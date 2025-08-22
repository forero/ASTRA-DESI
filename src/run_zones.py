import os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'main.py')

# Default release/version (override via environment)
RELEASE = os.environ.get("RELEASE", "edr")
VERSION = os.environ.get("VERSION", "v2.0")

CLUSTER_DIR = f'/global/cfs/cdirs/desi/public/{RELEASE}/vac/{RELEASE}/lss/{VERSION}/LSScats/clustering'
SCRATCH_BASE = '/pscratch/sd/v/vtorresg/cosmic-web'

RAW_OUT = f'{SCRATCH_BASE}/{RELEASE}/raw'
CLASS_OUT = f'{SCRATCH_BASE}/{RELEASE}/class'
GROUPS_OUT = f'{SCRATCH_BASE}/{RELEASE}/groups'
FIGS_OUT = f'{SCRATCH_BASE}/{RELEASE}/figs'
for d in (RAW_OUT, CLASS_OUT, GROUPS_OUT, FIGS_OUT):
    os.makedirs(d, exist_ok=True)

common_args = [sys.executable, MAIN_SCRIPT, '--base-dir', CLUSTER_DIR, '--raw-out',
               RAW_OUT, '--class-out', CLASS_OUT, '--groups-out', GROUPS_OUT,
               '--webtype', 'filament', '--source', 'data', '--n-random', '100',
               '--plot', '--plot-output', FIGS_OUT,] # '--only-plot',]

ENV = dict(os.environ)
ENV.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',})

def run_zone(zone: int):
    cmd = common_args + (['--zone', str(zone)] if RELEASE.lower()=="edr" else ['--zones', str(zone)])
    proc = subprocess.run(cmd, check=True, env=ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return zone, proc.stdout, proc.stderr

if __name__ == '__main__':
    if RELEASE.lower() == "edr":
        zones = list(range(20))
    else:
        zones = ["NGC1","NGC2"]
    max_workers = min(4, cpu_count())
    print(f'Using {max_workers} workers')
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_zone, z): z for z in zones}
        for fut in as_completed(futs):
            z = futs[fut]
            try:
                zone, out, err = fut.result()
                print(f'[OK] zone {zone:02d}')
                if err.strip():
                    print(f'[zone {zone:02d} STDERR]\n{err}')
                results.append(zone)
            except subprocess.CalledProcessError as e:
                print(f'[FAIL] zone {z:02d} exit={e.returncode}')
                print('STDOUT:\n', e.stdout)
                print('STDERR:\n', e.stderr)
            except Exception as e:
                print(f'[EXC] zone {z:02d}: {e!r}')

    done = sorted(results)
    missing = sorted(set(zones) - set(done))
    print(f'Done: {done}')
    print(f'Missing: {missing}')