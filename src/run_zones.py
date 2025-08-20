import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "main.py")

CLUSTER_DIR = "/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
SCRATCH_BASE = "/pscratch/sd/v/vtorresg/cosmic-web"

RAW_OUT   = f"{SCRATCH_BASE}/edr/raw"
CLASS_OUT = f"{SCRATCH_BASE}/edr/class"
GROUPS_OUT = f"{SCRATCH_BASE}/edr/groups"
FIGS_OUT  = f"{SCRATCH_BASE}/edr/figs"

os.makedirs(RAW_OUT, exist_ok=True)
os.makedirs(CLASS_OUT, exist_ok=True)
os.makedirs(GROUPS_OUT, exist_ok=True)
os.makedirs(FIGS_OUT, exist_ok=True)

common_args = ["python", MAIN_SCRIPT, "--base-dir", CLUSTER_DIR,
               "--raw-out", RAW_OUT, "--class-out", CLASS_OUT,
               "--groups-out", GROUPS_OUT, "--webtype", "filament",
               "--source", "data", "--n-random", "100", "--plot",
               "--plot-output", FIGS_OUT,]# "--only-plot",]

def run_zone(zone):
    print(f'--> Zone {zone}')
    subprocess.run(common_args + ["--zone", str(zone)], check=True)

if __name__ == "__main__":
    max_workers = cpu_count()
    ls = range(20)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ex.map(run_zone, ls)