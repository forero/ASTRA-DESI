import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time as t


def run_plot_add(script_path, raw_dir, class_dir, output_dir, zones, max_workers):
    """
    Launches plot_add.py for each zone in parallel using subprocess.

    Args:
        script_path: Path to the plot_add.py script.
        raw_dir: Directory with raw data files.
        class_dir: Directory with classification files.
        output_dir: Base output directory.
        zones: List of zone integers to process.
        max_workers: Number of parallel workers.
    """
    init_t = t.time()
    os.makedirs(output_dir, exist_ok=True)

    def _call_zone(z):
        cmd = ["python3", script_path, "--raw-dir", raw_dir, "--class-dir", class_dir,
               "--zones", str(z), "--output", output_dir]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return z, result.returncode, result.stdout, result.stderr

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_zone, z): z for z in zones}
        for future in as_completed(futures):
            zone = futures[future]
            try:
                z, code, out, err = future.result()
                if out:
                    print(f"[Zone {z:02d}] stdout:\n{out.strip()}")
                if code == 0:
                    print(f"[Zone {z:02d}] completed successfully")
                else:
                    print(f"[Zone {z:02d}] ERROR (code={code}):\n{err.strip()}")
            except Exception as e:
                print(f"[Zone {zone:02d}] raised exception: {e}")

    print(f"Elapsed: {t.time()-init_t:.2f} s")


if __name__ == "__main__":
    SCRIPT = "../src/plot_extra.py"
    RAW_DIR = "/pscratch/sd/v/vtorresg/cosmic-web/edr/raw"
    CLASS_DIR = "/pscratch/sd/v/vtorresg/cosmic-web/edr/class"
    OUTPUT = "../plots"
    ZONES = list(range(20))
    max_workers = max(1, os.cpu_count()-1)
    print(f"- Using {max_workers} workers")
    run_plot_add(SCRIPT, RAW_DIR, CLASS_DIR, OUTPUT, ZONES, max_workers=max_workers)