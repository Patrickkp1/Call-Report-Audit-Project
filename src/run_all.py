# run_all.py
# Runs the full pipeline from raw API data to compiled paper figures.
# Each script is idempotent — re-running skips cached quarterly parquets,
# overwrites the analysis panel, retrains models, and regenerates figures.

import os
import subprocess
import time

SCRIPTS = [
    "src/01_load_data.py",
    "src/02_build_ratios_stats.py",
    "src/03_run_models.py",
    "src/04_make_figures.py",
]

for script in SCRIPTS:
    print("\n" + "=" * 72)
    print(f"RUNNING: {script}")
    print("=" * 72)
    t0 = time.time()
    r = subprocess.run(["python3", script])
    if r.returncode != 0:
        print(f"\n!! {script} failed with code {r.returncode}; stopping.")
        break
    print(f"\n{script} finished in {time.time()-t0:.1f}s")

print("\nAll scripts complete.")
print("Outputs:")
print("  data/raw/         - one parquet per quarter (cached)")
print("  data/processed/   - analysis panel + trained models")
print("  tables/           - CSV results")
print("  figures/          - PNG + PDF charts")
