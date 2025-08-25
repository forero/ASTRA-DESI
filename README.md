# ASTRA-DESI

This repository provides an implementation of the **ASTRA algorithm** (https://arxiv.org/abs/2404.01124) adapted for the **Dark Energy Spectroscopic Instrument (DESI)** data.  
It supports both the **Early Data Release (EDR)** and **Data Release 1 (DR1)**, integrating directly with DESI clustering catalogs and generating classifications of the cosmic web into **voids, sheets, filaments, and knots**.


## Features

Full pipeline for:
  - Preprocessing real and random DESI clustering catalogs
  - Running ASTRA classification using Delaunay triangulation
  - Generating **pairs**, **classification**, and **probability** tables per zone
  - Group finding with **FoF** in each web environment
Visualization tools for:
  - Redshift and radial histograms
  - Cumulative distribution functions (CDFs)
  - Wedge plots of web classifications, by tracer and zone


## Structure

- `src/desiproc/` – Core data processing:
  - `read_data.py` – Loading and filtering DESI catalogs
  - `implement_astra.py` – ASTRA algorithm (pairs, classification, probabilities)
  - `gen_groups.py` – Group finding (FoF/DBSCAN) and group properties
- `src/plot/` – Visualization scripts:
  - `plot_extra.py` – Histograms, radial distributions, wedge plots (raw/probability)
  - `plot_groups.py` – Wedge plots of FoF groups.
  - `plot_wedge.py` – Tracer-based wedge plots across multiple zones
- `src/main.py` – Full pipeline driver (EDR or DR1).
- `jobs/` – Example SLURM batch scripts for running the pipeline on NERSC


## Usage

For **EDR** (zones by rosettes):

```bash
python -m src.main \
  --base-dir /path/to/edr/catalogs \
  --raw-out /path/to/output/raw \
  --class-out /path/to/output/class \
  --groups-out /path/to/output/groups \
  --release EDR \
  --n-random 100 \
  --plot
```

For **DR1** (zones by NGC/SGC regions):

```bash
python -m src.main \
  --base-dir /path/to/dr1/catalogs \
  --raw-out /path/to/output/raw \
  --class-out /path/to/output/class \
  --groups-out /path/to/output/groups \
  --release DR1 \
  --region N \
  --zones NGC1 NGC2 \
  --n-random 100 \
  --plot
```

### Plotting only (using existing outputs)

```bash
python -m src.main \
  --raw-out /path/to/output/raw \
  --class-out /path/to/output/class \
  --groups-out /path/to/output/groups \
  --only-plot \
  --plot-output /path/to/plots
```

---

## Outputs

- **Raw tables**: combined real + random catalogs per zone  
- **Pairs files**: Delaunay-based connections
- **Classification files**: counts and webtype assignments
- **Probability files**: void/sheet/filament/knot probabilities
- **Groups**: FoF groups
- **Plots**: histograms, CDFs, and wedge diagrams