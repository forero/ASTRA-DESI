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
  - `plot_groups.py` – Wedge plots of FoF groups
  - `plot_wedge.py` – Tracer-based wedge plots across multiple zones
- `src/main.py` – Full pipeline driver (EDR or DR1)
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


## Outputs

- **Raw tables**: combined real + random catalogs per zone  
- **Pairs files**: Delaunay-based connections
- **Classification files**: counts and webtype assignments
- **Probability files**: void/sheet/filament/knot probabilities
- **Groups**: FoF groups
- **Plots**: histograms, CDFs, and wedge diagrams


## Data sharing on Zenodo

This repository includes utility scripts to package and upload the generated data products to **Zenodo** for long-term archiving and sharing.  

- A staging area is created automatically in `/pscratch/.../zenodo_staging/`, ensuring the original pipeline outputs are never modified.  
- Each subfolder (`raw/`, `class/`, `groups/`) is compressed into a `.tar.gz` file (e.g., `raw.tar.gz`, `class.tar.gz`, `groups.tar.gz`).  
- These tarballs are then uploaded to Zenodo using the REST API, with metadata such as title, description, creators, keywords, and version provided via command-line arguments or JSON files.  
- Authentication is handled via a Zenodo API token stored in a local file (e.g., `~/.zenodo_token`).  

Example (sandbox mode, publishing after upload):

```bash
python src/utils/zenodo_push.py \
  --paths /pscratch/sd/v/vtorresg/cosmic-web/edr/raw \
         /pscratch/sd/v/vtorresg/cosmic-web/edr/class \
         /pscratch/sd/v/vtorresg/cosmic-web/edr/groups \
  --pscratch-dir /pscratch/sd/v/vtorresg/cosmic-web \
  --title "ASTRA-DESI EDR Release v0.1" \
  --description "Early Data Release products for ASTRA-DESI (raw, class, groups)." \
  --creators-json ./json/members.json \
  --keywords ASTRA DESI "cosmic web" \
  --sandbox \
  --publish \
  --token-file ~/.zenodo_token
```

This will produce tarballs in the staging directory and upload them to Zenodo.  
Use `--dry-run` to only generate the staging and tarballs without uploading.