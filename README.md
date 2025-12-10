# ASTRA-DESI

Implementation of the [ASTRA algorithm](https://arxiv.org/abs/2404.01124) adapted to the
Dark Energy Spectroscopic Instrument (DESI) clustering catalogues. The pipeline supports
the **Early Data Release (EDR)** plus **Data Releases 1 and 2 (DR1/DR2)** and produces
per-zone classifications of the cosmic web into **voids, sheets, filaments, and knots**.


## Requirements

- Linux environment (NERSC or equivalent HPC node recommended)
- Python 3.9+ (tested with 3.12)
- Packages: `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`
- Optional: `requests` for Zenodo uploads (pulled in by `zenodo_push.py`)


## Repository layout

- **`src/desiproc/`** – Core data-processing modules
  - `read_data.py`: helpers for loading DESI clustering catalogues and building Cartesian
    coordinates
  - `implement_astra.py`: Delaunay-based pair generation, web-type classification, and
    probability estimation
  - `gen_groups.py`: FoF group finder with configurable `r` thresholds
  - `paths.py`: canonical naming helpers for raw/classification/probability/pairs files
- **`src/plot/`** – Visualisation entry points
  - `common.py`: shared loaders and path resolvers used by all plotting scripts
  - `plot_wedges.py`: tracer-by-zone wedge plots for raw classifications (EDR/DR1/DR2), including FoF groups, global `--z-slice` cuts, per-tracer windows via `--tracer-z-slice`, and an optional `--view section` mode for annular “fan” wedges
  - `plot_extra.py`: histograms, CDFs, and supplementary wedges
- **`src/main.py`** – Command-line driver that orchestrates preprocessing, pair generation,
  classification, probabilities, and group finding (EDR/DR1/DR2)
- **`jobs/`** – Ready-to-run scripts for either interactive shells (`run_edr.sh`) or
  SLURM batch jobs (`run_edr.sbatch`, `run_dr1.sbatch`)
- **`zenodo/`** – Tools to stage pipeline outputs and push them to Zenodo (`zenodo_push.py`,
  `zenodo_upl.py`, `post_edr.sh`, and metadata templates under `zenodo/json/`)


## Pipeline Outputs

Each zone produces a consistent set of artefacts stored under the release root
(`classification/`, `probabilities/`, `pairs/`):

- **Raw tables** (`raw/zone_XX*.fits.gz`): combined real + random catalogue
- **Pairs** (`pairs/zone_XX*_pairs.fits.gz`): Delaunay edges
- **Classification** (`classification/zone_XX_*classified.fits.gz`): counts of data/random
  neighbours
- **Probabilities** (`probabilities/zone_XX*_probability.fits.gz`): void/sheet/filament/knot
  likelihoods using independent lower/upper `r` thresholds
- **Groups** (`groups/zone_XX*_groups_fof_WEBTYPE.fits.gz`): FoF group catalogues
- **Plots** (`figs/` or custom output): histograms, CDFs, standard wedges, FoF wedges


## Running the pipeline

### 1. Direct CLI (`src/main.py`)

Key CLI options:

- `--release {EDR,DR1,DR2}` selects the catalogue layout.
- `--r-lower` and `--r-upper` control the asymmetric thresholds used when classifying
  web types (defaults: `-0.9`, `0.9`).
- `--tracers` can restrict processing to a subset of tracer prefixes.
- `--plot` enables post-processing plots (written to `--plot-output` or `--groups-out`).
- `--only-plot` skips the heavy processing steps and reuses existing outputs.

**EDR example**

```bash
python src/main.py \
  --release EDR \
  --zone 0 \
  --base-dir /path/to/edr/catalogs \
  --raw-out /path/to/work/edr/raw \
  --class-out /path/to/work/edr/class \
  --groups-out /path/to/work/edr/groups \
  --plot-output /path/to/work/edr/figs \
  --n-random 100 \
  --r-lower -0.9 --r-upper 0.9 \
  --plot
```

**DR1 example**

```bash
python src/main.py \
  --release DR1 \
  --base-dir /path/to/dr1/catalogs \
  --raw-out /path/to/work/dr1/raw \
  --class-out /path/to/work/dr1/class \
  --groups-out /path/to/work/dr1/groups \
  --plot-output /path/to/work/dr1/figs \
  --zones NGC1 NGC2 \
  --tracers BGS_BRIGHT ELG \
  --n-random 100 \
  --r-lower -0.9 --r-upper 0.9 \
  --plot
```

Environment variables such as `PAIR_NJOBS_CAP` (maximum multiprocessing workers for
pair generation) can be exported beforehand when running on shared systems. When
`SLURM_CPUS_PER_TASK` is not set, the pipeline now defaults to using all visible CPU
cores (`os.cpu_count`).


### 2. Shell scripts in `jobs/`

The shell helpers wrap `src/main.py` with common configurations and directory layouts.

- `jobs/run_edr.sh [zone|all]` loads `python/3.12` on NERSC, points to the public EDR
  clustering directory, and produces/plots outputs in `/pscratch/.../edr/`. The script
  defaults to `--only-plot`, making it ideal for regenerating visualisations once the
  heavy processing has completed.

  ```bash
  # Regenerate plots for all EDR zones
  bash jobs/run_edr.sh all

  # Regenerate plots for a single zone
  bash jobs/run_edr.sh 05
  ```


### 3. SLURM batch jobs (`jobs/*.sbatch`)

- `jobs/run_edr.sbatch` submits one SLURM array per EDR zone, running the full pipeline
  (including plotting). Scratch outputs are written under `/pscratch/.../edr/`.
- `jobs/run_dr1.sbatch` is adapted to DR1; edit the `ZLABELS` and `TRACERS_BY_ZONE`
  arrays to match the desired zones/tracers. The script also enforces
  `PAIR_NJOBS_CAP`, capping multiprocessing workers based on `SLURM_CPUS_PER_TASK`.


## Visualisation tools

The plotting scripts under `src/plot/` share the loaders defined in `src/plot/common.py`.
Key entry points:

- `plot_wedges.py`: raw-classification wedges by tracer and FoF groups. Accepts the same release/tag layout as the main pipeline (EDR/DR1/DR2), supports both global `--z-slice zmin zmax` cuts, per-tracer windows via `--tracer-z-slice LRG:0.6:1.0`, and curved “fan” sections with `--view section` when you want to zoom into a thin shell.
- `plot_extra.py`: CDFs, histograms, and supplemental wedges. Supports on-disk caching
  (`--cache-dir`) to avoid repeated I/O.

Each script has an independent CLI.


## Zenodo packaging (`zenodo/`)

The `zenodo` directory provides automation for staging outputs and publishing them on
Zenodo:

- `zenodo_push.py`: orchestrates staging on `/pscratch`, compression of release folders,
  and upload via the Zenodo REST API. Supports sandbox mode, `--dry-run`, metadata JSON
  inputs (creators/related identifiers), and optional publication.
- `zenodo_upl.py`: lower-level helpers used by `zenodo_push.py` (copying staging trees,
  slugifying titles, etc.).
- `post_edr.sh` and `post_dr1.sh`: example shell wrappers invoking `zenodo_push.py` for the EDR and DR1 products.
- `json/members.json`: sample metadata template for Zenodo creators.

Basic usage (sandbox upload):

```bash
python zenodo/zenodo_push.py \
  --paths /pscratch/.../edr/raw /pscratch/.../edr/class /pscratch/.../edr/groups \
  --pscratch-dir /pscratch/.../cosmic-web \
  --title "ASTRA-DESI EDR Release v0.2" \
  --description "Early Data Release products for ASTRA-DESI (raw, class, groups)." \
  --creators-json zenodo/json/members.json \
  --keywords ASTRA DESI "cosmic web" \
  --sandbox --publish --token-file ~/.zenodo_token
```

Add `--dry-run` to generate the staging tarballs without performing the upload.