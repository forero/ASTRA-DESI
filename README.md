# ASTRA-DESI

Implementation of the [ASTRA algorithm](https://arxiv.org/abs/2404.01124) adapted to the
Dark Energy Spectroscopic Instrument (DESI) clustering catalogues. The pipeline supports
the **Early Data Release (EDR)** plus **Data Releases 1, 2, and 3 (DR1/DR2/DR3)** and produces
per-zone classifications of the cosmic web into **voids, sheets, filaments, and knots**.


## Requirements

- Linux environment (NERSC or equivalent HPC node recommended)
- Python 3.9+ (tested with 3.12)
- Packages: `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`
- Optional: `requests` for Zenodo uploads (pulled in by `zenodo_push.py`)


## Repository layout

- **`src/desiproc/`** – Core data-processing modules
- **`src/plot/`** – Visualisation entry points
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
- **DR1 properties** (`properties/zone_REGION_properties.fits.gz`): one row per real
  `TARGETID`, containing `SED_SFR`, `SED_MASS`, `FLUX_G`, and `FLUX_R`; successive
  runs reuse and, when needed, merge the regional file
- **Classification** (`classification/zone_XX_*classified.fits.gz`): counts of data/random
  neighbours
- **Probabilities** (`probabilities/zone_XX*_probability.fits.gz`): void/sheet/filament/knot
  likelihoods using independent lower/upper `r` thresholds
- **Plots** (`figs/` or custom output): histograms, CDFs, standard wedges, etc.


## Running the pipeline

### 1. Direct CLI (`src/main.py`)

Key CLI options:

- `--release {EDR,DR1,DR2,DR3}` selects the catalogue layout.
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

DR1 reads the native `{tracer}_NGC_*` and `{tracer}_SGC_*` catalogues directly;
it does not derive the two regions from RA/DEC and does not filter ASTRA inputs
with the auxiliary footprint masks.

```bash
python src/main.py \
  --release DR1 \
  --base-dir /path/to/dr1/catalogs \
  --raw-out /path/to/work/dr1/raw \
  --class-out /path/to/work/dr1/class \
  --groups-out /path/to/work/dr1/groups \
  --plot-output /path/to/work/dr1/figs \
  --zones NGC SGC \
  --tracers BGS_ANY BGS_BRIGHT ELG_LOPnotqso LRG QSO \
  --n-random 100 \
  --r-lower -0.9 --r-upper 0.9 \
  --plot
```


### 2. Shell scripts in `jobs/`

The shell helpers wrap `src/main.py` with common configurations and directory layouts.

- `jobs/run_edr.sh [zone|all]` loads `python/3.12` on NERSC, points to the public EDR
  clustering directory, and produces/plots outputs in `/pscratch/.../edr/`. The script
  defaults to `--only-plot`, making it ideal for regenerating visualisations once the
  heavy processing has completed.


### 3. SLURM batch jobs (`jobs/*.sbatch`)

- `jobs/run_edr.sbatch` submits one SLURM array per EDR zone, running the full pipeline
  (including plotting). Scratch outputs are written under `/pscratch/.../edr/`.
- `jobs/run_dr1.sbatch` submits a 10-task array covering NGC/SGC for `BGS_ANY`,
  `BGS_BRIGHT`, `LRG`, `ELG_LOPnotqso`, and `QSO`. Before ASTRA starts, it
  generates and saves auxiliary bright/dark HEALPix masks under
  `masks/bright_dark/`. NGC and SGC mask membership comes from the corresponding
  catalogue filenames, not from an RA/DEC split, and the masks are not applied
  to ASTRA's input rows. The script also enforces `PAIR_NJOBS_CAP`, capping
  multiprocessing workers based on `SLURM_CPUS_PER_TASK`.

  ```bash
  sbatch -J lrg_ngc --export=ALL,TRACER=LRG,ZONE=NGC jobs/run_dr3.sbatch
  sbatch -J lrg_sgc --export=ALL,TRACER=LRG,ZONE=SGC jobs/run_dr3.sbatch
  sbatch -J elg_ngc --export=ALL,TRACER=ELG,ZONE=NGC jobs/run_dr3.sbatch
  sbatch -J bgs_m2135_ngc --export=ALL,TRACER=BGS_BRIGHT-21.35,ZONE=NGC jobs/run_dr3.sbatch

  sbatch -J lrg_ngc_prob --export=ALL,MODE=prob,TRACER=LRG,ZONE=NGC jobs/run_dr3.sbatch
  ```


## Visualisation tools

The plotting scripts under `src/plot/` share the loaders defined in `src/plot/common.py`.
Key entry points:

- `plot_wedges.py`: raw-classification wedges by tracer and FoF groups. Accepts the same release/tag layout as the main pipeline (EDR/DR1/DR2), supports both global `--z-slice zmin zmax` cuts, per-tracer windows via `--tracer-z-slice LRG:0.6:1.0`, and curved “fan” sections with `--view section` when you want to zoom into a thin shell.
- `plot_extra.py`: CDFs, histograms, and supplemental wedges. Supports on-disk caching
  (`--cache-dir`) to avoid repeated I/O.


## Zenodo packaging (`zenodo/`)

The `zenodo` directory provides automation for staging outputs and publishing them on
Zenodo:

- `zenodo_push.py`: orchestrates staging and compression of release folders, or uploads
  existing archives unchanged with `--direct-upload`. It supports sandbox mode, a
  no-network `--dry-run`, resumable draft uploads, metadata JSON inputs, and optional
  publication.
- `zenodo_upl.py`: lower-level helpers used by `zenodo_push.py` (copying staging trees,
  slugifying titles, etc.).
- `post_edr.sh` and `post_dr1.sh`: example shell wrappers invoking `zenodo_push.py` for the EDR and DR1 products.
- `json/members.json`: sample metadata template for Zenodo creators.