# ASTRA-DESI

Implementation of the [ASTRA algorithm](https://arxiv.org/abs/2404.01124) adapted to the
Dark Energy Spectroscopic Instrument (DESI) clustering catalogues. The pipeline supports
the **Early Data Release (EDR)** plus **Data Releases 1 and 2 (DR1/DR2)** and produces
per-zone classifications of the cosmic web into **voids, sheets, filaments, and knots**.

> Zapata-Zuluaga et al., *The Cosmic Web in the DESI Early Data Release: A Probabilistic Environment Catalog*, arXiv:2604.01456.  
> https://doi.org/10.48550/arXiv.2604.01456


## Requirements

- Linux environment (NERSC or equivalent HPC node recommended)
- Python 3.9+ (tested with 3.12)
- Packages: `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`
- Optional: `requests` for Zenodo uploads (pulled in by `zenodo_push.py`)


## Repository layout

- **`src/desiproc/`** – Core data-processing modules
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
- **Classification** (`classification/zone_XX_*classified.fits.gz`): counts of data/random
  neighbours
- **Probabilities** (`probabilities/zone_XX*_probability.fits.gz`): void/sheet/filament/knot
  likelihoods using independent lower/upper `r` thresholds


## Running the pipeline

### Direct CLI (`src/main.py`)

Key CLI options:

- `--release {EDR,DR1,DR2}` selects the catalogue layout.
- `--r-lower` and `--r-upper` control the asymmetric thresholds used when classifying
  web types (defaults: `-0.9`, `0.9`).
- `--tracers` can restrict processing to a subset of tracer prefixes.

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


### SLURM batch jobs (`jobs/*.sbatch`)

- `jobs/run_edr.sbatch` submits one SLURM array per EDR zone, running the full pipeline
  (including plotting). Scratch outputs are written under `/pscratch/.../edr/`.
- `jobs/run_dr1.sbatch` is adapted to DR1; edit the `ZLABELS` and `TRACERS_BY_ZONE`
  arrays to match the desired zones/tracers. The script also enforces
  `PAIR_NJOBS_CAP`, capping multiprocessing workers based on `SLURM_CPUS_PER_TASK`.


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

Add `--dry-run` to generate the staging tarballs without performing the upload.
