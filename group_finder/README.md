# ASTRA group finder

`group_finder/` is the production raw-catalogue pipeline. It reads
`temp/raw/zone_NGC.fits.gz` and `temp/raw/zone_SGC.fits.gz`, constructs the
joint data/random Delaunay graph, applies the literal lowest-index ASTRA
watershed, prunes every group to the valid selection component connected to
its density minimum, measures moment-ellipsoid shapes, and writes FITS
catalogues.

The default command runs BGS, LRG, ELG, and QSO in both zones:

```bash
python -u -m group_finder
```

For a smaller selection:

```bash
python -u -m group_finder \
  --tracers LRG \
  --zones NGC SGC \
  --iteration 0 \
  --output-root temp/group_finder/lrg
```

Each case creates:

```text
catalogs/<TRACER>/<ZONE>/<TRACER>_<ZONE>_iter000_all.fits
catalogs/<TRACER>/<ZONE>/<TRACER>_<ZONE>_iter000_clean.fits
```

The `all` file contains post-mask measurable survivors and a `BORDER` flag.
The `clean` file excludes `BORDER=True` rows. Completely discarded groups are
not written. Required analysis columns are:

```text
VOID_ID RA DEC REDSHIFT R_EFF ELLIP
```

The default combined plot is:

```text
plots/all_tracers_zones_iter000_R_EFF_ELLIP.png
```

It uses every measurable post-mask survivor from the `all` catalog; the
strictly interior subset remains available in `clean`. The figure has a black
background and LaTeX rendering. Tracer colors are cyan for BGS, orange for
LRG, limegreen for ELG, and magenta for QSO. NGC is solid and SGC is dashed.
