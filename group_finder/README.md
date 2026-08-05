# ASTRA void finder

## Inputs

The pipeline accepts either combined files:

```text
<raw-dir>/zone_NGC.fits.gz
<raw-dir>/zone_SGC.fits.gz
```

or files split by tracer:

```text
<raw-dir>/zone_NGC_BGS.fits.gz
<raw-dir>/zone_NGC_LRG.fits.gz
<raw-dir>/zone_NGC_ELG.fits.gz
<raw-dir>/zone_NGC_QSO.fits.gz
...
```

For split DR2 files, separate raw labels such as `BGS_ANY_DATA` /
`BGS_ANY_RAND` and `ELG_DATA` / `ELG_RAND` are detected automatically.

## Algorithm

1. **Read one case:** Load all data and the requested random realization for
   one tracer and cap.

2. **Build the graph:** Combine data and random Cartesian positions and build their 3D Delaunay graph.

3. **Compute r:** For every graph vertex, count data and
   random neighbors and calculate

   ```text
   r = (N_data - N_random) / (N_data + N_random)
   ```

4. **Grow watershed basins:** Process selected vertices from lowest to highest
   `r`. A point with no assigned neighbor seeds a new group; otherwise it joins the lowest-ID neighboring group. Existing groups are not merged when basins meet. Groups with fewer than four total data-plus-random members are removed by default.

5. **Apply the selection mask:** Use only the requested random realization to
   build an angular HEALPix mask and a radial count mask. The defaults are
   `NSIDE=128`, at least 3 randoms per pixel, radial-bin width 5, and at least 3 randoms per radial bin.

6. **Boundaries:** Remove invalid members and internal edges that cross invalid angular or radial regions. Any affected group is
   marked `BORDER`.

7. **Measure the shape:** Use only the final random members. The center is their mean Cartesian position. The moment-tensor eigenvalues satisfy
   `lambda_1 >= lambda_2 >= lambda_3`, then

   ```text
   R_EFF = sqrt(5) (lambda_1 lambda_2 lambda_3)^(1/6)
   ELLIP = 1 - ((lambda_3 + lambda_2) / (lambda_2 + lambda_1))^(1/4).
   ```

## Outputs

Each tracer/cap case produces:

```text
catalogs/<TRACER>/<ZONE>/<TRACER>_<ZONE>_iter000_all.fits
catalogs/<TRACER>/<ZONE>/<TRACER>_<ZONE>_iter000_clean.fits
catalogs/<TRACER>/<ZONE>/<TRACER>_<ZONE>_iter000_membership.fits
```

### `all.fits`

One row per group after applying the mask. No minimum-random, bootstrap, or
shape-stability cut is applied. If a group does not define a full-rank 3D
moment ellipsoid, its undefined shape values are stored as `NaN`.

| Column | Meaning |
| --- | --- |
| `VOID_ID` | Globally unique void identifier. |
| `XCART`, `YCART`, `ZCART` | Mean Cartesian position of the retained random members, in `Mpc/h`. |
| `R_EFF` | Effective radius in `Mpc/h`. |
| `ELLIP` | Ellipticity. |
| `BORDER` | Whether the original group touched angular or radial selection. |

### `clean.fits`

The `BORDER=False` subset of `all.fits`. It contains:

```text
VOID_ID XCART YCART ZCART R_EFF ELLIP
```

### `membership.fits`

One row for every random point in the selected realization, including
unassigned randoms.

| Column | Meaning |
| --- | --- |
| `TARGETID` | Original random identifier. |
| `RA`, `DEC`, `Z` | Original observed coordinates and redshift. |
| `XCART`, `YCART`, `ZCART` | Original Cartesian coordinates. |
| `RANDITER` | Processed random realization. |
| `R_VALUE` | Local Delaunay r value. |
| `THRESHOLD_SELECTED` | Whether the point passed the `r` cut. |
| `GROUP_ID_PREMASK` | Group before selection; `-1` means unassigned. |
| `GROUP_ID` | Final post-mask group; `-1` means unassigned. |
| `VOID_ID` | Global ID for `GROUP_ID`; `-1` means no final group. |
| `MEMBER` | Whether `GROUP_ID >= 0`. |
| `PRUNED_BY_MASK` | Whether selection removed the point. |
| `BORDER` | Whether its pre-mask group touched invalid selection. |

The run also writes:

```text
run_iter000_summary.json
plots/all_tracers_zones_iter000_R_EFF_ELLIP.png
```

The JSON records parameters, paths, timings, and group/member counts. The plot compares `R_EFF` and `ELLIP` distributions from the `all` catalogs.

## Running

Run on the split DR2 inputs:

```bash
python -u -m group_finder \
  --raw-dir /pscratch/sd/v/vtorresg/cosmic-web/dr2/raw \
  --output-root /pscratch/sd/v/vtorresg/cosmic-web/dr2/group_finder \
  --mask-cache /pscratch/sd/v/vtorresg/cosmic-web/dr2/group_finder/healpix_masks
```
