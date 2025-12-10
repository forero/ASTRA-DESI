import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from matplotlib.lines import Line2D

from pathlib import Path

CLASS_COLORS = {
    'void': 'crimson',
    'sheet': 'orange',
    'filament': 'cornflowerblue',
    'knot': 'midnightblue',
}
WEBTYPES = ('void', 'sheet', 'filament', 'knot')
TRACERS = ('BGS', 'LRG', 'ELG', 'QSO')
Z_QUANTILE_BOUNDS = (10, 90)
SCATTER_SIZE = 2

TRACER_LIMITS = {
    'BGS': ((-800, -400), (-800, -400)),
    'LRG': ((-2000, -1100), (-2000, -1100)),
    'ELG': ((-2600, -1400), (-3000, -1800)),
    'QSO': ((-3100, -1900), (-3600, -2400)),
}

BASE_DIR = Path('/pscratch/sd/v/vtorresg/cosmic-web/dr1')
RAW_TEMPLATE = 'raw/zone_{zone}.fits.gz'
PROB_TEMPLATE = 'probabilities/zone_{zone}_probability.fits.gz'
ZONE = 'NGC2'


def _find_column_name(table, base):
    for suffix in ('', '_1', '_2', '_raw', '_prob'):
        name = f'{base}{suffix}' if suffix else base
        if name in table.colnames:
            return name
    raise KeyError(f'Column {base} not found in table.')


def _identify_tracer(label):
    if label is None:
        return ''
    value = str(label).strip()
    if not value:
        return ''
    core = value.split('_', 1)[0].upper()
    return core if core in TRACERS else ''


def _classify_webtypes(table):
    cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    for col in cols:
        if col not in table.colnames:
            raise KeyError(f'Missing probability column {col}.')
    arr = np.vstack([np.asarray(table[col], dtype=float) for col in cols]).T
    finite_mask = np.isfinite(arr).any(axis=1)
    safe_arr = np.where(np.isfinite(arr), arr, -np.inf)
    idx = np.argmax(safe_arr, axis=1)
    classes = np.array(WEBTYPES, dtype='U8')[idx]
    classes[~finite_mask] = ''
    return classes


def _compute_redshift_slice(z_values):
    finite = z_values[np.isfinite(z_values)]
    if finite.size == 0:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(finite, Z_QUANTILE_BOUNDS)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = np.nanmin(finite), np.nanmax(finite)
    if hi - lo < 1e-3:
        mid = 0.5 * (hi + lo)
        width = max(0.05, hi - lo)
        lo = mid - width
        hi = mid + width
    return lo, hi


def main():
    raw_path = BASE_DIR / RAW_TEMPLATE.format(zone=ZONE)
    prob_path = BASE_DIR / PROB_TEMPLATE.format(zone=ZONE)

    raw_tbl = Table.read(str(raw_path))
    prob_tbl = Table.read(str(prob_path))
    joined = join(raw_tbl, prob_tbl, keys='TARGETID')

    randiter_col = _find_column_name(joined, 'RANDITER')
    data_mask = np.asarray(joined[randiter_col]) == -1
    joined = joined[data_mask]

    tracer_col = _find_column_name(joined, 'TRACERTYPE')
    tracers = np.array([_identify_tracer(val) for val in joined[tracer_col]], dtype='U4')
    joined['TRACER'] = tracers

    webtypes = _classify_webtypes(joined)
    joined['WEBTYPE'] = webtypes

    mask_valid = (joined['TRACER'] != '') & (joined['WEBTYPE'] != '')
    joined = joined[mask_valid]

    x = np.asarray(joined[_find_column_name(joined, 'XCART')], dtype=float)
    y = np.asarray(joined[_find_column_name(joined, 'YCART')], dtype=float)
    z = np.asarray(joined[_find_column_name(joined, 'Z')], dtype=float)

    tracer_arr = np.asarray(joined['TRACER'])
    webtype_arr = np.asarray(joined['WEBTYPE'])

    fig, axes = plt.subplots(len(TRACERS), len(WEBTYPES), figsize=(12, 12))
    axes = np.atleast_2d(axes)

    for i, tracer in enumerate(TRACERS):
        ax_row = axes[i]
        tracer_mask = (tracer_arr == tracer)
        z_slice = _compute_redshift_slice(z[tracer_mask])
        z_min, z_max = z_slice
        slice_mask = tracer_mask & (z >= z_min) & (z <= z_max)

        for j, webtype in enumerate(WEBTYPES):
            ax = ax_row[j]
            if tracer == TRACERS[0]:
                ax.set_title(webtype.title())
            web_mask = slice_mask & (webtype_arr == webtype)
            if np.any(web_mask):
                x_sub = x[web_mask]
                y_sub = y[web_mask]
                ax.scatter(x_sub, y_sub, s=SCATTER_SIZE, c='black', alpha=0.8, linewidths=0.0)
                ax.set_aspect('equal', adjustable='box')
            else:
                # ax.text(0.5, 0.5, 'sin datos', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                ax.set_aspect('equal', adjustable='box')

            if i == len(TRACERS) - 1:
                ax.set_xlabel('$x$ [Mpc]')

        y_label = f'{tracer}\n{z_min:.2f} < z < {z_max:.2f}' if np.isfinite(z_min) and np.isfinite(z_max) else tracer
        axes[i, 0].set_ylabel(y_label)

        limits = TRACER_LIMITS.get(tracer)
        if limits:
            xlim, ylim = limits
            for ax in ax_row:
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

    # handles = [Line2D([0], [0], marker='o', linestyle='', color=CLASS_COLORS[w], label=w.title(), markersize=6) for w in WEBTYPES]
    # fig.legend(handles, [h.get_label() for h in handles], loc='upper center', ncol=len(WEBTYPES), frameon=False, markerscale=4)

    fig.suptitle(f'Zona {ZONE}: distribución por tracer y tipo de red')
    # fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(f'cosmic_web_{ZONE}.png', dpi=360)
    plt.show()


if __name__ == '__main__':
    main()