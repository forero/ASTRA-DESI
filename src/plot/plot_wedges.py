import argparse, glob, os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from astropy.cosmology import Planck18
from astropy.table import Table, join, vstack
from matplotlib.lines import Line2D

from pathlib import Path

if __package__ is None or __package__ == '':
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from desiproc.paths import safe_tag, zone_tag
    from desiproc.gen_groups import classify_by_probability
    from plot.common import resolve_raw_path, resolve_probability_path
    from plot.color_theme import load_theme, apply_matplotlib_theme
else:
    from desiproc.paths import safe_tag, zone_tag
    from desiproc.gen_groups import classify_by_probability
    from .common import resolve_raw_path, resolve_probability_path
    from .color_theme import load_theme, apply_matplotlib_theme
matplotlib.rcParams['text.usetex'] = True

# Change this value (or the PLOT_WEDGE_THEME / PLOT_THEME environment variables) to switch themes.
THEME_NAME, THEME = load_theme('PLOT_WEDGE_THEME', default='light')
apply_matplotlib_theme(THEME)
CLASS_COLORS = dict(THEME['class_colors'])
CLASS_ZORDER = {'void': 3, 'sheet': 0, 'filament': 1, 'knot': 2}
ALL_WEBTYPES = tuple(CLASS_COLORS.keys())
ORDERED_TRACERS = ['BGS', 'LRG', 'ELG', 'QSO']
TRACER_ZLIMS = {'BGS': 0.45, 'LRG': 1.0, 'ELG': 1.4, 'QSO': 2.2}

RAW_COLS = ['TARGETID','RA','Z','TRACERTYPE','RANDITER']
GROUPS_COLS = ['TARGETID','TRACERTYPE','RANDITER','GROUPID','NPTS','XCM','YCM','ZCM']

TEXT_COLOR = THEME['text']
main_color, sec_color = THEME['primary'], THEME['secondary']
SCATTER_EDGE_COLOR = THEME['scatter_edge']
HIGHLIGHT_EDGE_COLOR = THEME['highlight_edge']
CLASS_FALLBACK_COLOR = THEME['class_fallback']
GROUP_PALETTE_NAME = THEME['group_palette']
MONO_COLOR_DEFAULT = THEME['mono']
CENTER_SCATTER_COLOR = THEME['center_color']


def read_groups(groups_dir, zone, webtype, out_tag=None):
    """
    Return the FoF group table for ``zone`` and ``webtype``.

    Args:
        groups_dir (str): Directory containing FoF group catalogues.
        zone (int | str): Zone identifier.
        webtype (str): Web structure label (``'void'``, ``'sheet'``, ``'filament'``, ``'knot'``).
        out_tag (str | None): Optional suffix used when generating the catalogue.
    Returns:
        Table: Groups table restricted to the columns defined in ``GROUPS_COLS``.
    """

    tag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    if out_tag is not None:
        path = os.path.join(groups_dir, f'zone_{tag}{tsuf}_groups_fof_{webtype}.fits.gz')
        if os.path.exists(path):
            try:
                tbl = Table.read(path, memmap=True)
            except TypeError:
                tbl = Table.read(path)
            missing = [c for c in GROUPS_COLS if c not in tbl.colnames]
            if missing:
                raise KeyError(f'Missing columns {missing} in {path}')
            return tbl[GROUPS_COLS]

    legacy = os.path.join(groups_dir, f'zone_{tag}_groups_fof_{webtype}.fits.gz')
    if os.path.exists(legacy):
        try:
            tbl = Table.read(legacy, memmap=True)
        except TypeError:
            tbl = Table.read(legacy)
        missing = [c for c in GROUPS_COLS if c not in tbl.colnames]
        if missing:
            raise KeyError(f'Missing columns {missing} in {legacy}')
        return tbl[GROUPS_COLS]

    pattern = os.path.join(groups_dir, f'zone_{tag}_*_groups_fof_{webtype}.fits.gz')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No groups files found for zone {tag} and webtype {webtype} in {groups_dir}')
    tables = []
    for path in files:
        try:
            t = Table.read(path, memmap=True)
        except TypeError:
            t = Table.read(path)
        missing = [c for c in GROUPS_COLS if c not in t.colnames]
        if missing:
            raise KeyError(f'Missing columns {missing} in {path}')
        tables.append(t[GROUPS_COLS])
    return vstack(tables, metadata_conflicts='silent')


def read_raw_min(raw_dir, class_dir, zone, out_tag=None):
    """
    Return raw data restricted to ``RAW_COLS`` for plotting.

    Args:
        raw_dir (str): Directory containing raw catalogues.
        class_dir (str): Directory containing classification catalogues.
        zone (int | str): Zone identifier.
        out_tag (str | None): Optional suffix used when generating the catalogue.
    Returns:
        Table: Raw table containing the columns listed in ``RAW_COLS``.
    """

    try:
        raw_path = resolve_raw_path(raw_dir, zone, out_tag)
        tbl = Table.read(raw_path, hdu=1, include_names=RAW_COLS, memmap=True)
        return tbl
    except (FileNotFoundError, TypeError):
        pass

    zone_str = zone_tag(zone)
    legacy = os.path.join(raw_dir, f'zone_{zone_str}.fits.gz')
    candidates = [legacy]
    candidates.extend(sorted(glob.glob(os.path.join(raw_dir, f'zone_{zone_str}_*.fits.gz'))))

    tables = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            tbl = Table.read(path, hdu=1, include_names=RAW_COLS, memmap=True)
        except Exception:
            tbl = Table.read(path, memmap=True)
            missing = [c for c in RAW_COLS if c not in tbl.colnames]
            if missing:
                raise KeyError(f'Missing columns {missing} in {path}')
            tbl = tbl[RAW_COLS]
        tables.append(tbl)

    if not tables:
        raise FileNotFoundError(f'No raw files found for zone {zone_str} in {raw_dir}')
    if len(tables) == 1:
        return tables[0]
    return vstack(tables, metadata_conflicts='silent')


def mask_source(randiter, source):
    """
    Masks the data based on the source type.
    
    Args:
        randiter (np.ndarray): Array indicating whether the data is from the random sample (-1) or not.
        source (str): Source type, can be 'data', 'rand', or 'both'.
    Returns:
        np.ndarray: Boolean mask indicating which rows to keep based on the source type.
    """
    is_data = (randiter == -1)
    return is_data if source == 'data' else (~is_data if source == 'rand' else np.ones_like(is_data, dtype=bool))


def filter_randiters(table, iterations):
    """
    Return ``table`` filtered to the requested ``RANDITER`` values.

    Args:
        table (Table): Astropy table containing a ``RANDITER`` column.
        iterations (Iterable[int] | None): Accepted iteration identifiers.
    Returns:
        Table: Filtered table (or the original table when ``iterations`` is falsy).
    """

    if not iterations:
        return table

    arr = np.asarray(table['RANDITER'])
    keep = np.isin(arr, np.asarray(list(iterations), dtype=arr.dtype))
    return table[keep]


def classify_webtypes(prob_df):
    """
    Classify most likely web type per row from probability dataframe.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability scores for each web type.
    Returns:
        np.ndarray: Array of classified web types.
    """

    if prob_df.empty:
        return np.empty(0, dtype='U8')

    cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    missing = [col for col in cols if col not in prob_df.columns]
    if missing:
        raise ValueError(f'Probability table missing columns: {missing}')

    arr = prob_df[cols].to_numpy(dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=-np.inf)
    idx = np.argmax(arr, axis=1)
    mapping = np.array(['void', 'sheet', 'filament', 'knot'], dtype='U8')
    out = mapping[idx]
    bad = ~np.isfinite(arr).any(axis=1)
    if np.any(bad):
        out[bad] = ''
    return out


def resolve_zones(release, zone_arg):
    """
    Resolve the list of zones to process based on the release and CLI input.

    EDR accepts integers 0..19 or the token ``all`` (default). DR1 requires an
    explicit zone name (e.g., ``NGC1``).
    
    Args:
        release (str): The release version (e.g., "EDR" or "DR1").
        zone_arg (str | list): The zone(s) to process, can be a single zone name or a list of names.
    Returns:
        list: A list of resolved zone identifiers.
    """

    release = str(release).upper()

    def _tokenize(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            tokens = []
            for item in value:
                tokens.extend(str(item).replace(',', ' ').split())
            return tokens
        return str(value).replace(',', ' ').split()

    tokens = _tokenize(zone_arg)

    if release == 'EDR':
        if not tokens:
            return list(range(20))
        zones = []
        for tok in tokens:
            if str(tok).lower() == 'all':
                return list(range(20))
            try:
                val = int(tok)
            except ValueError:
                raise ValueError(f'Invalid EDR zone "{tok}" (expected integer 0-19 or "all")')
            if not (0 <= val <= 19):
                raise ValueError(f'EDR zone {val} outside [0, 19]')
            zones.append(val)
        return sorted(set(zones)) if zones else list(range(20))

    if not tokens:
        raise ValueError('DR1 release requires specifying --zone (e.g., NGC1).')

    zones = []
    for tok in tokens:
        tok = str(tok).strip().upper()
        if tok == 'ALL':
            raise ValueError('DR1 does not support the "all" zone selector.')
        zones.append(tok)

    if len(zones) != 1:
        raise ValueError('DR1 plotting expects a single zone name (e.g., NGC1).')
    return zones


def tracer_prefixes(tr_types):
    """
    Extracts the tracer prefixes from the TRACERTYPE strings.
    
    Args:
        tr_types (np.ndarray): Array of TRACERTYPE strings.
    Returns:
        np.ndarray: Array of tracer prefixes, which are the parts of the TRACERTYPE
    """
    return np.char.partition(tr_types, '_')[:, 0]


def pick_tracers(available, wanted):
    """
    Selects the tracers from the available ones based on the wanted list.
    
    Args:
        available (np.ndarray): Array of available tracer prefixes.
        wanted (list or None): List of wanted tracer prefixes. If None, all available tracers are returned.
    Returns:
        np.ndarray: Array of tracer prefixes that are both available and wanted.
    """
    if not wanted:
        return np.unique(available)
    w = np.asarray(wanted, dtype=str)

    present = set(available.tolist())
    return np.array([t for t in w if t in present], dtype=str)


def subplot_grid(n):
    """
    Computes the number of rows and columns for subplots based on the number of tracers.
    
    Args:
        n (int): Number of tracers.
    Returns:
        tuple: Number of rows and columns for the subplots.
    """
    ncols = min(4, max(1, n))
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _init_ax(ax, title):
    """
    Initializes the axis for the wedge plot with a title and removes spines and ticks.

    Args:
        ax (matplotlib.axes.Axes): The axis to initialize.
        title (str): The title for the plot.
    """
    ax.set_title(title, fontsize=18, y=1.09, color=TEXT_COLOR)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def _compute_zone_params(ravec, zvec, z_lim):
    """
    Computes parameters for the zone plot based on RA and redshift vectors.
    
    Args:
        ravec (np.ndarray): Array of RA values.
        zvec (np.ndarray): Array of redshift values.
        z_lim (float): Redshift limit for the plot.
    Returns:
        tuple: A tuple containing the computed parameters (ra_min, ra_max, ra_ctr, Dc, half_w, zmax).
    """
    if zvec.size:
        zmax_data = float(np.nanmax(zvec)) * 1.02
    else:
        zmax_data = 0.0
    if z_lim is not None:
        zmax = float(z_lim)
        if zmax <= 0:
            zmax = zmax_data
        else:
            zmax = max(zmax, 1e-8)
    else:
        zmax = zmax_data
    ra_min, ra_max = float(np.nanmin(ravec)), float(np.nanmax(ravec))
    ra_ctr = 0.5 * (ra_min + ra_max)
    Dc = Planck18.comoving_distance(zmax).value
    half_w = Dc * np.deg2rad(ra_max - ra_ctr)
    return ra_min, ra_max, ra_ctr, Dc, half_w, zmax


def _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc, half_w, y_max, n_ra, n_z, coord):
    """
    Draws a grid of lines on the wedge plot to indicate the redshift and RA ticks.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the grid.
        ra_min (float): Minimum RA value.
        ra_max (float): Maximum RA value.
        ra_ctr (float): Central RA value.
        Dc (float): Comoving distance at the maximum redshift.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
        coord (str): Coordinate system being used ('z' or 'dc').
    Returns:
        tuple: A tuple containing the redshift and RA tick values.
    """
    if coord == 'z':
        zs = np.linspace(0.0, y_max, 300)
        z_ticks = np.linspace(0.0, y_max, n_z)
    else:
        zs = np.linspace(0.0, y_max, 300)
        z_ticks = np.linspace(0.0, y_max, n_z)
    ra_ticks = np.linspace(ra_min, ra_max, n_ra)
    for z0 in z_ticks:
        w0 = half_w * (z0 / y_max) if y_max > 0 else 0
        ax.hlines(z0, -w0, w0, color=sec_color, lw=0.5, alpha=0.5)
    step = max(1, n_ra // 4)
    for rt in ra_ticks[::step]:
        dx = Dc * np.deg2rad(rt - ra_ctr)
        ax.plot((dx / y_max) * zs if y_max > 0 else np.zeros_like(zs), zs, color=sec_color, lw=0.5, alpha=0.5)
    return z_ticks, ra_ticks


def _draw_borders(ax, half_w, y_max):
    """
    Draws the borders of the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the borders.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
    """
    ax.plot([-half_w, 0], [y_max, 0], lw=0.6, c=main_color)
    ax.plot([ half_w, 0], [y_max, 0], lw=0.6, c=main_color)
    ax.plot([-half_w, half_w], [y_max, y_max], lw=1.5, c=main_color)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(0, y_max)


def _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc, y_max):
    """
    Annotates the top of the wedge plot with RA tick values.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the annotations.
        ra_ticks (np.ndarray): Array of RA tick values.
        ra_ctr (float): Central RA value.
        Dc (float): Comoving distance at the maximum redshift.
        y_max (float): Maximum y value for the plot.
    """
    top4 = np.linspace(ra_ticks.min(), ra_ticks.max(), 4)
    x_top = Dc * np.deg2rad(top4 - ra_ctr)
    for xt, rt in zip(x_top, top4):
        ax.text(xt, y_max + 0.01*y_max, f'{rt:.0f}', ha='center', va='bottom', fontsize=19, color=TEXT_COLOR)
    ax.text(0, y_max + 0.05*y_max, 'RA (deg)', ha='center', va='bottom', fontsize=19, color=TEXT_COLOR)


def _annotate_y_side(ax, z_ticks, half_w, y_max, idx, ylabel):
    """
    Annotates the side of the wedge plot with redshift tick values.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the annotations.
        z_ticks (np.ndarray): Array of redshift tick values.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
        idx (int): Index of the current subplot.
        ylabel (str): Label for the y-axis.
    """
    for idx_tick, z0 in enumerate(z_ticks):
        x0r = half_w * (z0 / y_max) if y_max > 0 else 0
        angle = np.degrees(np.arctan2(-z0, -x0r)) if y_max > 0 else 0
        offset = np.sign(x0r) * half_w * 0.11 if np.abs(x0r) > 1e-6 else half_w * 0.11
        ax.text(x0r + offset, z0, f'{z0:.2f}', ha='left', va='center', rotation=angle + 180, fontsize=20, color=TEXT_COLOR)
    if idx == 0:
        ax.set_ylabel(ylabel, fontsize=30, labelpad=15, color=TEXT_COLOR)


def plot_wedges(joined, tracers, zone, webtype, out_png, smin, max_z, n_ra=15, n_z=10, coord='z',
                connect_lines=False, line_min_npts=2,
                min_npts=2, top_groups=None, max_points=None,
                z_range=None, ra_range=None,
                use_presets=False, highlight_longest=None, highlight_connect=False,
                *, color_mode='group', title=None, webtype_order=None, mono_color=None,
                per_tracer_caps=None):
    """Plot wedge diagrams for the requested tracers.

    Supports three colouring modes:
      - ``group`` (default): reuse the historical palette keyed by GROUPID.
      - ``webtype``: colour by the WEBTYPE column (void/sheet/filament/knot).
      - ``mono``: draw all points with a single colour (structure overview).
    """

    color_mode = str(color_mode or 'group').lower()
    if color_mode not in {'group', 'webtype', 'mono'}:
        raise ValueError(f'Unsupported color_mode={color_mode}')

    if mono_color is None:
        mono_color = MONO_COLOR_DEFAULT

    tracers = [str(t).split('_', 1)[0].upper() for t in tracers]
    tracers = [t for t in ORDERED_TRACERS if t in tracers]

    if per_tracer_caps:
        caps_map = {str(k).split('_', 1)[0].upper(): float(v) for k, v in per_tracer_caps.items() if v is not None}
    else:
        caps_map = {}

    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)

    if 'GROUPID' in joined.colnames:
        gids = np.asarray(joined['GROUPID'], dtype=np.int64)
    else:
        gids = None
    npts = np.asarray(joined['NPTS'], int) if 'NPTS' in joined.colnames else np.ones(tr_types.size, dtype=int)

    webtypes = None
    if color_mode == 'webtype':
        if 'WEBTYPE' not in joined.colnames:
            raise ValueError('WEBTYPE column required for color_mode="webtype"')
        webtypes = np.char.lower(np.asarray(joined['WEBTYPE']).astype(str))

    tr_pref = tracer_prefixes(tr_types)

    nrows, ncols = subplot_grid(len(tracers))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12,30), sharex=False, sharey=False,
                             gridspec_kw={'wspace': 0.5, 'hspace': 0.})
    fig.subplots_adjust(top=0.85)
    axes = np.atleast_1d(axes).ravel()

    default_title = f'{webtype.capitalize()}s in zone {zone_tag(zone)}'
    if color_mode == 'webtype':
        sp_t = 1.04
    else:
        sp_t = 0.97
    plt.suptitle(title or default_title, fontsize=22, y=sp_t, color=TEXT_COLOR)

    cmap = plt.get_cmap(GROUP_PALETTE_NAME)
    if hasattr(cmap, 'colors'):
        palette = np.asarray(cmap.colors)
    else:
        palette = cmap(np.linspace(0.0, 1.0, 256))
    point_size = 0.2#max(int(smin or 1), 2)

    if webtype_order is None:
        webtype_order = list(ALL_WEBTYPES)
    webtype_order = [str(w).lower() for w in webtype_order]
    used_webtypes = set()

    has_groups = gids is not None

    for idx_ax, tr in enumerate(tracers):
        tr = str(tr)
        ax = axes[idx_ax]
        _init_ax(ax, tr)

        m = (tr_pref == tr)
        if color_mode == 'group':
            local_min_npts = 0
        elif use_presets:
            preset_min = {'BGS': 8, 'LRG': 12, 'ELG': 8, 'QSO': 6}.get(tr, min_npts)
            local_min_npts = max(int(min_npts or 0), int(preset_min or 0))
        else:
            local_min_npts = int(min_npts or 0)

        if has_groups and local_min_npts > 1:
            m &= (npts >= local_min_npts)

        if z_range is not None and len(z_range) == 2:
            zlo, zhi = float(z_range[0]), float(z_range[1])
            m &= (zvec >= zlo) & (zvec <= zhi)
        if ra_range is not None and len(ra_range) == 2:
            rlo, rhi = float(ra_range[0]), float(ra_range[1])
            m &= (ravec >= rlo) & (ravec <= rhi)

        if has_groups and top_groups is not None and int(top_groups) > 0:
            tg = gids[m]
            tn = npts[m]
            if tg.size > 0:
                uniq_g, idx_first = np.unique(tg, return_index=True)
                g_sizes = tn[idx_first]
                order = np.argsort(-g_sizes)
                keep_g = set(uniq_g[order][:int(top_groups)].tolist())
                m &= np.isin(gids, list(keep_g))

        tr_key = tr.split('_', 1)[0].upper()
        z_cap = None
        if tr_key in caps_map:
            z_cap = float(caps_map[tr_key])
            m &= (zvec <= z_cap)
        elif max_z is not None:
            z_cap = float(max_z)
            m &= (zvec <= z_cap)
        else:
            z_cap = TRACER_ZLIMS.get(tr_key, None)
            if z_cap is not None:
                z_cap = float(z_cap)
                m &= (zvec <= z_cap)

        if not np.any(m):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16, color=TEXT_COLOR)
            continue

        ra_min, ra_max, ra_ctr, Dc_maxz, half_w, zmax = _compute_zone_params(ravec[m], zvec[m], z_cap)
        y_max = Planck18.comoving_distance(zmax).value if coord == 'dc' else zmax

        ra = ravec[m]
        yvals = zvec[m] if coord == 'z' else Planck18.comoving_distance(zvec[m]).value
        gid = gids[m] if has_groups else None
        wt_local = webtypes[m] if webtypes is not None else None

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc_maxz, half_w, y_max, n_ra, n_z, coord)

        idx_mod = None
        if color_mode == 'group' and has_groups:
            _, color_idx = np.unique(gid, return_inverse=True)
            idx_mod = color_idx % palette.shape[0]

        scale = yvals / y_max
        x = scale * (Dc_maxz * np.deg2rad(ra - ra_ctr))
        w_at_y = half_w * scale
        mclip = np.abs(x) <= w_at_y

        if max_points is not None and int(max_points) > 0:
            sel_idx = np.nonzero(mclip)[0]
            if sel_idx.size > int(max_points):
                rng = np.random.default_rng(123)
                take = rng.choice(sel_idx.size, int(max_points), replace=False)
                mask_sel = np.zeros_like(mclip)
                mask_sel[sel_idx[take]] = True
                mclip = mask_sel

        x_sel = x[mclip]
        y_sel = yvals[mclip]

        if connect_lines and has_groups:
            gid_sub = gid[mclip]
            x_sub = x_sel
            y_sub = y_sel

            if color_mode == 'group' and idx_mod is not None:
                ci_sub = idx_mod[mclip]
                for gval in np.unique(gid_sub):
                    selg = (gid_sub == gval)
                    if np.count_nonzero(selg) < line_min_npts:
                        continue
                    order = np.argsort(y_sub[selg])
                    xs = x_sub[selg][order]
                    ys = y_sub[selg][order]
                    ax.plot(xs, ys, lw=0.5, alpha=0.9,
                            color=palette[int(ci_sub[selg][0])], zorder=0)
            elif color_mode == 'webtype' and wt_local is not None:
                wt_sub = wt_local[mclip]
                for gval in np.unique(gid_sub):
                    selg = (gid_sub == gval)
                    if np.count_nonzero(selg) < line_min_npts:
                        continue
                    order = np.argsort(y_sub[selg])
                    xs = x_sub[selg][order]
                    ys = y_sub[selg][order]
                    wt_g = str(wt_sub[selg][0])
                    ax.plot(xs, ys, lw=0.5, alpha=0.9,
                            color=CLASS_COLORS.get(wt_g, CLASS_FALLBACK_COLOR),
                            zorder=CLASS_ZORDER.get(wt_g, 1))
            elif color_mode == 'mono':
                for gval in np.unique(gid_sub):
                    selg = (gid_sub == gval)
                    if np.count_nonzero(selg) < line_min_npts:
                        continue
                    order = np.argsort(y_sub[selg])
                    xs = x_sub[selg][order]
                    ys = y_sub[selg][order]
                    ax.plot(xs, ys, lw=0.5, alpha=0.6, color=mono_color, zorder=0)

        if color_mode == 'group' and idx_mod is not None:
            ax.scatter(x_sel, y_sel, s=point_size, edgecolor=SCATTER_EDGE_COLOR, lw=0.15,
                       c=palette[idx_mod[mclip]], alpha=1.0)
        elif color_mode == 'webtype' and wt_local is not None:
            wt_sel = wt_local[mclip]
            for wt in webtype_order:
                sel = (wt_sel == wt)
                if not np.any(sel):
                    continue
                used_webtypes.add(wt)
                ax.scatter(x_sel[sel], y_sel[sel], s=point_size,
                           c=CLASS_COLORS.get(wt, CLASS_FALLBACK_COLOR), alpha=1.0,
                           zorder=2 + CLASS_ZORDER.get(wt, 0),
                           edgecolor=SCATTER_EDGE_COLOR, lw=0.15,)
        else:  # mono
            ax.scatter(x_sel, y_sel, s=point_size, c=mono_color,
                       alpha=0.65, edgecolor='none')

        if has_groups and highlight_longest is not None and int(highlight_longest) > 0:
            gid_all = gid
            x_all = x
            y_all = yvals
            uniq_g, inv = np.unique(gid_all, return_inverse=True)
            if uniq_g.size > 0:
                order_h = np.argsort(inv, kind='mergesort')
                inv_o = inv[order_h]
                cuts = np.r_[0, np.cumsum(np.bincount(inv_o, minlength=uniq_g.size))]
                lengths = np.empty(uniq_g.size, dtype=float)
                for j in range(uniq_g.size):
                    sl = slice(cuts[j], cuts[j+1])
                    if sl.stop - sl.start <= 1:
                        lengths[j] = 0.0
                        continue
                    xs = x_all[order_h][sl]
                    ys = y_all[order_h][sl]
                    dx = xs.max() - xs.min()
                    dy = ys.max() - ys.min()
                    lengths[j] = np.hypot(dx, dy)
                k = min(int(highlight_longest), uniq_g.size)
                if k > 0:
                    top_idx = np.argsort(-lengths)[:k]
                    top_g = set(uniq_g[top_idx].tolist())
                    mh = mclip & np.isin(gid_all, list(top_g))
                    ax.scatter(x_all[mh], y_all[mh], s=point_size*1.8, facecolors='none',
                               edgecolors=HIGHLIGHT_EDGE_COLOR, linewidths=0.6, zorder=5)
                    if highlight_connect:
                        for gval in top_g:
                            selg = (gid_all == gval) & mclip
                            if np.count_nonzero(selg) < max(2, line_min_npts):
                                continue
                            order2 = np.argsort(y_all[selg])
                            xs = x_all[selg][order2]
                            ys = y_all[selg][order2]
                            ax.plot(xs, ys, lw=1.1, alpha=0.9, color=HIGHLIGHT_EDGE_COLOR, zorder=6)

        _draw_borders(ax, half_w, y_max)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc_maxz, y_max)
        _annotate_y_side(ax, z_ticks, half_w, y_max, idx_ax, 'z' if coord == 'z' else r'$D_c$ [Mpc]')

    for j in range(len(tracers), len(axes)):
        axes[j].axis('off')

    legend_handles = None
    if color_mode == 'webtype' and used_webtypes:
        order = [wt for wt in webtype_order if wt in used_webtypes]
        handles = [Line2D([], [], marker='o', linestyle='', markersize=10,
                          markerfacecolor=CLASS_COLORS.get(wt, CLASS_FALLBACK_COLOR), markeredgecolor='none',
                          label=wt.capitalize()) for wt in order]
        if handles:
            legend_handles = handles

    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.012),
                   ncol=min(len(legend_handles), 4), frameon=True, fontsize=20,
                   labelcolor=TEXT_COLOR)

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(out_png)
    plt.close(fig)
    return out_png


def _aggregate_group_centers(joined):
    """
    Aggregate per-group centers from the joined table (which has RA, Z, TRACERTYPE, GROUPID, NPTS).
    Returns arrays per unique group with median RA/Z and associated tracer prefix and NPTS.
    
    Args:
        joined (pd.DataFrame): The joined data containing tracer information.
    Returns:
        tuple: A tuple containing arrays of tracer prefixes, group IDs, median RA, median Z, and number of points per group.
    """
    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    tracers = np.char.partition(tr_types, '_')[:, 0]
    gids = np.asarray(joined['GROUPID'], dtype=np.int64)
    ravec = np.asarray(joined['RA'], dtype=float)
    zvec = np.asarray(joined['Z'], dtype=float)
    npts = np.asarray(joined['NPTS'], dtype=int) if 'NPTS' in joined.colnames else np.ones_like(gids)

    keys = np.empty(gids.size, dtype=[('TR','U8'),('G','i8')])
    keys['TR'] = tracers
    keys['G'] = gids
    uniq, inv = np.unique(keys, return_inverse=True)

    order = np.argsort(inv, kind='mergesort')
    ri = inv[order]
    cuts = np.r_[0, np.cumsum(np.bincount(ri, minlength=uniq.size))]
    ra_med = np.array([np.median(ravec[order][cuts[i]:cuts[i+1]]) for i in range(uniq.size)])
    z_med  = np.array([np.median(zvec[order][cuts[i]:cuts[i+1]])  for i in range(uniq.size)])
    n_per = np.array([int(npts[order][cuts[i]]) for i in range(uniq.size)])

    return uniq['TR'], uniq['G'], ra_med, z_med, n_per


def plot_group_centers(joined, tracers, zone, webtype, out_png, min_npts=2, max_z=None, n_ra=15, n_z=10, coord='z'):
    """
    Plot per-group centers (median RA/z) for the selected tracers.
    Sizes points by group membership (NPTS). Reuses the wedge layout.
    
    Args:
        joined (pd.DataFrame): The joined data containing tracer information.
        tracers (list): List of tracer types to plot.
        zone (int): The zone number to plot.
        webtype (str): The type of web to plot.
        out_png (str): Output file path for the plot.
        min_npts (int): Minimum number of points in a group to be plotted.
        max_z (float | None): Global maximum redshift cap. Per-tracer limits (TRACER_ZLIMS) are applied and this
                              value acts as an additional upper bound.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
        coord (str): Coordinate system being used ('z' or 'dc').
    Returns:
        str: The output PNG file path.
    """
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)

    ag_tr, ag_gid, ag_ra, ag_z, ag_n = _aggregate_group_centers(joined)

    tracers = [str(t).split('_', 1)[0].upper() for t in tracers]
    tracers = [t for t in ORDERED_TRACERS if t in tracers]

    z_cap = max_z
    if z_cap is None:
        z_cap = float(np.nanmax(zvec)) * 1.02
    ra_min, ra_max = float(np.nanmin(ravec)), float(np.nanmax(ravec))
    ra_ctr = 0.5 * (ra_min + ra_max)
    Dc_maxz = Planck18.comoving_distance(z_cap).value
    half_w = Dc_maxz * np.deg2rad(ra_max - ra_ctr)
    y_max = Planck18.comoving_distance(z_cap).value if coord == 'dc' else z_cap

    nrows, ncols = subplot_grid(len(tracers))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10,15), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    plt.suptitle(f'{webtype.capitalize()} groups (centers) in zone {zone_tag(zone)}', fontsize=27, y=1.01, color=TEXT_COLOR)

    for i, tr in enumerate(tracers):
        tr = str(tr)
        ax = axes[i]
        _init_ax(ax, tr + ' centers')

        m = (ag_tr == tr) & (ag_n >= int(min_npts)) & (ag_z <= z_cap)
        if not np.any(m):
            ax.text(0.5, 0.5, 'No groups', ha='center', va='center', transform=ax.transAxes, fontsize=16, color=TEXT_COLOR)
            continue

        zvals = ag_z[m]
        yvals = zvals if coord == 'z' else Planck18.comoving_distance(zvals).value
        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc_maxz, half_w, y_max, n_ra, n_z, coord)

        scale = yvals / y_max
        x = scale * (Dc_maxz * np.deg2rad(ag_ra[m] - ra_ctr))
        w_at_y = half_w * scale
        mclip = np.abs(x) <= w_at_y

        sizes = np.clip(ag_n[mclip], 4, None)
        ax.scatter(x[mclip], yvals[mclip], s=sizes, c=CENTER_SCATTER_COLOR,
                   alpha=0.85, edgecolor='none')

        _draw_borders(ax, half_w, y_max)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc_maxz, y_max)
        _annotate_y_side(ax, z_ticks, half_w, y_max, i, 'z' if coord == 'z' else r'$D_c$ [Mpc]')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(out_png)
    plt.close(fig)
    return out_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--release', choices=['EDR','DR1'], default='edr', help='Dataset release: EDR (zones 00..19) or DR1 (NGC1/NGC2)')
    p.add_argument('--base-dir', default='/pscratch/sd/v/vtorresg/cosmic-web', help='Root directory containing release subdirectories (e.g., edr/raw, edr/groups, edr/probabilities)')
    p.add_argument('--output', default=None, help='Output directory for figures (defaults to {base}/{release}/figs/{mode})')
    p.add_argument('--out-tag', type=str, default=None, help='Tag appended to filenames (e.g., tracer)')
    p.add_argument('--zone', type=str, default='all', help='EDR: 0..19 or "all" (default processes every zone); DR1: NGC1/NGC2')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot','all'], default='filament', help='Web classification to load ("all" when plotting types)')
    p.add_argument('--mode', choices=['groups','types','structure'], default='groups', help='Plot style: group colours, web-type colours, or monochrome structure overview')
    p.add_argument('--tracers', nargs='*', default=None)
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--randiters', nargs='*', type=int, default=None, metavar='ITER', help='Restrict to specific RANDITER values (e.g., -1 for data, 0 for first random iteration)')
    p.add_argument('--smin', type=int, default=1)
    p.add_argument('--max-z', type=float, default=None)
    p.add_argument('--coord', choices=['z','dc'], default='z', help='Y-axis coordinate: redshift z (default) or comoving distance Dc')
    p.add_argument('--bins', type=int, default=10, help='Number of grid bins for RA and z/Dc')
    p.add_argument('--connect-lines', action='store_true', help='Connect points that belong to the same group with a line')
    p.add_argument('--line-min-npts', type=int, default=2, help='Minimum number of points in a group to draw its connecting line')
    p.add_argument('--min-npts', type=int, default=5, help='Filter: show only groups with at least this many members')
    p.add_argument('--top-groups', type=int, default=None, help='Filter: keep only the top-N largest groups per tracer')
    p.add_argument('--max-points', type=int, default=None, help='Cap points per tracer subplot (randomly subsamples if exceeded)')
    p.add_argument('--z-range', nargs=2, type=float, default=None, help='Optional z range [zmin zmax] to focus the view')
    p.add_argument('--ra-range', nargs=2, type=float, default=None, help='Optional RA range [ramin ramax] to focus the view')
    p.add_argument('--plot-centers', action='store_true', help='Also plot per-group centers (one dot per group)')
    p.add_argument('--center-min-npts', type=int, default=2, help='Minimum number of members for a group to be shown as center')
    p.add_argument('--use-presets', action='store_true', help='Apply tracer-specific defaults to emphasize filaments (e.g., higher min-npts for LRG)')
    p.add_argument('--highlight-longest', type=int, default=None, help='Highlight top-K longest groups per tracer (projected)')
    p.add_argument('--highlight-connect', action='store_true', help='Connect points for highlighted groups')
    return p.parse_args()


def main():
    args = parse_args()

    mode = args.mode.lower()
    webtype_arg = str(args.webtype).lower()
    if mode == 'groups' and webtype_arg == 'all':
        raise ValueError('Use --mode types to plot all web classifications together.')

    base_dir = os.path.abspath(str(args.base_dir))
    release_token = str(args.release).lower()
    release_dir = os.path.join(base_dir, release_token)
    raw_dir = os.path.join(release_dir, 'raw')
    groups_dir = os.path.join(release_dir, 'groups')
    prob_root = release_dir
    if args.output:
        output_dir = os.path.abspath(str(args.output))
    else:
        output_dir = os.path.join(release_dir, 'figs', mode)

    os.makedirs(output_dir, exist_ok=True)

    release = args.release.upper()
    zones = resolve_zones(release, args.zone)

    for zone in zones:
        raw = read_raw_min(raw_dir, release_dir, zone, out_tag=args.out_tag)
        rm = mask_source(np.asarray(raw['RANDITER']), args.source)
        raw = raw[rm]
        raw = filter_randiters(raw, args.randiters)

        if len(raw) == 0:
            print(f'Skipping zone {zone_tag(zone)}: raw catalogue empty after filters.', file=sys.stderr)
            continue

        zone_z = np.asarray(raw['Z'], dtype=float)
        if zone_z.size == 0:
            print(f'Skipping zone {zone_tag(zone)}: no Z values available.', file=sys.stderr)
            continue

        if args.max_z is not None:
            global_z_cap = float(args.max_z)
        else:
            global_z_cap = float(np.nanmax(zone_z)) * 1.02

        if args.z_range is not None and len(args.z_range) == 2:
            try:
                z_hi = float(args.z_range[1])
                global_z_cap = min(global_z_cap, z_hi)
            except Exception:
                pass

        raw_tr_pref = tracer_prefixes(np.asarray(raw['TRACERTYPE']).astype(str))
        tracer_caps = {}
        for tr in ORDERED_TRACERS:
            mask_tr = (raw_tr_pref == tr)
            if not np.any(mask_tr):
                continue
            zvals_tr = zone_z[mask_tr]
            if zvals_tr.size == 0:
                continue
            tracer_caps[tr] = float(np.nanmax(zvals_tr)) * 1.02
        if tracer_caps:
            for key, val in list(tracer_caps.items()):
                if args.z_range is not None and len(args.z_range) == 2:
                    try:
                        z_hi = float(args.z_range[1])
                        tracer_caps[key] = min(val, z_hi)
                    except Exception:
                        pass
                if args.max_z is not None:
                    tracer_caps[key] = min(tracer_caps[key], float(args.max_z))
            global_z_cap = max(tracer_caps.values())

        joined = None
        plot_table = raw
        legend_types = []

        if mode == 'groups':
            requested_webtypes = [webtype_arg]
            group_tables = []
            missing_webtypes = []
            active_webtypes = []
            for wtype in requested_webtypes:
                try:
                    g = read_groups(groups_dir, zone, wtype, out_tag=args.out_tag)
                except FileNotFoundError:
                    missing_webtypes.append(wtype)
                    continue
                gm = mask_source(np.asarray(g['RANDITER']), args.source)
                g = g[gm]
                g = filter_randiters(g, args.randiters)
                if len(g) == 0:
                    continue
                if wtype not in active_webtypes:
                    active_webtypes.append(wtype)
                g = g.copy()
                g['WEBTYPE'] = np.array([wtype] * len(g), dtype='U8')
                group_tables.append(g)

            if missing_webtypes:
                missing = ', '.join(sorted(missing_webtypes))
                raise RuntimeError(f'No groups files found for zone {zone_tag(zone)} and webtype(s): {missing}')

            if not group_tables:
                print(f'Skipping zone {zone_tag(zone)}: no groups available after filters.', file=sys.stderr)
                continue

            groups = group_tables[0] if len(group_tables) == 1 else vstack(group_tables, metadata_conflicts='silent')
            joined = join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')
            if len(joined) == 0:
                print(f'Skipping zone {zone_tag(zone)}: join between groups and raw catalog returned no rows.', file=sys.stderr)
                continue
            plot_table = joined
            legend_types = active_webtypes or requested_webtypes

        elif mode == 'types':
            try:
                prob_path = resolve_probability_path(prob_root, zone, out_tag=args.out_tag)
            except FileNotFoundError:
                print(f'Skipping zone {zone_tag(zone)}: probability file not found.', file=sys.stderr)
                continue

            prob_tbl = Table.read(prob_path, memmap=True)
            prob_tbl = classify_by_probability(prob_tbl)

            if args.source == 'data':
                prob_tbl = prob_tbl[prob_tbl['ISDATA'] == True]
            elif args.source == 'rand':
                prob_tbl = prob_tbl[prob_tbl['ISDATA'] == False]

            if len(prob_tbl) == 0:
                print(f'Skipping zone {zone_tag(zone)}: probability table empty after source filter.', file=sys.stderr)
                continue

            if args.randiters is not None:
                mask_iters = np.isin(np.asarray(prob_tbl['RANDITER'], dtype=np.int64),
                                     np.asarray(list(args.randiters), dtype=np.int64))
                prob_tbl = prob_tbl[mask_iters]
                if len(prob_tbl) == 0:
                    print(f'Skipping zone {zone_tag(zone)}: no rows left after RANDITER filter.', file=sys.stderr)
                    continue

            good = prob_tbl['WEBTYPE'] != ''
            prob_tbl = prob_tbl[good]
            if len(prob_tbl) == 0:
                print(f'Skipping zone {zone_tag(zone)}: no valid classifications after filtering.', file=sys.stderr)
                continue

            keep_cols = [col for col in ('TARGETID','RANDITER','WEBTYPE','ISDATA') if col in prob_tbl.colnames]
            prob_keep = prob_tbl[keep_cols]

            joined_types = join(raw, prob_keep, keys=['TARGETID','RANDITER'], join_type='inner')
            if len(joined_types) == 0:
                print(f'Skipping zone {zone_tag(zone)}: no overlap between raw and probability tables.', file=sys.stderr)
                continue

            plot_table = joined_types
            present = np.char.lower(np.unique(np.asarray(plot_table['WEBTYPE']).astype(str)))
            legend_types = [wt for wt in ALL_WEBTYPES if wt in present]

        else:
            legend_types = []

        available = tracer_prefixes(np.asarray(plot_table['TRACERTYPE']).astype(str))
        avail_set = set(map(str, available))

        if args.tracers:
            tokens = []
            for t in args.tracers:
                tokens.extend(str(t).replace(',', ' ').split())
            req_pref = [tok.split('_', 1)[0].upper() for tok in tokens]
            req_set = set(req_pref)
            tracers = [t for t in ORDERED_TRACERS if (t in req_set) and (t in avail_set)]
        else:
            tracers = [t for t in ORDERED_TRACERS if t in avail_set]

        if not tracers:
            raise RuntimeError(f'None of the requested tracers are present after filtering for zone {zone_tag(zone)}.')

        tag = zone_tag(zone)
        tsuf = safe_tag(args.out_tag)

        if mode == 'groups':
            out_png = os.path.join(output_dir, f'groups_zone_{tag}{tsuf}_{webtype_arg}_{args.coord}.png')
        elif mode == 'types':
            out_png = os.path.join(output_dir, f'webtypes_zone_{tag}{tsuf}_{webtype_arg}_{args.coord}.png')
        else:
            out_png = os.path.join(output_dir, f'data_zone_{tag}{tsuf}_{webtype_arg}_{args.coord}.png')

        zone_label = tag
        legend_order = legend_types
        mode_kwargs = {
            'groups': dict(color_mode='group', title=f"{webtype_arg.capitalize()}s in zone {int(zone_label)}"),
            'types': dict(color_mode='webtype', title=f"Web types in zone {int(zone_label)}", webtype_order=legend_order),
            'structure': dict(color_mode='mono', title=f"Zone {int(zone_label)}")
        }

        mode_args = mode_kwargs[mode]
        webtype_for_plot = 'all' if mode == 'types' else webtype_arg
        plot_wedges(plot_table, tracers, zone, webtype_for_plot, out_png, args.smin, global_z_cap,
                    n_ra=args.bins, n_z=args.bins, coord=args.coord, connect_lines=args.connect_lines,
                    line_min_npts=args.line_min_npts,
                    min_npts=args.min_npts, top_groups=args.top_groups, max_points=args.max_points,
                    z_range=args.z_range, ra_range=args.ra_range,
                    use_presets=args.use_presets, highlight_longest=args.highlight_longest,
                    highlight_connect=args.highlight_connect, per_tracer_caps=tracer_caps, **mode_args)

        if args.plot_centers:
            if joined is None:
                print(f'Skipping group centers for zone {zone_label}: no group catalogue available.', file=sys.stderr)
            else:
                center_label = 'types' if mode == 'types' else webtype_arg
                out_cent = os.path.join(output_dir, f'groups_centers_zone_{tag}{tsuf}_{center_label}_{args.coord}.png')
                plot_group_centers(joined, tracers, zone, center_label, out_cent, min_npts=args.center_min_npts,
                                   max_z=global_z_cap, n_ra=args.bins, n_z=args.bins, coord=args.coord)


if __name__ == '__main__':
    main()