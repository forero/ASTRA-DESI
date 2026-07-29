import os
from pathlib import Path
from typing import Mapping

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/astra-desi-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/astra-desi-cache')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif',
                         'axes.grid': True,
                         'grid.alpha': 0.18,
                         'figure.facecolor': 'black',
                         'axes.facecolor': 'black',
                         'savefig.facecolor': 'black'})

from .read_data import TRACER_DISPLAY, normalize_tracer, normalize_zone


TRACER_COLORS = {'BGS_BRIGHT': 'cyan',
                 'LRG': 'orange',
                 'ELG_LOPnotqso': 'limegreen',
                 'QSO': 'magenta'}
ZONE_LINESTYLES = {'NGC': '-', 'SGC': '--'}


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _bin_count(requested, samples):
    requested = int(requested)
    if requested < 1:
        raise ValueError('The requested bin count must be positive.')
    sizes = [len(_finite(values)) for values in samples.values()]
    positive = [size for size in sizes if size > 0]
    if not positive:
        raise ValueError('Cannot plot empty samples.')
    sample_limit = max(5, int(np.ceil(np.sqrt(2.0 * min(positive)))))
    return min(requested, sample_limit)


def _common_edges(variable, samples: Mapping, requested):
    arrays = [_finite(values) for values in samples.values()]
    pooled = np.concatenate([values for values in arrays if len(values)])
    if len(pooled) == 0:
        raise ValueError(f'No finite {variable} values are available.')
    n_bins = _bin_count(requested, samples)
    if variable == 'ELLIP':
        lower = 0.0
        upper = max(0.6, float(np.max(pooled)))
        upper = min(1.0, upper)
    elif variable == 'R_EFF':
        lower = 0.0
        maximum = float(np.max(pooled))
        upper = max(10.0, 10.0 * np.ceil(maximum / 10.0))
    else:
        raise ValueError(f'Unknown variable {variable!r}.')
    if upper <= lower:
        upper = lower + 1.0
    return np.linspace(lower, upper, n_bins + 1)


def _pdf_and_sigma(values, edges, n_bootstrap, rng: np.random.Generator):
    values = _finite(values)
    counts, _ = np.histogram(values, bins=edges)
    n_used = int(np.sum(counts))
    if n_used == 0:
        raise ValueError('A plotted sample has no values inside the bins.')
    widths = np.diff(edges)
    probabilities = counts / float(n_used)
    pdf = probabilities / widths
    bootstrap_counts = rng.multinomial(n_used, probabilities, size=int(n_bootstrap))
    bootstrap_pdf = bootstrap_counts / (float(n_used) * widths[None, :])
    sigma = np.std(bootstrap_pdf, axis=0, ddof=1)
    return {'pdf': pdf, 'sigma': sigma, 'lower': np.clip(pdf - sigma, 0.0, None),
            'upper': pdf + sigma, 'counts': counts, 'n': n_used}


def _zone_significance(sgc_values, ngc_values, edges, n_bootstrap, min_combined_count, rng: np.random.Generator):
    ngc = _finite(ngc_values)
    sgc = _finite(sgc_values)
    ngc_counts, _ = np.histogram(ngc, bins=edges)
    sgc_counts, _ = np.histogram(sgc, bins=edges)
    n_ngc = int(np.sum(ngc_counts))
    n_sgc = int(np.sum(sgc_counts))
    result = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    if n_ngc == 0 or n_sgc == 0:
        return result

    combined = ngc_counts + sgc_counts
    probability = combined / float(np.sum(combined))
    widths = np.diff(edges)
    ngc_null = rng.multinomial(n_ngc, probability, size=int(n_bootstrap))
    sgc_null = rng.multinomial(n_sgc, probability, size=int(n_bootstrap))
    null_delta = (ngc_null / (float(n_ngc) * widths[None, :]) - sgc_null / (float(n_sgc) * widths[None, :]))
    sigma = np.std(null_delta, axis=0, ddof=1)
    observed = (ngc_counts / (float(n_ngc) * widths) - sgc_counts / (float(n_sgc) * widths))
    supported = ((sigma > 0.0) & (combined >= int(min_combined_count)))
    np.divide(observed, sigma, out=result, where=supported)
    return result


def _normalized_samples(samples: Mapping):
    normalized = {}
    for key, values in samples.items():
        if len(key) != 2:
            raise ValueError('Sample keys must be (tracer, zone).')
        tracer = normalize_tracer(key[0])
        zone = normalize_zone(key[1])
        normalized[(tracer, zone)] = {'ELLIP': _finite(values['ELLIP']),
                                      'R_EFF': _finite(values['R_EFF'])}
    return normalized


def plot_all_tracers(samples: Mapping, output_path, iteration = 0, r_threshold = -0.25,
                     ellip_bins = 30, reff_bins = 30, n_bootstrap = 2000, seed = 12345,
                     min_combined_count = 5, use_tex = True):

    samples = _normalized_samples(samples)
    if not samples:
        raise ValueError('At least one tracer/zone sample is required.')
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 2:
        raise ValueError('n_bootstrap must be at least 2.')

    # plt.style.use('dark_background')
    plt.rcParams.update({'text.usetex': True})
    if not bool(use_tex):
        plt.rcParams.update({'text.usetex': False})

    variable_samples = {
        variable: {key: values[variable] for key, values in samples.items()
                   if len(values[variable])} for variable in ('ELLIP', 'R_EFF')}
    edges_by_variable = {'ELLIP': _common_edges('ELLIP', variable_samples['ELLIP'], ellip_bins),
                         'R_EFF': _common_edges('R_EFF', variable_samples['R_EFF'], reff_bins)}
    seed_sequence = np.random.SeedSequence(int(seed))
    child_seeds = iter(seed_sequence.spawn(2 * len(samples) + 2 * len(TRACER_COLORS) + 4))

    figure = plt.figure(figsize=(15, 6))
    outer = figure.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.10, top=0.82, wspace=0.16)
    labels = {'ELLIP': r'$\epsilon$', 'R_EFF': r'$R_{\mathrm{eff}}\,[\mathrm{Mpc}/h]$'}

    try:
        for column, variable in enumerate(('ELLIP', 'R_EFF')):
            edges = edges_by_variable[variable]
            centers = 0.5 * (edges[:-1] + edges[1:])
            inner = outer[0, column].subgridspec(2, 1, height_ratios=(3.1, 1.0), hspace=0.05)
            upper = figure.add_subplot(inner[0])
            lower = figure.add_subplot(inner[1], sharex=upper)

            for tracer in TRACER_COLORS:
                for zone in ('NGC', 'SGC'):
                    key = (tracer, zone)
                    if key not in samples or len(samples[key][variable]) == 0:
                        continue
                    estimate = _pdf_and_sigma(samples[key][variable], edges, n_bootstrap=n_bootstrap,
                                              rng=np.random.default_rng(next(child_seeds)))
                    color = TRACER_COLORS[tracer]
                    linestyle = ZONE_LINESTYLES[zone]
                    upper.fill_between(centers,
                                       estimate['lower'],
                                       estimate['upper'],
                                       color=color,
                                       alpha=0.09,
                                       linewidth=0.0)
                    upper.plot(centers,
                               estimate['pdf'],
                               color=color,
                               linestyle=linestyle,
                               linewidth=1.8)

                ngc_key = (tracer, 'NGC')
                sgc_key = (tracer, 'SGC')
                if ngc_key not in samples or sgc_key not in samples:
                    continue
                significance = _zone_significance(samples[ngc_key][variable],
                                                  samples[sgc_key][variable],
                                                  edges,
                                                  n_bootstrap=n_bootstrap,
                                                  min_combined_count=min_combined_count,
                                                  rng=np.random.default_rng(next(child_seeds)))
                lower.plot(centers, significance, color=TRACER_COLORS[tracer], linewidth=1.35)

            upper.set_title(labels[variable], fontsize=16)
            upper.set_ylabel(r'$\mathrm{Normalized\ PDF}$')
            upper.set_ylim(bottom=0.0)
            upper.tick_params(labelbottom=False)
            lower.axhspan(-1.0, 1.0, color='white', alpha=0.12, zorder=0)
            lower.axhline(0.0, color='white', alpha=0.5, linestyle='--')
            lower.axhline(2.0, color='white', alpha=0.35, linestyle=':')
            lower.axhline(-2.0, color='white', alpha=0.35, linestyle=':')
            lower.set_ylabel(r'$\Delta_{\mathrm{NGC-SGC}}/\sigma_{\mathrm{null}}$')
            lower.set_xlabel(labels[variable])
            lower.set_ylim(-3.0, 3.0)
            upper.set_xlim(edges[0], edges[-1])

        tracer_handles = [Line2D([0], [0],
                                 color=TRACER_COLORS[tracer],
                                 linewidth=2.2,
                                 label=TRACER_DISPLAY[tracer])
            for tracer in TRACER_COLORS
            if any(key[0] == tracer for key in samples)]
        zone_handles = [Line2D([0], [0],
                                color='white',
                                linestyle=ZONE_LINESTYLES[zone],
                                linewidth=1.8,
                                label=zone)
            for zone in ('NGC', 'SGC')
            if any(key[1] == zone for key in samples)]
        figure.legend(handles=tracer_handles + zone_handles,
                      loc='upper center',
                      ncol=max(1, len(tracer_handles) + len(zone_handles)),
                      frameon=False,
                      bbox_to_anchor=(0.5, 0.965))
        figure.suptitle(rf'$\mathrm{{ASTRA}}\quad'
                        rf'\mathrm{{RANDITER}}={int(iteration)},\quad{{}}'
                        rf'r_{{\mathrm{{threshold}}}}={float(r_threshold):.3f}$',
                        y=0.995, fontsize=15)

        output_path = Path(output_path)
        if not output_path.suffix:
            raise ValueError('output_path must have a figure suffix.')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f'.{output_path.stem}.{os.getpid()}{output_path.suffix}')
        figure.savefig(temporary, dpi=360, bbox_inches='tight')
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        if 'temporary' in locals() and temporary.exists():
            temporary.unlink()
    return output_path


__all__ = ['TRACER_COLORS',
           'ZONE_LINESTYLES',
           'plot_all_tracers']