import argparse
import os
from pathlib import Path


def _configure_matplotlib():
    cache = Path('/tmp') / 'astra-matplotlib-{}'.format(os.getuid())
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    font_cache = cache / 'fontconfig'
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('XDG_CACHE_HOME', str(cache))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'text.usetex': True,
                        #  'font.family': 'serif',
                        #  'font.serif': ['Computer Modern Roman'],
                         'figure.facecolor': 'white',
                         'axes.facecolor': 'white',
                         'savefig.facecolor': 'white',
                         'axes.edgecolor': 'black',
                         'axes.linewidth': 0.8,
                         'axes.grid': True,
                         'grid.color': '0.72',
                         'grid.linestyle': '-',
                         'grid.linewidth': 0.3,
                         'grid.alpha': 0.8,
                         'xtick.direction': 'in',
                         'ytick.direction': 'in',
                         'xtick.top': True,
                         'ytick.right': True,
                         'legend.frameon': False})
    return plt


STYLES = {'halo_void': (r'$\mathrm{Halos\ void}$', '#0072B2', '-'),
          'halo_sheet': (r'$\mathrm{Halos\ sheet}$', '#009E73', '-'),
          'halo_filament': (r'$\mathrm{Halos\ filament}$', '#E69F00', '-'),
          'halo_knot': (r'$\mathrm{Halos\ knot}$', '#D55E00', '-'),
          'random_void': (r'$\mathrm{Random\ void}$', '#CC79A7', '--'),
          'halo_all': (r'$\mathrm{All\ halos}$', '#000000', ':')}


def _symlog_threshold(values):
    import numpy as np
    finite = np.abs(np.concatenate([value[np.isfinite(value)] for value in values]))
    finite = finite[finite > 0.0]
    if len(finite) == 0:
        return 1.0
    return max(float(np.max(finite)) * 1e-3, float(np.min(finite)))


def plot(inputs, output_base, shot_noise):
    import numpy as np
    plt = _configure_matplotlib()

    spectra = []
    for path in inputs:
        with np.load(str(path), allow_pickle=False) as data:
            sample = str(data['sample'].item())
            if sample not in STYLES:
                raise ValueError('Unknown sample {!r} in {}'.format(sample, path))
            p0_field = ('Pk0_shot_subtracted'
                        if shot_noise == 'subtracted' else 'Pk0_raw')
            if p0_field not in data:
                p0_field = 'Pk0'
            spectra.append({'sample': sample,
                            'k': np.asarray(data['k'], dtype=np.float64).copy(),
                            'Pk0': np.asarray(data[p0_field], dtype=np.float64).copy(),
                            'Pk2': np.asarray(data['Pk2'], dtype=np.float64).copy(),
                            'Pk4': np.asarray(data['Pk4'], dtype=np.float64).copy()})

    order = {name: index for index, name in enumerate(STYLES)}
    spectra.sort(key=lambda item: order[item['sample']])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharex=True,
                             facecolor='white')
    fields = ('Pk0', 'Pk2', 'Pk4')
    p0_title = (r'$P_0(k)-P_{\mathrm{shot}}$'
                if shot_noise == 'subtracted' else r'$P_0(k)$')
    titles = (p0_title, r'$P_2(k)$', r'$P_4(k)$')

    for axis, field, title in zip(axes, fields, titles):
        values = []
        for spectrum in spectra:
            label, color, linestyle = STYLES[spectrum['sample']]
            mask = (np.isfinite(spectrum['k']) & np.isfinite(spectrum[field])
                    & (spectrum['k'] > 0.0))
            k = spectrum['k'][mask]
            power = spectrum[field][mask]
            values.append(power)
            axis.plot(k, power, color=color, linestyle=linestyle,
                      linewidth=1.45, label=label)
        axis.set_xscale('log')
        if field == 'Pk0' and all(np.all(value > 0.0) for value in values):
            axis.set_yscale('log')
        else:
            axis.set_yscale('symlog', linthresh=_symlog_threshold(values),
                            linscale=0.7)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(r'$k\,[h\,\mathrm{Mpc}^{-1}]$', fontsize=12)
        axis.grid(True, which='both', linewidth=0.3)

    axes[0].set_ylabel(
        r'$P_\ell(k)\,[(h^{-1}\,\mathrm{Mpc})^3]$', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), w_pad=1.2)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix('.png')
    pdf = output_base.with_suffix('.pdf')
    fig.savefig(str(png), dpi=250, bbox_inches='tight', facecolor='white')
    fig.savefig(str(pdf), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('pk-quijote --> wrote {}'.format(png), flush=True)
    print('pk-quijote --> wrote {}'.format(pdf), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-base', required=True)
    parser.add_argument('--shot-noise', choices=('raw', 'subtracted'), default='raw')
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()
    plot([Path(path) for path in args.inputs], Path(args.output_base),
         args.shot_noise)


if __name__ == '__main__':
    main()