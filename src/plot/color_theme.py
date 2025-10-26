import os
from typing import Dict, Tuple, Optional

import matplotlib

_COLOR_THEMES: Dict[str, Dict[str, object]] = {
    'light': {
        'text': '#111111',
        'primary': '#0f0f0f',
        'secondary': '#7a7a7a',
        'mono': '#151515',
        'highlight_edge': '#000000',
        'scatter_edge': '#222222',
        'group_palette': 'tab20',
        'center_color': '#2b7bb9',
        'figure_facecolor': '#ffffff',
        'axes_facecolor': '#ffffff',
        'class_colors': {
            'void': 'crimson',
            'sheet': 'orange',
            'filament': 'cornflowerblue',
            'knot': 'midnightblue',
        },
        'class_fallback': '#808080',
    },
    'dark': {
        'text': '#f2f2f2',
        'primary': '#f2f2f2',
        'secondary': '#9a9a9a',
        'mono': '#e0e0e0',
        'highlight_edge': '#f5f5f5',
        'scatter_edge': 'black',
        'group_palette': 'tab20',
        'center_color': '#6ea8ff',
        'figure_facecolor': 'black',
        'axes_facecolor': 'black',
        'class_colors': {
            'void': '#23EBC3',
            'sheet': '#005AFF',
            'filament': '#9B6A12',
            'knot': '#E6E68F',
        },
        'class_fallback': '#b0b0b0',
    },
}


def _normalise_name(name: Optional[str], default: str) -> str:
    value = name or default
    return str(value).strip().lower()


def available_themes() -> Tuple[str, ...]:
    """Return the list of registered theme names."""

    return tuple(sorted(_COLOR_THEMES))


def load_theme(env_var: str = 'PLOT_THEME', default: str = 'light') -> Tuple[str, Dict[str, object]]:
    """Resolve the theme dictionary.

    Looks first at ``env_var``; if unset, falls back to ``PLOT_THEME`` and
    finally to ``default``.
    """

    env_value = os.environ.get(env_var)
    if env_var != 'PLOT_THEME' and env_value is None:
        env_value = os.environ.get('PLOT_THEME')
    theme_name = _normalise_name(env_value, default)
    try:
        theme = _COLOR_THEMES[theme_name]
    except KeyError as exc:
        opts = ', '.join(available_themes())
        raise ValueError(f"Unknown colour theme '{theme_name}'. Options: {opts}") from exc
    return theme_name, theme


def apply_matplotlib_theme(theme: Dict[str, object]) -> None:
    """Apply the relevant ``matplotlib`` rcParams for *theme*."""

    text_color = theme['text']
    secondary = theme['secondary']

    matplotlib.rcParams.update({
        'axes.labelcolor': text_color,
        'axes.edgecolor': secondary,
        'axes.titlecolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'text.color': text_color,
        'axes.facecolor': theme['axes_facecolor'],
        'figure.facecolor': theme['figure_facecolor'],
        'savefig.facecolor': theme['figure_facecolor'],
    })


__all__ = ['available_themes', 'load_theme', 'apply_matplotlib_theme']