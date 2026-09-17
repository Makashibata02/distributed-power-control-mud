"""Shared publication-style plotting helpers for Part III figures."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

from .part2_style import (
    MARKERS,
    METHOD_LABELS,
    RECEIVER_LABELS,
    active_font,
    apply_part2_style as _apply_base_style,
    label_for_init,
    label_for_method,
    label_for_receiver,
    save_figure,
    style_axes,
)


CH4_COLORS = {
    "gray": "#979998",
    "rose": "#C69287",
    "coral": "#E79A90",
    "peach": "#EFBC91",
    "yellow": "#E4CD87",
    "cream": "#FAE5B8",
    "light_gray": "#DDDDDF",
    "dark": "#262626",
    "single": "#979998",
    "uniform": "#E4CD87",
    "random": "#C69287",
    "local": "#EFBC91",
    "pso": "#E79A90",
    "gain": "#E79A90",
    "global_od": "#979998",
    "group_od": "#C69287",
    "mf": "#EFBC91",
    "lmmse": "#E4CD87",
}

EXTRA_COLORS = [
    "#FAE5B8",
    "#DDDDDF",
    "#EFBC91",
    "#C69287",
    "#E79A90",
]

COLOR_ALIASES = {
    "single_power": "single",
    "structured_uniform_exp": "uniform",
    "random_dirichlet": "random",
    "random_dirichlet_best": "random",
    "local_search": "local",
    "structured_best": "peach",
    "structured_all": "peach",
    "PSO_MC": "pso",
    "deterministic": "gray",
    "mc": "yellow",
    "probability": "yellow",
    "beta": "rose",
    "power": "peach",
    "success": "coral",
    "global_od": "global_od",
    "group_od": "group_od",
    "group_mf": "mf",
    "group_lmmse": "lmmse",
    "od": "global_od",
    "mf": "mf",
    "lmmse": "lmmse",
}

COLOR_PALETTE = {**CH4_COLORS, **{key: CH4_COLORS[val] for key, val in COLOR_ALIASES.items()}}
PAPER_COLORS = CH4_COLORS

LINE_STYLES = {
    "single_power": "--",
    "structured_uniform_exp": "-",
    "random_dirichlet": "-.",
    "random_dirichlet_best": "-.",
    "local_search": "-",
    "PSO_MC": "-",
    "global_od": "-",
    "group_od": "--",
    "group_mf": "-.",
    "group_lmmse": ":",
}


def color_for(key: str, index: int | None = None) -> str:
    color_key = COLOR_ALIASES.get(key, key)
    if color_key in CH4_COLORS:
        return CH4_COLORS[color_key]
    if index is None:
        index = abs(hash(key))
    return EXTRA_COLORS[index % len(EXTRA_COLORS)]


def colors_for(keys: Iterable[str]) -> list[str]:
    return [color_for(key, idx) for idx, key in enumerate(keys)]


def apply_part3_style(verbose: bool = True) -> str:
    font = _apply_base_style(verbose=False)
    plt.rcParams.update(
        {
            "axes.edgecolor": CH4_COLORS["dark"],
            "grid.color": CH4_COLORS["light_gray"],
            "patch.edgecolor": "white",
            "patch.linewidth": 0.8,
        }
    )
    if verbose:
        print(f"[Part III plot style] 使用中文字体: {font}")
    return font


def apply_part2_style(verbose: bool = True) -> str:
    return apply_part3_style(verbose=verbose)


def setup_paper_style(verbose: bool = True) -> str:
    return apply_part3_style(verbose=verbose)
