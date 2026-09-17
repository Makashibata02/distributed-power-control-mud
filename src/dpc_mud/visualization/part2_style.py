"""Shared publication-style plotting helpers for Part II figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager


PAPER_COLORS = {
    "single": "#B7B7EB",
    "uniform": "#EAB883",
    "random": "#9D9EA3",
    "local": "#F09BA0",
    "gain": "#F09BA0",
    "global_od": "#B7B7EB",
    "group_od": "#9BBBE1",
    "mf": "#EAB883",
    "lmmse": "#F09BA0",
    "gray": "#9D9EA3",
    "light_gray": "#D9D9D9",
    "dark": "#262626",
}

EXTRA_COLORS = [
    "#9BBBE1",
    "#B7B7EB",
    "#EAB883",
    "#F09BA0",
    "#9D9EA3",
]

COLOR_ALIASES = {
    "single_power": "single",
    "structured_uniform_exp": "uniform",
    "random_dirichlet": "random",
    "random_dirichlet_best": "random",
    "local_search": "local",
    "pso": "gain",
    "PSO_MC": "gain",
    "structured_best": "global_od",
    "structured_all": "global_od",
    "group_mf": "mf",
    "group_lmmse": "lmmse",
    "deterministic": "single",
    "mc": "uniform",
    "power": "uniform",
    "probability": "single",
    "beta": "local",
    "success": "random",
}

COLOR_PALETTE = {**PAPER_COLORS, **{key: PAPER_COLORS[val] for key, val in COLOR_ALIASES.items()}}

PAPER_LINESTYLES = {
    "single": "--",
    "uniform": "-",
    "random": "-.",
    "local": "-",
    "global_od": "-",
    "group_od": "--",
    "mf": "-.",
    "lmmse": ":",
}

LINE_STYLES = {
    **PAPER_LINESTYLES,
    "single_power": "--",
    "structured_uniform_exp": "-",
    "structured_best": "-.",
    "structured_all": "-.",
    "random_dirichlet": ":",
    "random_dirichlet_best": ":",
    "local_search": "-",
    "PSO_MC": "-",
    "global_od": "-",
    "group_od": "--",
    "group_mf": "-.",
    "group_lmmse": ":",
}

PAPER_MARKERS = {
    "single": "o",
    "uniform": "s",
    "random": "^",
    "local": "D",
    "global_od": "o",
    "group_od": "s",
    "mf": "^",
    "lmmse": "D",
}

MARKERS = {
    **PAPER_MARKERS,
    "single_power": "o",
    "structured_uniform_exp": "s",
    "structured_best": "^",
    "structured_all": "^",
    "random_dirichlet": "D",
    "random_dirichlet_best": "D",
    "local_search": "P",
    "PSO_MC": "X",
    "global_od": "o",
    "group_od": "s",
    "group_mf": "^",
    "group_lmmse": "D",
}

METHOD_LABELS = {
    "single_power": "单功率接入基线",
    "structured_uniform_exp": "均匀概率-指数功率基线",
    "structured_best": "结构化初始化基线",
    "structured_all": "结构化初始化基线",
    "random_dirichlet": "随机初始化基线",
    "random_dirichlet_best": "随机初始化基线",
    "local_search": "局部搜索基线",
    "PSO_MC": "PSO优化",
}

RECEIVER_LABELS = {
    "global_od": "全局OD",
    "group_od": "分组OD",
    "group_mf": "MF",
    "group_lmmse": "LMMSE",
    "od": "OD",
    "mf": "MF",
    "lmmse": "LMMSE",
}

INIT_BETA_LABELS = {
    "uniform": "均匀分布",
    "exp_increasing": "指数递增",
    "exp_decreasing": "指数递减",
    "gaussian": "高斯型",
    "poisson": "泊松型",
    "single_highest_layer": "单最高层",
    "dirichlet": "狄利克雷",
}

_ACTIVE_FONT = "DejaVu Sans"


def color_for(key: str, index: int | None = None) -> str:
    color_key = COLOR_ALIASES.get(key, key)
    if color_key in PAPER_COLORS:
        return PAPER_COLORS[color_key]
    if index is None:
        index = abs(hash(key))
    return EXTRA_COLORS[index % len(EXTRA_COLORS)]


def label_for_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def label_for_receiver(receiver: str) -> str:
    return RECEIVER_LABELS.get(receiver, receiver)


def label_for_init(init_name: str) -> str:
    return INIT_BETA_LABELS.get(init_name, init_name)


def apply_part2_style(verbose: bool = True) -> str:
    global _ACTIVE_FONT
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            _ACTIVE_FONT = name
            break
    else:
        plt.rcParams["font.sans-serif"] = preferred_fonts
        _ACTIVE_FONT = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_COLORS["dark"],
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "grid.color": PAPER_COLORS["light_gray"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "patch.edgecolor": "white",
            "patch.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if verbose:
        print(f"[Part II plot style] 使用中文字体: {_ACTIVE_FONT}")
    return _ACTIVE_FONT


def setup_paper_style(verbose: bool = True) -> str:
    return apply_part2_style(verbose=verbose)


def active_font() -> str:
    return _ACTIVE_FONT


def style_axes(ax, *, grid_axis: str = "y") -> None:
    polish_axes(ax, grid_axis=grid_axis)


def polish_axes(ax, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color(PAPER_COLORS["dark"])
    ax.spines["bottom"].set_color(PAPER_COLORS["dark"])
    ax.tick_params(axis="both", direction="out", length=4, width=0.8, colors=PAPER_COLORS["dark"])


def _save_svg_without_titles(fig, path: Path, **kwargs) -> None:
    axis_titles = []
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            axis_titles.append((ax, loc, ax.get_title(loc=loc)))
            ax.set_title("", loc=loc)

    suptitle = getattr(fig, "_suptitle", None)
    suptitle_text = suptitle.get_text() if suptitle is not None else None
    if suptitle is not None:
        suptitle.set_text("")

    try:
        fig.savefig(path, **kwargs)
    finally:
        for ax, loc, title in axis_titles:
            ax.set_title(title, loc=loc)
        if suptitle is not None:
            suptitle.set_text(suptitle_text)


def save_figure(fig, fig_dir: Path, name: str, dpi: int = 300) -> tuple[Path, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.lower().endswith(".png") else name
    png_path = fig_dir / f"{stem}.png"
    pdf_path = fig_dir / f"{stem}.pdf"
    svg_path = fig_dir / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png_path, dpi=max(int(dpi), 300), bbox_inches="tight")
    _save_svg_without_titles(fig, svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def colors_for(keys: Iterable[str]) -> list[str]:
    return [color_for(key, idx) for idx, key in enumerate(keys)]
