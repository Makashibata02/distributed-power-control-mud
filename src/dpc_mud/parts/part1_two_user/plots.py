"""Plot helpers for Part I experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm, ListedColormap

from .experiments import RECEIVERS, Ch2Config
from .model import success_count_grid


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def _rows_for(rows: list[dict[str, float | str]], receiver: str) -> list[dict[str, float | str]]:
    return [row for row in rows if row["receiver"] == receiver]


def _save_svg_without_titles(fig: plt.Figure, path: Path) -> None:
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
        fig.savefig(path)
    finally:
        for ax, loc, title in axis_titles:
            ax.set_title(title, loc=loc)
        if suptitle is not None:
            suptitle.set_text(suptitle_text)


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(int(dpi), 300))
    _save_svg_without_titles(fig, path.with_suffix(".svg"))


def plot_feasible_regions(cfg: Ch2Config, out_dir: Path) -> None:
    grid = np.linspace(0.0, 8.0, 450)
    e1g, e2g = np.meshgrid(grid, grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    cmap = ListedColormap(["#e8e8e8", "#a9bfd6", "#4f6d8a"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    titles = {"OD": "OD接收机的可行发送功率域", "SUD": "SUD接收机的可行发送功率域", "SIC": "SIC接收机的可行发送功率域"}

    for ax, receiver in zip(axes, RECEIVERS):
        count_map = success_count_grid(receiver, e1g, e2g, cfg.rate, cfg.sigma2)
        im = ax.imshow(
            count_map,
            origin="lower",
            extent=[0, 8.0, 0, 8.0],
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        ax.set_title(titles[receiver])
        ax.set_xlabel(r"$P_1$")
        ax.set_ylabel(r"$P_2$")
        ax.grid(alpha=0.15)
        ax.plot([0, 8.0], [0, 8.0], "k--", linewidth=1.2, dashes=(4, 3))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["0用户", "1用户", "2用户"])

    fig.suptitle(f"不同接收机的可行发送功率域对比 (R={cfg.rate:.2f}, $\\sigma^2$={cfg.sigma2:.2f})", fontsize=13)
    _save_figure(fig, out_dir / "fig_2_5_1_feasible_regions.png", dpi=220)
    plt.close(fig)


def plot_throughput_vs_load(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}
    single_dashes = {"OD": (0, (6, 3)), "SUD": (3, (6, 3)), "SIC": (6, (6, 3))}
    single_markevery = {"OD": (0, 3), "SUD": (0, 2), "SIC": (1, 2)}
    single_markers = {"OD": "o", "SUD": "o", "SIC": "^"}
    curve_count = 0
    legend_handles = []
    legend_labels = []

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        loads = np.array([float(row["G"]) for row in part])
        single = np.array([float(row.get("T_single", row["T_fixed"])) for row in part])
        discrete = np.array([float(row.get("T_discrete", row["T_random"])) for row in part])

        discrete_line, = ax.plot(
            loads,
            discrete,
            color=colors[receiver],
            linestyle="-",
            marker="s",
            linewidth=1.8,
            markersize=3.0,
            label=f"{receiver} 离散分布式功率控制",
            zorder=2,
        )
        curve_count += 1
        single_line, = ax.plot(
            loads,
            single,
            color=colors[receiver],
            linestyle=single_dashes[receiver],
            marker=single_markers[receiver],
            linewidth=1.9 if receiver == "OD" else 1.2,
            markersize=3.2 if receiver == "OD" else 4.2,
            markevery=single_markevery[receiver],
            markerfacecolor=colors[receiver],
            markeredgecolor="white",
            markeredgewidth=0.55,
            alpha=0.95 if receiver == "OD" else 0.88,
            label=f"{receiver} 单一功率控制",
            zorder=3,
        )
        single_line.set_path_effects(
            [path_effects.Stroke(linewidth=3.2 if receiver == "OD" else 2.4, foreground="white"), path_effects.Normal()]
        )
        curve_count += 1
        legend_handles.extend([single_line, discrete_line])
        legend_labels.extend([f"{receiver} 单一功率控制", f"{receiver} 离散分布式功率控制"])

    if curve_count != 6:
        raise RuntimeError(f"图2.2应包含6条曲线，实际为{curve_count}条。")

    ax.set_title("图2.2 单一功率控制与离散分布式功率控制吞吐量对比")
    ax.set_xlabel("系统负载 G")
    ax.set_ylabel("系统吞吐量 T")
    ax.set_xlim(0.0, 2.0)
    ax.grid(alpha=0.15)
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=8.5)
    _save_figure(fig, out_dir / "fig2_2_single_vs_discrete_all_receivers.png", dpi=300)
    fig.savefig(out_dir / "fig2_2_single_vs_discrete_all_receivers.pdf")
    _save_figure(fig, out_dir / "fig_2_5_2_throughput_vs_load.png", dpi=220)
    plt.close(fig)


def plot_gain_vs_load(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}
    curve_count = 0

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        loads = np.array([float(row["G"]) for row in part])
        single = np.array([float(row.get("T_single", row["T_fixed"])) for row in part])
        discrete = np.array([float(row.get("T_discrete", row["T_random"])) for row in part])
        gains = discrete - single
        ax.plot(loads, gains, marker="o", linewidth=1.8, markersize=3.0, color=colors[receiver], label=f"{receiver} 增益")
        curve_count += 1

    if curve_count != 3:
        raise RuntimeError(f"图2.3应包含3条增益曲线，实际为{curve_count}条。")

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("图2.3 离散分布式功率控制相对单一功率控制的吞吐量增益")
    ax.set_xlabel("系统负载 G")
    ax.set_ylabel(r"吞吐量增益 $\Delta T$")
    ax.set_xlim(0.0, 2.0)
    ax.grid(alpha=0.15)
    ax.legend()
    _save_figure(fig, out_dir / "fig2_3_discrete_gain_over_single.png", dpi=300)
    fig.savefig(out_dir / "fig2_3_discrete_gain_over_single.pdf")
    _save_figure(fig, out_dir / "fig_2_5_3_gain_vs_load.png", dpi=220)
    plt.close(fig)


def plot_best_random_structure(rows: list[dict[str, float | str]], out_dir: Path, target_load: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    labels = [r"$\pi_0$", r"$\pi_1$", r"$\pi_2$", r"$\pi_{12}$"]
    receiver_colors = {"OD": "#4f6d8a", "SUD": "#89a8c0", "SIC": "#c7d7e5"}
    x = np.arange(len(labels))
    width = 0.23

    for idx, receiver in enumerate(RECEIVERS):
        part = _rows_for(rows, receiver)
        row = min(part, key=lambda x: abs(float(x["G"]) - target_load))
        vals = [float(row["pi0"]), float(row["pi1"]), float(row["pi2"]), float(row["pi12"])]
        offset = (idx - (len(RECEIVERS) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=receiver_colors[receiver],
            edgecolor="white",
            linewidth=0.8,
            label=receiver,
        )

    ax.set_title("离散分布式功率控制策略的最优条件概率结构 (G=1)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("功率选择概率分量")
    ax.set_ylabel("有消息用户的条件概率")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(title="接收机")
    _save_figure(fig, out_dir / "fig_2_5_4_best_random_structure.png", dpi=220)
    fig.savefig(out_dir / "fig_2_5_4_best_random_structure.pdf")
    plt.close(fig)


def plot_pi0_vs_load(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        loads = np.array([float(row["G"]) for row in part])
        pi0 = np.array([float(row["pi0"]) for row in part])
        ax.plot(loads, pi0, marker="o", linewidth=1.8, markersize=3.0, color=colors[receiver], label=receiver)

    ax.set_title("有消息用户的最优0功率选择概率")
    ax.set_xlabel("系统负载 G")
    ax.set_ylabel(r"条件概率 $\pi_0$")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15)
    ax.legend()
    _save_figure(fig, out_dir / "fig_2_5_5_pi0_vs_load.png", dpi=220)
    plt.close(fig)


def plot_gain_vs_rate(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        rates = np.array([float(row["R"]) for row in part])
        gains = np.array([float(row["max_gain_abs"]) for row in part])
        ax.plot(rates, gains, marker="o", linewidth=1.8, markersize=3.2, color=colors[receiver], label=receiver)

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("目标速率对离散分布式功率增益的影响")
    ax.set_xlabel("目标速率 R")
    ax.set_ylabel("最大吞吐量增益")
    ax.grid(alpha=0.15)
    ax.legend()
    _save_figure(fig, out_dir / "fig_2_5_6_gain_vs_rate.png", dpi=220)
    plt.close(fig)


def plot_throughput_vs_pbar(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)

    for ax, receiver in zip(axes, RECEIVERS):
        part = _rows_for(rows, receiver)
        pbar = np.array([float(row["Pbar"]) for row in part])
        fixed = np.array([float(row["T_fixed"]) for row in part])
        random = np.array([float(row["T_random"]) for row in part])
        ax.plot(pbar, fixed, color="#b55d60", marker="o", linewidth=1.8, markersize=3.2, label="单一功率")
        ax.plot(pbar, random, color="#3e6c8f", marker="s", linewidth=1.8, markersize=3.2, label="离散分布式功率")
        ax.set_title(receiver)
        ax.set_xlabel(r"平均发送功率约束 $\bar P$")
        ax.set_ylabel("吞吐量")
        ax.grid(alpha=0.15)
        ax.legend()

    fig.suptitle("平均发送功率约束对吞吐量的影响", fontsize=13)
    _save_figure(fig, out_dir / "fig_2_5_7_throughput_vs_pbar.png", dpi=220)
    plt.close(fig)


def plot_level_ablation(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        levels = np.array([int(row["num_nonzero_levels"]) for row in part])
        throughput = np.array([float(row["throughput"]) for row in part])
        ax.plot(levels, throughput, marker="o", linewidth=1.8, markersize=3.2, color=colors[receiver], label=receiver)

    ax.set_title("非零发送功率层数量消融实验")
    ax.set_xlabel("非零发送功率层数量 M")
    ax.set_ylabel("吞吐量")
    ax.grid(alpha=0.15)
    ax.legend()
    _save_figure(fig, out_dir / "fig_2_5_8_power_level_ablation.png", dpi=220)
    plt.close(fig)


def plot_bridge_to_ch3(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    threshold_rows = [row for row in rows if row["series"] == "two_user_threshold"]
    ax.scatter(
        [int(row["layer_index"]) for row in threshold_rows],
        [float(row["power"]) for row in threshold_rows],
        marker="D",
        s=64,
        color="#b55d60",
        label=r"离散分布式功率 $E_1,E_2,E_{12}$",
        zorder=4,
    )

    colors = {2: "#b55d60", 3: "#6a8f5a", 4: "#4f6d8a"}
    for layers in sorted({int(row["layer_count"]) for row in rows if row["series"] == "exponential_layers"}):
        part = [row for row in rows if row["series"] == "exponential_layers" and int(row["layer_count"]) == layers]
        x = np.arange(1, layers + 1, dtype=float)
        y = np.array([float(row["power"]) for row in part])
        ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.0, color=colors.get(layers), label=f"S={layers}指数分布式功率")

    ax.set_title("离散分布式功率到指数分布式功率结构的过渡")
    ax.set_xlabel("发送功率层编号")
    ax.set_ylabel("发送功率")
    ax.set_yscale("log")
    ax.grid(alpha=0.15, which="both")
    ax.legend()
    _save_figure(fig, out_dir / "fig_2_5_9_two_user_to_multilevel_power.png", dpi=220)
    plt.close(fig)


def plot_exponential_layer_compare(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    receiver_colors = {"OD": "#4f6d8a", "SUD": "#b55d60", "SIC": "#6a8f5a"}
    layer_styles = {
        "four_point": {"linestyle": "-", "marker": "s", "linewidth": 2.4, "markersize": 4.6, "alpha": 0.96, "label": "离散分布式功率"},
        2: {"linestyle": "--", "marker": "o", "linewidth": 1.7, "markersize": 4.2, "alpha": 0.82, "label": "2层指数分布式功率"},
        3: {"linestyle": "-.", "marker": "^", "linewidth": 1.9, "markersize": 4.8, "alpha": 0.90, "label": "3层指数分布式功率"},
        4: {"linestyle": ":", "marker": "D", "linewidth": 2.2, "markersize": 4.6, "alpha": 0.95, "label": "4层指数分布式功率"},
    }
    curve_count = 0

    for receiver in RECEIVERS:
        part = _rows_for(rows, receiver)
        loads_all = sorted({float(row["G"]) for row in part})
        four_by_load = {}
        for row in part:
            four_by_load[float(row["G"])] = float(row["T_four_point"])

        style = layer_styles["four_point"]
        line, = ax.plot(
            np.array(loads_all),
            np.array([four_by_load[x] for x in loads_all]),
            color=receiver_colors[receiver],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=receiver_colors[receiver],
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=style["alpha"],
            label=f"{receiver} {style['label']}",
            zorder=4,
        )
        line.set_path_effects([path_effects.Stroke(linewidth=style["linewidth"] + 1.6, foreground="white"), path_effects.Normal()])
        curve_count += 1

        for layers in sorted({int(row["exp_layers"]) for row in part}):
            layer_rows = [row for row in part if int(row["exp_layers"]) == layers]
            loads = np.array([float(row["G"]) for row in layer_rows])
            exp = np.array([float(row["T_exp_layer"]) for row in layer_rows])
            style = layer_styles[layers]
            line, = ax.plot(
                loads,
                exp,
                color=receiver_colors[receiver],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markerfacecolor=receiver_colors[receiver],
                markeredgecolor="white",
                markeredgewidth=0.6,
                alpha=style["alpha"],
                label=f"{receiver} {style['label']}",
                markevery={2: (0, 2), 3: (1, 2), 4: (0, 3)}.get(layers, None),
                zorder=3,
            )
            line.set_path_effects([path_effects.Stroke(linewidth=style["linewidth"] + 1.2, foreground="white"), path_effects.Normal()])
            curve_count += 1

    ax.set_title("离散分布式功率与不同层数指数分布式功率结构对比")
    ax.set_xlabel("系统负载 G")
    ax.set_ylabel("吞吐量")
    ax.set_xlim(0.0, 2.0)
    ax.grid(alpha=0.15)

    receiver_handles = [
        plt.Line2D([0], [0], color=receiver_colors[receiver], linewidth=2.2, label=receiver)
        for receiver in RECEIVERS
    ]
    style_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=5.2,
            markerfacecolor="#444444",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=style["label"],
        )
        for style in layer_styles.values()
    ]
    legend1 = ax.legend(handles=receiver_handles, title="接收机", loc="upper left", fontsize=8.5)
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, title="功率层结构", loc="lower right", fontsize=8.2)

    if curve_count != len(RECEIVERS) * len(layer_styles):
        raise RuntimeError(f"指数功率层数对比图曲线数量异常：{curve_count}")

    _save_figure(fig, out_dir / "fig_2_5_10_four_point_vs_exp_layers.png", dpi=220)
    fig.savefig(out_dir / "fig_2_5_10_four_point_vs_exp_layers.pdf")
    plt.close(fig)


def plot_exponential_best_r(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {2: "#b55d60", 3: "#6a8f5a", 4: "#7b6f9b"}

    part = _rows_for(rows, "SIC")
    for layers in sorted({int(row["exp_layers"]) for row in part}):
        layer_rows = [row for row in part if int(row["exp_layers"]) == layers]
        loads = np.array([float(row["G"]) for row in layer_rows])
        best_r = np.array([float(row["best_r"]) for row in layer_rows])
        ax.plot(loads, best_r, marker="o", linewidth=1.8, markersize=3.0, color=colors.get(layers), label=f"S={layers}")

    ax.set_title("SIC下不同层数指数发送功率的最优指数比")
    ax.set_xlabel("系统负载 G")
    ax.set_ylabel("最优指数比 r")
    ax.grid(alpha=0.15)
    ax.legend()
    _save_figure(fig, out_dir / "fig_2_5_11_exp_layers_best_r_sic.png", dpi=220)
    plt.close(fig)


def plot_exponential_three_layer_compare(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    """Backward-compatible alias."""
    plot_exponential_layer_compare(rows, out_dir)

