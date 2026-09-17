"""Plot Experiment 40 baseline Monte Carlo evaluation."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dpc_mud.visualization.part3_style import apply_part2_style, color_for, label_for_method, label_for_receiver, save_figure, style_axes


RECEIVER_ORDER = ["global_od", "group_od", "group_mf", "group_lmmse"]
BASELINE_ORDER = ["single_power", "structured_uniform_exp", "random_dirichlet_best", "local_search"]


def _save(fig, fig_dir: Path, name: str, dpi: int) -> None:
    save_figure(fig, fig_dir, name, dpi=dpi)


def _available_order(df: pd.DataFrame, column: str, preferred: list[str]) -> list[str]:
    existing = set(df[column].dropna().astype(str))
    return [item for item in preferred if item in existing]


def _plot_grouped_metric(
    summary: pd.DataFrame,
    *,
    value_column: str,
    receivers: list[str],
    baselines: list[str],
    ylabel: str,
    title: str,
    ylim_from_zero: bool,
) -> plt.Figure:
    pivot = summary.pivot_table(
        index="receiver",
        columns="baseline_type",
        values=value_column,
        aggfunc="first",
    ).reindex(index=receivers, columns=baselines)

    x = list(range(len(receivers)))
    width = min(0.18, 0.72 / max(len(baselines), 1))
    offsets = [(i - (len(baselines) - 1) / 2) * width for i in range(len(baselines))]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for idx, baseline in enumerate(baselines):
        values = pivot[baseline].to_numpy(dtype=float)
        ax.bar(
            [pos + offsets[idx] for pos in x],
            values,
            width=width,
            label=label_for_method(baseline),
            color=color_for(baseline, idx),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_receiver(receiver) for receiver in receivers])
    if ylim_from_zero:
        max_value = float(pivot.max().max())
        ax.set_ylim(0.0, max_value * 1.12 if max_value > 0 else 1.0)
    else:
        min_value = float(pivot.min().min())
        max_value = float(pivot.max().max())
        span = max(max_value - min_value, 1e-3)
        ax.set_ylim(min(min_value - 0.12 * span, 0.0), max(max_value + 0.15 * span, 0.0))
        ax.axhline(0.0, color="#262626", linewidth=0.9)
    style_axes(ax)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    return fig


def plot_all(result_dir: Path, dpi: int = 300) -> Path:
    apply_part2_style()
    summary = pd.read_csv(result_dir / "ch4_mc_baseline_summary.csv")
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    receivers = _available_order(summary, "receiver", RECEIVER_ORDER)
    baselines = _available_order(summary, "baseline_type", BASELINE_ORDER)

    fig = _plot_grouped_metric(
        summary,
        value_column="mc_mean_T_over_G",
        receivers=receivers,
        baselines=baselines,
        ylabel="归一化吞吐量 T/G",
        title="图4.1 随机负载下不同基线策略的归一化吞吐量对比",
        ylim_from_zero=True,
    )
    _save(fig, fig_dir, "fig4_1_random_load_baseline_T_over_G_grouped", dpi)
    _save(fig, fig_dir, "fig_ch4_ch3_baseline_det_vs_mc_T_over_G", dpi)

    summary = summary.copy()
    summary["normalized_drop"] = summary["deterministic_T_over_G"] - summary["mc_mean_T_over_G"]
    fig = _plot_grouped_metric(
        summary,
        value_column="normalized_drop",
        receivers=receivers,
        baselines=baselines,
        ylabel="归一化吞吐量退化 Δ(T/G)",
        title="图4.2 随机负载引起的基线策略吞吐量退化",
        ylim_from_zero=False,
    )
    _save(fig, fig_dir, "fig4_2_random_load_baseline_normalized_drop_grouped", dpi)
    _save(fig, fig_dir, "fig_ch4_ch3_baseline_mc_boxplot", dpi)
    _save(fig, fig_dir, "fig_ch4_ch3_baseline_relative_gap", dpi)

    receiver_labels = [label_for_receiver(receiver) for receiver in receivers]
    baseline_labels = [label_for_method(baseline) for baseline in baselines]
    print("图4.1使用的接收机类别:", "、".join(receiver_labels))
    print("图4.1使用的基线策略类别:", "、".join(baseline_labels))
    print("图4.2使用的接收机类别:", "、".join(receiver_labels))
    print("图4.2使用的基线策略类别:", "、".join(baseline_labels))
    print("图4.1保存路径:")
    print(f"  {fig_dir / 'fig4_1_random_load_baseline_T_over_G_grouped.png'}")
    print(f"  {fig_dir / 'fig4_1_random_load_baseline_T_over_G_grouped.pdf'}")
    print("图4.2保存路径:")
    print(f"  {fig_dir / 'fig4_2_random_load_baseline_normalized_drop_grouped.png'}")
    print(f"  {fig_dir / 'fig4_2_random_load_baseline_normalized_drop_grouped.pdf'}")
    print("确认横轴只包含全局OD、分组OD、MF、LMMSE四类。")
    return fig_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    fig_dir = plot_all(Path(args.result_dir), dpi=int(args.dpi))
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
