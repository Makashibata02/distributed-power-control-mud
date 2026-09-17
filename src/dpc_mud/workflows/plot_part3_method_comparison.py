"""Plot Experiment 42 baseline-vs-PSO comparison."""

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

from dpc_mud.visualization.part3_style import apply_part2_style, color_for, label_for_method, save_figure, style_axes


def _save(fig, fig_dir: Path, name: str, dpi: int) -> None:
    save_figure(fig, fig_dir, name, dpi=dpi)


def _labels(methods: list[str]) -> list[str]:
    return [label_for_method(m) for m in methods]


def _colors(methods: list[str]) -> list[str]:
    return [color_for(method, i) for i, method in enumerate(methods)]


def plot_all(result_dir: Path, dpi: int = 300) -> Path:
    apply_part2_style()
    summary = pd.read_csv(result_dir / "ch4_baseline_pso_compare.csv")
    samples = pd.read_csv(result_dir / "ch4_baseline_pso_compare_samples.csv")
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_order = ["single_power", "structured_uniform_exp", "local_search", "PSO_MC"]
    summary = summary[summary["method"].isin(method_order)].copy()
    summary["method_order"] = summary["method"].map({method: i for i, method in enumerate(method_order)})
    summary = summary.sort_values("method_order")
    methods = list(summary["method"])
    labels = _labels(methods)
    colors = _colors(methods)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    yerr = summary["mc_std_T"] / summary["G"].clip(lower=1e-12)
    ax.bar(labels, summary["mc_mean_T_over_G"], yerr=yerr, capsize=4, color=colors)
    ax.set_title("第三章基线与PSO的随机负载吞吐率对比")
    ax.set_ylabel("MC平均 T/G")
    ax.set_xlabel("方法")
    ax.tick_params(axis="x", rotation=18)
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch4_baseline_vs_pso_T_over_G", dpi)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, summary["mc_mean_T"], yerr=summary["mc_std_T"], capsize=4, color=colors)
    ax.set_title("第三章基线与PSO的随机负载吞吐量对比")
    ax.set_ylabel("MC平均吞吐量 T")
    ax.set_xlabel("方法")
    ax.tick_params(axis="x", rotation=18)
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch4_baseline_vs_pso_T", dpi)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(labels, summary["q_tx"], color=colors)
    ax.set_title("第三章基线与PSO的发送概率对比")
    ax.set_ylabel("q_tx")
    ax.set_xlabel("方法")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(summary["q_tx"].max()) * 1.1)))
    ax.tick_params(axis="x", rotation=18)
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch4_baseline_vs_pso_qtx", dpi)

    grouped = []
    grouped_labels = []
    grouped_methods = []
    for method in methods:
        group = samples[samples["method"] == method]["T_sample"].to_numpy()
        if len(group) > 0:
            grouped.append(group)
            grouped_labels.append(label_for_method(method))
            grouped_methods.append(method)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    box = ax.boxplot(grouped, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], _colors(grouped_methods)):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("第三章基线与PSO的MC样本吞吐量分布")
    ax.set_ylabel("样本吞吐量 T")
    ax.set_xlabel("方法")
    ax.set_xticklabels(grouped_labels, rotation=18, ha="right")
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch4_baseline_vs_pso_boxplot", dpi)

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
