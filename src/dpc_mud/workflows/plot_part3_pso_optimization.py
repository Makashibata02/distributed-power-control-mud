"""Plot Experiment 41 PSO optimization results."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpc_mud.visualization.part3_style import apply_part2_style, color_for, save_figure, style_axes


def _save(fig, fig_dir: Path, name: str, dpi: int) -> None:
    save_figure(fig, fig_dir, name, dpi=dpi)


def _vec(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=float)


def plot_all(result_dir: Path, dpi: int = 300) -> Path:
    apply_part2_style()
    trace = pd.read_csv(result_dir / "ch4_pso_trace.csv")
    summary = pd.read_csv(result_dir / "ch4_pso_summary.csv").iloc[0]
    samples = pd.read_csv(result_dir / "ch4_pso_eval_samples.csv")
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(trace["iter"], trace["best_score"], marker="o", color=color_for("pso"))
    ax.set_title("PSO搜索收敛曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("当前最佳目标值")
    style_axes(ax, grid_axis="both")
    _save(fig, fig_dir, "fig_ch4_pso_convergence", dpi)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(trace["iter"], trace["best_q_tx"], marker="s", color=color_for("uniform"))
    ax.set_title("PSO最优发送概率变化")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("q_tx")
    style_axes(ax, grid_axis="both")
    _save(fig, fig_dir, "fig_ch4_pso_qtx_trace", dpi)

    for field, title, ylabel, name, include_zero, color_key in [
        ("pi_full", "PSO最优 pi_full 分布", "概率", "fig_ch4_pso_best_pi_distribution", True, "probability"),
        ("beta", "PSO最优正功率层条件分布 beta", "条件概率", "fig_ch4_pso_best_beta_distribution", False, "beta"),
        ("P", "PSO最优功率层分布", "功率 P_s", "fig_ch4_pso_best_power_distribution", False, "power"),
    ]:
        values = _vec(summary[field])
        x = np.arange(0 if include_zero else 1, len(values) if include_zero else len(values) + 1)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(x, values, color=color_for(color_key))
        ax.set_title(title)
        ax.set_xlabel("功率层索引" if not include_zero else "层索引（0为零功率层）")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        _save(fig, fig_dir, name, dpi)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(samples["T_sample"], bins=24, color=color_for("local_search"), edgecolor="white", alpha=0.85)
    ax.set_title("PSO最优策略MC样本吞吐量分布")
    ax.set_xlabel("样本吞吐量 T")
    ax.set_ylabel("频数")
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch4_pso_sample_distribution", dpi)
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
