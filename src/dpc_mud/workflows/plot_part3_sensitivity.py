"""Plot one-factor sensitivity curves for Experiment 43."""

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

from dpc_mud.visualization.part3_style import LINE_STYLES, MARKERS, apply_part2_style, color_for, label_for_method, save_figure, style_axes


SCAN_ZH = {
    "G": ("总输入负载 G", "sensitivity_G"),
    "R": ("目标速率 R", "sensitivity_R"),
    "N": ("接收天线数 N", "sensitivity_N"),
    "Pbar": ("平均功率约束 Pbar", "sensitivity_Pbar"),
}


def plot_all(result_dir: Path, dpi: int = 300) -> Path:
    df = pd.read_csv(result_dir / "ch4_sensitivity_summary.csv")
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_part2_style()

    methods = ["single_power", "structured_uniform_exp", "local_search", "PSO_MC"]
    for scan_name, (xlabel, filename) in SCAN_ZH.items():
        sub = df[df["scan_name"] == scan_name].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method in methods:
            mdf = sub[sub["method"] == method].sort_values("scan_value")
            if mdf.empty:
                continue
            ax.plot(
                mdf["scan_value"],
                mdf["mc_mean_T_over_G"],
                marker=MARKERS.get(method, "o"),
                linestyle=LINE_STYLES.get(method, "-"),
                label=label_for_method(method),
                color=color_for(method),
            )
        ax.set_title(f"{xlabel}变化下的MC平均吞吐率")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MC平均 T/G")
        style_axes(ax, grid_axis="both")
        ax.legend(ncol=3)
        save_figure(fig, fig_dir, filename, dpi=int(dpi))

        if scan_name == "G":
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for method in methods:
                mdf = sub[sub["method"] == method].sort_values("scan_value")
                if mdf.empty:
                    continue
                ax.plot(
                    mdf["scan_value"],
                    mdf["mc_mean_T"],
                    marker=MARKERS.get(method, "o"),
                    linestyle=LINE_STYLES.get(method, "-"),
                    label=label_for_method(method),
                    color=color_for(method),
                )
            ax.set_title("总输入负载 G 变化下的MC平均总吞吐量")
            ax.set_xlabel("总输入负载 G")
            ax.set_ylabel("MC平均总吞吐量 T")
            style_axes(ax, grid_axis="both")
            ax.legend(ncol=3)
            save_figure(fig, fig_dir, "sensitivity_G_total_throughput", dpi=int(dpi))
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
