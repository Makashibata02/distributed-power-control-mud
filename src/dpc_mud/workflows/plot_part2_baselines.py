"""Plot Part II formula-baseline results with a unified project style."""

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

from dpc_mud.visualization.part2_style import (
    INIT_BETA_LABELS,
    LINE_STYLES,
    MARKERS,
    apply_part2_style,
    color_for,
    label_for_init,
    label_for_method,
    label_for_receiver,
    save_figure,
    style_axes,
)


def _save(fig, fig_dir: Path, name: str, dpi: int) -> None:
    save_figure(fig, fig_dir, name, dpi=dpi)


def _vec(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=float)


def _group_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, fig_dir: Path, name: str, dpi: int) -> None:
    receivers = list(df["receiver"].drop_duplicates())
    baselines = list(df["baseline_type"].drop_duplicates())
    x = np.arange(len(receivers))
    width = min(0.16, 0.82 / max(len(baselines), 1))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, baseline in enumerate(baselines):
        vals = []
        for receiver in receivers:
            row = df[(df["receiver"] == receiver) & (df["baseline_type"] == baseline)]
            vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + (i - (len(baselines) - 1) / 2) * width,
            vals,
            width=width,
            color=color_for(baseline, i),
            label=label_for_method(baseline),
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_receiver(r) for r in receivers], rotation=10)
    style_axes(ax)
    ax.legend(ncol=2)
    _save(fig, fig_dir, name, dpi)


def _best_local(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[summary["baseline_type"] == "local_search"]
    if rows.empty:
        rows = summary.loc[summary.groupby("receiver")["T_over_G"].idxmax()]
    return rows


def _layer_panels(
    rows: pd.DataFrame,
    field: str,
    title: str,
    ylabel: str,
    fig_dir: Path,
    name: str,
    dpi: int,
    include_zero: bool = False,
) -> None:
    receivers = list(rows["receiver"].drop_duplicates())
    fig, axes = plt.subplots(len(receivers), 1, figsize=(8.5, 2.35 * len(receivers)), sharex=False)
    if len(receivers) == 1:
        axes = [axes]
    field_color = {
        "pi_full": color_for("probability"),
        "beta": color_for("beta"),
        "P": color_for("power"),
    }.get(field, color_for("local_search"))
    for ax, receiver in zip(axes, receivers):
        row = rows[rows["receiver"] == receiver].iloc[0]
        y = _vec(row[field])
        x = np.arange(0 if include_zero else 1, len(y) if include_zero else len(y) + 1)
        ax.bar(x, y, color=field_color)
        ax.set_title(label_for_receiver(receiver), loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        style_axes(ax)
        if include_zero:
            ax.set_xticks(x)
            ax.set_xticklabels(["零功率层"] + [str(i) for i in range(1, len(y))])
        else:
            ax.set_xticks(x)
    axes[-1].set_xlabel("功率层索引")
    fig.suptitle(title, y=1.01, fontsize=12)
    _save(fig, fig_dir, name, dpi)


def plot_all(result_dir: Path, dpi: int = 300) -> Path:
    apply_part2_style()
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(result_dir / "ch3_formula_summary.csv")
    structured = pd.read_csv(result_dir / "ch3_structured_initializers.csv")
    random_df = pd.read_csv(result_dir / "ch3_random_initializers.csv")

    local_only = summary[summary["baseline_type"] == "local_search"]
    _group_bar(local_only, "T_over_G", "不同接收模型的公式化性能层级", "T/G", fig_dir, "fig_ch3_receiver_hierarchy_T_over_G", dpi)
    _group_bar(local_only, "T_over_Gtx", "不同接收模型的发送用户成功率层级", "T/G_tx", fig_dir, "fig_ch3_receiver_hierarchy_T_over_Gtx", dpi)
    _group_bar(summary, "T_over_G", "不同基线策略的公式化性能对比", "T/G", fig_dir, "fig_ch3_baseline_compare_T_over_G", dpi)
    _group_bar(summary, "T_over_Gtx", "不同基线策略的发送用户成功率对比", "T/G_tx", fig_dir, "fig_ch3_baseline_compare_T_over_Gtx", dpi)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped = structured.groupby(["receiver", "init_beta_mode"])["T_over_G"].max().reset_index()
    init_order = [key for key in INIT_BETA_LABELS if key in set(grouped["init_beta_mode"].astype(str))]
    for receiver, group in grouped.groupby("receiver"):
        group = group.copy()
        group["init_beta_mode"] = pd.Categorical(group["init_beta_mode"].astype(str), categories=init_order, ordered=True)
        group = group.sort_values("init_beta_mode")
        ax.plot(
            [label_for_init(str(x)) for x in group["init_beta_mode"]],
            group["T_over_G"],
            marker=MARKERS.get(receiver, "o"),
            linestyle=LINE_STYLES.get(receiver, "-"),
            color=color_for(receiver),
            label=label_for_receiver(receiver),
        )
    ax.set_title("不同结构化初始化的公式化性能差异")
    ax.set_xlabel("结构化 beta 初始化")
    ax.set_ylabel("最佳 T/G")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax, grid_axis="both")
    ax.legend(ncol=2)
    _save(fig, fig_dir, "fig_ch3_structured_init_compare", dpi)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    data = []
    labels = []
    colors = []
    for idx, (receiver, group) in enumerate(random_df.groupby("receiver")):
        data.append(group["T_over_G"].to_numpy())
        labels.append(label_for_receiver(receiver))
        colors.append(color_for(receiver, idx))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    struct_best = summary[summary["baseline_type"] == "structured_all"]
    if not struct_best.empty:
        ax.scatter(np.arange(1, len(labels) + 1), struct_best["T_over_G"], color=color_for("pso"), marker="D", label="结构化最优")
    ax.set_title("随机初始化分布与结构化最优对比")
    ax.set_ylabel("T/G")
    style_axes(ax)
    ax.legend()
    _save(fig, fig_dir, "fig_ch3_random_vs_structured_boxplot", dpi)

    best_rows = _best_local(summary)
    _layer_panels(best_rows, "pi_full", "局部搜索最优策略的 pi_full 分布", "概率", fig_dir, "fig_ch3_best_pi_distribution", dpi, include_zero=True)
    _layer_panels(best_rows, "beta", "局部搜索最优策略的正功率层条件分布 beta", "条件概率", fig_dir, "fig_ch3_best_beta_distribution", dpi)
    _layer_panels(best_rows, "P", "局部搜索最优策略的功率分布", "功率 P_s", fig_dir, "fig_ch3_best_power_distribution", dpi)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    gain_rows = []
    for receiver, group in summary.groupby("receiver"):
        local_row = group[group["baseline_type"] == "local_search"]
        nonlocal_rows = group[group["baseline_type"] != "local_search"]
        if local_row.empty or nonlocal_rows.empty:
            continue
        gain_rows.append(
            {
                "receiver": receiver,
                "gain": max(0.0, float(local_row["T_packets"].iloc[0]) - float(nonlocal_rows["T_packets"].max())),
            }
        )
    gain_df = pd.DataFrame(gain_rows)
    ax.bar([label_for_receiver(r) for r in gain_df["receiver"]], gain_df["gain"], color=color_for("local_search"))
    ax.set_title("局部搜索相对最佳非局部基线的吞吐量提升")
    ax.set_ylabel("绝对增益")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax)
    _save(fig, fig_dir, "fig_ch3_local_gain", dpi)

    trace_path = result_dir / "ch3_search_traces.json"
    if trace_path.exists():
        traces = json.loads(trace_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        selected = {}
        for key, payload in traces.items():
            receiver = key.split("_run_")[0]
            trace = payload.get("search_trace", [])
            if not trace:
                continue
            score = (
                round(float(payload.get("best_result", {}).get("T_packets", 0.0)), 9),
                round(float(payload.get("init_result", {}).get("T_packets", 0.0)), 9),
                -round(float(payload.get("local_gain", 0.0)), 9),
            )
            if receiver not in selected or score > selected[receiver][0]:
                selected[receiver] = (score, payload)
        for receiver, (_, payload) in selected.items():
            trace = payload.get("search_trace", [])
            ax.plot(
                [row["step"] for row in trace],
                [row["best_T_packets"] for row in trace],
                marker=MARKERS.get(receiver, "o"),
                linestyle=LINE_STYLES.get(receiver, "-"),
                color=color_for(receiver),
                label=label_for_receiver(receiver),
            )
        ax.set_title("代表性最优局部搜索轨迹")
        ax.set_xlabel("搜索步数")
        ax.set_ylabel("当前最佳吞吐量")
        style_axes(ax, grid_axis="both")
        ax.legend(ncol=2)
        _save(fig, fig_dir, "fig_ch3_search_trace", dpi)
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
