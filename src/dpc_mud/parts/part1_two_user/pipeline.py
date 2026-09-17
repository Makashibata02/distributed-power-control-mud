"""Run all Part I experiments and generate figures/tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import (
    Ch2Config,
    experiment_discrete_gain_summary,
    experiment_bridge_to_ch3,
    experiment_exponential_layer_compare,
    experiment_level_ablation,
    experiment_load_scan,
    experiment_params_table,
    experiment_pbar_sensitivity,
    experiment_rate_sensitivity,
    write_csv,
)
from .plots import (
    plot_best_random_structure,
    plot_bridge_to_ch3,
    plot_exponential_best_r,
    plot_exponential_layer_compare,
    plot_feasible_regions,
    plot_gain_vs_load,
    plot_gain_vs_rate,
    plot_level_ablation,
    plot_pi0_vs_load,
    plot_throughput_vs_load,
    plot_throughput_vs_pbar,
    setup_matplotlib,
)


def write_summary(out_dir: Path, cfg: Ch2Config, load_rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Part I Experiment Summary",
        "",
        f"- R: {cfg.rate}",
        f"- sigma^2: {cfg.sigma2}",
        f"- Pbar: {cfg.pbar}",
        f"- probability grid step: {cfg.grid_step}",
        "- The single power-control strategy uses pi0=0, pi1=1, and P1=Pbar.",
        "- The external load is G=2q; the single power-control strategy does not search pi0 or q_tx.",
        "",
        "## Best Discrete Distributed Power-Control Gain Over Nonzero Load",
    ]

    for receiver in ("OD", "SUD", "SIC"):
        part = [row for row in load_rows if row["receiver"] == receiver and float(row["G"]) > 0.0]
        best = max(part, key=lambda row: float(row["gain_abs"]))
        lines.append(
            f"- {receiver}: best discrete-minus-single gain {float(best['gain_abs']):.4f} at "
            f"G={float(best['G']):.2f}, pi0={float(best['pi0']):.2f}, "
            f"fixed_pi0={float(best['fixed_pi0']):.2f}, "
            f"T_single={float(best['T_single']):.4f}, T_discrete={float(best['T_discrete']):.4f}."
        )

    (out_dir / "simulation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def print_table_2_3(rows: list[dict[str, float | str]]) -> None:
    headers = [
        "接收机",
        "负载 G",
        "单一功率控制吞吐量",
        "离散分布式功率控制吞吐量",
        "绝对增益",
        "相对增益",
    ]
    widths = [8, 10, 22, 28, 12, 12]
    print("\n表2.3 G=2.0时离散分布式功率控制相对单一功率控制的吞吐量增益")
    print("".join(f"{header:<{width}}" for header, width in zip(headers, widths)))
    print("-" * sum(widths))
    for row in rows:
        values = [
            str(row["接收机"]),
            f"{float(row['负载 G']):.4f}",
            f"{float(row['单一功率控制吞吐量']):.6f}",
            f"{float(row['离散分布式功率控制吞吐量']):.6f}",
            f"{float(row['绝对增益']):.6f}",
            str(row["相对增益"]),
        ]
        print("".join(f"{value:<{width}}" for value, width in zip(values, widths)))


def run(output_dir: Path | None = None) -> Path:
    cfg = Ch2Config()
    out_dir = output_dir or (Path.cwd() / "results" / "generated" / "part1_two_user")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    params_rows = experiment_params_table(cfg)
    load_rows = experiment_load_scan(cfg)
    table_2_3_rows = experiment_discrete_gain_summary(load_rows, target_load=2.0)
    rate_rows = experiment_rate_sensitivity(cfg)
    pbar_rows = experiment_pbar_sensitivity(cfg)
    level_rows = experiment_level_ablation(cfg)
    bridge_rows = experiment_bridge_to_ch3(cfg)
    exp_compare_rows = experiment_exponential_layer_compare(cfg)

    write_csv(out_dir / "table_2_1_params.csv", params_rows)
    write_csv(out_dir / "table_2_2_best_policy.csv", load_rows)
    write_csv(out_dir / "table2_3_discrete_gain_over_single.csv", table_2_3_rows)
    write_csv(out_dir / "table_2_3_discrete_gain_over_single.csv", table_2_3_rows)
    write_csv(out_dir / "table_2_3_rate_sensitivity.csv", rate_rows)
    write_csv(out_dir / "table_2_4_pbar_sensitivity.csv", pbar_rows)
    write_csv(out_dir / "table_2_5_power_level_ablation.csv", level_rows)
    write_csv(out_dir / "table_2_6_two_user_to_multilevel_power.csv", bridge_rows)
    write_csv(out_dir / "table_2_7_four_point_vs_exp_layers.csv", exp_compare_rows)

    plot_feasible_regions(cfg, out_dir)
    plot_throughput_vs_load(load_rows, out_dir)
    plot_gain_vs_load(load_rows, out_dir)
    plot_best_random_structure(load_rows, out_dir)
    plot_pi0_vs_load(load_rows, out_dir)
    plot_gain_vs_rate(rate_rows, out_dir)
    plot_throughput_vs_pbar(pbar_rows, out_dir)
    plot_level_ablation(level_rows, out_dir)
    plot_bridge_to_ch3(bridge_rows, out_dir)
    plot_exponential_layer_compare(exp_compare_rows, out_dir)
    plot_exponential_best_r(exp_compare_rows, out_dir)
    write_summary(out_dir, cfg, load_rows)
    print_table_2_3(table_2_3_rows)

    print(f"Saved Part I outputs to: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part I experiments.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
