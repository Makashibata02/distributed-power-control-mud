"""Experiment 40: Monte Carlo re-evaluation of Part II baselines."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from dpc_mud.parts.part2_grouped_access.experiment_utils import flags_json, portable_value, vector_json
from dpc_mud.parts.part3_stochastic_optimization.io import config_from_mapping, created_at, flatten_ch3_best_payloads, load_yaml, make_result_dir, save_json
from dpc_mud.parts.part3_stochastic_optimization.monte_carlo import mc_evaluate_policy
from dpc_mud.parts.part2_grouped_access.model import GroupedPolicy


def _save_csv(rows: List[Dict[str, Any]], path: Path, precision: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            out = {k: (f"{v:.{precision}f}" if isinstance(v, float) else portable_value(v)) for k, v in row.items()}
            writer.writerow(out)


def run_experiment(*, ch3_dir: Path, num_mc: int | None = None, seed: int | None = None, quick: bool = False, config_path: Path | None = None) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path or (project_root / "configs" / "part3_stochastic_optimization.yaml")
    cfg = load_yaml(config_path)
    if quick:
        num_mc = int(cfg.get("quick", {}).get("num_mc_baseline", 300))
    num_mc = int(num_mc if num_mc is not None else cfg.get("mc", {}).get("num_mc_baseline", 10000))
    seed = int(seed if seed is not None else cfg.get("mc", {}).get("seed", 20260520))
    config = config_from_mapping(cfg)
    result_dir = make_result_dir(project_root, cfg.get("outputs", {}).get("root", "results"), "ch4_exp40_mc_eval_ch3_baselines")
    precision = int(cfg.get("outputs", {}).get("float_precision", 8))

    summary_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}
    for idx, payload in enumerate(flatten_ch3_best_payloads(ch3_dir)):
        receiver = payload["receiver"]
        baseline_type = payload["baseline_type"]
        policy_dict = payload["policy"]
        policy = GroupedPolicy(policy_dict["name"], pi_full=policy_dict["pi_full"], P=policy_dict["P"], extra=policy_dict.get("extra", {}))
        result = mc_evaluate_policy(config, policy, receiver, num_mc=num_mc, seed=seed, keep_samples=True, method=baseline_type)
        det = payload.get("best_result", {})
        deterministic_T = float(det.get("T_packets", 0.0))
        row = {
            "receiver": receiver,
            "baseline_type": baseline_type,
            "policy_name": policy.name,
            "G": config.G,
            "S": config.S,
            "N": config.N,
            "R": config.R,
            "Pbar": config.Pbar,
            "sigma2": config.sigma2,
            "num_mc": num_mc,
            "seed": seed,
            "deterministic_T": deterministic_T,
            "deterministic_T_over_G": float(det.get("T_over_G", 0.0)),
            "mc_mean_T": result.mc_mean_T,
            "mc_std_T": result.mc_std_T,
            "mc_p05_T": result.mc_p05_T,
            "mc_p50_T": result.mc_p50_T,
            "mc_p95_T": result.mc_p95_T,
            "mc_mean_T_over_G": result.mc_mean_T_over_G,
            "mc_mean_T_over_Gtx": result.mc_mean_T_over_Gtx,
            "mc_mean_G_tx": result.mc_mean_G_tx,
            "relative_gap": ((result.mc_mean_T - deterministic_T) / deterministic_T) if abs(deterministic_T) > 1e-12 else 0.0,
            "pi_full": vector_json(policy.pi_full),
            "pi0": policy.pi0,
            "q_tx": policy.q_tx,
            "beta": vector_json(policy.beta),
            "P": vector_json(policy.P),
            "avg_power": policy.avg_power,
        }
        summary_rows.append(row)
        for srow in result.samples:
            sample_rows.append({"receiver": receiver, "baseline_type": baseline_type, **srow})
        detail[f"{receiver}::{baseline_type}"] = {"summary": row, "policy": policy.to_dict()}

    paths = {
        "summary_csv": result_dir / "ch4_mc_baseline_summary.csv",
        "samples_csv": result_dir / "ch4_mc_baseline_samples.csv",
        "results_json": result_dir / "ch4_mc_baseline_results.json",
        "metadata_json": result_dir / "metadata.json",
    }
    _save_csv(summary_rows, paths["summary_csv"], precision)
    _save_csv(sample_rows, paths["samples_csv"], precision)
    save_json(detail, paths["results_json"])
    save_json(
        {
            "experiment_name": "exp_40_ch4_mc_eval_ch3_baselines",
            "created_at": created_at(),
            "ch3_source_dir": str(ch3_dir),
            "system": config.to_dict(),
            "num_mc": num_mc,
            "seed": seed,
            "statement": "Part II deterministic or theoretical baseline results are re-evaluated under Part III Poisson random-load Monte Carlo model. These re-evaluated results, rather than the original Part II deterministic results, are used for fair comparison with PSO. No Rayleigh fading or explicit channel matrix simulation is used.",
            "config_path": str(config_path),
        },
        paths["metadata_json"],
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    try:
        from dpc_mud.workflows.plot_part3_monte_carlo_baselines import plot_all

        fig_dir = plot_all(result_dir)
        print(f"figures: {fig_dir}")
    except Exception as exc:
        print(f"warning: failed to auto-plot exp40 figures: {exc}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ch3-dir", required=True)
    parser.add_argument("--num-mc", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_experiment(ch3_dir=Path(args.ch3_dir), num_mc=args.num_mc, seed=args.seed, quick=bool(args.quick), config_path=None if args.config is None else Path(args.config))


if __name__ == "__main__":
    main()
