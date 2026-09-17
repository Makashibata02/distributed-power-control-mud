"""Experiment 43: Part III one-factor sensitivity scan.

Each scan point is evaluated under the same Poisson random-load Monte Carlo
model. The compared methods are:

1. single_power: single-power access baseline imported from Part II.
2. structured_uniform_exp: MC grid search over uniform probability and exponential power.
3. local_search: MC-objective local search initialized from Part II local-search policy.
4. PSO_MC: Part III PSO optimization result for that scan point.
"""

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
import time
from pathlib import Path
from typing import Any, Dict, List

from dpc_mud.parts.part2_grouped_access.experiment_utils import portable_value, vector_json
from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy
from dpc_mud.parts.part3_stochastic_optimization.io import config_from_mapping, created_at, load_ch3_best_policies, load_yaml, make_result_dir, save_json
from dpc_mud.parts.part3_stochastic_optimization.monte_carlo import MCEvalResult, mc_evaluate_policy
from dpc_mud.parts.part3_stochastic_optimization.optimizer import run_pso
from dpc_mud.workflows.part3_method_comparison import _mc_local_search, _search_uniform_exp_mc


SCAN_VALUES = {
    "G": [50.0, 100.0, 150.0, 200.0, 250.0],
    "R": [0.10, 0.15, 0.20, 0.25, 0.30],
    "N": [2, 4, 8, 16],
    "Pbar": [0.5, 1.0, 1.5, 2.0],
}


def _save_csv(rows: List[Dict[str, Any]], path: Path, precision: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.{precision}f}" if isinstance(v, float) else portable_value(v)) for k, v in row.items()})


def _rescale_policy(policy: GroupedPolicy, config: GroupedConfig, suffix: str = "") -> GroupedPolicy:
    denom = float((policy.pi_pos * policy.P).sum())
    if denom <= 1e-15:
        raise ValueError(f"Cannot rescale policy {policy.name}: zero weighted power.")
    P = policy.P * (config.Pbar / denom)
    return GroupedPolicy(
        name=f"{policy.name}{suffix}",
        pi_full=policy.pi_full.copy(),
        P=P,
        extra=dict(policy.extra),
    )


def _summary_row(
    *,
    scan_name: str,
    scan_value: float,
    receiver: str,
    method: str,
    config: GroupedConfig,
    policy: GroupedPolicy,
    result: MCEvalResult,
    runtime_sec: float = 0.0,
) -> Dict[str, Any]:
    return {
        "scan_name": scan_name,
        "scan_value": scan_value,
        "receiver": receiver,
        "method": method,
        "G": config.G,
        "S": config.S,
        "N": config.N,
        "R": config.R,
        "Pbar": config.Pbar,
        "sigma2": config.sigma2,
        "mc_mean_T": result.mc_mean_T,
        "mc_std_T": result.mc_std_T,
        "mc_p05_T": result.mc_p05_T,
        "mc_p50_T": result.mc_p50_T,
        "mc_p95_T": result.mc_p95_T,
        "mc_mean_T_over_G": result.mc_mean_T_over_G,
        "mc_mean_T_over_Gtx": result.mc_mean_T_over_Gtx,
        "mc_mean_G_tx": result.mc_mean_G_tx,
        "pi0": policy.pi0,
        "q_tx": policy.q_tx,
        "avg_power": policy.avg_power,
        "pi_full": vector_json(policy.pi_full),
        "beta": vector_json(policy.beta),
        "P": vector_json(policy.P),
        "runtime_sec": runtime_sec,
    }


def run_experiment(
    *,
    ch3_dir: Path,
    receiver: str | None = None,
    quick: bool = False,
    config_path: Path | None = None,
) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path or (project_root / "configs" / "part3_stochastic_optimization.yaml")
    cfg = load_yaml(config_path)
    pso_cfg = dict(cfg.get("pso", {}))
    mc_cfg = dict(cfg.get("mc", {}))
    if quick:
        q = cfg.get("quick", {})
        pso_cfg["num_particles"] = int(q.get("num_particles", 8))
        pso_cfg["num_iters"] = int(q.get("num_iters", 5))
        mc_cfg["num_mc_opt"] = int(q.get("num_mc_opt", 100))
        mc_cfg["num_mc_eval"] = int(q.get("num_mc_eval", 300))

    base_config = config_from_mapping(cfg)
    receiver = receiver or str(pso_cfg.get("receiver", "group_lmmse"))
    ch3_meta, ch3_policies = load_ch3_best_policies(ch3_dir)
    if receiver not in ch3_policies:
        raise ValueError(f"No Part II policies found for receiver={receiver}.")
    missing = [name for name in ("single_power", "structured_uniform_exp", "local_search") if name not in ch3_policies[receiver]]
    if missing:
        raise ValueError(f"Missing required Part II policies for sensitivity scan: {missing}.")

    seed_opt = int(mc_cfg.get("seed_opt", mc_cfg.get("seed", 20260520)))
    seed_eval = int(mc_cfg.get("seed_eval", seed_opt + 1000003))
    num_mc_opt = int(mc_cfg.get("num_mc_opt", 1000))
    num_mc_eval = int(mc_cfg.get("num_mc_eval", 10000))
    q_tx_grid = list(ch3_meta.get("q_tx_grid", [0.2, 0.4, 0.6, 0.8, 1.0]))
    r_grid = list(ch3_meta.get("r_grid", [1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]))
    result_dir = make_result_dir(project_root, cfg.get("outputs", {}).get("root", "results"), "ch4_exp43_sensitivity_scan")
    precision = int(cfg.get("outputs", {}).get("float_precision", 8))

    rows: List[Dict[str, Any]] = []
    best_policies: Dict[str, Any] = {}

    for scan_name, values in SCAN_VALUES.items():
        for value in values:
            override = {scan_name: int(value) if scan_name == "N" else float(value)}
            config = base_config.with_overrides(**override)
            key = f"{scan_name}={value}"
            best_policies[key] = {}

            single_policy = _rescale_policy(ch3_policies[receiver]["single_power"], config, suffix=f"_{key}_mc")
            single_eval = mc_evaluate_policy(
                config,
                single_policy,
                receiver,
                num_mc=num_mc_eval,
                seed=seed_eval,
                keep_samples=False,
                method="single_power",
            )
            rows.append(
                _summary_row(
                    scan_name=scan_name,
                    scan_value=float(value),
                    receiver=receiver,
                    method="single_power",
                    config=config,
                    policy=single_policy,
                    result=single_eval,
                )
            )
            best_policies[key]["single_power"] = {
                "policy": single_policy.to_dict(),
                "eval": single_eval.to_summary_dict(),
                "note": "Single-power baseline re-evaluated under the same scan-point MC setting.",
            }

            uniform_policy, uniform_opt, uniform_eval, uniform_candidates = _search_uniform_exp_mc(
                config=config,
                receiver=receiver,
                q_tx_grid=q_tx_grid,
                r_grid=r_grid,
                num_mc_opt=num_mc_opt,
                seed_opt=seed_opt,
                num_mc_eval=num_mc_eval,
                seed_eval=seed_eval,
            )
            rows.append(
                _summary_row(
                    scan_name=scan_name,
                    scan_value=float(value),
                    receiver=receiver,
                    method="structured_uniform_exp",
                    config=config,
                    policy=uniform_policy,
                    result=uniform_eval,
                )
            )
            best_policies[key]["structured_uniform_exp"] = {
                "policy": uniform_policy.to_dict(),
                "opt_eval": uniform_opt.to_summary_dict(),
                "eval": uniform_eval.to_summary_dict(),
                "candidates": uniform_candidates,
            }

            init_local = _rescale_policy(ch3_policies[receiver]["local_search"], config, suffix=f"_{key}_mc_init")
            local_policy, local_opt, local_eval, local_trace = _mc_local_search(
                config=config,
                receiver=receiver,
                init_policy=init_local,
                num_steps=int(pso_cfg.get("num_iters", 50)),
                seed_search=seed_opt + 17,
                num_mc_opt=num_mc_opt,
                seed_opt=seed_opt,
                num_mc_eval=num_mc_eval,
                seed_eval=seed_eval,
                beta_scale=float(pso_cfg.get("beta_scale", 0.10)),
                power_scale=float(pso_cfg.get("power_scale", 0.10)),
                q_scale=float(pso_cfg.get("q_scale", 0.20)),
            )
            rows.append(
                _summary_row(
                    scan_name=scan_name,
                    scan_value=float(value),
                    receiver=receiver,
                    method="local_search",
                    config=config,
                    policy=local_policy,
                    result=local_eval,
                )
            )
            best_policies[key]["local_search"] = {
                "init_policy": init_local.to_dict(),
                "policy": local_policy.to_dict(),
                "opt_eval": local_opt.to_summary_dict(),
                "eval": local_eval.to_summary_dict(),
                "trace": local_trace,
            }

            warm_policies = [uniform_policy, local_policy]

            start = time.perf_counter()
            pso_result = run_pso(
                config=config,
                receiver=receiver,
                num_particles=int(pso_cfg.get("num_particles", 30)),
                num_iters=int(pso_cfg.get("num_iters", 50)),
                num_mc_opt=num_mc_opt,
                num_mc_eval=num_mc_eval,
                seed_opt=seed_opt,
                seed_eval=seed_eval,
                w=float(pso_cfg.get("w", 0.7)),
                c1=float(pso_cfg.get("c1", 1.5)),
                c2=float(pso_cfg.get("c2", 1.5)),
                vmax=pso_cfg.get("vmax", None),
                objective_metric=str(pso_cfg.get("objective_metric", "mc_mean_T_over_G")),
                warm_policies=warm_policies,
                warm_start_noise=float(pso_cfg.get("warm_start_noise", 0.15)),
                warm_ratio=float(pso_cfg.get("warm_ratio", 0.4)),
                clip_z_q=float(pso_cfg.get("clip_z_q", 10.0)),
                clip_z_beta=float(pso_cfg.get("clip_z_beta", 15.0)),
                clip_u=float(pso_cfg.get("clip_u", 10.0)),
                pmax_mode=str(pso_cfg.get("pmax_mode", "reject")),
            )
            runtime = time.perf_counter() - start
            rows.append(
                _summary_row(
                    scan_name=scan_name,
                    scan_value=float(value),
                    receiver=receiver,
                    method="PSO_MC",
                    config=config,
                    policy=pso_result.best_policy,
                    result=pso_result.final_eval,
                    runtime_sec=runtime,
                )
            )
            best_policies[key]["PSO_MC"] = {
                "policy": pso_result.best_policy.to_dict(),
                "eval": pso_result.final_eval.to_summary_dict(),
                "trace": pso_result.trace,
            }
            print(f"finished {key}: PSO T/G={pso_result.final_eval.mc_mean_T_over_G:.6f}")

    paths = {
        "summary_csv": result_dir / "ch4_sensitivity_summary.csv",
        "best_policies_json": result_dir / "ch4_sensitivity_best_policies.json",
        "metadata_json": result_dir / "metadata.json",
    }
    _save_csv(rows, paths["summary_csv"], precision)
    save_json(best_policies, paths["best_policies_json"])
    save_json(
        {
            "experiment_name": "exp_43_ch4_sensitivity_scan",
            "created_at": created_at(),
            "ch3_source_dir": str(ch3_dir),
            "receiver": receiver,
            "scan_values": SCAN_VALUES,
            "system_base": base_config.to_dict(),
            "num_mc_opt": num_mc_opt,
            "num_mc_eval": num_mc_eval,
            "seed_opt": seed_opt,
            "seed_eval": seed_eval,
            "q_tx_grid": q_tx_grid,
            "r_grid": r_grid,
            "pso": pso_cfg,
            "quick": bool(quick),
            "statement": "Part III sensitivity scan re-optimizes uniform-probability and local-search baselines under Poisson random-load Monte Carlo evaluation before comparing with PSO. No Rayleigh fading or explicit channel matrix simulation is used.",
            "ch3_metadata": ch3_meta,
        },
        paths["metadata_json"],
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    try:
        from dpc_mud.workflows.plot_part3_sensitivity import plot_all

        fig_dir = plot_all(result_dir)
        print(f"figures: {fig_dir}")
    except Exception as exc:
        print(f"warning: failed to auto-plot exp43 figures: {exc}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ch3-dir", required=True)
    parser.add_argument("--receiver", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_experiment(
        ch3_dir=Path(args.ch3_dir),
        receiver=args.receiver,
        quick=bool(args.quick),
        config_path=None if args.config is None else Path(args.config),
    )


if __name__ == "__main__":
    main()
