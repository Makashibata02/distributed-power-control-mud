"""Experiment 41: PSO optimization under Poisson random-load Monte Carlo."""

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
from dpc_mud.parts.part3_stochastic_optimization.io import config_from_mapping, created_at, load_ch3_best_policies, load_yaml, make_result_dir, save_json
from dpc_mud.parts.part3_stochastic_optimization.optimizer import run_pso


def _save_csv(rows: List[Dict[str, Any]], path: Path, precision: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.{precision}f}" if isinstance(v, float) else portable_value(v)) for k, v in row.items()})


def run_experiment(*, ch3_dir: Path, receiver: str | None = None, quick: bool = False, config_path: Path | None = None) -> Dict[str, Path]:
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
    receiver = receiver or str(pso_cfg.get("receiver", "group_lmmse"))
    seed_opt = int(mc_cfg.get("seed_opt", mc_cfg.get("seed", 20260520)))
    seed_eval = int(mc_cfg.get("seed_eval", seed_opt + 1000003))
    config = config_from_mapping(cfg)
    ch3_meta, ch3_policies = load_ch3_best_policies(ch3_dir)
    warm = []
    warm_policy_names = []
    if receiver in ch3_policies:
        for key in ("structured_uniform_exp", "local_search"):
            if key in ch3_policies[receiver]:
                warm.append(ch3_policies[receiver][key])
                warm_policy_names.append(key)

    result_dir = make_result_dir(project_root, cfg.get("outputs", {}).get("root", "results"), "ch4_exp41_pso_mc_opt")
    precision = int(cfg.get("outputs", {}).get("float_precision", 8))
    start = time.perf_counter()
    result = run_pso(
        config=config,
        receiver=receiver,
        num_particles=int(pso_cfg.get("num_particles", 30)),
        num_iters=int(pso_cfg.get("num_iters", 50)),
        num_mc_opt=int(mc_cfg.get("num_mc_opt", 1000)),
        num_mc_eval=int(mc_cfg.get("num_mc_eval", 10000)),
        seed_opt=seed_opt,
        seed_eval=seed_eval,
        w=float(pso_cfg.get("w", 0.7)),
        c1=float(pso_cfg.get("c1", 1.5)),
        c2=float(pso_cfg.get("c2", 1.5)),
        vmax=pso_cfg.get("vmax", None),
        objective_metric=str(pso_cfg.get("objective_metric", "mc_mean_T_over_G")),
        warm_policies=warm,
        warm_start_noise=float(pso_cfg.get("warm_start_noise", 0.15)),
        warm_ratio=float(pso_cfg.get("warm_ratio", 0.4)),
        clip_z_q=float(pso_cfg.get("clip_z_q", 10.0)),
        clip_z_beta=float(pso_cfg.get("clip_z_beta", 15.0)),
        clip_u=float(pso_cfg.get("clip_u", 10.0)),
        pmax_mode=str(pso_cfg.get("pmax_mode", "reject")),
    )
    runtime = time.perf_counter() - start
    policy = result.best_policy
    eval_result = result.final_eval
    summary = {
        "receiver": receiver,
        "method": "PSO_MC",
        "G": config.G,
        "S": config.S,
        "N": config.N,
        "R": config.R,
        "Pbar": config.Pbar,
        "sigma2": config.sigma2,
        "num_particles": int(pso_cfg.get("num_particles", 30)),
        "num_iters": int(pso_cfg.get("num_iters", 50)),
        "num_mc_opt": int(mc_cfg.get("num_mc_opt", 1000)),
        "num_mc_eval": int(mc_cfg.get("num_mc_eval", 10000)),
        "seed_opt": seed_opt,
        "seed_eval": seed_eval,
        "warm_ratio": float(pso_cfg.get("warm_ratio", 0.4)),
        "warm_start_noise": float(pso_cfg.get("warm_start_noise", 0.15)),
        "warm_policy_names": json.dumps(warm_policy_names, ensure_ascii=False),
        "best_opt_score": float(result.best_score),
        "eval_mc_mean_T": eval_result.mc_mean_T,
        "eval_mc_std_T": eval_result.mc_std_T,
        "eval_mc_mean_T_over_G": eval_result.mc_mean_T_over_G,
        "eval_mc_mean_T_over_Gtx": eval_result.mc_mean_T_over_Gtx,
        "eval_mc_mean_G_tx": eval_result.mc_mean_G_tx,
        "pi_full": vector_json(policy.pi_full),
        "pi0": policy.pi0,
        "q_tx": policy.q_tx,
        "beta": vector_json(policy.beta),
        "P": vector_json(policy.P),
        "avg_power": policy.avg_power,
        "runtime_sec": runtime,
    }
    paths = {
        "summary_csv": result_dir / "ch4_pso_summary.csv",
        "best_policy_json": result_dir / "ch4_pso_best_policy.json",
        "trace_csv": result_dir / "ch4_pso_trace.csv",
        "samples_csv": result_dir / "ch4_pso_eval_samples.csv",
        "metadata_json": result_dir / "metadata.json",
    }
    _save_csv([summary], paths["summary_csv"], precision)
    _save_csv(result.trace, paths["trace_csv"], precision)
    sample_rows = [{"receiver": receiver, "method": "PSO_MC", **row} for row in eval_result.samples]
    _save_csv(sample_rows, paths["samples_csv"], precision)
    save_json({"policy": policy.to_dict(), "final_eval": eval_result.to_summary_dict(), "summary": summary}, paths["best_policy_json"])
    save_json(
        {
            "experiment_name": "exp_41_ch4_pso_mc_opt",
            "created_at": created_at(),
            "ch3_source_dir": str(ch3_dir),
            "system": config.to_dict(),
            "receiver": receiver,
            "seed_opt": seed_opt,
            "seed_eval": seed_eval,
            "num_mc_opt": int(mc_cfg.get("num_mc_opt", 1000)),
            "num_mc_eval": int(mc_cfg.get("num_mc_eval", 10000)),
            "warm_ratio": float(pso_cfg.get("warm_ratio", 0.4)),
            "warm_start_noise": float(pso_cfg.get("warm_start_noise", 0.15)),
            "warm_policy_names": warm_policy_names,
            "pso": pso_cfg,
            "mc": mc_cfg,
            "statement": "Part III uses Poisson random-load Monte Carlo evaluation. No Rayleigh fading or explicit channel matrix simulation is used. PSO is an approximate optimization result, not a global optimum.",
            "ch3_metadata": ch3_meta,
        },
        paths["metadata_json"],
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    try:
        from dpc_mud.workflows.plot_part3_pso_optimization import plot_all

        fig_dir = plot_all(result_dir)
        print(f"figures: {fig_dir}")
    except Exception as exc:
        print(f"warning: failed to auto-plot exp41 figures: {exc}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ch3-dir", required=True)
    parser.add_argument("--receiver", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_experiment(ch3_dir=Path(args.ch3_dir), receiver=args.receiver, quick=bool(args.quick), config_path=None if args.config is None else Path(args.config))


if __name__ == "__main__":
    main()
