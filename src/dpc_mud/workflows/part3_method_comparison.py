"""Experiment 42: compare Part II MC baselines with Part III PSO.

This script only reads the outputs of exp_40 and exp_41. It does not rerun
Monte Carlo evaluation, so the comparison uses the exact same saved samples and
summary statistics produced by those experiments.
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
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dpc_mud.parts.part2_grouped_access.experiment_utils import portable_value, vector_json
from dpc_mud.parts.part2_grouped_access.initializers import make_initial_policy
from dpc_mud.parts.part2_grouped_access.local_search import perturb_policy
from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy
from dpc_mud.parts.part3_stochastic_optimization.io import config_from_mapping, created_at, load_ch3_best_policies, make_result_dir, save_json
from dpc_mud.parts.part3_stochastic_optimization.monte_carlo import MCEvalResult, mc_evaluate_policy


METHOD_ZH = {
    "single_power": "单功率接入基线",
    "structured_uniform_exp": "均匀概率初始化",
    "local_search": "局部搜索优化",
    "PSO_MC": "PSO算法优化",
}


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to save for {path}.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{key: portable_value(value) for key, value in row.items()} for row in rows])


def _first_existing(path_candidates: list[Path]) -> Path:
    for path in path_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected files exists: " + ", ".join(str(p) for p in path_candidates))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rescale_policy(policy: GroupedPolicy, config: GroupedConfig, suffix: str = "") -> GroupedPolicy:
    denom = float(np.sum(policy.pi_pos * policy.P))
    if denom <= 1e-15:
        raise ValueError(f"Cannot rescale policy {policy.name}: zero weighted power.")
    return GroupedPolicy(
        name=f"{policy.name}{suffix}",
        pi_full=policy.pi_full.copy(),
        P=policy.P * (config.Pbar / denom),
        extra=dict(policy.extra),
    )


def _row(
    *,
    receiver: str,
    method: str,
    config: GroupedConfig,
    policy: GroupedPolicy,
    result: MCEvalResult,
    source_dir: Path,
    search_note: str,
) -> Dict[str, Any]:
    return {
        "receiver": receiver,
        "method": method,
        "method_zh": METHOD_ZH.get(method, method),
        "G": float(config.G),
        "mc_mean_T": float(result.mc_mean_T),
        "mc_std_T": float(result.mc_std_T),
        "mc_mean_T_over_G": float(result.mc_mean_T_over_G),
        "mc_mean_T_over_Gtx": float(result.mc_mean_T_over_Gtx),
        "mc_mean_G_tx": float(result.mc_mean_G_tx),
        "pi0": float(policy.pi0),
        "q_tx": float(policy.q_tx),
        "avg_power": float(policy.avg_power),
        "pi_full": vector_json(policy.pi_full),
        "beta": vector_json(policy.beta),
        "P": vector_json(policy.P),
        "source_dir": str(source_dir),
        "search_note": search_note,
    }


def _sample_rows(receiver: str, method: str, result: MCEvalResult, source_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "receiver": receiver,
            "method": method,
            "sample_id": int(row["sample_id"]),
            "T_sample": float(row["T_sample"]),
            "T_over_G_sample": float(row["T_over_G_sample"]),
            "T_over_Gtx_sample": float(row["T_over_Gtx_sample"]),
            "G_tx_sample": float(row["G_tx_sample"]),
            "source_dir": str(source_dir),
        }
        for row in result.samples
    ]


def _search_uniform_exp_mc(
    *,
    config: GroupedConfig,
    receiver: str,
    q_tx_grid: List[float],
    r_grid: List[float],
    num_mc_opt: int,
    seed_opt: int,
    num_mc_eval: int,
    seed_eval: int,
) -> tuple[GroupedPolicy, MCEvalResult, MCEvalResult, List[Dict[str, Any]]]:
    best_policy: GroupedPolicy | None = None
    best_opt: MCEvalResult | None = None
    candidates: List[Dict[str, Any]] = []
    for q_tx in q_tx_grid:
        for r in r_grid:
            try:
                policy = make_initial_policy(
                    config,
                    beta_mode="uniform",
                    power_mode="exponential",
                    q_tx=float(q_tx),
                    r=float(r),
                )
                result = mc_evaluate_policy(
                    config,
                    policy,
                    receiver,
                    num_mc=int(num_mc_opt),
                    seed=int(seed_opt),
                    keep_samples=False,
                    method="structured_uniform_exp",
                )
            except Exception as exc:
                candidates.append({"method": "structured_uniform_exp", "q_tx": q_tx, "r": r, "valid": False, "error": str(exc)})
                continue
            candidates.append(
                {
                    "method": "structured_uniform_exp",
                    "q_tx": q_tx,
                    "r": r,
                    "valid": True,
                    "mc_mean_T_over_G": result.mc_mean_T_over_G,
                    "mc_mean_T": result.mc_mean_T,
                }
            )
            if best_opt is None or result.mc_mean_T_over_G > best_opt.mc_mean_T_over_G + 1e-12:
                best_policy = policy
                best_opt = result
    if best_policy is None or best_opt is None:
        raise RuntimeError("MC uniform-exp search did not produce any valid candidate.")
    final_eval = mc_evaluate_policy(
        config,
        best_policy,
        receiver,
        num_mc=int(num_mc_eval),
        seed=int(seed_eval),
        keep_samples=True,
        method="structured_uniform_exp",
    )
    return best_policy, best_opt, final_eval, candidates


def _mc_local_search(
    *,
    config: GroupedConfig,
    receiver: str,
    init_policy: GroupedPolicy,
    num_steps: int,
    seed_search: int,
    num_mc_opt: int,
    seed_opt: int,
    num_mc_eval: int,
    seed_eval: int,
    beta_scale: float,
    power_scale: float,
    q_scale: float,
) -> tuple[GroupedPolicy, MCEvalResult, MCEvalResult, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed_search))
    best_policy = init_policy
    best_opt = mc_evaluate_policy(
        config,
        best_policy,
        receiver,
        num_mc=int(num_mc_opt),
        seed=int(seed_opt),
        keep_samples=False,
        method="local_search",
    )
    trace: List[Dict[str, Any]] = []
    for step in range(int(num_steps)):
        try:
            candidate = perturb_policy(
                best_policy,
                config,
                rng,
                beta_scale=float(beta_scale),
                power_scale=float(power_scale),
                q_scale=float(q_scale),
            )
            candidate_opt = mc_evaluate_policy(
                config,
                candidate,
                receiver,
                num_mc=int(num_mc_opt),
                seed=int(seed_opt),
                keep_samples=False,
                method="local_search",
            )
            accepted = bool(candidate_opt.mc_mean_T_over_G > best_opt.mc_mean_T_over_G + 1e-12)
            if accepted:
                best_policy = candidate
                best_opt = candidate_opt
            trace.append(
                {
                    "step": int(step),
                    "candidate_T_over_G": float(candidate_opt.mc_mean_T_over_G),
                    "best_T_over_G": float(best_opt.mc_mean_T_over_G),
                    "candidate_q_tx": float(candidate.q_tx),
                    "best_q_tx": float(best_policy.q_tx),
                    "accepted": accepted,
                }
            )
        except Exception as exc:
            trace.append({"step": int(step), "accepted": False, "error": str(exc)})
    final_eval = mc_evaluate_policy(
        config,
        best_policy,
        receiver,
        num_mc=int(num_mc_eval),
        seed=int(seed_eval),
        keep_samples=True,
        method="local_search",
    )
    return best_policy, best_opt, final_eval, trace


def run_experiment(*, baseline_mc_dir: Path, pso_dir: Path) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    result_dir = make_result_dir(project_root, "results", "ch4_exp42_compare_baseline_pso")

    baseline_summary_path = _first_existing([baseline_mc_dir / "ch4_mc_baseline_summary.csv"])
    pso_summary_path = _first_existing([pso_dir / "ch4_pso_summary.csv"])
    pso_samples_path = _first_existing([pso_dir / "ch4_pso_eval_samples.csv"])
    baseline_metadata = _load_json(baseline_mc_dir / "metadata.json")
    pso_metadata = _load_json(pso_dir / "metadata.json")

    baseline_summary = pd.read_csv(baseline_summary_path)
    baseline_samples = pd.read_csv(baseline_mc_dir / "ch4_mc_baseline_samples.csv")
    pso_summary = pd.read_csv(pso_summary_path).iloc[0]
    pso_samples = pd.read_csv(pso_samples_path)

    receiver = str(pso_summary["receiver"])
    if baseline_summary[baseline_summary["receiver"] == receiver].empty:
        raise ValueError(f"No baseline MC rows found for receiver={receiver}.")
    config = config_from_mapping({"system": pso_metadata.get("system", {})})
    ch3_dir = Path(baseline_metadata.get("ch3_source_dir", pso_metadata.get("ch3_source_dir", "")))
    if not ch3_dir.exists():
        repo_root = Path(os.environ.get("DPC_MUD_REPO_ROOT", project_root))
        candidates = [
            repo_root / ch3_dir,
            project_root / ch3_dir,
            project_root / "results" / ch3_dir.name,
            project_root / "results_quick" / ch3_dir.name,
        ]
        ch3_dir = next((path for path in candidates if path.exists()), ch3_dir)
    if not ch3_dir.exists():
        raise FileNotFoundError(f"Cannot locate Part II source directory from metadata: {ch3_dir}")
    ch3_meta, ch3_policies = load_ch3_best_policies(ch3_dir)
    if receiver not in ch3_policies or "local_search" not in ch3_policies[receiver]:
        raise ValueError(f"Part II local_search policy is required for receiver={receiver}.")

    q_tx_grid = list(ch3_meta.get("q_tx_grid", [0.2, 0.4, 0.6, 0.8, 1.0]))
    r_grid = list(ch3_meta.get("r_grid", [1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]))
    pso_cfg = dict(pso_metadata.get("pso", {}))
    num_mc_opt = int(pso_metadata.get("num_mc_opt", pso_metadata.get("mc", {}).get("num_mc_opt", 1000)))
    num_mc_eval = int(pso_metadata.get("num_mc_eval", pso_metadata.get("mc", {}).get("num_mc_eval", 10000)))
    seed_opt = int(pso_metadata.get("seed_opt", pso_metadata.get("mc", {}).get("seed_opt", 20260520)))
    seed_eval = int(pso_metadata.get("seed_eval", pso_metadata.get("mc", {}).get("seed_eval", seed_opt + 1000003)))

    rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    search_detail: Dict[str, Any] = {}

    single_rows = baseline_summary[
        (baseline_summary["receiver"] == receiver) & (baseline_summary["baseline_type"] == "single_power")
    ]
    if single_rows.empty:
        raise ValueError(f"No single_power baseline row found for receiver={receiver}.")
    single_row = single_rows.iloc[0]
    rows.append(
        {
            "receiver": receiver,
            "method": "single_power",
            "method_zh": METHOD_ZH["single_power"],
            "G": float(single_row["G"]),
            "mc_mean_T": float(single_row["mc_mean_T"]),
            "mc_std_T": float(single_row["mc_std_T"]),
            "mc_mean_T_over_G": float(single_row["mc_mean_T_over_G"]),
            "mc_mean_T_over_Gtx": float(single_row["mc_mean_T_over_Gtx"]),
            "mc_mean_G_tx": float(single_row["mc_mean_G_tx"]),
            "pi0": float(single_row["pi0"]),
            "q_tx": float(single_row["q_tx"]),
            "avg_power": float(single_row["avg_power"]),
            "pi_full": str(single_row["pi_full"]),
            "beta": str(single_row["beta"]),
            "P": str(single_row["P"]),
            "source_dir": str(baseline_mc_dir),
            "search_note": "Single-power baseline imported from Experiment 40 MC evaluation",
        }
    )
    single_samples = baseline_samples[
        (baseline_samples["receiver"] == receiver) & (baseline_samples["baseline_type"] == "single_power")
    ]
    for _, row in single_samples.iterrows():
        sample_rows.append(
            {
                "receiver": receiver,
                "method": "single_power",
                "sample_id": int(row["sample_id"]),
                "T_sample": float(row["T_sample"]),
                "T_over_G_sample": float(row["T_over_G_sample"]),
                "T_over_Gtx_sample": float(row["T_over_Gtx_sample"]),
                "G_tx_sample": float(row["G_tx_sample"]),
                "source_dir": str(baseline_mc_dir),
            }
        )

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
        _row(
            receiver=receiver,
            method="structured_uniform_exp",
            config=config,
            policy=uniform_policy,
            result=uniform_eval,
            source_dir=result_dir,
            search_note="MC grid search over q_tx and exponential power ratio r",
        )
    )
    sample_rows.extend(_sample_rows(receiver, "structured_uniform_exp", uniform_eval, result_dir))
    search_detail["structured_uniform_exp"] = {
        "policy": uniform_policy.to_dict(),
        "opt_eval": uniform_opt.to_summary_dict(),
        "final_eval": uniform_eval.to_summary_dict(),
        "candidates": uniform_candidates,
    }

    init_local = _rescale_policy(ch3_policies[receiver]["local_search"], config, suffix="_mc_init")
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
        _row(
            receiver=receiver,
            method="local_search",
            config=config,
            policy=local_policy,
            result=local_eval,
            source_dir=result_dir,
            search_note="MC objective local search initialized from Part II local_search policy",
        )
    )
    sample_rows.extend(_sample_rows(receiver, "local_search", local_eval, result_dir))
    search_detail["local_search"] = {
        "init_policy": init_local.to_dict(),
        "policy": local_policy.to_dict(),
        "opt_eval": local_opt.to_summary_dict(),
        "final_eval": local_eval.to_summary_dict(),
        "trace": local_trace,
    }

    rows.append(
        {
            "receiver": receiver,
            "method": "PSO_MC",
            "method_zh": METHOD_ZH["PSO_MC"],
            "G": float(pso_summary["G"]),
            "mc_mean_T": float(pso_summary["eval_mc_mean_T"]),
            "mc_std_T": float(pso_summary["eval_mc_std_T"]),
            "mc_mean_T_over_G": float(pso_summary["eval_mc_mean_T_over_G"]),
            "mc_mean_T_over_Gtx": float(pso_summary["eval_mc_mean_T_over_Gtx"]),
            "mc_mean_G_tx": float(pso_summary["eval_mc_mean_G_tx"]),
            "pi0": float(pso_summary["pi0"]),
            "q_tx": float(pso_summary["q_tx"]),
            "avg_power": float(pso_summary["avg_power"]),
            "pi_full": str(pso_summary["pi_full"]),
            "beta": str(pso_summary["beta"]),
            "P": str(pso_summary["P"]),
            "source_dir": str(pso_dir),
            "search_note": "PSO optimized under Part III MC objective",
        }
    )

    for _, row in pso_samples.iterrows():
        sample_rows.append(
            {
                "receiver": receiver,
                "method": "PSO_MC",
                "sample_id": int(row["sample_id"]),
                "T_sample": float(row["T_sample"]),
                "T_over_G_sample": float(row["T_over_G_sample"]),
                "T_over_Gtx_sample": float(row["T_over_Gtx_sample"]),
                "G_tx_sample": float(row["G_tx_sample"]),
                "source_dir": str(pso_dir),
            }
        )

    paths = {
        "compare_csv": result_dir / "ch4_baseline_pso_compare.csv",
        "samples_csv": result_dir / "ch4_baseline_pso_compare_samples.csv",
        "search_json": result_dir / "ch4_mc_baseline_search_results.json",
        "metadata_json": result_dir / "metadata.json",
    }
    _save_csv(rows, paths["compare_csv"])
    _save_csv(sample_rows, paths["samples_csv"])
    save_json(search_detail, paths["search_json"])
    save_json(
        {
            "experiment_name": "exp_42_ch4_compare_baseline_pso",
            "created_at": created_at(),
            "baseline_mc_dir": str(baseline_mc_dir),
            "pso_dir": str(pso_dir),
            "ch3_source_dir": str(ch3_dir),
            "receiver": receiver,
            "required_methods": ["single_power", "structured_uniform_exp", "local_search", "PSO_MC"],
            "num_mc_opt": num_mc_opt,
            "num_mc_eval": num_mc_eval,
            "seed_opt": seed_opt,
            "seed_eval": seed_eval,
            "q_tx_grid": q_tx_grid,
            "r_grid": r_grid,
            "statement": "Part III compares three policies after optimizing them under the Part III Poisson random-load Monte Carlo objective: uniform-probability exponential-power grid search, MC-objective local search initialized from Part II local_search, and PSO_MC. No Rayleigh fading or explicit channel matrix simulation is used.",
        },
        paths["metadata_json"],
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    try:
        from dpc_mud.workflows.plot_part3_method_comparison import plot_all

        fig_dir = plot_all(result_dir)
        print(f"figures: {fig_dir}")
    except Exception as exc:
        print(f"warning: failed to auto-plot exp42 figures: {exc}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-mc-dir", required=True)
    parser.add_argument("--pso-dir", required=True)
    args = parser.parse_args()
    run_experiment(baseline_mc_dir=Path(args.baseline_mc_dir), pso_dir=Path(args.pso_dir))


if __name__ == "__main__":
    main()
