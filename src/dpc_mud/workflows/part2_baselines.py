"""Part II main experiment: formula-based grouped-load baselines.

Part II uses deterministic mean-load formula evaluation only. It does not
use Monte Carlo as a result source. The exported best policies are intended to
be read by Part III for Poisson-load Monte Carlo re-evaluation.
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dpc_mud.parts.part2_grouped_access.experiment_utils import (
    build_grouped_config,
    evaluate_grouped_policy,
    expand_random_specs,
    expand_structured_specs,
    get_q_tx_grid,
    get_r_grid,
    get_receivers,
    load_yaml,
    make_random_policy_from_spec,
    make_result_dir,
    make_structured_policy_from_spec,
    quicken_grouped_config,
    result_row,
    save_csv,
    save_json,
    single_power_policy,
    uniform_alpha_exp_power_policy,
)
from dpc_mud.parts.part2_grouped_access.local_search import local_search_grouped_policy


BASELINE_ORDER = [
    "single_power",
    "structured_uniform_exp",
    "structured_all",
    "random_dirichlet",
    "local_search",
]


def _best(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("Cannot select best item from an empty list.")
    return max(items, key=lambda item: item["result"].T_packets)


def _payload(policy, result, *, init_result=None, trace=None, **extra: Any) -> Dict[str, Any]:
    return {
        "policy": policy.to_dict(),
        "init_result": (init_result or result).to_dict(),
        "best_result": result.to_dict(),
        "trace": trace or [],
        **extra,
    }


def _run_single_power(config, receiver: str, q_grid: List[float]) -> Dict[str, Any]:
    items = []
    rows = []
    for idx, q_tx in enumerate(q_grid):
        policy = single_power_policy(config, q_tx=float(q_tx))
        result = evaluate_grouped_policy(config, policy, receiver)
        row = result_row(
            receiver=receiver,
            baseline_type="single_power",
            policy=policy,
            result=result,
            method="q_tx_grid",
            best_q_tx=float(q_tx),
            run_index=idx,
        )
        items.append({"policy": policy, "result": result, "row": row, "q_tx": float(q_tx)})
        rows.append(row)
    item = _best(items)
    summary = dict(item["row"])
    summary["method"] = "best_q_tx_grid"
    return {"summary_row": summary, "candidate_rows": rows, "payload": _payload(item["policy"], item["result"], best_q_tx=item["q_tx"])}


def _run_structured_uniform_exp(config, receiver: str, q_grid: List[float], r_grid: List[float]) -> Dict[str, Any]:
    items = []
    rows = []
    run_index = 0
    for q_tx in q_grid:
        for r in r_grid:
            policy = uniform_alpha_exp_power_policy(config, r=float(r), q_tx=float(q_tx))
            result = evaluate_grouped_policy(config, policy, receiver)
            row = result_row(
                receiver=receiver,
                baseline_type="structured_uniform_exp",
                policy=policy,
                result=result,
                method="q_tx_r_grid",
                best_r=float(r),
                best_q_tx=float(q_tx),
                run_index=run_index,
            )
            items.append({"policy": policy, "result": result, "row": row, "q_tx": float(q_tx), "r": float(r)})
            rows.append(row)
            run_index += 1
    item = _best(items)
    summary = dict(item["row"])
    summary["method"] = "best_q_tx_r_grid"
    return {"summary_row": summary, "candidate_rows": rows, "payload": _payload(item["policy"], item["result"], best_q_tx=item["q_tx"], best_r=item["r"])}


def _run_structured_all(config, full_cfg: Dict[str, Any], receiver: str, quick: bool, rng: np.random.Generator) -> Dict[str, Any]:
    items = []
    rows = []
    policies = []
    for idx, spec in enumerate(expand_structured_specs(full_cfg, quick=quick)):
        policy = make_structured_policy_from_spec(config, spec, rng)
        policy = type(policy)(
            name=policy.name,
            pi_full=policy.pi_full,
            P=policy.P,
            extra={**policy.extra, "baseline_type": "structured_all"},
        )
        result = evaluate_grouped_policy(config, policy, receiver)
        row = result_row(
            receiver=receiver,
            baseline_type="structured_all",
            policy=policy,
            result=result,
            method="structured_initializer",
            best_r=policy.extra.get("power_kwargs", {}).get("r"),
            best_q_tx=policy.q_tx,
            run_index=idx,
        )
        items.append({"policy": policy, "result": result, "row": row})
        rows.append(row)
        policies.append(policy)
    item = _best(items)
    summary = dict(item["row"])
    summary["method"] = "best_structured_initializer"
    return {"summary_row": summary, "candidate_rows": rows, "payload": _payload(item["policy"], item["result"]), "policies": policies}


def _run_random_dirichlet(config, full_cfg: Dict[str, Any], receiver: str, rng: np.random.Generator) -> Dict[str, Any]:
    specs = expand_random_specs(full_cfg)
    random_cfg = full_cfg.get("random_initialization", {})
    num_inits = int(random_cfg.get("num_inits", 32))
    items = []
    rows = []
    policies = []
    for idx in range(num_inits):
        spec = specs[idx % len(specs)]
        policy = make_random_policy_from_spec(config, spec, rng)
        policy = type(policy)(
            name=policy.name,
            pi_full=policy.pi_full,
            P=policy.P,
            extra={**policy.extra, "baseline_type": "random_dirichlet", "theta": spec.get("theta", 1.0)},
        )
        result = evaluate_grouped_policy(config, policy, receiver)
        row = result_row(
            receiver=receiver,
            baseline_type="random_dirichlet",
            policy=policy,
            result=result,
            method="random_dirichlet_initializer",
            best_q_tx=policy.q_tx,
            run_index=idx,
        )
        items.append({"policy": policy, "result": result, "row": row})
        rows.append(row)
        policies.append(policy)
    item = _best(items)
    summary = dict(item["row"])
    summary["method"] = "best_random_dirichlet"
    return {"summary_row": summary, "candidate_rows": rows, "payload": _payload(item["policy"], item["result"]), "policies": policies}


def _run_local_search(config, full_cfg: Dict[str, Any], receiver: str, init_policies: List[Any], seed: int) -> Dict[str, Any]:
    local_cfg = full_cfg.get("local_search", {})
    rng = np.random.default_rng(int(seed))
    items = []
    rows = []
    traces: Dict[str, Any] = {}
    for idx, policy in enumerate(init_policies):
        run, trace = local_search_grouped_policy(
            config=config,
            init_policy=policy,
            receiver=receiver,
            num_steps=int(local_cfg.get("num_steps", 50)),
            rng=rng,
            beta_scale=float(local_cfg.get("beta_scale", 0.10)),
            power_scale=float(local_cfg.get("power_scale", 0.10)),
            q_scale=float(local_cfg.get("q_scale", 0.20)),
        )
        row = result_row(
            receiver=receiver,
            baseline_type="local_search",
            policy=run.best_policy,
            result=run.best_result,
            method="local_search",
            local_gain=run.local_gain,
            accepted_steps=run.accepted_steps,
            best_q_tx=run.best_policy.q_tx,
            run_index=idx,
        )
        rows.append(row)
        items.append({"run": run, "policy": run.best_policy, "result": run.best_result, "row": row})
        traces[f"{receiver}_run_{idx}"] = run.to_dict()
    item = _best(items)
    summary = dict(item["row"])
    summary["method"] = "best_local_search"
    run = item["run"]
    return {
        "summary_row": summary,
        "candidate_rows": rows,
        "payload": _payload(
            run.best_policy,
            run.best_result,
            init_result=run.init_result,
            trace=run.search_trace,
            local_gain=run.local_gain,
            accepted_steps=run.accepted_steps,
        ),
        "traces": traces,
    }


def run_experiment(
    *,
    quick: bool = False,
    config_path: Path | None = None,
    auto_plot: bool = True,
) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path or (project_root / "configs" / "part2_grouped_access.yaml")
    full_cfg = load_yaml(config_path)
    if quick:
        full_cfg = quicken_grouped_config(full_cfg)
        print("[quick] Running reduced Part II formula baselines.")

    config = build_grouped_config(full_cfg, S=int(full_cfg.get("system", {}).get("S", 8)))
    receivers = get_receivers(full_cfg)
    q_grid = get_q_tx_grid(full_cfg, quick=quick)
    r_grid = get_r_grid(full_cfg, quick=quick)
    seed = int(full_cfg.get("local_search", {}).get("seed", 20260520))
    root_name = "results_quick" if quick else str(full_cfg.get("outputs", {}).get("root", "results"))
    result_dir = make_result_dir(project_root, root_name, "ch3_exp10_formula_baselines")
    float_precision = int(full_cfg.get("outputs", {}).get("float_precision", 8))

    summary_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    structured_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    traces: Dict[str, Any] = {}
    nested_best: Dict[str, Dict[str, Any]] = {}

    print("=" * 88)
    print("exp_10_ch3_formula_baselines")
    print("-" * 88)
    for ridx, receiver in enumerate(receivers):
        rng = np.random.default_rng(seed + 1000 * ridx)
        print(f"[receiver] {receiver}")
        single = _run_single_power(config, receiver, q_grid)
        uniform_exp = _run_structured_uniform_exp(config, receiver, q_grid, r_grid)
        structured = _run_structured_all(config, full_cfg, receiver, quick, rng)
        random = _run_random_dirichlet(config, full_cfg, receiver, rng)
        local = _run_local_search(
            config,
            full_cfg,
            receiver,
            structured["policies"] + random["policies"],
            seed + 5000 * ridx,
        )

        receiver_results = {
            "single_power": single,
            "structured_uniform_exp": uniform_exp,
            "structured_all": structured,
            "random_dirichlet": random,
            "local_search": local,
        }
        nested_best[receiver] = {
            "single_power": single["payload"],
            "structured_uniform_exp": uniform_exp["payload"],
            "structured_best": structured["payload"],
            "random_dirichlet_best": random["payload"],
            "local_search": local["payload"],
        }
        for key in BASELINE_ORDER:
            data = receiver_results[key]
            summary_rows.append(data["summary_row"])
            all_rows.extend(data["candidate_rows"])
            if key == "structured_all":
                structured_rows.extend(data["candidate_rows"])
            if key == "random_dirichlet":
                random_rows.extend(data["candidate_rows"])
            if key == "local_search":
                local_rows.extend(data["candidate_rows"])
                traces.update(data["traces"])
            row = data["summary_row"]
            print(
                f"  {key:<24s} "
                f"T={float(row['T_packets']):.6f}, "
                f"T/G={float(row['T_over_G']):.6f}, "
                f"T/G_tx={float(row['T_over_Gtx']):.6f}"
            )

    metadata = {
        "experiment": "exp_10_ch3_formula_baselines",
        "quick": bool(quick),
        "model_note": "Part II uses deterministic mean-load formula evaluation.",
        "mc_note": "Part II does not use Monte Carlo for results.",
        "part3_stochastic_optimization_note": "Part III re-evaluates Part II baselines under Poisson-load Monte Carlo.",
        "notation_note": "Probability notation uses pi, consistent with Part I.",
        "local_search_note": "derivative-free constrained local search, not convex optimization or global optimization",
        "rayleigh_note": "No Rayleigh fading or explicit channel matrices are used.",
        "config_path": str(config_path),
        "result_dir": str(result_dir),
        "config": config.to_dict(),
        "q_tx_grid": q_grid,
        "r_grid": r_grid,
        "receivers": receivers,
        "full_config": full_cfg,
    }
    best_json = {"metadata": metadata, "best_policies": nested_best}

    paths = {
        "summary_csv": result_dir / "ch3_formula_summary.csv",
        "all_candidates_csv": result_dir / "ch3_all_candidates.csv",
        "structured_csv": result_dir / "ch3_structured_initializers.csv",
        "random_csv": result_dir / "ch3_random_initializers.csv",
        "local_csv": result_dir / "ch3_local_search_runs.csv",
        "best_policies_json": result_dir / "ch3_best_policies.json",
        "search_traces_json": result_dir / "ch3_search_traces.json",
        "metadata_json": result_dir / "metadata.json",
    }
    save_csv(summary_rows, paths["summary_csv"], float_precision=float_precision)
    save_csv(all_rows, paths["all_candidates_csv"], float_precision=float_precision)
    save_csv(structured_rows, paths["structured_csv"], float_precision=float_precision)
    save_csv(random_rows, paths["random_csv"], float_precision=float_precision)
    save_csv(local_rows, paths["local_csv"], float_precision=float_precision)
    save_json(best_json, paths["best_policies_json"])
    save_json(traces, paths["search_traces_json"])
    save_json(metadata, paths["metadata_json"])
    if auto_plot:
        from dpc_mud.workflows.plot_part2_baselines import plot_all

        paths["figures_dir"] = plot_all(result_dir, dpi=300)
    print("-" * 88)
    for name, path in paths.items():
        print(f"{name:22s}: {path}")
    print("=" * 88)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Only save CSV/JSON, do not generate figures.")
    args = parser.parse_args()
    run_experiment(
        quick=bool(args.quick),
        config_path=None if args.config is None else Path(args.config),
        auto_plot=not bool(args.no_plot),
    )


if __name__ == "__main__":
    main()
