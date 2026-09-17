"""
Shared utilities for Part II grouped-load experiments.

The main probability notation is pi. Legacy alpha keys may be read for backward
compatibility, but new CSV/JSON outputs use pi_full, pi_pos, pi0, and beta.
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

from .initializers import make_initial_policy, make_pi_full, make_random_policy, scale_power_to_pbar
from .model import GroupedConfig, GroupedEvalResult, GroupedPolicy
from .receivers import evaluate_grouped_policy


DEFAULT_RECEIVERS = ["global_od", "group_od", "group_mf", "group_lmmse"]
DEFAULT_R_GRID = [1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]
DEFAULT_Q_TX_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def portable_value(value: Any) -> Any:
    """Return JSON/CSV-safe values without machine-specific absolute paths."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            repo_root = os.environ.get("DPC_MUD_REPO_ROOT")
            if repo_root:
                try:
                    return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
                except ValueError:
                    pass
            return path.name
        return value.replace("\\", "/")
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_value(item) for item in value]
    return value


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a dictionary: {path}")
    return data


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(portable_value(data), f, ensure_ascii=False, indent=2, default=json_default)


def save_csv(rows: Sequence[Dict[str, Any]], path: Path, *, float_precision: int = 8) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key, value in row.items():
                portable = portable_value(value)
                out[key] = f"{portable:.{float_precision}f}" if isinstance(portable, float) else portable
            writer.writerow(out)


def make_result_dir(project_root: Path, root_name: str, prefix: str) -> Path:
    override = os.environ.get("DPC_MUD_OUTPUT_ROOT")
    root = Path(override) if override else project_root / root_name
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{prefix}_{timestamp()}"
    for idx in range(1000):
        result_dir = base if idx == 0 else root / f"{base.name}_{idx:02d}"
        try:
            result_dir.mkdir(parents=True, exist_ok=False)
            return result_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique result directory for {base}.")


def build_grouped_config(full_cfg: Dict[str, Any], *, S: int | None = None) -> GroupedConfig:
    system = full_cfg.get("system", {})
    if "G" in system:
        G = float(system["G"])
    else:
        G = float(system.get("G_tx", 200.0))
        warnings.warn("Legacy G_tx found; treating it as total offered load G.", RuntimeWarning, stacklevel=2)
    return GroupedConfig(
        N=int(system.get("N", 8)),
        S=int(S if S is not None else system.get("S", 8)),
        R=float(system.get("R", 0.25)),
        G=G,
        Pbar=float(system.get("Pbar", 1.0)),
        sigma2=float(system.get("sigma2", 1.0)),
        stop_on_failure=bool(system.get("stop_on_failure", True)),
        use_full_power=bool(system.get("use_full_power", True)),
        q_tx_min=float(system.get("q_tx_min", full_cfg.get("transmission", {}).get("q_tx_min", 0.05))),
        q_tx_max=float(system.get("q_tx_max", full_cfg.get("transmission", {}).get("q_tx_max", 1.0))),
        Pmax=system.get("Pmax", None),
    )


def quicken_grouped_config(full_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(full_cfg))
    quick_cfg = cfg.get("quick", {})
    cfg.setdefault("transmission", {})["q_tx_grid"] = quick_cfg.get("q_tx_grid", [0.5, 1.0])
    cfg.setdefault("structured_initialization", {})["r_grid"] = quick_cfg.get("r_grid", [1.2, 2.0])
    cfg["structured_initialization"]["beta_modes"] = ["uniform", "exp_increasing", "exp_decreasing", "gaussian", "poisson"]
    cfg["structured_initialization"]["power_modes"] = ["exponential", "linear"]
    cfg.setdefault("random_initialization", {})["num_inits"] = int(quick_cfg.get("random_inits", 3))
    cfg["random_initialization"]["method"] = "dirichlet"
    cfg["random_initialization"]["theta_values"] = [1.0]
    cfg["random_initialization"]["power_modes"] = ["random_increments"]
    cfg.setdefault("local_search", {})["num_steps"] = int(quick_cfg.get("local_steps", 5))
    cfg.setdefault("outputs", {})["root"] = "results_quick"
    return cfg


def get_receivers(full_cfg: Dict[str, Any]) -> List[str]:
    if "receivers" in full_cfg:
        return [str(x) for x in full_cfg["receivers"]]
    exp = full_cfg.get("experiments", {}).get("s8_main_baseline", full_cfg.get("experiments", {}).get("grouped_baseline", {}))
    return [str(x) for x in exp.get("receivers", DEFAULT_RECEIVERS)]


def get_r_grid(full_cfg: Dict[str, Any], *, quick: bool = False) -> List[float]:
    grid = full_cfg.get("structured_initialization", {}).get(
        "r_grid",
        full_cfg.get("experiments", {}).get("s8_main_baseline", {}).get("r_grid", DEFAULT_R_GRID),
    )
    return [1.2, 2.0] if quick else [float(x) for x in grid]


def get_q_tx_grid(full_cfg: Dict[str, Any], *, quick: bool = False) -> List[float]:
    grid = full_cfg.get("transmission", {}).get("q_tx_grid", DEFAULT_Q_TX_GRID)
    return [0.5, 1.0] if quick else [float(x) for x in grid]


def expand_structured_specs(full_cfg: Dict[str, Any], *, q_tx_grid: Sequence[float] | None = None, quick: bool = False) -> List[Dict[str, Any]]:
    init = full_cfg.get("structured_initialization", full_cfg.get("initialization", {}).get("structured", {}))
    beta_modes = init.get("beta_modes", [x.get("mode", x) if isinstance(x, dict) else x for x in init.get("alpha_modes", ["uniform"])])
    power_modes = init.get("power_modes", ["exponential", "linear"])
    if power_modes and isinstance(power_modes[0], dict):
        power_modes = [x["mode"] for x in power_modes]
    q_values = list(q_tx_grid if q_tx_grid is not None else get_q_tx_grid(full_cfg, quick=quick))
    r_values = get_r_grid(full_cfg, quick=quick)
    a_grid = [1.5] if quick else [float(x) for x in init.get("a_grid", [1.2, 1.5, 2.0])]
    sigma_grid = [0.8] if quick else [float(x) for x in init.get("gaussian_sigma_grid", [0.2, 0.4, 0.8])]
    lambda_grid = [2.0] if quick else [float(x) for x in init.get("poisson_lambda_grid", [1.0, 2.0, 3.0])]

    beta_specs: List[Dict[str, Any]] = []
    for mode in [str(x) for x in beta_modes]:
        if mode in {"exp_increasing", "exp_decreasing"}:
            beta_specs.extend({"beta_mode": mode, "kwargs": {"a": a}} for a in a_grid)
        elif mode == "gaussian":
            beta_specs.extend({"beta_mode": mode, "kwargs": {"sigma": s}} for s in sigma_grid)
        elif mode == "poisson":
            beta_specs.extend({"beta_mode": mode, "kwargs": {"lambda_p": l}} for l in lambda_grid)
        else:
            beta_specs.append({"beta_mode": mode, "kwargs": {}})

    specs: List[Dict[str, Any]] = []
    for beta_spec in beta_specs:
        for power_mode in [str(x) for x in power_modes]:
            power_kwargs_list = [{"r": r} for r in r_values] if power_mode == "exponential" else [{}]
            for power_kwargs in power_kwargs_list:
                for q_tx in q_values:
                    specs.append(
                        {
                            "beta_mode": beta_spec["beta_mode"],
                            "power_mode": power_mode,
                            "q_tx": float(q_tx),
                            "kwargs": {**beta_spec["kwargs"], **power_kwargs, "q_tx": float(q_tx)},
                        }
                    )
    return specs


def expand_random_specs(full_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    init = full_cfg.get("random_initialization", full_cfg.get("initialization", {}).get("random", {}))
    method = str(init.get("method", "dirichlet"))
    beta_modes = [method] if method == "dirichlet" else [str(x) for x in init.get("beta_modes", init.get("alpha_modes", ["dirichlet"]))]
    power_modes = [str(x) for x in init.get("power_modes", ["random_increments"])]
    theta_values = [float(x) for x in init.get("theta_values", [1.0])]
    system = full_cfg.get("system", {})
    q_min = float(system.get("q_tx_min", full_cfg.get("transmission", {}).get("q_tx_min", 0.05)))
    q_max = float(system.get("q_tx_max", full_cfg.get("transmission", {}).get("q_tx_max", 1.0)))
    specs: List[Dict[str, Any]] = []
    for beta_mode in beta_modes:
        for power_mode in power_modes:
            for theta in (theta_values if beta_mode == "dirichlet" else [1.0]):
                specs.append(
                    {
                        "beta_mode": beta_mode,
                        "power_mode": power_mode,
                        "theta": float(theta),
                        "kwargs": {
                            "q_tx_mode": "random",
                            "q_tx_min": q_min,
                            "q_tx_max": q_max,
                            "theta": float(theta),
                        },
                    }
                )
    return specs


def single_power_policy(config: GroupedConfig, *, q_tx: float = 1.0) -> GroupedPolicy:
    beta = np.zeros(config.S, dtype=float)
    beta[-1] = 1.0
    pi_full = make_pi_full(beta, q_tx, eps=max(config.q_tx_min, 1e-12), beta_eps=0.0)
    P_raw = np.arange(1, config.S + 1, dtype=float)
    P = scale_power_to_pbar(pi_full[1:], P_raw, config.Pbar)
    return GroupedPolicy(
        name=f"single_power_baseline_q{q_tx:.3g}",
        pi_full=pi_full,
        P=P,
        extra={"baseline_type": "single_power_baseline", "init_type": "single_power", "beta_mode": "single_highest_layer", "power_mode": "embedded_linear", "q_tx": float(q_tx)},
    )


def uniform_alpha_exp_power_policy(config: GroupedConfig, *, r: float, q_tx: float = 1.0) -> GroupedPolicy:
    """Compatibility name for uniform beta plus exponential power."""

    beta = np.full(config.S, 1.0 / config.S, dtype=float)
    pi_full = make_pi_full(beta, q_tx, eps=max(config.q_tx_min, 1e-12))
    P_raw = np.asarray([float(r) ** s for s in range(config.S)], dtype=float)
    P = scale_power_to_pbar(pi_full[1:], P_raw, config.Pbar)
    return GroupedPolicy(
        name=f"uniform_beta_exp_power_q{q_tx:.3g}_r{r:.3g}",
        pi_full=pi_full,
        P=P,
        extra={"baseline_type": "structured_initialization_baseline", "init_type": "structured", "beta_mode": "uniform", "power_mode": "exponential", "r": float(r), "q_tx": float(q_tx)},
    )


def make_structured_policy_from_spec(config: GroupedConfig, spec: Dict[str, Any], rng: np.random.Generator) -> GroupedPolicy:
    return make_initial_policy(config, alpha_mode=str(spec.get("beta_mode", "uniform")), power_mode=str(spec["power_mode"]), rng=rng, **dict(spec.get("kwargs", {})))


def make_random_policy_from_spec(config: GroupedConfig, spec: Dict[str, Any], rng: np.random.Generator) -> GroupedPolicy:
    return make_random_policy(config, alpha_mode=str(spec.get("beta_mode", "dirichlet")), power_mode=str(spec["power_mode"]), rng=rng, **dict(spec.get("kwargs", {})))


def vector_json(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=float).tolist())


def flags_json(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=bool).tolist())


def result_row(
    *,
    receiver: str,
    baseline_type: str,
    policy: GroupedPolicy,
    result: GroupedEvalResult,
    method: str,
    init_type: str = "",
    init_beta_mode: str = "",
    init_alpha_mode: str = "",
    init_power_mode: str = "",
    local_gain: float = 0.0,
    accepted_steps: int = 0,
    best_r: float | None = None,
    best_q_tx: float | None = None,
    run_index: int = -1,
) -> Dict[str, Any]:
    beta_mode = init_beta_mode or init_alpha_mode or str(policy.extra.get("beta_mode", ""))
    return {
        "S": int(result.S),
        "receiver": receiver,
        "baseline_type": baseline_type,
        "policy_name": policy.name,
        "method": method,
        "T_packets": float(result.T_packets),
        "T_rate": float(result.T_rate),
        "T_over_G": float(result.T_over_G),
        "T_over_Gtx": float(result.T_over_Gtx),
        "success_ratio": float(result.success_ratio),
        "G": float(result.G),
        "G_tx": float(result.G_tx),
        "pi0": float(result.pi0),
        "q_tx": float(result.q_tx),
        "avg_power": float(result.avg_power),
        "failed_group": "" if result.failed_group is None else int(result.failed_group),
        "pi_full": vector_json(result.pi_full),
        "pi_pos": vector_json(result.pi_pos),
        "beta": vector_json(result.beta),
        "P": vector_json(result.P),
        "K0": float(result.K0),
        "K_groups": vector_json(result.K_groups),
        "group_success_users": vector_json(result.group_success_users),
        "group_success_flags": flags_json(result.group_success_flags),
        "init_type": init_type or str(policy.extra.get("init_type", "")),
        "init_beta_mode": beta_mode,
        "init_power_mode": init_power_mode or str(policy.extra.get("power_mode", "")),
        "local_gain": float(local_gain),
        "accepted_steps": int(accepted_steps),
        "best_r": "" if best_r is None else float(best_r),
        "best_q_tx": "" if best_q_tx is None else float(best_q_tx),
        "run_index": int(run_index),
    }


def evaluate_policy_dict(config: GroupedConfig, policy_dict: Dict[str, Any], receiver: str) -> GroupedEvalResult:
    if "pi_full" in policy_dict:
        pi_full = np.asarray(policy_dict["pi_full"], dtype=float)
    elif "alpha_full" in policy_dict:
        pi_full = np.asarray(policy_dict["alpha_full"], dtype=float)
    else:
        pi_full = np.concatenate([[0.0], np.asarray(policy_dict.get("alpha", policy_dict["pi_pos"]), dtype=float)])
    policy = GroupedPolicy(name=str(policy_dict.get("name", "loaded_policy")), pi_full=pi_full, P=np.asarray(policy_dict["P"], dtype=float), extra=dict(policy_dict.get("extra", {})))
    return evaluate_grouped_policy(config, policy, receiver)
