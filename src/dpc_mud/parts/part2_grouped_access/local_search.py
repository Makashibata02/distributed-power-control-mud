"""
Derivative-free constrained local search for grouped-load baselines.

The search perturbs q_tx, beta, and positive power increments. It is not convex
optimization and not a global optimization method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .initializers import (
    ensure_increasing_power,
    make_initial_policy,
    make_pi_full,
    make_random_policy,
    normalize_beta,
    scale_power_to_pbar,
)
from .model import GroupedConfig, GroupedPolicy, validate_grouped_policy
from .receivers import evaluate_grouped_policy


@dataclass(frozen=True)
class LocalSearchRun:
    init_policy: GroupedPolicy
    best_policy: GroupedPolicy
    init_result: Any
    best_result: Any
    search_trace: List[Dict[str, Any]]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def local_gain(self) -> float:
        return float(self.best_result.T_packets - self.init_result.T_packets)

    @property
    def accepted_steps(self) -> int:
        return int(sum(1 for row in self.search_trace if row.get("accepted", False)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "init_policy": self.init_policy.to_dict(),
            "best_policy": self.best_policy.to_dict(),
            "init_result": self.init_result.to_dict(),
            "best_result": self.best_result.to_dict(),
            "local_gain": self.local_gain,
            "accepted_steps": self.accepted_steps,
            "search_trace": self.search_trace,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class MultiStartSearchResult:
    receiver: str
    best_run: LocalSearchRun
    runs: List[LocalSearchRun]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_traces: bool = True) -> Dict[str, Any]:
        return {
            "receiver": self.receiver,
            "best_run": self.best_run.to_dict(),
            "num_runs": len(self.runs),
            "runs": [run.to_dict() if include_traces else {
                "init_policy": run.init_policy.to_dict(),
                "best_policy": run.best_policy.to_dict(),
                "init_result": run.init_result.to_dict(),
                "best_result": run.best_result.to_dict(),
                "local_gain": run.local_gain,
                "accepted_steps": run.accepted_steps,
                "extra": run.extra,
            } for run in self.runs],
            "extra": self.extra,
        }


def project_alpha(alpha: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compatibility helper: project positive-layer weights to beta."""

    return normalize_beta(alpha, eps=eps)


def project_power(pi_pos: np.ndarray, P_raw: np.ndarray, Pbar: float, eps: float = 1e-8) -> np.ndarray:
    return scale_power_to_pbar(pi_pos, ensure_increasing_power(P_raw, eps=eps), Pbar)


def _power_to_increments(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    P = ensure_increasing_power(P, eps=eps)
    return np.maximum(P - np.concatenate([[0.0], P[:-1]]), eps)


def _logit(x: float, eps: float = 1e-9) -> float:
    x = float(np.clip(x, eps, 1.0 - eps))
    return float(np.log(x / (1.0 - x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def perturb_policy(
    policy: GroupedPolicy,
    config: GroupedConfig,
    rng: np.random.Generator,
    beta_scale: float = 0.10,
    power_scale: float = 0.10,
    q_scale: float = 0.20,
    alpha_scale: float | None = None,
) -> GroupedPolicy:
    """Perturb q_tx, beta, and power increments, then project to constraints."""

    if alpha_scale is not None:
        beta_scale = float(alpha_scale)
    q_min = config.q_tx_min
    q_max = config.q_tx_max
    if q_max >= 1.0 and policy.q_tx >= 1.0 - 1e-12:
        q_new = 1.0
    else:
        q_new = _sigmoid(_logit(policy.q_tx) + float(rng.normal(0.0, q_scale)))
        q_new = float(np.clip(q_new, q_min, q_max))

    beta_new = normalize_beta(policy.beta * np.exp(rng.normal(0.0, beta_scale, size=config.S)))
    increments = _power_to_increments(policy.P)
    increments_new = increments * np.exp(rng.normal(0.0, power_scale, size=config.S))
    P_raw = np.cumsum(increments_new)
    pi_full = make_pi_full(beta_new, q_new, eps=max(q_min, 1e-12))
    P_new = project_power(pi_full[1:], P_raw, config.Pbar)

    candidate = GroupedPolicy(
        name=policy.name,
        pi_full=pi_full,
        P=P_new,
        extra={**policy.extra, "local_search_perturbed": True, "q_tx": q_new},
    )
    validate_grouped_policy(candidate, config, require_full_power=config.use_full_power)
    return candidate


def local_search_grouped_policy(
    *,
    config: GroupedConfig,
    init_policy: GroupedPolicy,
    receiver: str,
    num_steps: int,
    rng: np.random.Generator,
    beta_scale: float = 0.10,
    power_scale: float = 0.10,
    q_scale: float = 0.20,
    alpha_scale: float | None = None,
) -> Tuple[LocalSearchRun, List[Dict[str, Any]]]:
    if alpha_scale is not None:
        beta_scale = float(alpha_scale)
    validate_grouped_policy(init_policy, config, require_full_power=config.use_full_power)
    init_result = evaluate_grouped_policy(config, init_policy, receiver)
    best_policy = init_policy
    best_result = init_result
    trace: List[Dict[str, Any]] = []

    for step in range(int(num_steps)):
        candidate = perturb_policy(
            best_policy,
            config,
            rng,
            beta_scale=beta_scale,
            power_scale=power_scale,
            q_scale=q_scale,
        )
        candidate_result = evaluate_grouped_policy(config, candidate, receiver)
        accepted = bool(candidate_result.T_packets > best_result.T_packets + 1e-12)
        if accepted:
            best_policy = candidate
            best_result = candidate_result
        trace.append(
            {
                "step": int(step),
                "candidate_T_packets": float(candidate_result.T_packets),
                "best_T_packets": float(best_result.T_packets),
                "candidate_T_over_G": float(candidate_result.T_over_G),
                "best_T_over_G": float(best_result.T_over_G),
                "candidate_pi0": float(candidate.pi0),
                "candidate_q_tx": float(candidate.q_tx),
                "best_pi0": float(best_policy.pi0),
                "best_q_tx": float(best_policy.q_tx),
                "candidate_avg_power": float(candidate.avg_power),
                "best_avg_power": float(best_policy.avg_power),
                "accepted": accepted,
            }
        )

    run = LocalSearchRun(
        init_policy=init_policy,
        best_policy=best_policy,
        init_result=init_result,
        best_result=best_result,
        search_trace=trace,
        extra={
            "receiver": receiver,
            "num_steps": int(num_steps),
            "beta_scale": float(beta_scale),
            "power_scale": float(power_scale),
            "q_scale": float(q_scale),
            "note": "derivative-free constrained local search, not convex optimization",
        },
    )
    return run, trace


def _build_structured_initial_policies(config: GroupedConfig, specs: Sequence[Dict[str, Any]], rng: np.random.Generator) -> List[GroupedPolicy]:
    return [
        make_initial_policy(
            config,
            alpha_mode=str(spec.get("beta_mode", spec.get("alpha_mode", "uniform"))),
            power_mode=str(spec["power_mode"]),
            rng=rng,
            **dict(spec.get("kwargs", {})),
        )
        for spec in specs
    ]


def _build_random_initial_policies(
    config: GroupedConfig,
    specs: Sequence[Dict[str, Any]],
    num_random_inits: int,
    rng: np.random.Generator,
) -> List[GroupedPolicy]:
    policies: List[GroupedPolicy] = []
    for idx in range(int(num_random_inits)):
        if not specs:
            break
        spec = specs[idx % len(specs)]
        policies.append(
            make_random_policy(
                config,
                alpha_mode=str(spec.get("beta_mode", spec.get("alpha_mode", "dirichlet"))),
                power_mode=str(spec["power_mode"]),
                rng=rng,
                **dict(spec.get("kwargs", {})),
            )
        )
    return policies


def multi_start_local_search(
    *,
    config: GroupedConfig,
    receiver: str,
    structured_init_specs: Sequence[Dict[str, Any]],
    random_init_specs: Sequence[Dict[str, Any]],
    num_random_inits: int,
    local_steps: int,
    seed: int,
    beta_scale: float = 0.10,
    power_scale: float = 0.10,
    q_scale: float = 0.20,
    alpha_scale: float | None = None,
) -> MultiStartSearchResult:
    if alpha_scale is not None:
        beta_scale = float(alpha_scale)
    rng = np.random.default_rng(int(seed))
    init_policies = _build_structured_initial_policies(config, structured_init_specs, rng)
    init_policies.extend(_build_random_initial_policies(config, random_init_specs, num_random_inits, rng))
    if not init_policies:
        raise ValueError("No initial policies were generated for multi-start search.")

    runs: List[LocalSearchRun] = []
    for init_policy in init_policies:
        run, _ = local_search_grouped_policy(
            config=config,
            init_policy=init_policy,
            receiver=receiver,
            num_steps=local_steps,
            rng=rng,
            beta_scale=beta_scale,
            power_scale=power_scale,
            q_scale=q_scale,
        )
        runs.append(run)

    best_run = max(runs, key=lambda run: run.best_result.T_packets)
    return MultiStartSearchResult(
        receiver=receiver,
        best_run=best_run,
        runs=runs,
        extra={
            "seed": int(seed),
            "num_structured_inits": len(structured_init_specs),
            "num_random_inits": int(num_random_inits),
            "local_steps": int(local_steps),
            "beta_scale": float(beta_scale),
            "power_scale": float(power_scale),
            "q_scale": float(q_scale),
            "note": "derivative-free constrained local search, not convex optimization",
        },
    )
