"""Basic PSO optimizer using Part III Monte Carlo objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np

from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy
from .monte_carlo import MCEvalResult, mc_evaluate_policy
from .encoding import decode_particle, policy_to_particle_approx


@dataclass(frozen=True)
class PSOResult:
    best_policy: GroupedPolicy
    best_score: float
    opt_eval: MCEvalResult
    final_eval: MCEvalResult
    trace: list[dict[str, Any]] = field(default_factory=list)


def _score(result: MCEvalResult, metric: str) -> float:
    if metric == "mc_mean_T":
        return float(result.mc_mean_T)
    if metric == "mc_mean_T_over_G":
        return float(result.mc_mean_T_over_G)
    raise ValueError(f"Unsupported objective_metric: {metric}")


def _initial_positions(
    config: GroupedConfig,
    rng: np.random.Generator,
    num_particles: int,
    warm_policies: Sequence[GroupedPolicy],
    noise: float,
    warm_ratio: float = 0.4,
) -> np.ndarray:
    dim = 2 * config.S + 1
    positions = rng.normal(0.0, 1.0, size=(num_particles, dim))
    if warm_policies:
        warm_ratio = float(np.clip(warm_ratio, 0.0, 1.0))
        warm_count = int(round(warm_ratio * num_particles))
        if warm_ratio > 0.0:
            warm_count = max(1, warm_count)
        warm_count = min(num_particles, warm_count)
        for idx in range(warm_count):
            base = policy_to_particle_approx(warm_policies[idx % len(warm_policies)], config)
            positions[idx] = base + rng.normal(0.0, noise, size=dim)
    return positions


def _clip_positions(
    x: np.ndarray,
    config: GroupedConfig,
    *,
    clip_z_q: float = 10.0,
    clip_z_beta: float = 15.0,
    clip_u: float = 10.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    x[:, 0] = np.clip(x[:, 0], -float(clip_z_q), float(clip_z_q))
    x[:, 1 : 1 + config.S] = np.clip(x[:, 1 : 1 + config.S], -float(clip_z_beta), float(clip_z_beta))
    x[:, 1 + config.S :] = np.clip(x[:, 1 + config.S :], -float(clip_u), float(clip_u))
    return x


def run_pso(
    *,
    config: GroupedConfig,
    receiver: str,
    num_particles: int,
    num_iters: int,
    num_mc_opt: int,
    num_mc_eval: int,
    seed: int | None = None,
    seed_opt: int | None = None,
    seed_eval: int | None = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    objective_metric: str = "mc_mean_T_over_G",
    vmax: float | None = None,
    warm_policies: Sequence[GroupedPolicy] = (),
    warm_start_noise: float = 0.15,
    warm_ratio: float = 0.4,
    clip_z_q: float = 10.0,
    clip_z_beta: float = 15.0,
    clip_u: float = 10.0,
    pmax_mode: str = "reject",
) -> PSOResult:
    if seed_opt is None:
        if seed is None:
            seed_opt = 20260520
        else:
            seed_opt = int(seed)
    if seed_eval is None:
        seed_eval = int(seed_opt) + 1000003
    rng = np.random.default_rng(int(seed_opt))
    dim = 2 * config.S + 1
    x = _initial_positions(
        config,
        rng,
        int(num_particles),
        warm_policies,
        float(warm_start_noise),
        warm_ratio=float(warm_ratio),
    )
    x = _clip_positions(x, config, clip_z_q=clip_z_q, clip_z_beta=clip_z_beta, clip_u=clip_u)
    v = rng.normal(0.0, 0.1, size=(int(num_particles), dim))
    pbest_x = x.copy()
    pbest_score = np.full(int(num_particles), -np.inf, dtype=float)
    pbest_eval: list[MCEvalResult | None] = [None] * int(num_particles)
    gbest_x = x[0].copy()
    gbest_score = -np.inf
    gbest_eval: MCEvalResult | None = None
    trace: list[dict[str, Any]] = []

    for it in range(int(num_iters)):
        scores = []
        for i in range(int(num_particles)):
            try:
                policy = decode_particle(x[i], config, name=f"pso_iter{it}_particle{i}")
                if pmax_mode == "reject" and config.Pmax is not None and np.max(policy.P) > config.Pmax:
                    score = -np.inf
                    result = None
                else:
                    result = mc_evaluate_policy(
                        config,
                        policy,
                        receiver,
                        num_mc=int(num_mc_opt),
                        seed=int(seed_opt),
                        keep_samples=False,
                        method="PSO_MC",
                    )
                    score = _score(result, objective_metric)
            except Exception:
                result = None
                score = -np.inf
            scores.append(score)
            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_x[i] = x[i].copy()
                pbest_eval[i] = result
            if score > gbest_score:
                gbest_score = score
                gbest_x = x[i].copy()
                gbest_eval = result

        if gbest_eval is None:
            raise RuntimeError("PSO failed to evaluate any valid particle.")
        best_policy = decode_particle(gbest_x, config, name="pso_best_current")
        finite_scores = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
        num_valid = int(finite_scores.size)
        num_invalid = int(num_particles) - num_valid
        iter_best_score = float(np.max(finite_scores)) if num_valid else float("nan")
        trace.append(
            {
                "iter": int(it),
                "best_score": float(gbest_score),
                "iter_best_score": iter_best_score,
                "global_best_score": float(gbest_score),
                "mean_score": float(np.mean(finite_scores)) if num_valid else float("nan"),
                "std_score": float(np.std(finite_scores)) if num_valid else float("nan"),
                "valid_particle_ratio": float(num_valid / int(num_particles)),
                "num_valid_particles": num_valid,
                "num_invalid_particles": num_invalid,
                "best_T": float(gbest_eval.mc_mean_T),
                "best_T_over_G": float(gbest_eval.mc_mean_T_over_G),
                "best_T_over_Gtx": float(gbest_eval.mc_mean_T_over_Gtx),
                "best_pi0": float(best_policy.pi0),
                "best_q_tx": float(best_policy.q_tx),
                "best_avg_power": float(best_policy.avg_power),
                "best_P": best_policy.P.tolist(),
                "best_pi_full": best_policy.pi_full.tolist(),
            }
        )

        r1 = rng.uniform(size=x.shape)
        r2 = rng.uniform(size=x.shape)
        v = float(w) * v + float(c1) * r1 * (pbest_x - x) + float(c2) * r2 * (gbest_x - x)
        if vmax is not None:
            v = np.clip(v, -float(vmax), float(vmax))
        x = x + v
        x = _clip_positions(x, config, clip_z_q=clip_z_q, clip_z_beta=clip_z_beta, clip_u=clip_u)

    best_policy = decode_particle(gbest_x, config, name="pso_best_policy")
    opt_eval = mc_evaluate_policy(config, best_policy, receiver, num_mc=int(num_mc_opt), seed=int(seed_opt), method="PSO_MC")
    final_eval = mc_evaluate_policy(
        config,
        best_policy,
        receiver,
        num_mc=int(num_mc_eval),
        seed=int(seed_eval),
        keep_samples=True,
        method="PSO_MC",
    )
    return PSOResult(best_policy=best_policy, best_score=float(gbest_score), opt_eval=opt_eval, final_eval=final_eval, trace=trace)
