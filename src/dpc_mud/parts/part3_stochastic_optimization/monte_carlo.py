"""Poisson random-load Monte Carlo evaluator for Part III.

Only random load fluctuation is simulated. No Rayleigh fading, explicit channel
matrix, or per-user physical-layer simulation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy, validate_grouped_policy


@dataclass(frozen=True)
class MCEvalResult:
    receiver: str
    method: str
    num_mc: int
    seed: int
    mc_mean_T: float
    mc_std_T: float
    mc_p05_T: float
    mc_p50_T: float
    mc_p95_T: float
    mc_mean_T_over_G: float
    mc_mean_T_over_Gtx: float
    mc_mean_G_tx: float
    pi_full: np.ndarray
    P: np.ndarray
    avg_power: float
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pi0(self) -> float:
        return float(self.pi_full[0])

    @property
    def pi_pos(self) -> np.ndarray:
        return np.asarray(self.pi_full, dtype=float)[1:]

    @property
    def q_tx(self) -> float:
        return float(np.sum(self.pi_pos))

    @property
    def beta(self) -> np.ndarray:
        return self.pi_pos / self.q_tx if self.q_tx > 1e-15 else np.full(len(self.P), 1.0 / len(self.P))

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "receiver": self.receiver,
            "method": self.method,
            "num_mc": int(self.num_mc),
            "seed": int(self.seed),
            "mc_mean_T": float(self.mc_mean_T),
            "mc_std_T": float(self.mc_std_T),
            "mc_p05_T": float(self.mc_p05_T),
            "mc_p50_T": float(self.mc_p50_T),
            "mc_p95_T": float(self.mc_p95_T),
            "mc_mean_T_over_G": float(self.mc_mean_T_over_G),
            "mc_mean_T_over_Gtx": float(self.mc_mean_T_over_Gtx),
            "mc_mean_G_tx": float(self.mc_mean_G_tx),
            "pi_full": np.asarray(self.pi_full, dtype=float).tolist(),
            "pi0": self.pi0,
            "pi_pos": self.pi_pos.tolist(),
            "q_tx": self.q_tx,
            "beta": self.beta.tolist(),
            "P": np.asarray(self.P, dtype=float).tolist(),
            "avg_power": float(self.avg_power),
        }


def generate_poisson_load_samples(
    pi_full: np.ndarray,
    G: float,
    num_mc: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lam = float(G) * np.asarray(pi_full, dtype=float)
    return rng.poisson(lam=lam, size=(int(num_mc), lam.size)).astype(float)


def evaluate_given_counts(
    config: GroupedConfig,
    policy: GroupedPolicy,
    receiver: str,
    K_pos_sample: np.ndarray,
) -> Dict[str, Any]:
    K = np.asarray(K_pos_sample, dtype=float)
    P = np.asarray(policy.P, dtype=float)
    if K.shape != (config.S,):
        raise ValueError(f"K_pos_sample must have shape ({config.S},), got {K.shape}.")

    if receiver == "global_od":
        G_tx_sample = float(np.sum(K))
        P_sum = float(np.sum(K * P))
        d = float(min(config.N, G_tx_sample))
        if G_tx_sample <= 0 or d <= 0:
            T = 0.0
            failed_group: Optional[int] = None
        else:
            C_global = float(d * np.log2(1.0 + P_sum / config.sigma2))
            T = G_tx_sample if G_tx_sample * config.R <= C_global + 1e-12 else min(G_tx_sample, C_global / config.R)
            failed_group = None if T >= G_tx_sample - 1e-12 else -1
        return {
            "T_sample": float(T),
            "G_tx_sample": G_tx_sample,
            "failed_group": failed_group,
            "group_success_users": (K * (T / G_tx_sample)).tolist() if G_tx_sample > 0 else np.zeros(config.S).tolist(),
        }

    users = np.zeros(config.S, dtype=float)
    failed_group = None
    for s in reversed(range(config.S)):
        K_s = float(K[s])
        if K_s <= 1e-12:
            continue
        I_s = float(np.sum(K[:s] * P[:s]))
        intra = float(max(K_s - 1.0, 0.0) * P[s])
        if receiver == "group_od":
            d_s = float(min(config.N, K_s))
            lhs = float(K_s * config.R)
            rhs = float(d_s * np.log2(1.0 + K_s * P[s] / (config.sigma2 + I_s)))
            success = bool(lhs <= rhs + 1e-12)
        elif receiver == "group_mf":
            sinr_s = float(config.N * P[s] / (config.sigma2 + intra + I_s))
            success = bool(sinr_s >= config.gamma - 1e-12)
        elif receiver == "group_lmmse":
            K_remain = float(np.sum(K[: s + 1]))
            eta_s = float(1.0 / (1.0 + config.N / max(K_remain, 1e-12)))
            sinr_s = float(config.N * P[s] / (config.sigma2 + eta_s * (intra + I_s)))
            success = bool(sinr_s >= config.gamma - 1e-12)
        else:
            raise ValueError(f"Unsupported receiver: {receiver}")
        users[s] = K_s if success else 0.0
        if not success:
            failed_group = s + 1
            if config.stop_on_failure:
                break

    return {
        "T_sample": float(np.sum(users)),
        "G_tx_sample": float(np.sum(K)),
        "failed_group": failed_group,
        "group_success_users": users.tolist(),
    }


def mc_evaluate_policy(
    config: GroupedConfig,
    policy: GroupedPolicy,
    receiver: str,
    *,
    num_mc: int,
    seed: int,
    keep_samples: bool = False,
    method: str | None = None,
) -> MCEvalResult:
    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    samples = generate_poisson_load_samples(policy.pi_full, config.G, int(num_mc), int(seed))
    T = np.zeros(int(num_mc), dtype=float)
    T_over_G = np.zeros(int(num_mc), dtype=float)
    T_over_Gtx = np.zeros(int(num_mc), dtype=float)
    G_tx = np.zeros(int(num_mc), dtype=float)
    rows: list[dict[str, Any]] = []

    for idx, K_full in enumerate(samples):
        out = evaluate_given_counts(config, policy, receiver, K_full[1:])
        T[idx] = out["T_sample"]
        G_tx[idx] = out["G_tx_sample"]
        T_over_G[idx] = T[idx] / config.G if config.G > 0 else 0.0
        T_over_Gtx[idx] = T[idx] / G_tx[idx] if G_tx[idx] > 0 else 0.0
        if keep_samples:
            rows.append(
                {
                    "sample_id": int(idx),
                    "T_sample": float(T[idx]),
                    "T_over_G_sample": float(T_over_G[idx]),
                    "T_over_Gtx_sample": float(T_over_Gtx[idx]),
                    "G_tx_sample": float(G_tx[idx]),
                    "K0_sample": float(K_full[0]),
                    "failed_group": "" if out["failed_group"] is None else int(out["failed_group"]),
                }
            )

    return MCEvalResult(
        receiver=receiver,
        method=method or policy.name,
        num_mc=int(num_mc),
        seed=int(seed),
        mc_mean_T=float(np.mean(T)),
        mc_std_T=float(np.std(T, ddof=1)) if int(num_mc) > 1 else 0.0,
        mc_p05_T=float(np.percentile(T, 5)),
        mc_p50_T=float(np.percentile(T, 50)),
        mc_p95_T=float(np.percentile(T, 95)),
        mc_mean_T_over_G=float(np.mean(T_over_G)),
        mc_mean_T_over_Gtx=float(np.mean(T_over_Gtx)),
        mc_mean_G_tx=float(np.mean(G_tx)),
        pi_full=policy.pi_full.copy(),
        P=policy.P.copy(),
        avg_power=policy.avg_power,
        samples=rows,
    )
