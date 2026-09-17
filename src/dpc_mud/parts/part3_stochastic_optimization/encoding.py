"""Unconstrained PSO particle encoding for grouped policies."""

from __future__ import annotations

import numpy as np

from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy, validate_grouped_policy


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def logit(y: float, eps: float = 1e-9) -> float:
    y = float(np.clip(y, eps, 1.0 - eps))
    return float(np.log(y / (1.0 - y)))


def decode_particle(x: np.ndarray, config: GroupedConfig, *, name: str = "pso_candidate") -> GroupedPolicy:
    x = np.asarray(x, dtype=float)
    if x.size != 2 * config.S + 1:
        raise ValueError(f"Particle length must be {2 * config.S + 1}, got {x.size}.")
    z_q = x[0]
    z_beta = x[1 : 1 + config.S]
    u = x[1 + config.S :]
    q_unit = float(sigmoid(z_q))
    q_tx = config.q_tx_min + (config.q_tx_max - config.q_tx_min) * q_unit
    beta = softmax(z_beta)
    pi_pos = q_tx * beta
    pi_full = np.concatenate([[1.0 - q_tx], pi_pos])
    increments = np.exp(np.clip(u, -30.0, 30.0))
    P_raw = np.cumsum(increments)
    avg_raw = float(np.sum(pi_pos * P_raw))
    if avg_raw <= 0:
        raise ValueError("Invalid particle produced non-positive weighted power.")
    P = P_raw * (config.Pbar / avg_raw)
    policy = GroupedPolicy(name=name, pi_full=pi_full, P=P, extra={"source": "pso"})
    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    return policy


def policy_to_particle_approx(policy: GroupedPolicy, config: GroupedConfig, eps: float = 1e-9) -> np.ndarray:
    q_unit = (policy.q_tx - config.q_tx_min) / max(config.q_tx_max - config.q_tx_min, eps)
    z_q = logit(q_unit, eps=eps)
    z_beta = np.log(np.maximum(policy.beta, eps))
    P = np.asarray(policy.P, dtype=float)
    increments = np.maximum(P - np.concatenate([[0.0], P[:-1]]), eps)
    u = np.log(increments)
    return np.concatenate([[z_q], z_beta, u])
