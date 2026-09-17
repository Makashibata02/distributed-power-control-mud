"""
Initializers for Part II grouped power access baselines.

Policies are generated through the two-level probability parameterization:

    q_tx = sum(pi_1,...,pi_S)
    beta_s = pi_s / q_tx
    pi_full = [1 - q_tx, q_tx * beta_1, ..., q_tx * beta_S]

Power initializers only provide starting points. Local search may move to a
general strictly increasing positive power vector.
"""

from __future__ import annotations

from typing import Any, Optional
import math

import numpy as np

from .model import GroupedConfig, GroupedPolicy, validate_grouped_policy


def normalize_alpha(weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compatibility alias for normalizing a positive-layer distribution."""

    return normalize_beta(weights, eps=eps)


def normalize_beta(weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, eps)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Cannot normalize beta from non-positive weights.")
    return weights / total


def make_pi_full(beta: np.ndarray, q_tx: float, eps: float = 1e-12, beta_eps: float = 1e-12) -> np.ndarray:
    beta = normalize_beta(beta, eps=beta_eps)
    q_tx = float(np.clip(q_tx, eps, 1.0))
    pi_pos = q_tx * beta
    return np.concatenate([[1.0 - q_tx], pi_pos])


def make_alpha_full(beta: np.ndarray, q_tx: float, eps: float = 1e-12, beta_eps: float = 1e-12) -> np.ndarray:
    """Compatibility alias for older alpha0 experiment code."""

    return make_pi_full(beta, q_tx, eps=eps, beta_eps=beta_eps)


def ensure_increasing_power(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 1:
        raise ValueError(f"P must be 1-D, got shape {P.shape}.")
    P_sorted = np.sort(np.maximum(P, eps))
    for i in range(1, P_sorted.size):
        if P_sorted[i] <= P_sorted[i - 1] + eps:
            P_sorted[i] = P_sorted[i - 1] + eps
    return P_sorted


def scale_power_to_pbar(pi_pos: np.ndarray, P_raw: np.ndarray, Pbar: float) -> np.ndarray:
    pi_pos = np.asarray(pi_pos, dtype=float)
    P_inc = ensure_increasing_power(P_raw)
    avg = float(np.sum(pi_pos * P_inc))
    if avg <= 1e-15:
        raise ValueError("Cannot scale power with near-zero positive-layer average.")
    return P_inc * (float(Pbar) / avg)


def structured_beta(S: int, mode: str, **kwargs: Any) -> np.ndarray:
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}.")
    idx = np.arange(S, dtype=float)

    if mode == "uniform":
        weights = np.ones(S, dtype=float)
    elif mode == "exp_increasing":
        a = float(kwargs.get("a", kwargs.get("alpha_r", 1.5)))
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}.")
        weights = a**idx
    elif mode == "exp_decreasing":
        a = float(kwargs.get("a", kwargs.get("alpha_r", 1.5)))
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}.")
        weights = a ** (S - 1 - idx)
    elif mode == "gaussian":
        mu = float(kwargs.get("mu", (S - 1) / 2.0))
        sigma_g = float(kwargs.get("sigma_g", kwargs.get("sigma", max(S / 4.0, 0.5))))
        if sigma_g <= 0:
            raise ValueError(f"sigma_g must be positive, got {sigma_g}.")
        weights = np.exp(-((idx - mu) ** 2) / (2.0 * sigma_g**2))
    elif mode == "poisson":
        lambda_p = float(kwargs.get("lambda_p", kwargs.get("lambda", max(1.0, S / 2.0))))
        if lambda_p <= 0:
            raise ValueError(f"lambda_p must be positive, got {lambda_p}.")
        weights = np.asarray(
            [math.exp(k * math.log(lambda_p) - lambda_p - math.lgamma(k + 1)) for k in range(S)],
            dtype=float,
        )
    else:
        raise ValueError(
            "beta mode must be one of uniform, exp_increasing, exp_decreasing, gaussian, poisson."
        )
    return normalize_beta(weights)


def structured_alpha(S: int, mode: str, **kwargs: Any) -> np.ndarray:
    """Compatibility alias: returns beta over positive-power layers."""

    return structured_beta(S, mode, **kwargs)


def structured_power(
    S: int,
    mode: str,
    *,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> np.ndarray:
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}.")
    idx = np.arange(S, dtype=float)
    if mode == "exponential":
        r = float(kwargs.get("r", kwargs.get("power_r", 1.5)))
        if r <= 1.0:
            raise ValueError(f"r must be greater than 1, got {r}.")
        P_raw = r**idx
    elif mode == "linear":
        P_raw = idx + 1.0
    elif mode == "random_increments":
        rng = rng or np.random.default_rng()
        P_raw = np.cumsum(rng.exponential(scale=1.0, size=S) + 1e-8)
    elif mode == "log_increments":
        rng = rng or np.random.default_rng()
        scale = float(kwargs.get("scale", 0.6))
        P_raw = np.cumsum(np.exp(rng.normal(0.0, scale, size=S)))
    else:
        raise ValueError("power mode must be exponential, linear, random_increments, or log_increments.")
    return ensure_increasing_power(P_raw)


def _resolve_q_tx(config: GroupedConfig, rng: np.random.Generator, kwargs: dict[str, Any]) -> float:
    if "q_tx" in kwargs:
        q_tx = float(kwargs["q_tx"])
    elif kwargs.get("q_tx_mode") == "random":
        q_tx = float(rng.uniform(float(kwargs.get("q_tx_min", config.q_tx_min)), float(kwargs.get("q_tx_max", config.q_tx_max))))
    elif kwargs.get("q_tx_mode") == "beta":
        q_tx = float(rng.beta(float(kwargs.get("q_beta_a", 2.0)), float(kwargs.get("q_beta_b", 2.0))))
    else:
        q_tx = float(kwargs.get("q_tx_fixed", 1.0))
    return float(np.clip(q_tx, float(kwargs.get("q_tx_min", config.q_tx_min)), float(kwargs.get("q_tx_max", config.q_tx_max))))


def make_initial_policy(
    config: GroupedConfig,
    alpha_mode: Optional[str] = None,
    power_mode: str = "exponential",
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> GroupedPolicy:
    rng = rng or np.random.default_rng()
    beta_mode = str(kwargs.get("beta_mode", alpha_mode or kwargs.get("alpha_mode", "uniform")))
    beta_kwargs = dict(kwargs.get("beta_kwargs", kwargs.get("alpha_kwargs", {})))
    power_kwargs = dict(kwargs.get("power_kwargs", {}))
    for key in ("a", "alpha_r", "mu", "sigma", "sigma_g", "lambda_p", "lambda"):
        if key in kwargs:
            beta_kwargs[key] = kwargs[key]
    for key in ("r", "power_r", "scale"):
        if key in kwargs:
            power_kwargs[key] = kwargs[key]

    q_tx = _resolve_q_tx(config, rng, kwargs)
    beta = structured_beta(config.S, beta_mode, **beta_kwargs)
    pi_full = make_pi_full(beta, q_tx, eps=max(config.q_tx_min, 1e-12))
    P_raw = structured_power(config.S, power_mode, rng=rng, **power_kwargs)
    P = scale_power_to_pbar(pi_full[1:], P_raw, config.Pbar)
    policy = GroupedPolicy(
        name=f"init_{beta_mode}_{power_mode}_q{q_tx:.3g}",
        pi_full=pi_full,
        P=P,
        extra={
            "init_type": "structured",
            "beta_mode": beta_mode,
            "power_mode": power_mode,
            "q_tx": q_tx,
            "beta_kwargs": beta_kwargs,
            "power_kwargs": power_kwargs,
        },
    )
    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    return policy


def random_alpha(S: int, rng: np.random.Generator, mode: str, **kwargs: Any) -> np.ndarray:
    """Compatibility alias: returns beta over positive-power layers."""

    return random_beta(S, rng, mode, **kwargs)


def random_beta(S: int, rng: np.random.Generator, mode: str, **kwargs: Any) -> np.ndarray:
    if mode == "dirichlet":
        beta = float(kwargs.get("theta", kwargs.get("beta", 1.0)))
        if beta <= 0:
            raise ValueError(f"theta must be positive, got {beta}.")
        return rng.dirichlet(np.full(S, beta, dtype=float))
    if mode == "random_uniform":
        raw = rng.uniform(size=S)
    elif mode == "random_gaussian_weights":
        raw = np.abs(rng.normal(float(kwargs.get("loc", 0.0)), float(kwargs.get("scale", 1.0)), size=S)) + 1e-9
    elif mode == "random_exponential_weights":
        raw = rng.exponential(scale=float(kwargs.get("scale", 1.0)), size=S) + 1e-9
    else:
        raise ValueError(
            "random beta mode must be dirichlet, random_uniform, random_gaussian_weights, or random_exponential_weights."
        )
    return normalize_beta(raw)


def random_power_vector(S: int, rng: np.random.Generator, mode: str = "random_increments", **kwargs: Any) -> np.ndarray:
    if mode in {"random_increments", "increments", "exponential_increments"}:
        delta = rng.exponential(scale=float(kwargs.get("scale", 1.0)), size=S) + 1e-8
    elif mode == "log_increments":
        delta = np.exp(rng.normal(0.0, float(kwargs.get("scale", 0.6)), size=S))
    else:
        raise ValueError("random power mode must be random_increments or log_increments.")
    return ensure_increasing_power(np.cumsum(delta))


def make_random_policy(
    config: GroupedConfig,
    alpha_mode: Optional[str] = None,
    power_mode: str = "random_increments",
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> GroupedPolicy:
    rng = rng or np.random.default_rng()
    beta_mode = str(kwargs.get("beta_mode", alpha_mode or kwargs.get("alpha_mode", "dirichlet")))
    beta_kwargs = dict(kwargs.get("beta_kwargs", kwargs.get("alpha_kwargs", {})))
    power_kwargs = dict(kwargs.get("power_kwargs", {}))
    q_kwargs = {
        "q_tx_mode": kwargs.get("q_tx_mode", "random"),
        "q_tx_min": kwargs.get("q_tx_min", config.q_tx_min),
        "q_tx_max": kwargs.get("q_tx_max", config.q_tx_max),
    }
    if "q_tx" in kwargs:
        q_kwargs["q_tx"] = kwargs["q_tx"]

    beta = random_beta(config.S, rng, beta_mode, **beta_kwargs)
    q_tx = _resolve_q_tx(config, rng, q_kwargs)
    pi_full = make_pi_full(beta, q_tx, eps=max(config.q_tx_min, 1e-12))
    P_raw = random_power_vector(config.S, rng, mode=power_mode, **power_kwargs)
    P = scale_power_to_pbar(pi_full[1:], P_raw, config.Pbar)
    policy = GroupedPolicy(
        name=f"random_{beta_mode}_{power_mode}_q{q_tx:.3g}",
        pi_full=pi_full,
        P=P,
        extra={
            "init_type": "random",
            "beta_mode": beta_mode,
            "power_mode": power_mode,
            "q_tx": q_tx,
            "beta_kwargs": beta_kwargs,
            "power_kwargs": power_kwargs,
        },
    )
    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    return policy
