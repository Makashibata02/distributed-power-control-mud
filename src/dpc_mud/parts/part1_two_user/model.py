"""Core two-user random-access model for Part I experiments."""

from __future__ import annotations

import numpy as np


EPS = 1e-10


def gamma_from_rate(rate: float) -> float:
    """Return the SINR threshold gamma = 2^R - 1."""
    if rate <= 0:
        raise ValueError("rate must be positive.")
    return 2.0**rate - 1.0


def energy_levels(rate: float, sigma2: float) -> tuple[float, float, float]:
    """Return the paper's critical power coordinates E1, E2, and E12."""
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    gamma = gamma_from_rate(rate)
    e1 = gamma * sigma2
    e12 = gamma * (e1 + sigma2)
    e2 = 0.5 * (e1 + e12)
    return e1, e2, e12


def success_count(receiver: str, e1: float, e2: float, rate: float, sigma2: float) -> int:
    """
    Count successfully decoded users in one slot.

    The function treats P=0 as no transmission. If exactly one user transmits,
    that user is decoded only when its single-user SNR supports the target rate.
    """
    gamma = gamma_from_rate(rate)
    active1 = e1 > 0.0
    active2 = e2 > 0.0

    if not active1 and not active2:
        return 0

    if active1 and not active2:
        return int(e1 / sigma2 >= gamma)

    if active2 and not active1:
        return int(e2 / sigma2 >= gamma)

    if receiver == "OD":
        e_min, _, e_corner = energy_levels(rate, sigma2)
        e_sum = e_min + e_corner
        both_ok = (
            e1 >= e_min - EPS
            and e2 >= e_min - EPS
            and e1 + e2 >= e_sum - EPS
        )
        if both_ok:
            return 2

        one_ok = (e1 >= gamma * (sigma2 + e2) - EPS) or (e2 >= gamma * (sigma2 + e1) - EPS)
        return 1 if one_ok else 0

    if receiver == "SUD":
        ok1 = e1 >= gamma * (sigma2 + e2) - EPS
        ok2 = e2 >= gamma * (sigma2 + e1) - EPS
        return int(ok1) + int(ok2)

    if receiver == "SIC":
        if e1 >= e2:
            first_ok = e1 >= gamma * (sigma2 + e2) - EPS
            if not first_ok:
                return 0
            second_ok = e2 >= gamma * sigma2 - EPS
            return 2 if second_ok else 1

        first_ok = e2 >= gamma * (sigma2 + e1) - EPS
        if not first_ok:
            return 0
        second_ok = e1 >= gamma * sigma2 - EPS
        return 2 if second_ok else 1

    raise ValueError(f"Unsupported receiver: {receiver}")


def success_count_grid(receiver: str, e1g: np.ndarray, e2g: np.ndarray, rate: float, sigma2: float) -> np.ndarray:
    """Vectorized success-count map for feasible-region plots."""
    gamma = gamma_from_rate(rate)

    if receiver == "OD":
        e_min, _, e_corner = energy_levels(rate, sigma2)
        e_sum = e_min + e_corner
        active1 = e1g > 0.0
        active2 = e2g > 0.0
        two_user = (
            active1
            & active2
            & (e1g >= e_min - EPS)
            & (e2g >= e_min - EPS)
            & (e1g + e2g >= e_sum - EPS)
        )
        one_user = (
            ((active1 & (~active2) & (e1g >= gamma * sigma2 - EPS)))
            | ((active2 & (~active1) & (e2g >= gamma * sigma2 - EPS)))
            | ((active1 & active2 & (~two_user)) & ((e1g >= gamma * (sigma2 + e2g) - EPS) | (e2g >= gamma * (sigma2 + e1g) - EPS)))
        )
        return two_user.astype(int) * 2 + one_user.astype(int)

    if receiver == "SUD":
        s1 = e1g >= gamma * (sigma2 + e2g) - EPS
        s2 = e2g >= gamma * (sigma2 + e1g) - EPS
        return s1.astype(int) + s2.astype(int)

    if receiver == "SIC":
        first_if_1 = e1g >= gamma * (sigma2 + e2g) - EPS
        second_after_1 = e2g >= gamma * sigma2 - EPS
        count_if_1 = first_if_1.astype(int) * (1 + second_after_1.astype(int))

        first_if_2 = e2g >= gamma * (sigma2 + e1g) - EPS
        second_after_2 = e1g >= gamma * sigma2 - EPS
        count_if_2 = first_if_2.astype(int) * (1 + second_after_2.astype(int))
        return np.where(e1g >= e2g, count_if_1, count_if_2)

    raise ValueError(f"Unsupported receiver: {receiver}")


def throughput_exact(receiver: str, probs: np.ndarray, powers: np.ndarray, rate: float, sigma2: float) -> float:
    """Compute exact expected rate throughput by enumerating two users' choices."""
    probs = np.asarray(probs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    if probs.ndim != 1 or powers.ndim != 1:
        raise ValueError("probs and powers must be one-dimensional.")
    if probs.size != powers.size:
        raise ValueError("probs and powers must have the same length.")
    if np.any(probs < -1e-12):
        raise ValueError("probabilities must be non-negative.")
    if not np.isclose(np.sum(probs), 1.0, atol=1e-9):
        raise ValueError(f"probabilities must sum to 1, got {np.sum(probs)}.")

    expected_success = 0.0
    for i, p1 in enumerate(powers):
        for j, p2 in enumerate(powers):
            expected_success += probs[i] * probs[j] * success_count(receiver, float(p1), float(p2), rate, sigma2)

    return rate * expected_success


def average_power(probs: np.ndarray, powers: np.ndarray) -> float:
    """Return single-user average transmit power."""
    probs = np.asarray(probs, dtype=float)
    powers = np.asarray(powers, dtype=float)
    return float(np.dot(probs, powers))
