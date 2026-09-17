"""Low-dimensional policy searches for Part I."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .model import success_count
from .policies import Ch2Policy, fixed_power_policy, four_point_policy, interpolated_power_policy


@dataclass(frozen=True)
class SearchResult:
    receiver: str
    policy: Ch2Policy
    throughput: float


def grid_search_fixed_power_policy(
    *,
    receiver: str,
    q_arrival: float,
    pbar: float,
    rate: float,
    sigma2: float,
    step: float = 0.02,
) -> SearchResult:
    """Evaluate the non-optimized single-power baseline P=Pbar."""
    q_arrival = float(np.clip(q_arrival, 0.0, 1.0))
    policy = fixed_power_policy(pbar, q_arrival, pi0=0.0)
    powers = policy.powers
    probs = policy.probs

    success_matrix = np.zeros((2, 2), dtype=float)
    for i, p1 in enumerate(powers):
        for j, p2 in enumerate(powers):
            success_matrix[i, j] = success_count(receiver, float(p1), float(p2), rate, sigma2)

    throughput = rate * float(probs @ success_matrix @ probs)
    return SearchResult(receiver, policy, throughput)


def _quantized_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("step must lie in (0, 1].")
    count = int(round(1.0 / step))
    return np.arange(count + 1, dtype=float) / count


@lru_cache(maxsize=256)
def _probability_grid(num_nonzero: int, step_key: int) -> tuple[tuple[float, ...], ...]:
    """Return all non-zero probability vectors with sum <= 1."""
    if num_nonzero <= 0:
        raise ValueError("num_nonzero must be positive.")

    unit_count = step_key
    rows: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, depth: int) -> None:
        if depth == num_nonzero:
            rows.append(tuple(x / unit_count for x in prefix))
            return
        for val in range(remaining + 1):
            prefix.append(val)
            rec(prefix, remaining - val, depth + 1)
            prefix.pop()

    rec([], unit_count, 0)
    return tuple(rows)


def probability_grid(num_nonzero: int, step: float) -> np.ndarray:
    """Generate candidate non-zero probabilities on a simplex grid."""
    step_key = int(round(1.0 / step))
    return np.array(_probability_grid(num_nonzero, step_key), dtype=float)


def grid_search_policy(
    *,
    receiver: str,
    powers_nonzero: np.ndarray,
    q_arrival: float,
    pbar: float,
    rate: float,
    sigma2: float,
    step: float,
    name: str,
) -> SearchResult:
    """
    Search over probabilities for a fixed non-zero power support.

    q_arrival is the external packet-arrival probability. The searched pi is
    conditional on a user already having a packet:

        cond_pi = [pi0, pi1, ..., piM].

    The effective zero-power probability is 1-q_arrival + q_arrival*pi0.
    """
    powers_nonzero = np.asarray(powers_nonzero, dtype=float)
    q_arrival = float(np.clip(q_arrival, 0.0, 1.0))

    if q_arrival <= 0.0:
        cond_pi = np.zeros(powers_nonzero.size + 1, dtype=float)
        cond_pi[0] = 1.0
        policy = interpolated_power_policy(q_arrival, cond_pi, powers_nonzero, name=name)
        return SearchResult(receiver, policy, 0.0)

    candidates = probability_grid(powers_nonzero.size, step)

    avg_power = candidates @ powers_nonzero
    valid = avg_power <= pbar + 1e-12

    if np.any(valid):
        valid_nonzero = candidates[valid]
        cond_pi = np.column_stack((1.0 - np.sum(valid_nonzero, axis=1), valid_nonzero))
        powers = np.concatenate(([0.0], powers_nonzero))

        effective_probs = q_arrival * cond_pi
        effective_probs[:, 0] += 1.0 - q_arrival

        success_matrix = np.zeros((powers.size, powers.size), dtype=float)
        for i, p1 in enumerate(powers):
            for j, p2 in enumerate(powers):
                success_matrix[i, j] = success_count(receiver, float(p1), float(p2), rate, sigma2)

        scores = rate * np.einsum("ni,ij,nj->n", effective_probs, success_matrix, effective_probs)
        best_idx = int(np.argmax(scores))
        best_cond_pi = cond_pi[best_idx]
        best_policy = interpolated_power_policy(q_arrival, best_cond_pi, powers_nonzero, name=name)
        best_throughput = float(scores[best_idx])
    else:
        best_policy = None
        best_throughput = -1.0

    if best_policy is None:
        cond_pi = np.zeros(powers_nonzero.size + 1, dtype=float)
        cond_pi[0] = 1.0
        best_policy = interpolated_power_policy(q_arrival, cond_pi, powers_nonzero, name=name)
        best_throughput = 0.0

    return SearchResult(receiver, best_policy, best_throughput)


def grid_search_four_point_policy(
    *,
    receiver: str,
    q_arrival: float,
    pbar: float,
    rate: float,
    sigma2: float,
    step: float = 0.02,
) -> SearchResult:
    """Strict low-dimensional grid search on {0, E1, E2, E12}."""
    template = four_point_policy(rate, sigma2, q_arrival, np.array([1.0, 0.0, 0.0, 0.0]))
    return grid_search_policy(
        receiver=receiver,
        powers_nonzero=template.powers[1:],
        q_arrival=q_arrival,
        pbar=pbar,
        rate=rate,
        sigma2=sigma2,
        step=step,
        name="four_point_grid",
    )
