from __future__ import annotations

import math


def generate_bsr_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    atol: float = 1e-12,
) -> list[float]:
    """
    Generate p-banded BSR coefficients for SGD workload A_{alpha,beta}.

    Uses Theorem "Square-Root of SGD Workload Matrix" from
    "Banded Square Root Matrix Factorization for Differentially Private Model Training" (Kalinin et al., 2024)
      alpha := weight_decay (multiplicative), beta := momentum.

    Practical mapping for "no weight decay":
      - if weight_decay == 0, use alpha = 1.0.
    """
    if bands < 1:
        raise ValueError("bands must be >= 1")

    beta = float(momentum)
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)

    if not (0.0 <= beta < 1.0):
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            "weight_decay must satisfy 0 < weight_decay <= 1 for BSR coefficient generation "
            "(or be exactly 0 to represent no weight decay)"
        )

    # Paper assumes beta < alpha. We also support alpha == beta via the
    # generating-function limit:
    # sqrt(1 / ((1-alpha z)(1-beta z))) with alpha=beta -> 1 / (1-alpha z).
    if beta > alpha + atol:
        raise ValueError("BSR generation requires momentum <= effective weight decay")

    if abs(alpha - beta) <= atol:
        return [alpha**j for j in range(bands)]

    # r_i = |binom(-1/2, i)| with recurrence:
    # r_0 = 1, r_i = ((2i-1)/(2i)) r_{i-1}
    r = [0.0] * bands
    r[0] = 1.0
    for i in range(1, bands):
        r[i] = r[i - 1] * ((2.0 * i - 1.0) / (2.0 * i))

    coeffs = [0.0] * bands
    for j in range(bands):

        s = 0.0
        for i in range(j + 1):
            s += (alpha ** (j - i)) * r[j - i] * r[i] * (beta**i)

        coeffs[j] = s

    # Numerical hygiene for tiny negatives due to floating point.
    coeffs = [0.0 if (c < 0.0 and math.isclose(c, 0.0, abs_tol=atol)) else c for c in coeffs]

    return coeffs
