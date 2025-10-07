#%%
"""
Robust two-channel regression with possible per-row swaps via MM/S-estimation.

This module provides:
1) `MM_shuffled`: Alternates between robust regression fits for two
   response channels (Y1, Y2) and an exact, per-row "swap-or-not" decision
   to minimize total squared residual loss. It repeats several outer
   iterations and returns the best run (lowest loss).

2) Robust regression backends:
   - `fastsreg`: A fast S-estimator following Salibian-Barrera & Yohai (2005)
     using subsampling + local refinement and Tukey's biweight ρ.
   - `mmregres`: An MM-regression step (S-scale initialization +
     IRWLS with Tukey's biweight ψ).

The robust machinery uses Tukey’s biweight family with tuning constant `c`
selected either as a default (4.685, ~95% efficiency under Gaussian errors)
or via `Tbsc(bdp, p)` to target a breakdown point `bdp` for S-estimation.

Notation (robust functions):
- ρ_c(u): Tukey's biweight ρ at tuning c
- ψ_c(u): derivative of ρ (the ψ-function)
- w(u)   : ψ(u)/u (with the convention w(0)=ψ'(0))
- S-scale s solves   E[ ρ( (Y - Xβ)/s ) ] = κ, fixed κ determined by c and p.

References
----------
Salibian-Barrera, M., & Yohai, V. J. (2006).
A fast algorithm for S-regression estimates. Journal of Computational and
Graphical Statistics, 15(2), 414–427.

Implementation notes
--------------------
- Many inner loops are Numba-accelerated for speed (`@njit`).
- Normal equations are solved via pseudoinverse for robustness; you may
  switch to a stabilized solver if desired.
- Inputs are coerced to float64 arrays.

"""

from __future__ import annotations

import math
from itertools import permutations  # (currently unused; left for parity with original)
from typing import Optional, Tuple, List

import numpy as np
import scipy as sp
from numba import njit


def MM_shuffled(
    y1_in: np.ndarray,
    y2_in: np.ndarray,
    X_in: np.ndarray,
    bdp: float,
    outeriter: int = 5,
    project: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Robust two-channel regression with per-row swap decisions.

    Given two response vectors `Y1` and `Y2` measured on the same rows (samples)
    and a common design matrix `X`, this routine alternates between:
      (i) Robust fits β₁, β₂ for each channel using S/MM regression, and
     (ii) A *rowwise exact* decision q_i ∈ {0,1} choosing whether row i should
          keep (Y1_i, Y2_i) as-is (q_i=1) or be swapped to (Y2_i, Y1_i) (q_i=0),
          minimizing r11_i² + r22_i² vs r12_i² + r21_i².

    Across `outeriter` repetitions, it stores the total loss and returns the
    best (lowest) loss solution.

    Parameters
    ----------
    y1_in, y2_in : array-like, shape (N,)
        Two response channels.
    X_in : array-like, shape (N, p)
        Design matrix (include an intercept column if needed).
    bdp : float in (0, 0.5]
        Target breakdown point for the S-initialization in `fastsreg`.
        Typical choices: 0.5 (high robustness), 0.25 (more efficiency).
    outeriter : int, default=5
        Number of outer alternations (fit → recompute q → update Ŷ).
        The function returns the iteration with minimal loss.
    project : bool, default=True
        Kept for API parity; not used in the current implementation.

    Returns
    -------
    bm1 : ndarray, shape (p,)
        Robust regression coefficients for channel 1 (best iteration).
    bm2 : ndarray, shape (p,)
        Robust regression coefficients for channel 2 (best iteration).
    Y1_est : ndarray, shape (N,)
        Final reconstructed channel-1 outcomes after applying the best q.
    Y2_est : ndarray, shape (N,)
        Final reconstructed channel-2 outcomes after applying the best q.
    q : ndarray, shape (N,), dtype=uint8
        Swap indicator (1 = keep original order, 0 = swap Y1↔Y2) for the best iteration.
    losses : list[float]
        Total squared loss per outer iteration.

    Notes
    -----
    - The swap step is exact and vectorized: for each i,
         keep: s1_i = (Y1_i - Xβ₁)² + (Y2_i - Xβ₂)²
         swap: s0_i = (Y1_i - Xβ₂)² + (Y2_i - Xβ₁)²
      choose q_i = 1 iff s1_i ≤ s0_i.
    """
    Y1 = np.asarray(y1_in, float)
    Y2 = np.asarray(y2_in, float)
    X = np.asarray(X_in, float)
    N, p = X.shape

    # binary assignment (start with "no swap")
    q = np.ones(N, dtype=np.uint8)
    Y1_est = Y1.copy()
    Y2_est = Y2.copy()

    losses: List[float] = []
    qs: List[np.ndarray] = []
    betas1: List[np.ndarray] = []
    betas2: List[np.ndarray] = []

    for _ in range(outeriter):
        # robust starts (S) + MM
        bs1, ss1 = fastsreg(X, Y1_est, bdp, 10)
        bs2, ss2 = fastsreg(X, Y2_est, bdp, 10)
        bm1, _ = mmregres(X, Y1_est, bs1, ss1, 0)
        bm2, _ = mmregres(X, Y2_est, bs2, ss2, 0)

        # exact per-row decision for q (vectorized)
        r11 = Y1 - X @ bm1
        r22 = Y2 - X @ bm2
        r12 = Y1 - X @ bm2
        r21 = Y2 - X @ bm1
        s1 = r11 * r11 + r22 * r22  # keep (no swap)
        s0 = r12 * r12 + r21 * r21  # swap
        q = (s1 <= s0)

        # update reconstructed channels (no copies needed)
        Y1_est = np.where(q, Y1, Y2)
        Y2_est = np.where(q, Y2, Y1)

        betas1.append(bm1)
        betas2.append(bm2)
        qs.append(q.copy())
        losses.append(np.where(q, s1, s0).sum())

    j = int(np.argmin(losses))
    bm1 = betas1[j]
    bm2 = betas2[j]
    q = qs[j]

    # final projected channels
    Y1_est = np.where(q, Y1, Y2)
    Y2_est = np.where(q, Y2, Y1)
    return bm1, bm2, Y1_est, Y2_est, q.astype(np.uint8), losses


def mmregres(
    X: np.ndarray,
    Y: np.ndarray,
    b0: np.ndarray,
    s: float,
    bdp: float = 0,
    w_fun=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One MM-regression refinement step using IRWLS with Tukey's biweight.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (include intercept if needed).
    Y : ndarray, shape (n,)
        Response vector.
    b0 : ndarray, shape (p,)
        Initial coefficients (e.g., S-estimator).
    s : float
        Scale for standardizing residuals in the ψ-weights.
        Typically the S-scale from an S-estimator.
    bdp : float, default=0
        If 0, use default c=4.685. Otherwise compute c via `Tbsc(bdp, 1)`.
        (Here the scale step uses p=1 for c-lookup, consistent with original code.)
    w_fun : callable, optional
        Placeholder for custom weight function; currently ignored.

    Returns
    -------
    b_new : ndarray, shape (p,)
        Updated regression coefficients after IRWLS.
    w : ndarray, shape (n,)
        Final weights w_i = ψ(r_i/s)/|r_i/s| (with small-number protection).

    Notes
    -----
    The IRWLS update solves (Xᵀ W X) b = Xᵀ W y with W=diag(w(r/s)) and
    Tukey's biweight ψ. We use a pseudoinverse for robustness.

    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    b = np.asarray(b0, float).copy()

    if bdp == 0:
        c = 4.685
    else:
        c = Tbsc(bdp, 1)

    tol = 1e-10
    maxit = 100

    b_new = irwls_step(X, Y, b, s, c, tol, maxit)
    # Recompute final weights in Python for consistency
    r = (Y - X @ b_new) / s
    w = psibi(r, c) / np.maximum(np.abs(r), 1e-200)
    return b_new, w


@njit(cache=True, fastmath=True)
def _psibi_numba(x: np.ndarray, c: float) -> np.ndarray:
    """Numba-accelerated Tukey biweight ψ(x; c) with zero outside |x|≥c."""
    out = np.zeros_like(x)
    for i in range(x.size):
        xi = x[i]
        ax = abs(xi)
        if ax < c:
            z = xi * (1.0 - (xi / c) ** 2) ** 2
            out[i] = z
    return out


@njit(cache=True, fastmath=True)
def irwls_step(
    X: np.ndarray,
    Y: np.ndarray,
    b: np.ndarray,
    s: float,
    c: float,
    tol: float,
    maxit: int,
) -> np.ndarray:
    """
    Internal IRWLS loop (Numba) for MM-regression.

    Solves successive normal equations with weights w_i = ψ(r_i/s)/(r_i/s)
    using Tukey's biweight ψ. Uses pseudoinverse for stability.

    Parameters are as in `mmregres`.
    """
    n, p = X.shape
    for _ in range(maxit):
        r = (Y - X.dot(b)) / s

        # avoid 0/0 in w = ψ(r)/r
        for i in range(n):
            if -1e-200 < r[i] < 1e-200:
                r[i] = 1e-200 if r[i] >= 0 else -1e-200

        w = _psibi_numba(r, c) / r

        # Build X^T W X and X^T W y
        XWX = np.zeros((p, p))
        XWy = np.zeros(p)
        for i in range(n):
            wi = w[i]
            xi = X[i]
            XWy += wi * xi * Y[i]
            for a in range(p):
                va = wi * xi[a]
                for b2 in range(p):
                    XWX[a, b2] += va * xi[b2]

        b_new = np.linalg.pinv(XWX) @ XWy
        diff = np.max(np.abs(b_new - b))
        b = b_new
        if diff < tol:
            break
    return b


@njit(cache=True, fastmath=True)
def _median_abs(x: np.ndarray) -> float:
    """Numba median(|x|) using partition (works in nopython mode)."""
    n = x.size
    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = abs(x[i])
    k = n // 2
    part = np.partition(tmp, k)
    if n % 2 == 1:
        return part[k]
    else:
        j = k - 1
        return 0.5 * (np.partition(tmp, j)[j] + part[k])


@njit(cache=True, fastmath=True)
def _rhobi(u: np.ndarray, c: float) -> np.ndarray:
    """Numba Tukey biweight ρ(u; c) with cap c²/6 outside |u|>c."""
    n = u.size
    out = np.empty(n)
    cc2 = c * c
    cc4 = cc2 * cc2
    for i in range(n):
        ui = u[i]
        a = abs(ui)
        if a <= c:
            u2 = ui * ui
            out[i] = 0.5 * u2 * (1.0 - (u2 / cc2) + (u2 * u2) / (3.0 * cc4))
        else:
            out[i] = cc2 / 6.0
    return out


@njit(cache=True, fastmath=True)
def _fw(u: np.ndarray, c: float) -> np.ndarray:
    """
    Numba weight for S-iteration: proportional to ψ(u)/u under Tukey's biweight.

    Returns 0 outside |u|≥c and (1 - (u/c)²)² * (c²/6) inside.
    """
    n = u.size
    out = np.empty(n)
    cc2 = c * c
    for i in range(n):
        z = u[i] / c
        az = abs(z)
        if az < 1.0:
            t = 1.0 - z * z
            out[i] = (t * t) * (cc2 / 6.0)
        else:
            out[i] = 0.0
    return out


@njit(cache=True, fastmath=True)
def _scale1(u: np.ndarray, kp: float, c: float, initialsc: float) -> float:
    """
    Numba S-scale fixed-point iteration.

    Solves for s such that mean(ρ(u/s; c)) = kp, starting from `initialsc`.
    """
    sc = initialsc
    maxit = 200
    eps = 1e-20
    i = 0
    err = 1.0
    while (i < maxit) and (err > eps):
        r = u / sc
        rho = _rhobi(r, c)
        m = 0.0
        n = rho.size
        for j in range(n):
            m += rho[j]
        m /= n
        sc2 = (sc * sc) * m / kp
        if sc2 <= 0.0:
            sc2 = sc * sc
        sc_new = np.sqrt(sc2)
        err = abs(sc_new / sc - 1.0)
        sc = sc_new
        i += 1
    return sc


@njit(cache=True, fastmath=True)
def _ress_numba(
    X: np.ndarray,
    y: np.ndarray,
    beta0: np.ndarray,
    k_steps: int,
    conv: int,
    kp: float,
    c: float,
    initialscale: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Numba IRWLS refinement for S-estimator.

    Parameters
    ----------
    X, y : design and response
    beta0 : initial β
    k_steps : number of refining steps (ignored if conv==1)
    conv : if 1, iterate until tight convergence (≤ 50 steps)
    kp : constant derived from c via Tbsb(c, p)
    c : Tukey biweight tuning constant
    initialscale : starting scale

    Returns
    -------
    res : residuals with final β
    beta : refined coefficients
    sc : final scale
    """
    n, p = X.shape
    beta = beta0.copy()
    res = y - X.dot(beta)

    sc = initialscale
    if sc <= 0.0:
        med = _median_abs(res)
        sc = med / 0.6745

    maxk = 50 if conv == 1 else k_steps

    for _ in range(maxk):
        # scale update
        r = res / sc
        rho = _rhobi(r, c)
        m = 0.0
        for j in range(n):
            m += rho[j]
        m /= n
        sc2 = (sc * sc) * m / kp
        if sc2 <= 0.0:
            sc2 = sc * sc
        sc = np.sqrt(sc2)

        # weights and weighted LS
        r = res / sc
        w = _fw(r, c)
        XtWX = np.zeros((p, p))
        XtWy = np.zeros(p)
        for i in range(n):
            wi = w[i]
            if wi <= 0.0:
                continue
            sw = np.sqrt(wi)
            for a in range(p):
                xa = sw * X[i, a]
                XtWy[a] += xa * (sw * y[i])
                for b in range(p):
                    XtWX[a, b] += xa * (sw * X[i, b])

        beta_new = np.linalg.pinv(XtWX).dot(XtWy)

        if conv == 1:
            denom = 0.0
            for j in range(p):
                denom += abs(beta[j]) + 1e-31
            delta = 0.0
            for j in range(p):
                t = beta_new[j] - beta[j]
                delta = max(delta, abs(t))
            if delta / denom < 1e-20:
                beta = beta_new
                break

        beta = beta_new
        res = y - X.dot(beta)

    res = y - X.dot(beta)
    return res, beta, sc


def fastsreg(
    x: np.ndarray,
    y: np.ndarray,
    bdp: float,
    N: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Fast S-regression via subsampling + local refinement (Salibian-Barrera & Yohai).

    Parameters
    ----------
    x : ndarray, shape (n, p)
        Design matrix (include intercept if needed).
    y : ndarray, shape (n,)
        Response vector.
    bdp : float in (0, 0.5]
        Target breakdown point determining the Tukey tuning `c` through `Tbsc`.
    N : int, optional
        Number of random p-subset draws (subsamples). Default: 20.

    Returns
    -------
    beta : ndarray, shape (p,)
        Robust S-estimator coefficients.
    scale : float
        Associated S-scale.

    Notes
    -----
    - For each random p-subset with full rank, solve LS, then optionally refine
      with k=2 IRWLS steps (`ress`) to get a candidate. Keep `bestr=5` best
      candidates according to a scale screening and then fully refine them
      with convergence (`conv=1`). Return the best one.
    - The constant `kp = (c/6)*Tbsb(c, p)` enters the S-scale equation
      mean(ρ(r/s)) = kp.

    """
    if N is None:
        N = 20
    k = 2
    bestr = 5
    n, p = np.shape(x)
    c = Tbsc(bdp, 1)
    kp = (c / 6) * Tbsb(c, 1)

    bestbetas = np.zeros((bestr, p))
    bestscales = 1e20 * np.ones(bestr)
    sworst = 1e20

    for i in range(N):
        # draw a full-rank p-subset
        ok = False
        tries = 0
        while not ok and tries < 100:
            idx = np.random.choice(n, p, replace=False)
            xs = x[idx, :]
            if np.linalg.matrix_rank(xs) == p:
                ok = True
            tries += 1
        if not ok:
            continue

        ys = y[idx]
        beta = oursolve(xs, ys)

        if k > 0:
            # refining (k steps, no convergence)
            res, betarw, scalerw = ress(x, y, beta, k, 0, kp, c)
            resrw = res
        else:
            betarw = beta
            resrw = y - x @ betarw
            scalerw = np.median(np.absolute(resrw)) / 0.6745

        if i > 1:
            # screen by loss at sworst
            scaletest = lossS(resrw, sworst, c)
            if scaletest < kp:
                sbest = scale1(resrw, kp, c, scalerw)
                yi = np.argsort(bestscales)
                ind = yi[bestr - 1]
                bestscales[ind] = sbest
                bestbetas[ind, :] = betarw.T
                sworst = max(bestscales)
        else:
            bestscales[bestr - 1] = scale1(resrw, kp, c, scalerw)
            bestbetas[bestr - 1, :] = betarw.T

    superbestscale = 1e20
    superbestbeta = bestbetas[0, :].copy()

    for i in range(bestr - 1, 1, -1):
        _, betarw, scalerw = ress(x, y, bestbetas[i, :].T, 0, 1, kp, c, bestscales[i])
        if scalerw < superbestscale:
            superbestscale = scalerw
            superbestbeta = betarw

    beta = superbestbeta
    scale = superbestscale
    return beta, scale


# --------------------------------------------------------------------------
# Indirect calls and utilities: robust ψ/ρ, derivatives, scale equations
# --------------------------------------------------------------------------

def dpsibi(x: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Derivative ψ'(x; c) of Tukey's biweight ψ.

    Parameters
    ----------
    x : array-like
    c : float, default=4.685

    Returns
    -------
    ndarray
        ψ'(x; c), zero outside |x|≥c.
    """
    z = (abs(x) < c) * (1 - x ** 2 * (6 / c ** 2 - 5 * x ** 2 / c ** 4))
    return z


def d2psibi(x: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Second derivative ψ''(x; c) of Tukey's biweight ψ.

    Parameters
    ----------
    x : array-like
    c : float, default=4.685

    Returns
    -------
    ndarray
        ψ''(x; c), zero outside |x|≥c.
    """
    z = (abs(x) < c) * (20 * x ** 3 - 12 * c ** 2 * x) / c ** 4
    return z


def fw(u: np.ndarray, c: float) -> np.ndarray:
    """
    Weight function for S-iteration proportional to ψ(u)/u (Tukey biweight).

    Inside |u|<c:  (1 - (u/c)²)² * (c²/6);  outside: 0.

    Parameters
    ----------
    u : array-like
        Standardized residuals.
    c : float
        Tuning constant.

    Returns
    -------
    ndarray
        Weights.
    """
    tmp = (1 - (u / c) ** 2) ** 2
    tmp = tmp * (c ** 2 / 6)
    tmp[abs(u / c) > 1] = 0
    return tmp


def gint(k: int, c: float, p: int) -> float:
    """
    ∫₀^c r^k g(r²) dr, where g(·) is the pdf of ||Z||² for Z~N_p(0, I).

    This integral appears in constants for Tukey S-estimation (see Tbsb/Tbsc).

    Parameters
    ----------
    k : int
        Power on r inside the integral.
    c : float
        Upper limit and Tukey tuning constant.
    p : int
        Dimension for the Gaussian radial density.

    Returns
    -------
    float
        Integral value.
    """
    e = (k - p - 1) / 2
    numerator = (2 ** e) * sp.stats.gamma.cdf((c ** 2) / 2, (k + 1) / 2) * math.gamma((k + 1) / 2)
    res = numerator / (np.pi ** (p / 2))
    return res


def lossS(u: np.ndarray, s: float, c: float) -> float:
    """
    Objective for S-scale: mean(ρ(u/s; c)).

    Parameters
    ----------
    u : array-like
        Residuals.
    s : float
        Candidate scale.
    c : float
        Tukey tuning constant.

    Returns
    -------
    float
        Mean ρ at scale s.
    """
    res = np.mean(rhobi(u / s, c))
    return res


def oursolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares via pseudoinverse (shape (p,p)^+ * (p,))."""
    return np.linalg.pinv(a) @ b


def psibi(x: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey biweight ψ(x; c).

    ψ(x) = x (1 - (x/c)²)² for |x| < c; 0 otherwise.

    Parameters
    ----------
    x : array-like
    c : float, default=4.685

    Returns
    -------
    ndarray
        ψ(x; c).
    """
    z = (abs(x) < c) * x * (1 - (x / c) ** 2) ** 2
    return z


def rhobi(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey biweight ρ(u; c) with cap c²/6.

    Parameters
    ----------
    u : array-like
        Standardized residuals.
    c : float, default=4.685
        Tuning constant.

    Returns
    -------
    ndarray
        ρ(u; c).
    """
    w = (np.absolute(u) <= c)
    v = (u ** 2 / 2 * (1 - (u ** 2 / (c ** 2)) + (u ** 4) / (3 * c ** 4))) * w + (1 - w) * (c ** 2 / 6)
    return v


def ress(
    x: np.ndarray,
    y: np.ndarray,
    initialbeta: np.ndarray,
    k: int,
    conv: int,
    kp: float,
    c: float,
    initialscale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Refinement loop for S-estimation (Numba fast path, NumPy fallback).

    Parameters
    ----------
    x, y : arrays
        Design and response.
    initialbeta : ndarray, shape (p,)
        Starting β.
    k : int
        Number of IRWLS steps if `conv != 1`.
    conv : int (0 or 1)
        If 1, iterate to tight convergence (≤ 50 steps) and ignore `k`.
    kp : float
        Constant depending on c (see `Tbsb`); used in the S-scale equation.
    c : float
        Tukey tuning constant.
    initialscale : float, optional
        Starting scale. If None, use MAD/0.6745 from initial residuals.

    Returns
    -------
    res : ndarray
        Final residuals y - Xβ.
    beta1 : ndarray, shape (p,)
        Final coefficients.
    scale : float
        Final scale.

    Notes
    -----
    Uses a Numba-accelerated path if `initialscale is None` (to compute a robust
    start), otherwise falls back to the original NumPy implementation to
    replicate prior behavior exactly.

    """
    n, p = x.shape
    if initialscale is None:
        res0 = y - x @ initialbeta
        initialscale = np.median(np.abs(res0)) / 0.6745

        res, beta1, scale = _ress_numba(
            x,
            y,
            initialbeta.astype(np.float64),
            int(k),
            int(conv),
            float(kp),
            float(c),
            float(initialscale),
        )
        return res, beta1, scale

    # ---------- fallback pure NumPy (original) ----------
    res = y - x @ initialbeta
    scale = initialscale
    if conv == 1:
        k = 50
    beta = initialbeta.copy()
    for _ in range(k):
        r = res / scale
        scale = np.sqrt(scale ** 2 * np.mean(rhobi(r, c)) / kp)
        weights = fw(r, c)
        sw = np.sqrt(weights)
        xw = x * sw[:, None]
        yw = y * sw
        beta1 = np.linalg.pinv(xw.T @ xw) @ (xw.T @ yw)
        if np.isnan(beta1).any():
            beta1 = initialbeta
            scale = initialscale
            break
        if conv == 1:
            if np.linalg.norm(beta - beta1) / np.linalg.norm(beta + 1e-31) < 1e-20:
                beta = beta1
                break
        beta = beta1
        res = y - x @ beta
    res = y - x @ beta
    return res, beta, scale


def scale1(u: np.ndarray, kp: float, c: float, initialsc: Optional[float] = None) -> float:
    """
    S-scale fixed-point iteration (Numba fast path if initialsc is None).

    Parameters
    ----------
    u : ndarray
        Residuals.
    kp : float
        Constant derived from c (see `Tbsb`).
    c : float
        Tukey tuning constant.
    initialsc : float, optional
        Starting scale. If None, use MAD/0.6745.

    Returns
    -------
    float
        Final scale s satisfying mean(ρ(u/s; c)) ≈ kp.
    """
    if initialsc is None:
        initialsc = np.median(np.abs(u)) / 0.6745
        return _scale1(u.astype(np.float64), float(kp), float(c), float(initialsc))
    # fallback
    sc = initialsc
    for _ in range(200):
        sc2 = sc ** 2 * np.mean(rhobi(u / sc, c)) / kp
        sc_new = np.sqrt(sc2) if sc2 > 0 else sc
        if abs(sc_new / sc - 1) <= 1e-20:
            sc = sc_new
            break
        sc = sc_new
    return sc


def Tbsb(c: float, p: int) -> float:
    """
    Auxiliary constant for Tukey S-estimation: involves Gaussian radial integrals.

    Parameters
    ----------
    c : float
        Tukey tuning constant.
    p : int
        Dimension (used by underlying integrals).

    Returns
    -------
    float
        Tbsb(c, p) value entering kp = (c/6) * Tbsb(c, p).

    Notes
    -----
    Uses `gint` and chi-square tail probability.
    """
    y1 = gint(p + 1, c, p) / 2 - gint(p + 3, c, p) / (2 * c ** 2) + gint(p + 5, c, p) / (6 * c ** 4)
    y2 = (6 / c) * 2 * (np.pi ** (p / 2)) / math.gamma(p / 2)
    y3 = c * (1 - sp.stats.chi2.cdf(c ** 2, p))
    res = (y1 * y2 + y3)
    return res


def Tbsc(alpha: float, p: int) -> float:
    """
    Compute Tukey biweight tuning constant c from a target breakdown α.

    Parameters
    ----------
    alpha : float in (0, 0.5]
        Target breakdown point for S-estimator.
    p : int
        Dimension (enters via chi-square quantiles/integrals).

    Returns
    -------
    float
        Tuning constant c s.t. the induced S-estimator attains breakdown α.

    Notes
    -----
    Fixed-point iteration:
        c_{t+1} = Tbsb(c_t, p) / α,
    initialized at sqrt(χ²_{1-α, p}). Stops when |c_{t+1}-c_t| ≤ 1e-8 or max iters.

    """
    talpha = np.sqrt(sp.stats.chi2.ppf(1 - alpha, p))
    maxit = 1000
    eps = 1e-8
    diff = 1e6
    ctest = talpha
    it = 0
    while (diff > eps) and it < maxit:
        cold = ctest
        ctest = Tbsb(cold, p) / alpha
        diff = abs(cold - ctest)
        it += 1
    return ctest