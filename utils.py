# utils.py
from __future__ import annotations
"""
Utilities for spectral spike models and exponential-kernel inversion.

Model & conventions
-------------------
We consider a sparse Dirac train
    x(t) = Σ_{i=1}^K a_i δ(t - t_i),   t_i ∈ [0,1),
sampled on an N-point uniform grid. Its discrete Fourier samples are
    X[k] = Σ_i a_i e^{-j 2π k t_i},  k ∈ {-(N//2),...,N//2}.
Some observation models:
- "exp"   : Y[k] = X[k] / (α + j ω_k),      ω_k = 2π k (deconvolution)
- "spike" : Y[k] = H[k] X[k],               H is a low-pass (optional)

This module provides:
- Simulation: `simulate_noisy_signal`
- Cost function for α-search (smallest σ_min of Toeplitz): `F`
- Cadzow denoising (rank-K Toeplitz projection): `Cadzow`
- Prony pipeline with ADMM spectral denoising: `denoise_and_prony`
- Time-domain design for amplitude recovery: `reconstruction_params`
- ADMM solver for structured low-rank Toeplitz approx: `admm_denoise`
- Diagonal operators on Toeplitz matrices: `sum_diags`, `mean_diags`

FFT conventions
---------------
- Use NumPy scaling: ifft is unnormalized forward, 1/N on inverse.
- Spectral arrays marked with "*_shift" are fftshift'ed.
- Time-domain signals are real-valued by construction here.

"""

from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz, svd
from scipy.optimize import lsq_linear


# ---------- helpers (internal) ----------

def _wk(N: int) -> NDArray[np.float64]:
    """
    Angular frequencies for N equally spaced samples.

    Returns ω_k = 2π k with k given by fftshift(fftfreq(N) * N).

    Parameters
    ----------
    N : int
        Number of uniform samples in time domain.

    Returns
    -------
    ndarray of shape (N,), float64
        Angular frequency grid in fftshift convention.
    """
    return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N) * N)


def _fourier_of_diracs(
    ti: NDArray[np.float64],
    ai: NDArray[np.float64],
    wk: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Vectorized Fourier series of a Dirac train at frequencies wk.

    Computes Σ_i a_i e^{-j t_i ω_k} for each ω_k.

    Parameters
    ----------
    ti : (K,) float64
        Spike locations in [0,1).
    ai : (K,) float64
        Spike amplitudes.
    wk : (N,) float64
        Angular frequencies (fftshift grid).

    Returns
    -------
    (N,) complex128
        Spectrum samples.
    """
    phase = np.exp(-1j * ti[:, None] * wk[None, :])
    return (ai[:, None] * phase).sum(axis=0)


def _ifft_time(yk: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Real part of inverse DFT.

    Assumes caller handled fftshift alignment. Returns Re{ifft(yk)}.

    Parameters
    ----------
    yk : (N,) complex128
        Frequency-domain vector (unshifted for ifft).

    Returns
    -------
    (N,) float64
        Real time-domain signal.
    """
    return np.real(np.fft.ifft(yk))


# ---------- simulation ----------

def simulate_noisy_signal(
    N: int,
    spike_number: int,
    SNR_dB: float,
    *,
    mode: str = "exp",
    alpha: float | None = None,
    M: int | None = None,
    spike_times=None,
    amplitudes=None,
    snr_domain: str = "time",
    seed: int | None = None,
):
    """
    Simulate a sparse spectral model with optional exponential kernel and noise.

    Modes
    -----
    - "exp"   : Y[k] = X[k] / (α + j ω_k), requires `alpha`
    - "spike" : Y[k] = H[k] X[k], H[k] = 1{|k| ≤ M} if `M` given else 1

    SNR handling
    ------------
    - snr_domain="time"     : sets σ so that (E y_n^2) / σ^2 = 10^{SNR/10}
    - snr_domain="spectrum" : uses Parseval: mean|Y|^2 ↔ time power (NumPy scaling)

    Parameters
    ----------
    N : int
        Number of time samples.
    spike_number : int
        Number of Diracs (ignored if `spike_times` provided).
    SNR_dB : float
        Target signal-to-noise ratio in dB.
    mode : {"exp", "spike"}, default "exp"
        Observation model.
    alpha : float, optional
        Decay parameter for "exp" mode (required if mode="exp").
    M : int, optional
        Half-band for low-pass H in "spike" mode.
    spike_times : array-like, optional
        Predefined spike locations in [0,1). If None, sampled U[0,1).
    amplitudes : array-like, optional
        Predefined amplitudes. If None, sampled U[0.5,1].
    snr_domain : {"time","spectrum"}, default "time"
        Domain in which SNR_dB is enforced.
    seed : int, optional
        RNG seed.

    Returns
    -------
    yn : (N,) float
        Clean time-domain signal.
    yn_noise : (N,) float
        Noisy time-domain signal (white Gaussian).
    ti : (K,) float
        Spike locations (sorted).
    ai : (K,) float
        Spike amplitudes.
    sigma_noise : float
        Time-domain noise standard deviation.
    Yk_shift : (N,) complex
        fftshift(Y[k]) samples used to generate yn.
    Xk_shift : (N,) complex
        fftshift(X[k]) for the underlying Diracs.

    Raises
    ------
    ValueError
        If mode/parameters are inconsistent.

    Notes
    -----
    - All FFTs use NumPy’s conventions; Yk passed to ifft should be ifftshift'ed.
    - For reproducibility, pass `seed`.
    """
    rng = np.random.default_rng(seed)

    # amplitudes & locations
    ai = np.asarray(amplitudes, float) if amplitudes is not None else rng.uniform(0.5, 1.0, spike_number)
    if spike_times is None:
        ti = np.sort(rng.uniform(0.0, 1.0, spike_number))
    else:
        ti = np.sort(np.asarray(spike_times, float))
        spike_number = ti.size

    # frequency grids
    k  = np.fft.fftshift(np.fft.fftfreq(N) * N)      # integer grid
    wk = 2*np.pi*k                                   # angular frequency

    # Dirac spectrum (fftshift convention)
    Xk_shift = (ai[:, None] * np.exp(-1j * 2*np.pi * ti[:, None] * k[None, :])).sum(axis=0)

    if mode == "exp":
        if alpha is None:
            raise ValueError("alpha is required when mode='exp'.")
        Yk = np.fft.ifftshift(Xk_shift / (alpha + 1j*wk))
    elif mode == "spike":
        if M is None:
            H = np.ones_like(k, float)
        else:
            H = (np.abs(k) <= int(M)).astype(float)
        Yk = np.fft.ifftshift(H * Xk_shift)
    else:
        raise ValueError("mode must be 'exp' or 'spike'.")

    # clean time-domain signal
    yn = np.real(np.fft.ifft(Yk))

    # SNR → time-domain σ
    if snr_domain == "time":
        sig_pow = np.mean(yn**2)
        noise_pow = sig_pow / (10.0**(SNR_dB/10.0))
        sigma_noise = np.sqrt(noise_pow)
    elif snr_domain == "spectrum":
        # Parseval: sum|y|^2 = (1/N) sum|Y|^2
        Y_pow = np.mean(np.abs(Yk)**2)
        sigma_Y = np.sqrt(Y_pow / (10.0**(SNR_dB/10.0)))
        sigma_noise = sigma_Y / np.sqrt(N)
    else:
        raise ValueError("snr_domain must be 'time' or 'spectrum'")

    yn_noise = yn + rng.normal(0.0, sigma_noise, size=N)

    return yn, yn_noise, ti, ai, sigma_noise, np.fft.fftshift(Yk), Xk_shift


# ---------- cost / denoising ----------

def F(
    alpha: float,
    Zk: ArrayLike,
    wk: ArrayLike,
    K: int,
    cadzow: bool = False
) -> float:
    """
    Cost for decay-factor (α) search via Toeplitz minimal singular value.

    Constructs S_k = fftshift(Zk) * (α + j ω_k).
    Optionally applies Cadzow denoising (rank-K Toeplitz projection), then
    returns σ_min(Toeplitz(S_k)).

    Parameters
    ----------
    alpha : float
        Decay factor α in the exponential kernel.
    Zk : array-like, shape (N,)
        Observed spectrum in *unshifted* convention (will be fftshifted inside).
    wk : array-like, shape (N,)
        Angular frequency grid (fftshift convention).
    K : int
        Target rank (number of spikes).
    cadzow : bool, default False
        Whether to apply Cadzow denoising before taking σ_min.

    Returns
    -------
    float
        Smallest singular value of the Toeplitz matrix built from S_k.

    Notes
    -----
    The minimum is expected near the true α when the model/order are correct.
    """
    Sk = np.fft.fftshift(np.asarray(Zk)) * (alpha + 1j * np.asarray(wk))
    if cadzow:
        N = Sk.size
        Sk = Cadzow(Sk, K=K, N=N)
    s = svd(toeplitz(Sk[K:], Sk[np.arange(K, -1, -1)]), compute_uv=False)
    return float(s[-1])


def Cadzow(
    Xk: ArrayLike,
    K: int,
    N: int,
    tol_ratio: float = 1e4,
    max_iter: int = 10
) -> NDArray[np.complex128]:
    """
    Cadzow iterations (rank-K Toeplitz projection) on a 1-D spectrum.

    Iteratively:
      1) Build Toeplitz T(X)
      2) Truncated SVD to rank K
      3) Average along Toeplitz diagonals to map back to a spectrum

    Stop when s_K / s_{K+1} ≥ tol_ratio or `max_iter` reached.

    Parameters
    ----------
    Xk : array-like, shape (N,)
        Spectrum in fftshift convention.
    K : int
        Target rank (spikes).
    N : int
        Length of the desired spectrum.
    tol_ratio : float, default 1e4
        Separation threshold for singular values.
    max_iter : int, default 10
        Maximum iterations.

    Returns
    -------
    (N,) complex128
        Denoised spectrum (fftshift convention).
    """
    X = np.asarray(Xk, dtype=complex).copy()
    ratio = 0.0
    iters = 0

    while ratio < tol_ratio and iters < max_iter:
        iters += 1
        T = toeplitz(X[K:], X[np.arange(K, -1, -1)])
        U, s, Vh = svd(T)
        ratio = s[K - 1] / s[K]

        # Truncate to rank K
        S_K = np.diag(s[:K])
        A = U[:, :K] @ S_K @ Vh[:K, :]

        # Average along Toeplitz diagonals back to spectrum
        for idx, off in enumerate(np.arange(K, K - N, -1)):
            X[idx] = np.mean(np.diagonal(A, offset=off))
    return X


# ---------- Prony-style estimation ----------

def denoise_and_prony(
    signal,
    alpha,
    K,
    L=None,
    rho=0.5,
    iters=50,
    enforce_symmetry=True,
    mode="exp"
):
    """
    Denoise spectrum (ADMM Toeplitz low-rank) and estimate spike locations (Prony).

    Pipeline
    --------
    1) FFT of the time signal → Yk
    2) Build X̃_k:
         - mode="exp"   : X̃_k = fftshift(Yk) * (α + j ω_k)
         - mode="spike" : X̃_k = fftshift(Yk)
    3) ADMM denoising (rank-K Toeplitz)
    4) (Optional) enforce Hermitian symmetry
    5) Prony/annihilating filter from Toeplitz(Xk) → roots on unit circle → t̂

    Parameters
    ----------
    signal : array-like, shape (N,)
        Time-domain samples (real).
    alpha : float
        Decay parameter for "exp" mode.
    K : int
        Number of spikes.
    L : int, optional
        Toeplitz column size. Default: (N+1)//2.
    rho : float, default 0.5
        ADMM penalty parameter.
    iters : int, default 50
        ADMM iterations.
    enforce_symmetry : bool, default True
        Force Xk to satisfy conjugate symmetry (real time signal).
    mode : {"exp","spike"}, default "exp"
        Observation model.

    Returns
    -------
    Xk : (N,) complex
        Denoised (fftshift) spectrum after ADMM.
    spike_times : (K,) float
        Estimated locations t̂ ∈ [0,1), sorted.

    Raises
    ------
    ValueError
        If mode invalid or insufficient points (< 2K+1) for Prony.

    Notes
    -----
    Amplitude estimation is done separately from `reconstruction_params`.
    """
    y = np.asarray(signal, float)
    N = y.size
    if L is None:
        L = (N + 1) // 2

    Yk = np.fft.fft(y)
    if mode == "exp":
        wk = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N)*N)
        Xk_tilde = np.fft.fftshift(Yk) * (alpha + 1j*wk)
    elif mode == "spike":
        Xk_tilde = np.fft.fftshift(Yk)
    else:
        raise ValueError("mode must be 'exp' or 'spike'")

    Xk = admm_denoise(Xk_tilde, K, L, ADMM_RHO=rho, ADMM_MAXIT=iters)
    if enforce_symmetry:
        Xk = 0.5*(Xk + np.flip(np.conj(Xk)))

    # normalize (optional but stabilizes Prony)
    m = np.max(np.abs(Xk))
    if m > 0:
        Xk = Xk / m

    if Xk.size < 2*K + 1:
        raise ValueError(f"Need at least {2*K+1} points for Prony, have {Xk.size}")

    T = toeplitz(Xk[K:], Xk[np.arange(K, -1, -1)])
    _, _, Vh = svd(T)
    h = Vh[-1, :].conj()
    denom = h[0] if np.abs(h[0]) > 1e-14 else h[np.argmax(np.abs(h))]
    h = h / (denom + 1e-14)

    z = np.roots(h[::-1])
    idx = np.argsort(np.abs(np.abs(z) - 1))[:K]
    spike_times = np.sort(np.mod(np.angle(z[idx])/(2*np.pi), 1.0)).real
    return Xk, spike_times


def reconstruction_params(
    Xk,
    t_est,
    N: int,
    *,
    mode: str = "exp",
    alpha: float | None = None
):
    """
    Build the real time-domain design matrix mapping amplitudes → samples.

    Constructs Ẑ ∈ ℝ^{N×K} such that y ≈ Ẑ a for the chosen model:

    - mode="exp"   : y = ifft( (Σ_j a_j e^{-j2π k t_j}) / (α + j ω_k) )
    - mode="spike" : y = ifft( Σ_j a_j e^{-j2π k t_j} )

    Parameters
    ----------
    Xk : (N,) complex
        (Not used numerically here; kept for API symmetry and future extensions.)
    t_est : (K,) float
        Estimated spike locations t̂ ∈ [0,1).
    N : int
        Number of time samples.
    mode : {"exp","spike"}, default "exp"
        Observation model; "exp" requires `alpha`.
    alpha : float, optional
        Decay parameter for "exp" model.

    Returns
    -------
    Z_hat : (N, K) float
        Real-valued design matrix (columns are per-spike time responses).

    Raises
    ------
    ValueError
        If mode invalid or alpha missing for "exp".
    """
    Xk = np.asarray(Xk, complex)  # kept for symmetry; not used below
    t_est = np.asarray(t_est, float).reshape(-1)
    K = t_est.size

    wk = _wk(N)
    uk = np.exp(-1j * 2*np.pi * t_est).reshape((K,))

    # Frequency-domain Vandermonde stacking
    V = np.flipud(np.vander(uk, N//2 + 1).T)[1:, :]
    if N % 2 == 0:
        Z = np.concatenate((np.flipud(V.conj()), np.ones((1, K)), V[:-1, :]), axis=0)
    else:
        Z = np.concatenate((np.flipud(V.conj()), np.ones((1, K)), V), axis=0)

    if mode == "exp":
        if alpha is None:
            raise ValueError("alpha is required for mode='exp'.")
        inv_kernel = 1.0 / (alpha + 1j * wk)
        Z_hat = np.zeros((N, K), float)
        for j in range(K):
            Z_hat[:, j] = np.real(np.fft.ifft(np.fft.ifftshift(inv_kernel * Z[:, j])))
    elif mode == "spike":
        Z_hat = np.zeros((N, K), float)
        for j in range(K):
            Z_hat[:, j] = np.real(np.fft.ifft(np.fft.ifftshift(Z[:, j])))
    else:
        raise ValueError("mode must be 'exp' or 'spike'.")

    return Z_hat


def admm_denoise(
    Sk_in: ArrayLike,
    K: int,
    L: int,
    ADMM_RHO: float,
    ADMM_MAXIT: int
) -> NDArray[np.complex128]:
    """
    ADMM for structured low-rank Toeplitz approximation of a 1-D spectrum.

    Problem (informal)
    ------------------
      Given Sk (fftshift spectrum), seek x with Toeplitz T(x) close to T(Sk),
      but rank(T(x)) ≈ K. We alternate:
        - x-update via diagonalized normal equation (per anti-diagonal)
        - Z-update via rank-K truncated SVD on Toeplitz matrix
        - U dual variable update

    Parameters
    ----------
    Sk_in : (B,) complex
        Input spectrum (fftshift).
    K : int
        Target rank (spikes).
    L : int
        Toeplitz column size (Hankel height); typically (N+1)//2.
    ADMM_RHO : float
        ADMM penalty parameter ρ.
    ADMM_MAXIT : int
        Number of ADMM iterations.

    Returns
    -------
    (B,) complex
        Denoised spectrum (anti-diagonal averages of the rank-K projection).

    Notes
    -----
    - Uses diagonal weights consistent with Toeplitz anti-diagonal multiplicities.
    - This is a practical solver; not a full proximal proof.
    """
    Sk = np.asarray(Sk_in, dtype=complex).copy()
    B = Sk.size

    # Initialize variables
    Z = toeplitz(Sk[L - 1:], np.flipud(Sk[:L]))
    U = np.zeros(Z.shape, dtype=complex)

    rho = ADMM_RHO

    # Diagonal weights (anti-diagonal counts)
    w = np.ones(B, dtype=complex) * L
    w[:L] = np.arange(1, L + 1, dtype=complex)
    w[-L:] = np.arange(L, 0, -1, dtype=complex)

    W = np.ones(B, dtype=complex)
    G = rho * 1.0 / (W + rho * w)
    g = G * w * Sk

    for _ in range(ADMM_MAXIT):
        # e-update (diagonal-domain solve)
        z = sum_diags(U + Z)
        e = g - G * z
        x = Sk - e
        X = toeplitz(x[L - 1:], np.flipud(x[:L]))

        # rank-K projection
        H, sigmas, Vh = svd(X - U)
        S = np.diag(sigmas[:K])
        Z = H[:, :K] @ S @ Vh[:K, :]

        # dual ascent
        U = U + (Z - X)

    sadm = mean_diags(Z)
    return sadm


def sum_diags(X: ArrayLike) -> NDArray[np.complex128]:
    """
    Sum along each Toeplitz anti-diagonal (i.e., matrix diagonals).

    Parameters
    ----------
    X : (n, m) array-like
        Matrix whose (anti-)diagonals are to be summed.

    Returns
    -------
    d : (m+n-1,) complex
        Conjugated sums of diagonals, ordered from offset=-(n-1) to +(m-1).

    Notes
    -----
    The conjugation matches the spectral conventions used in `admm_denoise`.
    """
    X = np.asarray(X)
    n, m = X.shape
    d = np.zeros(m + n - 1, dtype=complex)
    for i, off in enumerate(range(-n + 1, m)):
        d[i] = np.trace(X, offset=off, dtype=complex)
    return np.conj(d)


def mean_diags(X: ArrayLike) -> NDArray[np.complex128]:
    """
    Average along each Toeplitz anti-diagonal with triangular weights.

    Parameters
    ----------
    X : (n, m) array-like
        Input matrix.

    Returns
    -------
    d_mean : (m+n-1,) complex
        Weighted average per diagonal, using counts per diagonal.
    """
    X = np.asarray(X)
    n, m = X.shape
    d = sum_diags(X)
    l = min(m, n)
    w = np.ones(m + n - 1, dtype=complex) * l
    w[:l] = np.arange(1, l + 1, dtype=complex)
    w[-l:] = np.arange(l, 0, -1, dtype=complex)
    return d / w