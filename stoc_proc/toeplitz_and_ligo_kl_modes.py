#!/usr/bin/env python3
"""
Toeplitz covariance eigenvectors → sinusoids
and empirical KL modes of LIGO noise, with Fourier spectra
before and after whitening, and direct comparison of KL modes.

Now with caching of GWOSC data to avoid repeated downloads / hangs.

Part 1: Synthetic Toeplitz covariance C_ij = rho^{|i-j|}
        -> eigenvectors look like discrete cosine modes.

Part 2: Fetch a chunk of LIGO strain data via GWPy (GWOSC) ONCE,
        cache it locally, then:
        - build an empirical covariance from many short windows,
        - compute its KL modes (eigenvectors),
        - compare the leading modes to cosines,
        - analyze their Fourier spectra (dominant frequencies),
        - do the same after whitening, and
        - overlay KL modes + their spectra before/after whitening.

Requirements:
    pip install gwpy numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gwpy.timeseries import TimeSeries


# ---------------------------------------------------------------------
# Utility: cached fetch of LIGO data
# ---------------------------------------------------------------------

def cached_fetch_open_data(ifo, start, stop, highpass_hz, resample_hz,
                           cache_dir="gw_cache", force_refresh=False):
    """
    Fetch LIGO data via GWPy, but cache the result locally as NPZ so we
    don't have to hit GWOSC every time.

    Returns
    -------
    t : ndarray
        Time stamps (seconds, GPS-based relative to epoch).
    x : ndarray
        Strain time series (already high-passed and resampled).
    fs : float
        Sample rate [Hz].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = (
        f"{ifo}_gps{start:.1f}-{stop:.1f}_"
        f"hp{highpass_hz:.1f}_"
        f"rs{resample_hz if resample_hz is not None else 'native'}.npz"
    )
    cpath = cache_dir / key

    if cpath.exists() and not force_refresh:
        print(f"Loading cached data from {cpath} ...")
        npz = np.load(cpath)
        t = npz["t"]
        x = npz["x"]
        fs = float(npz["fs"])
        return t, x, fs

    print(f"Fetching {ifo} strain from {start:.1f} to {stop:.1f} GPS ...")
    data = TimeSeries.fetch_open_data(ifo, start, stop)
    data = data.highpass(highpass_hz)
    if resample_hz is not None:
        data = data.resample(resample_hz)

    t = data.times.value
    x = data.value.astype(float)
    fs = float(data.sample_rate.value)

    np.savez_compressed(cpath, t=t, x=x, fs=fs)
    print(f"Saved cached data to {cpath}")

    return t, x, fs


# ---------------------------------------------------------------------
# Part 1: Toeplitz covariance → sinusoidal eigenvectors
# ---------------------------------------------------------------------

def build_toeplitz_cov(N=64, rho=0.9):
    """
    Build a Toeplitz covariance matrix C with entries C_ij = rho^{|i-j|}.
    This is the covariance of a zero-mean stationary AR(1)-like process.
    """
    idx = np.arange(N)
    C = rho ** np.abs(idx[:, None] - idx[None, :])
    return C


def cosine_modes(N, k_list):
    """
    Build cosine-like modes for comparison.

    Parameters
    ----------
    N : int
        Length of the vectors.
    k_list : list of int
        Mode indices (1,2,3,...) controlling a rough "frequency".

    Returns
    -------
    modes : ndarray, shape (N, len(k_list))
    """
    n = np.arange(N)
    modes = []
    for k in k_list:
        # Roughly DCT-style frequencies: (k - 0.5) * pi / N
        omega = np.pi * (k - 0.5) / N
        v = np.cos(omega * (n - (N - 1) / 2.0))
        modes.append(v)
    modes = np.stack(modes, axis=1)
    # Normalize to unit norm
    modes /= np.linalg.norm(modes, axis=0, keepdims=True)
    return modes


def demo_toeplitz_eig(N=64, rho=0.9, num_show=4):
    """
    Demonstrate that the eigenvectors of a Toeplitz covariance
    look like discrete cosines.
    """
    C = build_toeplitz_cov(N=N, rho=rho)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]  # sort descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k_list = list(range(1, num_show + 1))
    modes = cosine_modes(N, k_list)

    # Plot eigenvectors vs cosine modes
    n = np.arange(N)
    fig, axes = plt.subplots(num_show, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        f"Toeplitz covariance (rho={rho}) eigenvectors vs cosine modes",
        y=0.95
    )

    for i in range(num_show):
        ax = axes[i]
        v = eigvecs[:, i].copy()
        # Flip sign for better match (eigenvectors are defined up to sign)
        if np.dot(v, modes[:, i]) < 0:
            v = -v

        ax.plot(n, v, label=f"Eigenvector {i} (λ={eigvals[i]:.3f})")
        ax.plot(n, modes[:, i], "--", label=f"Cosine mode {i+1}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Index n")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # Plot eigenvalue spectrum
    plt.figure(figsize=(6, 3.5))
    plt.stem(np.arange(N), eigvals)
    plt.xlabel("Mode index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of Toeplitz Covariance Matrix")
    plt.tight_layout()


# ---------------------------------------------------------------------
# Part 2: Empirical KL modes of LIGO noise
# ---------------------------------------------------------------------

def build_cov_from_windows(x, window_length, step):
    """
    Build an empirical covariance matrix from many overlapping windows
    of a 1D time series x.

    Parameters
    ----------
    x : 1D ndarray
        Time series.
    window_length : int
        Number of samples per window.
    step : int
        Step between window starts (e.g. window_length//2 for 50% overlap).

    Returns
    -------
    C_emp : (window_length, window_length) ndarray
        Empirical covariance matrix (sample covariance).
    """
    windows = []
    for start in range(0, len(x) - window_length + 1, step):
        w = x[start:start + window_length]
        w = w - w.mean()
        windows.append(w)

    X = np.stack(windows, axis=1)  # shape: (window_length, n_windows)
    n_win = X.shape[1]
    C_emp = (X @ X.T) / n_win
    return C_emp


def kl_modes_and_spectra(x, fs, window_len_s, overlap, num_show, label_prefix):
    """
    Core analysis for KL modes:
      - build empirical covariance from windows,
      - eigen-decompose,
      - compare KL modes to cosines,
      - compute FFT of each KL mode,
      - print dominant frequencies,
      - plot spectra.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues in descending order.
    eigvecs : ndarray
        Eigenvectors in columns, matching eigvals.
    """
    window_length = int(round(window_len_s * fs))
    step = int(round(window_length * (1.0 - overlap)))

    print(f"\n[{label_prefix}] Building empirical covariance:")
    print(f"  window_length = {window_length} samples ({window_len_s:.3f} s)")
    print(f"  step          = {step} samples (overlap = {overlap:.2f})")

    C_emp = build_cov_from_windows(x, window_length=window_length, step=step)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C_emp)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Cosine modes
    k_list = list(range(1, num_show + 1))
    modes = cosine_modes(window_length, k_list)

    # Plot KL modes vs cosines
    n = np.arange(window_length)
    fig, axes = plt.subplots(num_show, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        f"{label_prefix}: KL modes vs cosine modes\n"
        f"(window_len={window_len_s:.3f}s, fs={fs:.0f}Hz)",
        y=0.95
    )

    for i in range(num_show):
        ax = axes[i]
        v = eigvecs[:, i].copy()
        # Align sign with cosine mode
        if np.dot(v, modes[:, i]) < 0:
            v = -v

        ax.plot(n, v, label=f"KL mode {i} (λ={eigvals[i]:.3e})")
        ax.plot(n, modes[:, i], "--", label=f"Cosine mode {i+1}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Sample index in window")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # Eigenvalue spectrum
    plt.figure(figsize=(6, 3.5))
    plt.semilogy(np.arange(len(eigvals)), eigvals, marker="o", linestyle="none")
    plt.xlabel("Mode index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(f"{label_prefix}: eigenvalues of empirical covariance")
    plt.tight_layout()

    # Fourier spectra of KL modes & dominant frequency mapping
    freqs = np.fft.rfftfreq(window_length, d=1.0 / fs)

    print(f"\n[{label_prefix}] Dominant frequencies of leading KL modes:")
    dom_freqs = []
    for i in range(num_show):
        v = eigvecs[:, i]
        V = np.fft.rfft(v)
        P = np.abs(V) ** 2
        # ignore DC for non-trivial mode
        if len(P) > 1:
            idx_max = np.argmax(P[1:]) + 1
        else:
            idx_max = 0
        f_dom = freqs[idx_max]
        dom_freqs.append(f_dom)
        print(f"  Mode {i}: dominant frequency ~ {f_dom:.1f} Hz (λ={eigvals[i]:.3e})")

    # Plot normalized spectra for first few KL modes
    plt.figure(figsize=(8, 4.5))
    for i in range(num_show):
        v = eigvecs[:, i]
        V = np.fft.rfft(v)
        P = np.abs(V) ** 2
        Pn = P / P.max() if P.max() > 0 else P
        plt.semilogy(freqs, Pn, label=f"KL mode {i} (~{dom_freqs[i]:.1f} Hz)")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized power")
    plt.title(f"{label_prefix}: Fourier spectra of leading KL modes")
    plt.legend()
    plt.tight_layout()

    return eigvals, eigvecs


def demo_ligo_kl_modes(
    ifo="L1",
    event_gps=1126259462.4,
    span_sec=32.0,
    highpass_hz=20.0,
    resample_hz=2048.0,
    window_len_s=0.5,
    overlap=0.5,
    num_show=4,
):
    """
    Empirically derive KL modes (eigenvectors) of LIGO noise and compare
    them to cosine modes, before and after whitening, and overlay them.

    Uses cached GWOSC data to avoid repeated downloads.

    Steps:
      - Fetch a short segment of strain data via cached_fetch_open_data.
      - KL analysis on this (colored) noise.
      - Reconstruct TimeSeries and whiten.
      - KL analysis on whitened noise.
      - Overlay corresponding KL modes and their spectra before/after whitening.
    """
    # Time interval
    start = event_gps - span_sec / 2
    stop = event_gps + span_sec / 2

    # Fetch with caching
    t, x, fs = cached_fetch_open_data(
        ifo, start, stop,
        highpass_hz=highpass_hz,
        resample_hz=resample_hz,
        cache_dir="gw_cache",
        force_refresh=False,
    )
    print(f"Sample rate: {fs:.1f} Hz, N = {len(x)}, duration ~ {len(x)/fs:.1f} s")

    # Show time series (colored)
    plt.figure(figsize=(10, 3.5))
    plt.plot(t - t[0], x, lw=0.5)
    plt.xlabel("Time since segment start [s]")
    plt.ylabel("Strain (hp)")
    plt.title(f"{ifo} strain segment (colored, high-passed)")
    plt.tight_layout()

    # KL analysis on colored noise
    eigvals_raw, eigvecs_raw = kl_modes_and_spectra(
        x, fs,
        window_len_s=window_len_s,
        overlap=overlap,
        num_show=num_show,
        label_prefix="RAW (colored) noise"
    )

    # Whitening: reconstruct a TimeSeries from cached data
    print("\nWhitening data for KL analysis ...")
    data_ts = TimeSeries(x, epoch=t[0], sample_rate=fs)
    fftlength = 4.0
    overlap_whiten = 2.0
    data_w = data_ts.whiten(fftlength=fftlength, overlap=overlap_whiten)

    t_w = data_w.times.value
    x_w = data_w.value.astype(float)
    fs_w = float(data_w.sample_rate.value)
    print(f"Whitened sample rate: {fs_w:.1f} Hz, N = {len(x_w)}")

    # Show whitened time series
    plt.figure(figsize=(10, 3.5))
    plt.plot(t_w - t_w[0], x_w, lw=0.5)
    plt.xlabel("Time since segment start [s]")
    plt.ylabel("Strain (whitened)")
    plt.title(f"{ifo} strain segment (whitened)")
    plt.tight_layout()

    # KL analysis on whitened noise (note: window_len_s, overlap same; fs_w ~ fs)
    eigvals_w, eigvecs_w = kl_modes_and_spectra(
        x_w, fs_w,
        window_len_s=window_len_s,
        overlap=overlap,
        num_show=num_show,
        label_prefix="WHITENED noise"
    )

    # -----------------------------------------------------------------
    # Direct comparison of KL modes before vs after whitening (time domain)
    # -----------------------------------------------------------------
    window_length = int(round(window_len_s * fs))
    n = np.arange(window_length)

    fig, axes = plt.subplots(num_show, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        "KL modes before vs after whitening (same mode index)",
        y=0.95
    )

    for i in range(num_show):
        ax = axes[i]
        v_raw = eigvecs_raw[:, i].copy()
        v_w   = eigvecs_w[:, i].copy()

        # Align whitened mode to raw mode via sign
        if np.dot(v_raw, v_w) < 0:
            v_w = -v_w

        ax.plot(n, v_raw, label=f"RAW KL mode {i}")
        ax.plot(n, v_w, "--", label=f"WHITENED KL mode {i}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Sample index in window")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # -----------------------------------------------------------------
    # Direct comparison of KL mode spectra before vs after whitening
    # -----------------------------------------------------------------
    freqs = np.fft.rfftfreq(window_length, d=1.0 / fs)

    fig, axes = plt.subplots(num_show, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        "KL mode spectra before vs after whitening (same mode index)",
        y=0.95
    )

    for i in range(num_show):
        ax = axes[i]
        v_raw = eigvecs_raw[:, i]
        v_w   = eigvecs_w[:, i]

        V_raw = np.fft.rfft(v_raw)
        V_w   = np.fft.rfft(v_w)
        P_raw = np.abs(V_raw) ** 2
        P_w   = np.abs(V_w) ** 2

        Pn_raw = P_raw / P_raw.max() if P_raw.max() > 0 else P_raw
        Pn_w   = P_w   / P_w.max()   if P_w.max()   > 0 else P_w

        ax.semilogy(freqs, Pn_raw, label=f"RAW KL mode {i}")
        ax.semilogy(freqs, Pn_w, "--", label=f"WHITENED KL mode {i}")
        ax.set_ylabel("Norm. power")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # --- Part 1: Synthetic Toeplitz example ---
    demo_toeplitz_eig(N=64, rho=0.9, num_show=4)

    # --- Part 2: Empirical KL modes of LIGO noise (raw + whitened) ---
    demo_ligo_kl_modes(
        ifo="L1",
        event_gps=1126259462.4,
        span_sec=32.0,
        highpass_hz=20.0,
        resample_hz=2048.0,
        window_len_s=0.5,   # 0.5 s windows
        overlap=0.5,
        num_show=4
    )

    plt.show()


if __name__ == "__main__":
    main()