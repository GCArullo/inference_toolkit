import numpy as np
import matplotlib.pyplot as plt

def colored_noise_from_psd(psd_onesided, fs, n, rng=None):
    """
    Generate real-valued colored Gaussian noise with a target one-sided PSD S1(f).

    psd_onesided : (n//2+1,) array
        One-sided PSD [signal^2/Hz] at rfftfreq(n, 1/fs).
    fs : float
        Sampling rate [Hz].
    n : int
        Number of samples.
    rng : np.random.Generator
        RNG (optional).

    Returns
    -------
    x : (n,) array
        Real noise with expected one-sided PSD ~ psd_onesided.
    """
    psd_onesided = np.asarray(psd_onesided, dtype=float)
    nfreq = n // 2 + 1
    if psd_onesided.shape[0] != nfreq:
        raise ValueError(f"psd_onesided must have length {nfreq}.")

    if rng is None:
        rng = np.random.default_rng()

    df = fs / n  # Δf

    # NumPy FFT is unnormalized: x = (1/N) * IFFT(X).
    # For k>0 (excluding DC/Nyquist), rFFT bin k represents ±f_k.
    # Draw complex Gaussian with Re,Im ~ N(0, σ^2) iid so E|X[k]|^2 = 2σ^2.
    # Match bin power for one-sided PSD:
    #   (1/N^2) * E|X[k]|^2 = S1(f_k) Δf  =>  σ = (N/2)*sqrt(S1(f_k)Δf)
    X = np.zeros(nfreq, dtype=np.complex128)

    # DC: purely real, no negative-frequency partner
    X[0] = rng.normal(0.0, n * np.sqrt(psd_onesided[0] * df))

    # Nyquist (if n even): purely real, no partner
    if n % 2 == 0:
        X[-1] = rng.normal(0.0, n * np.sqrt(psd_onesided[-1] * df))
        kmax = nfreq - 1
    else:
        kmax = nfreq

    k = np.arange(1, kmax)
    sigma = (n / 2.0) * np.sqrt(psd_onesided[k] * df)
    X[k] = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)

    return np.fft.irfft(X, n=n)


def welch_psd_onesided(x, fs, nperseg=1024, noverlap=None, window="hann"):
    """
    Simple Welch estimate of one-sided PSD using NumPy only.
    Returns (f, Pxx_onesided) with units [signal^2/Hz].
    """
    x = np.asarray(x)
    n = x.size
    if noverlap is None:
        noverlap = nperseg // 2
    if not (0 <= noverlap < nperseg):
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("nperseg - noverlap must be > 0")

    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "hamming":
        w = np.hamming(nperseg)
    elif window == "rect":
        w = np.ones(nperseg)
    else:
        raise ValueError("window must be 'hann', 'hamming', or 'rect'")

    # Window power normalization (PSD units)
    U = (w**2).sum()

    starts = np.arange(0, n - nperseg + 1, step)
    if starts.size == 0:
        raise ValueError("Signal too short for given nperseg")

    P_accum = None
    for s in starts:
        seg = x[s:s+nperseg]
        segw = seg * w
        X = np.fft.rfft(segw)
        P2 = (np.abs(X)**2) / (fs * U)  # two-sided periodogram on rfft bins
        P_accum = P2 if P_accum is None else (P_accum + P2)

    P_avg = P_accum / starts.size

    # Convert to one-sided PSD by doubling non-DC/non-Nyquist bins
    P1 = P_avg.copy()
    if nperseg % 2 == 0:
        P1[1:-1] *= 2.0
    else:
        P1[1:] *= 2.0

    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, P1


# ------------------- Example: generate + verify + plots -------------------

if __name__ == "__main__":
    fs = 1000.0
    n = 2**15

    # Target one-sided PSD on the rFFT grid of length n
    f = np.fft.rfftfreq(n, d=1/fs)
    psd_target = 1.0 / (1.0 + (f / 50.0)**2)  # example colored shape

    # Generate "FD strain" (frequency-domain realization) implicitly via PSD,
    # and return corresponding time series strain h(t)
    h = colored_noise_from_psd(psd_target, fs=fs, n=n, rng=np.random.default_rng(0))

    # Welch estimate
    fw, psd_hat = welch_psd_onesided(h, fs=fs, nperseg=2048, noverlap=1024)

    # Interpolate target onto Welch grid for overlay
    psd_target_w = np.interp(fw, f, psd_target)

    # ---- Plot 1: Time series h(t) ----
    t = np.arange(n) / fs
    plt.figure()
    plt.plot(t, h)
    plt.xlabel("Time [s]")
    plt.ylabel("Strain (arb. units)")
    plt.title("Generated colored Gaussian noise time series")
    plt.tight_layout()

    # ---- Plot 2: PSD overlay (Welch vs target) ----
    plt.figure()
    plt.loglog(fw, psd_hat, label="Welch estimate")
    plt.loglog(fw, psd_target_w, label="Target PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain^2/Hz]")
    plt.title("One-sided PSD: target vs Welch estimate")
    plt.legend()
    plt.tight_layout()

    plt.show()