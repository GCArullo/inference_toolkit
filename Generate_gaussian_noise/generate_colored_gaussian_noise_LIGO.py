import os
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# Updated Advanced LIGO design curve (ZeroDetHighPower), T1800044-v5
# Ccolumns: frequency [Hz], ASD [strain/sqrt(Hz)].
ALIGO_DESIGN_TXT = "https://dcc.ligo.org/public/0149/T1800044/005/aLIGO_O4_high_asd.txt"

def load_aligo_design_psd(cache_path="aLIGO_O4_high_asd.txt"):
    """
    Load Advanced LIGO design sensitivity as a one-sided PSD S1(f) [strain^2/Hz].

    Returns
    -------
    f : (M,) array
        Frequency grid [Hz] from the file.
    S1 : (M,) array
        One-sided PSD [strain^2/Hz], computed as ASD^2.
    """

    if not os.path.exists(cache_path): urlretrieve(ALIGO_DESIGN_TXT, cache_path)

    data = np.loadtxt(cache_path)
    f    = data[:, 0]
    asd  = data[:, 1]
    S1   = asd**2

    return f, S1


def colored_noise_from_psd(psd_onesided, fs, n, rng=None):

    """
    Generate time-domain real-valued colored Gaussian noise with target one-sided PSD S1(f)
    on the rFFT grid rfftfreq(n, 1/fs).

    Normalisation: x = irfft(X) where x[n] = (1/N) sum_k X[k] e^{i2πkn/N}

    For k>0 (excluding DC/Nyquist), draw complex Gaussian:
      Re,Im ~ N(0, σ^2) iid  =>  E|X[k]|^2 = 2σ^2
    Match one-sided PSD bin power:
      (1/N^2) E|X[k]|^2 = S1(f_k) Δf  =>  σ = (N/2) sqrt(S1(f_k) Δf)

    DC and Nyquist (if present) are purely real with std:
      σ = N sqrt(S1(f_k) Δf)
    """

    psd_onesided = np.asarray(psd_onesided, dtype=float)
    nfreq        = n // 2 + 1

    if psd_onesided.shape[0] != nfreq: raise ValueError(f"psd_onesided must have length {nfreq} (n//2+1).")

    if rng is None: rng = np.random.default_rng()

    df = fs / n
    X  = np.zeros(nfreq, dtype=np.complex128)

    # DC
    sigma_0 = n * np.sqrt(psd_onesided[0] * df)
    X[0]    = rng.normal(0.0, sigma_0)

    # Nyquist (only if n even)
    if n % 2 == 0:
        sigma_nyq = n * np.sqrt(psd_onesided[-1] * df)
        X[-1]     = rng.normal(0.0, sigma_nyq)
        kmax      = nfreq - 1
    else:
        kmax      = nfreq

    # Positive freqs excluding DC(/Nyquist)
    k     = np.arange(1, kmax)
    sigma = (n / 2.0) * np.sqrt(psd_onesided[k] * df)
    X[k]  = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)

    X_TD  = np.fft.irfft(X, n=n)

    return X, X_TD

if __name__ == "__main__":

    ###################
    # User parameters #
    ###################

    # Typical values
    fs       = 4096.0
    duration = 64.0  # seconds
    f_low    = 20.0
    nperseg  = 2048*10
    noverlap = 1024*10

    #######################
    # End user parameters #
    #######################

    n = int(fs * duration)

    # 1) Load aLIGO design curve and build PSD on your FFT grid
    f_curve, S1_curve = load_aligo_design_psd(cache_path="aLIGO_O4_high_asd.txt")
    f_grid            = np.fft.rfftfreq(n, d=1/fs)

    # Interpolate PSD onto rFFT frequency bins (fill outside with large noise)
    S1_grid = np.interp(f_grid, f_curve, S1_curve)#, left=S1_curve[0], right=S1_curve[-1])

    # Optional: impose a low-frequency cutoff (e.g., 10 or 20 Hz) by inflating PSD below f_low
    # S1_grid                 = S1_grid.copy()
    # S1_grid[f_grid < f_low] = S1_grid[f_grid < f_low].max()

    # 2) Generate colored Gaussian time-domain strain noise
    rng = np.random.default_rng(0)
    strain_FD, strain_TD   = colored_noise_from_psd(S1_grid, fs=fs, n=n, rng=rng)

    # 3) Welch PSD estimate for verification
    from scipy.signal import welch

    nperseg  = int(fs/2)*10
    noverlap = nperseg // 2

    # fw, S1_hat  = welch_psd_onesided(strain_TD, fs=fs, nperseg=nperseg, noverlap=noverlap)
    fw, S1_hat = welch(
    strain_TD,
    fs              = fs,
    window          = "hann",
    nperseg         = nperseg,
    noverlap        = noverlap,
    return_onesided = True)

    S1_target_w = np.interp(fw, f_grid, S1_grid)

    # 4) Plots
    t = np.arange(n) / fs

    plt.figure()
    plt.plot(f_grid, strain_FD.real)
    plt.plot(f_grid, strain_FD.imag)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain (FD)")
    plt.title("Simulated aLIGO-design colored Gaussian strain (frequency series)")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, strain_TD)
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.title("Simulated aLIGO-design colored Gaussian strain (time series)")
    plt.tight_layout()

    plt.figure()
    plt.loglog(fw, S1_hat, label="Welch estimate")
    plt.loglog(fw, S1_target_w, label="Target PSD (aLIGO design)")
    plt.loglog(f_curve, S1_curve, label="File aLIGO design curve", alpha=0.5, ls='--')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"One-sided PSD [$\mathrm{strain}^2/\mathrm{Hz}$]")
    plt.title("PSD check: aLIGO design vs Welch estimate")
    plt.legend()
    plt.tight_layout()

    plt.show()