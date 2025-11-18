#!/usr/bin/env python3
"""
Verify (approximate) wide-sense stationarity on a portion of real LIGO data.

We fetch public LIGO strain data (GWOSC) using GWPy, then:
  1) Split into overlapping windows.
  2) Check windowed mean/variance stability.
  3) Compare windowed ACFs to the global ACF.
  4) Compare windowed PSDs (Welch) for stability.

Refs:
- GWPy fetch_open_data docs: https://gwpy.github.io/docs/stable/timeseries/opendata/
- GW150914 GPS reference (≈1126259462.4): https://gwcat.cardiffgravity.org/data.html
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from gwpy.timeseries import TimeSeries
from scipy.signal import welch
import matplotlib.pyplot as plt


# ------------------------ configuration ------------------------

# Choose a 64 s span around the famous GW150914 event, but we'll evaluate WSS in shorter windows.
# You can move this away from the merger time if you want a 'quieter' segment.
EVENT_GPS = 1126259462.4  # GW150914 event time (approx)  [Livingston/Hanford]; public reference
SPAN_SEC = 64              # total data duration to fetch
IFO = "L1"                 # 'L1' (Livingston) or 'H1' (Hanford)

# Windowing for WSS checks
WIN_SEC = 4                # window length in seconds
OVERLAP = 0.5              # 50% overlap
MAX_LAG_S = 0.25           # compare ACF up to 0.25 s lag
PSD_FFTLEN = 2.0           # Welch segment length (s)
PSD_OVERLAP = 0.5          # Welch overlap fraction

# Optional: mask out ±0.2 s around the event time to reduce obvious nonstationarity
MASK_EVENT = True
MASK_RADIUS_S = 0.2

SEED = 123  # for any randomization if added later (not used now)

# ------------------------ helpers ------------------------

@dataclass
class WSSReport:
    fs: float
    n_windows: int
    mean_drift: float
    var_drift: float
    acf_rms_deviation: float
    psd_cv_median: float
    psd_cv_90pct: float


def sliding_windows(x: np.ndarray, fs: float, win_sec: float, overlap: float) -> np.ndarray:
    n = len(x)
    w = int(round(win_sec * fs))
    step = int(round(w * (1 - overlap)))
    if w <= 1 or step < 1:
        raise ValueError("Window or step too small; adjust WIN_SEC/OVERLAP.")
    starts = np.arange(0, n - w + 1, step)
    return np.stack([x[s:s + w] for s in starts], axis=0)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x) / n
    c = np.correlate(x, x, mode="full")
    mid = len(c) // 2
    c = c[mid:mid + max_lag + 1] / n
    return c / (denom + 1e-30)


def windowed_psd(x: np.ndarray, fs: float, seglen_s: float, overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = int(round(seglen_s * fs))
    noverlap = int(round(nperseg * overlap))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, Pxx


def summarize_psd_stability(psds: np.ndarray) -> Tuple[float, float]:
    # psds: shape (n_windows, n_freq)
    # coefficient of variation across windows, per freq
    mu = np.mean(psds, axis=0)
    sd = np.std(psds, axis=0, ddof=1)
    cv = sd / (mu + 1e-30)
    return float(np.median(cv)), float(np.percentile(cv, 90))


def verify_wss(x: np.ndarray, fs: float) -> WSSReport:
    # Build windows
    X = sliding_windows(x, fs, WIN_SEC, OVERLAP)
    nW, wlen = X.shape

    # Mean/variance stability
    means = X.mean(axis=1)
    vars_ = X.var(axis=1, ddof=0)
    mean_drift = float(np.max(np.abs(means - means.mean())) / (np.std(x) + 1e-30))
    var_drift = float(np.max(np.abs(vars_ - vars_.mean())) / (np.var(x) + 1e-30))

    # ACF stability (compare to global ACF)
    max_lag = int(round(MAX_LAG_S * fs))
    acf_global = acf(x, max_lag)
    acf_windows = np.stack([acf(win, max_lag) for win in X], axis=0)
    acf_rms_dev = float(np.sqrt(np.mean((acf_windows - acf_global) ** 2)))

    # PSD stability
    psd_list = []
    for win in X:
        _, Pxx = windowed_psd(win, fs, PSD_FFTLEN, PSD_OVERLAP)
        psd_list.append(Pxx)
    psds = np.stack(psd_list, axis=0)
    psd_cv_med, psd_cv_p90 = summarize_psd_stability(psds)

    return WSSReport(
        fs=fs,
        n_windows=nW,
        mean_drift=mean_drift,
        var_drift=var_drift,
        acf_rms_deviation=acf_rms_dev,
        psd_cv_median=psd_cv_med,
        psd_cv_90pct=psd_cv_p90,
    )


def mask_event_region(ts: np.ndarray, t: np.ndarray, t0: float, radius: float) -> np.ndarray:
    # Replace event-centered samples by linear interpolation across the gap
    m = (np.abs(t - t0) <= radius)
    if not np.any(m):
        return ts
    ts2 = ts.copy()
    idx = np.where(~m)[0]
    if len(idx) < 2:
        return ts  # can't sensibly interpolate
    # simple linear interpolation over masked region
    ts2[m] = np.interp(t[m], t[idx], ts[idx])
    return ts2


# ------------------------ main ------------------------

def main():
    start = EVENT_GPS - SPAN_SEC / 2
    stop = EVENT_GPS + SPAN_SEC / 2

    print(f"Fetching {SPAN_SEC}s of {IFO} strain data from GWOSC: [{start:.1f}, {stop:.1f}] GPS ...")
    # Fetch real data from GWOSC via GWPy
    # (GWPy docs: TimeSeries.fetch_open_data('L1', gps-5, gps+5))
    data = TimeSeries.fetch_open_data(IFO, start, stop)  # unit: dimensionless strain

    # Convert to numpy
    fs = float(data.sample_rate.value)
    t = data.times.value
    h = data.value

    # Optionally mask the merger moment (reduces obvious nonstationarity near the signal)
    if MASK_EVENT:
        h = mask_event_region(h, t, EVENT_GPS, MASK_RADIUS_S)

    # High-pass a little to remove slow drift (optional but common in LIGO handling)
    # Comment this out if you want to test raw stationarity
    hp = 20.0  # Hz
    h_filt = data.highpass(hp).value  # GWPy convenience filter

    # Run WSS verification
    rpt = verify_wss(h_filt, fs)

    print("\n=== WSS check (lower is better) ===")
    print(f"Sample rate: {rpt.fs:.1f} Hz   windows: {rpt.n_windows}")
    print(f"Mean drift (max |μ_i - μ̄| / σ): {rpt.mean_drift:.3g}")
    print(f"Var drift  (max |σ_i^2 - σ̄^2| / Var): {rpt.var_drift:.3g}")
    print(f"ACF RMS deviation (0..1 scale): {rpt.acf_rms_deviation:.3g}")
    print(f"PSD CV across windows — median: {rpt.psd_cv_median:.3g}   90th pct: {rpt.psd_cv_90pct:.3g}")

    # Simple “rule of thumb” flags (you can tune these)
    print("\nHeuristic verdict (tunable thresholds):")
    flags = []
    flags.append(rpt.mean_drift < 0.1)
    flags.append(rpt.var_drift < 0.2)
    flags.append(rpt.acf_rms_deviation < 0.05)
    flags.append(rpt.psd_cv_median < 0.25)
    verdict = "approximately WSS on this interval" if all(flags) else "not convincingly WSS (some metrics large)"
    print(f"→ {verdict}")

    # ---- Plots ----
    fig1 = plt.figure(figsize=(11, 4))
    plt.plot(t - t[0], h_filt, lw=0.7)
    plt.xlabel("Time since segment start [s]")
    plt.ylabel("Strain (high-passed)")
    plt.title(f"{IFO} strain segment (GWOSC), fs={fs:.0f} Hz — visual check")
    plt.tight_layout()

    # ACF: global vs. two windows
    max_lag = int(round(MAX_LAG_S * fs))
    global_acf = acf(h_filt, max_lag)
    X = sliding_windows(h_filt, fs, WIN_SEC, OVERLAP)
    acf_w1 = acf(X[0], max_lag)
    acf_wm = acf(X[len(X)//2], max_lag)
    tau = np.arange(max_lag + 1) / fs

    fig2 = plt.figure(figsize=(11, 4))
    plt.plot(tau, global_acf, label="Global ACF", lw=2)
    plt.plot(tau, acf_w1, label="ACF (first window)", alpha=0.8)
    plt.plot(tau, acf_wm, label="ACF (mid window)", alpha=0.8)
    plt.xlabel("Lag [s]")
    plt.ylabel("Autocorrelation")
    plt.title("ACF stability across windows")
    plt.legend()
    plt.tight_layout()

    # PSDs from a few windows
    fig3 = plt.figure(figsize=(11, 4))
    for idx in [0, len(X)//2, -1]:
        f, Pxx = welch(X[idx], fs=fs, nperseg=int(PSD_FFTLEN*fs), noverlap=int(PSD_OVERLAP*PSD_FFTLEN*fs))
        plt.semilogy(f, Pxx, alpha=0.8, label=f"window {idx}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain^2/Hz]")
    plt.title("Welch PSDs across windows (should be similar if WSS)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()