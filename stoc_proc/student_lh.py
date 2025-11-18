#!/usr/bin/env python3
"""
Demo: Gaussian vs Student-t likelihood on real LIGO data with a CBC injection.

- Downloads 32 s of L1 strain around GW150914 from GWOSC via GWPy.
- High-pass (20 Hz), optional resample to 2048 Hz.
- Injects a CBC (IMRPhenomD, non-spinning) at the center of the segment using PyCBC.
- Splits into overlapping 8 s segments; for each, computes:
    * rFFT (positive freqs),
    * Welch PSD (for that segment),
    * log-likelihood under Gaussian (ν=∞) and Student-t (ν in NU_LIST).
- Compares likelihoods for two models:
    (A) h = 0        (noise-only model)
    (B) h = template (correct signal model)

Author: you + ChatGPT
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from math import isinf

from gwpy.timeseries import TimeSeries
from scipy.signal import welch
from scipy.signal.windows import tukey
from scipy.special import gammaln, erfinv

# --- PyCBC for waveform generation ---
try:
    from pycbc.waveform import get_td_waveform
except Exception as e:
    raise SystemExit(
        "PyCBC is required for this demo.\n"
        "Install with: pip install pycbc\n"
        f"Import error: {e}"
    )

# ----------------------- configuration -----------------------

# Data span near GW150914 (can be anywhere; we pick this because it's public/demo-friendly)
EVENT_GPS = 1126259462.4
SPAN_SEC   = 32.0
IFO        = "L1"

# Preprocessing
HIGHPASS_HZ = 20.0
RESAMPLE_HZ = 2048.0       # set to None to keep native rate

# Likelihood segmentation
SEG_LEN_S   = 8.0
OVERLAP_FR  = 0.5
WINDOW_TUKEY_ALPHA = 0.1

# Student-t shapes to test (ν → ∞ ≡ Gaussian)
NU_LIST = [6.0, 10.0, 30.0, np.inf]

# Injection controls (IMRPhenomD, non-spinning)
DO_INJECTION   = True
M1_Msun        = 30.0
M2_Msun        = 30.0
DIST_Mpc       = 500.0         # increase for quieter injection, decrease to make it louder
F_LOWER        = 30.0          # Hz
APPROXIMANT    = "IMRPhenomD"
INJ_INCLINATION = 0.0          # face-on
INJ_PHI        = 0.0           # phase at coalescence
# Place the coalescence at the center of the fetched span (in GPS)
# You can shift by +/- a second if you want it away from segment boundaries.
INJ_GPS = EVENT_GPS

# ----------------------- likelihood core -----------------------

def _student_t_logpdf(x: np.ndarray, s: np.ndarray, nu: float) -> np.ndarray:
    """Log-pdf of Student-t with df=nu and per-sample scale s (real-valued)."""
    
    logC = gammaln(0.5*(nu+1)) - gammaln(0.5*nu) - 0.5*np.log(nu*np.pi) - np.log(s)
    z2 = (x / s)**2

    return logC - 0.5*(nu+1)*np.log1p(z2/nu)

def gaussian_loglik_segment(d: np.ndarray, h: np.ndarray, Sn: np.ndarray, df: float) -> float:
    """Standard Gaussian log-likelihood (up to a constant) for one segment."""
    r = d - h
    var = 0.5 * Sn * df  # per real component
    ll = -0.5 * ((r.real**2)/var + np.log(2*np.pi*var)) \
         -0.5 * ((r.imag**2)/var + np.log(2*np.pi*var))
    return float(np.sum(ll))

def student_t_loglik_segment(d: np.ndarray, h: np.ndarray, Sn: np.ndarray, df: float, nu: float) -> float:
    """Student-t log-likelihood for one segment (heavy-tailed)."""
    r = d - h
    s = np.sqrt(0.5 * Sn * df)
    ll = _student_t_logpdf(r.real, s, nu) + _student_t_logpdf(r.imag, s, nu)
    return float(np.sum(ll))

def robust_loglik(segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
                  nu: Optional[float]) -> float:
    """
    Sum log-likelihood over segments.
    Each item: (d_k, h_k, S_nk, df) with 1-sided PSD (positive freqs).
    """
    total = 0.0
    if nu is None or isinf(nu):
        for d, h, Sn, df in segments:
            total += gaussian_loglik_segment(d, h, Sn, df)
    else:
        for d, h, Sn, df in segments:
            total += student_t_loglik_segment(d, h, Sn, df, nu)
    return total

# ----------------------- utilities -----------------------

def segment_indices(n: int, seglen: int, step: int):
    starts = np.arange(0, n - seglen + 1, step, dtype=int)
    return [(s, s + seglen) for s in starts]

def segment_fft_and_psd(x: np.ndarray, fs: float, seg_len_s: float, overlap_fr: float,
                        win_alpha: float):
    """
    Split x into overlapping segments. For each, compute:
        - rFFT (positive frequencies, complex),
        - Welch PSD on same segment,
        - df and frequency grids.
    Returns: list of (D_k, f_psd, S_n(f_psd), df, f_fft, window) for reuse on template.
    """
    seglen = int(round(seg_len_s * fs))
    step = int(round(seglen * (1 - overlap_fr)))
    if seglen <= 1 or step < 1:
        raise ValueError("Bad segmentation parameters.")
    win = tukey(seglen, alpha=win_alpha)
    out = []
    for a, b in segment_indices(len(x), seglen, step):
        seg = x[a:b]
        if len(seg) != seglen:
            continue
        segw = seg * win
        D = np.fft.rfft(segw)
        df = fs / seglen
        f_fft = np.fft.rfftfreq(seglen, d=1.0/fs)
        nper = max(int(2.0 * fs), 8)   # ~2 s Welch chunks
        nover = int(0.5 * nper)
        f_psd, Pxx = welch(segw, fs=fs, nperseg=nper, noverlap=nover, detrend="constant")
        Pxx = np.maximum(Pxx, np.finfo(float).tiny)
        out.append((D, f_psd, Pxx, df, f_fft, win, slice(a, b)))
    return out

def interpolate_psd_to_fft_grid(f_psd: np.ndarray, Pxx: np.ndarray, f_fft: np.ndarray) -> np.ndarray:
    """Interpolate one-sided PSD onto the FFT frequency grid."""
    ff = np.clip(f_fft, f_psd[0], f_psd[-1])
    return np.interp(ff, f_psd, Pxx)

def make_td_injection(fs: float, n: int, start_gps: float, inj_gps: float,
                      m1: float, m2: float, dist_mpc: float,
                      f_lower: float, incl: float, phi0: float,
                      approximant: str) -> np.ndarray:
    """
    Generate a time-domain plus-polarization injection at sample rate fs,
    length n, with coalescence time at inj_gps. Aligned face-on by default.
    """
    dt = 1.0 / fs
    # PyCBC returns hp,hc with hp.end_time=0 at coalescence by default.
    hp, hc = get_td_waveform(approximant=approximant,
                             mass1=m1, mass2=m2, spin1z=0, spin2z=0,
                             f_lower=f_lower, delta_t=dt, distance=dist_mpc,
                             inclination=incl, coa_phase=phi0)
    h = hp.numpy()  # plus polarization
    # Place the last sample (coalescence) at inj_index
    inj_index = int(round((inj_gps - start_gps) * fs))
    L = len(h)
    x_model = np.zeros(n, dtype=float)
    a = inj_index - (L - 1)
    b = inj_index + 1
    # Trim to fit into [0, n)
    aa = max(a, 0); bb = min(b, n)
    ha = aa - a; hb = ha + (bb - aa)
    if bb > aa and hb > ha:
        x_model[aa:bb] += h[ha:hb]
    return x_model

# ----------------------- main demo -----------------------

def main():
    # ---- fetch real LIGO data (GWOSC) ----
    start = EVENT_GPS - SPAN_SEC/2
    stop  = EVENT_GPS + SPAN_SEC/2
    print(f"Fetching {IFO} strain {SPAN_SEC:.0f}s from {start:.1f} to {stop:.1f} (GPS) ...")
    data = TimeSeries.fetch_open_data(IFO, start, stop)   # dimensionless strain
    data = data.highpass(HIGHPASS_HZ)
    if RESAMPLE_HZ is not None:
        data = data.resample(RESAMPLE_HZ)

    t = data.times.value            # GPS times
    x = data.value.astype(float)
    fs = float(data.sample_rate.value)
    n  = len(x)

    # ---- inject CBC signal (time-domain) ----
    if DO_INJECTION:
        print("Generating and injecting CBC (IMRPhenomD) via PyCBC ...")
        template_td = make_td_injection(fs, n, t[0], INJ_GPS,
                                        M1_Msun, M2_Msun, DIST_Mpc,
                                        F_LOWER, INJ_INCLINATION, INJ_PHI,
                                        APPROXIMANT)
        x_inj = x + template_td
    else:
        template_td = np.zeros_like(x)
        x_inj = x

    # ---- segment data and build FFT+PSD packs ----
    packs_data = segment_fft_and_psd(x_inj, fs, SEG_LEN_S, OVERLAP_FR, WINDOW_TUKEY_ALPHA)

    # Build model FFTs on the same segments (window the *template* identically)
    segments_noise_model = []  # h = 0
    segments_signal_model = [] # h = template (correct model)
    for (D, f_psd, Pxx, df, f_fft, win, slc) in packs_data:
        # PSD on FFT grid
        Sn = interpolate_psd_to_fft_grid(f_psd, Pxx, f_fft)
        # Model h(f): FFT of windowed template segment on same slice
        H_seg = np.fft.rfft(template_td[slc] * win)
        # Drop very low frequencies (below highpass)
        keep = f_fft >= 20.0
        segments_noise_model.append((D[keep], np.zeros_like(D[keep]), Sn[keep], df))
        segments_signal_model.append((D[keep], H_seg[keep],               Sn[keep], df))

    # ---- evaluate log-likelihoods ----
    print("\nLog-likelihoods (sum over segments):")
    for nu in NU_LIST:
        tag = "Gaussian (ν=∞)" if (nu is None or np.isinf(nu)) else f"Student-t (ν={nu:g})"
        ll_noise  = robust_loglik(segments_noise_model, nu)
        ll_signal = robust_loglik(segments_signal_model, nu)
        print(f"  {tag:18s}  h=0: {ll_noise: .3f}   h=template: {ll_signal: .3f}   Δ: {ll_signal-ll_noise: .3f}")

    # ---- quick plots ----
    # 1) Time series with injection
    plt.figure(figsize=(11, 3.6))
    plt.plot(t - t[0], x_inj, lw=0.6, label="data + injection" if DO_INJECTION else "data")
    if DO_INJECTION:
        plt.plot(t - t[0], template_td, lw=0.6, alpha=0.8, label="injected template")
    plt.xlabel("Time since segment start [s]")
    plt.ylabel("Strain (hp)")
    ttl = f"{IFO} strain — {SPAN_SEC:.0f}s, fs={fs:.0f} Hz"
    if DO_INJECTION:
        ttl += f"  (inj: {M1_Msun:.0f}+{M2_Msun:.0f} Msun @ {DIST_Mpc:.0f} Mpc)"
    plt.title(ttl)
    plt.legend()
    plt.tight_layout()

    # 2) PSDs from a few windows
    plt.figure(figsize=(11, 3.6))
    for i, (D, f_psd, Pxx, df, f_fft, _, _) in enumerate(packs_data[:3]):
        plt.semilogy(f_psd, Pxx, alpha=0.85, label=f"Welch PSD (seg {i})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain^2/Hz]")
    plt.title("Welch PSDs across segments")
    plt.legend()
    plt.tight_layout()

    # 3) QQ plot of whitened residuals (noise-only model) on a middle segment
    D_mid, f_psd, Pxx, df_mid, f_fft, _, slc_mid = packs_data[len(packs_data)//2]
    keep = f_fft >= 20.0
    Sn_mid = interpolate_psd_to_fft_grid(f_psd, Pxx, f_fft)[keep]
    s_mid  = np.sqrt(0.5 * Sn_mid * df_mid)
    R = (D_mid - 0)[keep]  # residual under noise-only model
    z = np.concatenate([(R.real / s_mid), (R.imag / s_mid)])
    z = z[np.isfinite(z)]
    z.sort()
    p = (np.arange(len(z)) + 0.5)/len(z)
    qn = np.sqrt(2) * erfinv(2*p - 1)

    plt.figure(figsize=(4.8, 4.8))
    plt.plot(qn, z, ".", ms=2)
    lim = np.percentile(np.abs(z), 99.5)
    plt.plot([-lim, lim], [-lim, lim], lw=1)
    plt.xlabel("Normal quantiles")
    plt.ylabel("Whitened residuals")
    plt.title("QQ plot (heavier tails ⇒ Student-t helps)")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
