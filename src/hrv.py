import numpy as np
from typing import Dict, Sequence
from scipy.interpolate import CubicSpline
from scipy.signal import welch, detrend

# Frequency bands (Hz) from the 1996 Task Force HRV standards.
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Minimum NN-series length (s) needed to resolve each band.
# The Task Force recommends ~2 min for LF (its slowest wave lasts 25 s); PTB records
# are ~115 s, so LF is computed from 90 s upwards and should be read as a short-term estimate.
MIN_DURATION_LF = 90.0
MIN_DURATION_HF = 60.0


def rr_intervals_ms(r_peaks: Sequence[int], fs: float) -> np.ndarray:
    """
    Converts R-peak sample indices into R-R intervals in milliseconds.
    """
    return np.diff(np.asarray(r_peaks)) / fs * 1000.0


def clean_rr_intervals(rr_ms: np.ndarray, min_rr: float = 300.0, max_rr: float = 2000.0,
                       max_rel_change: float = 0.2) -> np.ndarray:
    """
    Removes physiologically implausible intervals 
    An interval is rejected if it lies outside [min_rr, max_rr] or deviates from the
    median interval by more than max_rel_change.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    in_range = (rr_ms >= min_rr) & (rr_ms <= max_rr)
    if not in_range.any():
        return np.array([])
    median_rr = np.median(rr_ms[in_range])
    stable = np.abs(rr_ms - median_rr) <= max_rel_change * median_rr
    return rr_ms[in_range & stable]


def time_domain_metrics(nn_ms: np.ndarray) -> Dict[str, float]:
    """
    time-domain HRV metrics from an NN interval series (ms).
    """
    if len(nn_ms) < 3:
        return {k: np.nan for k in ("mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "sd_hr", "cv_rr")}

    successive_diffs = np.diff(nn_ms)
    hr = 60000.0 / nn_ms

    return {
        "mean_rr": np.mean(nn_ms),
        "sdnn": np.std(nn_ms, ddof=1),
        "rmssd": np.sqrt(np.mean(successive_diffs ** 2)),
        "pnn50": np.mean(np.abs(successive_diffs) > 50.0) * 100.0,
        "mean_hr": np.mean(hr),
        "sd_hr": np.std(hr, ddof=1),
        "cv_rr": np.std(nn_ms, ddof=1) / np.mean(nn_ms),
    }


def frequency_domain_metrics(nn_ms: np.ndarray, interp_fs: float = 4.0) -> Dict[str, float]:
    """
    Spectral HRV metrics. The uneven NN tachogram is resampled to a uniform grid
    with a cubic spline, detrended, and its power spectrum estimated with Welch's method.
    Bands the recording is too short to resolve are returned as NaN.
    """
    nan_result = {k: np.nan for k in ("lf_power", "hf_power", "lf_hf_ratio", "lf_nu", "hf_nu")}
    if len(nn_ms) < 10:
        return nan_result

    beat_times = np.cumsum(nn_ms) / 1000.0
    duration = beat_times[-1] - beat_times[0]
    if duration < MIN_DURATION_HF:
        return nan_result

    grid = np.arange(beat_times[0], beat_times[-1], 1.0 / interp_fs)
    tachogram = detrend(CubicSpline(beat_times, nn_ms)(grid))

    nperseg = min(len(tachogram), int(interp_fs * 120))
    freqs, psd = welch(tachogram, fs=interp_fs, nperseg=nperseg)

    def band_power(band):
        mask = (freqs >= band[0]) & (freqs < band[1])
        return np.trapezoid(psd[mask], freqs[mask])

    hf = band_power(HF_BAND)
    lf = band_power(LF_BAND) if duration >= MIN_DURATION_LF else np.nan

    return {
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf / hf if hf > 0 else np.nan,
        "lf_nu": lf / (lf + hf) * 100.0,
        "hf_nu": hf / (lf + hf) * 100.0,
    }


def sample_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Sample entropy (Richman & Moorman, 2000): -ln(A/B), where B counts pairs of
    length-m templates within tolerance r and A counts pairs of length m+1.
    Lower values indicate a more regular, predictable rhythm. See docs/clinical-background
    for more info
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)

    def count_matches(length):
        # Use the same number of templates (n - m) for both lengths.
        templates = np.lib.stride_tricks.sliding_window_view(x, length)[: n - m]
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return (np.sum(dist <= r) - len(templates)) / 2  # exclude self-matches

    b = count_matches(m)
    a = count_matches(m + 1)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan


def nonlinear_metrics(nn_ms: np.ndarray) -> Dict[str, float]:
    """
    Poincaré plot descriptors and sample entropy.
    SD1 reflects short-term (beat-to-beat) variability, SD2 long-term variability.
    """
    if len(nn_ms) < 5:
        return {k: np.nan for k in ("sd1", "sd2", "sd1_sd2_ratio", "sample_entropy")}

    rr_n, rr_n1 = nn_ms[:-1], nn_ms[1:]
    sd1 = np.std((rr_n1 - rr_n) / np.sqrt(2), ddof=1)
    sd2 = np.std((rr_n1 + rr_n) / np.sqrt(2), ddof=1)

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1 / sd2 if sd2 > 0 else np.nan,
        "sample_entropy": sample_entropy(nn_ms),
    }


def extract_hrv_features(r_peaks: Sequence[int], fs: float) -> Dict[str, float]:
    """
    Full HRV feature set (time, frequency, non-linear) plus quality indicators.
    """
    rr_ms = rr_intervals_ms(r_peaks, fs)
    nn_ms = clean_rr_intervals(rr_ms)

    features = {
        "n_beats": len(r_peaks),
        "nn_fraction": len(nn_ms) / len(rr_ms) if len(rr_ms) else 0.0,
    }
    features.update(time_domain_metrics(nn_ms))
    features.update(frequency_domain_metrics(nn_ms))
    features.update(nonlinear_metrics(nn_ms))
    return features
