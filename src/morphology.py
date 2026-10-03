"""Defines all formal metrics used in main.py supported by hrv.py. See docs/clinical-background.md 
for more detailed explanations of all methods and reasoning
"""

import numpy as np
from typing import Dict, List, Sequence, Tuple

# Beat window around each R-peak (seconds).
PRE_R = 0.25
POST_R = 0.45

# The 12 standard clinical leads in the PTB database.
STANDARD_LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]


def extract_beats(signal: np.ndarray, r_peaks: Sequence[int], fs: float) -> np.ndarray:
    """
    Cuts a fixed window around every R-peak. Beats too close to the recording
    edges are skipped. Returns an array of shape (n_beats, window_length).
    """
    pre, post = int(PRE_R * fs), int(POST_R * fs)
    beats = [signal[r - pre:r + post] for r in r_peaks if r - pre >= 0 and r + post <= len(signal)]
    return np.array(beats)


def median_template(beats: np.ndarray) -> np.ndarray:
    """
    Median beat: robust to occasional ectopic beats and noise bursts.
    """
    return np.median(beats, axis=0)


def find_qrs_boundaries(templates: np.ndarray, fs: float) -> Tuple[int, int]:
    """
    Locates global QRS onset and offset from the multi-lead median templates.
    The absolute slopes of all leads are summed, so the boundaries mark where
    ventricular depolarisation starts and ends in any lead (as cardiologists measure it).
    """
    r_index = int(PRE_R * fs)
    slope = np.sum(np.abs(np.gradient(templates, axis=1)), axis=0)

    # Smooth with a 10 ms moving average to suppress noise.
    win = max(1, int(0.01 * fs))
    slope = np.convolve(slope, np.ones(win) / win, mode="same")

    # The detector's fiducial point is not always the QRS centre (it can sit on a
    # late R' or S wave), so anchor on the steepest point of the multi-lead slope.
    search_start, search_end = r_index - int(0.15 * fs), r_index + int(0.18 * fs)
    window = slope[search_start:search_end]
    peak = int(np.argmax(window))
    threshold = 0.15 * window[peak]

    # Grow the QRS outwards from the peak through steep samples, bridging short
    # dips (< 20 ms) between notches such as Q-R-S or R-R' deflections.
    max_gap = int(0.02 * fs)
    steep = np.where(window > threshold)[0]
    gaps = np.diff(steep) > max_gap
    segment_id = np.concatenate([[0], np.cumsum(gaps)])
    qrs_samples = steep[segment_id == segment_id[np.searchsorted(steep, peak)]]

    return search_start + qrs_samples[0], search_start + qrs_samples[-1]


def lead_morphology(template: np.ndarray, onset: int, offset: int, fs: float) -> Dict[str, float]:
    """
    Amplitude measurements (mV) on one lead's median beat, all relative to the
    isoelectric PR segment just before QRS onset.
    """
    baseline = np.median(template[max(0, onset - int(0.03 * fs)):max(1, onset - int(0.005 * fs))])
    beat = template - baseline
    qrs = beat[onset:offset + 1]

    peak_idx = int(np.argmax(qrs))
    q_amp = np.min(qrs[:peak_idx + 1])
    s_amp = np.min(qrs[peak_idx:])

    st_idx = min(len(beat) - 1, offset + int(0.06 * fs))

    t_window = beat[offset + int(0.08 * fs):]
    t_amp = t_window[np.argmax(np.abs(t_window))] if len(t_window) else np.nan

    return {
        "r_amp": qrs[peak_idx],
        "q_amp": q_amp,
        "s_amp": s_amp,
        "st_j": beat[offset],        # ST level at the J-point
        "st_60": beat[st_idx],       # ST level 60 ms after the J-point
        "t_amp": t_amp,
    }


def extract_morphology_features(signals: np.ndarray, lead_names: List[str], r_peaks: Sequence[int],
                                fs: float, leads: List[str] = None) -> Dict[str, float]:
    """
    Builds a median beat for each requested lead, finds global QRS boundaries and
    measures per-lead amplitudes. Features are named '<lead>_<measure>'.
    `signals` has shape (n_samples, n_leads), matching wfdb's p_signal.
    """
    if leads is None:
        leads = STANDARD_LEADS
    lead_idx = [lead_names.index(lead) for lead in leads]

    beats_per_lead = [extract_beats(signals[:, i], r_peaks, fs) for i in lead_idx]
    if len(beats_per_lead[0]) < 3:
        return {}

    templates = np.array([median_template(b) for b in beats_per_lead])
    onset, offset = find_qrs_boundaries(templates, fs)

    features = {"qrs_duration": (offset - onset) / fs * 1000.0}

    # Beat-to-template correlation: a signal quality / morphology stability index.
    ref_beats, ref_template = beats_per_lead[0], templates[0]
    correlations = [np.corrcoef(beat, ref_template)[0, 1] for beat in ref_beats]
    features["template_corr"] = float(np.median(correlations))

    for lead, template in zip(leads, templates):
        for name, value in lead_morphology(template, onset, offset, fs).items():
            features[f"{lead}_{name}"] = value

    return features
