import os
import wfdb
from typing import Optional, List, Dict
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Get the absolute path to the directory where this script (loading.py) is located.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up to get the project's root directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
# Construct the absolute path to the default raw data directory.
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw')

def load_ecg_record(record_name: str, database_name: str = 'ptbdb', data_dir: str = None) -> Optional[wfdb.Record]:
    """
    Checks for a local copy in 'data_dir'. If not found, it downloads the
    record from the specified PhysioNet database and saves it to the correct
    local subdirectory for future use.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    # Construct the full local path for the record
    local_record_path = os.path.join(data_dir, record_name)
    
    try:
        # Ensure the files exist locally by downloading if needed.
        if not os.path.exists(f"{local_record_path}.hea"):
            print(f"Record '{record_name}' not found locally. Downloading...")
            
            # Define the exact file paths as they exist on the PhysioNet server
            files_to_download = [f"{record_name}.dat", f"{record_name}.hea",f"{record_name}.xyz"]
            
            wfdb.dl_files(
                db=database_name, 
                dl_dir=data_dir, 
                files=files_to_download,
                keep_subdirs=True # This ensures the patient folder is created correctly.
            )
            print("Download complete.")
        
        print(f"Loading record '{record_name}' from local directory.")
        record = wfdb.rdrecord(local_record_path)
        return record

    except Exception as e:
        print(f"An error occurred while processing record '{record_name}': {e}")
        return None


def filter_signal(signal: np.ndarray, fs: int, low_cut: float = 0.5, high_cut: float = 45.0, order: int = 4) -> np.ndarray:
    """
    Applies a bandpass Butterworth filter to the input signal.
    """
    # Calculate the Nyquist frequency (half the sampling frequency)
    nyquist = 0.5 * fs
    
    # Normalize the cutoff frequencies with respect to the Nyquist frequency.
    low = low_cut / nyquist
    high = high_cut / nyquist
    
    # Design the Butterworth bandpass filter.
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering.
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def find_r_peaks(signal: np.ndarray, fs: int) -> List[int]:
    """
    Finds R-peaks in a filtered ECG signal using a simplified Pan-Tompkins algorithm.
    """
    # Differentiate the signal to highlight the QRS complexes
    diff_signal = np.diff(signal)

    # Square the signal to enhance peaks
    squared_signal = diff_signal ** 2

    # Integrate using a moving window average
    # The window size is 150ms.
    window_size = int(0.150 * fs)
    integrated_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')

    # Find peaks in the processed signal
    # A minimum distance is set to avoid detecting multiple peaks within one QRS.
    # A typical human heart rate is 60-100 bpm, so a safe minimum distance is >200ms.
    min_distance = int(0.3 * fs) # 300ms
    
    # set a dynamic threshold based on the signal's mean and std dev to avoid detecting noise as peaks.
    peak_threshold = np.mean(integrated_signal) + 2 * np.std(integrated_signal)
    
    # Use scipy's find_peaks function
    r_peaks, _ = find_peaks(
        integrated_signal, 
        height=peak_threshold, 
        distance=min_distance
    )
    
    return r_peaks

def calculate_hrv_metrics(r_peaks: List[int], fs: int) -> Dict[str, float]:
    """
    Calculates time-domain HRV metrics from a list of R-peak indices.
    """
    if len(r_peaks) < 2:
        return {"mean_rr": 0, "sdnn": 0, "rmssd": 0}

    # Calculate RR intervals in number of samples
    rr_intervals_samples = np.diff(r_peaks)
    
    # Convert RR intervals from samples to milliseconds, in case fs is not 1000Hz
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000

    # Calculate Mean RR (ms)
    mean_rr = np.mean(rr_intervals_ms)

    # Calculate Standard Deviation of NN intervals (ms)
    sdnn = np.std(rr_intervals_ms)

    # Calculate Root Mean Square of Successive Differences (ms)
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    hrv_metrics = {
        "mean_rr": round(mean_rr, 2),
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2)
    }
    
    return hrv_metrics