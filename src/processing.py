import os
import requests
import wfdb
from typing import Optional, List, Dict
import numpy as np
from scipy.signal import butter, sosfiltfilt
from biosppy.signals import ecg

# Get the absolute path to the directory where this script (loading.py) is located.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up to get the project's root directory.
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# Construct the absolute path to the default local PTB cache (gitignored).
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw', 'ptbdb')

# PhysioNet's official AWS Open Data mirror: same files, far less throttled than physionet.org.
_S3_MIRROR = 'https://physionet-open.s3.amazonaws.com/{db}/{version}/{path}'
_DB_VERSIONS = {'ptbdb': '1.0.0'}


def _download_from_mirror(database_name: str, files: List[str], data_dir: str) -> None:
    """
    Downloads files from the S3 mirror. Each file is written to a temporary '.part'
    path and renamed when complete, so an interrupted download never looks finished.
    """
    for path in files:
        url = _S3_MIRROR.format(db=database_name, version=_DB_VERSIONS[database_name], path=path)
        local_path = os.path.join(data_dir, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(local_path + '.part', 'wb') as f:
            f.write(response.content)
        os.replace(local_path + '.part', local_path)


def load_ecg_record(record_name: str, database_name: str = 'ptbdb', data_dir: str = None) -> Optional[wfdb.Record]:
    """
    Checks for a local copy in 'data_dir'. If not found, it downloads the
    record (from PhysioNet's S3 mirror, falling back to physionet.org) and saves
    it to the correct local subdirectory for future use.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    # Construct the full local path for the record
    local_record_path = os.path.join(data_dir, record_name)
    
    try:
        # Ensure the files exist locally by downloading if needed.
        if not os.path.exists(f"{local_record_path}.hea"):
            print(f"Record '{record_name}' not found locally. Downloading...")
            
            # Define the exact file paths as they exist on the PhysioNet server.
            # The header goes last: marks the record as fully downloaded.
            files_to_download = [f"{record_name}.dat", f"{record_name}.xyz", f"{record_name}.hea"]

            try:
                _download_from_mirror(database_name, files_to_download, data_dir)
            except (requests.RequestException, KeyError) as e:
                print(f"Mirror download failed ({e}); falling back to physionet.org.")
                wfdb.dl_files(
                    db=database_name, 
                    dl_dir=data_dir, 
                    files=files_to_download,
                    keep_subdirs=True # ensures the patient folder is created correctly.
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
    
    # Design the Butterworth bandpass filter as second-order sections, which stay
    # numerically stable at very low cutoffs (e.g. 0.05 Hz for ST-segment analysis), see docs/clinical-background for details.
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply the filter forwards and backwards for zero-phase filtering.
    filtered_signal = sosfiltfilt(sos, signal)
    
    return filtered_signal


def find_r_peaks(signal: np.ndarray, fs: int) -> List[int]:
    """
    Finds R-peaks in an ECG signal using biosppy's Hamilton QRS detector.
    """
    results = ecg.ecg(signal=signal, sampling_rate=fs, show=False)

    # The indices of the R-peaks are stored in the 'rpeaks' key.
    r_peaks = results['rpeaks']

    return r_peaks
