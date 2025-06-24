import os
import wfdb
from typing import Optional

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
