from utils.loading import load_ecg_record # Helper funct.

def main():
    """
    Main function to run the ECG analysis pipeline.
    """
    # --- Load Data ---
    # Define the record to analyze
    record_name = 'patient001/s0010_re'
    
    # Use  function to load the data
    record = load_ecg_record(record_name=record_name)

    if record:
        print("\n--- Data Loading Successful ---")
        print(f"Record: {record.record_name}")
        print(f"Sampling Frequency: {record.fs} Hz")
        print(f"Signal Shape: {record.p_signal.shape}")
        
    else:
        print("\n--- Data Loading Failed. Exiting Pipeline. ---")


if __name__ == "__main__":
    main()

