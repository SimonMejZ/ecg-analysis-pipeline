from processing import load_ecg_record, filter_signal, find_r_peaks, calculate_hrv_metrics # Helper functs.

def main():
    """
    Main function to run the ECG analysis pipeline.
    """
    # --- Load Data ---
    # Define the record to analyze
  
    record_name = 'patient001/s0010_re'
    record = load_ecg_record(record_name=record_name)

    if record:
        print("\n--- Data Loading Successful ---")
        print(f"Record: {record.record_name}, Sampling Freq: {record.fs} Hz")

        raw_signal = record.p_signal[:, 0]
        sampling_freq = record.fs
        
        #Filter Signal
        print("\n--- Filtering Signal ---")
        filtered_signal = filter_signal(
            signal=raw_signal, 
            fs=sampling_freq
        )
        print("Signal filtering complete.")
        
        # The 'filtered_signal' variable now holds the cleaned data.

        # --- 3. Find R-Peaks ---
        print("\n--- Finding R-Peaks ---")
        r_peaks = find_r_peaks(
            signal=filtered_signal,
            fs=sampling_freq
        )
        print(f"Detected {len(r_peaks)} R-peaks.")

        # --- 4. Calculate HRV Metrics ---
        print("\n--- Calculating HRV Metrics ---")
        hrv_metrics = calculate_hrv_metrics(r_peaks=r_peaks, fs=sampling_freq)
        print("HRV calculation complete.")
        
        print("\n--- Analysis Results ---")
        for metric, value in hrv_metrics.items():
            print(f"{metric.upper()}: {value} ms")
        
    else:
        print("\n--- Data Loading Failed. Exiting Pipeline. ---")


if __name__ == "__main__":
    main()

