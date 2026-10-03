from processing import load_ecg_record, filter_signal, find_r_peaks # Helper functs.
from hrv import extract_hrv_features
from morphology import extract_morphology_features

def main():
    """
    Main function to run the ECG analysis pipeline.
    """
    # --- Load Data ---
    # Define the record to analyze
  
    record_name = 'patient005/s0021are'
    record = load_ecg_record(record_name=record_name)

    if record:
        print("\n Data Loading Successful")
        print(f"Record: {record.record_name}, Sampling Freq: {record.fs} Hz")

        raw_signal = record.p_signal[:, 0]
        sampling_freq = record.fs
        
        #Filter Signal
        print("\n Filtering Signal")
        filtered_signal = filter_signal(
            signal=raw_signal, 
            fs=sampling_freq
        )
        print("Signal filtering complete.")
        
        # The 'filtered_signal' variable now holds the cleaned data.

        # --- 3. Find R-Peaks ---
        print("\nFinding R-Peaks")
        r_peaks = find_r_peaks(
            signal=filtered_signal,
            fs=sampling_freq
        )
        print(f"Detected {len(r_peaks)} R-peaks.")

        # --- 4. Calculate HRV Metrics ---
        print("\nCalculating HRV Metrics")
        hrv_metrics = extract_hrv_features(r_peaks=r_peaks, fs=sampling_freq)
        print("HRV calculation complete.")

        # --- 5. Beat Morphology (median beat per lead, 0.05 Hz high-pass preserves the ST segment) ---
        print("\nMeasuring Beat Morphology")
        diagnostic_signals = filter_signal(record.p_signal.T, sampling_freq, low_cut=0.05).T
        lead_names = [name.lower() for name in record.sig_name]
        morphology = extract_morphology_features(diagnostic_signals, lead_names, r_peaks, sampling_freq)
        print("Morphology measurement complete.")

        print("\nAnalysis Results")
        for metric, value in {**hrv_metrics, **morphology}.items():
            print(f"{metric}: {value:.3f}")
        
    else:
        print("\nData Loading Failed. Exiting Pipeline.")


if __name__ == "__main__":
    main()

