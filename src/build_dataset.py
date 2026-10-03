import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional

import pandas as pd
import wfdb

from processing import load_ecg_record, filter_signal, find_r_peaks
from hrv import extract_hrv_features
from morphology import extract_morphology_features

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'ptb_features.csv')

# Lead used for R-peak detection; morphology is measured on all 12 standard leads.
DETECTION_LEAD = 'ii'

# Map PTB 'Reason for admission' diagnoses onto classification labels.
LABEL_MAP = {
    'Healthy control': 'Normal',
    'Myocardial infarction': 'MI',
}


def parse_header_metadata(comments) -> Dict[str, str]:
    """
    Extracts patient metadata from the PTB header comment lines ('key: value').
    """
    fields = {}
    for line in comments:
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip().lower()] = value.strip()

    diagnosis = fields.get('reason for admission', 'n/a')
    return {
        'diagnosis': diagnosis,
        'label': LABEL_MAP.get(diagnosis, 'Other'),
        'age': pd.to_numeric(fields.get('age'), errors='coerce'),
        'sex': fields.get('sex', 'n/a'),
    }


def process_record(record_name: str) -> Optional[Dict]:
    """
    Runs the full pipeline on one PTB record: load -> filter -> detect R-peaks ->
    HRV + morphology features. Returns one feature row, or None on failure.
    """
    record = load_ecg_record(record_name)
    if record is None:
        return None

    fs = record.fs
    lead_names = [name.lower() for name in record.sig_name]

    # Two filtered versions: a narrower band for robust QRS detection, and a
    # diagnostic band (0.05 Hz high-pass) that preserves the ST segment.
    detection_signal = filter_signal(record.p_signal[:, lead_names.index(DETECTION_LEAD)], fs)
    # filter_signal works along the last axis, so filter the (leads, samples) transpose.
    diagnostic_signals = filter_signal(record.p_signal.T, fs, low_cut=0.05).T

    r_peaks = find_r_peaks(detection_signal, fs)
    if len(r_peaks) < 5:
        print(f"  --> Skipping {record_name}: only {len(r_peaks)} R-peaks found.")
        return None

    row = {
        'record': record_name,
        'patient': record_name.split('/')[0],
        'duration_s': record.sig_len / fs,
    }
    row.update(parse_header_metadata(record.comments))
    row.update(extract_hrv_features(r_peaks, fs))
    row.update(extract_morphology_features(diagnostic_signals, lead_names, r_peaks, fs))
    return row


def build_feature_dataset(output_csv_path: str, limit: Optional[int] = None, workers: int = 4) -> pd.DataFrame:
    """
    Processes every record in the PTB Diagnostic ECG Database (downloading and
    caching each one on first use) and saves the combined feature table.
    """
    record_names = wfdb.get_record_list('ptbdb')
    if limit:
        record_names = record_names[:limit]
    print(f"Processing {len(record_names)} PTB records with {workers} workers...")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_record, name): name for name in record_names}
        for i, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                row = future.result()
            except Exception as e:
                print(f"  --> Failed {name}: {e}")
                continue
            if row is not None:
                rows.append(row)
            print(f"[{i}/{len(record_names)}] {name}")

    features_df = pd.DataFrame(rows).sort_values('record').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    features_df.to_csv(output_csv_path, index=False)

    print(f"\nFeature engineering complete. Processed {len(features_df)} valid records.")
    print(features_df['label'].value_counts().to_string())
    print(f"Saved feature dataset to {output_csv_path}")
    return features_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the PTB HRV + morphology feature dataset.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N records (for testing).')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    build_feature_dataset(args.output, limit=args.limit, workers=args.workers)
