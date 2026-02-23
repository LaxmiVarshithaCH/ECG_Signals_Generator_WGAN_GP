# data/physionet_loader.py

import os
import wfdb
import numpy as np
from data.preprocess import preprocess_segment


def segment_signal(signal, seq_len):
    segments = []
    for i in range(0, len(signal) - seq_len + 1, seq_len):
        segments.append(signal[i:i + seq_len])
    return segments


def load_physionet(
    physionet_dir,
    seq_len,
    lowcut,
    highcut,
    lead_index=0
):

    if not os.path.isdir(physionet_dir):
        raise FileNotFoundError(f"Directory not found: {physionet_dir}")

    all_segments = []

    for file in os.listdir(physionet_dir):
        if file.endswith(".hea"):

            record_name = file.replace(".hea", "")
            record_path = os.path.join(physionet_dir, record_name)

            record = wfdb.rdrecord(record_path)

            signal = record.p_signal[:, lead_index]
            fs = record.fs

            segments = segment_signal(signal, seq_len)

            for seg in segments:
                processed = preprocess_segment(
                    seg,
                    fs,
                    seq_len,
                    lowcut,
                    highcut
                )
                all_segments.append(processed)

    if len(all_segments) == 0:
        raise RuntimeError("No ECG segments extracted.")

    ecg_array = np.array(all_segments)
    ecg_array = ecg_array[:, :, np.newaxis]  # (N, 1248, 1)

    return ecg_array