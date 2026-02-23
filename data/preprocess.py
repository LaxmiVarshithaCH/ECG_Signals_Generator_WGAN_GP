# data/preprocess.py

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut, highcut):
    b, a = butter(
        N=4,
        Wn=[lowcut / (fs / 2), highcut / (fs / 2)],
        btype="band"
    )
    return filtfilt(b, a, signal)


def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def enforce_length(signal, target_len):
    if len(signal) > target_len:
        signal = signal[:target_len]
    elif len(signal) < target_len:
        pad_len = target_len - len(signal)
        signal = np.pad(signal, (0, pad_len), mode="constant")
    return signal.astype(np.float32)


def preprocess_segment(segment, fs, target_len, lowcut, highcut):
    segment = bandpass_filter(segment, fs, lowcut, highcut)
    segment = normalize(segment)
    segment = enforce_length(segment, target_len)
    return segment