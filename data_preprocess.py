# Requires a YAML config file (e.g., config.yaml) with keys:
# train_metadata_path, train_data_path, test_metadata_path, test_data_path, (optional) sr, n_mfcc

import os
import librosa
import numpy as np
import yaml
import random
import torch
import torchaudio
from scipy.fftpack import dct


# -------- Linear filterbank for LFCC --------
def linear_fbanks(sr, n_fft, n_filters):
    """
    Create a linear triangular filterbank.
    Returns array of shape (n_filters, 1 + n_fft//2).
    """
    # FFT bin frequencies
    fft_freqs = np.linspace(0, sr/2, int(1 + n_fft//2))
    # Linearly spaced filter center frequencies
    center_freqs = np.linspace(0, sr/2, n_filters + 2)
    fb = np.zeros((n_filters, len(fft_freqs)))
    for i in range(n_filters):
        f_left, f_center, f_right = center_freqs[i], center_freqs[i+1], center_freqs[i+2]
        # Upward slope
        left_slope = (fft_freqs - f_left) / (f_center - f_left)
        # Downward slope
        right_slope = (f_right - fft_freqs) / (f_right - f_center)
        # Combine and clip
        fb_i = np.minimum(np.maximum(left_slope, 0), np.maximum(right_slope, 0))
        fb[i, :] = fb_i
    return fb

def extract_mfcc(wav_path, sr=16000, n_mfcc=20):
    """
    Extracts MFCC, delta, and delta-delta features and returns the full spectrogram.
    Returns:
        np.ndarray: Combined MFCC, delta, delta-delta (shape: 3*n_mfcc, time_steps)
    """
    y, sr_ret = librosa.load(wav_path, sr=sr)
    # Skip empty files
    if len(y) == 0:
        print(f"[WARN] Empty audio: {wav_path} — skipped.")
        return None
    # Adapt n_fft to short signals to silence librosa warning
    n_fft_use = 2048 if len(y) >= 2048 else 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr_ret, n_mfcc=n_mfcc, n_fft=n_fft_use)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
    # Return combined MFCC, delta, delta-delta spectrogram: shape (3*n_mfcc, time_steps)
    return mfcc_combined

def extract_log_mel(wav_path, sr=16000, n_mels=64, hop_length=None, win_length=None):
    """
    Extract log-scaled Mel spectrogram.
    Returns:
        np.ndarray: shape (n_mels, time_steps)
    """
    y, sr_ret = librosa.load(wav_path, sr=sr)
    if len(y) == 0:
        print(f"[WARN] Empty audio: {wav_path} — skipped.")
        return None
    # determine FFT and window/hop lengths
    n_fft_use = 2048 if len(y) >= 2048 else 512
    hop = hop_length if hop_length is not None else int(0.010 * sr_ret)
    win = win_length if win_length is not None else int(0.020 * sr_ret)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr_ret, n_mels=n_mels, n_fft=n_fft_use,
        hop_length=hop, win_length=win
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


# -------- LFCC extraction --------
def extract_lfcc(wav_path, sr=16000, n_lfcc=20, n_fft=2048):
    """
    Extracts LFCC (Linear Frequency Cepstral Coefficients) with delta and delta-delta.
    Returns a numpy array of shape (3*n_lfcc, time_steps).
    """
    y, sr_ret = librosa.load(wav_path, sr=sr)
    if len(y) == 0:
        print(f"[WARN] Empty audio: {wav_path} — skipped.")
        return None
    # Compute power spectrogram
    n_fft_use = n_fft if len(y) >= n_fft else 512
    S = np.abs(librosa.stft(y, n_fft=n_fft_use, hop_length=int(0.010*sr_ret), win_length=int(0.020*sr_ret)))**2
    # Create linear filterbank
    lin_fb = linear_fbanks(sr_ret, n_fft_use, n_lfcc)
    lin_spec = np.dot(lin_fb, S)  # shape: (n_lfcc, time_steps)
    # Log scaling
    log_lin = librosa.power_to_db(lin_spec, ref=np.max)
    # DCT to get LFCC
    lfcc = dct(log_lin, type=2, axis=0, norm='ortho')[:n_lfcc]
    # Delta and Delta-Delta
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
    return np.concatenate((lfcc, lfcc_delta, lfcc_delta2), axis=0)

def time_mask(spec, T=30, num_masks=1):
    """Apply time masking for SpecAugment. Supports 2D and 3D arrays."""
    spec = spec.copy()
    # handle (freq, time) or (channels, freq, time)
    t_max = spec.shape[-1]
    for _ in range(num_masks):
        t = random.randint(0, T)
        t0 = random.randint(0, max(0, t_max - t))
        if spec.ndim == 2:
            spec[:, t0:t0+t] = 0
        else:
            spec[:, :, t0:t0+t] = 0
    return spec

def freq_mask(spec, F=15, num_masks=1):
    """Apply frequency masking for SpecAugment. Supports 2D and 3D arrays."""
    spec = spec.copy()
    # handle (freq, time) or (channels, freq, time)
    f_max = spec.shape[-2]
    for _ in range(num_masks):
        f = random.randint(0, F)
        f0 = random.randint(0, max(0, f_max - f))
        if spec.ndim == 2:
            spec[f0:f0+f, :] = 0
        else:
            spec[:, f0:f0+f, :] = 0
    return spec

def load_data(metadata_path, data_path, sr=16000, n_mfcc=20, cfg=None):
    x, y, file_names = [], [], []
    label_map = {'fake': 0, 'real': 1}
    max_len = None
    if cfg is not None:
        max_len = cfg.get('max_time_steps', 200)
    else:
        max_len = 200
    with open(metadata_path, 'r') as f:
        for line in f:
            spk, file_name, _, _, label = line.strip().split(' ')
            wav_path = os.path.join(data_path, file_name)
            # Choose feature type(s)
            use_lfcc = cfg.get('use_lfcc', False) if cfg is not None else False
            use_log_mel = cfg.get('use_log_mel', False) if cfg is not None else False
            features_list = []
            if use_lfcc:
                lfcc_feat = extract_lfcc(wav_path, sr=sr, n_lfcc=cfg.get('n_lfcc', 20), n_fft=cfg.get('n_fft', 2048))
                if lfcc_feat is None: continue
                features_list.append(lfcc_feat)
            if use_log_mel:
                logmel_feat = extract_log_mel(
                    wav_path, sr=sr, n_mels=cfg.get('n_mels', 64),
                    hop_length=cfg.get('hop_length'), win_length=cfg.get('win_length')
                )
                if logmel_feat is None: continue
                features_list.append(logmel_feat)
            if not features_list:
                mfcc_feat = extract_mfcc(wav_path, sr=sr, n_mfcc=n_mfcc)
                if mfcc_feat is None: continue
                features_list.append(mfcc_feat)
            # Concatenate features along frequency axis: (freq_bins_total, time_steps)
            features = np.concatenate(features_list, axis=0)
            # Apply SpecAugment if enabled
            if cfg is not None and cfg.get('use_specaugment', False):
                features = time_mask(features, T=cfg.get('time_mask_param', 30),
                                     num_masks=cfg.get('time_masks', 1))
                features = freq_mask(features, F=cfg.get('freq_mask_param', 15),
                                     num_masks=cfg.get('freq_masks', 1))
            # Pad/truncate features along time axis to fixed length
            if features.ndim == 2:
                h, t = features.shape
            else:
                _, h, t = features.shape
            if t < max_len:
                if features.ndim == 2:
                    padding = np.zeros((h, max_len - t), dtype=features.dtype)
                    features = np.concatenate([features, padding], axis=1)
                else:
                    c = features.shape[0]
                    padding = np.zeros((c, h, max_len - t), dtype=features.dtype)
                    features = np.concatenate([features, padding], axis=2)
            else:
                if features.ndim == 2:
                    features = features[:, :max_len]
                else:
                    features = features[:, :, :max_len]
            # Sample-wise normalization: zero-mean, unit-variance
            features = (features - np.mean(features)) / (np.std(features) + 1e-6)
            x.append(features)
            label_int = label_map[label.lower()]
            y.append(label_int)
            file_names.append(file_name)
    return x, y, file_names

def load_config(path='config.yaml'):
    return yaml.safe_load(open(path))







