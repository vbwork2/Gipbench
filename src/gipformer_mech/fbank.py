"""Kaldi-compatible log-mel filterbank in NumPy, with the options sherpa-onnx uses for Gipformer.

Matches kaldi-native-fbank with dither 0, snip_edges False, remove_dc_offset, preemphasis 0.97,
povey window, power spectrum, 25 ms frames every 10 ms, low_freq 20 Hz and high_freq -400 Hz.
Samples are used as given (normalized to [-1, 1]), as in `rnnt_loader.encode`.
"""

import numpy as np


def mel_scale(frequency):
    return 1127.0 * np.log(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def mel_banks(num_bins: int, padded: int, sample_rate: int, low_freq: float, high_freq: float) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    high = nyquist + high_freq if high_freq <= 0 else high_freq
    if not 0 <= low_freq < high <= nyquist:
        raise ValueError("Invalid mel filterbank frequency range.")
    mel_low, mel_high = mel_scale(low_freq), mel_scale(high)
    delta = (mel_high - mel_low) / (num_bins + 1)
    mel = mel_scale(np.arange(padded // 2) * sample_rate / padded)
    left = mel_low + np.arange(num_bins)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    rising = (mel - left) / (center - left)
    falling = (right - mel) / (right - center)
    weights = np.where(mel <= center, rising, falling)
    return np.where((mel > left) & (mel < right), weights, 0.0)


def fbank(waveform: np.ndarray, sample_rate: int = 16000, num_bins: int = 80,
          low_freq: float = 20.0, high_freq: float = -400.0) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float64)
    length, shift = int(sample_rate * 0.025), int(sample_rate * 0.010)
    padded = 1 << (length - 1).bit_length()
    count = (len(x) + shift // 2) // shift
    if not count:
        return np.zeros((0, num_bins), dtype=np.float32)
    index = (np.arange(count) * shift + shift // 2 - length // 2)[:, None] + np.arange(length)[None]
    while (index < 0).any() or (index >= len(x)).any():
        index = np.where(index < 0, -index - 1, index)
        index = np.where(index >= len(x), 2 * len(x) - 1 - index, index)
    frames = x[index]
    frames = frames - frames.mean(axis=1, keepdims=True)
    emphasized = np.empty_like(frames)
    emphasized[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    emphasized[:, 0] = frames[:, 0] * (1 - 0.97)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / (length - 1))) ** 0.85
    power = np.abs(np.fft.rfft(emphasized * window, n=padded)) ** 2
    energies = power[:, :padded // 2] @ mel_banks(num_bins, padded, sample_rate, low_freq, high_freq).T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)
