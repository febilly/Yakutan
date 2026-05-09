import pytest
import numpy as np
from audio_resampler import AudioResampler

def test_audio_resampler_needs_no_resampling():
    """Test when input and output rates are the same, no resampling is needed and data is returned as-is."""
    resampler = AudioResampler(input_rate=16000, output_rate=16000)
    assert not resampler.needs_resample
    data = b'\x00\x00\x01\x00'
    assert resampler.resample(data) == data

def test_audio_resampler_empty_data():
    """Test resampling with an empty byte string."""
    resampler = AudioResampler(input_rate=48000, output_rate=16000)
    assert resampler.needs_resample
    assert resampler.resample(b'') == b''

def test_audio_resampler_zero_frames():
    """Test resampling with partial frame data resulting in 0 frames."""
    # channels=2, sample_width=2 -> 1 frame = 4 bytes
    resampler = AudioResampler(input_rate=48000, output_rate=16000, channels=2, sample_width=2)
    assert resampler.needs_resample

    # Pass 2 bytes: length of samples will be 1 (for int16), which means 1 sample
    # num_frames = len(samples) // channels = 1 // 2 = 0
    data = b'\x00\x00'
    assert resampler.resample(data) == data

def test_audio_resampler_normal_data():
    """Test resampling with normal data."""
    resampler = AudioResampler(input_rate=48000, output_rate=16000, channels=1, sample_width=2)
    assert resampler.needs_resample
    # 48000 / 16000 = 3, so every 3 frames will become 1 frame (roughly).
    # Provide 6 bytes = 3 int16 frames
    data = b'\x00\x00\x00\x00\x00\x00'
    result = resampler.resample(data)
    # The output might not strictly be exactly 1 frame (2 bytes) depending on soxr's buffering,
    # but let's just make sure it runs and returns bytes without raising.
    assert isinstance(result, bytes)
