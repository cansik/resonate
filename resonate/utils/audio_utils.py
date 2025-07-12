from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

ArrayOrTensor = np.ndarray | torch.Tensor


def save_wav(
        path: str | Path,
        audio: ArrayOrTensor,
        sample_rate: int,
        auto_clamp: bool = False
) -> None:
    """
    Save audio data (numpy array or torch tensor) to a WAV file using torchaudio.save.

    :param path:       Output file path (str or Path).
    :param audio:      Audio data, 1D or 2D numpy array or torch tensor.
                       If 2D, can be [channels, time] or [time, channels] (auto-fixed).
                       Accepts:
                         - float in [-1.0, 1.0]
                         - torch.uint8      (0–255 unsigned 8-bit PCM)
                         - torch.int16      (signed 16-bit PCM)
                         - torch.int32      (signed 32-bit PCM)
                         - torch.int64      (will be downcast to int32 with clipping)
    :param sample_rate: Sample rate in Hz.
    :param auto_clamp: Clamp the floating point signal if it is too loud.
    :return:            None
    """
    path = Path(path)

    # Convert numpy to torch, remember original dtype
    np_dtype = None
    if isinstance(audio, np.ndarray):
        np_dtype = audio.dtype
        audio = torch.from_numpy(audio)

    # Fix shape to [channels, time]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2:
        # Detect [time, channels] vs [channels, time]
        if audio.shape[0] >= audio.shape[1]:
            # assume [time, channels]
            audio = audio.transpose(0, 1)
    else:
        raise ValueError(f"Audio tensor must be 1D or 2D, got {audio.ndim}D")

    # Decide what to do based on dtype
    dt = audio.dtype
    if dt == torch.uint8 or dt == torch.int16 or dt == torch.int32:
        # Already a supported PCM format
        int_audio = audio
    elif dt == torch.int64:
        # WAV does not support int64; downcast with clipping
        info = torch.iinfo(torch.int32)
        clipped = audio.clamp(info.min, info.max)
        int_audio = clipped.to(torch.int32)
    elif dt.is_floating_point:
        # Float32 or Float64 → scale & convert to int16
        if auto_clamp:
            audio = audio.clamp(-1.0, 1.0)
        else:
            audio_min = audio.min()
            audio_max = audio.max()
            if audio_min < -1.0 or audio_max > 1.0:
                raise ValueError(f"Floating point audio is not normalized (min: {audio_min} max: {audio_max})")
        int_audio = (audio * 32767).to(torch.int16)
    else:
        raise TypeError(f"Unsupported audio dtype {dt}. "
                        "Supported: float, uint8, int16, int32, int64.")

    # Save
    torchaudio.set_audio_backend("sox_io")
    torchaudio.save(str(path), int_audio, sample_rate=sample_rate, format="wav")


def int16_to_float32(audio: ArrayOrTensor) -> ArrayOrTensor:
    """
    Convert 16-bit PCM audio to normalized float32 in [-1.0, +1.0].

    :param audio: 1D or 2D numpy array or torch tensor of dtype int16.
    :return:      Same shape, dtype float32, values in [-1.0, +1.0].
    """
    if isinstance(audio, np.ndarray):
        if audio.dtype != np.int16:
            raise TypeError(f"Expected numpy array with dtype int16, got {audio.dtype}")
        return audio.astype(np.float32) / 32768.0
    elif isinstance(audio, torch.Tensor):
        if audio.dtype != torch.int16:
            raise TypeError(f"Expected torch Tensor with dtype torch.int16, got {audio.dtype}")
        return audio.to(torch.float32) / 32768.0
    else:
        raise TypeError(f"Unsupported type {type(audio)}; expected numpy.ndarray or torch.Tensor")


def float32_to_int16(audio: ArrayOrTensor) -> ArrayOrTensor:
    """
    Convert normalized float32 audio in [-1.0, +1.0] to 16-bit PCM.

    :param audio: 1D or 2D numpy array or torch tensor of dtype float32.
                  Values are expected in [-1.0, +1.0], but will be clamped if outside.
    :return:      Same shape, dtype int16.
    """
    if isinstance(audio, np.ndarray):
        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError(f"Expected numpy array with float dtype, got {audio.dtype}")
        # Clamp and scale
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
    elif isinstance(audio, torch.Tensor):
        if not audio.dtype.is_floating_point:
            raise TypeError(f"Expected torch Tensor with floating dtype, got {audio.dtype}")
        # Clamp and scale
        clipped = audio.clamp(-1.0, 1.0)
        return (clipped * 32767).to(torch.int16)
    else:
        raise TypeError(f"Unsupported type {type(audio)}; expected numpy.ndarray or torch.Tensor")


def resample_audio(audio_data: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    return F.resample(audio_data, sample_rate, target_sample_rate, lowpass_filter_width=6)
