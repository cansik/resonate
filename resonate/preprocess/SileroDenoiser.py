from typing import Optional, Union, Callable

import numpy as np
import torch


class SileroDenoiser:
    """
    Wrapper for Silero denoise model from snakers4/silero-models.

    Provides methods to denoise numpy arrays or audio files.
    """

    INPUT_SAMPLE_RATE = 24000
    OUTPUT_SAMPLE_RATE = 48000

    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            name: str = "small_slow",
            device: Optional[torch.device] = None
    ):
        # set device
        self.device = device or torch.device("cpu")

        # load model and utilities
        if model is None:
            self.model, self.samples, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_denoise",
                name=name,
                device=self.device,
                verbose=False
            )
        else:
            self.model = model
            # user must set utils manually if custom model
            self.samples = []
            self.utils = None

        # unpack utils if available
        if hasattr(self, "utils") and self.utils is not None:
            self.read_audio, self.save_audio, self._denoise_fn = self.utils
        else:
            self.read_audio = None
            self.save_audio = None
            self._denoise_fn = None

        # ensure model in eval mode
        self.model.eval()

    def denoise_array(
            self,
            audio: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Denoise a waveform given as a numpy array or torch tensor.

        Returns a denoised torch.Tensor on CPU.
        """
        # convert to tensor if needed
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)

        # ensure batch dim and mono channel
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2 and audio.shape[0] != 1:
            # assume (channels, samples)
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(self.device)
        with torch.no_grad():
            denoised = self.model(audio)

        return denoised.squeeze(1).cpu()

    def process_chunked(
            self,
            audio: Union[np.ndarray, torch.Tensor],
            chunk_size: int,
            on_progress: Optional[Callable[[float], None]] = None
    ) -> torch.Tensor:
        """
        Denoise a long waveform (numpy array or torch tensor) in chunks.

        Args:
            audio: 1D numpy array or tensor, or 2D tensor with shape (1, N) or (channels, N).
            chunk_size: number of samples per chunk.
            on_progress: optional callback receiving a float [0.0, 1.0] after each chunk.

        Returns:
            A 1D torch.Tensor containing the full denoised waveform on CPU.
        """
        # to tensor
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)

        # flatten to mono 1D
        if audio.dim() == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            else:
                audio = audio.mean(dim=0)
        elif audio.dim() > 2:
            raise ValueError("Audio tensor must be 1D or 2D")

        total_samples = audio.shape[-1]
        denoised_chunks = []
        processed = 0

        while processed < total_samples:
            end = min(processed + chunk_size, total_samples)
            chunk = audio[processed:end]
            # denoise this chunk
            den_chunk = self.denoise_array(chunk)
            denoised_chunks.append(den_chunk)
            processed = end
            if on_progress:
                on_progress(processed / total_samples)

        # concatenate all chunks
        full = torch.cat(denoised_chunks, dim=-1)
        return full
