from typing import Callable

import numpy as np
import torchaudio.functional as F
from torch import Tensor

from resonate.stt.TransformersWhisper import TransformersWhisper
from resonate.utils import torch_utils
from resonate.vad.SileroVAD import SileroVAD
from resonate.vad.VADModels import convert_vad_results_to_segments


class ResonatePipeline:
    DEFAULT_SAMPLE_RATE: int = 16000
    UINT16_NORMALIZE_FACTOR: float = np.iinfo(np.uint16).max / 2

    def __init__(self):
        self.torch_device = torch_utils.get_device()

        self.vad = SileroVAD()
        self.tts = TransformersWhisper()

    def process(self, sample_rate: int, audio_data_uint16: np.ndarray,
                on_progress: Callable[[float, str], None]) -> str:
        audio_data_fp32 = audio_data_uint16.astype(np.float32) / self.UINT16_NORMALIZE_FACTOR
        mono_audio_data_fp32 = audio_data_fp32[:, 0].flatten()
        audio_data = Tensor(mono_audio_data_fp32)

        on_progress(0, "resampling")
        input_audio_cpu = F.resample(audio_data, sample_rate, self.DEFAULT_SAMPLE_RATE, lowpass_filter_width=6)

        # vad-analysis
        def on_vad_progress(value: float):
            on_progress(value, "voice activation detection")

        self.vad.reset_states()
        vad_results = self.vad.process(input_audio_cpu, on_progress=on_vad_progress)
        vad_segments = convert_vad_results_to_segments(vad_results, len(input_audio_cpu))

        # transcription
        input_audio = input_audio_cpu.numpy()
        chunks = []
        for segment in vad_segments:
            chunks.append(input_audio[segment.start:segment.end])

        transcriptions = []
        for i, chunk in enumerate(chunks):
            on_progress(i / len(chunks), "transcription")
            result = self.tts.process(chunk)
            transcriptions.append(result["text"])

        return " ".join(transcriptions).strip()
