from typing import Callable

import numpy as np
import torchaudio.functional as F
from torch import Tensor

from resonate.models.ResonateResult import ResonateResult
from resonate.models.ResonateSegment import ResonateSegment
from resonate.models.ResonateTiming import ResonateTiming
from resonate.stt.TransformersWhisper import TransformersWhisper
from resonate.utils import torch_utils
from resonate.utils.time_utils import samples_to_seconds
from resonate.vad.SileroVAD import SileroVAD
from resonate.vad.VADModels import convert_vad_results_to_segments


class ResonatePipeline:
    DEFAULT_SAMPLE_RATE: int = 16000
    UINT16_NORMALIZE_FACTOR: float = np.iinfo(np.uint16).max / 2

    def __init__(self,
                 model_id: str = "distil-whisper/distil-large-v3.5",
                 language: str | None = None,
                 batch_size: int = 8,
                 vad_min_silence: int = 1000):
        self.torch_device = torch_utils.get_device()

        self.vad = SileroVAD(min_silence_duration_ms=vad_min_silence)
        self.tts = TransformersWhisper(model_id, language, batch_size)

    def process(self,
                sample_rate: int,
                audio_data_uint16: np.ndarray,
                on_progress: Callable[[float, str], None]) -> ResonateResult:
        timing = ResonateTiming()

        with timing.time_block("pre-processing"):
            audio_data_fp32 = audio_data_uint16.astype(np.float32) / self.UINT16_NORMALIZE_FACTOR
            mono_audio_data_fp32 = audio_data_fp32[:, 0].flatten() if len(
                audio_data_fp32.shape) > 1 else audio_data_fp32
            audio_data = Tensor(mono_audio_data_fp32)

            on_progress(0, "resampling")
            input_audio_cpu = F.resample(audio_data, sample_rate, self.DEFAULT_SAMPLE_RATE, lowpass_filter_width=6)
            total_samples = len(input_audio_cpu)

        # vad-analysis
        with timing.time_block("voice activation detection"):
            def on_vad_progress(value: float):
                on_progress(value, "voice activation detection")

            self.vad.reset_states()
            vad_results = self.vad.process(input_audio_cpu, on_progress=on_vad_progress)
            vad_segments = convert_vad_results_to_segments(vad_results, total_samples)

        # transcription
        with timing.time_block("transcription"):
            input_audio = input_audio_cpu.numpy()

            transcriptions: list[ResonateSegment] = []
            for segment in vad_segments:
                chunk = input_audio[segment.start:segment.end]

                on_progress(segment.start / total_samples, "transcription")
                result = self.tts.process(chunk)

                text = result["text"].strip()

                transcriptions.append(
                    ResonateSegment(
                        samples_to_seconds(segment.start, self.DEFAULT_SAMPLE_RATE),
                        samples_to_seconds(segment.end, self.DEFAULT_SAMPLE_RATE),
                        text
                    )
                )

        return ResonateResult(timing, transcriptions)
