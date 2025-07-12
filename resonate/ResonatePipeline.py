import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from torch import Tensor

from resonate.models.ResonateResult import ResonateResult
from resonate.models.ResonateSegment import ResonateSegment
from resonate.models.ResonateTiming import ResonateTiming
from resonate.preprocess.SileroDenoiser import SileroDenoiser
from resonate.stt.TransformersWhisper import TransformersWhisper
from resonate.utils import audio_utils
from resonate.utils import torch_utils
from resonate.utils.time_utils import samples_to_seconds
from resonate.vad.SileroVAD import SileroVAD
from resonate.vad.VADModels import convert_vad_results_to_segments


class ResonatePipeline:
    DEFAULT_SAMPLE_RATE: int = 16000

    def __init__(self,
                 model_id: str = "distil-whisper/distil-large-v3.5",
                 language: str | None = None,
                 batch_size: int = 8,
                 use_denoise: bool = False,
                 vad_min_silence: int = 1000):
        self.torch_device = torch_utils.get_device()

        self.use_denoise = use_denoise

        self.denoiser: SileroDenoiser | None = None
        if self.use_denoise:
            self.denoiser = SileroDenoiser(device=self.torch_device)

        self.vad = SileroVAD(min_silence_duration_ms=vad_min_silence)
        self.tts = TransformersWhisper(model_id, language, batch_size)

        self.debug = False
        self.debug_output_path = Path("temp/debug")

    def process(self,
                audio_sample_rate: int,
                audio_data_uint16: np.ndarray,
                on_progress: Callable[[float, str], None]) -> ResonateResult:
        if self.debug:
            if self.debug_output_path.exists():
                shutil.rmtree(self.debug_output_path)
            self.debug_output_path.mkdir(parents=True, exist_ok=True)

        timing = ResonateTiming()

        with timing.time_block("pre-processing"):
            # todo: handle bits-per-sample in WAV
            on_progress(0, "pre-processing")

            # convert to fp32 normalized
            audio_data_fp32 = audio_utils.int16_to_float32(audio_data_uint16)

            # extract mono channel
            mono_audio_data_fp32 = audio_data_fp32
            if len(audio_data_fp32.shape) > 1:
                mono_audio_data_fp32 = audio_data_fp32[:, 0].flatten()
            audio_data = Tensor(mono_audio_data_fp32)
            on_progress(1, "pre-processing")

        sample_rate = audio_sample_rate

        if self.debug:
            audio_utils.save_wav(self.debug_output_path / "original.wav", audio_data, sample_rate)

        if self.use_denoise:
            with timing.time_block("denoising"):
                def on_denoise_progress(value: float):
                    on_progress(value, "denoising")

                audio_data_sr24 = audio_utils.resample_audio(audio_data, sample_rate, SileroDenoiser.INPUT_SAMPLE_RATE)
                audio_data_sr24 *= 0.95
                audio_data = self.denoiser.process_chunked(audio_data_sr24,
                                                           SileroDenoiser.INPUT_SAMPLE_RATE * 1,
                                                           on_progress=on_denoise_progress)
                audio_data = audio_data[0]
                sample_rate = SileroDenoiser.OUTPUT_SAMPLE_RATE

                if self.debug:
                    audio_utils.save_wav(self.debug_output_path / "denoised.wav", audio_data, sample_rate)

        with timing.time_block("re-sampling"):
            on_progress(0, "re-sampling")
            input_audio_cpu = audio_utils.resample_audio(audio_data, sample_rate, self.DEFAULT_SAMPLE_RATE)
            total_samples = len(input_audio_cpu)
            on_progress(1, "re-sampling")

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
            for i, segment in enumerate(vad_segments):
                chunk = input_audio[segment.start:segment.end]

                on_progress(i / len(vad_segments), "transcription")
                result = self.tts.process(chunk)

                text = result["text"].strip()

                transcriptions.append(
                    ResonateSegment(
                        samples_to_seconds(segment.start, self.DEFAULT_SAMPLE_RATE),
                        samples_to_seconds(segment.end, self.DEFAULT_SAMPLE_RATE),
                        text
                    )
                )

                if self.debug:
                    segment_path = self.debug_output_path / f"segment_{i:04d}.wav"
                    audio_utils.save_wav(segment_path, chunk, self.DEFAULT_SAMPLE_RATE, auto_clamp=True)
                    segment_path.with_suffix(".txt").write_text(text, encoding="utf-8")

        return ResonateResult(timing, transcriptions)
