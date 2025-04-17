from typing import Any, Dict

import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from resonate.utils import torch_utils


class TransformersWhisper:

    def __init__(self, model_id: str = "primeline/whisper-large-v3-turbo-german"):
        self.model_id = model_id

        self.torch_device = torch_utils.get_device()
        self.torch_dtype = torch_utils.get_dtype(allow_float_16=True)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.torch_device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.torch_device,
            generate_kwargs={"max_new_tokens": 128, "language": "de"}
        )

    def process(self, audio_data_fp32: np.ndarray) -> Dict[str, Any]:
        return self.pipe(audio_data_fp32)
