from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import numpy as np

from resonate.ResonatePipeline import ResonatePipeline
from resonate.utils.output_utils import segments_to_csv
from resonate.utils.time_utils import format_timestamp

model_options: dict[str, str] = {
    "whisper-large-v3-turbo-german": "primeline/whisper-large-v3-turbo-german",
    "whisper-tiny-german-1224": "primeline/whisper-tiny-german-1224",
    "whisper-large-v3-turbo-swiss-german": "Flurin17/whisper-large-v3-turbo-swiss-german",
    "whisper-distil-large-v3.5": "distil-whisper/distil-large-v3.5"
}

language_options: dict[str, str | None] = {
    "Auto": None,
    "German": "de",
    "English": "en",
    "French": "fr"
}


class ResonateApp:
    def __init__(self):
        self.pipeline: Optional[ResonatePipeline] = None

    def transcribe_text(self,
                        audio_input: Tuple[int, np.ndarray],
                        model_key: str,
                        language_key: str,
                        batch_size: int,
                        use_denoise: bool,
                        vad_min_silence: int,
                        progress=gr.Progress()):
        model_id = model_options[model_key]
        language = language_options[language_key]

        sample_rate, audio_data_uint16 = audio_input

        progress(0, desc="loading pipeline")

        self.pipeline = ResonatePipeline(model_id, language, batch_size, use_denoise, vad_min_silence)

        # post-process result
        result = self.pipeline.process(sample_rate, audio_data_uint16, progress)

        # text
        text = "\n".join([f"[{format_timestamp(s.start_ts)}-{format_timestamp(s.end_ts)}] {s.text}"
                          for s in result.segments])
        # csv
        csv_text = segments_to_csv(result.segments)

        temp_file_dir = Path("./temp")
        temp_file_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_file_dir / "segments.csv"
        temp_file.write_text(csv_text, encoding="utf-8")

        return result.timing.to_dict(), text, str(temp_file)

    def run(self) -> str:
        transcribe_tab = gr.Interface(fn=self.transcribe_text,
                                      inputs=[
                                          gr.Audio(label="Audio"),
                                          gr.Dropdown(label="Model", choices=list(model_options.keys())),
                                          gr.Dropdown(label="Language", choices=list(language_options.keys()),
                                                      value="German"),
                                          gr.Dropdown(label="Batch Size", choices=[1, 2, 4, 8, 16, 32], value=8),
                                          gr.Checkbox(label="Denoise", value=False),
                                          gr.Number(label="VAD Min Silence (ms)", value=1000)
                                      ],
                                      outputs=[
                                          gr.Json(label="Timing"),
                                          gr.Text(label="Text", lines=10, show_copy_button=True),
                                          gr.File(label="CSV")
                                      ],
                                      flagging_mode="never",
                                      clear_btn=gr.Button(visible=False),
                                      submit_btn="Transcribe",
                                      analytics_enabled=False,
                                      concurrency_limit=1)

        demo = gr.TabbedInterface([
            transcribe_tab
        ], [
            "Transcribe",
        ],
            title="Resonate",
            analytics_enabled=False,
            css="footer{display:none !important}",
            theme=gr.themes.Ocean())

        demo.enable_queue = True
        app, local_url, shared_url = demo.launch(prevent_thread_lock=True, quiet=True)
        return local_url
