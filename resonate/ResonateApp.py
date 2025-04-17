from typing import Tuple, Optional

import gradio as gr
import numpy as np

from resonate.ResonatePipeline import ResonatePipeline


class ResonateApp:
    def __init__(self):
        self.pipeline: Optional[ResonatePipeline] = None

    def transcribe_text(self, audio_input: Tuple[int, np.ndarray], progress=gr.Progress()):
        sample_rate, audio_data_uint16 = audio_input

        if self.pipeline is None:
            progress(0, desc="loading pipeline")
            self.pipeline = ResonatePipeline()

        result = self.pipeline.process(sample_rate, audio_data_uint16, progress)
        return result

    def run(self) -> str:
        transcribe_tab = gr.Interface(fn=self.transcribe_text,
                                      inputs=[
                                          gr.Audio(label="Audio")
                                      ],
                                      outputs=[
                                          gr.Text(label="Text")
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
