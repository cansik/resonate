from typing import Tuple

import gradio as gr
import numpy as np


class ResonateApp:
    def __init__(self):
        pass

    def transcribe_text(self, audio_input: Tuple[int, np.ndarray], progress=gr.Progress()):
        sample_rate, audio_data_uint16 = audio_input

        # load model if necessary
        progress(0, desc="loading generator")
        print(sample_rate)
        return "hello world"

    def run(self) -> str:
        transcribe_tab = gr.Interface(fn=self.transcribe_text,
                                      inputs=[
                                          gr.Audio(label="Audio")
                                      ],
                                      outputs=[
                                          gr.Text(label="Text")
                                      ],
                                      allow_flagging="never",
                                      clear_btn=gr.Button(visible=False),
                                      submit_btn="Transcribe",
                                      analytics_enabled=False)

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
