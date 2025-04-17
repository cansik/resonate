from pathlib import Path
from typing import Set

import requests
import webview


class ResonateWindow:

    def __init__(self, url: str):
        webview.settings["ALLOW_DOWNLOADS"] = True
        self.main_window = webview.create_window("Resonate", url, width=1024, height=800)

        self.main_window.events.loaded += self._on_loaded

        # download related
        self.download_folder: Path = Path.home() / "Downloads"
        self.allowed_download_formats: Set[str] = {".jpg", ".jpeg", ".png"}

    def _on_loaded(self):
        url: str = self.main_window.get_current_url()

        # allow specific files to be downloaded
        if any([url.endswith(e) for e in self.allowed_download_formats]):
            suffix = url.split(".")[-1]
            self._download_file(url, "resonate-{:04d}." + suffix)
            self.main_window.evaluate_js("history.back()")

    def _download_file(self, url: str, file_name_template: str):
        # find free filename
        file_id = 0
        file_name = file_name_template.format(file_id)
        while self.download_folder.joinpath(file_name).exists():
            file_id += 1
            file_name = file_name_template.format(file_id)

        r = requests.get(url, allow_redirects=True)
        open(self.download_folder.joinpath(file_name), 'wb').write(r.content)

    def open(self):
        webview.start()
