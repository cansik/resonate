import locale
import os

os.environ.setdefault("LANG", "en_US.UTF-8")

try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    # on some macs you might need a different name, e.g. "en_US"
    pass

import multiprocessing
import os
import sys

from resonate.ResonateApp import ResonateApp
from resonate.ResonateWindow import ResonateWindow

# redirect stdout and stderr
# fixes https://github.com/huggingface/diffusers/issues/3290
# todo: catch the streams and display in ui
if sys.stdout is None or sys.stderr is None:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

from resonate.utils import torch_utils


def main():
    # Pyinstaller fix
    multiprocessing.freeze_support()

    app = ResonateApp()
    local_url = app.run()

    print(local_url)

    window = ResonateWindow(local_url)
    window.open()

    torch_utils.clear_memory()


if __name__ == "__main__":
    main()
