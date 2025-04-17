import gc
import sys
from typing import Optional

import torch

from resonate.utils.os_utils import is_windows, is_macosx


def get_device_string() -> str:
    device = "cpu"

    if sys.platform.startswith("linux") or is_windows():
        if torch.cuda.is_available():
            device = "cuda"
    elif is_macosx():
        if torch.backends.mps.is_available():
            device = "mps"

    return device


def get_device() -> torch.device:
    return torch.device(get_device_string())


def get_variant() -> Optional[str]:
    if is_macosx():
        return None

    if is_windows() and is_cpu():
        return None

    return "fp16"


def get_dtype() -> torch.dtype:
    if is_macosx():
        return torch.float32

    if is_windows() and is_cpu():
        return torch.float32

    return torch.float16


def is_cpu() -> bool:
    return get_device_string() == "cpu"


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    return torch.backends.mps.is_available()


def clear_memory():
    gc.collect()

    if is_cuda_available():
        torch.cuda.empty_cache()

    if is_mps_available():
        torch.mps.empty_cache()
