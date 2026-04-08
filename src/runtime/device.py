"""Device selection with graceful fallback for unsupported CUDA builds."""

from __future__ import annotations

import warnings
from typing import Any

import torch


def _supported_arches() -> set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return {arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")}
        except Exception:
            return set()


def describe_cuda_support() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cuda_available = bool(torch.cuda.is_available())

    info: dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_name": None,
        "device_cc": None,
        "supported_arches": sorted(_supported_arches()),
        "supported": False,
    }
    if not info["cuda_available"]:
        return info

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        major, minor = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)

    cc = f"sm_{major}{minor}"
    info["device_name"] = device_name
    info["device_cc"] = cc

    supported_arches = set(info["supported_arches"])
    if not supported_arches:
        info["supported"] = True
    else:
        info["supported"] = cc in supported_arches
    return info


def choose_device() -> tuple[torch.device, str | None]:
    info = describe_cuda_support()
    if not info["cuda_available"]:
        return torch.device("cpu"), None
    if info["supported"]:
        return torch.device("cuda"), None

    supported = ", ".join(info["supported_arches"]) or "unknown"
    reason = (
        f"CUDA build is incompatible with GPU {info['device_name']} ({info['device_cc']}). "
        f"Installed PyTorch supports: {supported}. Falling back to CPU."
    )
    return torch.device("cpu"), reason
