"""Kernels of the bench: CUDA sources compiled by nvcc and launched through the driver API.

torch's own extension build does not work here - nvcc 12.5 cannot compile the headers of a torch
built against CUDA 12.8 - so a kernel is compiled to PTX without any torch header and launched with
cuLaunchKernel on torch's own memory. The PTX is cached next to the source and rebuilt when the
source is newer.

Invariant: a kernel runs in torch's context - the caller must have allocated on the device first,
which every path through the bench does.
Invariant: the compiled PTX matches the source it was built from; a stale cache is rebuilt, not used.
"""

from __future__ import annotations

import ctypes
import functools
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).parent
ARCH = "sm_86"  # RTX 3090 Ti; the bench pins one card (CLAUDE.md)
_VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")


class KernelError(RuntimeError):
    """A driver call failed; carries the driver's own message."""


@functools.cache
def _driver() -> ctypes.CDLL:
    return ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")


def _check(name: str, code: int) -> None:
    if code == 0:
        return
    message = ctypes.c_char_p()
    _driver().cuGetErrorString(code, ctypes.byref(message))
    raise KernelError(f"{name} failed with {code}: {message.value.decode() if message.value else 'unknown'}")


def compile_ptx(source: Path, arch: str = ARCH) -> Path:
    """Compile a .cu file to PTX with nvcc, reusing the cached PTX while it is newer than the source."""
    ptx = source.with_suffix(".ptx")
    if ptx.exists() and ptx.stat().st_mtime >= source.stat().st_mtime:
        return ptx
    command = ["nvcc", "-ptx", f"-arch={arch}", str(source), "-o", str(ptx)]
    # nvcc needs a host compiler on PATH; on Windows that means the MSVC environment
    if os.name == "nt" and _VCVARS.exists():
        joined = subprocess.list2cmdline(command)
        result = subprocess.run(f'call "{_VCVARS}" >nul && {joined}', shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise KernelError(f"nvcc failed: {result.stderr.strip() or result.stdout.strip()}")
    return ptx


@functools.cache
def load(name: str, source: Path | None = None) -> ctypes.c_void_p:
    """The handle of a kernel `name`, from the source of the same name in this package."""
    source = source or HERE / f"{name}.cu"
    torch.empty(1, device="cuda")  # torch's context has to be current before the driver API is used
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    _check("cuModuleLoad", _driver().cuModuleLoad(ctypes.byref(module), str(compile_ptx(source)).encode()))
    _check("cuModuleGetFunction", _driver().cuModuleGetFunction(ctypes.byref(function), module, name.encode()))
    return function


def _argument(value) -> ctypes.c_void_p:
    holder = ctypes.c_void_p(value.data_ptr()) if isinstance(value, torch.Tensor) else ctypes.c_int(value)
    _argument.keep.append(holder)  # the array holds pointers, so the values must outlive the call
    return ctypes.cast(ctypes.byref(holder), ctypes.c_void_p)


_argument.keep = []


def launch(function: ctypes.c_void_p, grid: tuple[int, int, int], block: tuple[int, int, int], *args) -> None:
    """Launch a kernel on the current stream with tensors and ints as arguments."""
    _argument.keep = []
    packed = (ctypes.c_void_p * len(args))(*[_argument(a) for a in args])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check("cuLaunchKernel", _driver().cuLaunchKernel(function, *grid, *block, 0, stream, packed, None))
