import ctypes

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_kernel(x, padded, TOTAL: tl.constexpr, SEQ: tl.constexpr,
                FFT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    col = offsets % FFT
    row = offsets // FFT
    values = tl.load(x + row * SEQ + col,
                     mask=mask & (col < SEQ), other=0.0)
    tl.store(padded + offsets, values, mask=mask)


@triton.jit
def _pad_2d_kernel(x, padded, SEQ: tl.constexpr, PITCH: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < PITCH
    values = tl.load(x + row * SEQ + cols,
                     mask=mask & (cols < SEQ), other=0.0)
    tl.store(padded + row * PITCH + cols, values, mask=mask)


# PyTorch's public FFT normalization is a separate tensor pass on ROCm.
# rocFFT can apply the same scale as part of the transform plan.  Calling the
# library directly also lets the single padding kernel above use a large,
# contiguous launch while retaining rocFFT's numerical implementation.
_rocfft = ctypes.CDLL("librocfft.so.0")
_handle = ctypes.c_void_p
_size = ctypes.c_size_t

_signatures = {
    "rocfft_setup": [],
    "rocfft_plan_description_create": [ctypes.POINTER(_handle)],
    "rocfft_plan_description_destroy": [_handle],
    "rocfft_plan_description_set_scale_factor": [_handle, ctypes.c_double],
    "rocfft_plan_create": [ctypes.POINTER(_handle), ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, _size, ctypes.POINTER(_size), _size,
                           _handle],
    "rocfft_plan_get_work_buffer_size": [_handle, ctypes.POINTER(_size)],
    "rocfft_execution_info_create": [ctypes.POINTER(_handle)],
    "rocfft_execution_info_set_work_buffer": [_handle, _handle, _size],
    "rocfft_execution_info_set_stream": [_handle, _handle],
    "rocfft_execute": [_handle, ctypes.POINTER(_handle),
                       ctypes.POINTER(_handle), _handle],
}
for _name, _args in _signatures.items():
    _function = getattr(_rocfft, _name)
    _function.argtypes = _args
    _function.restype = ctypes.c_int


def _check(status: int):
    if status != 0:
        raise RuntimeError(f"rocFFT error {status}")


_check(_rocfft.rocfft_setup())
_plans = {}


def _get_plan(device: int, rows: int, fft_size: int, inplace: bool,
              stream: int):
    key = (device, rows, fft_size, inplace, stream)
    state = _plans.get(key)
    if state is not None:
        return state

    description = _handle()
    _check(_rocfft.rocfft_plan_description_create(ctypes.byref(description)))
    _check(_rocfft.rocfft_plan_description_set_scale_factor(
        description, 1.0 / fft_size))

    plan = _handle()
    lengths = (_size * 1)(fft_size)
    # real-forward, single precision, one dimension
    _check(_rocfft.rocfft_plan_create(
        ctypes.byref(plan), 0 if inplace else 1, 2, 0, 1, lengths, rows,
        description))
    _check(_rocfft.rocfft_plan_description_destroy(description))

    work_size = _size()
    _check(_rocfft.rocfft_plan_get_work_buffer_size(
        plan, ctypes.byref(work_size)))
    info = _handle()
    _check(_rocfft.rocfft_execution_info_create(ctypes.byref(info)))
    _check(_rocfft.rocfft_execution_info_set_stream(info, _handle(stream)))
    workspace = None
    if work_size.value:
        workspace = torch.empty(work_size.value, dtype=torch.uint8,
                                device=f"cuda:{device}")
        _check(_rocfft.rocfft_execution_info_set_work_buffer(
            info, _handle(workspace.data_ptr()), work_size))

    state = (plan, info, workspace)
    _plans[key] = state
    return state


@torch.no_grad()
def run(x: torch.Tensor):
    seq = x.shape[-1]
    fft_size = 2 * seq
    rows = x.numel() // seq

    # rocFFT's in-place plan is preferable for large batches at these medium
    # transform sizes; outside that region its out-of-place plan is faster.
    inplace = rows >= 8192 and 3000 <= fft_size < 16384
    pitch = fft_size + 2 if inplace else fft_size
    padded = torch.empty((*x.shape[:-1], pitch), dtype=torch.float32,
                         device=x.device)
    block = 1024
    if 1536 <= pitch <= 8192:
        _pad_2d_kernel[(rows, triton.cdiv(pitch, block))](
            x, padded, SEQ=seq, PITCH=pitch, BLOCK=block, num_warps=4)
    else:
        total = padded.numel()
        _pad_kernel[(triton.cdiv(total, block),)](
            x, padded, TOTAL=total, SEQ=seq, FFT=pitch, BLOCK=block,
            num_warps=4)

    stream = torch.cuda.current_stream(x.device).cuda_stream
    plan, info, _workspace = _get_plan(
        x.device.index, rows, fft_size, inplace, stream)
    inputs = (_handle * 1)(padded.data_ptr())
    if inplace:
        _check(_rocfft.rocfft_execute(plan, inputs, None, info))
        output = torch.view_as_complex(
            padded.reshape(*x.shape[:-1], seq + 1, 2))
    else:
        output = torch.empty((*x.shape[:-1], seq + 1),
                             dtype=torch.complex64, device=x.device)
        outputs = (_handle * 1)(output.data_ptr())
        _check(_rocfft.rocfft_execute(plan, inputs, outputs, info))

    return output.real, output.imag
