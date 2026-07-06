# Architecture notes

## Design goals

VoidCLcompute is a thin, C-ABI-compatible wrapper around OpenCL for
elementwise array math on the GPU. It optimizes for the common case: apply
one operation across a large float array, get the result back.

## Buffer pooling

Device buffers are pooled and keyed by byte size rather than allocated fresh
on every call. Repeated calls at the same `count` reuse the same underlying
`cl_mem` objects, avoiding allocator churn — this matters a lot for
benchmark-style loops that call the same op thousands of times at a few
fixed sizes.

Call `GC_TrimBufferCache()` to release pooled buffers early if you're
switching to very different array sizes and want to free GPU memory sooner
than `GC_Shutdown()`.

## Pinned / mapped memory

Buffers are allocated with `CL_MEM_ALLOC_HOST_PTR` and accessed via
`clEnqueueMapBuffer`/`clEnqueueUnmapMemObject` rather than plain
`clEnqueueWriteBuffer`/`clEnqueueReadBuffer`. On integrated GPUs (shared
system memory), this is close to genuine zero-copy. On discrete GPUs it
still avoids an extra internal staging-buffer copy the driver would
otherwise perform.

## Vectorization

Elementwise ops (`add`, `subtract`, `multiply`, `divide`, and the trig
functions) run as `float4`-vectorized kernels — each work-item processes 4
elements via `vload4`/`vstore4`, with a scalar tail path for counts not
divisible by 4.

`gpu_heavy` is deliberately scalar. Its per-element work involves a
50-iteration loop of transcendental math, which doesn't vectorize cleanly
across vendors' math hardware paths — profiling showed no consistent win
from float4 there.

## Math precision

`sin`, `cos`, `tan`, and the `heavy` kernel use `native_*` OpenCL built-ins,
which map to dedicated Special Function Units on the GPU. These trade some
precision (~22-bit vs full 32-bit float) for a large speed win — appropriate
for bulk numerical/graphics work, not for applications needing IEEE-exact
trig results.

`asin`/`acos`/`atan` use the standard (non-`native_`) built-ins, since most
vendors don't expose hardware-accelerated inverse trig. Note `asin`/`acos`
are only valid for inputs in `[-1, 1]`; out-of-range inputs return `NaN`.

## Work-group sizing

Local work size is queried per-kernel via
`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` rather than hardcoded,
falling back to 64 if the query returns 0.

## Queue and synchronization

The command queue is created with
`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` where supported (falls back to an
in-order queue otherwise), allowing independent enqueued work to overlap on
the device.

Kernel completion is awaited via `clWaitForEvents` on that specific
kernel's event, not `clFinish` on the whole queue — this avoids stalling
unrelated in-flight work when multiple ops are in play.

## Kernel binary caching

Compiled kernel binaries are cached to disk under `gwcl_kernel_cache/`,
keyed by device name and operation ID. On a cache hit, `clCreateProgramWithBinary`
is used instead of `clCreateProgramWithSource` + `clBuildProgram` from
scratch, skipping the JIT compile step on subsequent runs.

If you update your GPU driver, delete `gwcl_kernel_cache/` manually — some
drivers silently accept incompatible cached binaries rather than rejecting
them outright.

## What this library is not

This is a bulk-array compute library, not a general-purpose calculator.
Dispatching a GPU kernel for a single scalar operation (e.g. adding two
numbers) is slower than doing it on the CPU directly — the overhead of
kernel dispatch and buffer mapping dwarfs the cost of one arithmetic
operation. The GPU path pays off once you're processing thousands to
millions of elements per call, or doing expensive per-element math
(like `gpu_heavy`'s trig loop) where dedicated hardware paths give a large
per-element speedup.
