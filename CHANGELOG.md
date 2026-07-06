# Changelog

## v0.1.0 — Initial release

- Core API: `GC_Init`, `GC_Shutdown`, `GC_TrimBufferCache`
- Elementwise binary ops: `gpu_add`, `gpu_subtract`, `gpu_multiply`, `gpu_divide`
- Elementwise unary ops: `gpu_sin`, `gpu_cos`, `gpu_tan`, `gpu_asin`, `gpu_acos`, `gpu_atan`
- `gpu_heavy` — compound multi-iteration benchmark kernel (sin/cos/sqrt loop)
- float4-vectorized kernels for all ops except `gpu_heavy` (scalar by design)
- `native_*` hardware math paths for `sin`/`cos`/`tan`/`heavy`
- Device buffer pooling keyed by byte size (no per-call alloc/free)
- Pinned/mapped host memory (`CL_MEM_ALLOC_HOST_PTR`) for near-zero-copy on
  integrated GPUs
- Per-kernel work-group sizing via `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`
- Out-of-order command queue (falls back to in-order if unsupported)
- Event-based kernel completion waits (no full-queue `clFinish` stalls)
- Disk-cached compiled kernel binaries (`gwcl_kernel_cache/`) — skips
  recompilation on subsequent runs
- `-cl-fast-relaxed-math -cl-mad-enable` build flags
- `/arch:AVX2` on the CPU-side build (benchmark comparison path)
- Included `benchmark` example comparing CPU multi-threaded vs GPU performance
