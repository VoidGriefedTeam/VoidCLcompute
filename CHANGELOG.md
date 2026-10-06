# Changelog

## Unreleased

- **New:** return-by-value overloads for every op, e.g.
  `std::vector<float> c = gpu_add(a, b);`. They return an empty vector on
  error. The out-parameter versions are kept for reusing storage in hot loops.
- **Breaking:** the public API now takes `std::vector<float>` instead of
  `const float*` + `int count`. `result` is resized to match the input.
  Scalar variants are overloads of the same names (`gpu_add(a, 2.0f, r)`),
  so `VoidCLcompute_overloads.h` and the `gpu_*_scalar` exports are gone.
- Empty input is a no-op; mismatched `a`/`b` sizes in binary ops print an
  error instead of reading out of bounds.
- Build system: `build.bat` replaced by CMake (`CMakeLists.txt`).
- Kernel-cache directory is created with `std::filesystem` instead of
  `<direct.h>`/`_mkdir`, and the export macro handles non-Windows
  compilers, so the library no longer depends on Windows-only headers.

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
