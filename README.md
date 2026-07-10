# VoidCLcompute

A lightweight C++ / C-ABI wrapper around OpenCL for elementwise GPU array
math — add, subtract, multiply, divide, and trig functions applied across
large float arrays, without hand-writing OpenCL boilerplate every time.
Also Check voidcl.vercel.app or voidclcompute.vercel.app.

Built by **Void** / **VoidGriefedTeam**.

## Why

OpenCL's raw API is verbose: platform/device discovery, context and queue
setup, program/kernel compilation, buffer management — all before you run a
single line of actual math. VoidCLcompute handles all of that once, and
exposes simple functions:

```cpp
gpu_add(a.data(), b.data(), result.data(), count);
```

## What this is (and isn't)

This is a **bulk-array** compute library. It's for processing thousands to
millions of values in one call — not for one-off scalar math. Dispatching a
GPU kernel to add two single numbers is slower than just doing `a + b` in
plain C++. GPU compute pays off at scale, and/or when the per-element math
is expensive (see `gpu_heavy` below). See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full reasoning.

## Quick example

```cpp
#include "VoidCLcompute.h"
#include <vector>
#include <cstdio>

int main() {
    GC_Init();

    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    std::vector<float> result(4);

    gpu_add(a.data(), b.data(), result.data(), 4);
    // result = {11, 22, 33, 44}

    for (float r : result) printf("%f\n", r);

    GC_Shutdown();
    return 0;
}
```

## API

```c
bool GC_Init();
void GC_Shutdown();
void GC_TrimBufferCache();   // optional: release pooled GPU buffers early

void gpu_add(const float* a, const float* b, float* result, int count);
void gpu_subtract(const float* a, const float* b, float* result, int count);
void gpu_multiply(const float* a, const float* b, float* result, int count);
void gpu_divide(const float* a, const float* b, float* result, int count);

void gpu_sin(const float* input, float* result, int count);
void gpu_cos(const float* input, float* result, int count);
void gpu_tan(const float* input, float* result, int count);
void gpu_asin(const float* input, float* result, int count);  // input must be in [-1, 1]
void gpu_acos(const float* input, float* result, int count);  // input must be in [-1, 1]
void gpu_atan(const float* input, float* result, int count);

void gpu_heavy(const float* a, const float* b, float* result, int count);
```

All ops are blocking — `result` is ready to read the moment the call
returns. `count` tells the library how many floats are in your arrays,
since a raw pointer alone doesn't carry that information.

## Performance notes

- Device buffers are **pooled by size** — repeated calls at the same
  `count` reuse existing GPU memory instead of reallocating.
- Uses pinned/mapped host memory, giving near-zero-copy behavior on
  integrated GPUs.
- Elementwise ops run `float4`-vectorized kernels.
- `sin`/`cos`/`tan`/`heavy` use hardware `native_*` math paths for a large
  speedup, trading a small amount of precision.
- Compiled kernel binaries are cached to disk (`gwcl_kernel_cache/`) —
  first run compiles, subsequent runs load the binary directly.

Full design rationale in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Building (Windows / MSVC)

1. Get the OpenCL SDK — see [third_party/README.md](third_party/README.md)
   for where to place it.
2. Open a **Developer Command Prompt for VS**.
3. From the repo root, run:
   ```
   build.bat
   ```
   This produces `VoidCLcompute.dll` / `.lib` and `benchmark.exe`.

## Benchmark example

`examples/benchmark/app.cpp` compares multi-threaded CPU performance
against `gpu_heavy` across a range of array sizes. On modest hardware (an
Intel Core i3-6100T with integrated HD 530 graphics), the GPU path wins
decisively at scale — sub-millisecond for large arrays — thanks to
dedicated hardware trig units outperforming software `sinf`/`cosf`
approximations on the CPU.

Run it after building:
```
benchmark.exe
```

## License

Mozilla Public License 2.0 (MPL-2.0) — see [LICENSE](LICENSE).

MPL-2.0 is a file-level ("weak") copyleft license: if you modify one of
this project's source files and distribute it, that modified file must
stay under MPL-2.0. You're free to combine this library with proprietary
code in a larger project — the copyleft only applies to VoidCLcompute's
own files, not your whole codebase.
