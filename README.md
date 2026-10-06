# VoidCLcompute

A lightweight C++ wrapper around OpenCL for elementwise GPU array
math — add, subtract, multiply, divide, and trig functions applied across
large `std::vector<float>`s, without hand-writing OpenCL boilerplate every time.
Also Check [voidcl.vercel.app](https://voidcl.vercel.app) or [voidclcompute.vercel.app](https://voidclcompute.vercel.app).

Built by **Void** / **VoidGriefedTeam**.

## Why

OpenCL's raw API is verbose: platform/device discovery, context and queue
setup, program/kernel compilation, buffer management — all before you run a
single line of actual math. VoidCLcompute handles all of that once, and
exposes simple functions:

```cpp
std::vector<float> result = gpu_add(a, b);
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

    std::vector<float> sum = gpu_add(a, b);
    // sum = {11, 22, 33, 44}

    std::vector<float> result = gpu_multiply(sum, 0.5f);
    // result = {5.5, 11, 16.5, 22}  (array * scalar)

    for (float r : result) printf("%f\n", r);

    GC_Shutdown();
    return 0;
}
```

## API

```cpp
bool GC_Init();
void GC_Shutdown();
void GC_TrimBufferCache();   // optional: release pooled GPU buffers early

using Vec = std::vector<float>;

// ---- Return-by-value (preferred): Vec c = gpu_add(a, b); ----
// array (op) array — a and b must be the same size
Vec gpu_add     (const Vec& a, const Vec& b);
Vec gpu_subtract(const Vec& a, const Vec& b);
Vec gpu_multiply(const Vec& a, const Vec& b);
Vec gpu_divide  (const Vec& a, const Vec& b);

// array (op) single number
Vec gpu_add     (const Vec& a, float scalar);
Vec gpu_subtract(const Vec& a, float scalar);
Vec gpu_multiply(const Vec& a, float scalar);
Vec gpu_divide  (const Vec& a, float scalar);

Vec gpu_sin (const Vec& input);
Vec gpu_cos (const Vec& input);
Vec gpu_tan (const Vec& input);
Vec gpu_asin(const Vec& input);   // input must be in [-1, 1]
Vec gpu_acos(const Vec& input);   // input must be in [-1, 1]
Vec gpu_atan(const Vec& input);

Vec gpu_heavy(const Vec& a, const Vec& b);

// ---- Out-parameter versions (reuse an existing vector's storage in hot loops) ----
// array (op) array — a and b must be the same size
void gpu_add     (const Vec& a, const Vec& b, Vec& result);
void gpu_subtract(const Vec& a, const Vec& b, Vec& result);
void gpu_multiply(const Vec& a, const Vec& b, Vec& result);
void gpu_divide  (const Vec& a, const Vec& b, Vec& result);

// array (op) single number — same names, picked by the 2nd argument's type
void gpu_add     (const Vec& a, float scalar, Vec& result);
void gpu_subtract(const Vec& a, float scalar, Vec& result);
void gpu_multiply(const Vec& a, float scalar, Vec& result);
void gpu_divide  (const Vec& a, float scalar, Vec& result);

void gpu_sin (const Vec& input, Vec& result);
void gpu_cos (const Vec& input, Vec& result);
void gpu_tan (const Vec& input, Vec& result);
void gpu_asin(const Vec& input, Vec& result);  // input must be in [-1, 1]
void gpu_acos(const Vec& input, Vec& result);  // input must be in [-1, 1]
void gpu_atan(const Vec& input, Vec& result);

void gpu_heavy(const Vec& a, const Vec& b, Vec& result);
```

All ops are blocking — the result is ready to read the moment the call
returns. The element count comes from the input vectors.

Return-by-value: on error (empty input, size mismatch between `a` and `b`)
you get an empty vector back.

Out-parameter versions: `result` is resized to match the input, so you
don't need to pre-size it (but you can, to avoid a reallocation in hot
loops). `result` may be the same vector as an input. Empty input is a no-op,
and a size mismatch prints an error and leaves `result` untouched.

> **ABI note:** these functions pass `std::vector` across the DLL boundary,
> so the library and your program must be built with the same compiler,
> standard library and configuration (e.g. both Release). Building through
> this repo's CMake project guarantees that. `GC_Init` / `GC_Shutdown` /
> `GC_TrimBufferCache` remain plain `extern "C"` functions.

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

## Building

Requires CMake 3.16+, a C++17 compiler, and an OpenCL SDK (headers + the
OpenCL library) — see [third_party/README.md](third_party/README.md) for
where to put it or how to point CMake at it.

```
cmake -B build
cmake --build build --config Release
```

This produces `VoidCLcompute.dll` (`libVoidCLcompute.so` on Linux) and the
`benchmark` executable, both in `build/` (or `build/Release/` with the
Visual Studio generator).

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `-DVOIDCL_BUILD_BENCHMARK=OFF` | `ON` | Skip the benchmark example |
| `-DVOIDCL_ENABLE_AVX2=OFF` | `ON` | Don't compile with AVX2 (for CPUs without it) |
| `-DCMAKE_PREFIX_PATH=<sdk>` | — | Where to find an OpenCL SDK installed elsewhere |

## Benchmark example

`examples/benchmark/app.cpp` compares multi-threaded CPU performance
against `gpu_heavy` across a range of array sizes. On modest hardware (an
Intel Core i3-6100T with integrated HD 530 graphics), the GPU path wins
decisively at scale — sub-millisecond for large arrays — thanks to
dedicated hardware trig units outperforming software `sinf`/`cosf`
approximations on the CPU.

Run it after building (from the folder containing the built binaries):
```
benchmark
```

## License

Mozilla Public License 2.0 (MPL-2.0) — see [LICENSE](LICENSE).

MPL-2.0 is a file-level ("weak") copyleft license: if you modify one of
this project's source files and distribute it, that modified file must
stay under MPL-2.0. You're free to combine this library with proprietary
code in a larger project — the copyleft only applies to VoidCLcompute's
own files, not your whole codebase.
