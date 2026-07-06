#include "../../include/VoidCLcompute.h"
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================
// HEAVY WORKLOAD (kept identical on CPU & GPU for fair timing)
// ============================================================
static inline float heavy(float x, float y) {
    for (int i = 0; i < 50; i++) {
        x = sinf(x) * cosf(y) + sqrtf(fabsf(x) + 1.0f);
        y = sinf(y) + cosf(x);
    }
    return x + y;
}

// ============================================================
// CPU Multi-Thread
// ============================================================
static void worker(const float* a, const float* b, float* out, size_t s, size_t e) {
    for (size_t i = s; i < e; i++) {
        out[i] = heavy(a[i], b[i]);
    }
}

static double cpuMT(const std::vector<float>& a,
                     const std::vector<float>& b,
                     std::vector<float>& out) {
    unsigned int threads = std::thread::hardware_concurrency();
    if (!threads) threads = 4;

    std::vector<std::thread> pool;
    pool.reserve(threads);

    size_t n = a.size();
    size_t chunk = (n + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned int t = 0; t < threads; t++) {
        size_t s = t * chunk;
        size_t e = std::min(s + chunk, n);
        if (s >= e) break;
        pool.emplace_back(worker, a.data(), b.data(), out.data(), s, e);
    }

    for (auto& th : pool) th.join();

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// GPU using the optimized heavy kernel (pooled + pinned buffers)
// ============================================================
static double gpu(const std::vector<float>& a,
                   const std::vector<float>& b,
                   std::vector<float>& out) {
    auto start = std::chrono::high_resolution_clock::now();

    gpu_heavy(a.data(), b.data(), out.data(), (int)a.size());

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// MAIN
// ============================================================
int main() {
    printf("CPU threads: %u\n", std::thread::hardware_concurrency());

    if (!GC_Init()) {
        printf("GPU init failed\n");
        return -1;
    }

    // GPU warmup — pre-builds kernel, pre-warms the 1024-element pool
    // entry, so the first timed size isn't paying compile/alloc cost.
    {
        std::vector<float> a(1024, 1.0f), b(1024, 2.0f), r(1024);
        gpu_heavy(a.data(), b.data(), r.data(), 1024);
    }

    const std::vector<size_t> sizes = {
        10000,
        100000,
        1000000,
        5000000,
        20000000
    };

    printf("\n%-12s %-14s %-14s %-10s\n",
           "Count", "CPU-MT", "GPU", "Winner");
    printf("-------------------------------------------------------------\n");

    for (size_t n : sizes) {
        std::vector<float> a(n), b(n), r2(n), r3(n);

        for (size_t i = 0; i < n; i++) {
            a[i] = (float)i * 0.001f;
            b[i] = (float)i * 0.002f;
        }

        // Warm CPU cache lines without polluting timed results.
        std::vector<float> dummy(std::min(n, (size_t)10000));
        worker(a.data(), b.data(), dummy.data(), 0, dummy.size());

        double t2 = cpuMT(a, b, r2);
        double t3 = gpu(a, b, r3);

        const char* win = "CPU-MT";
        double best = t2;
        if (t3 < best) { best = t3; win = "GPU"; }

        printf("%-12zu %-14.3f %-14.3f %-10s\n", n, t2, t3, win);
    }

    GC_TrimBufferCache();
    GC_Shutdown();
    printf("\nDone.\n");
    return 0;
}
