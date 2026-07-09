// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// ============================================================
// VoidCLcompute_overloads.h
//
// Optional C++-only convenience layer. The core library is C-ABI
// (extern "C"), so gpu_add(array, array, ...) and the true scalar
// version gpu_add_scalar(array, float, ...) are necessarily separate
// exported symbols underneath. This header lets C++ callers write
// the SAME function name for both:
//
//     gpu_add(array1, array2, result, count);   // array + array
//     gpu_add(array1, someFloat, result, count); // array + scalar
//
// The compiler picks the right overload based on the second argument's
// type — no code changes needed at the call site either way, and no
// runtime cost: this just resolves to the correct extern "C" function
// at compile time.
//
// Include this INSTEAD of VoidCLcompute.h in C++ translation units
// where you want this convenience. It includes VoidCLcompute.h itself,
// so you don't need to include both.
// ============================================================
#ifndef VOIDCLCOMPUTE_OVERLOADS_H
#define VOIDCLCOMPUTE_OVERLOADS_H

#include "VoidCLcompute.h"

#ifdef __cplusplus

// --- gpu_add ---
inline void gpu_add(const float* a, const float* b, float* result, int count) {
    ::gpu_add(a, b, result, count); // array + array (existing symbol)
}
inline void gpu_add(const float* a, float scalar, float* result, int count) {
    gpu_add_scalar(a, scalar, result, count); // array + scalar
}

// --- gpu_subtract ---
inline void gpu_subtract(const float* a, const float* b, float* result, int count) {
    ::gpu_subtract(a, b, result, count);
}
inline void gpu_subtract(const float* a, float scalar, float* result, int count) {
    gpu_subtract_scalar(a, scalar, result, count);
}

// --- gpu_multiply ---
inline void gpu_multiply(const float* a, const float* b, float* result, int count) {
    ::gpu_multiply(a, b, result, count);
}
inline void gpu_multiply(const float* a, float scalar, float* result, int count) {
    gpu_multiply_scalar(a, scalar, result, count);
}

// --- gpu_divide ---
inline void gpu_divide(const float* a, const float* b, float* result, int count) {
    ::gpu_divide(a, b, result, count);
}
inline void gpu_divide(const float* a, float scalar, float* result, int count) {
    gpu_divide_scalar(a, scalar, result, count);
}

#endif // __cplusplus
#endif // VOIDCLCOMPUTE_OVERLOADS_H
