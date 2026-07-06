// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef VOIDCLCOMPUTE_H
#define VOIDCLCOMPUTE_H

#ifdef VOIDCLCOMPUTE_EXPORTS
    #define VOIDCLCOMPUTE_API __declspec(dllexport)
#else
    #define VOIDCLCOMPUTE_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

VOIDCLCOMPUTE_API bool GC_Init();
VOIDCLCOMPUTE_API void GC_Shutdown();

VOIDCLCOMPUTE_API void gpu_add(const float* a, const float* b, float* result, int count);
VOIDCLCOMPUTE_API void gpu_subtract(const float* a, const float* b, float* result, int count);
VOIDCLCOMPUTE_API void gpu_multiply(const float* a, const float* b, float* result, int count);
VOIDCLCOMPUTE_API void gpu_divide(const float* a, const float* b, float* result, int count);
VOIDCLCOMPUTE_API void gpu_sin(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_cos(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_tan(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_asin(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_acos(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_atan(const float* input, float* result, int count);
VOIDCLCOMPUTE_API void gpu_heavy(const float* a, const float* b, float* result, int count);

// Release cached device buffers without a full context teardown.
VOIDCLCOMPUTE_API void GC_TrimBufferCache();

#ifdef __cplusplus
}
#endif

#endif // VOIDCLCOMPUTE_H
