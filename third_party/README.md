# Third-party dependencies

VoidCLcompute needs the OpenCL headers (`CL/cl.h`) and `OpenCL.lib` to build.
These are **not** committed to this repo — grab them yourself from one of:

- Khronos OpenCL-SDK releases: https://github.com/KhronosGroup/OpenCL-SDK/releases
- Or your GPU vendor's SDK (Intel, AMD, NVIDIA all ship OpenCL headers/libs)

## Expected layout

Place the SDK so the paths match what `build.bat` expects:

```
third_party/
└── OpenCL/
    ├── include/
    │   └── CL/
    │       └── cl.h
    └── lib/
        └── OpenCL.lib
```

If your SDK folder is named differently (e.g.
`OpenCL-SDK-v2026.05.29-Win-x64`), either rename it to `OpenCL` or edit the
`OPENCL_INC` / `OPENCL_LIB` variables at the top of `build.bat` to point at
wherever you extracted it.
