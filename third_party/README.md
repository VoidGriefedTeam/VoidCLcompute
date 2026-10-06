# Third-party dependencies

VoidCLcompute needs the OpenCL headers (`CL/cl.h`) and the OpenCL library
(`OpenCL.lib` on Windows, `libOpenCL.so` on Linux) to build.
These are **not** committed to this repo — grab them yourself from one of:

- Khronos OpenCL-SDK releases: https://github.com/KhronosGroup/OpenCL-SDK/releases
- Or your GPU vendor's SDK (Intel, AMD, NVIDIA all ship OpenCL headers/libs)

## Where CMake looks

CMake's stock `FindOpenCL` module checks the usual system and vendor-SDK
locations. In addition, if a `third_party/OpenCL` folder exists it is
searched first, using this layout:

```
third_party/
└── OpenCL/
    ├── include/
    │   └── CL/
    │       └── cl.h
    └── lib/
        └── OpenCL.lib
```

If your SDK lives somewhere else (e.g. `OpenCL-SDK-v2026.05.29-Win-x64`),
no renaming is needed — tell CMake where it is:

```
cmake -B build -DCMAKE_PREFIX_PATH=C:/path/to/OpenCL-SDK-v2026.05.29-Win-x64
```

or point at the pieces directly:

```
cmake -B build -DOpenCL_INCLUDE_DIR=<dir containing CL/> -DOpenCL_LIBRARY=<path to OpenCL.lib>
```

If you change these after a first configure, delete `build/CMakeCache.txt`
(or the whole `build/` folder) so the new paths are picked up.
