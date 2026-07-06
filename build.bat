@echo off
REM ============================================================
REM VoidCLcompute build script
REM Run from a Developer Command Prompt for VS.
REM
REM Requires the OpenCL SDK (headers + OpenCL.lib) - see
REM third_party/README.md for where to get it.
REM Adjust OPENCL_INC / OPENCL_LIB below to match your setup.
REM ============================================================

set OPENCL_INC=third_party\OpenCL\include
set OPENCL_LIB=third_party\OpenCL\lib

set PROJ_INC=include
set SRC_DIR=src
set EXAMPLE_DIR=examples\benchmark

REM /O2   - optimize for speed
REM /GL   - whole program optimization (paired with /LTCG at link)
REM /DNDEBUG - strip debug-only asserts in release build
REM /arch:AVX2 - vectorize CPU-side float math (heavy() also runs on
REM              CPU in the benchmark, for comparison against the GPU path)

cl.exe /EHsc /MD /LD /O2 /GL /arch:AVX2 /DNDEBUG ^
    /I %OPENCL_INC% /I %PROJ_INC% ^
    %SRC_DIR%\VoidCLcompute.cpp ^
    /link /LTCG /LIBPATH:%OPENCL_LIB% OpenCL.lib ^
    /OUT:VoidCLcompute.dll /IMPLIB:VoidCLcompute.lib

if %ERRORLEVEL% NEQ 0 (
    echo DLL build FAILED
    exit /b 1
)

cl.exe /EHsc /MD /O2 /GL /arch:AVX2 /DNDEBUG ^
    /I %OPENCL_INC% /I %PROJ_INC% ^
    %EXAMPLE_DIR%\app.cpp ^
    /link /LTCG /LIBPATH:. VoidCLcompute.lib /OUT:benchmark.exe

if %ERRORLEVEL% NEQ 0 (
    echo Benchmark build FAILED
    exit /b 1
)

echo Build succeeded. Run benchmark.exe.
