# NISAR Data Reader

An open-source exploratory visualizer for (NASA-ISRO Synthetic Aperture Radar) NISAR data packages. Reads GSLC/GCOV HDF5 files in seconds rather than minutes with Vulkan, CUDA acceleration, and an automatic CPU fallback.

![NISAR Data Reader displaying a GCOV product](docs/images/nisar-data-reader.png)

[Download NISAR Data Reader v0.6.0 for Windows](https://github.com/LynnAnalytics/NISAR-Data-Reader/releases/tag/v0.6.0)

## Highlights

- Continuous cursor-anchored zoom, drag panning, rotation, and Fit Scene.
- Exact native-resolution views covering 1x1, 2x2, or 4x4 source chunks.
- Guarded screen-space LOD pages for smooth navigation across large products.
- GSLC amplitude, power, power dB, phase, real, and imaginary modes.
- GCOV covariance linear/dB and complex cross-term magnitude/phase modes.
- Boxcar and adaptive Lee speckle filters with 3x3, 5x5, and 7x7 windows.
- Live finite/invalid counts, extrema, percentiles, and a log-count histogram.
- Grayscale, Turbo, cyclic phase, cmweather, and D3 colormaps.
- Native sibling-mask alignment and sample-spacing-aware display geometry.
- Blank startup with an in-app HDF5 file picker.
- F11 borderless fullscreen and a Tab-toggleable control panel.
- CUDA-first backend selection with a CPU fallback and opt-in HIP/SYCL builds.

Product discovery, metadata, chunk decoding, and raster reads use native HDF5.
LOD pages live in bounded memory for the current session.

## Requirements

- Windows 10 or 11.
- Vulkan 1.3-capable graphics driver.
- NVIDIA `sm_120` GPU and CUDA 13.3 for the CUDA path.
- Visual Studio 18 2026 Build Tools with MSVC and C++ CMake tools.
- CMake 4.2 or newer.

CUDA is selected automatically on a compatible device; otherwise the viewer
uses the CPU backend. The release CUDA code targets `sm_120`.

## Build

Dependencies are declared in [`vcpkg.json`](vcpkg.json) and resolved through
the Visual Studio vcpkg toolchain configured by `CMakePresets.json`.

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

& $cmake --preset release
& $cmake --build --preset release
```

The optimized viewer is written to:

```text
build/preset-release/viewer/Release/sat-viewer.exe
```

For a RelWithDebInfo build, replace `release` with `dev`.

Build the viewer without a CUDA toolchain:

```powershell
& $cmake --preset cpu-dev
& $cmake --build --preset cpu-dev
```

The CPU viewer is written to
`build/preset-cpu-dev/viewer/RelWithDebInfo/sat-viewer.exe`.

### Experimental GPU backends

ROCm/HIP and oneAPI/SYCL are source-build options. Both require the master
experimental switch plus their backend switch:

```text
# ROCm/HIP (configure with amdclang++ as CMAKE_HIP_COMPILER)
-DSATVIEW_ENABLE_CUDA=OFF
-DSATVIEW_EXPERIMENTAL_BACKENDS=ON
-DSATVIEW_ENABLE_EXPERIMENTAL_HIP=ON

# oneAPI/SYCL (configure with IntelLLVM icx and Ninja)
-DSATVIEW_ENABLE_CUDA=OFF
-DSATVIEW_EXPERIMENTAL_BACKENDS=ON
-DSATVIEW_ENABLE_EXPERIMENTAL_SYCL=ON
```

Run those builds with `--backend hip` or `--backend sycl`. They implement the
same scientific transforms, validity masks, and Boxcar/Lee filters as CUDA.

## Run

Open the blank viewer and select a product from the file picker:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe
```

Open a product directly:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe "E:\path\to\NISAR_product.h5"
```

Start at the complete-raster view:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe --fit-scene "E:\path\to\NISAR_product.h5"
```

Start with a centered 4x4 native source footprint:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe --zoom 4 "E:\path\to\NISAR_product.h5"
```

## Controls

| Input | Action |
| --- | --- |
| Mouse wheel | Zoom around the cursor |
| Left or middle drag | Pan |
| `+` / `-` | Zoom around the viewport center |
| `Q` / `E` | Rotate -90 / +90 degrees |
| `Tab` | Hide or restore controls |
| `F11` | Toggle borderless fullscreen |
| Fit Scene | Show the complete raster |


## Command-line options

```text
--zoom 1|2|4
--fit-scene
--rotation DEGREES
--backend auto|cuda|cpu|hip|sycl
--clean-view
--speckle none|boxcar|lee
--speckle-window 3|5|7
--speckle-looks N
--frames N
--smoke-test
```

Example:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe `
  --speckle lee --speckle-window 5 --speckle-looks 3 `
  --rotation 33 --clean-view "E:\path\to\NISAR_product.h5"
```

## Companion tools

Inspect product structure and metadata:

```powershell
.\build\preset-release\Release\sat-inspect.exe "E:\path\to\NISAR_product.h5"
.\build\preset-release\Release\sat-inspect.exe --json "E:\path\to\NISAR_product.h5"
```

Benchmark the overlapped HDF5/CUDA chunk pipeline:

```powershell
.\build\preset-release\Release\sat-bench.exe "E:\path\to\NISAR_product.h5" `
  --layer "/science/LSAR/GSLC/grids/frequencyA/HH" `
  --chunks 128 --warmup 8 --pipeline --csv
```

## Test

```powershell
$ctest = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --preset release
```

The suite covers navigation geometry, LOD planning, native/LOD selection,
science and mask sampling, CPU/CUDA transforms, speckle filters, distributions,
colormaps, cancellation, and publication behavior. Real-product integration
tests discover fixtures from `Test Data` or `SATVIEW_TEST_DATA_DIR`.

## Technical documentation

- [Architecture](ARCHITECTURE.md)
- [Performance measurements](PERFORMANCE.md)
- [Colormap attribution](COLORMAP_ATTRIBUTION.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

[MIT](LICENSE)
