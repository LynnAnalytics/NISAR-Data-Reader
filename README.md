# NISAR Data Reader

An exploratory visualizer for NISAR data releases. Reads raw HDF5 files in seconds rather than minutes with a custom rendering pipeline built in Vulkan & CUDA. 

![NISAR Data Reader displaying a GCOV product](docs/images/nisar-data-reader.png)

[Download NISAR Data Reader v0.5.4 for Windows](https://github.com/LynnAnalytics/NISAR-Data-Reader/releases/tag/v0.5.4)

## Highlights

- Continuous cursor-anchored zoom, drag panning, rotation, and Fit Scene.
- Exact native-resolution views covering 1x1, 2x2, or 4x4 source chunks.
- Guarded screen-space LOD pages for smooth navigation across large products.
- GSLC amplitude, power, power dB, phase, real, and imaginary modes.
- GCOV covariance linear/dB and complex cross-term magnitude/phase modes.
- CUDA Boxcar and adaptive Lee speckle filters with 3x3, 5x5, and 7x7 windows.
- Live finite/invalid counts, extrema, percentiles, and a log-count histogram.
- Grayscale, Turbo, cyclic phase, cmweather, and D3 colormaps.
- Native sibling-mask alignment and sample-spacing-aware display geometry.
- Blank startup with an in-app HDF5 file picker.
- F11 borderless fullscreen and a Tab-toggleable control panel.

Product discovery, metadata, chunk decoding, and raster reads use native HDF5.
LOD pages live in bounded memory for the current session.

## Requirements

- Windows 10 or 11.
- At the moment, NVIDIA RTX 5080 or better (`sm_120`).
- CUDA Toolkit 13.3.
- Vulkan driver with timeline semaphore and Win32 external-memory support.
- Visual Studio 18 2026 Build Tools with MSVC and C++ CMake tools.
- CMake 4.2 or newer.

The release configuration targets the RTX 5090. 

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
science and mask sampling, CUDA transforms, speckle filters, distributions,
colormaps, cancellation, and publication behavior. Real-product integration
tests discover fixtures from `Test Data` or `SATVIEW_TEST_DATA_DIR`.

## Technical documentation

- [Architecture](ARCHITECTURE.md)
- [Performance measurements](PERFORMANCE.md)
- [Colormap attribution](COLORMAP_ATTRIBUTION.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

[MIT](LICENSE)
