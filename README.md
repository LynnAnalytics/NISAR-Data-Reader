# NISAR Data Reader

NISAR Data Reader is a native, GPU-first desktop viewer and inspection toolkit for
NISAR GSLC and GCOV HDF5 products. It reads source HDF5 chunks directly, runs
scientific transforms on CUDA, and presents the result with Vulkan.

![NISAR Data Reader displaying a GCOV product](docs/images/nisar-data-reader.png)

The design has a hard boundary: **GDAL is not a dependency, fallback, adapter,
or preprocessing step.** The application does not invoke GDAL utilities, build
VRTs, create GDAL sidecars, or require a GDAL-generated intermediate. Product
discovery, metadata, datatype conversion, chunk alignment, and filter decoding
all use the native HDF5 library.

## What is implemented

- Native C++20 cataloging and chunk-aligned reads for NISAR GSLC and GCOV.
- A continuous, sample-spacing-aware camera with cursor-anchored wheel zoom,
  centered `+`/`-` zoom buttons, left- or middle-button drag panning, and
  **Fit Scene** for an all-the-way-out view of the complete raster. Arbitrary
  clockwise rotation is available from a degree slider; `Q`/`E` rotate by
  90 degrees while the viewport is hovered.
- A distraction-free view: `Tab` hides or restores the complete left control
  panel without resetting it, and a small restore button remains on the canvas.
- Exact native-resolution working sets up to 4x4 source chunks. Continuous
  navigation uses the 4x4 allocation as a pan guard band; explicit 1x1, 2x2,
  and 4x4 footprint controls remain exact.
- Guarded screen-space LOD pages for views beyond the exact 4x4 path. A
  power-of-two source stride targets at least 1.25 exact sparse samples per
  physical framebuffer pixel when the resident capacity permits. Page-aligned
  regional coverage has a hard 4096x4096 output cap. The current in-memory page
  survives small pan/zoom changes while it still covers the view at full
  framebuffer density, avoiding request churn at LOD boundaries.
- Native HDF5 co-sampling of raw science and the exact sibling mask on the same
  request-aligned lattice. Overview pages are bounded in-memory working data;
  the viewer creates no persistent page cache.
- An exact one-pass cold-page decoder for native-layout shuffle/DEFLATE
  chunks. HDF5 metadata/raw reads remain serialized for the installed
  non-thread-safe library; up to eight bounded CPU workers reverse DEFLATE and
  shuffle after releasing the HDF5 lock. Unsupported, oversized, multichunk,
  and unallocated reads use the ordinary HDF5 path. Interactive native reads
  keep HDF5's decompressed chunk cache.
- Caller-owned reads into a three-slot page-locked ring; pinned capacity stays
  chunk-sized even for a 16-chunk exact native view.
- Exact sibling `mask` discovery with dimension/type validation and direct
  mask alignment; `inputDataExceptionMask` is never substituted.
- Persistent CUDA science/mask mosaics assembled with asynchronous 2D H2D
  copies, plus direct raw LOD-page upload, followed by one scientific
  transform for the complete resident view.
- GSLC amplitude, power, power dB, phase, real, and imaginary transforms.
- GCOV diagonal covariance linear/dB and complex cross-term magnitude/phase.
- Vulkan/CUDA external-memory interop on the same Windows GPU, with an
  external timeline semaphore for ownership and no CUDA-output readback.
- Optional 3x3, 5x5, and 7x7 GPU Boxcar and adaptive Lee speckle filters.
  Amplitude and dB are filtered through linear-power neighborhood statistics;
  phase, signed components, and cross-term magnitude are deliberately excluded.
  Lee equivalent looks is a typed, positive user input. **None** remains the
  exact-source default and launches no filter kernel.
- An asynchronous 256-bin GPU distribution for each transformed resident page:
  exact finite/invalid counts and extrema, approximate p1/p2/p50/p98/p99,
  log-count plotting, hover counts, and one-click full/1-99/2-98 windows.
- An SDL3 + Dear ImGui shell with semantic layer, scientific mode, continuous
  camera navigation, focus-chunk controls, and live stage timings.
- `--rotation DEGREES` and `--clean-view` make rotated, panel-free startup
  views reproducible for presentation and smoke testing.
- A ready-only, coverage-aware 120 ms crossfade between resident sources. The
  previous Vulkan image remains available until its complete replacement has
  been prepared and uploaded.
- Typed, finite low/high controls, an unbounded-above gamma input with the exact
  shader minimum of `0.0001`, plus colormap and sampling controls. These remain
  draw-only. **Smooth** is validity-aware display interpolation; **Exact
  Pixels** is nearest-texel inspection. Wrapped phase is interpolated
  circularly across the `-pi`/`+pi` seam. The persistent
  256-by-20 RGBA8 LUT includes Grayscale, Turbo, Cyclic phase, nine cmweather
  maps, and eight additional D3 maps.
- Mode and speckle changes reuse the resident raw science/mask source and rerun
  only CUDA; they do not reread HDF5. Palette, window, gamma, sampling, and
  camera changes are draw-only while the camera remains within the resident
  guard.
- A native-grid viewport that preserves physical X/Y sample-spacing aspect
  ratio, plus human-readable/JSON metadata inspection and stage benchmarks.

## Current scope

The default view is a centered 1x1 footprint of the preferred `HH` GSLC or
`HHHH` GCOV layer. From there, navigation is continuous rather than limited to
three preset footprints. Scroll over the image to zoom around the source
coordinate under the cursor, drag with the left or middle mouse button to pan,
use `+`/`-` for centered zoom, or choose **Fit Scene** to see the entire
dataset. Use the rotation slider for arbitrary angles, or hover the viewport
and press `Q`/`E` for 90-degree steps. Positive angles rotate the image
clockwise. `Tab` toggles the left panel so the viewport can use the complete
window. Zoom-out stops at the physical-aspect-preserving fit for the current
rotation.

When the visible window fits within four source chunks on each axis, the
viewer keeps exact native values in a bounded GPU mosaic. It first reuses any
resident mosaic that still contains the camera window; otherwise automatic
camera navigation selects a canonical 4x4 guard band. This keeps ordinary
drags shader-only until the camera crosses that guard. Explicit chunk and
startup footprints still load the requested exact 1x1, 2x2, or 4x4 mosaic.
Near an edge, the footprint moves inward; small rasters use every available
chunk, and partial final chunks retain their true extent. At most 16 chunks
are resident in this exact path.

Farther out, the viewer requests a guarded regional LOD page rather than
switching directly from native data to one coarse full-scene image. The viewer
chooses a power-of-two source stride in screen space, including SDL framebuffer
scale, and targets at least 1.25 page texels per physical pixel when the
4096x4096 resident cap permits. Page origins are quantized to the global LOD
lattice, so adjacent levels retain the same native-grid registration.

Output sample `(r, c)` is the exact sparse native sample at
`source_origin + sample_stride * (r, c)`; science and mask always use the same
lattice. These stored values are selected source samples, not spatial or
radiometric aggregates. **Smooth** is display interpolation only: it preserves
the nearest texel's validity, renormalizes finite neighbors, and treats phase
circularly. **Exact Pixels** exposes nearest stored texels directly.

Speckle reduction operates on the current resident sample grid. On a native
page, an `N x N` window is exactly `N x N` native samples. On a sparse LOD page,
those `N x N` samples are separated by the labeled row/column stride, so the UI
reports the effective native footprint and never describes it as a dense native
filter. Boxcar and Lee both calculate in linear power, then return to the active
amplitude/power/dB display domain. Invalid centers remain invalid and invalid
neighbors do not contribute.

The distribution panel describes exactly the transformed, filtered resident
page currently being displayed. Counts and extrema are exact for that page;
percentiles are 256-bin estimates. A Fit Scene page provides whole-scene sparse
lattice coverage, not a scan of every native source pixel. Regional/native
pages and their origin, coverage, dimensions, and stride are labeled explicitly.

The currently displayed source remains available while another native mosaic
or LOD page is prepared. A replacement is never displayed partially. After a
complete page is uploaded, a coverage-aware 120 ms crossfade blends overlapping
valid coverage while retaining whichever source covers the remaining pixels.

An LOD regional request visits only its intersecting compressed HDF5
chunks and remains memory-only. **Fit Scene** uses the same planner for the
complete raster and can therefore visit every source chunk. Camera
movement and rotation inside the resident guard are shader-only. Rotating can
request new data only when it exposes source outside that guard. Builds
report progress and can be cancelled between chunks, but incomplete pages are
neither displayed nor resumable yet.

There is no reprojection, warping, basemap, or geographic resampling. EPSG and
grid-axis metadata are cataloged, and sample spacing controls display aspect
ratio, but pixels stay on their product-native grid. Neither the exact path nor
the LOD-page path uses GDAL.

`sat-inspect` catalogs auxiliary datasets. For a renderable float32 GSLC/GCOV
science layer, the viewer uses only the exact sibling leaf named `mask` when it
is readable uint8 and has the same dimensions; it never substitutes
`inputDataExceptionMask`.

## Requirements

The checked-in GPU configuration targets the local RTX 5090:

- Windows with a WDDM-visible CUDA/Vulkan adapter.
- NVIDIA RTX 5090-class GPU (`sm_120`).
- CUDA Toolkit 13.3.
- A Vulkan driver with timeline semaphores and Win32 external memory/semaphore
  support.
- Visual Studio Build Tools with the MSVC C++ toolchain.
- CMake 4.2 or newer (required for the Visual Studio 18 2026 generator).
- vcpkg dependencies declared in the source-tree `vcpkg.json`: HDF5 with
  zlib, nlohmann-json, SDL3/Vulkan, Dear ImGui, Vulkan loader/headers, and
  glslang.

The viewer verifies that CUDA and Vulkan selected the same adapter by Windows
LUID, node mask, and UUID. It fails explicitly if the required interop features
are unavailable; there is no CPU-rendering fallback.

## Build on this workstation

The checked-in presets are self-contained for this machine: they select the VS
18 2026 x64 generator, the installed vcpkg toolchain and dependency tree, and
`sm_120`. A fresh configure and build succeeds from ordinary PowerShell:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --preset dev
& $cmake --build --preset dev
```

If that CMake is already on `PATH`, the equivalent commands are `cmake
--preset dev` and `cmake --build --preset dev`.

The `dev` preset builds RelWithDebInfo viewer, tests, inspector, and benchmark.
`release` builds Release equivalents. `cpu-dev` builds diagnostics and tests
without the viewer; the native viewer requires CUDA.

## Package an optimized build

Build and test the `release` preset, install a fresh versioned app-local
package, and archive its contents without an extra enclosing directory:

```powershell
$releaseName = "SatDataReader-0.5.4-win64-Release"
$buildRoot = (Resolve-Path -LiteralPath "build").Path
$stage = Join-Path $buildRoot $releaseName
$archive = "$stage.zip"

if ((Test-Path -LiteralPath $stage) -or
    (Test-Path -LiteralPath $archive)) {
    throw "Refusing to overwrite an existing v0.5.4 package."
}

cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --install build\preset-release --config Release --prefix $stage

Compress-Archive `
  -Path (Join-Path $stage "*") `
  -DestinationPath $archive `
  -CompressionLevel Optimal

Get-FileHash -Algorithm SHA256 -LiteralPath $archive
```

The package places `sat-viewer.exe`, `sat-inspect.exe`, `sat-bench.exe`, and
the required HDF5, zlib, SDL3, Vulkan-loader, and MSVC runtime DLLs
at its root.
These documents are installed under `docs`, and compiled shaders under
`shaders`; the viewer resolves that shader directory relative to its
executable. The wildcard passed to `Compress-Archive` deliberately preserves
that flat runtime ZIP layout. CLI-only installs also work when
`SATVIEW_BUILD_VIEWER=OFF`. These redistribution instructions cover optimized
Release/RelWithDebInfo configurations only; no Debug redistribution package is
promised.

## Run

Launch without a path to open the blank viewer and choose a product from its
**Open HDF5 file...** button:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe
```

You can still open a product directly. The default camera starts on a centered
1x1 exact native footprint:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe "E:\path\to\NISAR_product.h5"
```

Over the scientific viewport, use the mouse wheel for cursor-anchored zoom,
left- or middle-button drag to pan, `+`/`-` for centered zoom, and **Fit Scene**
to zoom all the way out. Use the **Rotation** slider for arbitrary clockwise
angles, or hover the viewport and press `Q`/`E` for -/+90-degree steps. Press
`Tab` to hide or restore the left panel. Press `F11` to enter or leave
borderless fullscreen.
Camera movement is continuous; the 1x1/2x2/4x4 values describe exact native
working sets, not the only available zoom levels.

Sampling defaults to **Smooth** for continuous visual navigation. Choose
**Exact Pixels** to inspect nearest stored texels without display interpolation.
Neither mode reprojects the product or turns exact sparse LOD samples into
radiometric averages.

Low, High, and Gamma accept direct finite numeric input; the range drag remains
available for quick exploration. The resident-distribution panel shows exact
finite/invalid counts and extrema, approximate percentiles, and presets that
apply without rereading data. Select **Boxcar mean** or **Lee adaptive** under
**Speckle reduction** to redispatch the resident raw page on CUDA. Lee exposes
equivalent looks directly; supported windows are 3x3, 5x5, and 7x7.

The same initial settings are available for reproducible launches and smokes:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe --speckle lee `
  --speckle-window 5 --speckle-looks 3 --rotation 33 --clean-view `
  "E:\path\to\NISAR_product.h5"
```

Start directly at the complete-raster fit view:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe --fit-scene "E:\path\to\NISAR_product.h5"
```

A Fit Scene request can scan all intersecting compressed HDF5 chunks before
publishing its complete LOD page. The page remains in memory only for the
current process; restarting the viewer builds it again.

Start with a centered 4x4 exact native-chunk view:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe --zoom 4 "E:\path\to\NISAR_product.h5"
```

Smoke the exact native path:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe --smoke-test --zoom 4 "E:\path\to\NISAR_product.h5"
```

Smoke the Fit Scene native-HDF5 LOD path, including its in-memory build, raw
H2D, CUDA transform, Vulkan interop, and presentation:

```powershell
.\build\preset-dev\viewer\RelWithDebInfo\sat-viewer.exe --smoke-test --fit-scene "E:\path\to\NISAR_product.h5"
```

Smoke mode waits for the requested source to be published and four valid
post-upload frames. It prints the native footprint or LOD page origin, source
coverage, stride, output dimensions, and page status, plus preparation/HDF5,
H2D, and CUDA times. A fit-scene smoke allows up to five minutes for a
compressed scan; the native smoke allows one minute.

For automation, `--frames N` exits after `N` presented frames. `--zoom` accepts
only `1`, `2`, or `4` and selects the initial exact footprint;
`--fit-scene` selects the initial complete-raster view. `--speckle` accepts
`none`, `boxcar`, or `lee`; `--speckle-window` accepts `3`, `5`, or `7`; and
`--speckle-looks` accepts any finite positive float. `--rotation` accepts any
finite clockwise degree value, and `--clean-view` starts with the left panel
hidden.

Inspect a product without reading full science rasters:

```powershell
.\build\preset-dev\RelWithDebInfo\sat-inspect.exe "E:\path\to\NISAR_product.h5"
.\build\preset-dev\RelWithDebInfo\sat-inspect.exe --json "E:\path\to\NISAR_product.h5"
```

Benchmark the overlapped independent-chunk pipeline:

```powershell
.\build\preset-dev\RelWithDebInfo\sat-bench.exe "E:\path\to\NISAR_product.h5" `
  --layer "/science/LSAR/GSLC/grids/frequencyA/HH" `
  --chunks 128 --warmup 8 --pipeline --csv
```

`--pipeline` starts one HDF5 reader thread and overlaps read/decode with
asynchronous H2D and CUDA transforms through the same three-slot ring primitive
used by the viewer. It benchmarks independent chunks one by one; it does not
include sibling masks, multi-chunk mosaic assembly, CUDA/Vulkan interop, or
presentation. Two fixed 64-chunk timing banks ping-pong: the host waits only
when recycling a bank and on the final ordered event, never once per chunk or
with a device-wide synchronization in the normal measured path. Output includes
pipeline wall time, chunks/s, logical GiB/s, and per-stage p50/p95. Omit
`--pipeline` for isolated per-chunk timing. Pipeline mode requires CUDA.

If `--layer` is omitted, `sat-bench` selects the first readable science layer.
For meaningful CUDA results, pass a GSLC/GCOV science layer; an explicit path
is validated for readability, chunking, and float32/complex64 representation,
not semantic role. `--chunks` accepts `1..1000000`; `--warmup` accepts
`0..1000000`.

## Test

```powershell
$ctest = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --preset dev
```

The integration test discovers real products under `Test Data` by default. Set
`SATVIEW_TEST_DATA_DIR` to use another fixture directory.

Current validation includes exact real-data center-pixel goldens, CUDA
transform/mask edge cases, 1x1/2x2/4x4 geometry and allocation bounds, partial
bottom/right chunks, rotated continuous-camera mapping and clamping,
cursor-anchored zoom and pan invariants, and exact-native-versus-LOD resident
selection. LOD tests cover
guarded regional planning, physical-framebuffer density, the 4096 cap,
source-window/stride identity, globally aligned level transitions,
science/mask co-sampling, deterministic rebuilds, cancellation, and ready-only
publication. Shader checks cover validity-aware finite-neighbor
renormalization, circular phase interpolation, Exact Pixels, exact palette
indexing, and all 20 palette rows.

Distribution tests validate every sample in a 4096-by-4096 golden page,
including exact counts/extrema, histogram accounting, percentiles, presets,
invalid data, and numeric-control guards. Speckle tests compare GPU Boxcar and
Lee output against CPU references for all domains and windows, masks, clipped
borders, constants, impulses, ENL behavior, NaN/Inf, extreme finite values,
bitwise determinism, and invalid options.

The historical v0.4.1 optimized Release smoke suite passes the exact 4x4 path
on all seven supplied products: one GCOV and six GSLC, each producing a
2048-by-2048 native-grid mosaic and four valid Vulkan-presented frames.
Additional real-data
smokes pass with 5x5 Boxcar and 7x7 Lee on exact native pages, and with Boxcar
and Lee on whole-scene sparse LOD pages. The prior v0.3.1 wheel/pan and
transition evidence remains documented separately in `PERFORMANCE.md`.

The v0.5.4 Release suite passes the blank file-picker launch, all three CTest
targets, and direct-path GCOV smoke testing. Fit Scene smokes pass on real GCOV
and GSLC products. The v0.5.0 rotated Fit Scene smokes at
+33 degrees on GCOV and -27 degrees on GSLC each published a validated
whole-scene LOD page and four post-upload Vulkan frames.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership and synchronization,
[PERFORMANCE.md](PERFORMANCE.md) for measured native zoom scaling and the
historical v0.3.x/v0.4.x baselines plus current v0.5.4 validation measurements,
and [COLORMAP_ATTRIBUTION.md](COLORMAP_ATTRIBUTION.md) for pinned
palette provenance and redistribution notices.

