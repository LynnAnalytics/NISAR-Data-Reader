# Architecture

NISAR Data Reader is organized around a two-tier native-grid pipeline: exact
bounded source-chunk mosaics for near views and guarded, on-demand screen-space
LOD pages for far views. It never materializes a product-sized raster and never
routes data through GDAL. HDF5 is the only raster/container reader; CUDA owns
the production fast path, CPU is the automatic fallback, and Vulkan owns
presentation. HIP/ROCm and oneAPI/SYCL are opt-in experimental compute paths.

## Components

| Component | Responsibility |
| --- | --- |
| `satview_core` | Identify products, read bounded native-HDF5 pages, provide pageable staging, and run CPU scientific processing. |
| `satview_gpu` | Own the CUDA pinned ring, transforms, speckle filters, and asynchronous distributions. |
| `satview_hip_backend` / `satview_sycl_backend` | Run the experimental accelerator transforms and speckle filters. |
| `vulkan_interop` | Export a Vulkan device-local buffer to CUDA and import a Vulkan timeline semaphore into CUDA. |
| `sat-viewer` | Coordinate the reader, in-memory overview worker, selected compute backend, Vulkan renderer, SDL3 window, and controls. |
| `sat-inspect` | Emit a human or JSON catalog without loading full science rasters. |
| `sat-bench` | Measure open/catalog, HDF5 read/decode, pinned H2D, and CUDA transform stages. |

## Backend selection

`--backend auto` selects CUDA when an `sm_120` device and the compiled CUDA
path are available, then falls back to CPU. Explicit `cuda` reports an error
instead of falling back. HIP and SYCL require both the master experimental
CMake switch and their backend switch, and are selected explicitly.

CUDA writes directly into Vulkan-exported memory. CPU, HIP, and SYCL use one
reusable pageable read ring and one persistent host-visible Vulkan staging
buffer. HIP/SYCL execute transforms, validity handling, and Boxcar/Lee filters
on the accelerator; distribution summaries run after the bounded result is
staged back to the host. The detailed path below describes the CUDA fast path.

## Product discovery and reads

`Hdf5Product` opens the file with the native HDF5 C API and discovers the NISAR
identification, frequency groups, grid axes, EPSG metadata, science layers,
auxiliary layers, physical shapes, datatypes, storage layout, chunk dimensions,
fill metadata, and filter order.

Untrusted containers are kept self-contained: catalog traversal rejects soft,
external, and user-defined links, virtual datasets, and external raw storage.
Dynamic HDF5 plugin loading is disabled before the file opens. Aggregate link,
path, and metadata-string budgets bound discovery work; variable-length dataset
strings use a bounded allocator, while optional variable-length annotation
attributes are rejected explicitly because the attribute API has no bounded
allocator hook. HDF5 exposes plugin loading only as a process-wide setting, so
opening a product intentionally leaves dynamic plugins disabled for the rest of
these standalone tools' process lifetime.

Complex HDF5 compounds are exposed to callers as tightly packed,
native-endian `{float real, float imaginary}` pairs. A `ReadPlan` expands the
requested window to complete source chunks and clips it at dataset edges.
`read_into()` then fills a caller-owned span exactly matching the aligned
window. The core does not allocate a full raster.

The installed HDF5 library is not a thread-safe build, so HDF5 calls are
protected by a process-wide recursive mutex. Ordinary `read_into()` calls use
`H5Dread` and retain HDF5's decompressed raw-chunk cache for repeated exact
native tiles. A bounded four-entry MRU/LRU read-dataset handle cache gives each
retained dataset its own configured cache. The default target is 64 MiB per
retained dataset, with at least one complete physical chunk made cacheable.
Alternating science and mask reads therefore retains both handles and their
independent caches.

The cold one-pass overview path has a stricter optional fast path. It requires
an exact `H5Tequal` native file/memory layout, one physical chunk, at most 16
MiB decoded storage, and exactly DEFLATE or shuffle-then-DEFLATE. Chunk lookup
and `H5Dread_chunk2` stay under both HDF5/read locks; DEFLATE reversal,
unshuffle, sparse-sample copy, and output writes run after both locks are
released. Up to eight CPU workers overlap that work while HDF5 calls remain
serialized. Worker count is also bounded by hardware, job count, and the
request's aggregate scratch limit. Unallocated, multichunk, oversized,
non-native, or differently filtered data falls back to ordinary `H5Dread`.

Each retained dataset also keeps its file dataspace and one bounded latest-shape
memory dataspace under the same mutex. Chunk reads therefore avoid repeated
`H5Dget_space` and `H5Screate_simple` handle churn without changing selections,
decoded bytes, cache identity, or serialized HDF5 access.


GDAL does not participate in any part of discovery, decoding, caching, or
preprocessing. There is no GDAL fallback.

## Continuous camera and resident selection

Camera coordinates are continuous pixel-edge row/column coordinates. The
camera stores a center, clockwise rotation, and physical raster units per
logical screen pixel, so unequal X/Y sample spacing is preserved without
stretching the data. The fit scale uses the axis-aligned bounds of the rotated
physical raster: it is the maximum zoom-out for that angle and letterboxes the
shorter screen axis. Camera centers are clamped using the inverse-rotated
viewport extents at raster edges.

Wheel zoom keeps the raster coordinate under the cursor fixed unless edge
clamping intervenes. `+` and `-` zoom around the viewport center. Left- and
middle-button drags move through the rotated raster coordinate system. The
degree slider provides arbitrary rotation; viewport-hovered `Q`/`E` apply
-/+90-degree steps. **Fit Scene** sets the complete-raster fit at the current
angle. `Tab` removes or restores the left control panel without discarding its
state, and the canvas expands to the full main viewport while it is hidden.
`F11` toggles SDL borderless fullscreen; hidden controls have no restore overlay.
Without a command-line product path, the application starts in a blank ImGui
shell with an in-app file button. The Windows picker accepts an existing HDF5
path, then the product-specific reader, allocations, and workers are created.

Each frame, the visible half-open raster window selects one of two resident
sources. The request is the conservative raster-axis-aligned bound of all four
inverse-rotated viewport corners, clipped to the source raster:

1. If the window crosses at most four chunks on each axis, the viewer first
   retains any resident native mosaic that fully contains it. Otherwise,
   automatic camera navigation selects a canonical 4x4 native guard band;
   explicit chunk/startup footprints remain exact 1x1, 2x2, or 4x4 requests.
   The mosaic shifts inward at edges, clips partial final chunks, and contains
   at most 16 chunks. Pans inside the guard change only camera/shader state.
2. Otherwise, the viewer plans a guarded regional LOD page. It chooses a
   power-of-two native source stride from the camera scale and SDL framebuffer
   scale, targeting at least 1.25 page texels per physical pixel when capacity
   permits. The visible window plus guard must fit a page no larger than
   4096x4096; capacity selects a coarser adjacent level only when necessary.
   Page origins are quantized on the global LOD lattice, and pans inside the
   resident guard remain draw-only. A prepared or in-flight page also remains
   usable after the exact desired request changes if it still covers the view
   at one texel per physical framebuffer pixel. This hysteresis avoids needless
   cancellation/rebuild churn around page and LOD boundaries.

The currently published image and its scene-to-texture mapping are retained
while a replacement is prepared. A replacement is eligible only after its
complete native mosaic or LOD page has been uploaded into the inactive Vulkan
image. The renderer then performs a coverage-aware 120 ms crossfade: pixels
covered by both sources blend, while pixels covered by only one source retain
that valid source. Camera-only changes inside the resident source are draw-only.

## CUDA exact native read path

For a selected native mosaic:

1. The viewer creates a canonical request. New native requests supersede
   older work between source-chunk reads.
2. It accepts only the science dataset's exact sibling leaf `mask` when it is
   readable uint8 with identical dimensions.
3. The single HDF5 reader walks the mosaic row-major. For each source chunk,
   it reads science and the aligned mask window into one chunk-sized pinned
   slot; HDF5 timing covers both reads.
4. The slot transitions `Free -> Filling -> Ready`. The completion carries its
   exact destination row/column in the canonical mosaic.
5. The render thread issues a `cudaMemcpy2DAsync` from pinned science directly
   into the persistent CUDA science mosaic. If a mask exists, a second 2D H2D
   copy crops the requested mask subwindow while placing it directly into the
   co-sampled mask mosaic. There is no device-to-device mask-crop pass.
6. These private input writes do not touch the Vulkan-owned output, so all
   chunk H2Ds are queued before any wait for previous Vulkan consumption.
7. A bounded one-to-three-slot event/read pump drains ready completions and
   reclaims slots without tying a 16-chunk load to the presentation rate.
8. After the final chunk, one mask-aware CUDA kernel transforms the complete
   mosaic. With **None**, it writes the raw Vulkan-exported 32-bit buffer directly.
   With speckle enabled, it writes one persistent float scratch page instead.
9. The selected 3x3, 5x5, or 7x7 Boxcar/Lee kernel reads that scratch page and
   writes the final Vulkan-exported 32-bit field. No filter kernel launches for
   **None**.
10. One asynchronous distribution pass scans the final field, then copies only
    its fixed approximately 2 KiB summary through pinned memory. There is no
    full scalar-output readback.
11. CUDA signals an external timeline value. Vulkan waits at the transfer stage,
    then copies the complete buffer into the inactive R32_UINT image.
12. The same Vulkan submission signals the next timeline value after consuming
    the buffer; the next CUDA output write waits for that value.
13. The completed inactive image becomes the new resident source. If an older
    source is visible, both images are retained for a coverage-aware 120 ms
    crossfade before the old image is retired.

There is no CUDA-output readback. Obsolete native completions are discarded
with their matching ready slots before H2D, and source identities prevent a
completion for one layer or resident region from being published as another.

## Screen-space LOD page path

A guarded LOD request is specific to the source file, science dataset, optional
exact sibling mask, regional source window, and explicit power-of-two sample
stride. The viewer derives that level from camera scale and physical framebuffer
density, targeting at least 1.25 page texels per physical pixel when possible.
The visible window plus a stable guard must fit the resident page; capacity
selects a coarser level only when needed, and no output axis may exceed the
4096 hard cap.

For each request, the planner expands coverage to the selected LOD lattice,
quantizes the page origin, and clips the source window at raster edges. It then
stores exact sparse source samples at:

```text
source_origin + sample_stride * (r, c)
```

`source_origin` is the quantized regional page origin, not necessarily zero.
Adjacent levels remain registered to the same global power-of-two LOD lattice.
Science and mask are always co-sampled. The stored payload is raw float32 or
complex-float32 science plus an optional uint8 validity payload—not dB, phase,
RGB, or a radiometric aggregate. The usual CUDA kernel applies the selected
scientific mode only after upload.

One caller-owned background operation walks only source chunks intersecting a
regional page. A Fit Scene page covers the complete raster and can therefore
visit every source chunk. Eligible one-pass reads use up to eight bounded
decoder workers: HDF5 lookup/raw acquisition remains serialized, while exact
DEFLATE/unshuffle and disjoint sparse-output copies overlap outside the HDF5
lock. The configured scratch limit bounds aggregate source staging, not just a
single worker. Shuffle/deflate inputs still require each intersecting compressed
chunk to be decoded even though the output retains only exact sparse lattice
samples. Progress callbacks are serialized and monotonic. The operation yields
between completed chunks while the latency-sensitive native reader is active
and checks cancellation at each chunk boundary.

Overview output exists only in bounded memory owned by the active worker and
renderer. No `.svo` files or persistent page-cache directory are created.
Planning fingerprints the source and dataset metadata to reject stale work;
cancelled jobs return no partial output. Legacy `.svo` files are left untouched.

The build path is final-only: it does not progressively display an incomplete
page and cannot resume one. A complete in-memory page is transformed and
uploaded into the inactive image before the ready-only 120 ms
crossfade begins. Camera movement inside the resident guard is shader-only. A
scientific-mode or speckle-setting change reuses the resident raw samples and
reruns only the transform/filter/distribution sequence, without HDF5 or H2D.

For both source kinds, window, gamma, palette, sampling mode, camera crop, and
scene-to-texture mapping alter only Vulkan push constants and the draw. Typed
low/high/gamma edits therefore do not rescan either HDF5 or CUDA data.

## Pinned-ring lifetime

The viewer constructs one to three fixed-capacity `cudaHostAlloc` slots sized
for the largest single science chunk plus its worst-case aligned sibling-mask
read. Slot count is reduced as necessary so the complete read ring stays at or
below 512 MiB; a single packed chunk above that bound is rejected. Capacity
does not grow to 4x4: the larger working set exists only in persistent GPU
allocations. Alignment bounds account for partial mask chunks before and after
a request.

Each slot has a pre-created CUDA event:

```text
Free -> Filling -> Ready -> InFlight -> Free
```

After the per-chunk science and optional mask H2Ds are queued,
`mark_in_flight()` records the slot event. `reclaim_completed()` uses
`cudaEventQuery`; it does not issue `cudaDeviceSynchronize`. The slot can be
reused while later chunks are read and while the final transform/publication
waits on Vulkan ownership.

App-owned pinned storage and persistent CUDA mosaics are allocated once and
reused. Stream/device waits are reserved for controlled teardown; Vulkan
`deviceWaitIdle` is used for resize or teardown, not ordinary view loading.

## CUDA/Vulkan ownership protocol

The interop layer requires:

- the current CUDA device and selected Vulkan physical device to match by
  Windows adapter LUID, node mask, and UUID;
- Vulkan timeline-semaphore support;
- `VK_KHR_external_memory_win32`;
- `VK_KHR_external_semaphore_win32`; and
- exportable `OPAQUE_WIN32` buffer memory and semaphore handles.

One monotonically increasing timeline carries both ownership directions. Values
are allocated in odd/even pairs:

```text
CUDA prepares an exact mosaic or guarded raw LOD page in private memory
CUDA waits for previous even value (Vulkan consumed)
CUDA transform writes the shared output directly or one private scratch page
optional speckle kernel writes scratch -> shared output
CUDA distribution summarizes the final shared field to a fixed pinned result
CUDA signals odd value N (CUDA ready)
Vulkan waits for N at TRANSFER
Vulkan copies buffer -> inactive R32_UINT image
Vulkan signals N + 1 (Vulkan consumed)
next CUDA write waits for N + 1
```

Once that complete image is ready, presentation may retain the preceding image
for the independent 120 ms crossfade; it does not extend CUDA ownership.

The Win32 export handles are closed after CUDA import. CUDA and Vulkan resource
objects retain their respective references until coordinated teardown.

## Scientific transforms

The viewer exposes these one-output-value-per-pixel CUDA paths:

| Input | Modes |
| --- | --- |
| GSLC complex polarization | amplitude, power, power dB, phase, real, imaginary |
| GCOV real diagonal covariance | linear, power dB |
| GCOV complex cross covariance | magnitude, phase |

Non-finite values become NaN. Negative GCOV diagonal terms are invalid. Every
viewer transform receives the paired mask when one passes the sibling,
dimension, and uint8 checks. It uses NISAR mask semantics (`1..254` valid; `0`
and `255` invalid). The reusable API also provides normalized complex
correlation magnitude, which is tested but is not currently connected to the
desktop controls.

Neighborhood speckle filtering is supported only for non-negative SAR domains:
GSLC amplitude/power/power dB and GCOV diagonal linear/power dB. Boxcar returns
the valid-neighbor arithmetic mean in linear power. Lee uses population variance
and `w = clamp((v - m*m/ENL) / v, 0, 1)`, with zero weight when variance is zero.
Amplitude and dB are converted to relative linear power for statistics and
converted back afterward; the kernel never averages logarithmic dB values.

Each neighborhood is normalized by its local finite maximum, which keeps
intermediate statistics bounded even for extreme finite amplitude or dB input.
Invalid/non-finite centers remain NaN, invalid neighbors are excluded, and image
edges use clipped windows. Window size is resident-grid space: a native page has
dense native semantics, while an LOD page spans sparse samples separated by its
labeled stride. Filter options participate in exact render identity, and the
resident distribution always scans the post-filter field.

## Polarimetric decomposition and layer comparison

Derived analysis uses a serial, latest-request-wins worker separate from the
single-layer reader. Its aligned reader streams up to four science rasters on
one exact source lattice, deduplicates a shared sibling mask, and visits a
block only after all of its members are complete. Every member retains its own
bounded HDF5 `ReadPlan`; planning rejects incompatible grids, stale source
metadata, arithmetic overflow, or output/scratch budgets before publication.
Cancellation returns no partial page. The result is held only in bounded
worker/renderer memory and is never written to a persistent cache.
Canonical page identity excludes only the scheduling serial, so unchanged
guarded windows reuse active, prepared, or resident composite data. Switching
between split and swipe reuses the same packed pair and changes only draw state.

Pauli RGB is resolved per frequency and is available only when one GCOV grid
contains readable, identically aligned float32 `HHHH`, `HVHV`, and `VVVV`
diagonal terms plus complex64 `HHVV`. For reciprocal covariance input the
linear channel powers are:

```text
R (double bounce) = 0.5 * (HHHH + VVVV - 2*Re(HHVV))
G (cross-pol)     = 2 * HVHV
B (surface)       = 0.5 * (HHHH + VVVV + 2*Re(HHVV))
```

Non-finite terms, negative diagonal powers, or an invalid shared mask make the
output invalid. Negative derived channel power is retained by the scientific
calculation but clipped to zero only for RGB display conversion. The displayed
channels are converted to dB and packed as three 10-bit values over
`[-100, +50] dB`, with a two-bit validity tag, in one R32_UINT transport word.

Layer comparison currently requires two different science rasters on the same
frequency with identical non-empty dimensions, grid mapping, usable coordinate
metadata, and compatible units. Thus split, swipe, `A - B`, and power-ratio
views are exact-grid comparisons only; no resampling or reprojection is hidden
in the operation. Split and swipe pack the independently valid transformed
values as two range-preserving bfloat16 values in one R32_UINT transport word.
Difference remains a full float32 scalar. Ratio is computed from non-negative linear power and shown
as `10*log10(A/B)`; valid zero numerators use the transform epsilon floor
(`-200 dB` at the current `1e-20` epsilon), while zero/near-zero denominators
are invalid. Speckle filtering is disabled for these derived pages.

## Rendering and display controls

The renderer always receives an R32_UINT image. Scalar pages retain their exact
float32 bits and are decoded with `uintBitsToFloat`; composite pages use one of
the two packed display transports above. The fragment shader:

- maps the axis-aligned draw rectangle through the inverse camera rotation into
  a resident native mosaic or LOD page, discarding fragments outside the
  rotated source window;
- in **Exact Pixels**, performs one authoritative nearest-texel science fetch;
- in **Smooth**, keeps the nearest texel authoritative for validity, manually
  fetches four neighbors, and renormalizes the finite samples;
- interpolates wrapped phase as weighted unit directions so the `-pi`/`+pi`
  seam follows the short arc;
- maps low/high to `[0, 1]` and applies display gamma;
- rounds to one of 256 samples in a persistent 256-by-20 RGBA8 UNORM atlas;
- selects a palette row with an exact nearest `texelFetch`;
- discards NaN/Inf and unused image texels rather than sampling stale data.

Packed Pauli and split/swipe views force authoritative nearest-texel sampling;
the latter maps both panels to the same complete source window, with swipe
using a screen-space divider. Scalar difference and ratio retain the ordinary
float distribution, palette, and range controls. Packing is a display/storage
transport: it is not used for the float32 Pauli arithmetic, difference, or
ratio calculations.

Smooth is display interpolation over exact sparse stored samples; it does not
change the cache payload, perform a radiometric aggregate, or alter the native
grid. Exact Pixels exposes the stored sample lattice directly.

Low and high are direct finite float inputs constrained only by `low < high`;
gamma is a direct finite positive input clamped to the shader's exact `0.0001`
minimum. Adaptive edit steps and a range drag are UI conveniences, not limits.

The asynchronous distribution reducer reports exact finite/invalid counts and
extrema plus a 256-bin histogram. Percentiles are derived from that histogram
and are therefore approximate. The fixed result is copied through pinned host
memory and polled by event; there is no per-frame scan or full output readback.
The UI binds a result to its exact layer/source/mode/filter request and labels
whether its scope is regional native, regional sparse, whole-scene native, or
whole-scene sparse, including origin, coverage, dimensions, and stride.

The 20 stable rows are Grayscale, Turbo, Cyclic phase, nine maps pinned from
openradar/cmweather, and D3 Viridis, Cividis, Inferno, Magma, Plasma, reversed
RdBu, PuOr, and Cubehelix Default. The atlas is 20,480 bytes, uploaded once at
startup to device-local Vulkan memory, and has no runtime D3/cmweather/Python
dependency. Full pinned provenance and licenses are installed as
`docs/COLORMAP_ATTRIBUTION.md`.

Defaults follow data topology: D3 Viridis for sequential amplitude/power/dB,
linear covariance, and magnitude; cmweather balance for signed real/imaginary;
and the endpoint-closed cyclic map for phase. Changing layer or scientific mode
reapplies the corresponding preset. Palette, low/high, gamma, sampling mode,
camera rotation, and transition opacity are Vulkan push constants and never
trigger HDF5, H2D, or CUDA work.

Swapchain creation prefers `VK_PRESENT_MODE_MAILBOX_KHR` and falls back to
FIFO. Camera scale is expressed in physical raster units per logical screen
pixel. For physical raster width `W`, height `H`, and clockwise angle `theta`,
**Fit Scene** uses these rotated bounds:

```text
rotated_width  = abs(cos(theta)) * W + abs(sin(theta)) * H
rotated_height = abs(sin(theta)) * W + abs(cos(theta)) * H
fit_scale      = max(rotated_width / viewport_width,
                     rotated_height / viewport_height)
```

The shorter screen axis is letterboxed. SDL framebuffer scaling does not
alter camera geometry or the visible source window, but it does inform LOD
density: the planner targets 1.25 page texels per physical framebuffer pixel
when the 4096 resident cap permits. This preserves physical sample spacing
without reprojecting the native grid.

## Native-grid and map boundary

The near-view working set contains at most 16 adjacent native source chunks.
Far views use on-demand regional pages on the globally registered power-of-two
LOD lattice. Each page contains exact sparse source samples and has a hard 4096
sample cap on either axis. Neither path is a map engine: there is no
reprojection, warp, basemap, geodetic tile scheme, or geographic resampling,
and GDAL is not linked or invoked.

Skipped source samples are never spatially or radiometrically aggregated.
Smooth may interpolate the displayed scalar field, with validity-aware finite
renormalization and circular phase handling, but it does not alter the stored
samples. The viewer never averages dB, wrapped scalar phase, or rendered RGB
into an LOD cache product.

These are on-demand working-set pages at adjacent levels, not a precomputed
geospatial tile pyramid. Fit Scene is the whole-raster case of the same planner.
Cold generation remains complete-before-display and cannot resume an incomplete
page.

## Shutdown and error behavior

Automatic selection falls back to CPU only when the CUDA path is unavailable;
an explicitly requested accelerator reports its error. The native reader and
overview worker are stopped and joined before their source product or staging
storage is destroyed. The CUDA stream is drained before persistent CUDA
resources are released, and Vulkan is made idle before swapchain and interop
objects are destroyed.

## Runtime package layout

The optimized release install places `sat-viewer.exe`, `sat-inspect.exe`,
`sat-bench.exe`, and their app-local
HDF5/zlib/SDL3/Vulkan/MSVC runtime DLLs in
one directory. The viewer loads compiled SPIR-V from `shaders` relative to its
executable, not the process working directory. `README.md`, `ARCHITECTURE.md`,
`PERFORMANCE.md`, and `COLORMAP_ATTRIBUTION.md` are installed under `docs` when
the viewer is enabled. CLI-only installs remain supported; the application
bundle is not an SDK.
