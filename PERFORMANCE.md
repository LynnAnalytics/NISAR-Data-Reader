# Performance

Performance is measured stage by stage on the RTX 5090 rather than inferred
from architecture. The current benchmark shows that native HDF5 read and
decompression dominate steady-state chunk latency; pinned transfer and CUDA
scientific transforms are already small. Continuous navigation adds a second
far-view regime: guarded regional screen-space LOD pages whose native-HDF5
scan, physical-pixel density, and ready-only transition must be measured
separately from exact native mosaics.

No result in this document uses GDAL. Older benchmark sections retain results
for the retired `.svo` cache; current builds keep LOD pages only in bounded
memory and create no intermediate raster.

## Benchmark modes

Both modes use:

- the `RelWithDebInfo` CUDA build targeting `sm_120`;
- native HDF5 with the product's physical shuffle/deflate pipeline;
- a three-slot page-locked host ring;
- asynchronous H2D on a nonblocking CUDA stream; and
- the default transform selected by `sat-bench` for each layer.

Without `--pipeline`, `sat-bench` synchronizes each sample to isolate HDF5,
H2D, and CUDA stage timings. The reported isolated runs use 64 measured chunks
after four warmup chunks. `HDF5 GiB/s` is logical decoded bytes divided by time
in the HDF5 read/decode stage, not compressed storage throughput.

With `--pipeline`, warmup is performed before the timed interval and the CUDA
stream is drained once after all warmup chunks. Plans, result storage, and CUDA
timing resources are allocated before timing starts. One HDF5 reader thread
publishes into a three-slot pinned ring while the consumer enqueues H2D copies
and transforms on one nonblocking stream. Timing uses two fixed 64-chunk banks:
four CUDA events per slot, 512 events total, independent of `--chunks`. There
is no per-chunk host wait and no device-wide synchronization in the normal
measured path. The host waits on a bank's final ordered event only when that
bank must be recycled, then waits on the final ordered event after all chunks.

Pipeline chunks/s is measured chunks divided by wall time from reader release
through final CUDA completion. Pipeline logical GiB/s is the sum of canonical
decoded chunk bytes divided by that same wall time. Stage p50/p95 remains
per-chunk timing and is not additive because HDF5 is overlapped with the CUDA
queue.

The CLI validates `--chunks` as an unsigned decimal integer in
`[1, 1000000]` and `--warmup` in `[0, 1000000]`. It defaults to the first
readable science layer; the documented throughput runs explicitly select
science layers. An explicit path is not role-enforced. `sat-bench` does not
automatically read the sibling validity mask and
does not include CUDA/Vulkan interop or presentation.

`sat-bench` also does not exercise the continuous viewer camera or the
regional/Fit Scene LOD-page builder; those require viewer measurements.

## Current diskless overview and zoom pass (2026-08-08)

The overview builder now validates bounds once per sampled block and uses
fixed-size strided copies (or a row copy when contiguous). Exact-output tests
pass. On the Release Fit Scene smoke, three-run warm-OS observations changed
GCOV from a 252.464 ms baseline median to 246.231 ms, and GSLC from 7386.63 ms
to 5924.73 ms. A later GSLC repeat ranged from 6529.55 to 7272.05 ms (median
6610.63 ms), so storage and OS-cache variance is material; the observed median
gain is 10.5-19.8%, not a guaranteed fixed speedup.

Zoom scheduling now reuses the single prepared or in-flight in-memory page
after an exact request boundary while coverage remains complete and density is
at least one texel per physical framebuffer pixel. The planner still targets
1.25, creating a bounded hysteresis band before a finer rebuild. Navigation
tests cover reuse across an LOD boundary and expiry below the density floor.
This adds no page cache, disk writes, or extra resident page.

The v0.6.0 blank launcher initializes only a 1x1 scientific backing resource;
product-sized resident allocations and HDF5 workers are deferred until a file
is selected. A four-frame blank Release launch exits successfully, and the
same 4x4 GCOV smoke passed at 41.95 ms HDF5, 0.666 ms upload, and 0.977 ms
CUDA. Forced CPU in that build measured 51.01 ms HDF5, 2.22 ms upload, and
2.85 ms processing; the CUDA-free automatic path also passed.

## Overlapped pipeline results

| Product / layer | Measured / warmup chunks | Pipeline ms | Chunks/s | Logical GiB/s | HDF5 p50/p95 ms | H2D p50/p95 ms | CUDA p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCOV frequency A `HHHH` | 256 / 8 | 407.749 | 627.837 | 0.602513 | 1.3637 / 2.3625 | 0.077824 / 0.085536 | 0.034944 / 0.057184 |
| GSLC `025_128`, frequency A `HH` | 128 / 8 | 76.5036 | 1673.12 | 3.23511 | 0.5797 / 0.6853 | 0.098464 / 0.113440 | 0.035232 / 0.094336 |

These are end-to-end reader-to-final-CUDA-event throughput measurements for the
benchmark pipeline. They do not include the validity mask or Vulkan
presentation and are not viewer FPS.

## Nsight Systems device evidence

A Release GCOV pipeline trace with 128 measured chunks and 8 warmup chunks
captured 136 H2D operations and 136 transform-kernel launches. The Nsight
Systems device summaries reported:

| GPU operation | Operations | Mean device duration |
| --- | ---: | ---: |
| H2D copy | 136 | 26.205 us |
| Scientific transform kernel | 136 | 1.010 us |

These are pure GPU execution durations from the profiler. They are not
`sat-bench` CUDA-event stage percentiles, CPU API durations, or pipeline wall
time, and should not be substituted for those measurements. Their scale
confirms that storage/read/decode—not GPU math—is the dominant current cost.

## Isolated per-chunk results

| Product / layer | Open/catalog ms | HDF5 p50 ms | HDF5 p95 ms | HDF5 GiB/s | Pinned H2D p50 ms | Pinned H2D p95 ms | CUDA p50 ms | CUDA p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCOV frequency A `HHHH` | 8.5389 | 1.3362 | 1.6249 | 0.690314 | 0.074592 | 0.083040 | 0.036640 | 0.052128 |
| GSLC `025_128`, frequency A `HH` | 7.3604 | 0.5317 | 0.6561 | 3.49690 | 0.093184 | 0.106464 | 0.033280 | 0.079904 |
| GSLC `025_169`, frequency A `HH` | 7.4467 | 0.5707 | 0.7628 | 3.26582 | 0.088192 | 0.117472 | 0.037472 | 0.061408 |

These measurements support two narrow conclusions:

1. The CUDA transform is not the current steady-state bottleneck for a 512 by
   512 source chunk.
2. Further latency work should first target native HDF5 read/decode behavior,
   adjacent-chunk reuse, and request scheduling, while retaining exact output
   checks.

The table is not an FPS claim. It excludes general UI/render time and should
not be added across columns as an end-to-end latency result because the
interactive pipeline overlaps independently ordered work.

## Exact native working-set scaling

These existing measurements exercise only the exact 1x1/2x2/4x4 viewer mosaic
path, including exact sibling mask reads, per-chunk 2D H2D placement, one
complete-mosaic CUDA transform, CUDA/Vulkan interop, and four valid presented
frames. They do not measure regional or Fit Scene LOD-page construction, cache
loading, or transitions. They used the RelWithDebInfo `dev` build on the RTX
5090. Each
product/zoom pair was first warmed once, then measured in five fresh viewer processes; the table reports
the median printed stage values. The OS file cache was warm. Process startup and
four-frame presentation wall time are not part of the printed stage totals.

For multi-chunk views, HDF5 and H2D are sums over all chunks in the footprint;
CUDA is one transform over the completed mosaic.

| Product / default layer | Footprint | Chunks | HDF5 total median ms | H2D total median ms | CUDA median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCOV `026_012` / `HHHH` | 1x1 | 1 | 3.15 | 0.09 | 0.89 |
| GCOV `026_012` / `HHHH` | 2x2 | 4 | 10.99 | 0.21 | 1.06 |
| GCOV `026_012` / `HHHH` | 4x4 | 16 | 43.31 | 0.86 | 0.86 |
| GSLC `025_128` / `HH` | 1x1 | 1 | 7.63 | 0.12 | 1.02 |
| GSLC `025_128` / `HH` | 2x2 | 4 | 27.83 | 0.37 | 1.13 |
| GSLC `025_128` / `HH` | 4x4 | 16 | 103.96 | 1.40 | 0.98 |

The footprint grows by 4x and 16x in source chunks. Native HDF5 read/decode
therefore dominates scaling, while the full 4x4 H2D sum stays at or below
1.40 ms in these medians and the one transform remains near 1 ms. This is the
intended GPU-first balance: larger host allocation is avoided, assembly is
direct to persistent device mosaics, and the view is published once.

Continuous-camera native navigation uses the existing 4x4 capacity as a
guard band. An already-published native mosaic is retained while it contains
the visible window; after the first automatic upgrade, ordinary pans inside
that guard are shader-only. Crossing its boundary still schedules a bounded
replacement mosaic. Explicit `--zoom` and chunk-footprint requests remain
exact 1x1, 2x2, or 4x4 loads for reproducible inspection and benchmarking.

### Optimized Release end-to-end validation

The final `Release` build passed `--smoke-test --zoom 4` on all seven real
fixtures. Every view was 2048 by 2048 native pixels, assembled 16 science/mask
chunks, transformed once, and presented four valid frames.

| Product group | Products | HDF5 total ms | H2D total ms | CUDA ms |
| --- | ---: | ---: | ---: | ---: |
| GCOV `026_012` | 1 | 42.2745 | 0.824224 | 0.924352 |
| GSLC (six fixtures) | 6 | 106.867-119.240 | 0.986336-1.40838 | 0.660416-1.26048 |

These are smoke/startup observations, not steady-state FPS claims. They validate
the native HDF5 science-plus-mask reads, chunk-sized pinned-ring reuse, direct
2D device assembly, one CUDA transform, external timeline handoff, Vulkan
buffer-to-image copy, shader draw, and presentation together.

## Historical v0.3.0 full-scene overview baseline

The retired v0.3.0 full-scene path had a deliberately different cost profile
from the exact native tables above. The v0.3.0 `Release` viewer produced these
smoke observations on the RTX 5090. They are preserved as a historical baseline
and are not v0.3.1 measurements:

| Source path | Source rowsxcolumns / stride | Resident widthxheight | Cache state | Prepare/HDF5 ms | H2D ms | CUDA ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| GCOV `026_012`, exact native 4x4 | 17892x18108 / 1x1 | 2048x2048 | Not applicable | 42.2745 (HDF5) | 0.824 | 0.924 |
| GCOV `026_012`, Fit Scene | 17892x18108 / 9x9 | 2012x1988 | Cache hit | 34.2602 | 2.012 | 1.169 |
| GSLC `026_006`, Fit Scene | 67104x34200 / 34x17 | 2012x1974 | Cache hit | 59.5305 | 3.434 | 1.187 |
| GSLC `025_128`, Fit Scene, cold then warm | 67824x34488 / 34x17 | 2029x1995 | Built / cache hit | 43,798.6 / 57.9339 | 3.019-3.345 | 0.974-1.177 |

These are single smoke observations, not medians, throughput benchmarks, or
FPS. Each smoke waited for the requested source and four valid post-upload
frames, but UI frame rate was not measured. The OS filesystem cache was not
reset or controlled, so “cold” and “warm” above refer only to absence/presence
of the application's validated `.svo` entry. The native GCOV row is context for
one exact-path run; it does not replace the multi-process medians above.

The v0.3.0 plan chose independent integer row and column strides from physical
sample spacing, with neither output axis larger than 2048. Its cache stored raw
float32/complex-float32 science and, when present, the exactly co-sampled
uint8 mask. It did not store transformed dB, phase, RGB, or a product-sized
raster.

On a cold v0.3.0 application cache, the native HDF5 worker visited all source
chunks and retained only samples on the globally anchored lattice.
Shuffle/deflate filters still required source-chunk decode; sparse selection
reduced the bounded output and GPU upload but did not make the first compressed
scan free. Cold cost therefore depended on source dimensions, chunk/filter
layout, storage, and OS cache state. Progress was reported per source chunk.
The worker yielded to foreground exact reads and observed cancellation between
chunks, but that build was complete-before-display and could not resume a partial
result.

During a build, sampled science and mask values are collected directly into the
configured bounded output buffers. They are neither checksummed nor written to
disk. The completed in-memory page is uploaded once and transformed on CUDA.
After that source is resident, camera pan/zoom is shader-only;
scientific-mode changes reran CUDA from the raw overview without HDF5, and
window/gamma/palette changes remained draw-only.

The correctness suite gates output bounds, physical row/column strides, exact
globally anchored science/mask co-sampling, deterministic rebuilds,
cancellation, and absence of partial output. Real-product Fit Scene smokes validate raw overview
upload, CUDA transformation, external-timeline handoff, Vulkan copy/draw, and
four valid presented frames. None of those checks turns a single observation
into a general latency or FPS claim.

## v0.3.1 guarded regional LOD validation

Fresh `Release` observations on the RTX 5090 are recorded below. They are
single-run validation measurements, not general latency or FPS claims. Storage
and OS-cache state were not controlled beyond using a fresh application cache
for the cold Fit Scene run and reusing it for the warm run.

In each promoted run, verify these behavioral gates before recording timing:

- the page contains exact sparse source samples with science and mask on one
  request-aligned lattice;
- the guarded regional page is no larger than 4096x4096;
- the planner achieves at least 1.25 page texels per physical framebuffer pixel
  when resident capacity permits and records when capacity selects a coarser LOD;
- no replacement is displayed until its complete page is ready; and
- overlapping coverage crossfades for 120 ms while singly covered pixels retain
  their valid resident source.

| Scenario | Source/layer | Page origin and source coverage | Stride and output dimensions | Cache state | Prepare/HDF5 ms | H2D ms | CUDA ms | Transition/frame-pacing result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Fit Scene, cold cache | GCOV 026/012, frequency A `HHHH` | origin 0,0; coverage 17892x18108 | stride 5x5; output 3579x3622 | Built | 2787.57 | 7.14666 | 1.1967 | Four valid post-upload frames; 4.04796 page texels per physical framebuffer pixel; no outgoing source on initial publication |
| Fit Scene, warm cache | Same GCOV layer | origin 0,0; coverage 17892x18108 | stride 5x5; output 3579x3622 | Hit | 103.824 | 5.80624 | 1.47462 | Four valid post-upload frames; 4.04796 page texels per physical framebuffer pixel; no outgoing source on initial publication |
| Fit Scene, cold cache | GSLC 026/006, frequency A `HH` complex layer | origin 0,0; coverage 67104x34200 | stride 17x9; output 3948x3800 | Built | 29113.2 | 11.8794 | 1.6344 | Four valid post-upload frames; 4.21719 page texels per physical framebuffer pixel; no outgoing source on initial publication |
| Fit Scene, warm cache | Same GSLC layer | origin 0,0; coverage 67104x34200 | stride 17x9; output 3948x3800 | Hit | 210.611 | 12.3017 | 1.47322 | Four valid post-upload frames; 4.21719 page texels per physical framebuffer pixel; no outgoing source on initial publication |
| Continuous wheel zoom across LOD levels | Same GCOV product; default renderable layer | Five exact regional page identities during the cold interactive run | Request-specific power-of-two strides; each output bounded to 4096x4096 | Five builds on the first replay; zero new cache writes on the recorded warm replay | Not instrumented per page | Not instrumented per page | Not instrumented per page | PID-scoped replay sent 14 wheel-out and 8 wheel-in events in a 1456x938 window; the 10-second, 30 FPS recording had no observed legacy hard swap or axis-aligned resident gap. Across 100 viewport samples at 10 Hz, the maximum adjacent valid-detail gradient ratio was 1.107 versus about 9.05 in the reported v0.3.0 recording |
| Pan across a regional guard boundary | Same warm replay | 192-pixel drag within the replayed view | Existing ready regional identities | Zero new cache writes during the recorded replay | Not instrumented per page | Not instrumented per page | Not instrumented per page | Viewer exited successfully; no additional resident gap was observed during the drag |

For each row, also record logical viewport size, SDL framebuffer scale, achieved
page-texel density on each physical axis, driver/toolkit, storage and OS-cache
state, request-to-ready latency, presented-frame pacing, and observed crossfade
duration. Compare Smooth and Exact Pixels visually, but distinguish that display
sampling choice from the exact sparse cache payload.

The interactive replay is visual and image-gradient evidence rather than an FPS
benchmark. It does not instrument the exact number or duration of crossfades,
and it does not prove that every possible page boundary is imperceptible.

## v0.4.0 filters, distribution, and exact I/O optimization

These are fresh optimized `Release` observations on the RTX 5090. Kernel values
are CUDA-event timings; viewer values are single startup smokes through real
HDF5, CUDA, Vulkan interop, and four valid presented frames. They are not FPS
claims. **None** bypasses the filter API entirely, so the default transform and
output path remain the prior exact path.

### GPU feature cost

The standalone filter benchmark warms three launches, then averages 20 launches
with no validity-mask pointer. Correctness tests separately cover real NISAR mask
semantics. Each row is a 5x5 resident-grid window:

| Resident grid / domain | Boxcar ms | Lee ms |
| --- | ---: | ---: |
| 2048x2048 amplitude | 0.099867 | 0.108658 |
| 2048x2048 linear power | 0.092614 | 0.099629 |
| 2048x2048 power dB | 0.084882 | 0.091904 |
| 4096x4096 amplitude | 0.453216 | 0.452875 |
| 4096x4096 linear power | 0.389506 | 0.417736 |
| 4096x4096 power dB | 0.363805 | 0.390722 |

The 4096x4096 distribution golden scanned all 16,777,216 samples in
`0.220544 ms`. Its finite/invalid counts and extrema are exact; p1/p2/p50/p98/p99
are estimates from 256 bins. Host transfer is a fixed approximately 2 KiB pinned
summary, not a scalar-page readback, and the scan occurs once per transformed
resident request rather than once per frame.

GPU tests compare Boxcar and Lee to CPU references across 3x3/5x5/7x7 windows,
amplitude/power/dB equivalence, masks, non-finite and negative inputs, clipped
borders, all-invalid neighborhoods, constant and impulse fields, ENL response,
extreme finite values, validation errors, and bitwise determinism. CUDA 13.3
Compute Sanitizer was attempted in three bounded runs but did not finish within
180 seconds, so no sanitizer result is claimed.

### Real viewer publication

| Scenario | Resident rowsxcolumns / stride | Prepare/HDF5 ms | H2D ms | CUDA processing ms |
| --- | ---: | ---: | ---: | ---: |
| GCOV exact 4x4, None | 2048x2048 / 1x1 | 42.2775 | 0.701856 | 0.899744 |
| GSLC exact 4x4, None, six products | 2048x2048 / 1x1 | 104.000-114.947 | 0.983904-1.26653 | 0.728160-1.09965 |
| GCOV exact 4x4, 5x5 Boxcar | 2048x2048 / 1x1 | 41.7425 | 0.659776 | 1.88182 |
| GSLC 025/128 exact 4x4, 7x7 Lee, ENL 3.5 | 2048x2048 / 1x1 | 104.714 | 1.52320 | 1.73693 |
| GCOV Fit Scene cache hit, 5x5 Boxcar | 3579x3622 / 5x5 | 102.158 | 5.73206 | 2.38742 |
| GSLC 026/006 Fit Scene cache hit, 5x5 Lee, ENL 3 | 3948x3800 / 17x9 | 202.686 | 12.4863 | 2.22653 |

The integrated CUDA column includes the scientific transform and, when enabled,
the first filter launch in a fresh process; it therefore should not be
substituted for the warmed kernel microbenchmark. The Fit Scene filter window is
5x5 in the labeled sparse resident grid. It spans 21x21 native pixels for the
GCOV stride-5 page and 69x37 native pixels for the GSLC stride-17x9 page; neither
is presented as a dense native 5x5 filter.

All seven default-path Release smokes passed. Additional Boxcar and Lee smokes
passed on both exact native and complete-raster sparse LOD sources. Release CTest
passed all three targets: core/integration/distribution, navigation, and the
standalone CUDA speckle suite.

### Exact HDF5 handle-reuse optimization

The retained-dataset cache now reuses its file dataspace and one bounded
latest-shape memory dataspace under the existing HDF5 mutex. It removes repeated
`H5Dget_space`/`H5Screate_simple`/close churn; it does not change selections,
decoded values, cache keys, mask alignment, or publication rules.

| Representative exact cache payload | SHA-256 before and after |
| --- | --- |
| GCOV | `CF98B0BB2003B16D196F3998B6864C693B484B9262618E1B3CF2BBA4E537C2D8` |
| GSLC | `6C96C2C297FE226E3BA16DC755D657FF239286174B0421CF0D584593970FDF8B` |

The controlled GSLC cold Fit Scene path improved from `30.94 s` to
`28.40-28.85 s` (about 7.5%); its warm path improved from `211.5 ms` to
`205.9 ms`. GCOV cold results were neutral within run-to-run noise. Profiling of
the optimized GSLC build attributed 26,992 ms to science HDF5 read/decode and
1,206 ms to mask HDF5 work, versus 254 ms sample copies, 148 ms hashes, and
102 ms write/flush. Combined HDF5 decode/read remains about 98% of cold time.

A separate warm-OS-cache 128-chunk pipeline observation produced:

| Product/layer | Pipeline ms | Chunks/s | HDF5 p50/p95 ms | H2D p50/p95 ms | CUDA p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCOV `HHHH` | 189.522 | 675.384 | 1.3572 / 2.3637 | 0.079616 / 0.093696 | 0.037408 / 0.049856 |
| GSLC 025/128 `HH` | 71.4556 | 1791.32 | 0.5494 / 0.6159 | 0.094208 / 0.114240 | 0.036800 / 0.051136 |

Those two pipeline rows are current-state observations, not controlled
before/after attribution; OS-cache warmth can move them materially. They also
exercise ordinary `read_into()`, which intentionally retains HDF5's
decompressed raw-chunk cache for repeated exact native tiles.

## v0.4.1 exact cold-overview acceleration

The one-pass overview builder now has a narrowly gated raw-chunk path. It is
eligible only when `H5Tequal` proves the file bytes already match the native
memory representation, the plan covers one physical chunk, decoded storage is
at most 16 MiB, and the filter pipeline is exactly DEFLATE or
shuffle-then-DEFLATE with the expected element width. Unsupported, unallocated,
oversized, multichunk, differently filtered, or conversion-requiring reads use
ordinary `H5Dread`.

The installed HDF5 reports thread safety disabled. Chunk lookup and
`H5Dread_chunk2` therefore remain serialized under the process-wide HDF5/read
locks. The locks are released before exact DEFLATE reversal, unshuffle, and
disjoint sparse-output copies. The builder uses at most eight decode workers,
also bounded by hardware, source-job count, and the request's aggregate scratch
limit. Filter masks reported by chunk lookup and raw read must agree. Progress
callbacks remain serialized and monotonic, and cancellation still prevents
partial cache publication.

An isolated 256-chunk scaling probe selected the eight-worker cap. Raw chunks
were acquired once, every decoded sample buffer contributed to a deterministic
checksum, and all worker counts produced the same checksum. These are sizing
measurements, not end-to-end viewer timings:

| Product/layer | Serial raw acquisition ms | 1-worker decode ms | 8-worker decode ms |
| --- | ---: | ---: | ---: |
| GCOV `HHHH` | 27.346 | 186.554 | 25.965 |
| GSLC 025/128 `HH` | 90.615 | 645.959 | 88.637 |

The controlled Release A/B used the same products, view requests, machine, and
isolated application-cache procedure. “Cold” means the exact `.svo` entry was
absent; it does not claim a reset operating-system disk cache. Current ranges
contain three independent v0.4.1 runs:

| Product / Fit Scene layout | v0.4.0 cold ms | v0.4.1 cold range ms | v0.4.1 median ms | Median speedup | Warm median, old -> new ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCOV 026/012, 3579x3622 / stride 5x5 | 2780.06 | 363.112-519.803 | 367.213 | 7.57x | 103.109 -> 100.444 |
| GSLC 026/006, 3948x3800 / stride 17x9 | 30656.9 | 6988.90-8211.96 | 7388.74 | 4.15x | 217.311 -> 219.671 |

Every old/new and repeated output cache was byte-identical:

| Product | SHA-256 |
| --- | --- |
| GCOV | `CF98B0BB2003B16D196F3998B6864C693B484B9262618E1B3CF2BBA4E537C2D8` |
| GSLC | `6C96C2C297FE226E3BA16DC755D657FF239286174B0421CF0D584593970FDF8B` |

The generic interactive read path did not adopt raw-chunk decoding, so its hot
HDF5 cache behavior remains intact. Validation includes direct-versus-`H5Dread`
byte comparisons on one GCOV and two GSLC products, strict native-layout and
filter-mask guards, compressed synthetic interior and partial-edge chunks,
multichunk/unsupported fallback, corruption and cancellation, repeated
determinism, CPU-only tests, and the full CUDA/Vulkan Release suite. GDAL is not
used or packaged.

## v0.5.1 cache checksum acceleration

Cache v2 replaces only the large science and mask payload checksums with
XXH3-128. The source identity, metadata key, sparse sampling, stored science and
mask bytes, atomic publication, and post-load validation rules are unchanged.
The small metadata/header identities retain FNV-1a. Legacy v1 entries use a
distinct filename/magic/version and are rebuilt rather than interpreted as v2.
The stronger 128-bit checksum is also less collision-prone than the retired
64-bit payload checksum.

The controlled Release A/B alternated the retained v1 and v2 binaries on the
same products and view requests. Each cold sample used a new application-cache
directory; the operating-system file cache was not reset. Warm samples reused
the matching validated entry. Values are three-run medians:

| Product / Fit Scene layout | Cold v1 range / median ms | Cold v2 range / median ms | Cold speedup | Warm v1 range / median ms | Warm v2 range / median ms | Warm speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GCOV 026/012, 3579x3622 / stride 5x5 | 380.991-385.194 / 384.630 | 327.380-337.844 / 334.597 | 1.15x | 98.1230-100.628 / 99.8935 | 51.3213-58.3679 / 54.0865 | 1.85x |
| GSLC 026/006, 3948x3800 / stride 17x9 | 6832.45-7791.38 / 7373.10 | 6783.36-6985.87 / 6803.23 | 1.08x | 202.219-212.379 / 209.509 | 97.0386-110.430 / 100.898 | 2.08x |

The v1 and v2 headers differ by design, so parity was computed over the complete
combined science-plus-mask payload after the metadata boundary:

| Product | Payload bytes | SHA-256, v1 and v2 |
| --- | ---: | --- |
| GCOV | 64,815,690 | `EFA2CA28ADAEAC75D400F56764C11464DB4FE3D45D13B975646712EC1BA9EBD9` |
| GSLC | 135,021,600 | `44289F25EF35125B4ADB0CC7825E35730B0610034074D96A48EA85D3B0A4EC5A` |

Validation covers the official empty XXH3-128 vector, one-shot versus 64 KiB
streaming equality, explicit v1 rejection/rebuild, same-size payload corruption,
atomic recovery, all three Release CTest targets, and cold/warm Vulkan smokes on
the two real products above.

## Correctness gates

Performance changes must retain:

- exact real-data center-pixel goldens for one GCOV and two GSLC fixtures;
- correct dimensions, EPSG metadata, sample spacing, chunks, filters, and mask
  semantics;
- CUDA transform, invalid-mask, non-finite, and normalized-correlation cases;
- CPU-reference Boxcar/Lee agreement for amplitude, linear power, and dB across
  3x3/5x5/7x7 windows, exact mask semantics, clipped borders, ENL response,
  extreme finite values, and deterministic repeats;
- a zero-filter-kernel **None** path that preserves the original transform;
- exact distribution finite/invalid counts, extrema, and histogram accounting,
  with percentile/preset behavior bounded by the documented 256-bin estimate;
- request identity that includes filter type, window, and Lee ENL so stale GPU
  pages or distributions cannot be mislabeled;
- byte-identical LOD cache output across HDF5 handle reuse, direct decoding, and
  cache-checksum version changes;
- direct-versus-`H5Dread` equality plus strict native-layout, filter-pipeline,
  filter-mask, one-chunk, allocation, and decoded-size gates with safe fallback;
- canonical 1x1/2x2/4x4 geometry, edge shifting, partial bottom/right chunks,
  overflow/safety caps, and differently aligned mask staging bounds;
- continuous rotated-camera fit/clamp, cursor-anchor zoom, rotated drag-pan,
  physical-aspect, draw/source mapping, and exact-native-versus-LOD
  resident-selection invariants;
- guarded regional power-of-two selection, the 1.25-texel physical-pixel target,
  4096x4096 capacity fallback, globally quantized origins, and level alignment;
- exact sparse science/mask co-sampling, source-window/stride cache identity,
  cache validation/corruption recovery, stale-completion rejection, and
  cancellation without a partial final entry;
- validity-authoritative Smooth sampling, finite-neighbor renormalization,
  circular phase interpolation, and nearest-texel Exact Pixels;
- ready-only dual-image publication and a coverage-aware 120 ms crossfade that
  retains valid singly covered pixels;
- exact 20-row palette IDs, RGBA endpoints, grayscale ramp, cyclic seam, and
  LUT indexing; and
- successful real-product Release smokes through CUDA/Vulkan presentation for
  each source path whose performance is being promoted.

Both far and near paths remain on the product-native grid. A faster path is not
a promotion if it changes exact sparse stored values, de-synchronizes the
science/mask lattice, treats invalid samples as valid, confuses Smooth display
interpolation with a radiometric aggregate, or introduces reprojection, basemap
work, geographic resampling, or a hidden GDAL path.

## Reproduce

Reproduce the independent-chunk throughput benchmarks:

```powershell
.\build\preset-dev\RelWithDebInfo\sat-bench.exe "E:\path\to\NISAR_product.h5" `
  --layer "/science/LSAR/GCOV/grids/frequencyA/HHHH" `
  --chunks 256 --warmup 8 --pipeline

.\build\preset-dev\RelWithDebInfo\sat-bench.exe "E:\path\to\NISAR_product.h5" `
  --layer "/science/LSAR/GSLC/grids/frequencyA/HH" `
  --chunks 128 --warmup 8 --pipeline
```

Append `--csv` for machine-readable output; omit `--pipeline` for isolated
per-chunk timing. These benchmark modes exclude masks, mosaic assembly, interop,
and presentation.

Reproduce the exact native viewer footprints:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --zoom 1 "E:\path\to\NISAR_product.h5"
.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --zoom 2 "E:\path\to\NISAR_product.h5"
.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --zoom 4 "E:\path\to\NISAR_product.h5"
```

Reproduce the filter and distribution tests/microbenchmarks, then exercise the
integrated filtered path:

```powershell
.\build\preset-release\Release\satview-speckle-tests.exe
.\build\preset-release\Release\satview-tests.exe

.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --zoom 4 `
  --speckle boxcar --speckle-window 5 "E:\path\to\NISAR_GCOV.h5"

.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --fit-scene `
  --speckle lee --speckle-window 5 --speckle-looks 3 `
  "E:\path\to\NISAR_GSLC.h5"
```

Reproduce the Fit Scene LOD-page publication and four valid frames:

```powershell
.\build\preset-release\viewer\Release\sat-viewer.exe --smoke-test --fit-scene "E:\path\to\NISAR_product.h5"
```

This builds the Fit Scene page in memory on each launch; no persistent cache is
created.
Fit-scene smoke mode allows up to five minutes for the first compressed scan.

For the continuous wheel-zoom and guard-boundary rows, run the Release viewer
without smoke mode, record logical viewport and physical framebuffer dimensions,
then zoom and pan across adjacent LOD requests. Record request-to-ready latency
and frame pacing through the coverage-aware 120 ms transition.

For comparable results, record product/layer, build type, warmup strategy, CUDA
driver/toolkit, storage location, OS cache state, selected footprint/source
kind, page origin and source coverage, cache state, stride/output dimensions,
physical-pixel density, sampling mode, and transition result.
