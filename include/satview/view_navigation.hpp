#pragma once

#include <cstdint>

namespace satview::viewer {

// Raster coordinates are continuous pixel-edge coordinates: row in [0, rows]
// and column in [0, columns]. Pixel centers lie at integer + 0.5. Navigation
// uses spacing magnitudes because screen rows and columns increase down/right.
struct RasterMetrics {
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    double row_spacing = 0.0;
    double column_spacing = 0.0;
};

struct ScreenViewport {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct Camera2D {
    double center_row = 0.0;
    double center_column = 0.0;
    // Physical raster units represented by one screen pixel. Larger values
    // zoom out and smaller values zoom in.
    double world_units_per_screen_pixel = 0.0;
    // Clockwise image rotation in radians. Navigation canonicalizes this to
    // [-pi, pi].
    double rotation_radians = 0.0;
};

struct ScreenPoint { double x = 0.0; double y = 0.0; };
struct RasterPoint { double row = 0.0; double column = 0.0; };

struct RasterWindow {
    double row_begin = 0.0;
    double column_begin = 0.0;
    double row_end = 0.0;
    double column_end = 0.0;
};

struct PixelWindow {
    std::uint64_t row = 0;
    std::uint64_t column = 0;
    std::uint64_t height = 0;
    std::uint64_t width = 0;
    [[nodiscard]] bool operator==(const PixelWindow&) const = default;
};

struct PixelExtent {
    std::uint64_t height = 0;
    std::uint64_t width = 0;
    [[nodiscard]] bool operator==(const PixelExtent&) const = default;
};

struct OverviewRequest {
    std::uint32_t level = 0;
    std::uint64_t sample_stride = 1;
    // Source coverage aligned to the overview grid and clipped at the raster.
    PixelWindow source_window;
    // The same coverage in coordinates of the complete overview level.
    PixelWindow level_window;
    PixelExtent level_extent;
    [[nodiscard]] bool operator==(const OverviewRequest&) const = default;
};

struct GuardedOverviewOptions {
    std::uint32_t maximum_level = 63;
    std::uint64_t page_extent = 512;
    std::uint64_t maximum_resident_extent = 4096;
    // This is expressed per logical viewport pixel. A DPI-aware caller should
    // multiply its physical-pixel target by the framebuffer scale.
    double target_texels_per_logical_pixel = 1.25;
    // Fraction of resident pages reserved as total pan guard across an axis.
    double guard_fraction = 0.25;
};

// A deterministic page-aligned working set around the visible overview
// request. Each resident axis is the smallest visible-pages-plus-guard span,
// bounded by the configured capacity and raster extent, and remains on the
// same global power-of-two sample lattice as `visible`.
struct GuardedOverviewRequest {
    OverviewRequest visible;
    PixelWindow resident_source_window;
    PixelWindow resident_level_window;

    [[nodiscard]] bool operator==(
        const GuardedOverviewRequest&) const = default;
};

// Furthest-out camera containing the complete rotated raster. The shorter
// screen axis is letterboxed.
[[nodiscard]] Camera2D fit_camera(
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    double rotation_radians = 0.0);

// Prevents zooming farther out than fit and clamps each center axis. There is
// no built-in maximum magnification; the UI may impose a minimum scale first.
[[nodiscard]] Camera2D clamp_camera(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport);

[[nodiscard]] RasterPoint screen_to_raster(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const ScreenPoint& screen);

[[nodiscard]] ScreenPoint raster_to_screen(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const RasterPoint& point);

// scale_factor < 1 zooms in and > 1 zooms out. The source coordinate under
// anchor_screen stays fixed unless raster-edge clamping is required.
[[nodiscard]] Camera2D zoom_about(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const ScreenPoint& anchor_screen,
    double scale_factor);

// Positive deltas mean the image was dragged right/down. The camera center
// moves oppositely through the rotated raster coordinate system.
[[nodiscard]] Camera2D pan_by_screen_delta(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    double delta_x,
    double delta_y);

[[nodiscard]] RasterWindow visible_raster_window(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport);

// Axis-aligned screen-space bounds occupied by valid rotated data, clipped to
// the viewport.
[[nodiscard]] ScreenViewport visible_raster_screen_rect(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport);

// Coarsest power-of-two overview whose source spacing is no larger than one
// screen pixel on either physical axis. Level zero is native resolution.
[[nodiscard]] std::uint32_t choose_overview_level(
    const Camera2D& camera,
    const RasterMetrics& raster,
    std::uint32_t maximum_level);

[[nodiscard]] PixelExtent overview_level_extent(
    const RasterMetrics& raster, std::uint32_t level);

// Level-aligned request covering the visible raster. This only plans; it never
// allocates or reads source data.
[[nodiscard]] OverviewRequest make_overview_request(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    std::uint32_t level);

// Minimum resident-texel density across the raster axes, expressed per
// physical framebuffer pixel. framebuffer_scale_x/y are physical pixels per
// logical viewport pixel; row/column strides are source samples per resident
// texel. This is suitable for auditing the final capacity-constrained LOD.
[[nodiscard]] double minimum_resident_texels_per_framebuffer_pixel(
    const Camera2D& camera,
    const RasterMetrics& raster,
    std::uint64_t row_sample_stride,
    std::uint64_t column_sample_stride,
    double framebuffer_scale_x,
    double framebuffer_scale_y);

// True while an already-built sampled window still covers the visible raster
// and supplies at least the requested physical-pixel density. This lets the UI
// keep one resident overview across small pan/zoom changes without another
// cache or a rebuild on every camera update.
[[nodiscard]] bool sampled_window_remains_usable(
    const PixelWindow& source_window,
    std::uint64_t row_sample_stride,
    std::uint64_t column_sample_stride,
    const RasterWindow& visible,
    const Camera2D& camera,
    const RasterMetrics& raster,
    double framebuffer_scale_x,
    double framebuffer_scale_y,
    double minimum_density);

// Chooses a screen-density overview level, increasing it only when necessary
// for the visible window plus a stable page guard to fit in the configured
// resident texture. If the density target and guard cannot both fit, capacity
// wins and the planner selects a coarser power-of-two level. Resident origins
// are quantized to the global page lattice; source coordinates remain on the
// global power-of-two sampling lattice. Throws std::length_error if
// maximum_level is too small for the capacity.
[[nodiscard]] GuardedOverviewRequest make_guarded_overview_request(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const GuardedOverviewOptions& options = {});

}  // namespace satview::viewer
