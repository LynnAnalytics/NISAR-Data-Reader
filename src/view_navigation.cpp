#include "satview/view_navigation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace satview::viewer {
namespace {

struct Geometry {
    double rows;
    double columns;
    double row_spacing;
    double column_spacing;
};

struct Rotation {
    double cosine;
    double sine;
};

constexpr double pi = 3.141592653589793238462643383279502884;

[[nodiscard]] bool finite_positive(const double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] double canonical_rotation(const double radians) {
    if (!std::isfinite(radians)) {
        throw std::invalid_argument("navigation rotation must be finite");
    }
    const double normalized = std::remainder(radians, 2.0 * pi);
    return normalized == 0.0 ? 0.0 : normalized;
}

[[nodiscard]] Rotation rotation_for(const double radians) {
    const double normalized = canonical_rotation(radians);
    return {std::cos(normalized), std::sin(normalized)};
}

[[nodiscard]] double fit_scale(
    const Geometry& geometry,
    const ScreenViewport& viewport,
    const Rotation rotation) {
    const double physical_height = geometry.rows * geometry.row_spacing;
    const double physical_width = geometry.columns * geometry.column_spacing;
    if (!finite_positive(physical_height) || !finite_positive(physical_width)) {
        throw std::overflow_error("navigation physical extent overflow");
    }
    const double rotated_width =
        std::abs(rotation.cosine) * physical_width +
        std::abs(rotation.sine) * physical_height;
    const double rotated_height =
        std::abs(rotation.sine) * physical_width +
        std::abs(rotation.cosine) * physical_height;
    const double scale = std::max(
        rotated_height / viewport.height,
        rotated_width / viewport.width);
    if (!finite_positive(scale)) {
        throw std::overflow_error("navigation fit scale overflow");
    }
    return scale;
}

[[nodiscard]] RasterPoint screen_to_raster_unchecked(
    const Camera2D& camera,
    const Geometry& geometry,
    const ScreenViewport& viewport,
    const ScreenPoint& screen,
    const Rotation rotation) noexcept {
    const double center_x = viewport.x + 0.5 * viewport.width;
    const double center_y = viewport.y + 0.5 * viewport.height;
    const double screen_x =
        (screen.x - center_x) * camera.world_units_per_screen_pixel;
    const double screen_y =
        (screen.y - center_y) * camera.world_units_per_screen_pixel;
    const double physical_column =
        rotation.cosine * screen_x + rotation.sine * screen_y;
    const double physical_row =
        -rotation.sine * screen_x + rotation.cosine * screen_y;
    return {
        camera.center_row + physical_row / geometry.row_spacing,
        camera.center_column + physical_column / geometry.column_spacing};
}

[[nodiscard]] ScreenPoint raster_to_screen_unchecked(
    const Camera2D& camera,
    const Geometry& geometry,
    const ScreenViewport& viewport,
    const RasterPoint& point,
    const Rotation rotation) noexcept {
    const double physical_x =
        (point.column - camera.center_column) * geometry.column_spacing;
    const double physical_y =
        (point.row - camera.center_row) * geometry.row_spacing;
    const double center_x = viewport.x + 0.5 * viewport.width;
    const double center_y = viewport.y + 0.5 * viewport.height;
    return {
        center_x +
            (rotation.cosine * physical_x - rotation.sine * physical_y) /
                camera.world_units_per_screen_pixel,
        center_y +
            (rotation.sine * physical_x + rotation.cosine * physical_y) /
                camera.world_units_per_screen_pixel};
}

[[nodiscard]] Geometry validate(
    const RasterMetrics& raster,
    const ScreenViewport& viewport) {
    if (raster.rows == 0 || raster.columns == 0) {
        throw std::invalid_argument("navigation raster dimensions must be positive");
    }
    if (!std::isfinite(raster.row_spacing) ||
        !std::isfinite(raster.column_spacing) ||
        raster.row_spacing == 0.0 || raster.column_spacing == 0.0) {
        throw std::invalid_argument("navigation spacing must be finite and nonzero");
    }
    if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) ||
        !finite_positive(viewport.width) || !finite_positive(viewport.height)) {
        throw std::invalid_argument("navigation viewport must be finite and positive");
    }
    const Geometry result{
        static_cast<double>(raster.rows),
        static_cast<double>(raster.columns),
        std::abs(raster.row_spacing),
        std::abs(raster.column_spacing)};
    if (!finite_positive(result.rows) || !finite_positive(result.columns) ||
        !finite_positive(result.row_spacing) ||
        !finite_positive(result.column_spacing)) {
        throw std::overflow_error("navigation geometry is not representable");
    }
    return result;
}

[[nodiscard]] Geometry validate(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport) {
    const auto geometry = validate(raster, viewport);
    if (!std::isfinite(camera.center_row) ||
        !std::isfinite(camera.center_column) ||
        !finite_positive(camera.world_units_per_screen_pixel) ||
        !std::isfinite(camera.rotation_radians)) {
        throw std::invalid_argument("navigation camera must be finite and positive");
    }
    return geometry;
}

void validate(const ScreenPoint& point) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        throw std::invalid_argument("screen point must be finite");
    }
}

[[nodiscard]] bool covers(
    const double visible,
    const double dimension) noexcept {
    const double magnitude =
        std::max({1.0, std::abs(visible), std::abs(dimension)});
    return visible + 32.0 * std::numeric_limits<double>::epsilon() * magnitude >=
        dimension;
}

[[nodiscard]] double clamp_axis(
    const double center,
    const double visible,
    const double dimension) noexcept {
    if (covers(visible, dimension)) {
        return 0.5 * dimension;
    }
    const double half = 0.5 * visible;
    return std::clamp(center, half, dimension - half);
}

[[nodiscard]] std::uint64_t stride_for(const std::uint32_t level) {
    if (level > 63) {
        throw std::invalid_argument("overview level must be in [0, 63]");
    }
    return std::uint64_t{1} << level;
}

[[nodiscard]] constexpr std::uint64_t ceil_div(
    const std::uint64_t value,
    const std::uint64_t divisor) noexcept {
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

[[nodiscard]] std::uint64_t clipped_product(
    const std::uint64_t value,
    const std::uint64_t multiplier,
    const std::uint64_t limit) noexcept {
    return value > limit / multiplier ? limit : std::min(value * multiplier, limit);
}

[[nodiscard]] std::uint64_t clipped_floor(
    const double value,
    const std::uint64_t limit) {
    if (value <= 0.0) {
        return 0;
    }
    if (value >= static_cast<double>(limit)) {
        return limit;
    }
    return static_cast<std::uint64_t>(std::floor(value));
}

[[nodiscard]] std::uint64_t clipped_ceil(
    const double value,
    const std::uint64_t limit) {
    if (value <= 0.0) {
        return 0;
    }
    if (value >= static_cast<double>(limit)) {
        return limit;
    }
    return static_cast<std::uint64_t>(std::ceil(value));
}

struct PageSpan {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
};

[[nodiscard]] PageSpan touched_pages(
    const std::uint64_t begin,
    const std::uint64_t extent,
    const std::uint64_t page_extent) {
    if (extent == 0 || page_extent == 0) {
        throw std::invalid_argument(
            "overview page coverage must be non-empty");
    }
    const auto first = begin / page_extent;
    const auto last = (begin + extent - 1) / page_extent;
    return {first, last - first + 1};
}

[[nodiscard]] bool guarded_axis_fits(
    const PageSpan required,
    const std::uint64_t total_pages,
    const std::uint64_t resident_pages,
    const std::uint64_t maximum_visible_pages) noexcept {
    return required.count <=
        (total_pages <= resident_pages
             ? total_pages
             : maximum_visible_pages);
}

[[nodiscard]] std::uint64_t canonical_resident_page_start(
    const PageSpan required,
    const std::uint64_t total_pages,
    const std::uint64_t requested_resident_pages) {
    const auto actual_pages = std::min(requested_resident_pages, total_pages);
    if (required.count == 0 || required.count > actual_pages ||
        required.first >= total_pages ||
        required.count > total_pages - required.first) {
        throw std::logic_error(
            "visible overview pages cannot fit the resident page span");
    }
    const auto spare = actual_pages - required.count;
    const auto leading_guard = spare / 2 + spare % 2;
    const auto preferred = required.first > leading_guard
        ? required.first - leading_guard
        : 0;
    return std::min(preferred, total_pages - actual_pages);
}

}  // namespace

Camera2D fit_camera(
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const double rotation_radians) {
    const auto geometry = validate(raster, viewport);
    const double normalized = canonical_rotation(rotation_radians);
    const Rotation rotation{std::cos(normalized), std::sin(normalized)};
    return {
        0.5 * geometry.rows,
        0.5 * geometry.columns,
        fit_scale(geometry, viewport, rotation),
        normalized};
}

Camera2D clamp_camera(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport) {
    const auto geometry = validate(camera, raster, viewport);
    Camera2D result = camera;
    result.rotation_radians = canonical_rotation(camera.rotation_radians);
    const Rotation rotation{
        std::cos(result.rotation_radians),
        std::sin(result.rotation_radians)};
    result.world_units_per_screen_pixel = std::min(
        camera.world_units_per_screen_pixel,
        fit_scale(geometry, viewport, rotation));
    const double visible_rows =
        (std::abs(rotation.sine) * viewport.width +
         std::abs(rotation.cosine) * viewport.height) *
        result.world_units_per_screen_pixel / geometry.row_spacing;
    const double visible_columns =
        (std::abs(rotation.cosine) * viewport.width +
         std::abs(rotation.sine) * viewport.height) *
        result.world_units_per_screen_pixel / geometry.column_spacing;
    if (!finite_positive(visible_rows) || !finite_positive(visible_columns)) {
        throw std::overflow_error("navigation visible extent overflow");
    }
    result.center_row = clamp_axis(camera.center_row, visible_rows, geometry.rows);
    result.center_column =
        clamp_axis(camera.center_column, visible_columns, geometry.columns);
    return result;
}

RasterPoint screen_to_raster(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const ScreenPoint& screen) {
    const auto geometry = validate(camera, raster, viewport);
    validate(screen);
    return screen_to_raster_unchecked(
        camera, geometry, viewport, screen,
        rotation_for(camera.rotation_radians));
}

ScreenPoint raster_to_screen(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const RasterPoint& point) {
    const auto geometry = validate(camera, raster, viewport);
    if (!std::isfinite(point.row) || !std::isfinite(point.column)) {
        throw std::invalid_argument("raster point must be finite");
    }
    return raster_to_screen_unchecked(
        camera, geometry, viewport, point,
        rotation_for(camera.rotation_radians));
}

Camera2D zoom_about(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const ScreenPoint& anchor_screen,
    const double scale_factor) {
    const auto geometry = validate(camera, raster, viewport);
    validate(anchor_screen);
    if (!finite_positive(scale_factor)) {
        throw std::invalid_argument("zoom factor must be finite and positive");
    }
    if (anchor_screen.x < viewport.x ||
        anchor_screen.x > viewport.x + viewport.width ||
        anchor_screen.y < viewport.y ||
        anchor_screen.y > viewport.y + viewport.height) {
        throw std::out_of_range("zoom anchor is outside the viewport");
    }
    const auto canonical = clamp_camera(camera, raster, viewport);
    const auto rotation = rotation_for(canonical.rotation_radians);
    const auto anchor = screen_to_raster_unchecked(
        canonical, geometry, viewport, anchor_screen, rotation);
    const double scale =
        canonical.world_units_per_screen_pixel * scale_factor;
    if (!finite_positive(scale)) {
        throw std::overflow_error("navigation zoom scale overflow");
    }
    const double center_x = viewport.x + 0.5 * viewport.width;
    const double center_y = viewport.y + 0.5 * viewport.height;
    const double screen_x = (anchor_screen.x - center_x) * scale;
    const double screen_y = (anchor_screen.y - center_y) * scale;
    const double physical_column =
        rotation.cosine * screen_x + rotation.sine * screen_y;
    const double physical_row =
        -rotation.sine * screen_x + rotation.cosine * screen_y;
    return clamp_camera(
        Camera2D{
            anchor.row - physical_row / geometry.row_spacing,
            anchor.column - physical_column / geometry.column_spacing,
            scale,
            canonical.rotation_radians},
        raster,
        viewport);
}

Camera2D pan_by_screen_delta(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const double delta_x,
    const double delta_y) {
    const auto geometry = validate(camera, raster, viewport);
    if (!std::isfinite(delta_x) || !std::isfinite(delta_y)) {
        throw std::invalid_argument("pan delta must be finite");
    }
    auto result = clamp_camera(camera, raster, viewport);
    const auto rotation = rotation_for(result.rotation_radians);
    const double screen_x =
        delta_x * result.world_units_per_screen_pixel;
    const double screen_y =
        delta_y * result.world_units_per_screen_pixel;
    result.center_column -=
        (rotation.cosine * screen_x + rotation.sine * screen_y) /
        geometry.column_spacing;
    result.center_row -=
        (-rotation.sine * screen_x + rotation.cosine * screen_y) /
        geometry.row_spacing;
    return clamp_camera(result, raster, viewport);
}

RasterWindow visible_raster_window(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport) {
    const auto geometry = validate(camera, raster, viewport);
    const auto canonical = clamp_camera(camera, raster, viewport);
    const auto rotation = rotation_for(canonical.rotation_radians);
    const std::array corners{
        ScreenPoint{viewport.x, viewport.y},
        ScreenPoint{viewport.x + viewport.width, viewport.y},
        ScreenPoint{viewport.x, viewport.y + viewport.height},
        ScreenPoint{
            viewport.x + viewport.width,
            viewport.y + viewport.height}};
    double row_begin = std::numeric_limits<double>::infinity();
    double column_begin = std::numeric_limits<double>::infinity();
    double row_end = -std::numeric_limits<double>::infinity();
    double column_end = -std::numeric_limits<double>::infinity();
    for (const auto corner : corners) {
        const auto point = screen_to_raster_unchecked(
            canonical, geometry, viewport, corner, rotation);
        row_begin = std::min(row_begin, point.row);
        column_begin = std::min(column_begin, point.column);
        row_end = std::max(row_end, point.row);
        column_end = std::max(column_end, point.column);
    }
    return {
        std::clamp(row_begin, 0.0, geometry.rows),
        std::clamp(column_begin, 0.0, geometry.columns),
        std::clamp(row_end, 0.0, geometry.rows),
        std::clamp(column_end, 0.0, geometry.columns)};
}

ScreenViewport visible_raster_screen_rect(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport) {
    const auto geometry = validate(camera, raster, viewport);
    const auto canonical = clamp_camera(camera, raster, viewport);
    const auto rotation = rotation_for(canonical.rotation_radians);
    const std::array corners{
        RasterPoint{0.0, 0.0},
        RasterPoint{0.0, geometry.columns},
        RasterPoint{geometry.rows, 0.0},
        RasterPoint{geometry.rows, geometry.columns}};
    double minimum_x = std::numeric_limits<double>::infinity();
    double minimum_y = std::numeric_limits<double>::infinity();
    double maximum_x = -std::numeric_limits<double>::infinity();
    double maximum_y = -std::numeric_limits<double>::infinity();
    for (const auto corner : corners) {
        const auto point = raster_to_screen_unchecked(
            canonical, geometry, viewport, corner, rotation);
        minimum_x = std::min(minimum_x, point.x);
        minimum_y = std::min(minimum_y, point.y);
        maximum_x = std::max(maximum_x, point.x);
        maximum_y = std::max(maximum_y, point.y);
    }
    const double left = std::max(viewport.x, minimum_x);
    const double top = std::max(viewport.y, minimum_y);
    const double right = std::min(viewport.x + viewport.width, maximum_x);
    const double bottom = std::min(viewport.y + viewport.height, maximum_y);
    return {left, top, std::max(0.0, right - left),
            std::max(0.0, bottom - top)};
}

std::uint32_t choose_overview_level(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const std::uint32_t maximum_level) {
    const ScreenViewport unit{0.0, 0.0, 1.0, 1.0};
    const auto geometry = validate(camera, raster, unit);
    if (maximum_level > 63) {
        throw std::invalid_argument("maximum overview level must be <= 63");
    }
    const double limiting_samples = std::min(
        camera.world_units_per_screen_pixel / geometry.row_spacing,
        camera.world_units_per_screen_pixel / geometry.column_spacing);
    std::uint32_t level = 0;
    std::uint64_t stride = 1;
    while (level < maximum_level && level < 63 &&
           static_cast<double>(stride) <= 0.5 * limiting_samples) {
        stride <<= 1;
        ++level;
    }
    return level;
}

PixelExtent overview_level_extent(
    const RasterMetrics& raster,
    const std::uint32_t level) {
    static_cast<void>(validate(raster, ScreenViewport{0.0, 0.0, 1.0, 1.0}));
    const auto stride = stride_for(level);
    return {ceil_div(raster.rows, stride), ceil_div(raster.columns, stride)};
}

OverviewRequest make_overview_request(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const std::uint32_t level) {
    static_cast<void>(validate(camera, raster, viewport));
    const auto stride = stride_for(level);
    const auto visible = visible_raster_window(camera, raster, viewport);
    const auto row_begin = clipped_floor(visible.row_begin, raster.rows);
    const auto column_begin = clipped_floor(visible.column_begin, raster.columns);
    const auto row_end = clipped_ceil(visible.row_end, raster.rows);
    const auto column_end = clipped_ceil(visible.column_end, raster.columns);
    const auto level_row = row_begin / stride;
    const auto level_column = column_begin / stride;
    const auto level_row_end = ceil_div(row_end, stride);
    const auto level_column_end = ceil_div(column_end, stride);
    if (level_row_end <= level_row || level_column_end <= level_column) {
        throw std::logic_error("overview request became empty");
    }
    const auto source_row = clipped_product(level_row, stride, raster.rows);
    const auto source_column =
        clipped_product(level_column, stride, raster.columns);
    const auto source_row_end =
        clipped_product(level_row_end, stride, raster.rows);
    const auto source_column_end =
        clipped_product(level_column_end, stride, raster.columns);
    return {
        level,
        stride,
        PixelWindow{source_row, source_column,
                    source_row_end - source_row,
                    source_column_end - source_column},
        PixelWindow{level_row, level_column,
                    level_row_end - level_row,
                    level_column_end - level_column},
        overview_level_extent(raster, level)};
}

double minimum_resident_texels_per_framebuffer_pixel(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const std::uint64_t row_sample_stride,
    const std::uint64_t column_sample_stride,
    const double framebuffer_scale_x,
    const double framebuffer_scale_y) {
    const ScreenViewport unit{0.0, 0.0, 1.0, 1.0};
    const auto geometry = validate(camera, raster, unit);
    if (row_sample_stride == 0 ||
        column_sample_stride == 0) {
        throw std::invalid_argument(
            "resident sample strides must be positive");
    }
    if (!finite_positive(framebuffer_scale_x) ||
        !finite_positive(framebuffer_scale_y)) {
        throw std::invalid_argument(
            "framebuffer scales must be finite and positive");
    }

    const long double row_density_per_logical_pixel =
        static_cast<long double>(camera.world_units_per_screen_pixel) /
        static_cast<long double>(geometry.row_spacing) /
        static_cast<long double>(row_sample_stride);
    const long double column_density_per_logical_pixel =
        static_cast<long double>(camera.world_units_per_screen_pixel) /
        static_cast<long double>(geometry.column_spacing) /
        static_cast<long double>(column_sample_stride);
    const auto rotation = rotation_for(camera.rotation_radians);
    const long double cosine = rotation.cosine;
    const long double sine = rotation.sine;
    const long double matrix_00 =
        column_density_per_logical_pixel * cosine /
        static_cast<long double>(framebuffer_scale_x);
    const long double matrix_01 =
        column_density_per_logical_pixel * sine /
        static_cast<long double>(framebuffer_scale_y);
    const long double matrix_10 =
        -row_density_per_logical_pixel * sine /
        static_cast<long double>(framebuffer_scale_x);
    const long double matrix_11 =
        row_density_per_logical_pixel * cosine /
        static_cast<long double>(framebuffer_scale_y);
    const long double matrix_scale = std::max({
        std::abs(matrix_00),
        std::abs(matrix_01),
        std::abs(matrix_10),
        std::abs(matrix_11)});
    if (!(matrix_scale > 0.0L) || !std::isfinite(matrix_scale)) {
        throw std::overflow_error(
            "resident framebuffer density is not representable");
    }
    const long double a = matrix_00 / matrix_scale;
    const long double b = matrix_01 / matrix_scale;
    const long double c = matrix_10 / matrix_scale;
    const long double d = matrix_11 / matrix_scale;
    const long double frobenius_squared = a * a + b * b + c * c + d * d;
    const long double determinant = a * d - b * c;
    const long double discriminant = std::max(
        0.0L,
        frobenius_squared * frobenius_squared -
            4.0L * determinant * determinant);
    const long double maximum_eigenvalue = 0.5L *
        (frobenius_squared + std::sqrt(discriminant));
    const long double minimum = matrix_scale *
        std::abs(determinant) / std::sqrt(maximum_eigenvalue);
    if (!(minimum > 0.0L) ||
        minimum > static_cast<long double>(
            std::numeric_limits<double>::max())) {
        throw std::overflow_error(
            "resident framebuffer density is not representable");
    }
    const double result = static_cast<double>(minimum);
    if (!finite_positive(result)) {
        throw std::overflow_error(
            "resident framebuffer density is not representable");
    }
    return result;
}

bool sampled_window_remains_usable(
    const PixelWindow& source_window,
    const std::uint64_t row_sample_stride,
    const std::uint64_t column_sample_stride,
    const RasterWindow& visible,
    const Camera2D& camera,
    const RasterMetrics& raster,
    const double framebuffer_scale_x,
    const double framebuffer_scale_y,
    const double minimum_density) {
    if (!finite_positive(minimum_density)) {
        throw std::invalid_argument(
            "minimum resident density must be finite and positive");
    }
    const long double row_end =
        static_cast<long double>(source_window.row) +
        static_cast<long double>(source_window.height);
    const long double column_end =
        static_cast<long double>(source_window.column) +
        static_cast<long double>(source_window.width);
    const bool contains_visible =
        static_cast<long double>(visible.row_begin) >= source_window.row &&
        static_cast<long double>(visible.column_begin) >= source_window.column &&
        static_cast<long double>(visible.row_end) <= row_end &&
        static_cast<long double>(visible.column_end) <= column_end;
    return contains_visible &&
        minimum_resident_texels_per_framebuffer_pixel(
            camera,
            raster,
            row_sample_stride,
            column_sample_stride,
            framebuffer_scale_x,
            framebuffer_scale_y) >= minimum_density;
}

GuardedOverviewRequest make_guarded_overview_request(
    const Camera2D& camera,
    const RasterMetrics& raster,
    const ScreenViewport& viewport,
    const GuardedOverviewOptions& options) {
    if (options.maximum_level > 63) {
        throw std::invalid_argument(
            "maximum overview level must be <= 63");
    }
    if (options.page_extent == 0 ||
        options.maximum_resident_extent == 0 ||
        options.maximum_resident_extent % options.page_extent != 0 ||
        options.maximum_resident_extent >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "overview resident extent must be a positive page multiple");
    }
    if (!finite_positive(options.target_texels_per_logical_pixel)) {
        throw std::invalid_argument(
            "overview target texel density must be finite and positive");
    }
    if (!std::isfinite(options.guard_fraction) ||
        options.guard_fraction < 0.0 ||
        options.guard_fraction >= 1.0) {
        throw std::invalid_argument(
            "overview guard fraction must be finite and in [0, 1)");
    }

    const auto resident_pages =
        options.maximum_resident_extent / options.page_extent;
    const auto requested_guard_pages = static_cast<std::uint64_t>(
        std::ceil(
            static_cast<long double>(resident_pages) *
            static_cast<long double>(options.guard_fraction)));
    const auto maximum_visible_pages = std::max<std::uint64_t>(
        1, resident_pages - std::min(
            resident_pages - 1, requested_guard_pages));

    Camera2D density_camera = camera;
    density_camera.world_units_per_screen_pixel /=
        options.target_texels_per_logical_pixel;
    if (!finite_positive(density_camera.world_units_per_screen_pixel)) {
        throw std::overflow_error(
            "overview density-adjusted camera scale is not representable");
    }

    auto level = choose_overview_level(
        density_camera, raster, options.maximum_level);
    OverviewRequest visible;
    PageSpan visible_rows;
    PageSpan visible_columns;
    std::uint64_t total_page_rows = 0;
    std::uint64_t total_page_columns = 0;
    for (;;) {
        visible = make_overview_request(
            camera, raster, viewport, level);
        visible_rows = touched_pages(
            visible.level_window.row,
            visible.level_window.height,
            options.page_extent);
        visible_columns = touched_pages(
            visible.level_window.column,
            visible.level_window.width,
            options.page_extent);
        total_page_rows = ceil_div(
            visible.level_extent.height, options.page_extent);
        total_page_columns = ceil_div(
            visible.level_extent.width, options.page_extent);
        if (guarded_axis_fits(
                visible_rows,
                total_page_rows,
                resident_pages,
                maximum_visible_pages) &&
            guarded_axis_fits(
                visible_columns,
                total_page_columns,
                resident_pages,
                maximum_visible_pages)) {
            break;
        }
        if (level == options.maximum_level) {
            throw std::length_error(
                "maximum overview level cannot fit the guarded resident");
        }
        ++level;
    }

    const auto resident_page_rows = std::min({
        resident_pages,
        total_page_rows,
        visible_rows.count + requested_guard_pages});
    const auto resident_page_columns = std::min({
        resident_pages,
        total_page_columns,
        visible_columns.count + requested_guard_pages});
    const auto resident_page_row = canonical_resident_page_start(
        visible_rows, total_page_rows, resident_page_rows);
    const auto resident_page_column = canonical_resident_page_start(
        visible_columns,
        total_page_columns,
        resident_page_columns);
    const auto resident_level_row = clipped_product(
        resident_page_row,
        options.page_extent,
        visible.level_extent.height);
    const auto resident_level_column = clipped_product(
        resident_page_column,
        options.page_extent,
        visible.level_extent.width);
    const auto resident_level_row_end = clipped_product(
        resident_page_row + resident_page_rows,
        options.page_extent,
        visible.level_extent.height);
    const auto resident_level_column_end = clipped_product(
        resident_page_column + resident_page_columns,
        options.page_extent,
        visible.level_extent.width);
    const PixelWindow resident_level{
        resident_level_row,
        resident_level_column,
        resident_level_row_end - resident_level_row,
        resident_level_column_end - resident_level_column};

    const auto resident_source_row = clipped_product(
        resident_level.row, visible.sample_stride, raster.rows);
    const auto resident_source_column = clipped_product(
        resident_level.column, visible.sample_stride, raster.columns);
    const auto resident_source_row_end = clipped_product(
        resident_level.row + resident_level.height,
        visible.sample_stride,
        raster.rows);
    const auto resident_source_column_end = clipped_product(
        resident_level.column + resident_level.width,
        visible.sample_stride,
        raster.columns);

    return GuardedOverviewRequest{
        .visible = visible,
        .resident_source_window = PixelWindow{
            resident_source_row,
            resident_source_column,
            resident_source_row_end - resident_source_row,
            resident_source_column_end - resident_source_column},
        .resident_level_window = resident_level,
    };
}

}  // namespace satview::viewer
