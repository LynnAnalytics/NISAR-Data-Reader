#include "satview/view_navigation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>

namespace {
using namespace satview::viewer;

struct Test {
    int failures = 0;
    void expect(bool value, std::string_view label) {
        if (!value) {
            ++failures;
            std::cerr << "Navigation test failed: " << label << '\n';
        }
    }
    void close(double actual, double expected, double tolerance,
               std::string_view label) {
        expect(std::isfinite(actual) &&
               std::abs(actual - expected) <= tolerance, label);
    }
    template <typename Exception, typename Function>
    void throws(Function&& function, std::string_view label) {
        try {
            function();
            expect(false, label);
        } catch (const Exception&) {
        } catch (...) {
            expect(false, label);
        }
    }
};

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr ScreenViewport canvas{0.0, 0.0, 1080.0, 900.0};
constexpr RasterMetrics gcov{17'892, 18'108, -20.0, 20.0};
constexpr RasterMetrics gslc{67'824, 34'488, -5.0, 5.0};

[[nodiscard]] bool contains(
    const PixelWindow& outer,
    const PixelWindow& inner) {
    return inner.row >= outer.row && inner.column >= outer.column &&
        inner.row + inner.height <= outer.row + outer.height &&
        inner.column + inner.width <= outer.column + outer.width;
}

[[nodiscard]] Camera2D density_camera(
    Camera2D camera,
    const double target) {
    camera.world_units_per_screen_pixel /= target;
    return camera;
}

void fit_and_mapping(Test& test) {
    const auto fit = fit_camera(gcov, canvas);
    test.close(fit.center_row, 8'946.0, 0.0, "fit row center");
    test.close(fit.center_column, 9'054.0, 0.0, "fit column center");
    test.close(fit.world_units_per_screen_pixel, 397.6, 1e-12,
               "fit physical scale");
    const auto window = visible_raster_window(fit, gcov, canvas);
    test.close(window.row_begin, 0.0, 1e-10, "fit reaches top");
    test.close(window.column_begin, 0.0, 1e-10, "fit reaches left");
    test.close(window.row_end, 17'892.0, 1e-10, "fit reaches bottom");
    test.close(window.column_end, 18'108.0, 1e-10, "fit reaches right");

    const auto rect = visible_raster_screen_rect(fit, gcov, canvas);
    const double expected_width = 18'108.0 * 20.0 / 397.6;
    test.close(rect.width, expected_width, 1e-10, "letterbox width");
    test.close(rect.height, 900.0, 1e-10, "fit fills limiting axis");
    test.close(rect.x, 0.5 * (1080.0 - expected_width), 1e-10,
               "letterbox centered");

    const Camera2D camera{8'000.25, 9'000.75, 160.0};
    const ScreenViewport offset{123.5, 77.25, 1080.0, 900.0};
    const ScreenPoint screen{925.125, 312.75};
    const auto raster = screen_to_raster(camera, gcov, offset, screen);
    const auto round_trip = raster_to_screen(camera, gcov, offset, raster);
    test.close(round_trip.x, screen.x, 1e-11, "screen X round trip");
    test.close(round_trip.y, screen.y, 1e-11, "screen Y round trip");
}

void zoom_and_pan(Test& test) {
    const Camera2D start{8'946.0, 9'054.0, 320.0};
    const ScreenPoint anchor{810.0, 225.0};
    const auto before = screen_to_raster(start, gcov, canvas, anchor);
    test.close(before.row, 5'346.0, 1e-12, "anchor source row");
    test.close(before.column, 13'374.0, 1e-12, "anchor source column");
    const auto zoomed = zoom_about(start, gcov, canvas, anchor, 0.5);
    test.close(zoomed.center_row, 7'146.0, 1e-12, "zoom row center");
    test.close(zoomed.center_column, 11'214.0, 1e-12,
               "zoom column center");
    test.close(zoomed.world_units_per_screen_pixel, 160.0, 1e-12,
               "zoom scale");
    const auto after = screen_to_raster(zoomed, gcov, canvas, anchor);
    test.close(after.row, before.row, 1e-12, "zoom preserves row anchor");
    test.close(after.column, before.column, 1e-12,
               "zoom preserves column anchor");

    const Camera2D pan_start{8'946.0, 9'054.0, 160.0};
    const auto panned = pan_by_screen_delta(pan_start, gcov, canvas, 100.0, 50.0);
    test.close(panned.center_row, 8'546.0, 1e-12,
               "drag down moves camera up");
    test.close(panned.center_column, 8'254.0, 1e-12,
               "drag right moves camera left");
    const auto low = pan_by_screen_delta(pan_start, gcov, canvas, 1e9, 1e9);
    test.close(low.center_row, 3'600.0, 1e-12, "pan top clamp");
    test.close(low.center_column, 4'320.0, 1e-12, "pan left clamp");
    const auto high = pan_by_screen_delta(pan_start, gcov, canvas, -1e9, -1e9);
    test.close(high.center_row, 14'292.0, 1e-12, "pan bottom clamp");
    test.close(high.center_column, 13'788.0, 1e-12, "pan right clamp");
    const auto fitted = fit_camera(gcov, canvas);
    const auto locked = pan_by_screen_delta(fitted, gcov, canvas, 1e9, -1e9);
    test.close(locked.center_row, fitted.center_row, 0.0, "fit locks row pan");
    test.close(locked.center_column, fitted.center_column, 0.0,
               "fit locks column pan");
}

void rotation_navigation(Test& test) {
    constexpr RasterMetrics raster{1'000, 1'000, 2.0, 1.0};
    constexpr ScreenViewport viewport{0.0, 0.0, 400.0, 400.0};
    constexpr Camera2D quarter_turn{500.0, 500.0, 1.0, 0.5 * pi};

    const auto source_right = screen_to_raster(
        quarter_turn, raster, viewport, ScreenPoint{210.0, 200.0});
    test.close(source_right.row, 495.0, 1e-12,
               "clockwise rotation maps screen right toward raster top");
    test.close(source_right.column, 500.0, 1e-12,
               "quarter-turn screen right preserves raster column");

    const auto column_right = raster_to_screen(
        quarter_turn, raster, viewport, RasterPoint{500.0, 510.0});
    test.close(column_right.x, 200.0, 1e-12,
               "clockwise rotation maps raster right to screen center X");
    test.close(column_right.y, 210.0, 1e-12,
               "clockwise rotation maps raster right toward screen bottom");

    const auto row_down = raster_to_screen(
        quarter_turn, raster, viewport, RasterPoint{505.0, 500.0});
    test.close(row_down.x, 190.0, 1e-12,
               "clockwise rotation maps raster down toward screen left");
    test.close(row_down.y, 200.0, 1e-12,
               "quarter-turn raster down preserves screen Y");

    const Camera2D arbitrary{510.25, 480.75, 1.75, 0.371};
    const ScreenPoint screen{77.125, 318.875};
    const auto source = screen_to_raster(
        arbitrary, raster, viewport, screen);
    const auto round_trip = raster_to_screen(
        arbitrary, raster, viewport, source);
    test.close(round_trip.x, screen.x, 1e-11,
               "rotated screen X round trip");
    test.close(round_trip.y, screen.y, 1e-11,
               "rotated screen Y round trip");

    const ScreenPoint anchor{275.0, 145.0};
    const auto anchor_before = screen_to_raster(
        arbitrary, raster, viewport, anchor);
    const auto zoomed = zoom_about(
        arbitrary, raster, viewport, anchor, 0.5);
    const auto anchor_after = screen_to_raster(
        zoomed, raster, viewport, anchor);
    test.close(anchor_after.row, anchor_before.row, 1e-11,
               "rotated zoom preserves row anchor");
    test.close(anchor_after.column, anchor_before.column, 1e-11,
               "rotated zoom preserves column anchor");
    test.close(zoomed.rotation_radians, arbitrary.rotation_radians, 1e-15,
               "zoom preserves rotation");

    const auto panned = pan_by_screen_delta(
        quarter_turn, raster, viewport, 10.0, 0.0);
    test.close(panned.center_row, 505.0, 1e-12,
               "rotated right drag moves camera toward raster bottom");
    test.close(panned.center_column, 500.0, 1e-12,
               "quarter-turn right drag preserves camera column");

    constexpr RasterMetrics rectangle{100, 200, 1.0, 1.0};
    constexpr ScreenViewport wide{0.0, 0.0, 200.0, 100.0};
    const auto rotated_fit = fit_camera(rectangle, wide, 0.5 * pi);
    test.close(rotated_fit.world_units_per_screen_pixel, 2.0, 1e-12,
               "rotated fit uses the rotated physical bounds");
    const auto rotated_rect = visible_raster_screen_rect(
        rotated_fit, rectangle, wide);
    test.close(rotated_rect.x, 75.0, 1e-12,
               "rotated fit letterbox is centered");
    test.close(rotated_rect.width, 50.0, 1e-12,
               "rotated fit letterbox width");
    test.close(rotated_rect.height, 100.0, 1e-12,
               "rotated fit fills limiting height");

    constexpr RasterMetrics square_raster{1'000, 1'000, 1.0, 1.0};
    constexpr ScreenViewport landscape{0.0, 0.0, 200.0, 100.0};
    const auto clamped = clamp_camera(
        Camera2D{0.0, 0.0, 1.0, 0.5 * pi},
        square_raster,
        landscape);
    test.close(clamped.center_row, 100.0, 1e-12,
               "rotated clamp uses screen width for raster rows");
    test.close(clamped.center_column, 50.0, 1e-12,
               "rotated clamp uses screen height for raster columns");
    const auto visible = visible_raster_window(
        Camera2D{500.0, 500.0, 1.0, 0.5 * pi},
        square_raster,
        landscape);
    test.close(visible.row_begin, 400.0, 1e-12,
               "rotated visible row begin");
    test.close(visible.row_end, 600.0, 1e-12,
               "rotated visible row end");
    test.close(visible.column_begin, 450.0, 1e-12,
               "rotated visible column begin");
    test.close(visible.column_end, 550.0, 1e-12,
               "rotated visible column end");
}

void physical_spacing_and_overviews(Test& test) {
    constexpr RasterMetrics unequal{2'000, 4'000, -2.0, 4.0};
    constexpr ScreenViewport square{0.0, 0.0, 1'000.0, 1'000.0};
    const auto unequal_fit = fit_camera(unequal, square);
    test.close(unequal_fit.world_units_per_screen_pixel, 16.0, 1e-12,
               "unequal spacing fit");
    const Camera2D unequal_camera{1'000.0, 2'000.0, 8.0};
    const auto moved = screen_to_raster(
        unequal_camera, unequal, square, ScreenPoint{501.0, 501.0});
    test.close(moved.row, 1'004.0, 1e-12, "row spacing mapping");
    test.close(moved.column, 2'002.0, 1e-12, "column spacing mapping");
    test.expect(choose_overview_level(unequal_camera, unequal, 10) == 1,
                "LOD limited by coarser physical axis");

    const auto gcov_fit = fit_camera(gcov, canvas);
    test.expect(choose_overview_level(gcov_fit, gcov, 20) == 4,
                "GCOV fit chooses LOD4");
    test.expect(overview_level_extent(gcov, 4) == PixelExtent{1'119, 1'132},
                "partial edge overview shape");
    const auto complete = make_overview_request(gcov_fit, gcov, canvas, 4);
    test.expect(complete == OverviewRequest{
        4, 16, PixelWindow{0, 0, 17'892, 18'108},
        PixelWindow{0, 0, 1'119, 1'132}, PixelExtent{1'119, 1'132}},
        "fit request covers complete overview");

    const auto gslc_fit = fit_camera(gslc, canvas);
    test.expect(choose_overview_level(gslc_fit, gslc, 20) == 6,
                "GSLC fit chooses LOD6");
    test.expect(overview_level_extent(gslc, 6) == PixelExtent{1'060, 539},
                "GSLC overview shape");

    const Camera2D zoomed{8'946.0, 9'054.0, 160.0};
    test.expect(choose_overview_level(zoomed, gcov, 20) == 3,
                "eight samples per screen pixel chooses LOD3");
    const auto request = make_overview_request(zoomed, gcov, canvas, 3);
    test.expect(request.source_window == PixelWindow{5'344, 4'728, 7'208, 8'648},
                "visible source aligns outward to LOD blocks");
    test.expect(request.level_window == PixelWindow{668, 591, 901, 1'081},
                "aligned level coordinates");
    const Camera2D edge{14'292.0, 13'788.0, 160.0};
    const auto edge_request = make_overview_request(edge, gcov, canvas, 3);
    test.expect(edge_request.source_window.row + edge_request.source_window.height ==
                    gcov.rows &&
                edge_request.source_window.column + edge_request.source_window.width ==
                    gcov.columns,
                "edge request clips partial blocks");
}

void guarded_overviews(Test& test) {
    const GuardedOverviewOptions options{
        .maximum_level = 20,
        .page_extent = 512,
        .maximum_resident_extent = 4096,
        .target_texels_per_logical_pixel = 1.25,
        .guard_fraction = 0.25,
    };
    const auto fitted = make_guarded_overview_request(
        fit_camera(gcov, canvas), gcov, canvas, options);
    test.expect(fitted.visible.level == 3 &&
                    fitted.visible.sample_stride == 8,
                "guarded fit honors subpixel density target");
    test.expect(
        fitted.resident_level_window == PixelWindow{0, 0, 2'237, 2'264} &&
            fitted.resident_source_window ==
                PixelWindow{0, 0, 17'892, 18'108},
        "small complete overview uses one partial-edge resident");

    const Camera2D zoomed{8'946.0, 9'054.0, 160.0};
    const auto first = make_guarded_overview_request(
        zoomed, gcov, canvas, options);
    test.expect(first.visible.level == 2 &&
                    first.visible.sample_stride == 4,
                "guarded zoom keeps more than one texel per logical pixel");
    test.expect(first.resident_level_window.width <=
                    options.maximum_resident_extent &&
                    first.resident_level_window.height <=
                    options.maximum_resident_extent &&
                    first.resident_level_window.row % options.page_extent == 0 &&
                    first.resident_level_window.column % options.page_extent == 0,
                "guarded resident is bounded and page aligned");
    test.expect(contains(
                    first.resident_level_window,
                    first.visible.level_window) &&
                    contains(
                        first.resident_source_window,
                        first.visible.source_window),
                "guarded resident contains visible coverage");

    const auto moved_camera = pan_by_screen_delta(
        zoomed, gcov, canvas, 10.0, -10.0);
    const auto moved = make_guarded_overview_request(
        moved_camera, gcov, canvas, options);
    test.expect(
        moved.visible.level == first.visible.level &&
            moved.resident_level_window == first.resident_level_window &&
            moved.resident_source_window == first.resident_source_window,
        "small pan reuses the guarded resident");

    const Camera2D hysteresis_zoom{8'946.0, 9'054.0, 96.0};
    const auto finer = make_guarded_overview_request(
        hysteresis_zoom, gcov, canvas, options);
    test.expect(
        finer.visible.sample_stride == 2 &&
            first.visible.sample_stride == 4,
        "planner requests finer data at the target-density boundary");
    test.expect(
        sampled_window_remains_usable(
            first.resident_source_window,
            first.visible.sample_stride,
            first.visible.sample_stride,
            visible_raster_window(hysteresis_zoom, gcov, canvas),
            hysteresis_zoom,
            gcov,
            1.0,
            1.0,
            1.0),
        "resident overview bridges a small zoom across an LOD boundary");

    const Camera2D underdense_zoom{8'946.0, 9'054.0, 79.0};
    test.expect(
        !sampled_window_remains_usable(
            first.resident_source_window,
            first.visible.sample_stride,
            first.visible.sample_stride,
            visible_raster_window(underdense_zoom, gcov, canvas),
            underdense_zoom,
            gcov,
            1.0,
            1.0,
            1.0),
        "resident overview expires before zoom softness becomes visible");

    test.throws<std::invalid_argument>(
        [&] {
            (void)sampled_window_remains_usable(
                first.resident_source_window,
                first.visible.sample_stride,
                first.visible.sample_stride,
                visible_raster_window(zoomed, gcov, canvas),
                zoomed,
                gcov,
                1.0,
                1.0,
                0.0);
        },
        "overview reuse rejects a non-positive density threshold");

    constexpr RasterMetrics large{100'000, 100'000, 1.0, 1.0};
    constexpr ScreenViewport small_canvas{0.0, 0.0, 256.0, 256.0};
    constexpr Camera2D one_page{50'048.0, 50'048.0, 1.0};
    const auto small = make_guarded_overview_request(
        one_page, large, small_canvas, options);
    test.expect(
        small.visible.level == 0 &&
            small.resident_level_window.height == 1'536 &&
            small.resident_level_window.width == 1'536 &&
            small.resident_source_window.height == 1'536 &&
            small.resident_source_window.width == 1'536,
        "one-page viewport allocates only visible plus guard pages");

    constexpr ScreenViewport wide{0.0, 0.0, 1'500.0, 1'000.0};
    constexpr RasterMetrics regional{5'000, 5'000, 1.0, 1.0};
    constexpr Camera2D four_pages{1'000.0, 1'261.0, 1.0};
    const GuardedOverviewOptions compact{
        .maximum_level = 10,
        .page_extent = 512,
        .maximum_resident_extent = 2048,
        .target_texels_per_logical_pixel = 1.0,
        .guard_fraction = 0.25,
    };
    const auto capacity_limited = make_guarded_overview_request(
        four_pages, regional, wide, compact);
    test.expect(capacity_limited.visible.level == 1 &&
                    capacity_limited.visible.sample_stride == 2,
                "planner relaxes density by only one adjacent LOD for guard");

    std::uint64_t previous_stride = 1;
    for (int index = 0; index <= 600; ++index) {
        const double fraction = static_cast<double>(index) / 600.0;
        const double scale = 20.0 * std::pow(397.6 / 20.0, fraction);
        const auto selected = make_guarded_overview_request(
            Camera2D{8'946.0, 9'054.0, scale},
            gcov,
            canvas,
            options);
        test.expect(
            selected.visible.sample_stride == previous_stride ||
                selected.visible.sample_stride == previous_stride * 2,
            "continuous zoom changes guarded LOD by at most two-to-one");
        previous_stride = selected.visible.sample_stride;

        const auto density_level = choose_overview_level(
            density_camera(
                Camera2D{8'946.0, 9'054.0, scale},
                options.target_texels_per_logical_pixel),
            gcov,
            options.maximum_level);
        if (selected.visible.level == density_level &&
            selected.visible.level > 0) {
            const double texels_per_logical_pixel =
                scale / 20.0 /
                static_cast<double>(selected.visible.sample_stride);
            test.expect(
                texels_per_logical_pixel + 1e-12 >=
                    options.target_texels_per_logical_pixel,
                "unconstrained guarded LOD remains subpixel");
        }
    }
}

void framebuffer_density(Test& test) {
    constexpr RasterMetrics anisotropic{2'000, 4'000, -2.0, 4.0};
    constexpr Camera2D camera{1'000.0, 2'000.0, 80.0};

    const auto x_limited =
        minimum_resident_texels_per_framebuffer_pixel(
            camera, anisotropic, 8, 4, 2.0, 1.25);
    test.close(
        x_limited,
        2.5,
        0.0,
        "anisotropic density uses column stride and framebuffer X scale");

    const auto y_limited =
        minimum_resident_texels_per_framebuffer_pixel(
            camera, anisotropic, 8, 4, 1.0, 1.25);
    test.close(
        y_limited,
        4.0,
        0.0,
        "anisotropic density uses row stride and framebuffer Y scale");

    const auto rotated_limited =
        minimum_resident_texels_per_framebuffer_pixel(
            Camera2D{1'000.0, 2'000.0, 80.0, 0.5 * pi},
            anisotropic,
            8,
            2,
            2.0,
            1.0);
    test.close(rotated_limited, 2.5, 1e-12,
               "rotation carries anisotropic density across framebuffer axes");
}

void randomized_invariants(Test& test) {
    std::mt19937_64 random(0x5090C0DEULL);
    std::uniform_int_distribution<std::uint64_t> dimension(1, 1'000'000);
    std::uniform_real_distribution<double> spacing(0.25, 100.0);
    std::uniform_real_distribution<double> extent(1.0, 4'000.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> angle(-pi, pi);
    for (int index = 0; index < 1'000; ++index) {
        const RasterMetrics raster{dimension(random), dimension(random),
            index % 2 == 0 ? -spacing(random) : spacing(random), spacing(random)};
        const ScreenViewport viewport{17.25, 33.5, extent(random), extent(random)};
        const auto camera = fit_camera(
            raster, viewport, angle(random));
        const ScreenPoint point{viewport.x + unit(random) * viewport.width,
                                viewport.y + unit(random) * viewport.height};
        const auto source = screen_to_raster(camera, raster, viewport, point);
        const auto round_trip = raster_to_screen(camera, raster, viewport, source);
        const double tolerance = 2e-9 * std::max(viewport.width, viewport.height);
        test.expect(std::abs(round_trip.x - point.x) <= tolerance &&
                    std::abs(round_trip.y - point.y) <= tolerance,
                    "random coordinate round trip");
        const auto level = choose_overview_level(camera, raster, 20);
        const auto request = make_overview_request(camera, raster, viewport, level);
        test.expect(request.sample_stride == (std::uint64_t{1} << level),
                    "random stride is power of two");
        test.expect(request.source_window.row + request.source_window.height <=
                        raster.rows &&
                    request.source_window.column + request.source_window.width <=
                        raster.columns,
                    "random source request bounded");
        test.expect(request.level_window.row + request.level_window.height <=
                        request.level_extent.height &&
                    request.level_window.column + request.level_window.width <=
                        request.level_extent.width,
                    "random level request bounded");

        const auto zoom_scale = 0.02 + 0.98 * unit(random);
        Camera2D random_camera = camera;
        random_camera.world_units_per_screen_pixel *= zoom_scale;
        random_camera.center_row = unit(random) *
            static_cast<double>(raster.rows);
        random_camera.center_column = unit(random) *
            static_cast<double>(raster.columns);
        random_camera = clamp_camera(random_camera, raster, viewport);
        const GuardedOverviewOptions guarded_options{
            .maximum_level = 40,
            .page_extent = 512,
            .maximum_resident_extent = 4096,
            .target_texels_per_logical_pixel =
                1.25 * (1.0 + 2.0 * unit(random)),
            .guard_fraction = 0.25,
        };
        const auto guarded = make_guarded_overview_request(
            random_camera, raster, viewport, guarded_options);
        test.expect(
            guarded.resident_level_window.height <= 4096 &&
                guarded.resident_level_window.width <= 4096 &&
                guarded.resident_level_window.row % 512 == 0 &&
                guarded.resident_level_window.column % 512 == 0,
            "random guarded resident is bounded and quantized");
        test.expect(
            contains(
                guarded.resident_level_window,
                guarded.visible.level_window) &&
                contains(
                    guarded.resident_source_window,
                    guarded.visible.source_window),
            "random guarded resident contains visible request");
        test.expect(
            guarded.visible.sample_stride ==
                (std::uint64_t{1} << guarded.visible.level),
            "random guarded stride remains power of two");
    }
}

void invalid_inputs(Test& test) {
    test.throws<std::invalid_argument>([] { static_cast<void>(fit_camera(
        RasterMetrics{0, 1, 1.0, 1.0}, canvas)); }, "zero dimension rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(fit_camera(
        RasterMetrics{1, 1, 0.0, 1.0}, canvas)); }, "zero spacing rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(fit_camera(
        gcov, ScreenViewport{0.0, 0.0, 0.0, 1.0})); },
        "zero viewport rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(screen_to_raster(
        Camera2D{1.0, 1.0, 0.0}, gcov, canvas, ScreenPoint{})); },
        "zero camera scale rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(fit_camera(
        gcov,
        canvas,
        std::numeric_limits<double>::quiet_NaN())); },
        "nonfinite fit rotation rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(screen_to_raster(
        Camera2D{1.0, 1.0, 1.0, std::numeric_limits<double>::infinity()},
        gcov, canvas, ScreenPoint{})); }, "nonfinite camera rotation rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(zoom_about(
        fit_camera(gcov, canvas), gcov, canvas,
        ScreenPoint{540.0, 450.0}, 0.0)); },
        "zero zoom factor rejected");
    test.throws<std::out_of_range>([] { static_cast<void>(zoom_about(
        fit_camera(gcov, canvas), gcov, canvas,
        ScreenPoint{-1.0, 450.0}, 0.5)); },
        "outside zoom anchor rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(pan_by_screen_delta(
        fit_camera(gcov, canvas), gcov, canvas,
        std::numeric_limits<double>::quiet_NaN(), 0.0)); },
        "nonfinite pan rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        overview_level_extent(gcov, 64)); },
        "LOD above 63 rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        choose_overview_level(fit_camera(gcov, canvas), gcov, 64)); },
        "maximum LOD above 63 rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        make_guarded_overview_request(
            fit_camera(gcov, canvas),
            gcov,
            canvas,
            GuardedOverviewOptions{.maximum_level = 64})); },
        "guarded maximum LOD above 63 rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        make_guarded_overview_request(
            fit_camera(gcov, canvas),
            gcov,
            canvas,
            GuardedOverviewOptions{.page_extent = 0})); },
        "zero overview page extent rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        make_guarded_overview_request(
            fit_camera(gcov, canvas),
            gcov,
            canvas,
            GuardedOverviewOptions{
                .page_extent = 512,
                .maximum_resident_extent = 4097})); },
        "nonintegral overview page span rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        make_guarded_overview_request(
            fit_camera(gcov, canvas),
            gcov,
            canvas,
            GuardedOverviewOptions{
                .target_texels_per_logical_pixel = 0.0})); },
        "zero overview density target rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        make_guarded_overview_request(
            fit_camera(gcov, canvas),
            gcov,
            canvas,
            GuardedOverviewOptions{.guard_fraction = 1.0})); },
        "complete overview guard fraction rejected");
    test.throws<std::length_error>([] { static_cast<void>(
        make_guarded_overview_request(
            Camera2D{500'000.0, 500'000.0, 1.0},
            RasterMetrics{1'000'000, 1'000'000, 1.0, 1.0},
            ScreenViewport{0.0, 0.0, 10'000.0, 10'000.0},
            GuardedOverviewOptions{
                .maximum_level = 0,
                .maximum_resident_extent = 2048})); },
        "insufficient guarded maximum level rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            fit_camera(gcov, canvas), gcov, 0, 1, 1.0, 1.0)); },
        "zero resident row stride rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            fit_camera(gcov, canvas), gcov, 1, 0, 1.0, 1.0)); },
        "zero resident column stride rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            fit_camera(gcov, canvas), gcov, 1, 1, 0.0, 1.0)); },
        "zero framebuffer X scale rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            fit_camera(gcov, canvas),
            gcov,
            1,
            1,
            1.0,
            std::numeric_limits<double>::infinity())); },
        "nonfinite framebuffer Y scale rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            Camera2D{1.0, 1.0, 0.0}, gcov, 1, 1, 1.0, 1.0)); },
        "invalid density camera rejected");
    test.throws<std::invalid_argument>([] { static_cast<void>(
        minimum_resident_texels_per_framebuffer_pixel(
            Camera2D{1.0, 1.0, 1.0},
            RasterMetrics{1, 1, 0.0, 1.0},
            1,
            1,
            1.0,
            1.0)); },
        "invalid density raster rejected");
}
}  // namespace

int main() {
    Test test;
    fit_and_mapping(test);
    zoom_and_pan(test);
    rotation_navigation(test);
    physical_spacing_and_overviews(test);
    guarded_overviews(test);
    framebuffer_density(test);
    randomized_invariants(test);
    invalid_inputs(test);
    if (test.failures == 0) {
        std::cout << "All navigation tests passed\n";
        return 0;
    }
    std::cerr << test.failures << " navigation test(s) failed\n";
    return 1;
}
