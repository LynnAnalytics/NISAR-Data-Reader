#include "satview/hdf5_product.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

int run_distribution_tests();
#ifdef SATVIEW_HAS_CUDA
int run_gpu_distribution_tests();
int run_gpu_transform_tests();
#endif
int run_colormap_tests();
int run_cpu_scientific_tests();
int run_overview_tests(const satview::Hdf5Product& product);
int run_overview_spacing_tests(const satview::Hdf5Product& product);
int run_real_mosaic_plan_tests(const satview::Hdf5Product& product);
int run_resident_view_tests();
int run_viewer_math_tests();

namespace {

struct TestContext {
  int failures = 0;

  void expect(bool condition, std::string_view message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

std::filesystem::path data_directory() {
#if defined(_MSC_VER)
  char *value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, "SATVIEW_TEST_DATA_DIR") == 0 &&
      value != nullptr) {
    const std::filesystem::path result(value);
    std::free(value);
    return result;
  }
#else
  if (const char *value = std::getenv("SATVIEW_TEST_DATA_DIR")) {
    return std::filesystem::path(value);
  }
#endif
  return std::filesystem::path(SATVIEW_DEFAULT_TEST_DATA_DIR);
}

std::filesystem::path find_product(const std::filesystem::path &directory,
                                   std::string_view marker) {
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".h5" &&
        entry.path().filename().string().find(marker) != std::string::npos) {
      return entry.path();
    }
  }
  throw std::runtime_error("missing test product containing " +
                           std::string(marker));
}

struct FloatPair {
  float real;
  float imaginary;
};

std::span<const std::byte> element_at(const std::vector<std::byte> &bytes,
                                      const satview::ReadPlan &plan,
                                      std::uint64_t row, std::uint64_t column) {
  const auto local_row = row - plan.aligned.row;
  const auto local_column = column - plan.aligned.column;
  const auto element_index = local_row * plan.aligned.width + local_column;
  const auto offset =
      static_cast<std::size_t>(element_index) * plan.data_type.element_size;
  return {bytes.data() + offset, plan.data_type.element_size};
}

std::size_t window_bytes(const satview::Window2D &window,
                         std::size_t element_size) {
  return static_cast<std::size_t>(window.height * window.width) * element_size;
}

bool same_window(const satview::Window2D &actual,
                 const satview::Window2D &expected) {
  return actual.row == expected.row && actual.column == expected.column &&
         actual.height == expected.height && actual.width == expected.width;
}

void expect_overlapping_reads_equal(TestContext &test,
                                    const std::vector<std::byte> &first_bytes,
                                    const satview::ReadPlan &first,
                                    const std::vector<std::byte> &second_bytes,
                                    const satview::ReadPlan &second,
                                    std::string_view message) {
  test.expect(first.data_type.element_size == second.data_type.element_size,
              "overlapping reads have the same element size");
  if (first.data_type.element_size != second.data_type.element_size) {
    return;
  }

  const auto first_row_end = first.aligned.row + first.aligned.height;
  const auto first_column_end = first.aligned.column + first.aligned.width;
  const auto second_row_end = second.aligned.row + second.aligned.height;
  const auto second_column_end = second.aligned.column + second.aligned.width;
  const auto row_begin = std::max(first.aligned.row, second.aligned.row);
  const auto row_end = std::min(first_row_end, second_row_end);
  const auto column_begin =
      std::max(first.aligned.column, second.aligned.column);
  const auto column_end = std::min(first_column_end, second_column_end);

  test.expect(row_begin < row_end && column_begin < column_end,
              "overlapping-read comparison has a non-empty intersection");
  if (row_begin >= row_end || column_begin >= column_end) {
    return;
  }

  const auto element_size = first.data_type.element_size;
  const auto row_bytes =
      static_cast<std::size_t>(column_end - column_begin) * element_size;
  bool equal = true;
  for (auto row = row_begin; row < row_end; ++row) {
    const auto first_element = (row - first.aligned.row) * first.aligned.width +
                               (column_begin - first.aligned.column);
    const auto second_element =
        (row - second.aligned.row) * second.aligned.width +
        (column_begin - second.aligned.column);
    const auto *first_row =
        first_bytes.data() +
        static_cast<std::size_t>(first_element) * element_size;
    const auto *second_row =
        second_bytes.data() +
        static_cast<std::size_t>(second_element) * element_size;
    if (std::memcmp(first_row, second_row, row_bytes) != 0) {
      equal = false;
      break;
    }
  }
  test.expect(equal, message);
}

void test_gcov_chunk_boundaries(TestContext &test,
                                const satview::Hdf5Product &product,
                                std::string_view layer) {
  constexpr std::uint64_t chunk = 512;
  constexpr std::uint64_t rows = 17892;
  constexpr std::uint64_t columns = 18108;
  constexpr std::size_t element_size = sizeof(float);
  constexpr std::uint64_t bottom_chunk_row = (rows / chunk) * chunk;
  constexpr std::uint64_t right_chunk_column = (columns / chunk) * chunk;

  const auto full_plan = product.make_read_plan(layer, chunk, chunk, 1, 1);
  test.expect(same_window(full_plan.aligned, {chunk, chunk, chunk, chunk}),
              "GCOV interior request aligns to one full chunk");
  test.expect(full_plan.expected_bytes ==
                  window_bytes(full_plan.aligned, element_size),
              "GCOV full-chunk byte count");
  std::vector<std::byte> full_bytes(full_plan.expected_bytes);
  product.read_into(full_plan, full_bytes);

  const auto four_chunk_plan =
      product.make_read_plan(layer, chunk - 1, chunk - 1, 2, 2);
  test.expect(
      same_window(four_chunk_plan.aligned, {0, 0, 2 * chunk, 2 * chunk}),
      "GCOV boundary-crossing request aligns to four chunks");
  test.expect(four_chunk_plan.requested_row_offset == chunk - 1 &&
                  four_chunk_plan.requested_column_offset == chunk - 1,
              "GCOV four-chunk request offsets");
  test.expect(four_chunk_plan.expected_bytes ==
                  window_bytes(four_chunk_plan.aligned, element_size),
              "GCOV four-chunk byte count");
  std::vector<std::byte> four_chunk_bytes(four_chunk_plan.expected_bytes);
  product.read_into(four_chunk_plan, four_chunk_bytes);
  expect_overlapping_reads_equal(
      test, full_bytes, full_plan, four_chunk_bytes, four_chunk_plan,
      "GCOV full chunk is stable inside a four-chunk read");

  const auto bottom_plan =
      product.make_read_plan(layer, rows - 1, right_chunk_column - 1, 1, 2);
  test.expect(same_window(bottom_plan.aligned,
                          {bottom_chunk_row, right_chunk_column - chunk,
                           rows - bottom_chunk_row,
                           columns - (right_chunk_column - chunk)}),
              "GCOV bottom-edge read clips partial bottom and right chunks");
  test.expect(bottom_plan.expected_bytes ==
                  window_bytes(bottom_plan.aligned, element_size),
              "GCOV bottom-edge byte count");
  std::vector<std::byte> bottom_bytes(bottom_plan.expected_bytes);
  product.read_into(bottom_plan, bottom_bytes);

  const auto right_plan =
      product.make_read_plan(layer, bottom_chunk_row - 1, columns - 1, 2, 1);
  test.expect(same_window(right_plan.aligned,
                          {bottom_chunk_row - chunk, right_chunk_column,
                           rows - (bottom_chunk_row - chunk),
                           columns - right_chunk_column}),
              "GCOV right-edge read clips partial right and bottom chunks");
  test.expect(right_plan.expected_bytes ==
                  window_bytes(right_plan.aligned, element_size),
              "GCOV right-edge byte count");
  std::vector<std::byte> right_bytes(right_plan.expected_bytes);
  product.read_into(right_plan, right_bytes);

  expect_overlapping_reads_equal(
      test, bottom_bytes, bottom_plan, right_bytes, right_plan,
      "GCOV bottom-right partial chunk is stable across overlapping reads");
}

void test_gcov(TestContext &test, const std::filesystem::path &file) {
  const satview::Hdf5Product product(file);
  test.failures += run_overview_tests(product);
  test.failures += run_real_mosaic_plan_tests(product);
  test.expect(product.identification().product_type ==
                  satview::ProductType::gcov,
              "GCOV product type");
  test.expect(product.frequencies().size() == 2, "GCOV has two frequencies");

  const auto *frequency_a = product.find_frequency("A");
  const auto *frequency_b = product.find_frequency("B");
  test.expect(frequency_a != nullptr && frequency_b != nullptr,
              "GCOV A/B catalogs exist");
  if (frequency_a != nullptr) {
    test.expect(frequency_a->grid.y.count == 17892 &&
                    frequency_a->grid.x.count == 18108,
                "GCOV A grid dimensions");
    test.expect(frequency_a->grid.epsg == 32616, "GCOV EPSG:32616");
    test.expect(frequency_a->grid.y.spacing == -20.0,
                "GCOV negative Y spacing");
  }

  const std::string layer = "/science/LSAR/GCOV/grids/frequencyA/HHHH";
  const auto *descriptor = product.find_dataset(layer);
  test.expect(descriptor != nullptr, "GCOV HHHH catalog entry");
  if (descriptor != nullptr) {
    test.expect(descriptor->dimensions[0] == 17892 &&
                    descriptor->dimensions[1] == 18108,
                "GCOV HHHH dimensions");
    test.expect(
        descriptor->data_type.file_representation_is_native,
        "GCOV HHHH file bytes exactly match the native representation");
    test.expect(descriptor->chunk_dimensions ==
                    std::array<std::uint64_t, 2>{512, 512},
                "GCOV HHHH chunks");
    test.expect(
        descriptor->filters.size() == 2 &&
            descriptor->filters[0].name.find("shuffle") != std::string::npos &&
            descriptor->filters[1].name.find("deflate") != std::string::npos,
        "GCOV HHHH filter order");
  }

  const auto *exception_b = product.find_dataset(
      "/science/LSAR/GCOV/grids/frequencyB/inputDataExceptionMask");
  test.expect(exception_b != nullptr, "GCOV B exception-mask catalog entry");
  if (exception_b != nullptr) {
    test.expect(exception_b->dimensions ==
                    std::array<std::uint64_t, 2>{17892, 18108},
                "GCOV B exception mask preserves physical A-sized shape");
  }

  const std::uint64_t row = 8946;
  const std::uint64_t column = 9054;
  const auto plan = product.make_read_plan(layer, row, column, 1, 1);
  test.expect(
      product.supports_direct_chunk_decode(plan),
      "GCOV center chunk supports exact direct decoding");
  test.expect(plan.aligned.height <= 512 && plan.aligned.width <= 512,
              "GCOV single-pixel request remains one source chunk");
  std::vector<std::byte> bytes(plan.expected_bytes);
  product.read_direct_chunk_into(plan, bytes);
  float value = 0.0F;
  const auto source = element_at(bytes, plan, row, column);
  std::memcpy(&value, source.data(), sizeof(value));
  test.expect(std::abs(value - 0.0583252906799316F) < 1.0e-7F,
              "GCOV center-pixel golden");
  std::vector<std::byte> ordinary_bytes(plan.expected_bytes);
  product.read_into(plan, ordinary_bytes);
  test.expect(
      bytes == ordinary_bytes,
      "GCOV direct chunk is byte-identical to ordinary H5Dread");

  const std::string mask_layer =
      "/science/LSAR/GCOV/grids/frequencyA/mask";
  const auto *mask_descriptor = product.find_dataset(mask_layer);
  test.expect(mask_descriptor != nullptr, "GCOV exact sibling mask exists");
  if (mask_descriptor != nullptr) {
    const auto mask_plan =
        product.make_read_plan(mask_layer, row, column, 1, 1);
    std::vector<std::byte> first_science(plan.expected_bytes);
    std::vector<std::byte> second_science(plan.expected_bytes);
    std::vector<std::byte> first_mask(mask_plan.expected_bytes);
    std::vector<std::byte> second_mask(mask_plan.expected_bytes);

    product.read_into(plan, first_science);
    product.read_into(mask_plan, first_mask);
    product.read_into(plan, second_science);
    product.read_into(mask_plan, second_mask);

    test.expect(first_science == second_science,
                "GCOV science read is stable across mask alternation");
    test.expect(first_mask == second_mask,
                "GCOV mask read is stable across science alternation");
  }

  test_gcov_chunk_boundaries(test, product, layer);
}

void test_open_option_validation(TestContext &test,
                                 const std::filesystem::path &file) {
  satview::Hdf5OpenOptions options;
  options.read_dataset_cache_entries = 0;
  bool rejected = false;
  try {
    const satview::Hdf5Product product(file, options);
  } catch (const satview::Hdf5Error &) {
    rejected = true;
  }
  test.expect(rejected, "zero HDF5 read dataset cache entries are rejected");
}

void test_gslc(TestContext &test, const std::filesystem::path &file,
               std::uint64_t expected_rows, std::uint64_t expected_columns,
               std::uint32_t expected_epsg, std::uint64_t row,
               std::uint64_t column, FloatPair expected) {
  const satview::Hdf5Product product(file);
  test.expect(product.identification().product_type ==
                  satview::ProductType::gslc,
              "GSLC product type");
  const auto *frequency_a = product.find_frequency("A");
  test.expect(frequency_a != nullptr, "GSLC frequency A exists");
  if (frequency_a != nullptr) {
    test.expect(frequency_a->grid.y.count == expected_rows &&
                    frequency_a->grid.x.count == expected_columns,
                "GSLC A grid dimensions");
    test.expect(frequency_a->grid.epsg == expected_epsg, "GSLC EPSG");
    test.expect(frequency_a->grid.y.spacing == -5.0, "GSLC negative Y spacing");
  }

  const std::string layer = "/science/LSAR/GSLC/grids/frequencyA/HH";
  const auto *descriptor = product.find_dataset(layer);
  test.expect(descriptor != nullptr, "GSLC HH catalog entry");
  if (descriptor != nullptr) {
    test.expect(descriptor->data_type.kind ==
                        satview::ScalarKind::compound_complex &&
                    descriptor->data_type.element_size == sizeof(FloatPair),
                "GSLC canonical float2 layout");
    test.expect(
        descriptor->data_type.file_representation_is_native,
        "GSLC complex file bytes exactly match the canonical native layout");
  }

  const auto plan = product.make_read_plan(layer, row, column, 1, 1);
  test.expect(
      product.supports_direct_chunk_decode(plan),
      "GSLC center chunk supports exact direct decoding");
  std::vector<std::byte> bytes(plan.expected_bytes);
  product.read_direct_chunk_into(plan, bytes);
  FloatPair value{};
  const auto source = element_at(bytes, plan, row, column);
  std::memcpy(&value, source.data(), sizeof(value));
  test.expect(std::abs(value.real - expected.real) < 1.0e-7F &&
                  std::abs(value.imaginary - expected.imaginary) < 1.0e-7F,
              "GSLC center-pixel complex golden");
  std::vector<std::byte> ordinary_bytes(plan.expected_bytes);
  product.read_into(plan, ordinary_bytes);
  test.expect(
      bytes == ordinary_bytes,
      "GSLC direct complex chunk is byte-identical to ordinary H5Dread");
  if (expected_rows == 67824) {
    test.failures += run_overview_spacing_tests(product);
  }
}

} // namespace

int main() {
  TestContext test;
  test.failures += run_colormap_tests();
  test.failures += run_cpu_scientific_tests();
  test.failures += run_distribution_tests();
  test.failures += run_resident_view_tests();
  test.failures += run_viewer_math_tests();
  try {
    const auto directory = data_directory();
    test.expect(std::filesystem::is_directory(directory),
                "test-data directory exists");
    if (std::filesystem::is_directory(directory)) {
      const auto gcov = find_product(directory, "_GCOV_");
      test_open_option_validation(test, gcov);
      test_gcov(test, gcov);
      test_gslc(test, find_product(directory, "_128_"), 67824, 34488, 32640,
                33912, 17244, {-0.0526123046875F, -0.036163330078125F});
      test_gslc(test, find_product(directory, "_169_"), 66672, 34056, 32650,
                33336, 17028, {-0.046600341796875F, 0.67578125F});
    }
  } catch (const std::exception &error) {
    ++test.failures;
    std::cerr << "FAIL: real-data tests threw: " << error.what() << '\n';
  }

#ifdef SATVIEW_HAS_CUDA
  test.failures += run_gpu_distribution_tests();
  test.failures += run_gpu_transform_tests();
#endif

  if (test.failures == 0) {
    std::cout << "All satview tests passed\n";
    return 0;
  }
  std::cerr << test.failures << " test(s) failed\n";
  return 1;
}
