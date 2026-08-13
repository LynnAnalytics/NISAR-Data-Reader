#pragma once

#include "satview/aligned_reader.hpp"
#include "satview/analysis_catalog.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace satview::composite {

enum class PageKind : std::uint8_t {
    pauli_rgb,
    compare_pair,
    compare_difference,
    compare_ratio_db,
};

enum class SourceTransform : std::uint8_t {
    amplitude,
    power,
    power_db,
    phase,
    real,
    imaginary,
    linear,
    magnitude,
};

struct PageRequest {
    std::uint64_t serial = 0;
    // Camera generation whose window/lattice produced this request. It is not
    // part of page data identity, but late completions retain it for stale-view
    // publication checks.
    std::uint64_t camera_generation = 0;
    PageKind kind = PageKind::pauli_rgb;
    std::optional<analysis::PauliRecipe> pauli;
    std::optional<analysis::CompareRecipe> compare;
    SourceTransform transform = SourceTransform::power_db;
    Window2D source_window;
    std::array<std::uint64_t, 2> sample_stride{1, 1};
    std::size_t maximum_output_bytes =
        aligned::kDefaultMaximumOutputBytes;
    std::size_t maximum_scratch_bytes =
        aligned::kDefaultMaximumScratchBytes;
    bool prefer_direct_chunk_decode = false;
};

struct PagePlan {
    PageRequest request;
    aligned::AlignedPlan aligned_plan;
};

struct PageProgress {
    std::uint64_t serial = 0;
    aligned::AlignedProgress aligned_progress;
    bool active = false;
};

struct PageCompletion {
    std::uint64_t serial = 0;
    std::optional<PagePlan> plan;
    std::vector<float> values;
    double elapsed_milliseconds = 0.0;
    std::string error;

    [[nodiscard]] bool ready() const noexcept {
        return error.empty() && plan.has_value() && !values.empty();
    }
};

// Data identity for reuse and supersession decisions. Scheduling serial and
// camera generation are deliberately excluded; all recipe, lattice,
// transform, budget, and I/O policy fields participate.
[[nodiscard]] bool same_page_source(
    const PageRequest& left,
    const PageRequest& right);

[[nodiscard]] PagePlan make_page_plan(
    const Hdf5Product& product,
    const PageRequest& request);

// Executes one bounded plan synchronously. A cancelled execution returns an
// empty payload, so no partially assembled composite can be published.
[[nodiscard]] PageCompletion build_page(
    const Hdf5Product& product,
    const PagePlan& plan,
    aligned::CancelCallback cancel = {},
    aligned::ProgressCallback progress = {});

// Serial latest-request-wins worker used by the viewer. The Hdf5Product must
// outlive it; stop() joins before product destruction.
class PageWorker final {
public:
    explicit PageWorker(const Hdf5Product& product);
    ~PageWorker();

    PageWorker(const PageWorker&) = delete;
    PageWorker& operator=(const PageWorker&) = delete;

    void request(PageRequest request);
    // Cancels older work only when serial is strictly newer than every prior
    // request/cancellation serial. Callers own this monotonic generation.
    void supersede(std::uint64_t serial) noexcept;
    [[nodiscard]] PageProgress progress() const;
    [[nodiscard]] std::optional<PageCompletion> try_take_completion();
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace satview::composite
