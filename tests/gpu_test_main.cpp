#include <iostream>

int run_gpu_distribution_tests();
int run_gpu_transform_tests();

namespace {

constexpr int kTestSkipped = 77;

}  // namespace

int main() {
    const int distribution_status = run_gpu_distribution_tests();
    const int transform_status = run_gpu_transform_tests();

    const bool distribution_skipped =
        distribution_status == kTestSkipped;
    const bool transform_skipped = transform_status == kTestSkipped;
    const int failures =
        (distribution_skipped ? 0 : distribution_status) +
        (transform_skipped ? 0 : transform_status);
    if (failures != 0) {
        std::cerr << failures << " CUDA test(s) failed\n";
        return 1;
    }
    if (distribution_skipped || transform_skipped) {
        std::cout << "CUDA runtime tests skipped\n";
        return kTestSkipped;
    }

    std::cout << "All CUDA runtime tests passed\n";
    return 0;
}
