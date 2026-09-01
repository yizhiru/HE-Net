#include "henet/cryptonets.h"
#include "test_harness.h"

#include <cstdint>
#include <vector>

using henet::cryptonets::Tensor;

HENET_TEST(square_and_flatten) {
    Tensor x{{2.0, -3.0}, {1, 1, 2}};
    auto y = henet::cryptonets::square(x);
    HENET_CHECK_NEAR(y.data[0], 4.0, 1e-12);
    HENET_CHECK_NEAR(y.data[1], 9.0, 1e-12);

    auto flat = henet::cryptonets::flatten(y);
    HENET_CHECK_EQ(flat.shape.size(), 1u);
    HENET_CHECK_EQ(flat.shape[0], 2u);
}

HENET_TEST(avg_pool2d_non_overlapping) {
    Tensor x{{1.0, 2.0, 3.0, 4.0}, {1, 2, 2}};
    auto y = henet::cryptonets::avg_pool2d(x, 2);
    HENET_CHECK_EQ(y.shape[1], 1u);
    HENET_CHECK_EQ(y.shape[2], 1u);
    HENET_CHECK_NEAR(y.data[0], 2.5, 1e-12);
}

HENET_TEST(dense_is_affine) {
    Tensor x{{1.0, 2.0}, {2}};
    const vec_t w = {3.0, 4.0};
    const vec_t b = {5.0};
    auto y = henet::cryptonets::dense(x, w, b, 1);
    HENET_CHECK_NEAR(y.data[0], 16.0, 1e-12);
}

HENET_TEST(conv2d_full_kernel_no_padding) {
    Tensor x{{1.0, 2.0, 3.0, 4.0}, {1, 2, 2}};
    const vec_t w = {1.0, 1.0, 1.0, 1.0};
    const vec_t b = {0.0};
    auto y = henet::cryptonets::conv2d(x, w, b, 1, 2, 1, 0);
    HENET_CHECK_EQ(y.shape[1], 1u);
    HENET_CHECK_EQ(y.shape[2], 1u);
    HENET_CHECK_NEAR(y.data[0], 10.0, 1e-12);
}

HENET_TEST(rgb_to_gray_averages_channels) {
    std::vector<uint8_t> rgb(3 * 32 * 32, 0);
    rgb[0] = 255;          // R
    rgb[32 * 32] = 0;      // G
    rgb[2 * 32 * 32] = 0;  // B
    auto gray = henet::cryptonets::rgb_to_gray(rgb);
    HENET_CHECK_EQ(gray.shape[0], 1u);
    HENET_CHECK_NEAR(gray.data[0], 255.0 / 255.0 / 3.0, 1e-9);
    HENET_CHECK_NEAR(gray.data[1], 0.0, 1e-12);
}
