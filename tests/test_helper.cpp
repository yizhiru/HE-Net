#include "henet/helper.h"
#include "test_harness.h"

HENET_TEST(reshape_2d_fills_row_major) {
    dmat out;
    reshape(out, {1.0, 2.0, 3.0, 4.0}, 2, 2);
    HENET_CHECK_EQ(out.size(), 2u);
    HENET_CHECK_NEAR(out[0][0], 1.0, 1e-12);
    HENET_CHECK_NEAR(out[0][1], 2.0, 1e-12);
    HENET_CHECK_NEAR(out[1][0], 3.0, 1e-12);
    HENET_CHECK_NEAR(out[1][1], 4.0, 1e-12);
}

HENET_TEST(reshape_4d_preserves_order) {
    ften out;
    vec_t flat(2 * 2 * 1 * 1);
    flat[0] = 9.0;
    flat[1] = 8.0;
    flat[2] = 7.0;
    flat[3] = 6.0;
    reshape(out, flat, 2, 2, 1, 1);
    HENET_CHECK_NEAR(out[0][0][0][0], 9.0, 1e-12);
    HENET_CHECK_NEAR(out[1][1][0][0], 6.0, 1e-12);
}

HENET_TEST(aggregate_bias_bn_act_identity_bn) {
    dmat poly(1, vec_t(3, 0.0));
    const vec_t bias = {0.0};
    const vec_t gamma = {1.0};
    const vec_t beta = {0.0};
    const vec_t mean = {0.0};
    const vec_t var = {1.0};
    const vec_t act = {1.0, 2.0, 3.0};  // c, linear, quadratic
    aggregate_bias_bn_act(poly, bias, gamma, beta, mean, var, act, 1.0, 0.0);
    HENET_CHECK_NEAR(poly[0][0], 1.0, 1e-9);
    HENET_CHECK_NEAR(poly[0][1], 2.0, 1e-9);
    HENET_CHECK_NEAR(poly[0][2], 3.0, 1e-9);
}

HENET_TEST(generate_fixed_vector_strided) {
    vec_t out;
    generate_fixed_vector(out, 7.0, 4, 4, 2);
    HENET_CHECK_EQ(out.size(), 16u);
    HENET_CHECK_NEAR(out[0], 7.0, 1e-12);
    HENET_CHECK_NEAR(out[2], 7.0, 1e-12);
    HENET_CHECK_NEAR(out[8], 7.0, 1e-12);
    HENET_CHECK_NEAR(out[10], 7.0, 1e-12);
    HENET_CHECK_NEAR(out[1], 0.0, 1e-12);
    HENET_CHECK_NEAR(out[5], 0.0, 1e-12);
}

HENET_TEST(rotate_left_and_right) {
    vec_t left = {1.0, 2.0, 3.0, 4.0};
    rotate_left_vec_inplace(left, 1);
    HENET_CHECK_NEAR(left[0], 2.0, 1e-12);
    HENET_CHECK_NEAR(left[3], 1.0, 1e-12);

    vec_t right = {1.0, 2.0, 3.0, 4.0};
    rotate_right_vec_inplace(right, 1);
    HENET_CHECK_NEAR(right[0], 4.0, 1e-12);
    HENET_CHECK_NEAR(right[1], 1.0, 1e-12);
}

HENET_TEST(arg_max_returns_first_peak) {
    HENET_CHECK_EQ(arg_max({0.1, 0.5, 0.2}), 1);
    HENET_CHECK_EQ(arg_max({3.0, 1.0, 2.0}), 0);
}

HENET_TEST(kernel_center_weight_is_placed_on_grid) {
    dten kernels(1, std::vector<vec_t>(3, vec_t(3, 0.0)));
    kernels[0][1][1] = 5.0;
    vec_t slots;
    kernel_to_interlaced_vector(slots, kernels, 4, 1);
    HENET_CHECK_EQ(slots.size(), 1024u);
    HENET_CHECK_NEAR(slots[0], 5.0, 1e-12);
    HENET_CHECK_NEAR(slots[1], 5.0, 1e-12);
    HENET_CHECK_NEAR(slots[32], 5.0, 1e-12);
}
