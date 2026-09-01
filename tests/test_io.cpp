#include "henet/utils.h"
#include "test_harness.h"

#include <stdexcept>
#include <string>

HENET_TEST(read_vec_from_fixture) {
    const std::string path = std::string(HENET_TEST_FIXTURE_DIR) + "/sample.csv";
    vec_t v;
    read_vec_from_file(v, path);
    HENET_CHECK_EQ(v.size(), 3u);
    HENET_CHECK_NEAR(v[0], 1.5, 1e-12);
    HENET_CHECK_NEAR(v[1], 2.25, 1e-12);
    HENET_CHECK_NEAR(v[2], -4.0, 1e-12);
}

HENET_TEST(read_val_from_fixture) {
    const std::string path = std::string(HENET_TEST_FIXTURE_DIR) + "/scalar.csv";
    HENET_CHECK_NEAR(read_val_from_file(path), 0.125, 1e-12);
}

HENET_TEST(read_vec_missing_file_throws) {
    vec_t v;
    bool threw = false;
    try {
        read_vec_from_file(v, "/no/such/henet_weights.txt");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    HENET_CHECK(threw);
}
