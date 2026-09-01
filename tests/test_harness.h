#ifndef HE_NET_TEST_HARNESS_H
#define HE_NET_TEST_HARNESS_H

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace henet {
namespace test {

inline int &failed_count() {
    static int n = 0;
    return n;
}

inline std::vector<std::pair<std::string, void (*)()>> &registry() {
    static std::vector<std::pair<std::string, void (*)()>> tests;
    return tests;
}

struct RegisterTest {
    RegisterTest(const char *name, void (*fn)()) {
        registry().emplace_back(name, fn);
    }
};

}  // namespace test
}  // namespace henet

#define HENET_TEST(name)                                                       \
    static void name();                                                        \
    static henet::test::RegisterTest _henet_reg_##name(#name, name);           \
    static void name()

#define HENET_CHECK(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__              \
                      << "  " << #cond << "\n";                                \
            ++henet::test::failed_count();                                     \
        }                                                                      \
    } while (0)

#define HENET_CHECK_EQ(a, b) HENET_CHECK((a) == (b))

#define HENET_CHECK_NEAR(a, b, eps)                                            \
    do {                                                                       \
        const double _va = static_cast<double>(a);                             \
        const double _vb = static_cast<double>(b);                             \
        if (std::fabs(_va - _vb) > (eps)) {                                    \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__              \
                      << "  " << _va << " !~ " << _vb << "\n";                 \
            ++henet::test::failed_count();                                     \
        }                                                                      \
    } while (0)

#endif
