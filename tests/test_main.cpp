#include "test_harness.h"

#include <iostream>

int main() {
    std::cout << "Running " << henet::test::registry().size() << " tests\n";
    for (const auto &entry: henet::test::registry()) {
        std::cout << " - " << entry.first << "\n";
        entry.second();
    }
    if (henet::test::failed_count() != 0) {
        std::cerr << henet::test::failed_count() << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "All tests passed\n";
    return 0;
}
