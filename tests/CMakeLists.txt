add_executable(henet-tests
        test_main.cpp
        test_helper.cpp
        test_cryptonets.cpp
        test_cli.cpp
        test_io.cpp
)
target_include_directories(henet-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(henet-tests PRIVATE henet_common)
target_compile_definitions(henet-tests
        PRIVATE
        HENET_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

add_test(NAME henet-tests COMMAND henet-tests)

# CryptoNets 端到端样例：依赖真实 CIFAR 与权重，不注册为 ctest
add_executable(henet-plain henet_plain.cpp)
target_link_libraries(henet-plain PRIVATE henet_common)
