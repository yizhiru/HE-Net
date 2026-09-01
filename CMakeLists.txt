cmake_minimum_required(VERSION 3.16)

# CMake 4 + Apple Clang 不会自动填 SDK，导致 <cstdlib>/<thread> 找不到
if(CMAKE_HOST_APPLE AND NOT CMAKE_OSX_SYSROOT)
    execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )
    if(CMAKE_OSX_SYSROOT)
        set(CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT}" CACHE PATH "Apple SDK" FORCE)
        message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")
    endif()
endif()

project(HE-Net VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
    add_link_options(-stdlib=libc++)
    # 新版 Apple Clang 会优先搜 CLT 自带的残缺 libc++，需显式加上 SDK 中的完整头文件
    if(CMAKE_OSX_SYSROOT AND EXISTS "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1/cstdlib")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-isystem${CMAKE_OSX_SYSROOT}/usr/include/c++/v1>)
        message(STATUS "Using libc++ headers: ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(HENET_BUILD_SEAL "Build CKKS encrypted inference (requires Microsoft SEAL)" ON)
option(HENET_BUILD_TESTS "Build unit tests" ON)

set(HENET_DEFAULT_DATA_DIR "${CMAKE_SOURCE_DIR}/data/cifar-10/cifar-10-batches-bin")
set(HENET_DEFAULT_MODEL_DIR "${CMAKE_SOURCE_DIR}/python/checkpoints/CIFAR_10/fp32")

add_library(henet_common
        src/helper.cpp
        src/thread_pool.cpp
        src/cryptonets.cpp
)
target_include_directories(henet_common
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_definitions(henet_common
        PUBLIC
        HENET_DEFAULT_DATA_DIR="${HENET_DEFAULT_DATA_DIR}"
        HENET_DEFAULT_MODEL_DIR="${HENET_DEFAULT_MODEL_DIR}"
)

if(HENET_BUILD_SEAL)
    if(CMAKE_HOST_APPLE)
        list(PREPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
    endif()

    find_package(SEAL 4.1 QUIET)
    if(NOT SEAL_FOUND)
        message(STATUS "Microsoft SEAL 4.1 not found; fetching v4.1.2")
        include(FetchContent)
        # SEAL 4.1 的 zlib 依赖声明了过旧的 cmake_minimum_required，CMake 4 需要此兼容项
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
        set(SEAL_BUILD_DEPS ON CACHE BOOL "" FORCE)
        set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
        set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)
        set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
        set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(SEAL_BUILD_SEAL_C OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
                SEAL
                GIT_REPOSITORY https://github.com/microsoft/SEAL.git
                GIT_TAG v4.1.2
        )
        FetchContent_MakeAvailable(SEAL)
    endif()
    find_package(OpenMP)

    add_executable(henet-seal
            src/henet_seal.cpp
            src/encryptor.cpp
            src/evaluator.cpp
    )
    target_link_libraries(henet-seal PRIVATE henet_common)

    if(TARGET SEAL::seal)
        target_link_libraries(henet-seal PRIVATE SEAL::seal)
    elseif(TARGET SEAL::seal_shared)
        target_link_libraries(henet-seal PRIVATE SEAL::seal_shared)
    else()
        message(FATAL_ERROR "Cannot find SEAL::seal or SEAL::seal_shared")
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(henet-seal PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(HENET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
