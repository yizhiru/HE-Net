/**
 * @file thread_pool.cpp
 * 简单线程池，供密态卷积并行求值。
 */

#include "henet/thread_pool.h"

namespace Thread {

    ThreadPool * threadPool_ptr__;

    void initThreadPool(size_t n) {
        threadPool_ptr__ = new ThreadPool(n);
    }
}
