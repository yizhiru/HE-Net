#include "henet/cli.h"
#include "test_harness.h"

#include <string>
#include <vector>

HENET_TEST(join_path_inserts_slash) {
    HENET_CHECK_EQ(henet::join_path("a", "b"), std::string("a/b"));
    HENET_CHECK_EQ(henet::join_path("a/", "b"), std::string("a/b"));
    HENET_CHECK_EQ(henet::join_path("", "b"), std::string("b"));
}

HENET_TEST(with_trailing_slash) {
    HENET_CHECK_EQ(henet::with_trailing_slash("ckpt"), std::string("ckpt/"));
    HENET_CHECK_EQ(henet::with_trailing_slash("ckpt/"), std::string("ckpt/"));
}

HENET_TEST(parse_run_config_reads_flags) {
    henet::RunConfig cfg;
    char arg0[] = "henet-tests";
    char arg1[] = "--data";
    char arg2[] = "/tmp/cifar";
    char arg3[] = "--samples";
    char arg4[] = "3";
    char arg5[] = "--threads";
    char arg6[] = "2";
    char *argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
    HENET_CHECK(henet::parse_run_config(7, argv, cfg));
    HENET_CHECK_EQ(cfg.data_dir, std::string("/tmp/cifar"));
    HENET_CHECK_EQ(cfg.num_samples, 3);
    HENET_CHECK_EQ(cfg.num_threads, 2);
    HENET_CHECK_EQ(cfg.help, false);
}

HENET_TEST(parse_run_config_rejects_unknown) {
    henet::RunConfig cfg;
    char arg0[] = "henet-tests";
    char arg1[] = "--nope";
    char *argv[] = {arg0, arg1};
    HENET_CHECK(!henet::parse_run_config(2, argv, cfg));
}
