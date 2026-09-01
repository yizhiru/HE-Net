#ifndef HE_NET_CLI_H
#define HE_NET_CLI_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#ifndef HENET_DEFAULT_DATA_DIR
#define HENET_DEFAULT_DATA_DIR "data/cifar-10/cifar-10-batches-bin"
#endif

#ifndef HENET_DEFAULT_MODEL_DIR
#define HENET_DEFAULT_MODEL_DIR "python/checkpoints/CIFAR_10/fp32"
#endif

namespace henet {

struct RunConfig {
    std::string data_dir = HENET_DEFAULT_DATA_DIR;
    std::string model_dir = HENET_DEFAULT_MODEL_DIR;
    int num_samples = 1;
    int num_threads = 0;
    bool help = false;
};

inline std::string with_trailing_slash(std::string path) {
    if (path.empty()) {
        return path;
    }
    const char last = path.back();
    if (last != '/' && last != '\\') {
        path.push_back('/');
    }
    return path;
}

inline std::string join_path(const std::string &dir, const std::string &file) {
    if (dir.empty()) {
        return file;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

inline void print_usage(const char *prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --data PATH     CIFAR-10 cifar-10-batches-bin directory\n"
        << "  --model PATH    Exported weight directory\n"
        << "  --samples N     Number of test samples (default: 1; 0 = all)\n"
        << "  --threads N     Worker threads (default: hardware concurrency)\n"
        << "  -h, --help      Show this help\n";
}

/** 解析 --data / --model / --samples / --threads。失败时打印用法。 */
inline bool parse_run_config(int argc, char **argv, RunConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            cfg.help = true;
            return true;
        } else if (arg == "--data") {
            const char *value = require_value("--data");
            if (!value) {
                return false;
            }
            cfg.data_dir = value;
        } else if (arg == "--model") {
            const char *value = require_value("--model");
            if (!value) {
                return false;
            }
            cfg.model_dir = value;
        } else if (arg == "--samples") {
            const char *value = require_value("--samples");
            if (!value) {
                return false;
            }
            cfg.num_samples = std::atoi(value);
        } else if (arg == "--threads") {
            const char *value = require_value("--threads");
            if (!value) {
                return false;
            }
            cfg.num_threads = std::atoi(value);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (cfg.num_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        cfg.num_threads = hw > 0 ? static_cast<int>(hw) : 4;
    }
    cfg.model_dir = with_trailing_slash(cfg.model_dir);
    return true;
}

}  // namespace henet

#endif
