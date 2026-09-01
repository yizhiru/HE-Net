/**
 * @file henet_plain.cpp
 * CryptoNets 明文推理入口。与 henet-seal 使用的 CNN-128 不是同一套网络。
 */

#include "henet/cli.h"
#include "henet/cryptonets.h"
#include "henet/utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    henet::RunConfig cfg;
    if (!henet::parse_run_config(argc, argv, cfg)) {
        return 1;
    }
    if (cfg.help) {
        henet::print_usage(argv[0]);
        return 0;
    }

    std::cout << "henet-plain: CryptoNets grayscale baseline (not CNN-128)\n";
    std::cout << "data=" << cfg.data_dir << std::endl;
    std::cout << "model=" << cfg.model_dir << std::endl;

    henet::cryptonets::Weights weights;
    try {
        weights = henet::cryptonets::load_weights(cfg.model_dir);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n"
                  << "Need CryptoNets weights: conv.weight.txt, conv.bias.txt, "
                  << "fc1.weight.txt, fc1.bias.txt, fc2.weight.txt, fc2.bias.txt\n"
                  << "These are not the CNN-128 files exported by python/train.py.\n";
        return 1;
    }

    auto [x_test, y_test] = load_cifar_dataset(cfg.data_dir);
    size_t n = y_test.size();
    if (cfg.num_samples > 0) {
        n = std::min(n, static_cast<size_t>(cfg.num_samples));
    }

    double correct = 0.0;
    for (size_t i = 0; i < n; ++i) {
        auto logits = henet::cryptonets::infer(henet::cryptonets::rgb_to_gray(x_test[i]), weights);
        const auto best = std::max_element(logits.data.begin(), logits.data.end());
        const auto pred = static_cast<uint8_t>(std::distance(logits.data.begin(), best));
        if (pred == y_test[i]) {
            correct += 1.0;
        }
    }

    std::cout << "samples=" << n << std::endl;
    std::cout << "accuracy=" << (n == 0 ? 0.0 : correct / static_cast<double>(n)) << std::endl;
    return 0;
}
