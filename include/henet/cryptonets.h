/**
 * @file cryptonets.h
 * CryptoNets 明文小网络：灰度图、一层卷积、平方激活、两层全连接。
 */
#ifndef HE_NET_CRYPTONETS_H
#define HE_NET_CRYPTONETS_H

#include "henet/utils.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace henet {
namespace cryptonets {

constexpr size_t kImageSize = 32;
constexpr size_t kRgbChannels = 3;
constexpr size_t kConvOutChannels = 8;
constexpr size_t kKernel = 3;
constexpr size_t kStride = 1;
constexpr size_t kPadding = 1;
constexpr size_t kPool = 4;
constexpr size_t kFc1 = 128;
constexpr size_t kNumClasses = 10;

/** 行优先张量，shape 为 {C,H,W} 或 {N}。 */
struct Tensor {
    vec_t data;
    std::vector<size_t> shape;

    size_t numel() const;
};

/** CryptoNets 明文权重，对应 conv / fc1 / fc2 的 txt 导出。 */
struct Weights {
    vec_t conv_w;
    vec_t conv_b;
    vec_t fc1_w;
    vec_t fc1_b;
    vec_t fc2_w;
    vec_t fc2_b;
};

Tensor conv2d(const Tensor &input,
              const vec_t &weight,
              const vec_t &bias,
              size_t out_channels,
              size_t kernel_size,
              size_t stride,
              size_t padding);

Tensor avg_pool2d(const Tensor &input, size_t kernel_size);

Tensor flatten(const Tensor &input);

Tensor dense(const Tensor &input, const vec_t &weight, const vec_t &bias, size_t out_features);

Tensor square(const Tensor &input);

/** Conv -> x^2 -> AvgPool -> Flatten -> Dense -> x^2 -> Dense -> x^2 */
Tensor infer(const Tensor &input, const Weights &weights);

Tensor rgb_to_gray(const std::vector<uint8_t> &rgb);

Weights load_weights(const std::string &model_dir);

}  // namespace cryptonets
}  // namespace henet

#endif
