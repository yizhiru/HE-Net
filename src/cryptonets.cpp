/**
 * @file cryptonets.cpp
 * CryptoNets 明文算子实现，供 henet-plain 与单元测试共用。
 */

#include "henet/cryptonets.h"

#include "henet/cli.h"

namespace henet {
namespace cryptonets {

size_t Tensor::numel() const {
    size_t n = 1;
    for (size_t d: shape) {
        n *= d;
    }
    return n;
}

namespace {

size_t at3(size_t c, size_t h, size_t w, size_t height, size_t width) {
    return c * height * width + h * width + w;
}

}  // namespace

Tensor conv2d(const Tensor &input,
              const vec_t &weight,
              const vec_t &bias,
              size_t out_channels,
              size_t kernel_size,
              size_t stride,
              size_t padding) {
    const size_t in_h = input.shape[1];
    const size_t in_w = input.shape[2];
    const size_t pad_h = in_h + 2 * padding;
    const size_t pad_w = in_w + 2 * padding;

    vec_t padded(pad_h * pad_w, 0.0);
    for (size_t h = 0; h < in_h; ++h) {
        for (size_t w = 0; w < in_w; ++w) {
            padded[(h + padding) * pad_w + (w + padding)] = input.data[h * in_w + w];
        }
    }

    const size_t out_h = (pad_h - kernel_size) / stride + 1;
    const size_t out_w = (pad_w - kernel_size) / stride + 1;
    vec_t out(out_channels * out_h * out_w, 0.0);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t h = 0; h < out_h; ++h) {
            for (size_t w = 0; w < out_w; ++w) {
                double acc = bias[oc];
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        const size_t ih = h * stride + kh;
                        const size_t iw = w * stride + kw;
                        const size_t wi = oc * kernel_size * kernel_size + kh * kernel_size + kw;
                        acc += weight[wi] * padded[ih * pad_w + iw];
                    }
                }
                out[at3(oc, h, w, out_h, out_w)] = acc;
            }
        }
    }
    return {out, {out_channels, out_h, out_w}};
}

Tensor avg_pool2d(const Tensor &input, size_t kernel_size) {
    const size_t channels = input.shape[0];
    const size_t in_h = input.shape[1];
    const size_t in_w = input.shape[2];
    const size_t out_h = in_h / kernel_size;
    const size_t out_w = in_w / kernel_size;
    const double inv_area = 1.0 / static_cast<double>(kernel_size * kernel_size);

    vec_t out(channels * out_h * out_w, 0.0);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < out_h; ++h) {
            for (size_t w = 0; w < out_w; ++w) {
                double acc = 0.0;
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        acc += input.data[at3(c, h * kernel_size + kh, w * kernel_size + kw, in_h, in_w)];
                    }
                }
                out[at3(c, h, w, out_h, out_w)] = acc * inv_area;
            }
        }
    }
    return {out, {channels, out_h, out_w}};
}

Tensor flatten(const Tensor &input) {
    return {input.data, {input.numel()}};
}

Tensor dense(const Tensor &input, const vec_t &weight, const vec_t &bias, size_t out_features) {
    const size_t in_features = input.shape[0];
    vec_t out(out_features, 0.0);
    for (size_t i = 0; i < out_features; ++i) {
        double acc = bias[i];
        for (size_t j = 0; j < in_features; ++j) {
            acc += input.data[j] * weight[i * in_features + j];
        }
        out[i] = acc;
    }
    return {out, {out_features}};
}

Tensor square(const Tensor &input) {
    vec_t out = input.data;
    for (double &v: out) {
        v *= v;
    }
    return {out, input.shape};
}

Tensor infer(const Tensor &input, const Weights &weights) {
    Tensor x = conv2d(input, weights.conv_w, weights.conv_b, kConvOutChannels, kKernel, kStride, kPadding);
    x = square(x);
    x = avg_pool2d(x, kPool);
    x = flatten(x);
    x = dense(x, weights.fc1_w, weights.fc1_b, kFc1);
    x = square(x);
    x = dense(x, weights.fc2_w, weights.fc2_b, kNumClasses);
    return square(x);
}

Tensor rgb_to_gray(const std::vector<uint8_t> &rgb) {
    vec_t gray(kImageSize * kImageSize, 0.0);
    for (size_t h = 0; h < kImageSize; ++h) {
        for (size_t w = 0; w < kImageSize; ++w) {
            double acc = 0.0;
            for (size_t c = 0; c < kRgbChannels; ++c) {
                acc += static_cast<double>(rgb[at3(c, h, w, kImageSize, kImageSize)]) / 255.0;
            }
            gray[h * kImageSize + w] = acc / static_cast<double>(kRgbChannels);
        }
    }
    return {gray, {1, kImageSize, kImageSize}};
}

Weights load_weights(const std::string &model_dir) {
    Weights w;
    read_vec_from_file(w.conv_w, join_path(model_dir, "conv.weight.txt"));
    read_vec_from_file(w.conv_b, join_path(model_dir, "conv.bias.txt"));
    read_vec_from_file(w.fc1_w, join_path(model_dir, "fc1.weight.txt"));
    read_vec_from_file(w.fc1_b, join_path(model_dir, "fc1.bias.txt"));
    read_vec_from_file(w.fc2_w, join_path(model_dir, "fc2.weight.txt"));
    read_vec_from_file(w.fc2_b, join_path(model_dir, "fc2.bias.txt"));
    return w;
}

}  // namespace cryptonets
}  // namespace henet
