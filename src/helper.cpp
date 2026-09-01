#include "henet/cli.h"
#include "henet/helper.h"

/** 从 txt 导出目录加载 CNN-128 权重，并把 BN / bias / 激活融合成二次多项式。 */
void load_model_weights(ModelWeights &mwp, const string &model_dir) {
    vector<int> nch_vec = {3, OUT_CHANNELS1, OUT_CHANNELS2, OUT_CHANNELS3, OUT_CHANNELS4};
    vector<int> pool_size_vec = {4, 4, 1, 64};

    // 读取卷积层权重、偏置项，BN层参数，激活函数参数
    for (int i = 0; i < 4; ++i) {
        string index_name = to_string(i + 1);
        string weight_file = henet::join_path(model_dir, "conv" + index_name + ".weight.txt");
        vec_t conv_weight;
        read_vec_from_file(conv_weight, weight_file);
        reshape(mwp.conv_weights[i], conv_weight, nch_vec[i + 1], nch_vec[i], 3, 3);

        string bias_file = henet::join_path(model_dir, "conv" + index_name + ".bias.txt");
        vec_t conv_bias;
        read_vec_from_file(conv_bias, bias_file);

        vec_t mean, var, gamma, beta;
        string file_name = henet::join_path(model_dir, "bn" + index_name + ".running_mean.txt");
        read_vec_from_file(mean, file_name);
        file_name = henet::join_path(model_dir, "bn" + index_name + ".running_var.txt");
        read_vec_from_file(var, file_name);
        file_name = henet::join_path(model_dir, "bn" + index_name + ".weight.txt");
        read_vec_from_file(gamma, file_name);
        file_name = henet::join_path(model_dir, "bn" + index_name + ".bias.txt");
        read_vec_from_file(beta, file_name);

        vec_t act_coeff;
        act_coeff.push_back(read_val_from_file(henet::join_path(model_dir, "act" + index_name + ".c.txt")));
        act_coeff.push_back(read_val_from_file(henet::join_path(model_dir, "act" + index_name + ".beta.txt")));
        act_coeff.push_back(read_val_from_file(henet::join_path(model_dir, "act" + index_name + ".alpha.txt")));

        mwp.act_poly[i].resize(nch_vec[i + 1], vec_t(3));
        aggregate_bias_bn_act(mwp.act_poly[i], conv_bias,
                              gamma, beta, mean, var,
                              act_coeff, pool_size_vec[i], 1e-05);
    }

    // 读取全连接层权重、偏置项
    vec_t dense_w;
    read_vec_from_file(dense_w, henet::join_path(model_dir, "linear.weight.txt"));
    reshape(mwp.dense_weight, dense_w, 10, nch_vec[4]);

    read_vec_from_file(mwp.dense_bias, henet::join_path(model_dir, "linear.bias.txt"));
}

/** 将一维向量排成 [dim1][dim2]。 */
void reshape(dmat &res, vec_t input, int dim1, int dim2) {
    res.resize(dim1, vector<double>(dim2));

    int index = 0;
    for (int k = 0; k < dim1; ++k) {
        for (int l = 0; l < dim2; ++l) {
            res[k][l] = input[index];
            index++;
        }
    }
}

/** 将一维向量排成 [dim1][dim2][dim3]。 */
void reshape(dten &res, vec_t input, int dim1, int dim2, int dim3) {
    res.resize(dim1, vector<vector<double>>(dim2, vector<double>(dim3)));

    int index = 0;
    for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
            for (int l = 0; l < dim3; ++l) {
                res[j][k][l] = input[index];
                index++;
            }
        }
    }
}

/** 将一维向量排成 [dim1][dim2][dim3][dim4]。 */
void reshape(ften &res, vec_t input, int dim1, int dim2, int dim3, int dim4) {
    res.resize(dim1,
               vector<vector<vector<double>>>(dim2, vector<vector<double>>(dim3, vector<double>(dim4))));

    int index = 0;
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                for (int l = 0; l < dim4; ++l) {
                    res[i][j][k][l] = input[index];
                    index++;
                }
            }
        }
    }
}

/**
 * 融合 conv-bias、BN 与二次激活。
 * y = x + bias
 * z = gamma / sqrt(var+eps) * y + (beta - gamma*mean/sqrt(var+eps)) = d0*x + d1
 * poly(x) = a0 + a1*z + a2*z^2，再除以 pool_size。
 */
void aggregate_bias_bn_act(dmat &real_poly, vec_t bias,
                           vec_t gamma, vec_t beta, vec_t mean, vec_t var,
                           vec_t act_poly, double pool_size, double epsilon) {
    if (real_poly.size() != bias.size()) {
        throw invalid_argument("Error: initialization of real_poly");
    }

    for (size_t i = 0; i < bias.size(); ++i) {
        double d0 = gamma[i] / sqrt(var[i] + epsilon);
        double d1 = d0 * (bias[i] - mean[i]) + beta[i];

        // 常数项
        real_poly[i][0] = ((act_poly[2] * d1 * d1) + (act_poly[1] * d1) + act_poly[0]) / pool_size;
        // 一次项
        real_poly[i][1] = (2 * act_poly[2] * d0 * d1 + act_poly[1] * d0) / pool_size;
        // 二次项
        real_poly[i][2] = (act_poly[2] * d0 * d0) / pool_size;
    }
}

/** 在 [i*dist][j*dist] 位置写入 val，其余为 0。 */
void generate_fixed_vector(vec_t &res, double val, int rows, int cols, int dist) {
    int len = rows * cols;
    res.assign(len, 0.0);

    for (int i = 0; i < rows; i += dist) {
        for (int j = 0; j < cols; j += dist) {
            res[i * cols + j] = val;
        }
    }
}


/** 把 3x3 核的第 f 个位置交错铺进 32x32 槽位。 */
void kernel_to_interlaced_vector(vec_t &res, dten kernels, int f, int dist) {
    int nrows = 32;
    int ncols = 32;
    // 截止有效列索引
    int ncols1 = ncols;

    res.assign(nrows * ncols, 0ULL);

    // 行批大小，batch_in/dist = dist
    int packed_nrows = kernels.size() / dist;
    // 列批大小，dist
    int packed_ncols = dist;

    switch (f) {
        case 0:
            // 因为有padding，所以要去掉头边界、左边界；因为是输入张量是输入通道交错编码，所以迭代 += dist
            for (int i = dist; i < nrows; i += dist) {
                for (int j = dist; j < ncols1; j += dist) {
                    // k 为行批索引，l 为列批索引
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][0];
                        }
                    }
                }
            }
            break;
        case 1:
            // 去掉头边界
            for (int i = dist; i < nrows; i += dist) {
                for (int j = 0; j < ncols1; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][1];
                        }
                    }
                }
            }
            break;
        case 2:
            // 去掉头边界、右边界
            for (int i = dist; i < nrows; i += dist) {
                for (int j = 0; j < ncols1 - dist; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][0][2];
                        }
                    }
                }
            }
            break;
        case 3:
            // 去掉左边界
            for (int i = 0; i < nrows; i += dist) {
                for (int j = dist; j < ncols1; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][0];
                        }
                    }
                }
            }
            break;
        case 4:
            // 无须去掉
            for (int i = 0; i < nrows; i += dist) {
                for (int j = 0; j < ncols1; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][1];
                        }
                    }
                }
            }
            break;
        case 5:
            // 去掉右边界
            for (int i = 0; i < nrows; i += dist) {
                for (int j = 0; j < ncols1 - dist; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][1][2];
                        }
                    }
                }
            }
            break;
        case 6:
            // 去掉下边界、左边界
            for (int i = 0; i < nrows - dist; i += dist) {
                for (int j = dist; j < ncols1; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][0];
                        }
                    }
                }
            }
            break;
        case 7:
            // 去掉下边界
            for (int i = 0; i < nrows - dist; i += dist) {
                for (int j = 0; j < ncols1; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][1];
                        }
                    }
                }
            }
            break;
        case 8:
            // 去掉下边界、右边界
            for (int i = 0; i < nrows - dist; i += dist) {
                for (int j = 0; j < ncols1 - dist; j += dist) {
                    for (int k = 0; k < packed_nrows; k++) {
                        for (int l = 0; l < packed_ncols; l++) {
                            res[i * ncols + j + k * ncols + l] = kernels[k * dist + l][2][2];
                        }
                    }
                }
            }
            break;
        default:
            break;
    }
}

/** 向量循环左移 steps 步。 */
void rotate_left_vec_inplace(vec_t &vec, int steps) {
    vec_t res;
    int dim = (int) vec.size();
    int shift = steps % dim;
    int k = dim - shift;

    // 先取 vec[shift:dim]
    for (int j = 0; j < k; ++j) {
        res.push_back(vec[j + shift]);
    }
    // 再取 vec[0:shift]
    for (int j = k; j < dim; ++j) {
        res.push_back(vec[j - k]);
    }

    for (int j = 0; j < dim; ++j) {
        vec[j] = res[j];
    }
}

/** 向量循环右移 steps 步。 */
void rotate_right_vec_inplace(vec_t &vec, int steps) {
    vec_t res;
    int dim = (int) vec.size();
    int shift = steps % dim;
    int k = dim - shift;

    // 先取 vec[k:dim]
    for (int j = 0; j < dim - k; ++j) {
        res.push_back(vec[j + k]);
    }
    // 再取 vec[0:k]
    for (int j = 0; j < k; ++j) {
        res.push_back(vec[j]);
    }

    // 结果复制到 vec
    for (int j = 0; j < dim; ++j) {
        vec[j] = res[j];
    }
}

/** 返回最大值下标。 */
int arg_max(vec_t scores) {
    int label = 0;
    double max_score = scores[0];
    for (int i = 0; i < scores.size(); ++i) {
        if (max_score < scores[i]) {
            max_score = scores[i];
            label = i;
        }
    }
    return label;
}


