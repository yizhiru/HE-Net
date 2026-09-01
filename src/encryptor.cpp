/**
 * @file encryptor.cpp
 * 将 CIFAR 样本加密为 CKKS 密文，并把 CNN-128 权重量化到对应 level 的明文槽。
 */

#include "henet/encryptor.h"


/**
 * 测试集样本加密
 * @param x_cipher 出参，样本密文，形状[27]，3通道*9卷积核大小，单个密文表示：[置换后2D样本数据]*B次
 * @param x_test 入参，输入测试集单个样本
 * @param sample_index 入参，标识在测试集的索引
 */
void HENetEncryptor::encrypt_sample(vector<Ciphertext> &x_cipher,
                                    const vec_t &x_test) {
    x_cipher.resize(3 * 3 * 3);

    dten test_sample;
    reshape(test_sample, x_test, 3, 32, 32);

    for (int ch = 0; ch < 3; ch++) {
        // CNN padding操作，填充后数据形状 [34][34]
        vector<vector<double>> input_padded;
        vector<double> zero_vec(34, 0.0);
        // 上边界填充
        input_padded.push_back(zero_vec);
        for (int i = 0; i < 32; i++) {
            vector<double> temp_vec;
            // 左边界填充
            temp_vec.push_back(0.0);
            temp_vec.insert(temp_vec.end(), test_sample[ch][i].begin(), test_sample[ch][i].end());
            // 右边界填充
            temp_vec.push_back(0.0);
            input_padded.push_back(temp_vec);
        }
        // 下边界填充
        input_padded.push_back(zero_vec);

        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                // 取填充数据32*32
                vector<double> msg_one_block;
                for (int k = 0; k < 32; k++) {
                    for (int l = 0; l < 32; l++) {
                        // 置换
                        msg_one_block.push_back(input_padded[k + 1 + i][l + 1 + j]);
                    }
                }

                // 将拼接后数据重复组成长向量，编码明文
                vector<double> msg_full;
                for (int l = 0; l < NUM_BLOCKS; ++l) {
                    msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
                }

                Plaintext plain;
                encoder.encode(msg_full, Q_SCALE, plain);
                encryptor.encrypt(plain, x_cipher[9 * ch + 3 * (i + 1) + (j + 1)]);
            }
        }
    }
}

/**
 * B1 块卷积权重编码明文
 * @param w_plain 出参，权重编码明文，形状 [out_channels/16][in_channels*kernel_size*kernel_size], [nch/16][27]
 * @param conv_w 入参，conv_w[0] 卷积核权重形状 [out_channels][in_channels][kernel_size][kernel_size], [nch][3][3][3]
 * @param out_channels 入参，卷积层输出通道数
 */
void HENetEncryptor::encode_conv1_w(vector<vector<Plaintext>> &w_plain,
                                    const vector<ften> &conv_w,
                                    int out_channels) {
    int n_out = out_channels / NUM_BLOCKS;

    w_plain.resize(n_out, vector<Plaintext>(3 * 3 * 3));
    size_t slot_count = encoder.slot_count();

    for (int ch = 0; ch < 3; ++ch) {
        for (int ht = 0; ht < 3; ++ht) {
            for (int wt = 0; wt < 3; ++wt) {
                for (int n = 0; n < n_out; n++) {
                    vector<double> ker_full;
                    // 32*32次重复卷积核权重，BL 组输出信道（该位置）权重拼接，编码成一个明文
                    for (int l = n * NUM_BLOCKS; l < (n + 1) * NUM_BLOCKS; l++) {
                        vector<double> ker_one_block(32 * 32, conv_w[0][l][ch][ht][wt]);
                        ker_full.insert(ker_full.end(), ker_one_block.begin(), ker_one_block.end());
                    }

                    if (ker_full.size() != slot_count) {
                        throw invalid_argument("Error: encode_conv1_kernel");
                    }
                    encoder.encode(ker_full, QC_SCALE, w_plain[n][9 * ch + 3 * ht + wt]);
                }
            }
        }
    }
}

/**
 * 卷积层权重交错编码，用于 B2 块或 B3 块，单个明文表示 (8输出通道*batch_in 输入通道) 个某位置的卷积核权重
 * @param conv_w_p 出参，卷积层参数编码明文，形状 (out_channels/8,in_channels/batch_in,9)
 * @param conv_w 入参，卷积层权重参数，形状 (out_channels,in_channels,kernel_size,kernel_size)
 * @param conv_block 入参，标识 B2 块或 B3 块
 * @param in_channels 入参，输入通道数
 * @param out_channels 入参，输出通道数
 */
void HENetEncryptor::encode_conv_interlace(vector<vector<vector<Plaintext>>> &conv_w_p,
                                           ften conv_w,
                                           int conv_block,
                                           int in_channels,
                                           int out_channels,
                                           int level_gap) {
    int conv_block1 = conv_block - 1;
    // 输出通道数/BL
    size_t num_out = (out_channels / NUM_BLOCKS);
    // 列批大小，通常等于 sqrt(batch_in)
    int dist = 2;
    // 交错数据打包每批输入通道数（批大小），B2 块对应4，B3 块对应 16
    int batch_in = 4;
    if (conv_block == 2) {
        dist = 2;
        batch_in = 4;
    } else if (conv_block == 3 || conv_block == 4) {
        dist = 4;
        batch_in = 16;
    }

    auto param_id_type = get_param_id(level_gap);

    int num_in = (int) in_channels / batch_in;
    conv_w_p.resize(num_out, vector<vector<Plaintext>>(num_in, vector<Plaintext>(NUM_KERNELS)));

    /*
     * 因为每个输入张量密文由BL 个数据块构成，每个数据块又由 batch_in 个输入通道交错编码组成，输入张量密文对应输入通道数 BL*batch_in
     * 输入通道索引可以由三维数组表示 [分组索引][数据块索引][批索引]，也可以由二维数组表示 [整体数据块索引][批索引]
     * 其中，[整体数据块索引] = BL*[分组索引] + [数据块索引]
     */
    MT_EXEC_RANGE(num_out, first, last)
                    // i 为输出通道分组索引，区间 [0:out_channels/16]
                    for (int i = first; i < last; ++i) {
                        // option 为卷积核权重索引，区间 [0:9]
                        for (int f = 0; f < NUM_KERNELS; ++f) {
                            // j 为输入通道整体数据块索引，区间 [0:in_channels/batch_in]
                            for (int j = 0; j < num_in; ++j) {
                                vec_t temp_slots;
                                // 分组结束元素（输入通道）对应整体数据块索引结果再加一，等于 BL*[分组索引] + BL
                                int jend = 8 * ceil((double) (j + 1) / 8.0);
                                // 分组开始元素（输入通道）对应整体数据块索引，等于 BL*[分组索引]
                                int jstart = jend - NUM_BLOCKS;
                                // 分组开始元素（输入通道）在整体输入通道中的索引
                                int jstart_batch = jstart * batch_in;

                                // j1 为输入通道整体数据块索引，区间 [j:jend]
                                for (int j1 = j; j1 < jend; ++j1) {
                                    vec_t temp;
                                    dten kernels;
                                    // k 为批索引
                                    for (int k = 0; k < batch_in; k++) {
                                        /*
                                         * 输出通道整体索引 = 分组开始元素（输出通道）整体索引 + 输出通道数据块索引（从0开始）
                                         * 输出通道数据块索引区间 [0:jend-j]
                                         *
                                         * 输入通道整体索引 = 分组开始元素（输入通道）整体索引 + 数据块索引 + 批索引*BL
                                         * 之所以批索引乘以BL，是因为交错编码时，相邻元素（输入信道）相隔 BL
                                         */
                                        kernels.push_back(
                                                conv_w[i * 8 + (j1 - j)][jstart_batch + (j1 - jstart) + k * 8]);
                                    }
                                    kernel_to_interlaced_vector(temp, kernels, f, dist);
                                    temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                                }

                                // j1 为输入通道整体数据块索引，区间 [jstart:j]
                                for (int j1 = jstart; j1 < j; ++j1) {
                                    vec_t temp;
                                    dten kernels;
                                    for (int k = 0; k < batch_in; k++) {
                                        // 整体输出通道索引 = 分组开始元素（输出通道）对应整体索引 + 输出通道数据块索引
                                        // 输出通道数据块索引区间 [8-j+jstart:8]=[jend-j:8]
                                        int idx = jstart_batch + (j1 - jstart) + k * 8;
                                        kernels.push_back(conv_w[i * 8 + (8 - j + j1)][idx]);
                                    }
                                    kernel_to_interlaced_vector(temp, kernels, f, dist);
                                    temp_slots.insert(temp_slots.end(), temp.begin(), temp.end());
                                }

                                if (f < 4) {
                                    // rho^{-r_k}(.)
                                    rotate_left_vec_inplace(temp_slots, STEPS_CONV[conv_block1][7 - f]);
                                } else if (f > 4) {
                                    // rho^{-r_k}(.)
                                    rotate_right_vec_inplace(temp_slots, STEPS_CONV[conv_block1][f - 1]);
                                }
                                encoder.encode(temp_slots, param_id_type, QC_SCALE, conv_w_p[i][j][f]);
                            }
                        }
                    }
    MT_EXEC_RANGE_END

}


/**
 * B2 块 mask 编码明文，索引 [2*i][2*j] 置1，其余位置置0，提取卷积核的卷积结果，从而mask 其他值
 * @param mask 出参，mask 明文
 */
void HENetEncryptor::encode_mask_conv2(Plaintext &mask) {
    int num_rows = 32;
    int num_cols = 32;
    vector<double> msg_one_block(num_rows * num_cols, 0ULL);
    // 对应二维数组位置 [2*i][2*j] 置1，其余位置置0
    for (int i = 0; i < num_rows; i += 2) {
        for (int j = 0; j < num_cols; j += 2) {
            msg_one_block[i * 32 + j] = 1.0;
        }
    }
    vector<double> msg_full;
    // 重复
    for (int l = 0; l < NUM_BLOCKS; ++l) {
        msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
    }

    auto param_id_type = get_param_id(3);
    encoder.encode(msg_full, param_id_type, QC_SMALL_SCALE, mask);
}

/**
 * B2 块 mask 编码明文，索引 [4*i][4*j] 置1，其余位置置0，提取卷积核的卷积结果，从而mask 其他值
 * @param mask_poly2 出参，mask 明文
 */
void HENetEncryptor::encode_mask_conv3(Plaintext &mask, int level_gap) {
    int num_rows = 32;
    int num_cols = 32;
    vector<double> msg_one_block(num_rows * num_cols, 0ULL);
    // 对应二维数组位置 [4*i][4*j] 置1，其余位置置0
    for (int i = 0; i < num_rows; i += 4) {
        for (int j = 0; j < num_cols; j += 4) {
            msg_one_block[i * 32 + j] = 1.0;
        }
    }
    vector<double> msg_full;
    // 重复
    for (int l = 0; l < NUM_BLOCKS; ++l) {
        msg_full.insert(msg_full.end(), msg_one_block.begin(), msg_one_block.end());
    }

    auto param_id_type = get_param_id(level_gap);
    encoder.encode(msg_full, param_id_type, QC_SMALL_SCALE, mask);
}



/**
 * 聚合激活层编码明文
 * @param act_plain 出参，聚合激活层权重编码明文，形状 (out_channels/16,3)
 * @param act_poly 入参，聚合激活层权重，形状 (out_channels,3)
 * @param out_channels 入参，卷积层输出通道数
 * @param dist 入参，相邻权重间隔距离
 */
void HENetEncryptor::encode_act(vector<vector<Plaintext>> &act_plain,
                                const dmat &act_poly,
                                int out_channels,
                                int dist) {
    int n_out = out_channels / NUM_BLOCKS;
    act_plain.resize(n_out, vector<Plaintext>(3));
    size_t slot_count = encoder.slot_count();

    for (int k = 0; k < n_out * 3; ++k) {
        // 输出通道分组索引
        int i = (int) floor(k / 3.0);
        // 集合激活层多项式次数索引，0为常数项，1为一次项，2为二次项
        int l = (k % 3);

        vec_t temp_full_slots;
        for (int j = 0; j < NUM_BLOCKS; ++j) {
            vec_t temp_short;
            generate_fixed_vector(temp_short, act_poly[i * NUM_BLOCKS + j][l], 32, 32, dist);
            temp_full_slots.insert(temp_full_slots.end(), temp_short.begin(), temp_short.end());
        }
        if (temp_full_slots.size() != slot_count)
            throw invalid_argument("Error: encode_act_kernel");

        if (l == 0) {
            encoder.encode(temp_full_slots, Q_SCALE, act_plain[i][0]);
        } else if (l == 1) {
            encoder.encode(temp_full_slots, QC_SCALE, act_plain[i][1]);
        } else if (l == 2) {
            encoder.encode(temp_full_slots, QC_SCALE, act_plain[i][2]);
        }
    }
}

/**
 * 全连接层权重明文编码
 * @param dense_w_p 出参，全连接层权重明文，形状 (in_features/16,16)，单个明文包括了10个输出特征对应的权重
 * @param dense_w 入参，全连接层权重，形状 (out_features,in_features)，(10,in_features)
 * @param in_features 入参，输入特征数
 * @param level_gap 入参，距离首level 的层数，用来计算当前处于level
 */
void HENetEncryptor::encode_dense_w(vector<vector<vector<Plaintext>>> &dense_w_p,
                                    dmat dense_w,
                                    int in_features,
                                    int level_gap) {
    size_t slot_count = encoder.slot_count();
    int num_in = in_features / NUM_BLOCKS;
    // (2, in_features/8,8)
    dense_w_p.resize(2, vector<vector<Plaintext>>(num_in, vector<Plaintext>(NUM_BLOCKS)));
    auto param_id_type = get_param_id(level_gap);

    MT_EXEC_RANGE(num_in, first, last)
                    // i 为输入特征的分组索引，取值区间 [0:in_features/8]
                    for (int i = first; i < last; ++i) {
                        // i1 = i*8
                        int i1 = (i << 3);
                        int diff = i1;

                        // j 为输入特征的数据块索引，取 [0:4]，保证j+l 小于8
                        for (int j = 0; j < 4; ++j) {
                            vec_t temp_slots(slot_count, 0.0);
                            vec_t temp_slots2(slot_count, 0.0);
                            // l 为输出特征的索引，取 [0:5]
                            for (int l = 0; l < 5; ++l) {
                                // 明文索引 l1 = (j+l)*1024，对应明文数据块索引 [j+l]
                                int l1 = ((j + l) << LOG_SIZE_OF_BLOCK);
                                // 权重索引 [l][8*i+j+l]
                                temp_slots[l1] = dense_w[l][diff + l];
                                temp_slots2[l1] = dense_w[l + 5][diff + l];
                            }
                            encoder.encode(temp_slots, param_id_type, QC_SCALE, dense_w_p[0][i][j]);
                            encoder.encode(temp_slots2, param_id_type, QC_SCALE, dense_w_p[1][i][j]);
                            diff++;
                        }


                        // j 取 [4:8]
                        for (int j = 4; j < 8; ++j) {
                            vec_t temp_slots(slot_count, 0.0);
                            vec_t temp_slots2(slot_count, 0.0);
                            // 头部
                            for (int l = 0; l < j - 3; ++l) {
                                // 明文索引 l1 = l*1024
                                int l1 = (l << LOG_SIZE_OF_BLOCK);
                                int xcord = (i1 + l);
                                // 权重索引 [l-j+8][i*8+l]
                                temp_slots[l1] = dense_w[l - j + 8][xcord];
                                temp_slots2[l1] = dense_w[l - j + 13][xcord];
                            }

                            // 尾部
                            for (int l = j; l < 8; ++l) {
                                // 明文索引 l1 = l*1024
                                int l1 = (l << LOG_SIZE_OF_BLOCK);
                                int ycord = l - j;
                                // 权重索引 [l-j][i*8+l]
                                temp_slots[l1] = dense_w[ycord][diff + ycord];
                                temp_slots2[l1] = dense_w[ycord + 5][diff + ycord];
                            }
                            encoder.encode(temp_slots, param_id_type, QC_SCALE, dense_w_p[0][i][j]);
                            encoder.encode(temp_slots2, param_id_type, QC_SCALE, dense_w_p[1][i][j]);
                            diff++;
                        }
                    }
    MT_EXEC_RANGE_END
}

/**
 * 全连接层偏置项明文编码
 * @param dense_b_p 出参，全连接层偏置项明文
 * @param dense_b 入参，全连接层偏置项，形状 (10,)
 * @param level_gap 入参，距离首level 的层数
 */
void HENetEncryptor::encode_dense_b(vector<Plaintext> &dense_b_p,
                                    const vec_t dense_b,
                                    int level_gap) {
    size_t slot_count = encoder.slot_count();
    // 平均分为2组，每组对应5个输出特征
    int num_out = 2;
    dense_b_p.resize(num_out);
    auto param_id_type = get_param_id(level_gap);

    MT_EXEC_RANGE(num_out, first, last)
                    for (int i = first; i < last; ++i) {
                        vec_t temp_slots(slot_count, 0ULL);
                        int end = 5 * (i + 1) - 1;
                        int start = end - 4;
                        for (int j = start; j < end; ++j) {
                            int idx = (i << LOG_SIZE_OF_BLOCK);
                            temp_slots[idx] = dense_b[j];
                        }
                        encoder.encode(temp_slots, param_id_type, Q_SCALE, dense_b_p[i]);
                    }
    MT_EXEC_RANGE_END
}

/*
 * 模型明文编码
 */
void HENetEncryptor::encode_model(EncodedHENetModel &model,
                                  const ModelWeights &mwp) {
    // B1 块
    encode_conv1_w(model.conv1_w_p, mwp.conv_weights, OUT_CHANNELS1);
    encode_act(model.act1_poly, mwp.act_poly[0], OUT_CHANNELS1, 1);

    // B2 块
    encode_conv_interlace(model.conv2_w_p, mwp.conv_weights[1], 2,
                          OUT_CHANNELS1, OUT_CHANNELS2, 4);
    encode_mask_conv2(model.mask_conv2);
    encode_act(model.act2_poly, mwp.act_poly[1], OUT_CHANNELS2, 2);

    // B3 块
    encode_conv_interlace(model.conv3_w_p, mwp.conv_weights[2], 3,
                          OUT_CHANNELS2, OUT_CHANNELS3, 8);
    encode_mask_conv3(model.mask_conv3, 7);
    encode_act(model.act3_poly, mwp.act_poly[2], OUT_CHANNELS3, 4);

    // B4 块
//    encode_conv_interlace(model.conv4_w_p, mwp.conv_weights[3], 4,
//                          OUT_CHANNELS3, OUT_CHANNELS4, 12);
//    encode_mask_conv3(model.mask_conv4, 11);
//    encode_act(model.act4_poly, mwp.act_poly[3], OUT_CHANNELS4, 4);


    // Dense
//    encode_dense_w(model.dense_w_p, mwp.dense_weight, OUT_CHANNELS4, 15);
//    encode_dense_b(model.dense_b_p, mwp.dense_bias, 16);
}



