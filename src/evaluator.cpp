
/**
 * @file evaluator.cpp
 * CNN 各层的密态求值：交错卷积、多项式激活、平均池化与全连接。
 */

#include "henet/evaluator.h"

/**
 * B1 块卷积的密态计算，nn.Conv2d(3, first_f, kernel_size=(3, 3), stride=1, padding=1)
 * @param res 出参，卷积结果，形状 (out_channels/8,)，单个密文表示8个输出通道卷积结果（32,32）
 * @param x_cipher 入参，样本数据密文，形状 [27]，单个密文表示 [置换后2D样本数据]重复8次
 * @param conv1_w 入参，卷积核权重明文，形状 [out_channels/8][27]，单个明文表示 8个输出通道 ([权重值]*(32*32))
 * @param out_channels 入参，B1 块卷积输出通道数
 * @param evaluator 入参，CKKS 评估器
 */
void HENetEvaluator::conv1_fhe_eval(vector<Ciphertext> &res,
                                    const vector<Ciphertext> &x_cipher,
                                    const vector<vector<Plaintext>> &conv1_w,
                                    int out_channels) {
    int n_out = out_channels / 8;
    res.resize(n_out);

    MT_EXEC_RANGE(n_out, first, last)
                    for (int n = first; n < last; ++n) {
                        // 单个卷积核权重乘以所有对应2D 图像值
                        evaluator.multiply_plain(x_cipher[0], conv1_w[n][0], res[n]);
                        // 累加，3输入通道*9卷积核大小
                        for (int i = 1; i < 27; i++) {
                            Ciphertext ctemp;
                            evaluator.multiply_plain(x_cipher[i], conv1_w[n][i], ctemp);
                            evaluator.add_inplace(res[n], ctemp);
                        }
                        evaluator.rescale_to_next_inplace(res[n]);
                    }
    MT_EXEC_RANGE_END
}


/**
 * 聚合激活层（包括Conv-bias、激活函数、BN层）密态计算
 * @param ct 入参出参，密态数据，形状 (ct.size,)
 * @param act_poly 入参，聚合激活层权重明文，形状 (ct.size,3)
 */
void HENetEvaluator::bn_act_fhe_eval(vector<Ciphertext> &ct,
                                     vector<vector<Plaintext>> act_poly) {
    MT_EXEC_RANGE(ct.size(), first, last)
                    for (int i = first; i < last; ++i) {
//    for (int i = 0; i < ct.size(); ++i) {
                        // 二次项 ct[i]^2 * act_poly[i][2]
                        Ciphertext ct2;
                        evaluator.square(ct[i], ct2);
                        evaluator.relinearize_inplace(ct2, relin_keys);
                        evaluator.rescale_to_next_inplace(ct2);

                        evaluator.mod_switch_to_inplace(act_poly[i][2], ct2.parms_id());
                        evaluator.multiply_plain_inplace(ct2, act_poly[i][2]);

                        // 一次项 ct[i] * act_poly[i][1]
                        evaluator.mod_switch_to_inplace(ct[i], ct2.parms_id());
                        evaluator.mod_switch_to_inplace(act_poly[i][1], ct[i].parms_id());
                        evaluator.multiply_plain_inplace(ct[i], act_poly[i][1]);

                        // 结果相加 ct[i]^2 * act_poly[i][2] + ct[i] * act_poly[i][1]
                        ct[i].scale() = ct2.scale();
                        evaluator.add_inplace(ct2, ct[i]);
                        evaluator.rescale_to_next_inplace(ct2);

                        // 常数项 act_poly[i][0]
                        act_poly[i][0].scale() = ct2.scale();
                        evaluator.mod_switch_to_inplace(act_poly[i][0], ct2.parms_id());
                        // 结果相加 ct[i]^2 * act_poly[i][2] + ct[i] * act_poly[i][1] + act_poly[i][0]
                        evaluator.add_plain_inplace(ct2, act_poly[i][0]);
                        ct[i] = ct2;
                    }
    MT_EXEC_RANGE_END
}

/**
 * B1 块平均池化的密态计算，nn.AvgPool2d(kernel_size=2, stride=2)
 * @param ct 入参出参，输入形状 (out_channels/8,)，单个密文数据块有效数据形状 (32,32)；
 *              输出平均池化结果，形状 (out_channels/8,)，单个密文数据块表示池化结果，有效数据形状 (16,16)
 *              结果索引 (i,j)  对应一维数组索引 64*i+2*j
 */
void HENetEvaluator::avg_pool1_fhe_eval(vector<Ciphertext> &ct) {
    MT_EXEC_RANGE(ct.size(), first, last)
                    for (int i = first; i < last; ++i) {
                        Ciphertext temp;
                        // 池化窗口的大小为(2,2)，旋转两次相加
                        evaluator.rotate_vector(ct[i], 1, gal_keys, temp);
                        evaluator.add_inplace(ct[i], temp);

                        evaluator.rotate_vector(ct[i], 32, gal_keys, temp);
                        evaluator.add_inplace(ct[i], temp);
                    }
    MT_EXEC_RANGE_END
}

/**
 * 交错编码，参看论文 prepocessing-step 示意图
 * @param res 出参，旋转相加结果，形状 (4,) 或 (2,)
 * @param ct 入参，输入张量
 * @param mask_poly 入参，mask 向量，用于提取有用元素，mask 无用元素
 * @param conv_block 入参，标识 B2 块或 B3 块
 */
void HENetEvaluator::interlace_input(vector<Ciphertext> &res,
                                     vector<Ciphertext> &ct,
                                     Plaintext mask_poly,
                                     int conv_block) {
    if (conv_block == 2) {
        MT_EXEC_RANGE(ct.size(), first, last)
                        for (int i = first; i < last; ++i) {
                            // 仅保留偶数位索引 [2*i][2*j]，其余索引置0
                            evaluator.multiply_plain_inplace(ct[i], mask_poly);
                        }
        MT_EXEC_RANGE_END

        if (ct.size() == 16) {
            MT_EXEC_RANGE(12, first, last)
                            for (int i = first; i < last; ++i) {
                                if (i == 0) {
                                    evaluator.rotate_vector_inplace(ct[1], -1, gal_keys);
                                } else if (i == 1) {
                                    evaluator.rotate_vector_inplace(ct[2], -32, gal_keys);
                                } else if (i == 2) {
                                    evaluator.rotate_vector_inplace(ct[3], -33, gal_keys);
                                } else if (i == 3) {
                                    evaluator.rotate_vector_inplace(ct[5], -1, gal_keys);
                                } else if (i == 4) {
                                    evaluator.rotate_vector_inplace(ct[6], -32, gal_keys);
                                } else if (i == 5) {
                                    evaluator.rotate_vector_inplace(ct[7], -33, gal_keys);
                                } else if (i == 6) {
                                    evaluator.rotate_vector_inplace(ct[9], -1, gal_keys);
                                } else if (i == 7) {
                                    evaluator.rotate_vector_inplace(ct[10], -32, gal_keys);
                                } else if (i == 8) {
                                    evaluator.rotate_vector_inplace(ct[11], -33, gal_keys);
                                } else if (i == 9) {
                                    evaluator.rotate_vector_inplace(ct[13], -1, gal_keys);
                                } else if (i == 10) {
                                    evaluator.rotate_vector_inplace(ct[14], -32, gal_keys);
                                } else if (i == 11) {
                                    evaluator.rotate_vector_inplace(ct[15], -33, gal_keys);
                                }
                            }
            MT_EXEC_RANGE_END

            /*
             * 交错编码批大小为4，密文数组大小为16，则可以分为4组
             * 第一组：ct[0] + rho^{-1}(ct[1]) + rho^{-32}(ct[2]) + rho^{-33}(ct[3])
             * 第二组：ct[4] + rho^{-1}(ct[5]) + rho^{-32}(ct[6]) + rho^{-33}(ct[7])
             * 第三组：ct[8] + rho^{-1}(ct[9]) + rho^{-32}(ct[10]) + rho^{-33}(ct[11])
             * 第四组：ct[12] + rho^{-1}(ct[13]) + rho^{-32}(ct[14]) + rho^{-33}(ct[15])
             */
            res.resize(4);
            MT_EXEC_RANGE(4, first, last)
                            for (int i = first; i < last; ++i) {
                                res[i] = ct[4 * i];
                                for (int j = 1; j < 4; ++j) {
                                    evaluator.add_inplace(res[i], ct[4 * i + j]);
                                }
                                evaluator.rescale_to_next_inplace(res[i]);
                            }
            MT_EXEC_RANGE_END
        }
    } else if (conv_block == 3) {
        MT_EXEC_RANGE(ct.size(), first, last)
                        // 仅保留索引 [4*i][j*j]，其余索引置0
                        for (int i = first; i < last; ++i) {
                            evaluator.multiply_plain_inplace(ct[i], mask_poly);
                        }
        MT_EXEC_RANGE_END

        /*
         * 交错编码批大小为16，密文数组大小为32，则交错编码可将密文数组共分为2组
         * 第一组：
         * ct[0] + rho^{-1}(ct[1]) + rho^{-2}(ct[2]) + rho^{-3}(ct[3])
         *   + rho^{-32}(ct[4]) + ... + rho^{-35}(ct[7])
         *   + rho^{-64}(ct[8]) + ... + rho^{-67}(ct[11])
         *   + rho^{-96}(ct[12]) + ... + rho^{-99}(ct[15])
         * 第二组：
         * ct[16] + rho^{-1}(ct[17]) + rho^{-2}(ct[18]) + rho^{-3}(ct[19])
         *   + rho^{-32}(ct[20]) + ... + rho^{-35}(ct[23])
         *   + rho^{-64}(ct[24]) + ... + rho^{-67}(ct[37])
         *   + rho^{-96}(ct[28]) + ... + rho^{-99}(ct[31])
         */
        MT_EXEC_RANGE(15, first, last)
                        for (int n = first; n < last; ++n) {
                            int n1 = n + 1;
                            int rot_amount = -((n1 % 4) + 32 * (int) floor((double) n1 / (double) 4));
                            evaluator.rotate_vector_inplace(ct[n + 1], rot_amount, gal_keys);
                        }
        MT_EXEC_RANGE_END

        MT_EXEC_RANGE(15, first, last)
                        for (int n = first; n < last; ++n) {
                            int n1 = n + 1;
                            int rot_amount = -((n1 % 4) + 32 * (int) floor((double) n1 / (double) 4));
                            evaluator.rotate_vector_inplace(ct[n + 17], rot_amount, gal_keys);
                        }
        MT_EXEC_RANGE_END


        res.resize(2);
        res[0] = ct[0];
        for (int j = 1; j < 16; ++j) {
            evaluator.add_inplace(res[0], ct[j]);
        }
        evaluator.rescale_to_next_inplace(res[0]);

        res[1] = ct[16];
        for (int j = 17; j < 32; j++) {
            evaluator.add_inplace(res[1], ct[j]);
        }
        evaluator.rescale_to_next_inplace(res[1]);
    }
}

/**
 * 旋转输入张量，对应论文中giant 策略，计算 rho^{(2^10)*j}(ct[i])
 * @param res 出参，旋转输入张量的结果，形状 (1, ct.size, 8)
 * @param ct 入参，输入张量
 * @param conv_block 入参，标识 B2 块或 B3 块
 */
void HENetEvaluator::generate_rotations_giant(vector<vector<Ciphertext>> &res,
                                              vector<Ciphertext> ct,
                                              int conv_block) {

    int num_in = (int) ct.size();
    int num_rot = (int) STEPS_GIANT.size();
    int total_num = num_in * num_rot;
    res.resize(num_in, vector<Ciphertext>(num_rot + 1));

    for (int i = 0; i < num_in; ++i) {
        res[i][0] = ct[i];
    }

    // res[i][j+1] = rho^{(j+1)*(2^10)}(ct[i])
    MT_EXEC_RANGE(total_num, first, last)
                    for (int k = first; k < last; k++) {
                        // 输入张量数组索引，区间 [0:ct.size]
                        int i = k / num_rot;
                        // 旋转步数数组索引（数据块索引-1），区间 [0:num_rot]
                        int j = k % num_rot;
                        evaluator.rotate_vector(ct[i], STEPS_GIANT[j],
                                                gal_keys, res[i][j + 1]);
                    }
    MT_EXEC_RANGE_END
}


/**
 * B2 块卷积的密态计算，nn.Conv2d(nch, 2*nch, kernel_size=3, stride=1, padding=1)
 * @param res 出参，卷积结果，形状 (out_channels/8,)，单个密文表示8个输出通道卷积结果，有效数据形状 (16,16)
 *              卷积结果索引[i][j] 对应结果密文向量索引 [64*i+2*j]
 * @param input 入参，输入张量，形状 (in_channels/8,)
 * @param conv2_w 入参，卷积核权重明文，形状 (out_channels/8,in_channels/4,9)，单个明文表示 (8输出通道*4输入通道) 个卷积核权重
 * @param mask_poly2 入参，mask 编码明文
 * @param in_channels 入参，卷积输入通道数，nch
 * @param out_channels 入参，卷积输出通道数，2*nch
 */
void HENetEvaluator::conv2_fhe_eval(vector<Ciphertext> &res,
                                    vector<Ciphertext> &input,
                                    const vector<vector<vector<Plaintext>>> &conv_w_p,
                                    const Plaintext &conv_mask,
                                    int out_channels) {
    // 第一步：预处理，数据交错打包，参看论文 Pre-processing 示意图
    vector<Ciphertext> in_packed;
    interlace_input(in_packed, input, conv_mask, 2);
    int num_in = (int) in_packed.size();

    // 输出通道数除以8
    int num_out = out_channels / NUM_BLOCKS;

    /*
     * 第二步，普通同态卷积（Ordinary homomorphic convolution）
     * 输入：
     *   输入张量密文 in_packed，形状 (4,)，单个密文有8个数据块，每个数据块中4个输入通道为一组进行交错编码
     *   权重卷积核明文 conv_w_p，形状 (out_channels/8,in_channels/4,9)，单个明文表示 (8输出通道*4输入通道) 个卷积核权重
     * 输出：
     *   卷积结果 res，形状 (out_channels/8,)，单个密文表示8个输出通道的卷积中间结果
     *     单个数据块对应4个输入通道卷积中间结果
     */
    vector<vector<Ciphertext>> rot;
    // 旋转输入张量, 形状 (4,8)，计算 rho^{(2^10)*j}(ct[i])，用于后续计算 MultPlain
    generate_rotations_giant(rot, in_packed, 2);
    res.resize(num_out);

    MT_EXEC_RANGE(num_out, first, last)
                    // i 为输出通道分组索引，区间 [0:out_channels/8]
                    for (int i = first; i < last; ++i) {
                        // l 为卷积核位置索引，区间 [0:9]
                        for (int l = 0; l < NUM_KERNELS; l++) {
                            int ker_index;
                            if (l == 0) {
                                ker_index = MIDDEL_INDEX_OF_KERNELS;
                            } else if (l < (MIDDEL_INDEX_OF_KERNELS + 1)) {
                                ker_index = l - 1;
                            } else {
                                ker_index = l;
                            }

                            Ciphertext ctemp;
                            Ciphertext ctempl;
                            // j 为输入通道数据块索引
                            for (int j = 0; j < NUM_BLOCKS; ++j) {
                                // k 为输入通道分组索引，区间 [0:4]
                                for (int k = 0; k < num_in; ++k) {
                                    // j0 为输入通道整体数据块索引，区间 [0:in_channels/batch_in]
                                    int j0 = j + NUM_BLOCKS * k;
                                    if (j0 == 0) {
                                        // MultPlain(rho^{j*(2^10)}(res_packed[k]), rho^{-r_k}(ptF_{i,j,k,l}))
                                        evaluator.multiply_plain(rot[k][j],
                                                                 conv_w_p[i][j0][ker_index], ctempl);
                                    } else {
                                        evaluator.multiply_plain(rot[k][j],
                                                                 conv_w_p[i][j0][ker_index], ctemp);
                                        evaluator.add_inplace(ctempl, ctemp);
                                    }
                                }
                            }

                            if (l == 0) {
                                res[i] = ctempl;
                            } else {
                                // rho^{r_k}(.)
                                evaluator.rotate_vector_inplace(
                                        ctempl, STEPS_CONV[1][l - 1],gal_keys);
                                evaluator.add_inplace(res[i], ctempl);
                            }
                        }
                        evaluator.rescale_to_next_inplace(res[i]);
                    }
    MT_EXEC_RANGE_END

    /*
     * 第三步，后处理 Post-processing
     * 将一个数据块内4（批大小）个输入通道卷积结果累加，从而得到最终卷积结果；计算方法类似于滑动步幅等于2的池化操作
     */
    MT_EXEC_RANGE(num_out, first, last)
                    for (int i = first; i < last; ++i) {
                        Ciphertext ctemp;
                        evaluator.rotate_vector(res[i], 1, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);

                        evaluator.rotate_vector(res[i], 32, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);
                    }
    MT_EXEC_RANGE_END
}


/**
 * B2 块平均池化的密态计算，nn.AvgPool2d(kernel_size=2, stride=2)
 * @param ct 入参出参，输入形状 (out_channels/8,)，单个密文有效数据 (16,16)
 *              输出平均池化结果，形状 (out_channels/8,)，单个密文数据块表示池化结果，有效数据形状 (8,8)
 */
void HENetEvaluator::avg_pool2_fhe_eval(vector<Ciphertext> &ct) {
    MT_EXEC_RANGE(ct.size(), first, last)
                    for (int i = first; i < last; ++i) {
                        Ciphertext ctemp;
                        // 旋转两次，r[0,0] = ct[0,0] + ct[0,2] + ct[2,0] + ct[2,2]
                        evaluator.rotate_vector(ct[i], 2, gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], 64, gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);
                    }
    MT_EXEC_RANGE_END
}


void HENetEvaluator::conv3_fhe_eval(vector<Ciphertext> &res,
                                    vector<Ciphertext> &input,
                                    const vector<vector<vector<Plaintext>>> &conv_w_p,
                                    const Plaintext &conv_mask,
                                    int out_channels) {
    // 第一步：预处理，数据交错打包，参看论文 Pre-processing 示意图
    vector<Ciphertext> in_packed;
    interlace_input(in_packed, input, conv_mask, 3);
    int num_in = (int) in_packed.size();

    // 输出通道数除以8
    int num_out = out_channels / NUM_BLOCKS;

    vector<vector<Ciphertext>> rot;
    // 旋转输入张量, 形状 (4,8)，计算 rho^{(2^10)*j}(ct[i])，用于后续计算 MultPlain
    generate_rotations_giant(rot, in_packed, 3);
    res.resize(num_out);

    MT_EXEC_RANGE(num_out, first, last)
                    // i 为输出通道分组索引，区间 [0:out_channels/8]
                    for (int i = first; i < last; ++i) {
                        // l 为卷积核位置索引，区间 [0:9]
                        for (int l = 0; l < NUM_KERNELS; l++) {
                            int ker_index;
                            if (l == 0) {
                                ker_index = MIDDEL_INDEX_OF_KERNELS;
                            } else if (l < (MIDDEL_INDEX_OF_KERNELS + 1)) {
                                ker_index = l - 1;
                            } else {
                                ker_index = l;
                            }

                            Ciphertext ctemp;
                            Ciphertext ctempl;
                            // j 为输入通道数据块索引
                            for (int j = 0; j < NUM_BLOCKS; ++j) {
                                // k 为输入通道分组索引，区间 [0:4]
                                for (int k = 0; k < num_in; ++k) {
                                    // j0 为输入通道整体数据块索引，区间 [0:in_channels/batch_in]
                                    int j0 = j + NUM_BLOCKS * k;
                                    if (j0 == 0) {
                                        // MultPlain(rho^{j*(2^10)}(res_packed[k]), rho^{-r_k}(ptF_{i,j,k,l}))
                                        evaluator.multiply_plain(rot[k][j],
                                                                 conv_w_p[i][j0][ker_index], ctempl);
                                    } else {
                                        evaluator.multiply_plain(rot[k][j],
                                                                 conv_w_p[i][j0][ker_index], ctemp);
                                        evaluator.add_inplace(ctempl, ctemp);
                                    }
                                }
                            }

                            if (l == 0) {
                                res[i] = ctempl;
                            } else {
                                // rho^{r_k}(.)
                                evaluator.rotate_vector_inplace(ctempl, STEPS_CONV[2][l - 1], gal_keys);
                                evaluator.add_inplace(res[i], ctempl);
                            }
                        }
                        evaluator.rescale_to_next_inplace(res[i]);
                    }
    MT_EXEC_RANGE_END

    /*
     * 第三步，后处理 Post-processing
     * 将一个数据块内16（批大小）个输入通道卷积结果累加，从而得到最终卷积结果
     */
    MT_EXEC_RANGE(num_out, first, last)
                    for (int i = first; i < last; ++i) {
                        Ciphertext ctemp;
                        evaluator.rotate_vector(res[i], 1, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);

                        evaluator.rotate_vector(res[i], 2, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);

                        evaluator.rotate_vector(res[i], 32, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);

                        evaluator.rotate_vector(res[i], 64, gal_keys, ctemp,
                                                MemoryPoolHandle().New(false));
                        evaluator.add_inplace(res[i], ctemp);
                    }
    MT_EXEC_RANGE_END
}

/**
 * B2 块自适应平均池化的密态计算，nn.AdaptiveAvgPool2d((1, 1))
 * @param ct 入参出参，输入形状 (out_channels/8,)，单个密文的数据块有效数据形状 (8,8)
 *              输出平均池化结果，形状 (out_channels/8,)，单个密文的数据块有效数据形状 (1,)
 */
void HENetEvaluator::adaptive_avg_pool_fhe_eval(vector<Ciphertext> &ct) {
    int row_shift_amount[] = {32 * 4, 32 * 4 * 2, 32 * 4 * 4};
    int column_shift_amount[] = {4, 8, 16};

    MT_EXEC_RANGE(ct.size(), first, last)
                    for (int i = first; i < last; ++i) {
                        Ciphertext ctemp;
                        evaluator.rotate_vector(ct[i], row_shift_amount[0], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], row_shift_amount[1], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], row_shift_amount[2], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], column_shift_amount[0], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], column_shift_amount[1], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);

                        evaluator.rotate_vector(ct[i], column_shift_amount[2], gal_keys, ctemp);
                        evaluator.add_inplace(ct[i], ctemp);
                    }
    MT_EXEC_RANGE_END
}


/**
 * 全连接层密态推理
 * @param res 出参，计算结果，形状 (2,)，每条密文表示 5 个类别的输出结果
 * @param input 入参，输入张量，形状 (in_features/8,)，每条密文包括 8个输入特征
 * @param dense_w_p 入参，全连接权重明文，形状 ()
 * @param dense_b_p 入参
 * @param in_features 入参，输入特征数
 */
void HENetEvaluator::dense_fhe_eval(vector<Ciphertext> &res,
                                    vector<Ciphertext> &input,
                                    vector<vector<vector<Plaintext>>> &dense_w_p,
                                    vector<Plaintext> &dense_b_p,
                                    int in_features) {
    int num_in = in_features / NUM_BLOCKS;
    res.resize(2);
    // 输出特征分组数2*数据块数8
    int group_blocks = 16;
    vector<vector<Ciphertext>> ctemp;
    ctemp.resize(2, vector<Ciphertext>(NUM_BLOCKS));

    MT_EXEC_RANGE(group_blocks, first, last)
                    // k 为整体数据块索引, k = 8*g+j
                    for (int k = first; k < last; ++k) {
                        // g 为输出特征分组索引
                        int g = floor(k / NUM_BLOCKS);
                        // j 为数据块索引
                        int j = k % NUM_BLOCKS;
                        // 第一分组相乘
                        evaluator.multiply_plain(input[0], dense_w_p[g][0][j], ctemp[g][j]);

                        Ciphertext temp;
                        // i 为输入特征分组索引，不同分组的同一数据块累加
                        for (int i = 1; i < num_in; ++i) {
                            evaluator.multiply_plain(input[i], dense_w_p[g][i][j], temp);
                            evaluator.add_inplace(ctemp[g][j], temp);
                        }

                        // rho^{j*(2^10)}(.)
                        if (j != 0) {
                            evaluator.rotate_vector_inplace(ctemp[g][j], STEPS_GIANT[j - 1], gal_keys);
                        }
                    }
    MT_EXEC_RANGE_END

    // 数据块累加
    MT_EXEC_RANGE(2, first, last)
                    for (int g = first; g < last; g++) {
                        res[g] = ctemp[g][0];
                        for (int j = 1; j < NUM_BLOCKS; j++) {
                            evaluator.add_inplace(res[g], ctemp[g][j]);
                        }
                        evaluator.rescale_to_next_inplace(res[g]);

                        // 加上偏置项
                        dense_b_p[g].scale() = res[g].scale();
                        evaluator.add_plain_inplace(res[g], dense_b_p[g]);
                    }
    MT_EXEC_RANGE_END
}

/*
 * 模型密态推理
 */
void HENetEvaluator::model_fhe_inference(vector<Ciphertext> &res,
                                         const vector<Ciphertext> &input,
                                         EncodedHENetModel &model) {
    // B1 块
    vector<Ciphertext> res1;
    conv1_fhe_eval(res1, input, model.conv1_w_p, OUT_CHANNELS1);
    bn_act_fhe_eval(res1, model.act1_poly);
    avg_pool1_fhe_eval(res1);

    // B2 块
    vector<Ciphertext> res2;
    conv2_fhe_eval(res2, res1, model.conv2_w_p, model.mask_conv2, OUT_CHANNELS2);
    bn_act_fhe_eval(res2, model.act2_poly);
    avg_pool2_fhe_eval(res2);

    // B3 块
//    vector<Ciphertext> res3;
//    conv3_fhe_eval(res3, res2, model.conv3_w_p, model.mask_conv3, OUT_CHANNELS3);
//    bn_act_fhe_eval(res3, model.act3_poly);

//    adaptive_avg_pool_fhe_eval(res3);

    // Dense
//    dense_fhe_eval(res, res3, model.dense_w_p, model.dense_b_p, out_channels3);

    res = res2;
}

