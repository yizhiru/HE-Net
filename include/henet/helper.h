#ifndef HE_NET_HELPER_H
#define HE_NET_HELPER_H

#include "henet/utils.h"

/** CKKS 系数模数位宽。 */
const int LOG_Q = 24;
const int LOG_QC = 24;
const int LOG_QC_SMALL = 21;
const int LOG_Q0 = 33;
const int LOG_P0 = 33;

const double Q_SCALE = pow(2.0, LOG_Q);
const double QC_SCALE = pow(2.0, LOG_QC);
const double QC_SMALL_SCALE = pow(2.0, LOG_QC_SMALL);

/** 一条密文中的 32x32 数据块个数：8192 / 1024 = 8。 */
const int NUM_BLOCKS = 8;
const int SIZE_OF_BLOCK = 1024;
const int LOG_SIZE_OF_BLOCK = 10;
const int NUM_KERNELS = 9;
const int MIDDEL_INDEX_OF_KERNELS = 4;

const int OUT_CHANNELS1 = 128;
const int OUT_CHANNELS2 = 2 * OUT_CHANNELS1;
const int OUT_CHANNELS3 = 4 * OUT_CHANNELS1;
const int OUT_CHANNELS4 = 1 * OUT_CHANNELS1;

const vector<vector<int>> STEPS_CONV = {
        {-33,     -32,     -31,     -1,     1,     31,     32,     33},
        {-33 * 2, -32 * 2, -31 * 2, -1 * 2, 1 * 2, 31 * 2, 32 * 2, 33 * 2},
        {-33 * 4, -32 * 4, -31 * 4, -1 * 4, 1 * 4, 31 * 4, 32 * 4, 33 * 4},
        {-33 * 4, -32 * 4, -31 * 4, -1 * 4, 1 * 4, 31 * 4, 32 * 4, 33 * 4},
};

const vector<int> STEPS_GIANT = {
        (1 << 10), (2 << 10), (3 << 10), (4 << 10), (5 << 10),
        (6 << 10), (7 << 10)
};

const vector<int> STEPS_POOL = {4, 8, 16, 32 * 4, 32 * 4 * 2, 32 * 4 * 4};

const vector<int> STEPS_INTERLACE = {-3, -33, -34, -35, -64, -65, -66, -67, -96, -97, -98, -99};

typedef vector<vector<double>> dmat;
typedef vector<vector<vector<double>>> dten;
typedef vector<vector<vector<vector<double>>>> ften;

/** CNN-128 明文权重：卷积、融合后的二次激活、全连接。 */
struct ModelWeights {
    vector<ften> conv_weights;
    vector<dmat> act_poly;
    dmat dense_weight;
    vec_t dense_bias;
};

void reshape(dmat &res, vec_t input, int dim1, int dim2);
void reshape(dten &res, vec_t input, int dim1, int dim2, int dim3);
void reshape(ften &res, vec_t input, int dim1, int dim2, int dim3, int dim4);

void load_model_weights(ModelWeights &mwp, const string &model_dir);

void aggregate_bias_bn_act(dmat &real_poly, vec_t bias,
                           vec_t gamma, vec_t beta, vec_t mean, vec_t var,
                           vec_t act_poly, double pool_size, double epsilon);

void generate_fixed_vector(vec_t &res, double val, int rows, int cols, int dist);

void kernel_to_interlaced_vector(vec_t &res, dten kernels, int f, int dist);

void rotate_left_vec_inplace(vec_t &vec, int steps);
void rotate_right_vec_inplace(vec_t &vec, int steps);

int arg_max(vec_t scores);

#endif
