#ifndef HE_NET_HENET_EVALUATOR_H
#define HE_NET_HENET_EVALUATOR_H

#include <vector>
#include <seal/seal.h>
#include "henet/encryptor.h"
#include "henet/helper.h"
#include "henet/thread_pool.h"
#include "henet/utils.h"

using namespace std;
using namespace seal;

/** 在密文上评估 CNN 各层：卷积、多项式激活、池化、全连接。 */
class HENetEvaluator {
private:
    Evaluator &evaluator;
    RelinKeys &relin_keys;
    GaloisKeys &gal_keys;

public:
    HENetEvaluator(Evaluator &evaluator, RelinKeys &relin_keys, GaloisKeys &gal_keys) :
            evaluator(evaluator), relin_keys(relin_keys), gal_keys(gal_keys) {};

    void conv1_fhe_eval(vector<Ciphertext> &res,
                        const vector<Ciphertext> &x_cipher,
                        const vector<vector<Plaintext>> &conv1_w,
                        int out_channels);

    void bn_act_fhe_eval(vector<Ciphertext> &ct,
                         vector<vector<Plaintext>> act_poly);

    void avg_pool1_fhe_eval(vector<Ciphertext> &ct);

    void interlace_input(vector<Ciphertext> &res,
                         vector<Ciphertext> &ct,
                         Plaintext mask_poly,
                         int conv_block);

    void generate_rotations_giant(vector<vector<Ciphertext>> &res,
                                  vector<Ciphertext> ct,
                                  int conv_block);

    void conv2_fhe_eval(vector<Ciphertext> &res,
                        vector<Ciphertext> &input,
                        const vector<vector<vector<Plaintext>>> &conv_w_p,
                        const Plaintext &conv_mask,
                        int out_channels);

    void avg_pool2_fhe_eval(vector<Ciphertext> &ct);

    void conv3_fhe_eval(vector<Ciphertext> &res,
                        vector<Ciphertext> &input,
                        const vector<vector<vector<Plaintext>>> &conv_w_p,
                        const Plaintext &conv_mask,
                        int out_channels);

    void conv4_fhe_eval(vector<Ciphertext> &res,
                        vector<Ciphertext> &input,
                        const vector<vector<vector<Plaintext>>> &conv_w_p,
                        const Plaintext &conv_mask,
                        int out_channels);

    void adaptive_avg_pool_fhe_eval(vector<Ciphertext> &ct);


    void dense_fhe_eval(vector<Ciphertext> &res,
                        vector<Ciphertext> &input,
                        vector<vector<vector<Plaintext>>> &dense_w_p,
                        vector<Plaintext> &dense_b_p,
                        int in_features);

    void model_fhe_inference(vector<Ciphertext> &res,
                             const vector<Ciphertext> &input,
                             EncodedHENetModel &model);

};

#endif
