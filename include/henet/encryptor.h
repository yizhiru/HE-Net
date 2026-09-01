#ifndef HE_NET_HENET_ENCRYPTOR_H
#define HE_NET_HENET_ENCRYPTOR_H

#include <vector>
#include <seal/seal.h>
#include "henet/helper.h"
#include "henet/thread_pool.h"
#include "henet/utils.h"

using namespace std;
using namespace seal;

/** CKKS 明文槽中的编码后 CNN-128 权重。 */
struct EncodedHENetModel {
    // B1
    vector<vector<Plaintext>> conv1_w_p;
    vector<vector<Plaintext>> act1_poly;

    // B2
    vector<vector<vector<Plaintext>>> conv2_w_p;
    Plaintext mask_conv2;
    vector<vector<Plaintext>> act2_poly;

    // B3
    vector<vector<vector<Plaintext>>> conv3_w_p;
    Plaintext mask_conv3;
    vector<vector<Plaintext>> act3_poly;

    // B4
    vector<vector<vector<Plaintext>>> conv4_w_p;
    Plaintext mask_conv4;
    vector<vector<Plaintext>> act4_poly;

    // Dense
    vector<vector<vector<Plaintext>>> dense_w_p;
    vector<Plaintext> dense_b_p;
};

class HENetEncryptor {
private:
    SEALContext &context;
    Encryptor &encryptor;
    CKKSEncoder &encoder;

    inline parms_id_type get_param_id(int level_gap) {
        auto context_data = context.first_context_data();
        for (int i = 0; i < level_gap && context_data; i++) {
            context_data = context_data->next_context_data();
        }
        return context_data->parms_id();
    }

public:
    HENetEncryptor(SEALContext &context, Encryptor &encryptor, CKKSEncoder &encoder) :
            context(context), encryptor(encryptor), encoder(encoder) {};

    void encrypt_sample(vector<Ciphertext> &x_cipher,
                        const vec_t &x_test);

    void encode_conv1_w(vector<vector<Plaintext>> &w_plain,
                        const vector<ften> &conv_w,
                        int out_channels);


    void encode_act(vector<vector<Plaintext>> &act_plain,
                    const dmat &act_poly,
                    int out_channels,
                    int dist);

    void encode_conv_interlace(vector<vector<vector<Plaintext>>> &conv_w_p,
                               ften conv_w,
                               int conv_block,
                               int in_channels,
                               int out_channels,
                               int level_gap);

    void encode_mask_conv2(Plaintext &mask);

    void encode_mask_conv3(Plaintext &mask, int level_gap);

    void encode_dense_w(vector<vector<vector<Plaintext>>> &dense_w_p,
                        dmat dense_w,
                        int in_features,
                        int level_gap);

    void encode_dense_b(vector<Plaintext> &dense_b_p,
                        const vec_t dense_b,
                        int level_gap);

    void encode_model(EncodedHENetModel &model,
                      const ModelWeights &mwp);

};

#endif
