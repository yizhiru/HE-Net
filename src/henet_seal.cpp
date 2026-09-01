/**
 * @file henet_seal.cpp
 * CNN-128 的 CKKS 密态推理入口：加密样本、编码权重、密文求值、解密类别。
 */

#include "cifar/cifar10_reader.hpp"
#include "henet/cli.h"
#include "henet/encryptor.h"
#include "henet/evaluator.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif


inline std::ostream &operator<<(std::ostream &stream, seal::parms_id_type parms_id) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    stream << std::hex << std::setfill('0') << std::setw(16) << parms_id[0] << " " << std::setw(16) << parms_id[1]
           << " " << std::setw(16) << parms_id[2] << " " << std::setw(16) << parms_id[3] << " ";

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);

    return stream;
}


/** 从密文 logits 中取出 10 类分数并返回 argmax。 */
int decrypt_result(Decryptor &decryptor,
                   CKKSEncoder &encoder,
                   vector<Ciphertext> pred_ct) {
    vec_t pred_scores(pred_ct.size() * 5);
    for (int i = 0, j = 0; i < pred_ct.size(); ++i) {
        Plaintext plain;
        decryptor.decrypt(pred_ct[i], plain);
        vec_t v;
        encoder.decode(plain, v);
        for (int k = 0; k < 5; k++) {
            pred_scores[j] = v[k * SIZE_OF_BLOCK];
            j++;
        }
    }

    int pred_label = arg_max(pred_scores);
    return pred_label;
}


/** 对 num_samples 张已归一化图像做完整密态推理并打印准确率。 */
void model_fhe_eval(const dmat &x_test_vec,
                    const vector<unsigned char> y_true,
                    const ModelWeights &mwp,
                    int num_samples) {
    vector<int> bit_sizes_vec = {
            // The last modulus
            LOG_Q0,
            // dense
            LOG_QC,
            // act4-conv4
//            LOG_QC, LOG_Q, LOG_QC,
            // pre-processing
//            LOG_QC_SMALL,
            // act3-conv3
            LOG_QC, LOG_Q, LOG_QC,
            // pre-processing
            LOG_QC_SMALL,
            // act2-conv2
            LOG_QC, LOG_Q, LOG_QC,
            // pre-processing
            LOG_QC_SMALL,
            // act1-conv1
            LOG_QC, LOG_Q, LOG_QC,
            LOG_P0
    };

    EncryptionParameters params(scheme_type::ckks);
    const size_t n = (1 << 14);
    params.set_poly_modulus_degree(n);
    params.set_coeff_modulus(CoeffModulus::Create(n, bit_sizes_vec));

    auto context = SEALContext(params);

    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    Encryptor encryptor(context, pk);
    Decryptor decryptor(context, sk);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    vector<int> steps_all;
    steps_all.insert(steps_all.end(), STEPS_GIANT.begin(), STEPS_GIANT.end());
    steps_all.insert(steps_all.end(), STEPS_CONV[0].begin(), STEPS_CONV[0].end());
    steps_all.insert(steps_all.end(), STEPS_CONV[1].begin(), STEPS_CONV[1].end());
    steps_all.insert(steps_all.end(), STEPS_CONV[2].begin(), STEPS_CONV[2].end());
    steps_all.insert(steps_all.end(), STEPS_POOL.begin(), STEPS_POOL.end());
    steps_all.insert(steps_all.end(), STEPS_INTERLACE.begin(), STEPS_INTERLACE.end());

    GaloisKeys gal_keys;
    keygen.create_galois_keys(steps_all, gal_keys);

    auto context_data = context.key_context_data();
    cout << "----> Level (chain index): " << context_data->chain_index();
    cout << " ...... key_context_data()" << endl;
    cout << "      parms_id: " << context_data->parms_id() << endl;
    cout << "      coeff_modulus primes: ";
    cout << hex;
    for (const auto &prime: context_data->parms().coeff_modulus()) {
        cout << prime.value() << " ";
    }
    cout << dec << endl;
    cout << "\\" << endl;
    cout << " \\-->";

    /*
    Next iterate over the remaining (data) levels.
    */
    context_data = context.first_context_data();
    while (context_data) {
        cout << " Level (chain index): " << context_data->chain_index();
        if (context_data->parms_id() == context.first_parms_id()) {
            cout << " ...... first_context_data()" << endl;
        } else if (context_data->parms_id() == context.last_parms_id()) {
            cout << " ...... last_context_data()" << endl;
        } else {
            cout << endl;
        }
        cout << "      parms_id: " << context_data->parms_id() << endl;
        cout << "      coeff_modulus primes: ";
        cout << hex;
        for (const auto &prime: context_data->parms().coeff_modulus()) {
            cout << prime.value() << " ";
        }
        cout << dec << endl;
        cout << "\\" << endl;
        cout << " \\-->";

        /*
        Step forward in the chain.
        */
        context_data = context_data->next_context_data();
    }
    cout << " End of chain reached" << endl << endl;

    HENetEncryptor heNetEncryptor(context, encryptor, encoder);
    HENetEvaluator heNetEvaluator(evaluator, relin_keys, gal_keys);


    // 加密数据集
    vector<vector<Ciphertext>> x_ct_vec;
    x_ct_vec.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        vector<Ciphertext> ct;
        heNetEncryptor.encrypt_sample(ct, x_test_vec[i]);
        x_ct_vec[i].insert(x_ct_vec[i].end(), ct.begin(), ct.end());
    }

    // 模型明文编码
    EncodedHENetModel model;
    heNetEncryptor.encode_model(model, mwp);

    // 密态模型推理
    auto start_time = std::chrono::high_resolution_clock::now();
    vector<int> y_pred(num_samples);
    MT_EXEC_RANGE(num_samples, first, last)
                    for (int i = first; i < last; ++i) {
                        vector<Ciphertext> res;
                        heNetEvaluator.model_fhe_inference(res, x_ct_vec[i], model);
                        int pred = decrypt_result(decryptor, encoder, res);
                        y_pred[i] = pred;
                    }
    MT_EXEC_RANGE_END

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // 输出耗时
    std::cout << "cost time：" << duration.count() / 1000 << " (s)" << std::endl;

    double num_correct = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        if (y_pred[i] == y_true[i]) {
            num_correct += 1;
        }
    }
    double acc = num_correct / num_samples;
    cout << "accuracy: " << acc << endl;

}


int main(int argc, char **argv) {
    henet::RunConfig cfg;
    if (!henet::parse_run_config(argc, argv, cfg)) {
        return 1;
    }
    if (cfg.help) {
        henet::print_usage(argv[0]);
        return 0;
    }

    Thread::initThreadPool(static_cast<size_t>(cfg.num_threads));
#ifdef _OPENMP
    omp_set_num_threads(cfg.num_threads);
#endif
    cout << "threads=" << cfg.num_threads << endl;
    cout << "data=" << cfg.data_dir << endl;
    cout << "model=" << cfg.model_dir << endl;

    ModelWeights mwp = {
            vector<ften>(4),
            vector<dmat>(4),
            dmat(10, vec_t(1024)),
            vec_t()};
    load_model_weights(mwp, cfg.model_dir);

    auto [test_X, test_Y] = load_cifar_dataset(cfg.data_dir);
    size_t available = test_Y.size();
    size_t num_samples = available;
    if (cfg.num_samples > 0) {
        num_samples = std::min(available, static_cast<size_t>(cfg.num_samples));
    }
    cout << "samples=" << num_samples << endl;

    dmat x_test_vec(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        const size_t channels = 3, height = 32, width = 32;
        auto rgb_img = test_X[i];
        vec_t x_test(channels * height * width);
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    auto index = c * height * width + h * width + w;
                    x_test[index] = static_cast<double>(rgb_img[index]) / 255.0;
                }
            }
        }
        x_test_vec[i].insert(x_test_vec[i].end(), x_test.begin(), x_test.end());
    }

    model_fhe_eval(x_test_vec, test_Y, mwp, static_cast<int>(num_samples));
    return 0;
}

