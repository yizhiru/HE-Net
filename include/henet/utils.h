#ifndef HE_NET_UTILS_H
#define HE_NET_UTILS_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "cifar/cifar10_reader.hpp"

using namespace std;

typedef vector<double> vec_t;

/** 按逗号读取一维浮点向量。 */
inline void read_vec_from_file(vec_t &vec, const string &FileName) {
    ifstream source(FileName);
    string line;
    if (!source.is_open()) {
        throw invalid_argument("Error: cannot read file: " + FileName);
    }
    while (getline(source, line, ',')) {
        if (line.empty()) {
            continue;
        }
        vec.push_back(stod(line));
    }
    cout << "Read file: " << FileName << ", size: " << vec.size() << endl;
}

/** 读取文件中的最后一个浮点值。 */
inline double read_val_from_file(const string &FileName) {
    ifstream openFile(FileName.data());
    string line;
    double res = 0.0;
    if (!openFile.is_open()) {
        throw invalid_argument("Error: cannot read file: " + FileName);
    }
    while (getline(openFile, line, ',')) {
        if (!line.empty()) {
            res = stod(line);
        }
    }
    return res;
}

inline auto load_cifar_dataset(const string &path) {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);
    cout << "Loaded CIFAR test image size: " << dataset.test_images.size() << endl;
    return make_tuple(dataset.test_images, dataset.test_labels);
}

#endif
