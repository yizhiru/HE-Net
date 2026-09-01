# HE-Net

基于 CKKS 同态加密的 CIFAR-10 卷积神经网络密态推理实现。

本仓库从 [K-miran/HEAR](https://github.com/K-miran/HEAR) fork，并改写为图像分类场景。HEAR 的论文为 [*Secure Human Action Recognition by Encrypted Neural Network Inference*](https://arxiv.org/abs/2104.09164)，使用 RNS-CKKS 在密文上评估 CNN。HE-Net 沿用其同态卷积（交错编码、giant 旋转、多项式激活）思路，将任务从人体动作识别换成 CIFAR-10 分类。

上游 HEAR 以 MIT License 发布，详见 [HEAR LICENSE](https://github.com/K-miran/HEAR/blob/main/LICENSE)。

## 这个项目做什么

客户端把图片加密后交给服务端，服务端用 [Microsoft SEAL](https://github.com/microsoft/SEAL) 的 CKKS 方案直接在密文上做卷积、BN、激活和池化，再把密文分数返回给客户端解密。CKKS 只能高效做加法和乘法，因此网络使用二次多项式激活 \(\alpha x^2 + \beta x + c\)，而不是 ReLU。

当前密态推理入口会跑完 B1（conv1 + BN/act + pool）和 B2（交错卷积 + BN/act + pool）。B3 / B4 / 全连接的编码与求值代码仍保留在源码中，但默认路径尚未接回 `model_fhe_inference`。

## 目录结构

```
HE-Net/
├── CMakeLists.txt
├── include/
│   ├── cifar/                 # CIFAR-10 二进制读取
│   └── henet/                 # 公共头文件
├── src/
│   ├── encryptor.cpp          # 样本加密与权重明文编码
│   ├── evaluator.cpp          # 密态卷积 / 激活 / 池化
│   ├── helper.cpp             # 权重加载、BN+激活融合
│   ├── henet_seal.cpp         # CKKS 密态推理（CNN-128）
│   ├── henet_plain.cpp        # CryptoNets 明文推理入口
│   ├── cryptonets.cpp         # CryptoNets 算子
│   └── thread_pool.cpp
├── tests/                     # 单元测试与固定样例
├── python/                    # PyTorch 训练与权重导出
└── data/                      # 数据集（不入库）
```

## 依赖

- CMake >= 3.16
- C++17 编译器
- [Microsoft SEAL](https://github.com/microsoft/SEAL) 4.1（仅 `henet-seal` 需要；未安装时 CMake 会自动拉取 v4.1.2）
- OpenMP（可选）
- Python 3，以及 `python/requirements.txt` 中的包（训练用）

macOS 可用 Homebrew 安装 SEAL：

```bash
brew install seal
```

## 准备数据

C++ 推理读取官方 **binary** 格式：

```bash
mkdir -p data/cifar-10
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar -xz -C data/cifar-10
```

目录中应出现 `data/cifar-10/cifar-10-batches-bin/test_batch.bin`。

PyTorch 训练会通过 torchvision 自动下载 Python 版本到 `data/cifar-10/`。

## 训练并导出权重

```bash
pip install -r python/requirements.txt
python python/train.py --epochs 50 --batch-size 128
```

最佳 checkpoint 和导出的 `*.txt` 权重写到 `python/checkpoints/CIFAR_10/fp32/`。明文评估：

```bash
python python/test.py
```

## 编译

```bash
cmake -S . -B build
cmake --build build -j
```

macOS 上若出现 `'cstdlib' file not found` 或 `'thread' file not found`，删掉 `build/` 后重配一次。这通常是 Command Line Tools 自带的 libc++ 头文件不完整；CMake 会自动改用 macOS SDK 里的完整头文件。conda 环境有时也会干扰系统头文件，可先 `conda deactivate` 再编译。

## 密态推理

```bash
./build/henet-seal \
  --data data/cifar-10/cifar-10-batches-bin \
  --model python/checkpoints/CIFAR_10/fp32 \
  --samples 1 \
  --threads 8
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--data` | CIFAR-10 `cifar-10-batches-bin` 目录 |
| `--model` | 训练导出的 txt 权重目录 |
| `--samples` | 测试样本数；`0` 表示全部 |
| `--threads` | 线程数，默认取硬件并发 |

未指定路径时，CMake 会把工程内的默认 `data/` 与 `python/checkpoints/` 编译进二进制。

## 明文基线 `henet-plain`（另一套模型）

`src/henet_plain.cpp` **不是** CNN-128 的明文对照，而是早期 CryptoNets 风格的独立小网络，不使用 SEAL。

- 输入：CIFAR-10 RGB 三通道取平均，变成 1×32×32 灰度
- 结构：`Conv(1→8, 3×3) → x² → AvgPool(4) → Flatten(512) → Dense(128) → x² → Dense(10) → x²`
- 权重文件：`conv.weight.txt`、`conv.bias.txt`、`fc1.weight.txt`、`fc1.bias.txt`、`fc2.weight.txt`、`fc2.bias.txt`

不能加载 `python/train.py` 导出的 `conv1` / `bn` / `act` / `linear` 权重。

```bash
./build/henet-plain \
  --data data/cifar-10/cifar-10-batches-bin \
  --model <cryptonets-weight-dir> \
  --samples 0
```

`--samples 0` 表示跑完整测试集。只编该明文程序、不拉 SEAL：

```bash
cmake -S . -B build -DHENET_BUILD_SEAL=OFF
cmake --build build -j --target henet-plain
```

## 测试

C++ 单测覆盖 helper（reshape / BN 融合 / 旋转）、CryptoNets 算子、命令行解析和权重文件读取：

```bash
cmake -S . -B build
cmake --build build -j --target henet-tests
ctest --test-dir build --output-on-failure
```

Python 激活与网络输出形状：

```bash
python tests/test_model.py
```

## 网络结构（训练用 CNN）

输入为 \(3 \times 32 \times 32\) 的 CIFAR-10 图像，默认第一层 128 通道：

1. Conv 3×3 → BN → 多项式激活 → AvgPool 2×2
2. Conv 3×3 → BN → 多项式激活 → AvgPool 2×2
3. Conv 3×3 → BN → 多项式激活
4. Conv 3×3 → BN → 多项式激活 → AdaptiveAvgPool → Linear(10)

卷积层、BN 和激活在导出时融合成二次多项式系数，供密态求值使用。

## 致谢

- 同态卷积与密文打包来自 [HEAR](https://github.com/K-miran/HEAR)（Kim 等）
- CKKS 实现来自 [Microsoft SEAL](https://github.com/microsoft/SEAL)
- CIFAR-10 读取来自 Baptiste Wicht 的 MIT 许可头文件
