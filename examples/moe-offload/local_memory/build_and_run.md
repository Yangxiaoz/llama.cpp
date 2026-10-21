# 编译、运行与测试

本文件是从 [feature.md](../feature.md) 拆出来的工程操作手册 (编译、参数、测试、标定、常见问题). 特性本身的设计见 feature.md.


开发分两个阶段:
1. 当前阶段: 笔记本纯 CPU 构建, 硬件拓扑是 DDR 内存 + SSD. 独显 (4060) 属于 "CPU + 独立 GPU" 结构, 不在本特性的讨论范围内.
2. 之后的阶段: Jetson 统一内存 (CUDA 构建, 设备类型是 IGPU), 见第 6 节.

## 1. 编译

```bash
# 日常调试 / 查正确性 (Debug, -O0)
cmake --preset cpu-debug
cmake --build build-cpu-debug --target llama-moe-offload -j

# 性能测试 (Release)
cmake --preset cpu-release
cmake --build build-cpu-release --target llama-moe-offload -j

# Jetson 路径的编译检查
cmake --build build-cuda-moe --target llama-moe-offload -j
```

| 构建 | 用途 |
|---|---|
| `build-cpu-debug` | 查正确性. `-O0`, 计算慢很多倍, **不要**用它比较性能 |
| `build-cpu-release` | 所有时间 / IO 对比都用它 |
| `build-cuda-moe` | Jetson 路径的编译检查, 还没在设备上运行过 |

- 改源码后**必须重新编译**再测: 旧二进制曾造成过一次误报.
- `ENABLE_CUSTOM_MOE=ON` -> `add_compile_definitions(CUSTOM_MOE)`, 全局生效, 包括 ggml (在 `add_subdirectory(ggml)` 之前定义).
- 依赖 liburing (`sudo apt install liburing-dev`).
- 示例没有 `-t` 参数, 使用默认线程数 (`GGML_DEFAULT_N_THREADS` = 4).

## 2. 运行参数

示例参数只在 `examples/moe-offload` 里解析, 对应 `llama_model_params` 的同名字段:

| 参数 | 字段 | 含义 |
|---|---|---|
| `--moe-mode on-demand\|cross-layer\|io-aware` | `moe_mode` | 运行模式, 示例默认 `io-aware` (库默认 `on-demand`) |
| `--moe-pool-ratio F` | `moe_pool_ratio` | 专家池占 "设备总内存" 的比例, 0 < F < 1, 示例默认 0.46 (库默认 0.3). CPU 构建按系统内存总量计算, 例如 31 GB x 0.15 约为 4.7 GB. 池必须至少能放下一整层专家, 否则加载时报错 |
| `--moe-io-queue N` | `moe_io_queue` | 只对 `io-aware` 有效: prefill 只在 "正在读盘的专家数 < N" 时才预测新的一层, 见 5.4. 默认 0 = 每层专家数 (DeepSeek-Lite 64, Qwen3 128) |
| `--moe-decode-prefetch N` | `moe_decode_prefetch` | 只对 `io-aware` 有效: decode 每层最多预取几个缺失专家, 默认 2, 0 = 下一层所有缺失专家. `cross-layer` 固定为全部, 不看这个参数 |
| `--moe-arpe-affinity FILE` | `moe_arpe_affinity` | 只对 `io-aware` 有效: ARPE 离线亲和先验 A0 的路由计数文件 (由 `--moe-arpe-calib` 生成), 见 5.7. 不给 = 均匀先验 |
| `--moe-arpe-calib FILE` | `moe_arpe_calib` | 只对 `io-aware` 有效: 模型释放时把本次运行的相邻层路由计数累加进 FILE (标定用) |

- 示例在 CUSTOM_MOE 下: `load_mode = LLAMA_LOAD_MODE_NONE` (不用 mmap); 空 prompt 直接报错.
- 运行时约束:
  - 一旦编译进 CUSTOM_MOE, **每次加载模型都走 offload 路径**, 没有运行时开关. 不是 deepseek2 / qwen3moe 的模型会在构造函数中触发断言.
  - 只支持单文件 GGUF.

## 3. 测试准备 (把这段复制到同一个 shell)

```bash
cd ~/workspace/github/llama.cpp
R=build-cpu-release/bin/llama-moe-offload
D=build-cpu-debug/bin/llama-moe-offload
DS=/home/shawn/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
QW=/home/shawn/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
P='James creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?'

# run <命令...>: 清页缓存, 跑一次, 只打印关键行
#   标准输出 -> t.out (打印的 prompt + 生成的文本), 标准错误 -> t.err (日志和统计)
run() {
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    "$@" > t.out 2> t.err
    echo "exit=$?"
    grep -E "custom_moe: mode|custom_moe stats|custom_moe ARPE|no room for a whole layer|GGML_ASSERT|error|prompt eval time|decoded" t.err
}
```

- `run` 每次先清 page cache (`sudo tee`, 会要一次密码). 不清的话模型文件留在 page cache 里, 读盘其实是内存拷贝, IO 对比完全没意义 (实测 decode 能到 10 GB/s, 那是内存带宽).
- prompt 用 ASCII 单引号: 双引号会被 shell 展开 `$2000`, 中文引号不会被当成引号.
- 每次运行都覆盖 `t.out` / `t.err`, 需要对比时及时 `cp`.

## 4. 三种模式的测试命令

```bash
# 基线: prefill 按层加载/释放, decode 逐个专家, 不预测
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode on-demand "$P"

# 跨层预测: 第 i 层预测第 i+1 层, decode 预取下一层全部缺失专家
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode cross-layer "$P"

# IO 最大化: 前沿层预测 + ARPE decode 淘汰
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode io-aware "$P"

# io-aware: 改 IO 队列深度 (0 = 每层专家数)
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-io-queue 32 "$P"

# io-aware: decode 每层最多预取 2 个专家 (默认); 0 = 下一层全部都预取
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-decode-prefetch 2 "$P"

# Qwen3-30B: 每层 128 个专家, 池子用 0.2
run $R -m $QW -n 96 --moe-pool-ratio 0.2 --moe-mode io-aware "$P"
```

- `--moe-pool-ratio 0.15` 在 31 GB 内存的机器上约 4.7 GB 池, DeepSeek-Lite 预加载 12 层 (日志第一行会打印 `n_slot of pool` 和预加载层数).
- 三种模式用同一个 prompt 时, 生成结果应当完全一致 (见第 5 节).

## 5. 正确性对比

同一 prompt、同一构建、贪婪采样: 三种模式的输出必须逐字一致.

```bash
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode on-demand "$P";   cp t.out ref.out
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode cross-layer "$P"; diff -q ref.out t.out
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode io-aware "$P";    diff -q ref.out t.out
```

- `diff` 没有输出 = 一致. 有输出说明加载或淘汰改坏了结果, 把 `t.out` 和 `ref.out` 的差异贴出来.
- 两次运行的 prompt 必须完全相同 (包括 shell 引号方式), 否则比的是不同的 prompt.

## 6. 性能对比

看 `run` 打印的两类行:

```
llama_perf_context_print: prompt eval time = ... ms / ... tokens        <- prefill 总时间
main: decoded 96 tokens in ... s, speed: ... t/s                        <- decode 速度
custom_moe stats [prefill]: ... hit, ... wait, ... miss | prefetch N (used M) | read ... MiB | ensure ... ms, predict ... ms
custom_moe stats [decode]:  ...
```

- `ensure` = 计算线程等这一层专家的总时间, 也就是没有被计算掩盖的 IO. `predict` = 预测器自身的时间 (同样在关键路径上).
- `prompt eval time - ensure - predict` 约等于纯计算时间. 各模式应该接近; 如果 io-aware 明显更大, 说明后台 IO 拖慢了计算.
- `miss` = 按需读的专家数; `prefetch (used)` = 预取数, 括号内是其中被用到的数. 命中率 = used / prefetch.
- 单次测量有 1 秒左右的波动, 建议每条跑 2-3 次.
- 换模型或换池大小时预加载层数会变, 对比必须在同一个 `--moe-pool-ratio` 下做.



## 7. 长 prompt / 多 ubatch

超过 `n_ubatch` (默认 512, 并且不超过 `n_batch`) 的 prompt 会被切成多个 ubatch, 每个 ubatch 从第 0 层完整跑一遍:

```bash
mkdir -p /tmp/moe_test
python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 80).strip())" > /tmp/moe_test/p_long.txt

run $R -m $DS -n 48 --moe-pool-ratio 0.15 --moe-mode io-aware "$(cat /tmp/moe_test/p_long.txt)"
```

- 上面这条约 800 token, 会产生 2 个 ubatch.
- 长度不是 512 的整数倍时, 最后剩下的 1 个 token 会单独成为一个 ubatch 并被当成 decode (feature.md 第 8 节), 输出仍然正确.
- 每个新 ubatch 的第 0 层是冷启动 (没有跨 ubatch 预测), 那一层的 `ensure` 会明显变大, 属于已知情况.

## 8. ARPE 标定

ARPE 的亲和先验 A0 来自 "相邻两层路由" 的统计数据. 不标定也能跑 (A0 均匀), 标定只是让它从第一个 token 就有信息; 指南的结论是去掉离线先验平均少约 0.3 个百分点.

```bash
# 1) 用 3-5 条和评测无关的 prompt 标定; 每条都会把本次运行的计数累加进同一个文件
run $R -m $DS -n 64 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-arpe-calib /tmp/moe_test/ds.arpe 'Write a Python function that checks whether a string is a palindrome, ignoring spaces, punctuation and letter case, and explain how it works step by step.'
run $R -m $DS -n 64 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-arpe-calib /tmp/moe_test/ds.arpe 'Explain the difference between TCP and UDP, with one example use case for each. Keep the answer under 150 words.'
run $R -m $DS -n 64 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-arpe-calib /tmp/moe_test/ds.arpe 'Summarize the plot of Romeo and Juliet in five sentences.'

# 2) 用标定结果跑评测 prompt
run $R -m $DS -n 96 --moe-pool-ratio 0.15 --moe-mode io-aware --moe-arpe-affinity /tmp/moe_test/ds.arpe "$P"
```

- 标定文件在**模型释放时**写出 (正常退出). 每次会打印 `custom_moe ARPE: calibration counts written to ... (N pairs from this run, M in total)`.
- 不要用评测的 prompt 去标定 (数据泄漏); 也不要只用一条 (样本太少).
- 文件是模型相关的: 头里有 MoE 层数 / 专家数 / top-k. 不匹配时 `--moe-arpe-affinity` 打印 warning 并退回均匀先验, `--moe-arpe-calib` 不写入.
- 大小: DeepSeek-Lite 约 0.8 MB, Qwen3-30B 约 6.2 MB. 删掉文件就等于清空标定.
- prefill 和 decode 的 token 都会计入统计; 标定时的 `-n` 决定 decode 贡献多少.

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `error: the prompt is empty` | 没有传 prompt, 或 prompt 被 shell 吃掉; 用单引号 |
| `moe pool too small` | `--moe-pool-ratio` 太小, 池放不下一整层 |
| `no room for a whole layer, layer N is loaded expert by expert` | on-demand 的池被 decode 打散, 退化为一层内逐个专家加载; 输出正确, 只是性能略差 |
| `GGML_ASSERT ...` | 先对照 feature.md 第 7 节的不变量, 把 `t.err` 末尾贴出来 |
| 不同模式结果不一致 | 用同一构建、同一 prompt 复现; 确认没有用旧二进制 |

## 10. Jetson 板端编译

在 Orin Nano 8GB 上直接编译 (走的不是交叉编译, 设计见 feature.md 第 6 节):

```bash
sudo apt install ccache liburing-dev        # 只需一次
cmake --preset jetson
cmake --build --preset jetson               # 只编 llama-moe-offload, 4 线程
```

- preset `jetson` 继承 `linux-base`, 内容: `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `GGML_CUDA_GRAPHS=OFF`, `CMAKE_BUILD_TYPE=Release`, `LLAMA_BUILD_TESTS/TOOLS/SERVER=OFF`, `LLAMA_BUILD_EXAMPLES=ON`, `GGML_CCACHE=ON`, build preset 的 `jobs = 4`, 只构建 `llama-moe-offload`.
- `ENABLE_CUSTOM_MOE=ON` 和 `LLAMA_CURL=OFF` 继承自 `linux-base`.
- 板子上的 CMake 太老时, `cmake --preset jetson` 会报 `Unrecognized "version" field`: preset 文件是 version 4, 需要 CMake >= 3.23 (JetPack 6 自带的 Ubuntu 22.04 是 3.22, 不够; JetPack 5 的 3.16 更不行). 两个办法:
  1. 不用 preset, 直接传参数 (和 preset 等价):
     ```bash
     cmake -B build-jetson -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
       -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87 \
       -DGGML_CUDA_GRAPHS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
       -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DGGML_CCACHE=ON \
       -DENABLE_CUSTOM_MOE=ON -DLLAMA_CURL=OFF
     cmake --build build-jetson --target llama-moe-offload -j4
     ```
  2. 在板子上装一个新 CMake (项目本身只要求 3.14, 升级只为了用 preset):
     ```bash
     wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-aarch64.tar.gz
     tar xf cmake-3.28.3-linux-aarch64.tar.gz -C ~/.local/
     export PATH=~/.local/cmake-3.28.3-linux-aarch64/bin:$PATH
     cmake --version
     ```
- CUDA 编译器默认 `/usr/local/cuda/bin/nvcc` (JetPack 的标准位置). 不在那里时加 `-DCMAKE_CUDA_COMPILER=<路径>`.
- 8 GB 内存不要提高并发: 6 核全开容易 OOM, 在 swap 上抖动会更慢.
- 先验证 offload 逻辑时可以编纯 CPU 版:
  `cmake -B build-jetson-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DENABLE_CUSTOM_MOE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-jetson-cpu --target llama-moe-offload -j4`.
- 模型必须放在 NVMe SSD 上; eMMC / SD 卡的随机读会成为瓶颈, IO 数据没有参考价值.
