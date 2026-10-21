# CUSTOM_MOE: MoE expert offload to disk

本文记录 `CUSTOM_MOE` 特性的实现细节与代码位置, 用于后续排查问题. 改动基本都用 `#ifdef CUSTOM_MOE` 包裹, 关闭宏即恢复上游代码 (例外: CMake 中的 liburing 和 `custom-moe-offload.cpp` 没有宏保护).

## 1. 目标与场景

- 场景: 内存 (CPU DDR, 或 Jetson 这类统一内存) 装不下整个 MoE 模型. 非专家权重常驻内存, **专家权重 (ffn_up/gate/down_exps) 留在 SSD 上的 GGUF 文件里**, 运行时按需读入一个固定大小的 "专家池" (expert pool).
- 与上游的 `llama-moe-cache` / `-ot` / `--cpu-moe` 等机制无关. 上游机制解决的是 "显存放不下 -> 放到 DDR", 本特性解决的是 "DDR 放不下 -> 放到磁盘".
- 读盘使用 `io_uring` (liburing). 运行时一律以**单个专家**为单位加载.
- 三种运行模式 (第 5 节): `on-demand` (基线), `cross-layer` (跨层预测), `io-aware` (IO 最大化, 本工程后续优化的核心).
- 目标使用场景: 单用户. prompt 可以被拆成多个 ubatch, 也可以多轮对话. 不考虑 MTP、`llama-moe-cache`、合并的 `gate_up_exps` 张量等特殊情况.

## 2. 编译与运行

开发分两个阶段:
1. 当前阶段: 笔记本纯 CPU 构建, 硬件拓扑是 DDR 内存 + SSD. 独显 (4060) 属于 "CPU + 独立 GPU" 结构, 不在本特性的讨论范围内.
2. 之后的阶段: Jetson 统一内存 (CUDA 构建, 设备类型是 IGPU), 见第 6 节.

### 2.1 编译

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

### 2.2 运行参数

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

### 2.3 测试准备 (把这段复制到同一个 shell)

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

### 2.4 三种模式的测试命令

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
- 三种模式用同一个 prompt 时, 生成结果应当完全一致 (见 2.5).


### 2.5 性能对比

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



### 2.8 ARPE 标定

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

### 2.9 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `error: the prompt is empty` | 没有传 prompt, 或 prompt 被 shell 吃掉; 用单引号 |
| `moe pool too small` | `--moe-pool-ratio` 太小, 池放不下一整层 |
| `no room for a whole layer, layer N is loaded expert by expert` | on-demand 的池被 decode 打散, 退化为一层内逐个专家加载; 输出正确, 只是性能略差 |
| `GGML_ASSERT ...` | 先对照第 7 节的不变量, 把 `t.err` 末尾贴出来 |
| 不同模式结果不一致 | 用同一构建、同一 prompt 复现; 确认没有用旧二进制 |

### 2.10 Jetson 板端编译

在 Orin Nano 8GB 上直接编译 (走的不是交叉编译, 见第 6 节):

```bash
sudo apt install ccache liburing-dev        # 只需一次
cmake --preset jetson
cmake --build --preset jetson               # 只编 llama-moe-offload, 4 线程

# 运行: 专家池在 managed 设备内存里, 必须设置这个环境变量
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
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

## 3. 代码地图

| 位置 | 作用 |
|---|---|
| [CMakeLists.txt:116-122](../../CMakeLists.txt#L116-L122) | `ENABLE_CUSTOM_MOE` 选项 -> 宏 `CUSTOM_MOE` |
| [CMakePresets.json](../../CMakePresets.json) | 新增 `linux-base` / `cuda` / `jetson` / `cpu-debug` / `cpu-release` preset |
| [src/CMakeLists.txt:7-15](../../src/CMakeLists.txt#L7-L15), [79-90](../../src/CMakeLists.txt#L79-L90) | 查找并链接 liburing |
| [src/CMakeLists.txt:53-54](../../src/CMakeLists.txt#L53-L54) | 把 `custom-moe-offload.cpp`、`custom-moe-arpe.cpp` 加入 libllama |
| [include/llama.h:320-327](../../include/llama.h#L320-L327) | `enum llama_moe_mode` |
| [include/llama.h:358-364](../../include/llama.h#L358-L364) | `llama_model_params` 新增 `moe_mode` / `moe_pool_ratio` / `moe_io_queue` / `moe_arpe_affinity` / `moe_arpe_calib` |
| [src/llama-model.cpp:3364-3370](../../src/llama-model.cpp#L3364-L3370) | 上述参数的默认值 (on-demand / 0.3 / 0 / NULL / NULL) |
| [src/llama-model-loader.h:73-76](../../src/llama-model-loader.h#L73-L76) | 新 flag `TENSOR_EXPERT_WEIGHT = 1 << 7`, 成员 `moe_table_ctx` |
| [src/llama-model-loader.cpp:1367-1378](../../src/llama-model-loader.cpp#L1367-L1378) | `create_tensor`: 带 `TENSOR_EXPERT_WEIGHT` 的张量放进 `moe_table_ctx`, 不进 `ctx_map`, 所以**不分配 buffer, 也不会被 `load_all_data` 读入** |
| [src/llama-model.cpp:1657-1662](../../src/llama-model.cpp#L1657-L1662) | `get_layer_buft_list`: 层的 device 取 `devices.front()` (GPU 或 iGPU), buft list 用 `cpu_buft_list`. 纯 CPU 构建不会走到这里 |
| [src/llama-model.cpp:1682-1698](../../src/llama-model.cpp#L1682-L1698) | 创建 `moe_table_ctx` (no_alloc, 每个 MoE 层 3 个张量头) |
| [src/llama-model.cpp:1871-1873](../../src/llama-model.cpp#L1871-L1873) | `model.table_ctx` 接管 `moe_table_ctx` 的所有权 |
| [src/llama-model.h:17-19](../../src/llama-model.h#L17-L19), [772-775](../../src/llama-model.h#L772-L775) | `llama_model::table_ctx`, `llama_model::moe_unified` |
| [src/llama.cpp:372-378](../../src/llama.cpp#L372-L378) | `load_tensors` 之后构造 `custom_moe_unified` 并调用 `prefill_init()` |
| [src/llama-context.cpp:2522-2526](../../src/llama-context.cpp#L2522-L2526) | `graph_max_nodes`: 节点上限 8x -> 10x `n_tensors` |
| [src/llama-graph.h:26-28](../../src/llama-graph.h#L26-L28), [1093-1109](../../src/llama-graph.h#L1093-L1109) | `build_moe_ffn_offload` 声明 |
| [src/llama-graph.cpp:1581](../../src/llama-graph.cpp#L1581) | `build_moe_ffn_offload`: 替代上游 `build_moe_ffn` 的建图函数, 包括预测; io-aware 的动态目标层节点在 [1677](../../src/llama-graph.cpp#L1677) |
| [src/models/deepseek2.cpp:136-143](../../src/models/deepseek2.cpp#L136-L143), [655-680](../../src/models/deepseek2.cpp#L655-L680) | deepseek2: 专家张量加 flag; MoE 分支改调 `build_moe_ffn_offload` |
| [src/models/qwen3moe.cpp:51-60](../../src/models/qwen3moe.cpp#L51-L60), [142-171](../../src/models/qwen3moe.cpp#L142-L171) | qwen3moe: 同上 |
| [ggml/src/ggml-cuda/ggml-cuda.cu:310](../../ggml/src/ggml-cuda/ggml-cuda.cu#L310) | 保持上游的 `integrated = false`; CUSTOM_MOE 不再修改它, 池改用设备内存 (见第 6 节) |
| [src/custom-moe-offload.h](../../src/custom-moe-offload.h) | 全部数据结构 + 部分 inline 实现 |
| [src/custom-moe-offload.cpp](../../src/custom-moe-offload.cpp) | 核心逻辑 + 自定义 ggml 算子 (`id_pos_map`, `custom_predict`, `custom_frontier`, `custom_predict_frontier`) |
| [src/custom-moe-arpe.h](../../src/custom-moe-arpe.h), [.cpp](../../src/custom-moe-arpe.cpp) | ARPE 淘汰策略 `custom_evict_arpe` (io-aware 的 decode), 见 5.7 |
| [examples/moe-offload/moe-offload.cpp](moe-offload.cpp) | 基于 `simple.cpp` 的示例: 解析 `--moe-*` 参数, 关闭 mmap, 拒绝空 prompt |

只有 deepseek2 和 qwen3moe 两个架构接入了本特性.

## 4. 数据结构 (src/custom-moe-offload.h)

### 4.1 `custom_hparams` ([h:81](../../src/custom-moe-offload.h#L81))

| 字段 | 含义 |
|---|---|
| `utilization` | 池大小比例 (= `moe_pool_ratio`) |
| `io_queue` | io-aware 的 IO 队列阈值 Q, 0 在构造时换成 `n_expert` |
| `n_preload_layers` | 池能完整放下的层数 = `min(n_slots / n_expert, n_moe_layer)`, 至少为 1 |
| `reserve_budget` | 预约专家数的上限 = `n_slots - n_expert`, 见 4.4 |

**层下标约定**: 表、池、淘汰策略里的 `il` 都是 "MoE 层下标" = 模型层号 - `n_layer_dense_lead`. DeepSeek-V2-Lite 有 1 个 dense 层, 所以模型第 1 层 = MoE 层 0. Qwen3-MoE 没有 dense 层.

### 4.2 专家池 `custom_expert_pool` ([h:123](../../src/custom-moe-offload.h#L123), 构造函数 [cpp:140](../../src/custom-moe-offload.cpp#L140))

- 三个 3D 张量 `pool_ups` / `pool_gates` / `pool_downs`, 形状 `[ne0, ne1, n_slots]`.
- 一个 "slot" = 一个专家的 up+gate+down. 对 up/gate/down 分别在所有 MoE 层中选**字节数最大**的专家张量作为模板, 池的类型和 `nbyte_slot_*` (`size_t`) 都取自它. 其他层的专家较小时, slot 末尾留空.
- 专家可以放在任意 slot, 同一层的专家**不需要连续**: `mul_mat_id` 用 slot 下标直接在整个池张量上取专家 (见 5.2).
- `n_slots = utilization * props.memory_total / nbyte_group`. `props` 来自 `model.devices.front()` (GPU 或 iGPU); 没有设备 (纯 CPU 构建) 时用 CPU device, 即系统内存. `n_slots < n_expert` 时抛异常: 最坏情况下一层的全部专家都会被用到.
- buffer 类型: 设备的默认 buffer type (`ggml_backend_dev_buffer_type(dev)`). CPU 构建是普通内存; CUDA 构建是**设备内存**, 需要 `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (managed 分配) 才能让 CPU (io_uring) 也写它, 没设时构造直接报错. buffer usage 设为 `WEIGHTS`.
- io-aware 模式下, 同一个 buffer 里还有 `routers_all` (所有 MoE 层的 router 按层拼接, `[n_embd, n_expert * n_moe_layer]`) 和 `router_biases_all` (有 `exp_probs_b` 时), 由 `routers_init` ([cpp:719](../../src/custom-moe-offload.cpp#L719)) 从模型权重复制. 用途见 5.2. 大小: DeepSeek-Lite 约 13 MB, Qwen3-30B 约 48 MB (F32 router).
- `free_slots`: `std::map<pos, len>` 空闲区间表. `pool_alloc(n)` first-fit; `pool_free` 当前没有调用者. 实际的回收方式是: 淘汰一个专家后直接把它的 `pos` 交给新专家用.

参考值 (`--moe-pool-ratio 0.46` 对应约 3.7 GB 池):

| 模型 | MoE 层 x 专家 | 单专家 (up+gate+down, 取最大层) | slot 数 | 可放下整层数 |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite Q4_K_M | 26 x 64, top-6 | 1.62+1.62+3.06 MB (down: Q8_0) | ~590 | 9 |
| Qwen3-30B-A3B Q4_K_M | 48 x 128, top-8 | 0.84+0.84+1.23 MB (down: Q6_K) | ~1270 | 9 |

### 4.3 专家表 `custom_expert_table` ([h:220](../../src/custom-moe-offload.h#L220), `table_init` [cpp:739](../../src/custom-moe-offload.cpp#L739))

- `experts[n_moe_layer][n_expert]`, 每项 `custom_expert_group` ([h:114](../../src/custom-moe-offload.h#L114)):
  - `up/gate/down.offs`: 该专家在 GGUF 文件中的**绝对偏移** = `weight->offs + j * layer_nbyte`, `idx` 是文件下标 (未使用).
  - `state`: `OnDisk` / `Loading` / `InMemory`.
  - `pos`: 在池中的 slot 下标, `OnDisk` 时为 -1.
  - `predicted`: 被预测预约, 所在层还没计算. 预约期间不可淘汰 (见 4.4).
  - `prefetched`: 由预测器读入, 所在层计算时清除. 只用于统计 (5.6).
- `layer_type[il]`: 每层 up/gate/down 的实际 ggml_type. 混合量化时, `get_ups/get_gates/get_downs` ([cpp:676-704](../../src/custom-moe-offload.cpp#L676-L704)) 返回池张量的 view, 并把 view 的 `type` 和 `nb` 改成该层类型.
- `layer_nbyte[il]`: 该层单个专家 up/gate/down 在文件中的精确字节数. 每次读盘只读这么多字节, 并以此检查短读.
- `pending_io[il][id]`: 该专家还没完成的异步读请求数 (0..3). `n_loading`: `Loading` 的专家数 (= IO 队列长度). `n_reserved`: `predicted` 的专家数 (由 `reserve` / `clear_predicted` 维护). `n_resident[il]`: 该层有 slot 的专家数 (`Loading` + `InMemory`), 由 `mark` 维护.
- `map()/unmap()`: `(row, col, type) <-> io_uring user_data`, 即 `(row * n_col + col) * 3 + type + 1` (加 1 是为了避开 0).
- `refresh(async_io, wait)`: 收割异步 ring 上已完成的 CQE, 检查字节数, 专家 3 块都到齐后标记 `InMemory`. `wait = true` 时先睡眠等待至少一个完成 (要求 `n_loading > 0`).

### 4.4 淘汰策略与预约

**淘汰策略** `custom_evict_policy` ([h:375](../../src/custom-moe-offload.h#L375)) 是一个接口, 记录池中所有层的专家 (`Loading` 和 `InMemory`), 并选出被淘汰的专家:

| 接口 | 调用位置 | 作用 |
|---|---|---|
| `insert` / `insert_layer` | `load_expert` / `load_layer` | 专家拿到 slot (开始加载) |
| `use` | `id_pos_map` | 按 token 顺序记录每次使用 |
| `erase` | `_free_expert` | 专家离开池 |
| `victim(can_evict)` | `alloc_slot` | 在 `can_evict` 为真的专家中选一个 |
| `layer_begin(il, ids, prefill)` | `ensure` 开头, 加载缺失专家之前 | 可选钩子 (默认空): 当前层和它用到的专家 |
| `observe(il, top-k, prefill)` | `id_pos_map`, `ensure` 之后、预测器之前 | 可选钩子 (默认空): 这一层所有 token 的路由 |
| `lock_layer()` | `alloc_slot` | 默认 true: 当前层整层不可淘汰 (`can_evict`); false: 只锁当前层用到的专家 (`can_evict_unused`) |

实现:
- `custom_evict_lru` ([h:400](../../src/custom-moe-offload.h#L400)): 标准 LRU, 一条全局链表, 头部 = 最近使用, 从尾部选.
- `custom_evict_random` ([h:460](../../src/custom-moe-offload.h#L460)): 从一个随机起点扫描, 选第一个可淘汰的. `use` 不做任何事.
- `custom_evict_arpe` ([custom-moe-arpe.h](../../src/custom-moe-arpe.h)): ARPE, 见 5.7. `insert/use/erase` 都不做事, 驻留信息全部通过 `can_evict` 回调从专家表读取.
- 新的缓存管理方法 = 新的子类 + 在 `custom_strategy_for` 中把某个模式的 `evict` 指向它 (见 5.1).
- 所有子类都只能从 `can_evict` 接受的专家中选; 状态检查 (`Loading`、`predicted`、当前层) 不在子类里重复实现, 所以子类不会淘汰正在加载或被预约的专家.

**prefill 的淘汰: 按层距离 (`prefill_victim = farthest_layer`)**, `victim_farthest` ([cpp:369](../../src/custom-moe-offload.cpp#L369)). cross-layer 和 io-aware 在 prefill 时不用 LRU, 而用这个规则; decode 仍用淘汰策略对象.
- 原因: prefill 按层号顺序扫一遍所有层, 是典型的循环访问. 这时 LRU 选错: 预加载层 1..N-1 只在加载时 "用过", 比刚算完的层 0 更 "旧", LRU 会先淘汰马上要用的层 1, 而不是已经用完的层 0.
- 规则: 层 l 的下一次使用在 `(l - cur_il) mod n_moe_layer` 层之后 (算完的层要等下一个 ubatch 或 decode). 从距离最大的层开始淘汰, 即**刚算完的层最先被淘汰**, 然后是更早算完的层, 最后才是最远的未来层. 这是这种访问顺序下的 Belady 最优.
- 为层 t 取 slot 时, 只能淘汰距离大于 t 的层: 预取永远不会挤掉比自己更早要用的专家. 当前层的按需加载 (t = `cur_il`) 可以淘汰任何其他层.
- **释放是惰性的**: 不会在一层算完时整层释放. 只有当某个读请求确实需要 slot 时才淘汰一个专家, 所以 IO 慢时, 算完的层大部分专家仍留在池里 (对下一个 ubatch 和 decode 有用).

**on-demand 的 prefill: 按层加载和释放** (`ensure_layer`). 与上面两种模式不同, on-demand 在 prefill 时整层进出:
- 一层开始时如果这一层不是整层驻留 (包括只有部分专家在池里的情况), 先释放这一层自己的专家, 再找一块连续的空 slot 整层读入 (`load_layer`: 先读完所有 up, 再所有 gate, 再所有 down).
- 腾位置时只释放**整层**: `victim_layer` 找距离最远的、所有专家都可以释放的层, `free_layer` 释放它的全部专家, slot 通过 `pool_free` 归还并合并成连续区间.
- 效果就是滑动窗口: 预加载 0..N-1 层之后, 第 N 层淘汰第 N-1 层的位置, 第 N+1 层淘汰第 N 层, 以此类推.
- 池被 decode 打散 (空闲 slot 不连续) 时无法整层放入, 退化为一层内逐个专家加载, 打印一次 warning.
- decode 仍是逐个专家阻塞读 (一个 token 只用 k 个专家, 整层读会多读约 10 倍).

**淘汰条件** `can_evict(il, id, cur_il)` ([cpp:862](../../src/custom-moe-offload.cpp#L862)), 两种选择方式共用:
- `il != cur_il`: 当前正在计算的层不能淘汰, 因为它的专家已经映射给了本 ubatch.
- `state == InMemory`: 正在加载的专家有读请求在途.
- `!predicted`: 预约的专家在所在层计算前不能淘汰. 否则用纯 LRU 时, 很早之前预测的专家比刚算完的层更 "旧", 会先被淘汰.

`can_evict_unused(il, id)` (策略的 `lock_layer() == false` 时使用, 目前只有 ARPE): 第一条改为 "不是当前层用到的专家" (`cur_used`, 由 `ensure` 设置). 安全性: 映射给本 ubatch 的正好是 `ensure` 收到的那些专家, 当前层的其他专家没有被映射. 后两条不变.

**预约** (`reserve_and_load`, [cpp:539](../../src/custom-moe-offload.cpp#L539)): 所有预测器都通过它工作. 已经预约的直接返回; 预约总数达到 `reserve_budget` 时失败; `OnDisk` 的专家先用 `alloc_slot(目标层, wait = false, prefill)` 拿 slot 并提交异步读, 拿不到就失败; 最后标记 `predicted`. 某一层开始计算时 (`ensure`) 清除该层所有专家的 `predicted`, 不论预测是否命中.

**不会死锁的理由**: 预约总数 <= `n_slots - n_expert`, 所以当前层需要的 (最多 `n_expert` 个) slot 一定来自空闲 slot、可淘汰的专家, 或者在途读请求完成后的专家. `alloc_slot(cur_il, wait = true, prefill)` ([cpp:391](../../src/custom-moe-offload.cpp#L391)) 在没有可淘汰专家时等待一个读请求完成后再试.

### 4.5 IO `custom_async_io` ([h:135](../../src/custom-moe-offload.h#L135), [cpp:19-98](../../src/custom-moe-offload.cpp#L19-L98))

- 用 `O_RDONLY` 打开模型文件 (没有 O_DIRECT), 创建**两个** io_uring, 都注册这个 fd (fixed file 下标 0). mlock 只是尽力而为.
  - `ring` (SQ 深度 640, CQ 32768): 运行时的所有异步读都走这里, 只有 `refresh_completions` 收割它的 CQE. 每次提交都立即 `io_uring_submit`, SQ 不会积压; 但 CQE 收割得晚, CQ 必须能容纳所有在途读请求 (最多每个 slot 3 个), 所以 CQ 设为 32768 (`IORING_SETUP_CQSIZE | CLAMP`).
  - `block_ring` (深度 4): 阻塞读, 用于模型加载时的 `load_layer` 和 on-demand 模式.
- 分开的原因: `block_read` 里的 `io_uring_wait_cqe` 会拿到**任意一个** CQE, 如果两种读共用一个 ring, 就可能吞掉异步读的完成事件.
- 异步读带 `IOSQE_ASYNC`. 文件是普通的 buffered 读 (ext4, 没有 O_DIRECT). 不带这个标志时, 内核在 `io_uring_submit` 里先同步尝试读: page cache 里已有的数据 (相邻专家的预读、decode 时之前读过的专家) 当场拷贝; 没有的部分发起 128 KB 的预读, 之后的重试以 task_work 的形式在**提交线程**上执行. 提交线程是计算线程, 其他线程在屏障处等它, 所以这些拷贝实际上都在计算的关键路径上, "异步" 预取和按需读几乎没有区别. 带上之后, 读和拷贝都由内核的 io-wq 工作线程完成.
- `refresh_completions(wait)`: 返回 `{user_data, 读到的字节数}`; `wait = true` 时先 `io_uring_wait_cqe` 睡眠, 不空转. `res < 0` 或 `user_data == 0` 时抛异常.
- 兼容性: 只用 liburing 一直存在的接口. SQE 数组的 mlock 长度用 `*r.sq.kring_entries` (2.5 里标为 deprecated, 但 liburing < 2.2 只有它; 没有 `sq.ring_entries`). Ubuntu 22.04 / JetPack 6 自带的是 liburing 2.1.

### 4.6 自定义算子的 userdata `runtime_ctx` ([h:539](../../src/custom-moe-offload.h#L539))

- `{moe_unified, il_moe}`, `custom_moe_unified::rt_ctxs` 每个 MoE 层一个, 构造时创建, 之后不再 resize.
- `id_pos_map`、`custom_frontier`、`custom_predict_frontier` 的 `il` = 正在计算的层; `custom_predict` 的 `il` = **被预测的目标层** (固定目标). 正在计算的层由 `ensure` 写入 `moe_unified->cur_il`.
- 生命周期和 `custom_moe_unified` 相同. 算子里**不能 delete** 它: 图会被复用, 同一个算子节点会执行很多次.

## 5. 运行模式与生命周期

### 5.1 模式 = 独立组件的组合

`custom_strategy_for(mode)` ([cpp:118](../../src/custom-moe-offload.cpp#L118)) 是**唯一**把模式映射到组件的地方:

| 模式 | `load` | `evict` (decode) | `prefill_victim` | `prefill` 预测 | `decode` 预测 |
|---|---|---|---|---|---|
| `on-demand` | block (prefill 整层, decode 逐个专家阻塞读) | random | policy (= random) | none | none |
| `cross-layer` | async (一层缺的专家并行读) | lru | farthest_layer | cross_layer (第 i 层预测第 i+1 层) | next_layer |
| `io-aware` | async | arpe | farthest_layer | io_aware (前沿层, 按 IO 节奏) | next_layer |

- 组件之间互相独立, 代码里只按组件分派 (`strategy.prefill` / `strategy.decode` 的 switch, `evict` 的虚函数), 从不判断 "当前是哪种模式".
- io-aware 的 decode 预测目前和 cross-layer 用的是同一个组件 (`next_layer`); 淘汰策略已经换成 ARPE (5.7), cross-layer 仍是 LRU. 每个模型实例各自创建自己的淘汰策略对象, 没有共享状态. 以后再换 io-aware 的 decode 预测: 新增一个 `custom_decode_kind` 值 (在 `predict_targets` 和 `on_predicted` 里各加一个 case), 然后只改 `custom_strategy_for` 里 io-aware 那一行.

### 5.2 加载与建图

**加载阶段**:
1. `load_arch_tensors` (deepseek2/qwen3moe) 创建专家张量时带上 `TENSOR_EXPERT_WEIGHT` -> 张量头放在 `moe_table_ctx`, **没有数据**.
2. [llama.cpp:372-378](../../src/llama.cpp#L372-L378) 构造 `custom_moe_unified`: 选模式的组件; 选出最大的专家张量确定池类型和 slot 大小; `table_init`; 分配池 (io-aware 时还有 `routers_all`, 由 `routers_init` 填充); 计算 `n_preload_layers`、`reserve_budget`、`io_queue`; 创建淘汰策略对象和 `rt_ctxs`.
3. `prefill_init()` ([cpp:346](../../src/custom-moe-offload.cpp#L346)): 用 MoE 层 `0 .. n_preload_layers-1` 的全部专家把空池填满 (阻塞顺序读). 加载结束时打印模式、预加载层数、预约上限, 以及 Q (io-aware).

**建图阶段**: `build_moe_ffn_offload` ([graph.cpp:1581](../../src/llama-graph.cpp#L1581))

```
build_select(router, bias) = router x cur -> softmax/sigmoid -> (+bias) -> ggml_top_k        [专家 id]

selected = build_select(gate_inp, exp_probs_b)
select_map = ggml_map_custom2(selected, selection_probs, id_pos_map, rt_ctxs[il])                [专家 id -> 池 slot]
for target in moe_unified->predict_targets(il, n_tokens > 1):                                     固定目标 (cross-layer, decode)
    predicted = build_select(router(target), router_bias(target))                                 用当前层的 cur 预测 target 层
    ggml_map_custom1_inplace(predicted, custom_predict, rt_ctxs[target])
if moe_unified->predicts_frontier(il, n_tokens > 1):                                              动态目标 (io-aware prefill)
    rows = ggml_custom_4d(I32 [n_expert], args = {select_map}, custom_frontier)                   运行时选目标层 t, 写出 t*n_expert + j
    predicted = build_select(get_rows(routers_all, rows), get_rows(router_biases_all, rows))
    ggml_map_custom1_inplace(predicted, custom_predict_frontier, rt_ctxs[il])
up/gate/down = mul_mat_id(pool_ups/gates/downs, cur, select_map)   <- 用 slot 下标做索引
weights = get_rows(probs, selected)  (norm + clamp, scale 为 0 或 1 时跳过)
silu(gate) * up -> down -> * weights -> 各专家求和
```

- `predict_targets(il, prefill)` ([cpp:633](../../src/custom-moe-offload.cpp#L633)) 给出**建图时就确定**的目标层: decode `next_layer` -> `il + 1`; prefill `cross_layer` -> `il + 1`; 其他为空.
- io-aware 的目标层在**运行时**才确定 (取决于 IO 进度), 而图是固定的 (还会被复用). 做法: 所有层的 router 拼成 `routers_all`, `custom_frontier` 在运行时把目标层的行号写进一个 I32 张量, `get_rows` 取出这一层的 router, 后面的 matmul / top-k 和普通预测完全相同. 不预测时也会算一次 (行号指向当前层), 结果被忽略; 代价是每层一次 router matmul, 约占专家计算量的 0.3%.
- 预测节点必须在 `select_map` 之后执行: `id_pos_map` 先写入 `cur_il`. `custom_frontier` 把 `select_map` 作为参数, 保证这个顺序. 自定义算子都在 CPU 上按图的顺序执行.
- 与上游相比缺少: expert group 选择, `SOFTMAX_WEIGHT` 等门控, `gate_up_exps` 合并张量, 专家 bias/scale, `swiglu_clamp`, `ggml_swiglu_split` 融合. 目标模型用不到这些.

### 5.3 每层开始: `id_pos_map` ([cpp:991](../../src/custom-moe-offload.cpp#L991)) -> `ensure` ([cpp:479](../../src/custom-moe-offload.cpp#L479))

**线程模型**: 当前 ggml-cpu 中, 线程池里的**所有线程都会执行每个节点**. `n_tasks = 1` 只影响规划阶段, 执行时 `nth` = 线程数, 每个算子要自己按 `ith` 分工. prefill 时由第一个拿到 `CallSync` 锁的线程执行加载, 其他线程等待, 然后所有线程按 token 分段并行写 slot 下标; decode 时只有 `ith == 0` 执行.

`ensure(il, 这一层用到的专家, prefill)`:
1. `cur_il = il`, `cur_used` = 这一层用到的专家; 清除该层的 `predicted`: 这一层的预测已经被消费. prefill 且 `il == 0` 时 (新的 ubatch) 把 io-aware 的前沿 `frontier` 清零. 调用 `evict->layer_begin`.
2. 加载缺失的专家, 按 `strategy.load`:
   - block + prefill (on-demand): 整层加载 `ensure_layer`, 整层不在池里时先整层释放再整层读入, 见 4.4.
   - block + decode (on-demand): 逐个 `alloc_slot(il, wait = true, prefill)` + 阻塞读.
   - async (cross-layer / io-aware): 先把所有缺失专家都提交异步读 (并行 IO), 再等待.
3. 等待被预测过、仍在加载的专家.

随后调用 `evict->observe` (整层路由), 再按 token 顺序对每个 (token, k) 调用 `evict->use`. 这些都在同一个线程里执行 (prefill: `CallSync` 的初始化函数; decode: `ith == 0`).

**prefill / decode 的判断** (`ubatch_prefill`): 在**第一个 MoE 层**按 token 数决定 (`> 1` 为 prefill), 本 ubatch 的后续层沿用. 原因: 最后一层只计算输出 token (`inp_out_ids`), prompt 的最后一个 ubatch 在最后一层只有 1 个 token, 以前会被当成 decode (统计、淘汰方式都错). 预测器的 `prefill` 参数用 `in_prefill()`. 线程分工 (`CallSync` 还是 `ith == 0`) 仍按本层的 token 数.

prefill 只加载这一层**实际用到的**专家. 每层平均用到的不同专家数约为 `n_expert x (1 - (1 - k/n_expert)^n_tokens)` (路由均匀时), 例如 Qwen (128 选 8): 10 个 token 约 61 个, 70 个 token 约 127 个.

### 5.4 预测

`custom_predict` ([cpp:1044](../../src/custom-moe-offload.cpp#L1044), 固定目标) 和 `custom_predict_frontier` ([cpp:1081](../../src/custom-moe-offload.cpp#L1081), 动态目标) 把预测结果 (去重, 按首次出现的顺序; 单 token 时就是 top-k 顺序) 交给 `on_predicted` ([cpp:615](../../src/custom-moe-offload.cpp#L615)), 由组件决定怎么处理.

**decode `next_layer`** (`predict_next_layer`, [cpp:583](../../src/custom-moe-offload.cpp#L583)): 用当前层的 `cur` 预测下一层的 top-k, 按组件决定预取几个:
- `cross-layer` (`decode_load = 0`): 下一层**所有缺失**的专家都预约并异步读入 (最多 top-k 个).
- `io-aware` (`decode_load = --moe-decode-prefetch`, 默认 2): 只预取前 N 个缺失专家; 缺失数 <= N 时全部预取. 依据是 "一层 decode 的计算时间大约只够读 N 个专家", 多预取只会积压在队列里, 还会挤占当前层按需读的带宽. 0 = 不限.
- 已在池中的专家不重复读; 预约在下一层 `ensure` 时清除. 预约总数仍受 `reserve_budget` 限制, 池满或额度用尽时提前停止.

#### prefill: cross-layer 和 io-aware 的共同点

- 都是**跨层预测**: 用第 k 层的输入 `cur` 乘目标层的 router, 得到目标层的 top-k. prefill 时结果是所有 token 用到的专家的并集.
- 都通过 `reserve_and_load`: 已在池里的专家只预约 (保护), 缺的异步读入, 不等待.
- 都用按层距离淘汰 (4.4), 预取不会挤掉比目标层更早要用的专家.
- 模型加载后池里是 MoE 层 `0..N-1` (N = `n_preload_layers`).
- **最后一个 MoE 层只计算输出 token**: 模型在最后一层之前把 `cur` 裁剪到输出行 (`inp_out_ids`), 所以它的专家只来自输出 token. 预测它的规则因此不同:
  - 固定目标 (cross-layer, 在第 L-2 层预测第 L-1 层): 用 `get_rows(cur, out_ids)` 预测. 输出行数为 0 的 ubatch (prompt 不是最后一个 ubatch, 例如 1024 token 的第一个 ubatch) 直接不预测.
  - 前沿层 (io-aware): 目标在运行时确定, 无法静态裁剪, 所以第 L-1 层**永远不当预测目标**, 它需要的 k 个专家按需加载 (比整层预取便宜得多).
  - 原因: 以前用全部 token 预测最后一层, 读进整层 (~64 个专家), 而最后一层只用 ~k 个; 更糟的是没有输出 token 的 ubatch 里最后一层整层不执行, `ensure` 不会调用, 预约永远不清除, 一直占着 slot 到下一个 ubatch.
  - 保底: 新的 prefill ubatch 开始时 (`ensure` 的 `il == 0`) 清除所有预约 (`clear_all_predicted`), 即使将来又出现没被消费的预测, 也不会跨 ubatch 泄漏.

#### prefill `cross_layer`: 下一层 (`prefetch_cross_layer`, [cpp:561](../../src/custom-moe-offload.cpp#L561))

这个方法关注的是**跨层预测本身**, 不是 IO 最大化. 预测距离固定为 1, 没有超参数.

- 计算到第 i 层时, 用第 i 层的 `cur` 预测第 i+1 层, 预测出的专家异步读入, 和第 i 层的专家计算并行. 第 i+1 层开始时 (`ensure`) 等这些读完成, 预测漏掉的专家这时按需加载.
- 不看 IO 状态; 放不下的部分 (预约已满, 或没有比第 i+1 层更远的专家可以淘汰) 直接放弃, 由第 i+1 层按需加载.
- 第 0..N-2 层预测的第 1..N-1 层已经在池里 (预加载), 只预约, 不读盘. 第 N-1 层预测第 N 层时开始读盘, 这时算完的第 0..N-2 层最先被淘汰.
- decode 的 `next_layer` 也是预测下一层, 区别是每层的预取数量由组件决定 (cross-layer 全部缺失, io-aware 至多 N 个, 见 5.4).

#### prefill `io_aware`: 前沿层 + IO 节奏 (`next_frontier` [cpp:589](../../src/custom-moe-offload.cpp#L589), `prefetch_frontier` [cpp:568](../../src/custom-moe-offload.cpp#L568))

**前沿层** `frontier`: 本 ubatch 中第一个还没被预测的层. 新 ubatch 开始时清零, 然后跳过所有专家都在池里的层, 所以预加载之后就是第 N 层.

从第一个 MoE 层开始, 第 k 层计算时 `custom_frontier` 调用 `next_frontier()`:
1. `frontier = max(frontier, k+1)`, 并跳过所有专家都在池里的层.
2. 如果 `frontier == k+1` (下一层马上要用), 或者 IO 队列长度 `loading_count() < Q`: 用**第 k 层刚算出的 `cur`** 预测 frontier 层.
3. 否则这一层不预测. 此时 IO 是慢的一方, 现在提交只会排在队尾; 不如让后面的层用更新的 `cur` 预测同一个 frontier, 距离更近, 也更准.

`prefetch_frontier`: 预约并读入预测出的全部专家. 全部成功时 `frontier + 1`; 有失败 (没有 slot 或预约已满) 时 frontier 不变, 下一层用新的 `cur` 再预测这一层 (结果取并集, 已预约的专家不会重复读). 前沿层推进到最后一个 MoE 层时停止预测 (见上一条).

所以**每层最多预测一层, 每次都用最新的 `cur`**, 预测距离 = `frontier - k`, 随 IO 进度自动变化.

**生产者 (IO) / 消费者 (计算)**:
- IO 慢 (常见情况): 队列 >= Q 时跳过预测, frontier 不动, 预测距离自动缩短. 淘汰是惰性的 (4.4), 不提交读就不淘汰, 所以不会出现 "每层算完就释放一整层, 但 IO 跟不上, 白白丢掉专家" 的情况. 这正是原来 D 要解决的问题, 现在由 IO 队列的实际长度驱动, 而不是一个固定的层数.
- IO 快: 每层都预测, frontier 每层前进 1, 预测距离保持在 N 左右. 上限由池决定: 只能淘汰比目标层更远的专家.
- Q 的含义是 "IO 队列里至少保持多少专家才不会断流". Q 太小时 IO 可能在两次预测之间空闲; Q 太大时预测距离变远、准确率下降, 也会更早挤掉算完的层. 默认 Q = 一层的专家数.

**IO 从哪一层开始**: 预加载把池填满时, 第 0 层 (第一个 MoE 层) 没有可以淘汰的专家: 层 0 正在计算, 层 1..N-1 比目标层 N 更早要用. 所以第 0 层的预测只能用到池里剩余的空 slot (`n_slots - N * n_expert` 个); 没有空 slot 时这次不读盘, frontier 保持为 N. 到第 1 层时, 层 0 已经算完, 成为距离最远的层, 第 1 层用新的 `cur` 预测第 N 层, IO 开始. 也就是说, IO 最晚从第二个 MoE 层开始.

#### 对比

| | cross-layer | io-aware |
|---|---|---|
| 关注点 | 跨层预测 | IO 最大化 (计算和 IO 尽量重叠) |
| 目标层 | 第 i+1 层, 建图时确定 | 前沿层, 运行时确定 |
| 预测距离 | 固定 1 | 自适应: 开始约为 N, IO 慢时缩短, 最小为 1 |
| 是否看 IO 状态 | 否 | 是, 队列 >= Q 时跳过 (下一层除外) |
| 放不下时 | 放弃, 计算时按需加载 | 下一层用新的 `cur` 重试 |
| 预加载的层 | 第 N-1 层才开始读盘 | 跳过已在池里的层, 最晚第二个 MoE 层开始读第 N 层 |
| 超参数 | 无 | Q (`--moe-io-queue`) |
| 每层的图节点 | `router(i+1)` | `custom_frontier` + `get_rows(routers_all)` |

### 5.5 decode

decode 和 prefill 走同一个 `ensure`. on-demand: 逐个专家阻塞读 (decode 不整层), random 淘汰; cross-layer: 并行异步读 + 预取下一层全部缺失专家, LRU 淘汰; io-aware: 同样的预测器但每层最多预取 N 个 (2.2 的 `--moe-decode-prefetch`), ARPE 淘汰 (5.7).

### 5.6 统计 (`custom_moe_stats`)

prefill 和 decode 各一份, 模型释放时 (`~custom_moe_unified`) 打印一行 `custom_moe stats [prefill|decode]`:

| 字段 | 含义 |
|---|---|
| `layer runs` | `ensure` 的调用次数 (prefill: ubatch 数 x MoE 层数; decode: token 数 x MoE 层数) |
| `hit / wait / miss` | 每层实际用到的专家在 `ensure` 开始时的状态: 已在池中 / 已预取但还没读完 / 不在池中 (按需读) |
| `prefetch (used)` | 预测器读入的专家数, 括号内是其中在所在层被用到的个数 |
| `read` | 读盘总字节数 (不含加载模型时的预加载) |
| `ensure` | `ensure` 的总时间. 这段时间里所有计算线程都在等这一层的专家, 即 IO 没有被计算掩盖的部分 |
| `predict` | 预测器的总时间 (分配 slot、淘汰、提交读请求), 也在计算的关键路径上 |

分析方法: `prompt eval time - ensure - predict` 约等于纯计算时间. 如果某个模式的这个值明显大于 on-demand, 说明后台 IO 拖慢了计算 (例如 page cache 到池的拷贝、内存带宽竞争); 如果 `miss` 多, 说明预测或淘汰有问题; 如果 `wait` 多且 `ensure` 长, 说明预取太晚或 IO 本身太慢.

### 5.7 ARPE: io-aware 的 decode 淘汰 (`custom_evict_arpe`)

方法本身见 [ARPE_施工指南.md](ARPE_施工指南.md), 本节只写和本工程的对接. 参数用指南的默认值 (`custom_arpe_params`: W0 = 40, alpha = 0.16, rho = 0.1, lam = 0.6, w_aff = 20, 在线亲和和 prefill 先验都打开), 目前没有命令行参数.

**调用时机** (对应指南 6.1 的顺序):

| 指南 | 本工程 |
|---|---|
| 每层开始, 拿到 S_c | `ensure` 开头 `layer_begin(c, S_c)`: 记下当前层和锁定集合, `observed = false` |
| 按需加载前 `select_victim` (observe 之前) | `ensure` 里的 `alloc_slot(wait = true)` -> `victim`, 用 t = `seen[c]` |
| `observe_prefill` / `observe_decode` | `id_pos_map` 中 `ensure` 之后调用 `observe` |
| observe 之后的预取 (`observed = True`) | 下一层预测器 `predict_next_layer` -> `reserve_and_load` -> `alloc_slot(wait = false)` -> `victim`, 用 t = `seen[c] - 1` |

- **prefill 不用 ARPE 淘汰**: io-aware 的 prefill 用按层距离淘汰 (`prefill_victim = farthest_layer`, 4.4), 指南允许沿用系统已有的 prefill 策略. ARPE 只在 prefill 里收集统计 (先验、`last = -1`、在线亲和).
- **多 ubatch 的 prompt**: 先验按所有 ubatch 累加; `last` 取最后一个 ubatch 的最后一个 token; 相邻层配对只在同一个 ubatch 的相邻层之间 (`prev_layer == il - 1`).
- **最后一层只有输出 token**: 按指南 3.2, 与上一层的**最后 n 行**对齐配对. 这依赖 5.3 中的 prefill 判断修正, 否则最后一层会被当成一个 decode token.
- **请求边界** (指南 11.4 的 "单请求模式"): decode 之后的第一个 prefill ubatch 在 MoE 层 0 重置所有请求级统计. 同一个 prompt 的多个 ubatch 不重置. 多轮对话时每一轮算一个请求.

**和已有机制的兼容** (排查时先看这里):
- ARPE 只在 `can_evict_unused` 接受的专家里选: 只锁当前层用到的专家 (指南 8.1), 当前层的其他专家可以淘汰; `Loading` 的专家不可淘汰 (指南 13.6); `predicted` 的专家 (decode 下一层预测刚预约的专家) 不可淘汰. 最后这一条是**和指南的差异**: 指南里没有预约, 但这里必须保留, 否则下一层预测刚读入的专家可能马上被淘汰, `_free_expert` 的断言也会失败.
- 没有候选时 `victim` 返回 false: 按需加载 (`wait = true`) 等一个在途读请求完成后重试 (指南 8.3 的 "等待"); 预取 (`wait = false`) 直接放弃. 预约上限保证当前层总能拿到 slot, 不会死锁 (4.4).
- 当前层计算期间 `p` 只算一次 (`p_valid`), 同一层多次淘汰复用; `layer_begin` 和 `observe` 使其失效 (指南 11.3 第 8 条).
- 所有调用都在单线程中 (5.3), ARPE 不加锁.

**离线亲和先验 A0** (指南 3): `A0 = (C + 1) / (N + E/K)`, 计数文件不存在或不匹配时等于均匀先验 K/E (C = N = 0).
- 标定: `--moe-arpe-calib FILE` 跑若干条**不用于评测**的 prompt (指南建议 5-10 条), 每次运行结束时把本次所有 token (prefill + decode) 的相邻层计数累加进 FILE.
- 使用: `--moe-arpe-affinity FILE`.
- 文件格式: 头 `{magic "ARPE", version 1, n_moe_layer, n_expert, n_expert_used}` (5 个 uint32), 然后 `N[(L-1) x E]`、`C[(L-1) x E x E]` (double). 头与模型不符时忽略 (affinity) 或不写 (calib), 打印 warning. 大小: DeepSeek-Lite 0.8 MB, Qwen3-30B 6.2 MB.

**日志**: 加载时打印 A0 的来源; 模型释放时打印 decode 淘汰的层分布 `custom_moe ARPE: N decode evictions: x% current layer, y% next layer, z% previous layer` (指南 12.3: 不应 "总是淘汰刚用过的上一层"), 以及标定文件的写入情况.

## 6. Jetson 统一内存 (CUDA 构建)

纯 CPU 构建时, 所有权重、专家池和 KV cache 都在系统内存中, 计算全部在 CPU 上, 本节不适用.

### 6.1 设计目标

专家池必须**同时**被 CPU 和 iGPU 访问:
- CPU: io_uring 把专家从 SSD 读进池.
- iGPU: `mul_mat_id` 读池里的专家.

做法: 池分配在**设备内存** (`ggml_backend_dev_buffer_type(dev)`), 运行时设置 `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, ggml-cuda 的设备分配改用 `cudaMallocManaged` (managed memory). managed 内存在 Linux 64 位 UVA 下 CPU 和 GPU 使用同一个指针, io_uring 可以直接写入, GPU 把它当普通设备 buffer 计算. Jetson 上物理上仍是同一块 DDR.

### 6.2 需要满足的条件和对应改动

1. **找到 iGPU 设备**. Jetson 的 CUDA 设备类型是 `GGML_BACKEND_DEVICE_TYPE_IGPU`, `ggml_backend_dev_by_type(GPU)` 找不到它. 池和设备层都取 `model.devices.front()` (llama 在没有独显时会把 iGPU 放进 `devices`).
2. **池在设备内存**. 调度器把用到池的 `MUL_MAT_ID` 调度到 iGPU, 没有拷贝, 也不需要 `integrated` 补丁 (上游保持 `false`, 见 6.5). 构造时检查: 如果池的 buffer type 不是 host 内存而 `GGML_CUDA_ENABLE_UNIFIED_MEMORY` 没有设置, 直接报错 (否则 io_uring 会写 cudaMalloc 的设备指针).
3. **非专家权重**走上游路径: 放在设备的默认 buffer (`gpu_buft_list`), 也是 managed 内存. KV cache 和计算 buffer 同样在设备内存.
4. `routers_all` / `router_biases_all` 也在池的 buffer 里, 初始化时用 `ggml_backend_tensor_set` 写入 (不能直接写 `tensor->data`).

### 6.3 CPU 写入和 GPU 读取的顺序

每层的图被切成 `GPU (router, top_k) -> CPU (id_pos_map, custom_predict) -> GPU (mul_mat_id ...)`. CPU 段开始前, 调度器要把 top_k 结果拷到 CPU, 这时 GPU 已经执行完前面所有的 kernel. 所以:
- 被淘汰的 slot 只属于前面的层 (当前层受保护), 它们的 kernel 已经执行完, 可以安全覆盖.
- 预测的异步读写入空闲 slot 或被淘汰的 slot, 不会和当前层的 `mul_mat_id` 冲突. 使用前 `ensure` 会等待读完成.

### 6.4 风险与待验证

- managed 分配在这块板子上的正确性和性能还没有验证 (6.5).
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 会让所有设备分配 (权重、KV cache、计算 buffer) 都变成 managed; 在 Jetson 上它们本来就是同一块 DRAM, 预期没有额外开销, 需要用实测确认.
- 每层都有 CPU 段, CUDA graph 用不上 (Jetson preset 已关闭).
- 旧方案 (池放 CUDA_Host pinned 内存 + 恢复 `integrated`) 已废弃, 原因见 6.5.

### 6.5 已知问题: iGPU 直接读 host 内存时 batch > 8 输出错误 (2026-10-21 定位)

现象 (Jetson Orin Nano 8GB, JetPack 6.2 / L4T R36.4.3, CUDA 12.6, liburing 2.1):
- prompt <= 8 token 输出正常, >= 9 token 开始输出 `!!!!` 这类乱码.
- 与 `--moe-pool-ratio`、运行模式无关; `-ngl 0` (层全放 CPU) 也一样.

定位过程:
1. 把 ggml-cuda 里 CUSTOM_MOE 的 `integrated = prop.integrated` 临时改回 `false` (上游行为), 同样的 prompt 输出正确. 因此问题出在 "iGPU 直接访问 CUDA_Host 上的权重" 这条路径, 也就是上游 #15034 关闭 `integrated` 的原因.
2. 边界正好是 8: 7/8 token 正常, 9 token 起出错. `ggml/src/ggml-cuda/mmvq.cuh` 里 `MMVQ_MAX_BATCH_SIZE == 8`: batch <= 8 时 `mul_mat_id` 走 MMVQ 内核 (能正确读 host 内存), batch > 8 走 MMQ / 其它路径 (读 host 内存得到错误结果).
3. CUSTOM_MOE 下 `get_layer_buft_list` 让**所有层权重**都在 CUDA_Host, 不只专家池, 所以整个 prefill 都受影响.

规避 (当前可用): 示例新增 `-ub N` (设置 `ctx_params.n_ubatch`), 在 Jetson 上用 `-ub 8`, 所有 ubatch <= 8, 走 MMVQ. 实测 DeepSeek-Lite / 66 token prompt: cross-layer prefill 4.6 t/s, io-aware 4.1 t/s, on-demand 1.1 t/s; decode 约 3 t/s (前几个 token 池是冷的). 代价: 每个 ubatch 都从第 0 层开始, 池小的时候 prefill 会重复读盘 (on-demand 最明显).

**已实施的修改 (2026-10-21, 待板端验证)**: 按 6.1/6.2 的方案, 专家池改到设备内存 (managed 分配), 去掉 CUSTOM_MOE 的 `integrated` 补丁和 `get_layer_buft_list` 分支; 运行时要设置 `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`. 预期不再有 host 权重被 iGPU 读取的问题, 可以去掉 `-ub 8`. 验证方法: 用同一段长 prompt 分别跑 "不带 -ub" 和 "带 -ub 8", 输出应完全一致; 若一致, 删除示例里的 `-ub` 参数.

## 7. 不变量 (排查问题时先检查这些)

1. `state == InMemory/Loading` <=> `pos >= 0` <=> 该专家被淘汰策略记录.
2. `state == Loading` <=> `pending_io > 0` (仅异步读); `n_loading` = `Loading` 专家的个数.
3. 当前层 (`cur_il`) 的专家、`Loading` 的专家、`predicted` 的专家都不能被淘汰 (`can_evict`). 例外: `lock_layer() == false` 的策略 (ARPE) 只锁当前层用到的专家 (`can_evict_unused`).
4. `n_reserved` = `predicted` 专家的个数 <= `reserve_budget`; `predicted` 只出现在还没计算的层上, 每层计算开始时清除, 新的 prefill ubatch 开始时全部清除.
10. `n_resident[il]` = 该层 `Loading` + `InMemory` 的专家数, 只由 `mark` 修改.
11. prefill 中 `frontier > cur_il` (由 `next_frontier` 保证); 只在 `il == 0` 的 prefill `ensure` 中清零.
12. 一个 ubatch 是 prefill 还是 decode 只在 MoE 层 0 决定 (`ubatch_prefill`), 同一个 ubatch 的所有层一致.
13. 淘汰策略子类只从 `can_evict` 回调接受的专家里选, 自己不判断专家状态.
5. 代码按组件分派, 不按模式分派; 模式到组件的映射只在 `custom_strategy_for` 中.
6. 自定义算子在 CPU 线程池里执行, **所有线程都会进入**. 新增的状态修改要么只让 `ith == 0` (或 `CallSync` 的初始化函数) 做, 要么加锁.
7. 池张量在图里的地址固定, 所以图可以被复用. 被固化在算子参数里的 `runtime_ctx` 指针必须在整个模型生命周期内有效.
8. 池内的字节偏移一律用 `size_t` 计算. 读盘长度 = `layer_nbyte` (<= `nbyte_slot_*`).
9. 异步读只走 `ring`, 只通过 `refresh` 收割; `block_ring` 只用于阻塞读.

## 8. 未解决的问题

- Jetson 路径 (第 6 节) 还没有运行验证, 见 6.4.
- io-aware 只在层边界 (`custom_frontier`) 决定是否预测. 如果某一层计算时间很长, IO 队列可能在这一层中途读空. 要彻底解决需要后台线程, 暂不处理.
- 预加载把池填满时, io-aware 的 IO 从第二个 MoE 层开始, 见 5.4.
- `routers_all` 是 router 权重的一份拷贝 (DeepSeek-Lite 约 13 MB, Qwen3-30B 约 48 MB).
- ARPE 暂不做与 Python 参考实现的差分验证 (指南 12.2), 参数暂用默认值. 本工程里的 ARPE 与指南的已知差异: `predicted` 的专家不可淘汰 (5.7).
- ARPE 只支持单请求 (单 context 串行 decode).
- prompt 长度不是 ubatch 的整数倍时, 最后剩下的 1 个 token 会单独成为一个 ubatch, 在 MoE 层 0 被判定为 decode (5.3), 按 decode 的加载和淘汰方式处理. 输出正确, 只是那一层不做 prefill 预测. 例: 513 token = 512 + 1.
- 专家大小不一致导致池有约 7%-9% 的浪费 (按测试模型估算). 暂不处理.
- 已知限制 (不在目标场景内): 分片 GGUF (只打开一个 fd); 合并的 `gate_up_exps` 张量; deepseek2 的 MTP 层; 专家张量没有 buffer, 依赖 `tensor->buffer` 的上游功能 (memory breakdown, `common_fit_params`, `llama-moe-cache`) 行为未知.

## 9. 变更记录
- 2026-10-16: **三种运行模式 + IO-Aware prefill + 参数改名**:
  - 新增 `enum llama_moe_mode` (on-demand / cross-layer / io-aware). 每个模式 = `custom_strategy` 中四个独立组件 (load, evict, prefill, decode) 的组合, 映射只在 `custom_strategy_for` 中.
  - 淘汰策略改为接口 `custom_evict_policy`, 实现 `custom_evict_lru` 和 `custom_evict_random` (原来的 `custom_expert_manage`、`free_lru`、`free_random` 删除).
  - io-aware prefill: MoE 层 0 预测所有后续层, 由泵在预约上限内按层顺序持续加载; 泵在每层 `ensure` 结束时继续. cross-layer prefill 改为纯跨层预测 (每层只预测第 il+D 层, 不再在层 0 填充第 1..D-1 层).
  - 所有预测器统一通过 `reserve_and_load`, 预约总数上限 `reserve_budget = n_slots - n_expert`, 保证当前层总有位置 (原来的不死锁论证依赖 D 的截断, 现在对所有模式成立).
  - on-demand 改为逐个专家阻塞读 (原来的 `-mo 0` 也是并行异步读).
  - 异步 ring 的 CQ 增大到 32768.
  - 参数改名: `-mo` / `-mu` / `-mp` -> `--moe-mode` / `--moe-pool-ratio` / `--moe-predict-distance`; `llama_model_params` 的字段改为 `moe_mode` / `moe_pool_ratio` / `moe_predict_distance`.
- 2026-10-16: **io-aware prefill 改为前沿层预测 + IO 节奏; prefill 淘汰改为按层距离**:
  - 删除 "层 0 一次性预测所有层" 和泵 (`io_plan`, `pump`). 改为每层用最新的 `cur` 只预测一层 (前沿层); IO 队列 >= Q 时跳过, 由后面的层用更新的 `cur` 预测; 放不下时下一层重试. 新参数 `--moe-io-queue` / `moe_io_queue`.
  - 运行时选择目标层: 新增 `routers_all` / `router_biases_all`, 自定义算子 `custom_frontier` 和 `custom_predict_frontier`.
  - 新组件 `prefill_victim`: cross-layer 和 io-aware 在 prefill 时按层距离淘汰 (`victim_farthest`), 刚算完的层最先被淘汰, 预取只淘汰比目标层更远的专家. 原因是 LRU 在 prefill 中会先淘汰马上要用的预加载层.
  - 专家表新增 `n_resident`.
- 2026-10-16: cross-layer 的 prefill 预测距离固定为 1 (第 i 层预测第 i+1 层). 删除参数 `--moe-predict-distance` / `moe_predict_distance` 和 `custom_hparams::predict_distance`.
- 2026-10-16: 新增统计 `custom_moe_stats` (5.6) 和 Release 构建 preset `cpu-release`. 用户在 Debug 构建上测得 cross-layer / io-aware 的 prefill 比 on-demand 慢 (on-demand 12.2 s, cross-layer 12.5 s, io-aware 14.7 s, DeepSeek-Lite, pool 0.15, 66 token), decode 三者接近. 原因待用 Release 构建 + 统计确认.
- 2026-10-16: Release 构建的统计 (DeepSeek-Lite, pool 0.15, 66 token) 显示: prefill 的 `ensure` 只有 58 ms (cross-layer) / 102 ms (io-aware), 但 `predict` 有 755 / 649 ms; decode 的 `predict` 1161 ms 只预取了 2130 个专家 (每个约 0.55 ms, 相当于 memcpy 的速度). 结论: 预取的拷贝是在提交时同步完成的, 并没有和计算并行. 异步读改为带 `IOSQE_ASYNC` (4.5).
- 2026-10-20: **io-aware 的 decode 淘汰改为 ARPE** (5.7):
  - 新文件 `src/custom-moe-arpe.h/.cpp`: `custom_evict_arpe`; 新组件值 `custom_evict_kind::arpe`, `custom_strategy_for` 中 io-aware 的 `evict` 改为 arpe. cross-layer 仍是 LRU, on-demand 仍是 random.
  - `custom_evict_policy` 新增可选钩子 `layer_begin` / `observe` / `lock_layer` (默认空 / 空 / true, LRU 和 random 的行为不变); 新增 `can_evict_unused` 和 `cur_used`.
  - 修正 prefill / decode 的判断: 改为在 MoE 层 0 决定 (`ubatch_prefill`), 以前 prompt 最后一个 ubatch 的最后一层 (只有输出 token) 被当成 decode. 影响所有模式: 这一层现在按 prefill 统计, 用 prefill 的淘汰方式.
  - 新参数 `--moe-arpe-affinity` / `--moe-arpe-calib` (`moe_arpe_affinity` / `moe_arpe_calib`), 用于离线亲和先验的标定和加载.
  - 由于本会话的 shell 被安全检查拦截, 这一版先在没有编译的情况下写完, 之后编译修正: `n_expert_used` 是按层的函数, ARPE 的 K 取第一个 MoE 层的值, 并断言所有 MoE 层相同. `build-cpu-release` / `build-cpu-debug` / `build-cuda-moe` 均编译通过, 无警告.
  - 差分验证 (指南 12.2) 和 ARPE 参数调优暂不做, 先用默认参数调通.
- 2026-10-20: **最后一层的预测 + on-demand prefill 按层**:
  - 最后一层预测只用输出行: `build_moe_ffn_offload` 新增 `out_ids` 参数 (模型的 `inp_out_ids`), 预测最后一个 MoE 层时用 `get_rows(cur, out_ids)`, 输出行为 0 时不预测. io-aware 的前沿层不把最后一层当目标. 修复: 整层白读, 以及没有输出 token 的 ubatch 里预约永远不清除.
  - 新的 prefill ubatch 开始时 `clear_all_predicted`, 防止预约跨 ubatch 泄漏.
  - on-demand 的 prefill 改为按层: `ensure_layer` 整层读入, `victim_layer` / `free_layer` 整层释放, `pool_free` 归并连续区间. decode 仍逐个专家.
  - 三个构建 (cpu-release / cpu-debug / cuda-moe) 编译通过, 无警告.
- 2026-10-20: 第 2 节改写为可直接复制粘贴的测试手册: 2.1 三个构建的用途, 2.2 参数表, 2.3 变量和 `run()` 函数, 2.4 三种模式的命令, 2.5 正确性对比, 2.6 性能对比和统计解读, 2.7 长 prompt / 多 ubatch, 2.8 ARPE 标定流程, 2.9 常见问题.
- 2026-10-20: CMakePresets 的 `jetson` preset 从隐藏的片段改为完整的 configure preset (CUDA sm_87, CUDA graphs 关, Release, tests/tools/server 关, ccache, 4 线程), 并新增同名 build preset (jobs = 4, 只编 `llama-moe-offload`). feature.md 新增 2.10 节.
- 2026-10-20: decode 预测改为按缺失数预取: cross-layer 预取下一层全部缺失专家; io-aware 新增 `--moe-decode-prefetch N` (默认 2, 0 = 全部), 组件字段 `custom_strategy::decode_load`, `llama_model_params::moe_decode_prefetch`.
- 2026-10-21: 记录 Jetson 上 CMake < 3.23 无法读取 version 4 的 preset (`Unrecognized "version" field`), 2.10 节补了等价的手工 cmake 命令和升级 CMake 的方法.
- 2026-10-21: 修复 Jetson (liburing 2.1) 上的编译错误: `sq.ring_entries` 只有 liburing >= 2.2 才有, 改用 `*sq.kring_entries` 计算 SQE 数组长度.
- 2026-10-21: 定位 Jetson 上 batch > 8 输出乱码: CUSTOM_MOE 的 `integrated` 补丁让 iGPU 直接读 CUDA_Host 权重, 而 ggml-cuda 只有 MMVQ (batch <= 8) 能正确读 host 内存. 新增示例参数 `-ub` 作为规避 (`-ub 8`), 结论和根治方向写在 6.5.
- 2026-10-21: 尝试 managed memory 方案: 专家池改用设备 buffer (`ggml_backend_dev_buffer_type`), 依赖 `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 的 managed 分配让 CPU 仍可写; 撤销 `integrated` 补丁和 `get_layer_buft_list` 的 CUSTOM_MOE 分支; `routers_init` 改用 `ggml_backend_tensor_set`. 第 6 节重写, 2.10 加运行时的环境变量. 板端验证通过后删除示例的 `-ub` 参数.
