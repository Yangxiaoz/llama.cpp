# CUSTOM_MOE: MoE expert offload to disk

本文记录 `CUSTOM_MOE` 特性的实现细节与代码位置, 用于后续排查问题. 改动基本都用 `#ifdef CUSTOM_MOE` 包裹, 关闭宏即恢复上游代码 (例外: CMake 中的 liburing 和 `custom-moe-offload.cpp` 没有宏保护).

## 1. 目标与场景

- 场景: 内存 (CPU DDR, 或 Jetson 这类统一内存) 装不下整个 MoE 模型. 非专家权重常驻内存, **专家权重 (ffn_up/gate/down_exps) 留在 SSD 上的 GGUF 文件里**, 运行时按需读入一个固定大小的 "专家池" (expert pool).
- 与上游的 `llama-moe-cache` / `-ot` / `--cpu-moe` 等机制无关. 上游机制解决的是 "显存放不下 -> 放到 DDR", 本特性解决的是 "DDR 放不下 -> 放到磁盘".
- 读盘使用 `io_uring` (liburing), 支持阻塞读和异步读 (预取).
- 目标使用场景 (当前代码只保证这个场景): 单用户, 单个 prompt, 一次 prefill, 之后逐 token decode, 不会有第二次 prefill. 不考虑 MTP、`llama-moe-cache`、合并的 `gate_up_exps` 张量等特殊情况.

## 2. 编译与运行

开发分两个阶段:
1. 当前阶段: 在笔记本上用纯 CPU 构建 `build-cpu-debug` (preset `cpu-debug`, Debug, GGML_CUDA=OFF), 硬件拓扑是 DDR 内存 + SSD. 笔记本的独显 (4060) 属于 "CPU + 独立 GPU" 结构, 不在本特性的讨论范围内.
2. 之后的阶段: Jetson 统一内存 (CUDA 构建, 设备类型是 IGPU, 物理上仍是 DDR + SSD), 见第 6 节.

```bash
cmake --preset cpu-debug                       # 生成 build-cpu-debug
cmake --build build-cpu-debug --target llama-moe-offload -j
build-cpu-debug/bin/llama-moe-offload -m model.gguf -n 32 [-mu 0.2] [-mo 0|1] [-mp 1] '<prompt>'
```

- prompt 建议用 ASCII 单引号括起来: 用双引号时 shell 会展开 `$2000` 这类文本; 用中文引号时 shell 根本不把它当引号.
- 修改源码后**必须重新编译** `build-cpu-debug`, 再进行测试.
- `ENABLE_CUSTOM_MOE=ON` -> `add_compile_definitions(CUSTOM_MOE)`, 全局生效, 包括 ggml (在 `add_subdirectory(ggml)` 之前定义).
- 依赖 liburing (`sudo apt install liburing-dev`).
- 示例参数 (只在 `examples/moe-offload` 里解析):
  - `-mu F`: 专家池占 "设备总内存" 的比例, 0 < F < 1, 默认 0.46 (`llama_model_default_params()` 里的默认值是 0.3). CPU 构建按系统内存总量计算, 例如 31 GB x 0.2 约为 6.2 GB. 池必须至少能放下一整层专家, 否则加载时报错.
  - `-mo 0|1`: `moe_enable_optimizations`. 1 = LRU 淘汰 + 下一层专家预测/预取; 0 = 随机淘汰, 不预取.
  - `-mp N`: `n_prefetch_layers`, prefill 时向前预取的层数.
- 示例在 CUSTOM_MOE 下固定的设置:
  - `load_mode = LLAMA_LOAD_MODE_NONE` (不用 mmap).
  - `n_ubatch = n_batch = n_prompt`: prompt 只占一个 ubatch, 只做一次 prefill. 原因是 `LoadContext.is_finished` 只置位不清零 (见不变量 3): 第二次 prefill 时已经被换出的层仍会被当成常驻层, 映射到 slot -1. 另外 decode 的单专家淘汰会打散常驻层 (见不变量 2). 将来要支持长 prompt 或多轮对话, 必须先重新设计这一部分.
- 运行时约束:
  - 一旦编译进 CUSTOM_MOE, **每次加载模型都走 offload 路径**, 没有运行时开关. 不是 deepseek2 / qwen3moe 的模型会在构造函数中触发断言.
  - 只支持单文件 GGUF.

## 3. 代码地图

| 位置 | 作用 |
|---|---|
| [CMakeLists.txt:116-122](../../CMakeLists.txt#L116-L122) | `ENABLE_CUSTOM_MOE` 选项 -> 宏 `CUSTOM_MOE` |
| [CMakePresets.json](../../CMakePresets.json) | 新增 `linux-base` / `cuda` / `jetson` / `cpu-debug` preset |
| [src/CMakeLists.txt:7-15](../../src/CMakeLists.txt#L7-L15), [79-90](../../src/CMakeLists.txt#L79-L90) | 查找并链接 liburing |
| [src/CMakeLists.txt:53](../../src/CMakeLists.txt#L53) | 把 `custom-moe-offload.cpp` 加入 libllama |
| [include/llama.h:349-353](../../include/llama.h#L349-L353) | `llama_model_params` 新增 `moe_memory_utilization` / `moe_enable_optimizations` / `n_prefetch_layers` |
| [src/llama-model.cpp:3364-3368](../../src/llama-model.cpp#L3364-L3368) | 上述三个参数的默认值 (0.3 / false / 1) |
| [src/llama-model-loader.h:73-76](../../src/llama-model-loader.h#L73-L76) | 新 flag `TENSOR_EXPERT_WEIGHT = 1 << 7`, 成员 `moe_table_ctx` |
| [src/llama-model-loader.cpp:1367-1378](../../src/llama-model-loader.cpp#L1367-L1378) | `create_tensor`: 带 `TENSOR_EXPERT_WEIGHT` 的张量放进 `moe_table_ctx`, 不进 `ctx_map`, 所以**不分配 buffer, 也不会被 `load_all_data` 读入** |
| [src/llama-model.cpp:1657-1662](../../src/llama-model.cpp#L1657-L1662) | `get_layer_buft_list`: 层的 device 取 `devices.front()` (GPU 或 iGPU), buft list 用 `cpu_buft_list` -> 非专家权重放在 device 的 host buffer (CUDA_Host). 纯 CPU 构建不会走到这里 |
| [src/llama-model.cpp:1682-1698](../../src/llama-model.cpp#L1682-L1698) | 创建 `moe_table_ctx` (no_alloc, 每个 MoE 层 3 个张量头) |
| [src/llama-model.cpp:1871-1873](../../src/llama-model.cpp#L1871-L1873) | `model.table_ctx` 接管 `moe_table_ctx` 的所有权 |
| [src/llama-model.h:17-19](../../src/llama-model.h#L17-L19), [772-775](../../src/llama-model.h#L772-L775) | `llama_model::table_ctx`, `llama_model::moe_unified` |
| [src/llama.cpp:372-380](../../src/llama.cpp#L372-L380) | `load_tensors` 之后构造 `custom_moe_unified` 并调用 `prefill_init()` |
| [src/llama-context.cpp:2522-2526](../../src/llama-context.cpp#L2522-L2526) | `graph_max_nodes`: 节点上限 8x -> 10x `n_tensors` |
| [src/llama-graph.h:26-28](../../src/llama-graph.h#L26-L28), [1093-1120](../../src/llama-graph.h#L1093-L1120) | `build_moe_ffn_predict` / `build_moe_ffn_offload` 声明 |
| [src/llama-graph.cpp:1582](../../src/llama-graph.cpp#L1582) | `build_moe_ffn_predict` (当前没有调用者) |
| [src/llama-graph.cpp:1637](../../src/llama-graph.cpp#L1637) | `build_moe_ffn_offload`: 替代上游 `build_moe_ffn` 的建图函数 |
| [src/models/deepseek2.cpp:136-143](../../src/models/deepseek2.cpp#L136-L143), [655-675](../../src/models/deepseek2.cpp#L655-L675) | deepseek2: 专家张量加 flag; MoE 分支改调 `build_moe_ffn_offload` |
| [src/models/qwen3moe.cpp:51-60](../../src/models/qwen3moe.cpp#L51-L60), [142-162](../../src/models/qwen3moe.cpp#L142-L162) | qwen3moe: 同上 |
| [ggml/src/ggml-cuda/ggml-cuda.cu:308-313](../../ggml/src/ggml-cuda/ggml-cuda.cu#L308-L313) | 恢复 `integrated` 检测, 让 iGPU 直接在 CUDA_Host 上计算 (见第 6 节) |
| [src/custom-moe-offload.h](../../src/custom-moe-offload.h) | 全部数据结构 + 大量 inline 实现 |
| [src/custom-moe-offload.cpp](../../src/custom-moe-offload.cpp) | 核心逻辑 + 两个自定义 ggml 算子 |
| [examples/moe-offload/moe-offload.cpp](moe-offload.cpp) | 基于 `simple.cpp` 的示例: 解析 `-mu/-mo/-mp`, 关闭 mmap, 拒绝空 prompt, 设置 `n_ubatch = n_prompt` |

只有 deepseek2 和 qwen3moe 两个架构接入了本特性.

## 4. 数据结构 (src/custom-moe-offload.h)

### 4.1 `custom_hparams` ([h:80](../../src/custom-moe-offload.h#L80))

| 字段 | 含义 |
|---|---|
| `optimization` | = `moe_enable_optimizations` |
| `utilization` | 池大小比例 |
| `n_prefetch_layers` | prefill 预取层数 |
| `n_preload_layers` | 池能完整放下的层数 = `min(n_slots / n_expert, n_moe_layer)`, 至少为 1 |
| `head_il` / `tail_il` | prefill 时的 "常驻层窗口" `[head_il, tail_il)`, 单位是 MoE 层下标 |
| `n_mean_expert` | `n_slots / n_moe_layer`, 目前没用到 |

**层下标约定**: 表、池、LRU 里的 `il` 都是 "MoE 层下标" = 模型层号 - `n_layer_dense_lead`. DeepSeek-V2-Lite 有 1 个 dense 层, 所以模型第 1 层 = MoE 层 0. Qwen3-MoE 没有 dense 层.

### 4.2 专家池 `custom_expert_pool` ([h:104](../../src/custom-moe-offload.h#L104), 构造函数 [cpp:107-252](../../src/custom-moe-offload.cpp#L107-L252))

- 三个 3D 张量 `pool_ups` / `pool_gates` / `pool_downs`, 形状 `[ne0, ne1, n_slots]`.
- 一个 "slot" = 一个专家的 up+gate+down. 对 up/gate/down 分别在所有 MoE 层中选**字节数最大**的专家张量作为模板, 池的类型和 `nbyte_slot_*` (`size_t`) 都取自它. 其他层的专家较小时, slot 末尾留空.
- `n_slots = utilization * props.memory_total / nbyte_group`. `props` 来自 `model.devices.front()` (GPU 或 iGPU); 没有设备 (纯 CPU 构建) 时用 CPU device, 即系统内存. `n_slots < n_expert` (放不下一整层) 时抛异常.
- buffer 类型: 该设备的 host buffer type (CUDA 下是 CUDA_Host, 即 `cudaMallocHost` 的 pinned 内存), 没有就用 CPU buffer. 如果 CUDA_Host 分配失败, ggml 会退回普通 CPU buffer, 这时打印 warning. buffer usage 设为 `WEIGHTS`, 让调度器把用它的算子当作 "带权重的算子".
- `free_slots`: `std::map<pos, len>` 空闲区间表. `pool_alloc(n)` first-fit; `pool_free` 会合并相邻区间, 但**当前没有任何调用者**. 实际的回收方式是: 淘汰一个专家后直接把它的 `pos` 交给新专家用.

参考值 (`-mu 0.46` 对应约 3.7 GB 池):

| 模型 | MoE 层 x 专家 | 单专家 (up+gate+down, 取最大层) | slot 数 | 可常驻整层数 |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite Q4_K_M | 26 x 64, top-6 | 1.62+1.62+3.06 MB (down: Q8_0) | ~590 | 9 |
| Qwen3-30B-A3B Q4_K_M | 48 x 128, top-8 | 0.84+0.84+1.23 MB (down: Q6_K) | ~1270 | 9 |

### 4.3 专家表 `custom_expert_table` ([h:241](../../src/custom-moe-offload.h#L241), `table_init` [cpp:414](../../src/custom-moe-offload.cpp#L414))

- `experts[n_moe_layer][n_expert]`, 每项 `custom_expert_group`:
  - `up/gate/down.offs`: 该专家在 GGUF 文件中的**绝对偏移** = `weight->offs + j * layer_nbyte`, `idx` 是文件下标 (未使用).
  - `state`: `OnDisk` / `Loading` / `InMemory`.
  - `pos`: 在池中的 slot 下标, `OnDisk` 时为 -1.
- `layer_type[il]`: 每层 up/gate/down 的实际 ggml_type. 混合量化时, `get_ups/get_gates/get_downs` ([cpp:371-412](../../src/custom-moe-offload.cpp#L371-L412)) 返回池张量的 view, 并把 view 的 `type` 和 `nb` 改成该层类型 (slot 步长 `nb[2]` 仍按池类型). 例如 DeepSeek-Lite 的 down 有的层是 Q5_0, 有的层是 Q8_0, 池按 Q8_0 分配.
- `layer_nbyte[il]`: 该层单个专家 up/gate/down 在文件中的精确字节数 (`ggml_nbytes / n_expert`). 每次读盘只读这么多字节, 并以此检查短读.
- `load_ctx[il]` (`LoadContext`, [h:193](../../src/custom-moe-offload.h#L193)): 异步加载的计数. `pending_io[e]` 初始为 3 (up/gate/down 三个读请求), 降到 0 表示该专家加载完成; `finished_expert_count == n_expert` 时 `is_finished = true`. `is_started` 表示这一层的异步加载已经触发.
- `map()/unmap()`: `(row, col, type) <-> io_uring user_data`, 即 `(row * n_col + col) * 3 + type + 1` (加 1 是为了避开 0).
- `refresh(async_io)`: 收割异步 ring 上所有已完成的 CQE, 检查读到的字节数, 更新 `LoadContext`, 专家 3 块都到齐后标记 `InMemory`.

### 4.4 LRU 管理 `custom_expert_manage` ([h:362](../../src/custom-moe-offload.h#L362))

标准 LRU: 一条**全局**链表, 包含池中所有层的专家 (`Loading` 和 `InMemory` 都在里面). 头部 = 最近使用, 尾部 = 最久未使用. `iters[il][id]` 指向链表节点, `end()` 表示不在池中. 所有操作都加锁 (prefill 时所有线程都会调用 `use`).

| 操作 | 调用位置 | 作用 |
|---|---|---|
| `insert` / `insert_layer` | `load_expert` / `load_layer` | 专家拿到 slot (开始加载), 放到头部 |
| `use` | `id_pos_map` | 专家被当前 token 使用, 移到头部 |
| `erase` / `erase_layer` | `_free_expert` / `_free_layer` | 专家离开池 |
| `victim(can_evict)` | `free_lru` | 从尾部往前找第一个 `can_evict` 为真的专家 |

淘汰条件 `can_evict(il, id, cur_il)` ([cpp:550](../../src/custom-moe-offload.cpp#L550)), `free_lru` (`-mo 1`) 和 `free_random` (`-mo 0`) 共用:
- `il != cur_il`: 当前正在计算的层不能淘汰, 因为它的专家可能已经映射给了本 token, 后面的 `mul_mat_id` 还要用.
- `state == InMemory`: 正在加载的专家有读请求在途, 不能淘汰.
- 找不到可淘汰的专家时返回 -1, 调用方触发断言 (池太小).

### 4.5 IO `custom_async_io` ([h:116](../../src/custom-moe-offload.h#L116), [cpp:19-93](../../src/custom-moe-offload.cpp#L19-L93))

- 用 `O_RDONLY` 打开模型文件 (没有 O_DIRECT), 创建**两个** io_uring, 都注册这个 fd (fixed file 下标 0). SQ/CQ/SQE 的 mlock 只是尽力而为, 失败只打印信息.
  - `ring` (深度 640): 只用于异步读 (预取). 只有 `refresh_completions` 收割它的 CQE.
  - `block_ring` (深度 4): 只用于阻塞读. 同一时刻最多只有 1 个请求在途.
- 分开的原因: 如果两种读共用一个 ring, `block_read` 里的 `io_uring_wait_cqe` 会拿到**任意一个** CQE, 可能把异步预取的完成事件吞掉 -> 自己的读还没完成就返回, 被吞掉的预取永远停在 `Loading`.
- `block_read`: 在 `block_ring` 上提交并等待; `res < 0` 或者短读都抛异常.
- `async_submit(idx, ...)`: 加锁, 在 `ring` 上提交, `user_data = idx`.
- `refresh_completions`: 非阻塞收割 `ring` 上所有 CQE, 返回 `{user_data, 读到的字节数}`; `res < 0` 或 `user_data == 0` 时抛异常. 短读由 `table.refresh` 检查.

### 4.6 自定义算子的 userdata `runtime_ctx` ([h:443](../../src/custom-moe-offload.h#L443))

- `{moe_unified, il_moe}`. `custom_moe_unified::rt_ctxs` 在构造函数里一次性创建, 每个 MoE 层一个, 之后不再 resize. 建图时用 `get_runtime_ctx(il_moe)` 取得指针, 传给 `ggml_map_custom*`.
- 生命周期和 `custom_moe_unified` 相同. 算子里**不能 delete** 它: 图会被复用 (decode 的每个 token、reserve 图), 同一个算子节点会执行很多次.

## 5. 生命周期

### 5.1 加载阶段

1. `load_arch_tensors` (deepseek2/qwen3moe) 创建专家张量时带上 `TENSOR_EXPERT_WEIGHT` -> 张量头放在 `moe_table_ctx`, **没有数据**. 非专家权重照常分配和加载.
2. [llama.cpp:372-380](../../src/llama.cpp#L372-L380): 检查 `0 < utilization < 1`, 构造 `custom_moe_unified`:
   - 选出最大的专家张量, 确定池类型和 slot 大小 (见 4.2).
   - `table_init` 填写每个专家的文件偏移、每层类型和字节数.
   - 分配池, 计算 `n_preload_layers`, `head_il = 0`, `tail_il = n_preload_layers`, 创建 LRU 和 `rt_ctxs`.
3. `prefill_init()` ([cpp:268](../../src/custom-moe-offload.cpp#L268)): 对 MoE 层 `0 .. n_preload_layers-1` 各 `pool_alloc(n_expert)` 一段连续 slot, 然后用 `block_read` 阻塞读入整层 (`load_layer(..., io_mode::block)`). 整层的专家放在连续 slot 里, 专家 e 的 slot = 层首 slot + e (不变量 2).

### 5.2 建图阶段: `build_moe_ffn_offload` ([graph.cpp:1637](../../src/llama-graph.cpp#L1637))

与上游 `build_moe_ffn` 的区别:

```
logits = gate_inp x cur -> softmax/sigmoid -> (+exp_probs_b) -> ggml_top_k -> selected_experts   [专家 id]
select_map = ggml_map_custom2(selected_experts, selection_probs, id_pos_map, n_tasks=1, runtime_ctx{moe_unified, il_moe})
                                                                                                   [专家 id -> 池 slot]
(optimization && 有下一层 gate_inp):
    predict: 下一层 gate_inp x 当前 cur -> softmax -> top_k -> ggml_map_custom1_inplace(custom_predict)
up/gate/down = mul_mat_id(pool_ups/gates/downs, cur, select_map)   <- 用 slot 下标而不是专家 id 做索引
weights = get_rows(probs, selected_experts)  (norm + clamp, scale 为 0 或 1 时跳过)
silu(gate) * up -> down -> * weights -> 各专家求和
```

- 关键思路: 自定义算子把专家 id **改写成池中的 slot 下标**, 后续 `mul_mat_id` 直接在池张量上做 gather. 这个算子在 CPU 上执行 (自定义算子只有 CPU 后端支持), 并在执行时完成缺失专家的加载. 也就是说, "加载" 是在图执行过程中, 在 `mul_mat_id` 之前同步完成的.
- `selection_probs` 作为第二个输入只是为了建立图依赖, 算子不读它.
- "预测" 用**下一层**的 router 权重乘当前层的输入, 近似得到下一层会选的专家.
- 与上游相比缺少: expert group 选择 (`n_expert_groups > 1`), `SOFTMAX_WEIGHT` 等门控, `gate_up_exps` 合并张量, 专家 bias/scale (`*_exps_s`), `swiglu_clamp`, 以及 `ggml_swiglu_split` 融合. 目标模型 (deepseek2 / qwen3moe 的常见 GGUF) 用不到这些.

### 5.3 执行阶段: `id_pos_map` ([cpp:687](../../src/custom-moe-offload.cpp#L687))

`prefill = (n_tokens > 1)`.

**线程模型**: 当前 ggml-cpu 中, 线程池里的**所有线程都会执行每个节点** ([ggml-cpu.c](../../ggml/src/ggml-cpu/ggml-cpu.c) 中的 `ggml_graph_compute_thread`). 传给 `ggml_map_custom*` 的 `n_tasks = 1` 只影响规划阶段, 执行时 `nth` = 线程数, 每个算子要自己按 `ith` 分工. 节点之间有 barrier.
- prefill: 第一个拿到 `CallSync` 锁的线程执行 `check_layer`, 其他线程等它完成; 然后按 token 分段并行做映射; 最后一个线程在 `mark_completion` 中重置 `CallSync` (它是 `static` 的, 被所有层共用).
- decode 和 `custom_predict`: 只有 `ith == 0` 干活, 其他线程直接返回.

**Prefill (按层整层加载)**

1. `check_layer(il)` ([cpp:278](../../src/custom-moe-offload.cpp#L278)):
   - `layer_check(il)` (即 `load_ctx.is_finished`) 为真 -> 认为整层已在内存, 直接返回.
   - 已触发异步加载 (`is_started`) -> 忙等 `refresh` 直到完成, 然后标记 `InMemory`.
   - 否则阻塞加载: 把 `head_il` 层的 slot 段 (`table.at(head_il, 0).pos` 开始) 让给 `il`, `_free_layer(head_il)`, `load_layer(il, pos, block)`, 然后 `head_il++`, `tail_il++`. 即一个 "滑动窗口": 窗口内保持 `n_preload_layers` 个整层.
2. 把每个 token 的专家 id 映射成 slot.
3. prompt 最后 30 个 token (不足 30 时为全部 token) 选中的专家调用 `use`, 让 decode 开始时 LRU 里这些专家最 "新". 被窗口换出的层会通过 `erase_layer` 从 LRU 中移除.

**Decode (按专家加载, 单线程)**

对每个选中的专家:
1. `check_expert(il, id)` ([cpp:317](../../src/custom-moe-offload.cpp#L317)):
   - `OnDisk`: 先 `pool_alloc(1)`; 没有空闲 slot 时淘汰一个专家: `optimization ? free_lru(il) : free_random(il)`, 然后用 `block_ring` **阻塞**读入.
   - `Loading` (由预测触发): 忙等 `refresh` 直到 `InMemory`.
2. `use(il, id)` 更新 LRU.
3. 写入 slot 下标.

**Predict (`custom_predict`, [cpp:742](../../src/custom-moe-offload.cpp#L742), 仅 `optimization=1`)**

- Decode: 按 top-k 顺序找到下一层第一个 `OnDisk` 的预测专家, 分配 slot 或淘汰一个专家 (`free_lru(il)`, 保护当前层 il) 后**异步**读入, 然后停止. 每层最多预取 1 个专家. 预取的专家在 `load_expert` 中插入 LRU 头部.
- Prefill: 跳过 MoE 层 0 和最后两层; 如果不是全部常驻, `head_il < il` (窗口头不是当前层), 并且 `il <= n_prefetch_layers` 或 `il == tail_il - 1`, 就 `predict_layer(head_il, tail_il)`: 释放窗口头层, 把它的 slot 段用于**异步**加载 `tail_il` 层, 然后窗口右移. 这样下一层的 `check_layer` 会走 "已触发, 等待完成" 分支. 窗口只有 1 层时不预取, 否则会释放正在计算的层.

## 6. Jetson 统一内存 (CUDA 构建)

纯 CPU 构建时, 所有权重、专家池和 KV cache 都在系统内存中, 计算全部在 CPU 上, 本节不适用.

### 6.1 设计目标

专家池必须**同时**被 CPU 和 iGPU 直接访问, 不能有拷贝:
- CPU: io_uring 把专家从 SSD 读进池; 自定义算子 (CPU) 维护映射.
- iGPU: `mul_mat_id` 直接读池里的专家.

做法: 池分配为 iGPU 设备的 host buffer type, 即 CUDA_Host (`cudaMallocHost` pinned 内存, 在 Linux 64 位 UVA 下 CPU 和 GPU 使用同一个指针). 物理上它就是同一块 DDR.

### 6.2 需要满足的条件和对应改动

1. **找到 iGPU 设备**. Jetson 的 CUDA 设备类型是 `GGML_BACKEND_DEVICE_TYPE_IGPU` ([ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu) `ggml_backend_cuda_device_get_type`), `ggml_backend_dev_by_type(GPU)` 找不到它. 以前这会导致: 池退回普通 CPU 内存; 层的 device 为 NULL, 分配 KV cache 时崩溃. 现在两处都改用 `model.devices.front()` (llama 在没有独显时会把 iGPU 放进 `devices`).
2. **调度器把用到池的算子放到 iGPU 上, 并且不拷贝**. 调度器用 `ggml_backend_sched_backend_from_buffer` 选择第一个 `supports_buft(buffer) && supports_op` 的后端. CUDA 的 `supports_buft` 只有在 `integrated == true` 时才接受 CUDA_Host. 上游把 `integrated` 强制设成 false (#15034), 导致:
   - batch 较小时 (decode), 使用池的 `MUL_MAT_ID` 在 CPU 上执行;
   - batch >= 32 时 (prefill), `op_offload` 把算子放到 GPU, 并且把**整个池**拷贝进 GPU 内存 (原 P5), 内存翻倍.
   
   改动: 在 `#ifdef CUSTOM_MOE` 下恢复 `integrated = prop.integrated` ([ggml-cuda.cu:308-313](../../ggml/src/ggml-cuda/ggml-cuda.cu#L308-L313)). `integrated` 只影响 `supports_buft` 和 Debug 构建中的几条断言, 对独显 (`prop.integrated == 0`) 没有影响.
3. **非专家权重**放在 CUDA_Host (`cpu_buft_list` 的第一项), `integrated == true` 时同样由 iGPU 直接计算. KV cache 和计算 buffer 在 CUDA 设备内存中 (在 Jetson 上也是同一块 DDR).

### 6.3 CPU 写入和 GPU 读取的顺序

每层的图被切成 `GPU (router, top_k) -> CPU (id_pos_map, custom_predict) -> GPU (mul_mat_id ...)`. CPU 段开始前, 调度器要把 top_k 结果拷到 CPU, 这时 GPU 已经执行完前面所有的 kernel. 所以:
- `check_expert` 淘汰的 slot 只属于前面的层 (`can_evict` 排除当前层), 它们的 kernel 已经执行完, 可以安全覆盖.
- 预取 (异步读) 写入的 slot, 要么是空闲 slot, 要么是被淘汰的前面层的 slot, 不会和当前层的 `mul_mat_id` 冲突. 使用前 `check_expert` 会等待读完成.

### 6.4 风险与待验证

- 上游关闭 `integrated` 的原因是 #15034 中的 "输出错误" (我没能查到 issue 的具体内容). 在 Jetson 上需要对比 `ENABLE_CUSTOM_MOE=OFF` 的输出来确认.
- 只在独显机器上做过编译检查 (`build-cuda-moe`), 还没有在 Jetson 上运行过.
- `GGML_CUDA_NO_PINNED` 或 `cudaMallocHost` 失败时池会退回 CPU 内存, 加载时打印 warning, 这时又会出现拷贝.
- Jetson preset 关闭了 CUDA graphs (`GGML_CUDA_GRAPHS=OFF`). 每层都有 CPU 段, CUDA graph 本来也用不上.

## 7. 不变量 (排查问题时先检查这些)

1. `state == InMemory/Loading` <=> `pos >= 0` <=> 该专家在 LRU 链表中.
2. 被 `check_layer` / `predict_layer` 复用的层 (`head_il`) 必须**整层在内存, 且 slot 连续**: 专家 e 的 slot = `table.at(head_il, 0).pos + e`. decode 的单专家淘汰会破坏这一点, 所以 decode 之后不能再 prefill.
3. `load_ctx[il].is_finished == true` 应当表示 "整层都在内存". 当前实现只置位, 从不清零, 所以每层最多只能经历一次 prefill (示例通过 `n_ubatch = n_prompt` 保证).
4. 自定义算子在 CPU 线程池里执行, **所有线程都会进入** (见 5.3 线程模型). 新增的状态修改要么只让 `ith == 0` 做, 要么加锁.
5. 池张量在图里的地址固定, 所以图拓扑和池内容无关, 图可以被复用. 被固化在算子参数里的 `runtime_ctx` 指针必须在整个模型生命周期内有效 (见 4.6).
6. 池内的字节偏移一律用 `size_t` 计算. 读盘长度 = `layer_nbyte` (<= `nbyte_slot_*`).
7. 阻塞读只走 `block_ring`, 异步读只走 `ring`. 不要在 `ring` 上等待 CQE, 只能通过 `refresh` 收割 (见 4.5).
8. 当前层的专家和正在加载的专家不能被淘汰 (`can_evict`).

## 8. 未解决的问题

- Jetson 路径 (第 6 节) 还没有运行验证, 见 6.4.
- 已知限制 (不在目标场景内, 暂不处理): 多次 prefill (不变量 2、3); 分片 GGUF (只打开一个 fd, 忽略 `custom_tensor_mmap::idx`); 合并的 `gate_up_exps` 张量; deepseek2 的 MTP 层 (专家张量的 flag 被替换成 `TENSOR_EXPERT_WEIGHT`, 丢掉了 `TENSOR_SKIP`); 专家张量没有 buffer, 依赖 `tensor->buffer` 的上游功能 (memory breakdown, `common_fit_params`, `llama-moe-cache`) 行为未知.

## 9. 变更记录

- 2026-10-14: 修复 P1: 图复用导致 `runtime_ctx` 被提前 delete. 改为由 `custom_moe_unified::rt_ctxs` 持有.
- 2026-10-14: 修复 P2: slot 地址计算 32 位溢出. `nbyte_slot_*` 改为 `size_t`.
- 2026-10-14: 规避 P3 (多次 prefill 时 `LoadContext` 不清零): 示例设置 `n_ubatch = n_prompt`.
- 2026-10-14: 修复 P4 (阻塞读和异步读共用 io_uring): 增加专用的 `block_ring`; `block_read` 检查短读; `io_uring_register_files` 检查返回值; mlock 失败不再中止加载.
- 2026-10-14: 当前开发环境改为纯 CPU 构建 `build-cpu-debug`. Qwen `-mo 1` decode 卡住的原因是旧二进制不包含 P4 的修复, 重新编译后已解决.
- 2026-10-15: P5/P6 (Jetson): 池和层的 device 改用 `model.devices.front()`, 能找到 IGPU; CUSTOM_MOE 下恢复 ggml-cuda 的 `integrated` 检测, 让 iGPU 直接在 CUDA_Host 池上计算, 不再拷贝整个池. 只做了编译检查.
- 2026-10-15: P7: 去掉 `> 30 token` 断言, 预热改为 prompt 的最后 30 个 token (不足 30 时为全部); 示例不再检查 prompt 长度. P8: 删除 (不处理).
- 2026-10-15: P9 硬性错误: `GGML_ASSERT(a = b)` 改为比较; `total_size()` 改为 `size_t`; slot 大小改为按所有层中最大的专家计算; 每次读盘只读该层专家的实际字节数, 异步读也检查短读; 权重 scale 为 0 或 1 时跳过, `weights_sum` 加 clamp (与上游一致); `free_random` 不再死循环; 池放不下一整层时报错; prefill 窗口只有 1 层时不再预取 (原来会释放正在计算的层).
- 2026-10-15: LRU 改为标准 LRU (全局链表, 加载时插入, 使用时移到头部, 淘汰时从尾部取). 淘汰时跳过当前层和正在加载的专家, 修复了 "淘汰正在加载的专家触发断言" 和 `get_back_n` 在空链表上的未定义行为. 去掉了按 router 概率调整位置的逻辑.
- 2026-10-15: 示例在 prompt 为空时直接报错. 原来空 prompt 在没有 BOS 的模型 (Qwen3) 上得到 0 个 token, 导致 `n_batch = n_ubatch = 0`, 只报 "n_batch and n_ubatch cannot both be zero". 实际遇到的原因是 shell 里的 `$P` 没有设置.
