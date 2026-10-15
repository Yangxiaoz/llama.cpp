# CUSTOM_MOE: MoE expert offload to disk

本文记录 `CUSTOM_MOE` 特性的实现细节与代码位置, 用于后续排查问题. 所有改动都用 `#ifdef CUSTOM_MOE` 包裹, 关闭宏即恢复上游代码 (例外见 "代码地图" 中标注的无条件改动).

## 1. 目标与场景

- 场景: 内存 (CPU DDR, 或 Jetson 这类统一内存的 "显存") 装不下整个 MoE 模型. 非专家权重常驻内存, **专家权重 (ffn_up/gate/down_exps) 留在 SSD 上的 GGUF 文件里**, 运行时按需读入一个固定大小的 "专家池" (expert pool).
- 与上游的 `llama-moe-cache` / `-ot` / `--cpu-moe` 等机制无关. 上游机制解决的是 "显存放不下 -> 放到 DDR", 本特性解决的是 "DDR 放不下 -> 放到磁盘".
- 读盘使用 `io_uring` (liburing), 支持阻塞读和异步读 (预取).
- 目标使用场景 (当前代码只保证这个场景): 单用户, 单个 prompt, 一次 prefill, 之后逐 token decode, 不会有第二次 prefill.

## 2. 编译与运行

开发分两个阶段:
1. 当前阶段: 在笔记本上用纯 CPU 构建 `build-cpu-debug` (preset `cpu-debug`, Debug, GGML_CUDA=OFF), 硬件拓扑是 DDR 内存 + SSD. 笔记本的独显 (4060) 属于 "CPU + 独立 GPU" 结构, 不在本特性的讨论范围内, 不用 CUDA 构建.
2. 之后的阶段: Jetson 统一内存 (CUDA 构建, 但物理上仍是 DDR + SSD). 第 6 节和 P5/P6 属于这个阶段.

```bash
cmake --preset cpu-debug                       # 生成 build-cpu-debug
cmake --build build-cpu-debug --target llama-moe-offload -j
build-cpu-debug/bin/llama-moe-offload -m model.gguf -n 32 [-mu 0.2] [-mo 0|1] [-mp 1] '<prompt>'
```

- prompt 建议用 ASCII 单引号括起来: 用双引号时 shell 会展开 `$2000` 这类文本; 用中文引号时 shell 根本不把它当引号.
- 修改源码后**必须重新编译** `build-cpu-debug`, 再进行测试.
- CPU 构建下, 池大小 = `utilization x 系统内存总量` (CPU device 的 `memory_total`). 例如 31 GB 内存, `-mu 0.2` 约为 6.2 GB.

- `ENABLE_CUSTOM_MOE=ON` -> `add_compile_definitions(CUSTOM_MOE)` (全局, 所有 target 都带这个宏).
- 依赖 liburing (`sudo apt install liburing-dev`), 找不到直接 FATAL_ERROR, **即使 ENABLE_CUSTOM_MOE=OFF 也会检查** (见 P8).
- 示例参数 (只在 `examples/moe-offload` 里解析):
  - `-mu F`: 专家池占 "设备总内存" 的比例, 0 < F < 1, 默认 0.46 (`llama_model_default_params()` 里的默认值是 0.3).
  - `-mo 0|1`: `moe_enable_optimizations`. 1 = LRU 淘汰 + 下一层专家预测/预取; 0 = 随机淘汰, 不预取.
  - `-mp N`: `n_prefetch_layers`, prefill 时向前预取的层数.
- 示例在 CUSTOM_MOE 下固定的设置:
  - `load_mode = LLAMA_LOAD_MODE_NONE` (不用 mmap).
  - `n_ubatch = n_batch = n_prompt`: prompt 只占一个 ubatch, 只做一次 prefill. 原因是 `LoadContext.is_finished` 只置位不清零 (见不变量 3): 第二次 prefill 时已经被换出的层仍会被当成常驻层, 映射到 slot -1. 另外 decode 的单专家淘汰会打散常驻层 (见不变量 2). 将来要支持长 prompt 或多轮对话, 必须先重新设计这一部分.
  - prompt 必须超过 30 个 token, 否则示例直接报错退出 (见 P7).
- 运行时约束:
  - 一旦编译进 CUSTOM_MOE, **每次加载模型都走 offload 路径**, 没有运行时开关. 不是 deepseek2 / qwen3moe 的模型会在 `custom_moe_unified` 构造函数里解引用空的 `ffn_up_exps`.
  - 2..30 token 的 batch 会触发断言 (P7), 所以 `llama-cli` / `llama-server` (warmup 用 2 个 token) 无法使用.
  - 只支持单文件 GGUF (见 P9).

## 3. 代码地图

| 位置 | 作用 |
|---|---|
| [CMakeLists.txt:116-122](../../CMakeLists.txt#L116-L122) | `ENABLE_CUSTOM_MOE` 选项 -> 宏 `CUSTOM_MOE` |
| [CMakePresets.json](../../CMakePresets.json) | 新增 `linux-base` / `cuda` / `jetson` / `cpu-debug` preset (无宏保护) |
| [src/CMakeLists.txt:7-15](../../src/CMakeLists.txt#L7-L15), [79-90](../../src/CMakeLists.txt#L79-L90) | 查找并链接 liburing, 定义 `LLAMA_USE_LIBURING` (无宏保护) |
| [src/CMakeLists.txt:53](../../src/CMakeLists.txt#L53) | 把 `custom-moe-offload.cpp` 加入 libllama (无宏保护, 文件本身也没有 `#ifdef`) |
| [include/llama.h:349-353](../../include/llama.h#L349-L353) | `llama_model_params` 新增 `moe_memory_utilization` / `moe_enable_optimizations` / `n_prefetch_layers` |
| [src/llama-model.cpp:3363-3367](../../src/llama-model.cpp#L3363-L3367) | 上述三个参数的默认值 (0.3 / false / 1) |
| [src/llama-model-loader.h:73-76](../../src/llama-model-loader.h#L73-L76) | 新 flag `TENSOR_EXPERT_WEIGHT = 1 << 7`, 成员 `moe_table_ctx` |
| [src/llama-model-loader.cpp:1367-1378](../../src/llama-model-loader.cpp#L1367-L1378) | `create_tensor`: 带 `TENSOR_EXPERT_WEIGHT` 的张量放进 `moe_table_ctx`, 不进 `ctx_map`, 所以**不分配 buffer, 也不会被 `load_all_data` 读入** |
| [src/llama-model.cpp:1657-1661](../../src/llama-model.cpp#L1657-L1661) | `get_layer_buft_list`: GPU 层的 device 设为第一个 GPU, 但 buft list 用 `cpu_buft_list` -> 非专家权重放在主机内存 (CUDA_Host pinned), KV cache 放 GPU |
| [src/llama-model.cpp:1681-1697](../../src/llama-model.cpp#L1681-L1697) | 创建 `moe_table_ctx` (no_alloc, 每个 MoE 层 3 个张量头) |
| [src/llama-model.cpp:1870-1872](../../src/llama-model.cpp#L1870-L1872) | `model.table_ctx` 接管 `moe_table_ctx` 的所有权 |
| [src/llama-model.h:17-19](../../src/llama-model.h#L17-L19), [772-775](../../src/llama-model.h#L772-L775) | `llama_model::table_ctx`, `llama_model::moe_unified` |
| [src/llama.cpp:372-380](../../src/llama.cpp#L372-L380) | `load_tensors` 之后构造 `custom_moe_unified` 并调用 `prefill_init()` |
| [src/llama-context.cpp:2522-2526](../../src/llama-context.cpp#L2522-L2526) | `graph_max_nodes`: 节点上限 8x -> 10x `n_tensors` (注意是直接 `return`) |
| [src/llama-graph.h:26-28](../../src/llama-graph.h#L26-L28), [1093-1120](../../src/llama-graph.h#L1093-L1120) | `build_moe_ffn_predict` / `build_moe_ffn_offload` 声明 |
| [src/llama-graph.cpp:1582](../../src/llama-graph.cpp#L1582) | `build_moe_ffn_predict` (当前没有调用者) |
| [src/llama-graph.cpp:1637](../../src/llama-graph.cpp#L1637) | `build_moe_ffn_offload`: 替代上游 `build_moe_ffn` 的建图函数 |
| [src/models/deepseek2.cpp:136-143](../../src/models/deepseek2.cpp#L136-L143), [655-675](../../src/models/deepseek2.cpp#L655-L675) | deepseek2: 专家张量加 flag; MoE 分支改调 `build_moe_ffn_offload` |
| [src/models/qwen3moe.cpp:51-60](../../src/models/qwen3moe.cpp#L51-L60), [142-162](../../src/models/qwen3moe.cpp#L142-L162) | qwen3moe: 同上 |
| [src/custom-moe-offload.h](../../src/custom-moe-offload.h) | 全部数据结构 + 大量 inline 实现 |
| [src/custom-moe-offload.cpp](../../src/custom-moe-offload.cpp) | 核心逻辑 + 两个自定义 ggml 算子 |
| [examples/moe-offload/moe-offload.cpp](moe-offload.cpp) | 基于 `simple.cpp` 的示例: 解析 `-mu/-mo/-mp`, 关闭 mmap, 设置 `n_ubatch = n_prompt`, 检查 prompt > 30 token |

只有 deepseek2 和 qwen3moe 两个架构接入了本特性.

## 4. 数据结构 (src/custom-moe-offload.h)

### 4.1 `custom_hparams` ([h:80](../../src/custom-moe-offload.h#L80))

| 字段 | 含义 |
|---|---|
| `optimization` | = `moe_enable_optimizations` |
| `utilization` | 池大小比例 |
| `n_prefetch_layers` | prefill 预取层数 |
| `n_preload_layers` | 池能完整放下的层数 = `min(n_slots / n_expert, n_moe_layer)` |
| `head_il` / `tail_il` | prefill 时的 "常驻层窗口" `[head_il, tail_il)`, 单位是 MoE 层下标 |
| `n_mean_expert` | `n_slots / n_moe_layer`, 目前没用到 |

**层下标约定**: 表和池里的 `il` 都是 "MoE 层下标" = 模型层号 - `n_layer_dense_lead`. DeepSeek-V2-Lite 有 1 个 dense 层, 所以模型第 1 层 = MoE 层 0. Qwen3-MoE 没有 dense 层.

### 4.2 专家池 `custom_expert_pool` ([h:104](../../src/custom-moe-offload.h#L104))

- 三个 3D 张量 `pool_ups` / `pool_gates` / `pool_downs`, 形状 `[ne0, ne1, n_slots]`, 类型取自**最后一层**的专家张量 (`model.layers.back()`).
- 一个 "slot" = 一个专家的 up+gate+down. `nbyte_slot_*` (`size_t`) = 最后一层张量字节数 / `n_expert`.
- `n_slots = utilization * props.memory_total / nbyte_group`, 其中 `props` 是**第一个 GPU** 的属性 (没有 GPU 时用 CPU device, 即系统内存).
- buffer 类型: `ggml_backend_dev_host_buffer_type(dev)` (CUDA 下是 pinned host 内存), 没有就用 CPU buffer. buffer usage 设为 `WEIGHTS`, 让调度器把用它的算子当作 "带权重的算子".
- `free_slots`: `std::map<pos, len>` 空闲区间表. `pool_alloc(n)` first-fit; `pool_free` 会合并相邻区间, 但**当前没有任何调用者**. 实际的回收方式是: 淘汰一个专家后直接把它的 `pos` 交给新专家用.

参考值 (8 GB 显存, `-mu 0.46`, 约 3.7 GB 池):

| 模型 | MoE 层 x 专家 | 单专家 (up+gate+down) | slot 数 | 可常驻整层数 |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite Q4_K_M | 26 x 64, top-6 | 1.62+1.62+3.06 MB (down 取最后一层 Q8_0) | ~590 | 9 |
| Qwen3-30B-A3B Q4_K_M | 48 x 128, top-8 | 0.84+0.84+1.23 MB (down 取最后一层 Q6_K) | ~1270 | 9 |

### 4.3 专家表 `custom_expert_table` ([h:241](../../src/custom-moe-offload.h#L241))

- `experts[n_moe_layer][n_expert]`, 每项 `custom_expert_group`:
  - `up/gate/down.offs`: 该专家在 GGUF 文件中的**绝对偏移** (`weight->offs + j * 单专家字节数`, 按该层实际类型计算), `idx` 是文件下标 (未使用).
  - `state`: `OnDisk` / `Loading` / `InMemory`.
  - `pos`: 在池中的 slot 下标, `OnDisk` 时为 -1.
- `layer_type[il]`: 每层 up/gate/down 的实际 ggml_type. 混合量化时, `get_ups/get_gates/get_downs` ([cpp:358-399](../../src/custom-moe-offload.cpp#L358-L399)) 返回池张量的 view, 并把 view 的 `type` 和 `nb` 改成该层类型 (slot 步长 `nb[2]` 仍按池类型). 例如 DeepSeek-Lite 的 down 有的层是 Q5_0, 有的层是 Q8_0, 池按 Q8_0 分配.
- `load_ctx[il]` (`LoadContext`, [h:193](../../src/custom-moe-offload.h#L193)): 异步加载的计数. `pending_io[e]` 初始为 3 (up/gate/down 三个读请求), 降到 0 表示该专家加载完成; `finished_expert_count == n_expert` 时 `is_finished = true`. `is_started` 表示这一层的异步加载已经触发.
- `map()/unmap()`: `(row, col, type) <-> io_uring user_data`, 即 `(row * n_col + col) * 3 + type + 1` (加 1 是为了避开 0).
- `refresh(async_io)`: 收割异步 ring 上所有已完成的 CQE, 更新 `LoadContext`, 专家 3 块都到齐后标记 `InMemory`.

### 4.4 LRU 管理 `custom_expert_manage` ([h:358](../../src/custom-moe-offload.h#L358))

- 每层一个 `std::list<expert_id>`, 链表**头部 = 最热**, 尾部 = 最冷; `layer_iters` 用于 O(1) 定位; `n_active[il]` = 该层在链表里的专家数.
- `hit(il, id, prob)`: prob >= 0.1 移到头部; >= 0.008 前移 20 位; 否则前移 10 位. 不在链表里的先 `active` (插到尾部) 再前移 10 位.
- `touch()`: 预测加载时把专家插到**尾部**.
- `get_freeLayer_id(cur, load)`: 从 `cur-1` 往前绕一圈 (不含 `cur` 和 `load`), 选 `n_active` 最大的层.
- `get_back_n(il, 0)`: 返回该层最冷的专家.

### 4.5 IO `custom_async_io` ([h:116](../../src/custom-moe-offload.h#L116), [cpp:19-96](../../src/custom-moe-offload.cpp#L19-L96))

- 用 `O_RDONLY` 打开模型文件 (没有 O_DIRECT), 创建**两个** io_uring, 都注册这个 fd (fixed file 下标 0). SQ/CQ/SQE 的 mlock 只是尽力而为, 失败只打印信息.
  - `ring` (深度 640): 只用于异步读 (预取). 只有 `refresh_completions` 收割它的 CQE.
  - `block_ring` (深度 4): 只用于阻塞读. 同一时刻最多只有 1 个请求在途.
- 分开的原因: 如果两种读共用一个 ring, `block_read` 里的 `io_uring_wait_cqe` 会拿到**任意一个** CQE, 可能把异步预取的完成事件吞掉 -> 自己的读还没完成就返回, 被吞掉的预取永远停在 `Loading`. 另外, 阻塞读不设置 `user_data`, liburing 2.5 的 `io_uring_prep_rw` 也不会清零它, 它的 CQE 会被 `refresh` 当成别的专家的完成事件.
- `block_read`: 在 `block_ring` 上提交并等待; `res < 0` 或者短读都抛异常.
- `async_submit(idx, ...)`: 加锁, 在 `ring` 上提交, `user_data = idx`.
- `refresh_completions`: 非阻塞收割 `ring` 上所有 CQE, 返回 user_data 列表; `res < 0` 或 `user_data == 0` 时抛异常.

### 4.6 自定义算子的 userdata `runtime_ctx` ([h:526](../../src/custom-moe-offload.h#L526))

- `{moe_unified, il_moe}`. `custom_moe_unified::rt_ctxs` 在构造函数里一次性创建, 每个 MoE 层一个, 之后不再 resize. 建图时用 `get_runtime_ctx(il_moe)` 取得指针, 传给 `ggml_map_custom*`.
- 生命周期和 `custom_moe_unified` 相同. 算子里**不能 delete** 它: 图会被复用 (decode 的每个 token、reserve 图), 同一个算子节点会执行很多次.

## 5. 生命周期

### 5.1 加载阶段

1. `load_arch_tensors` (deepseek2/qwen3moe) 创建专家张量时带上 `TENSOR_EXPERT_WEIGHT` -> 张量头放在 `moe_table_ctx`, **没有数据**. 非专家权重照常分配和加载.
2. [llama.cpp:372-380](../../src/llama.cpp#L372-L380): 检查 `0 < utilization < 1`, 构造 `custom_moe_unified`:
   - 以最后一层专家张量为模板确定池类型和 slot 大小 ([cpp:110-233](../../src/custom-moe-offload.cpp#L110-L233)).
   - `table_init` 填写每个专家的文件偏移和每层类型 ([cpp:401](../../src/custom-moe-offload.cpp#L401)).
   - 分配池, 计算 `n_preload_layers`, `head_il = 0`, `tail_il = n_preload_layers`, 创建 `rt_ctxs`.
3. `prefill_init()` ([cpp:249](../../src/custom-moe-offload.cpp#L249)): 对 MoE 层 `0 .. n_preload_layers-1` 各 `pool_alloc(n_expert)` 一段连续 slot, 然后用 `block_read` 阻塞读入整层 (`load_layer(..., io_mode::block)`). 整层的专家放在连续 slot 里, 专家 e 的 slot = 层首 slot + e (不变量 2).

### 5.2 建图阶段: `build_moe_ffn_offload` ([graph.cpp:1637](../../src/llama-graph.cpp#L1637))

与上游 `build_moe_ffn` 的区别:

```
logits = gate_inp x cur -> softmax/sigmoid -> (+exp_probs_b) -> ggml_top_k -> selected_experts   [专家 id]
select_map = ggml_map_custom2(selected_experts, selection_probs, id_pos_map, n_tasks=1, runtime_ctx{moe_unified, il_moe})
                                                                                                   [专家 id -> 池 slot]
(optimization && 有下一层 gate_inp):
    predict: 下一层 gate_inp x 当前 cur -> softmax -> top_k -> ggml_map_custom1_inplace(custom_predict)
up/gate/down = mul_mat_id(pool_ups/gates/downs, cur, select_map)   <- 用 slot 下标而不是专家 id 做索引
weights = get_rows(probs, selected_experts)  (norm / scale)
silu(gate) * up -> down -> * weights -> 各专家求和
```

- 关键思路: 自定义算子把专家 id **改写成池中的 slot 下标**, 后续 `mul_mat_id` 直接在池张量上做 gather. 这个算子在 CPU 上执行 (自定义算子只有 CPU 后端支持), 并在执行时完成缺失专家的加载. 也就是说, "加载" 是在图执行过程中, 在 `mul_mat_id` 之前同步完成的.
- `runtime_ctx` 由 `custom_moe_unified` 持有 (见 4.6), 建图时只取指针.
- "预测" 用**下一层**的 router 权重乘当前层的输入, 近似得到下一层会选的专家.
- 与上游相比缺少: expert group 选择 (`n_expert_groups > 1`), `weights_sum` 的 clamp, `SOFTMAX_WEIGHT` 等门控, `gate_up_exps` 合并张量, 专家 bias/scale (`*_exps_s`), `swiglu_clamp`, 以及 `ggml_swiglu_split` 融合. deepseek2 与 qwen3moe 的常见 GGUF 用不到这些.

### 5.3 执行阶段: `id_pos_map` ([cpp:677](../../src/custom-moe-offload.cpp#L677))

`prefill = (n_tokens > 1)`.

**线程模型**: 当前 ggml-cpu 中, 线程池里的**所有线程都会执行每个节点** ([ggml-cpu.c](../../ggml/src/ggml-cpu/ggml-cpu.c) 中的 `ggml_graph_compute_thread`). 传给 `ggml_map_custom*` 的 `n_tasks = 1` 只影响规划阶段, 执行时 `nth` = 线程数 (默认 4), 每个算子要自己按 `ith` 分工. 节点之间有 barrier.
- prefill: 第一个拿到 `CallSync` 锁的线程执行 `check_layer`, 其他线程等它完成; 然后按 token 分段并行做映射; 最后一个线程在 `mark_completion` 中重置 `CallSync` (它是 `static` 的, 被所有层共用).
- decode 和 `custom_predict`: 只有 `ith == 0` 干活, 其他线程直接返回.

**Prefill (按层整层加载)**

1. `check_layer(il)` ([cpp:259](../../src/custom-moe-offload.cpp#L259)):
   - `layer_check(il)` (即 `load_ctx.is_finished`) 为真 -> 认为整层已在内存, 直接返回.
   - 已触发异步加载 (`is_started`) -> 忙等 `refresh` 直到完成, 然后标记 `InMemory`.
   - 否则阻塞加载: 把 `head_il` 层的 slot 段 (`table.at(head_il, 0).pos` 开始) 让给 `il`, `_free_layer(head_il)`, `load_layer(il, pos, block)`, 然后 `head_il++`, `tail_il++`. 即一个 "滑动窗口": 窗口内保持 `n_preload_layers` 个整层.
2. 把每个 token 的专家 id 映射成 slot.
3. 对最后几层 (`il > warmup_layer`), ubatch 的最后 30 个 token 调用 `hit(il, id, 1)`, 用来预热 LRU.

**Decode (按专家加载, 单线程)**

对每个选中的专家:
1. `check_expert(il, id)` ([cpp:298](../../src/custom-moe-offload.cpp#L298)):
   - `OnDisk`: 先 `pool_alloc(1)`; 没有空闲 slot 时淘汰一个专家: `optimization ? free_lru(il, il+1) : free_random({il, il+1})`, 然后用 `block_ring` **阻塞**读入.
   - `Loading` (由预测触发): 忙等 `refresh` 直到 `InMemory`.
2. `hit(il, id, prob)` 更新 LRU.
3. 写入 slot 下标.

**Predict (`custom_predict`, [cpp:733](../../src/custom-moe-offload.cpp#L733), 仅 `optimization=1`)**

- Decode: 按 top-k 顺序找到下一层第一个 `OnDisk` 的预测专家, 分配或淘汰 (`free_lru(il, il+1)`) 一个 slot 后**异步**读入, `touch` 进 LRU, 然后停止. 每层最多预取 1 个专家.
- Prefill: 跳过 MoE 层 0 和最后两层; 如果不是全部常驻, 且 `il <= n_prefetch_layers` 或 `il == tail_il - 1`, 就 `predict_layer(head_il, tail_il)`: 释放窗口头层, 把它的 slot 段用于**异步**加载 `tail_il` 层, 然后窗口右移. 这样下一层的 `check_layer` 会走 "已触发, 等待完成" 分支.

## 6. 内存与后端放置 (CUDA 构建, Jetson 阶段再处理)

纯 CPU 构建时, 所有权重、专家池和 KV cache 都在系统内存中, 计算全部在 CPU 上, 不存在本节的问题.

- 非专家权重: `cpu_buft_list` 的第一项, 一般是 CUDA_Host (pinned).
- 专家池: GPU device 的 host buffer (CUDA_Host pinned), 大小 = `utilization x GPU 总显存`.
- KV cache: GPU 显存 (`dev_layer` = GPU, `offload_kqv` 默认开启).
- 调度: 权重在 host buffer 上的算子默认在 CPU 上执行; batch >= `op_offload_min_batch_size` (默认 32) 的 `MUL_MAT_ID` 会被 offload 到 GPU, **这时整个池张量会被拷贝到显存** (见 P5).
- 当前 ggml-cuda 中 `integrated` 被强制设为 false ([ggml-cuda.cu:308](../../ggml/src/ggml-cuda/ggml-cuda.cu#L308)), 所以 Jetson 上 CUDA_Host 也**不会**被当作零拷贝内存.

## 7. 不变量 (排查问题时先检查这些)

1. `state == InMemory/Loading` <=> `pos >= 0`, 且该专家在 LRU 链表中.
2. 被 `check_layer` / `predict_layer` 复用的层 (`head_il`) 必须**整层在内存, 且 slot 连续**: 专家 e 的 slot = `table.at(head_il, 0).pos + e`. decode 的单专家淘汰会破坏这一点, 所以 decode 之后不能再 prefill.
3. `load_ctx[il].is_finished == true` 应当表示 "整层都在内存". 当前实现只置位, 从不清零, 所以每层最多只能经历一次 prefill (示例通过 `n_ubatch = n_prompt` 保证).
4. 自定义算子在 CPU 线程池里执行, **所有线程都会进入** (见 5.3 线程模型). 新增的状态修改要么只让 `ith == 0` 做, 要么加锁.
5. 池张量在图里的地址固定, 所以图拓扑和池内容无关, 图可以被复用. 被固化在算子参数里的 `runtime_ctx` 指针必须在整个模型生命周期内有效 (见 4.6).
6. 池内的字节偏移一律用 `size_t` 计算 (`nbyte_slot_*` 是 `size_t`), 池张量可以超过 4 GiB.
7. 阻塞读只走 `block_ring`, 异步读只走 `ring`. 不要在 `ring` 上等待 CQE, 只能通过 `refresh` 收割 (见 4.5).

## 8. 未解决的问题

"场景内" 指第 1 节的目标场景. 编号沿用最初的审查, 已解决的问题已删除 (见第 9 节).

**P5. Prefill 时整个池可能被拷贝到显存** - 只出现在 CUDA 构建中, 推迟到 Jetson 阶段处理; CPU 构建不受影响
- 池在 CUDA_Host 上, buffer usage 是 WEIGHTS. 当 batch >= 32 (场景中 prefill 一般都满足) 时, `MUL_MAT_ID` 会被 offload 到 GPU, 调度器要把整个 `pool_ups/gates/downs` (合计 `utilization x 显存`) 拷到显存.
- 在 Jetson 上, 这相当于内存翻倍, 可能 OOM, 违背了 utilization 的本意.
- 确认方法: 看日志中 `CUDA0 compute buffer size` 和 `graph splits`. 如果 compute buffer 达到 GB 级, 就说明发生了拷贝.

**P6. Jetson 的零拷贝在新版 ggml-cuda 中被关闭** - 只影响 Jetson, 推迟到 Jetson 阶段处理, 会放大 P5
- [ggml-cuda.cu:308](../../ggml/src/ggml-cuda/ggml-cuda.cu#L308) 中 `integrated = false` (上游 #15034). 旧版本在 Jetson 上可能直接在 CUDA_Host 上计算, 新版本一律拷贝. 这是上游的改动, 不属于本特性的代码.

**P7. 2..30 token 的 batch 会 abort** - 场景内是使用约束, 示例已提前检查
- [cpp:708](../../src/custom-moe-offload.cpp#L708) `GGML_ASSERT(a->ne[1] > 30)`.
- 从代码看, 这个断言并不是必需的. 它只服务于 LRU 预热 (`i > ie1 - warmup_token`); token 少于 30 时, 这个循环只是对所有 token 都 hit 一次. 可以考虑去掉断言.

**P8. 编译宏不完整** - 和运行场景无关, 只影响 "一键恢复"
- liburing 的查找和链接、`custom-moe-offload.cpp` 的编译, 以及 `custom-moe-offload.h` 中的 `#include <liburing.h>` 都没有宏保护. 关闭 `ENABLE_CUSTOM_MOE` 后仍然要求安装 liburing.
- `graph_max_nodes` 在 CUSTOM_MOE 分支直接 `return`, 跳过了 lora 和 backend sampler 的节点数. 场景内不使用 lora / backend sampler, 所以没有影响.

**P9. 其他**
- 被预测的专家被 `touch` 到 LRU **尾部** (最先被淘汰). 如果这时它还处于 `Loading` 状态, 并在处理第 il+2 层时被 `free_lru` 选中, 就会触发 [cpp:573](../../src/custom-moe-offload.cpp#L573) 的 `InMemory` 断言. 场景内 `-mo 1` 时有可能发生, 取决于读盘时间.
- [cpp:434](../../src/custom-moe-offload.cpp#L434) `GGML_ASSERT(a = b)` 是赋值, 不是比较.
- `total_size()` 返回 `uint32_t`, 超过 4 GiB 会截断. 当前没有调用者.
- 池的类型和 slot 大小取自最后一层. 如果其他层的某个专家张量比最后一层大, 读入时会截断. 两个测试模型的最后一层恰好是最大的.
- 只打开一个 fd, 忽略 `custom_tensor_mmap::idx` -> 不支持分片 GGUF.
- 异步读的 CQE 只检查 `res < 0`, 不检查短读.
- `free_random` 在其他层都没有 `InMemory` 专家时会死循环; `free_lru` 在所选层的链表为空时 `get_back_n` 是未定义行为.
- deepseek2: 专家张量的 flags 被替换成单独的 `TENSOR_EXPERT_WEIGHT`, 丢掉了 MTP 层的 `TENSOR_SKIP` 等 flag; 使用合并 `gate_up_exps` 的 GGUF 会因为 `ffn_gate_exps` 为空而崩溃; `expert_weights_scale` 缺省为 0 的模型会被 `ggml_scale(..., 0)` 清零.
- 专家张量没有 buffer, 依赖 `tensor->buffer` / `data` 的上游代码 (memory breakdown, `common_fit_params`, `llama-moe-cache`) 行为未知.

## 9. 变更记录

- 2026-10-14: 修复 P1: 图复用导致 `runtime_ctx` 被提前 delete. 改为由 `custom_moe_unified::rt_ctxs` 持有.
- 2026-10-14: 修复 P2: slot 地址计算 32 位溢出. `nbyte_slot_*` 改为 `size_t`.
- 2026-10-14: 规避 P3 (多次 prefill 时 `LoadContext` 不清零): 示例设置 `n_ubatch = n_prompt`, 并在 prompt <= 30 token 时报错退出.
- 2026-10-14: 修复 P4 (阻塞读和异步读共用 io_uring): 增加专用的 `block_ring`; `block_read` 检查短读; `io_uring_register_files` 检查返回值; mlock 失败不再中止加载.
- 2026-10-14: 当前开发环境改为纯 CPU 构建 `build-cpu-debug`; P5/P6 推迟到 Jetson 阶段. Qwen `-mu 0.2 -mo 1` decode 卡住的现象, 是因为 `build-cpu-debug` 还没有包含 P4 的修复 (现象与 P4 一致), 已重新编译, 等待复测.
