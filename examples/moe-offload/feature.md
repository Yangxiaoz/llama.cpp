# CUSTOM_MOE: MoE expert offload to disk

本文记录 `CUSTOM_MOE` 特性的实现细节与代码位置, 用于后续排查问题. 改动基本都用 `#ifdef CUSTOM_MOE` 包裹, 关闭宏即恢复上游代码 (例外: CMake 中的 liburing 和 `custom-moe-offload.cpp` 没有宏保护).

## 1. 目标与场景

- 场景: 内存 (CPU DDR, 或 Jetson 这类统一内存) 装不下整个 MoE 模型. 非专家权重常驻内存, **专家权重 (ffn_up/gate/down_exps) 留在 SSD 上的 GGUF 文件里**, 运行时按需读入一个固定大小的 "专家池" (expert pool).
- 与上游的 `llama-moe-cache` / `-ot` / `--cpu-moe` 等机制无关. 上游机制解决的是 "显存放不下 -> 放到 DDR", 本特性解决的是 "DDR 放不下 -> 放到磁盘".
- 读盘使用 `io_uring` (liburing). 运行时一律以**单个专家**为单位加载.
- 三种运行模式 (第 5 节): `on-demand` (基线), `cross-layer` (跨层预测), `io-aware` (IO 最大化, 本工程后续优化的核心).
- 目标使用场景: 单用户. prompt 可以被拆成多个 ubatch, 也可以多轮对话. 不考虑 MTP、`llama-moe-cache`、合并的 `gate_up_exps` 张量等特殊情况.

## 2. 附加文档 (local_memory/)

本文件只描述特性本身 (设计、数据结构、模式、不变量). 工程操作和修改记录拆到 `local_memory/`:

| 文件 | 内容 |
|---|---|
| [local_memory/build_and_run.md](local_memory/build_and_run.md) | 编译 (cpu-release / cpu-debug / cuda-moe / Jetson preset)、运行参数、测试命令、正确性 / 性能对比、长 prompt、ARPE 标定、常见问题 |
| [local_memory/changelog.md](local_memory/changelog.md) | 修改记录 (按时间, 最新的在最后) |
| [local_memory/jetson_host_memory.md](local_memory/jetson_host_memory.md) | Jetson 上 "iGPU 直接读 host 内存导致 batch > 8 输出错误" 的定位过程, 以及为什么放弃 CUDA_Host 和 managed 方案 |
| [ARPE_施工指南.md](ARPE_施工指南.md) | ARPE 方法本身 (方法文档, 5.7 按它实现) |

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
| [ggml/src/ggml-cuda/ggml-cuda.cu:310](../../ggml/src/ggml-cuda/ggml-cuda.cu#L310) | 保持上游的 `integrated = false`; CUSTOM_MOE 不再修改它 (见 6.5) |
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
- buffer 类型: 设备的默认 buffer type (`ggml_backend_dev_buffer_type(dev)`), CUDA 下是**设备内存**. buffer usage 设为 `WEIGHTS`. io_uring 不能直接写设备内存, 所以另有一块 host staging 环 (6.1).
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

decode 和 prefill 走同一个 `ensure`. on-demand: 逐个专家阻塞读 (decode 不整层), random 淘汰; cross-layer: 并行异步读 + 预取下一层全部缺失专家, LRU 淘汰; io-aware: 同样的预测器但每层最多预取 N 个 (见 local_memory/build_and_run.md 的参数表), ARPE 淘汰 (5.7).

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

### 6.1 设计目标与做法

专家池要被 GPU 读取, 而 CPU 要用 io_uring 写入, 两者不能直接共享同一块内存:
- io_uring 只能写 host 内存.
- iGPU 直接在 host 内存上计算时, batch > 8 的路径 (MMQ) 会读错 (6.5).

做法 (设备池 + host staging):
```
SSD --io_uring--> host staging (pinned) --ggml_backend_tensor_set--> 设备池 (cudaMalloc) --> GPU 计算
                  CPU 可写                     (CUDA 下是 H2D 拷贝 + 同步)
```
- 池分配在**设备内存**, GPU 走的是和上游相同的正常路径.
- io_uring 的目标改为 staging 环里的一个槽; 一个专家的 up/gate/down 三块都到齐后, 由 `refresh` 的回调 `copy_expert_to_pool` 拷进池里的 slot, 然后才标记 `InMemory`.
- CPU 构建 (x86) 没有真正的拷贝: 池和 staging 都是普通内存, `ggml_backend_tensor_set` 内部就是 memcpy.

### 6.2 需要满足的条件和对应改动

1. **找到 iGPU 设备**. Jetson 的 CUDA 设备类型是 `GGML_BACKEND_DEVICE_TYPE_IGPU`, `ggml_backend_dev_by_type(GPU)` 找不到它. 池、staging 和层设备都取 `model.devices.front()`.
2. **池在设备内存**. `ggml_backend_dev_buffer_type(dev)`; 调度器把用到池的 `MUL_MAT_ID` 放到 iGPU, 没有拷贝. 不再需要 `integrated` 补丁 (上游保持 false).
3. **staging 环**. `ggml_backend_dev_host_buffer_type(dev)` (CUDA 下是 pinned 的 CUDA_Host) 分配 `n_stage x slot` 字节, 默认 `n_stage = 16` (DeepSeek-Lite 约 100 MB, Qwen3-30B 约 46 MB). 槽数同时限制并行读的专家数.
4. **非专家权重**走上游路径, 在设备内存里初始化. KV cache 和计算 buffer 也在设备内存.
5. `routers_all` / `router_biases_all` 在池的 buffer 里, 初始化用 `ggml_backend_tensor_set`.

### 6.3 顺序与并发

每个专家的生命周期: 占一个 staging 槽 -> 读 (async 或 block) -> 拷进池的 slot -> `InMemory` -> 释放 staging 槽.
- `Loading` 状态覆盖 "读 + 拷贝" 两个阶段; 这期间 pool slot 和 staging 槽都不能被复用或淘汰 (`can_evict` 只接受 `InMemory`).
- 预取拿不到 staging 槽时直接放弃 (不影响当前层); 当前层按需加载时等待一个槽位释放 (`stage_alloc(wait = true)` 内部收割完成事件).
- 每层的图被切成 `GPU (router, top_k) -> CPU (id_pos_map, custom_predict) -> GPU (mul_mat_id ...)`. CPU 段开始前调度器会同步; 拷贝在 CPU 段内完成, 后续 kernel 看到的是新数据.
- 当前阶段 (阶段一) 的拷贝是同步的 (`ggml_backend_tensor_set` 内部 `cudaMemcpyAsync` + `cudaStreamSynchronize`), 会占用加载线程; 阶段二再改成异步拷贝 + 流顺序.

### 6.4 代价与风险

- **多一次内存拷贝** (staging -> 池). Jetson 上两者都在 DRAM, 单个 6 MB 专家约 0.3-0.6 ms; decode 每 token 若缺 100 个专家, 多 30-60 ms; prefill 每 ubatch 补 19 层 (约 7.3 GB), 多 0.5-1 s.
- **staging 内存**: `n_stage x slot` (见 6.2).
- **状态机更复杂**: 读、拷贝两阶段, staging 槽位分配, 拷贝完成前不能淘汰或复用.
- **预取提前量变短**: 数据要等拷贝完成才可用, 有效预取窗口缩小; 拷贝还会和计算抢内存带宽.
- 阶段一的同步拷贝会串行化加载线程, 预取的并行度受 `n_stage` 限制.
- 还没有在板子上验证.

### 6.5 为什么是设备池 + staging

旧方案把池放在 CUDA_Host, 让 iGPU 原地读. 在 Jetson 上这条路径只在 batch <= 8 (MMVQ) 时正确, batch > 8 会输出 `!!!!`; managed 内存没有改变这一点, 而且 decode 慢约 35 倍. 所以把池放到设备内存, io_uring 先读进 host staging 再拷进池. 定位过程和实验数据见 [local_memory/jetson_host_memory.md](local_memory/jetson_host_memory.md).

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

- Jetson 路径 (第 6 节) 还没有运行验证, 见 6.4. 定位记录在 [local_memory/jetson_host_memory.md](local_memory/jetson_host_memory.md).
- io-aware 只在层边界 (`custom_frontier`) 决定是否预测. 如果某一层计算时间很长, IO 队列可能在这一层中途读空. 要彻底解决需要后台线程, 暂不处理.
- 预加载把池填满时, io-aware 的 IO 从第二个 MoE 层开始, 见 5.4.
- `routers_all` 是 router 权重的一份拷贝 (DeepSeek-Lite 约 13 MB, Qwen3-30B 约 48 MB).
- ARPE 暂不做与 Python 参考实现的差分验证 (指南 12.2), 参数暂用默认值. 本工程里的 ARPE 与指南的已知差异: `predicted` 的专家不可淘汰 (5.7).
- ARPE 只支持单请求 (单 context 串行 decode).
- prompt 长度不是 ubatch 的整数倍时, 最后剩下的 1 个 token 会单独成为一个 ubatch, 在 MoE 层 0 被判定为 decode (5.3), 按 decode 的加载和淘汰方式处理. 输出正确, 只是那一层不做 prefill 预测. 例: 513 token = 512 + 1.
- 专家大小不一致导致池有约 7%-9% 的浪费 (按测试模型估算). 暂不处理.
- 已知限制 (不在目标场景内): 分片 GGUF (只打开一个 fd); 合并的 `gate_up_exps` 张量; deepseek2 的 MTP 层; 专家张量没有 buffer, 依赖 `tensor->buffer` 的上游功能 (memory breakdown, `common_fit_params`, `llama-moe-cache`) 行为未知.
