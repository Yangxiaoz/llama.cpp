# Jetson: iGPU 直接读 host 内存时 batch > 8 输出错误

本文件记录 2026-10-21 的定位过程和结论. 它解释了为什么当前的专家池放在设备内存、并额外使用 host staging (见 [feature.md](../feature.md) 6.1-6.5).

现象 (Jetson Orin Nano 8GB, JetPack 6.2 / L4T R36.4.3, CUDA 12.6, liburing 2.1). 本文记录为什么放弃了"池放 CUDA_Host、iGPU 原地读"的旧方案.

- 旧方案 (池在 CUDA_Host + `integrated = prop.integrated`) 下, prompt <= 8 token 输出正常, >= 9 token 开始输出 `!!!!`; 与 `--moe-pool-ratio`、运行模式无关, `-ngl 0` 也一样.
- 把 ggml-cuda 的 `integrated` 临时改回 `false` 后输出正确. 边界正好是 `MMVQ_MAX_BATCH_SIZE == 8` (`ggml/src/ggml-cuda/mmvq.cuh`): batch <= 8 走 MMVQ 内核 (能正确读 host 内存), batch > 8 走 MMQ / 其它路径 (读 host 内存得到错误结果). 旧方案下 `get_layer_buft_list` 让**所有层权重**都在 CUDA_Host, 所以整个 prefill 都受影响.
- 尝试过 managed memory (池和权重都用 `cudaMallocManaged`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`): **仍然在 9 token 起出错**, 8/9 边界不变; 而且 decode 从 7.4 t/s 掉到 0.21 t/s. 结论: Jetson 上 managed 内存仍属于"系统内存"这一类, 没有改变 MMQ 的访问路径, 反而带来 UVM 的缺页/迁移开销. 该方案已放弃.
- 当前方案 (设备池 + staging) 让 GPU 读真正的 cudaMalloc 设备内存, 和上游 llama.cpp 的默认路径一致.
- 阶段的临时规避 (旧方案下仍可用): 示例的 `-ub 8` 把 ubatch 限制在 8, 全部走 MMVQ. 实测 DeepSeek-Lite / 66 token: cross-layer prefill 4.6 t/s, io-aware 4.1 t/s, on-demand 1.1 t/s. 验证新方案正确后删除这个参数.

## 2026-10-21 补充: 设备池上的 `!!!!` 也是同一个内核行为

设备池 + staging 之后, 4060 上仍然复现 `-ub 8` 正常、`-ub 9` 输出 `!!!!` (和 Jetson 无关). 最小复现程序证明这**不是** type-patched 视图本身的问题 (视图和连续张量逐位一致), 而是 MMQ 会读过专家末尾:

- MMQ 每行按 `MMQ_ITER_K` 分块, 最后一块会读过专家最后一行最多 `MATRIX_ROW_PADDING` = 512 个值. 上游的连续权重后面是另一个张量 (有限值), 所以无害.
- 池里混合量化层的专家后面是空的 slot 尾部, 残留上一个更大专家的字节; 按 fp16 解码常是 NaN/inf. 类型一致的层则读到下一个 slot 的开头, 也可能是别的类型.
- 修复: slot 步长 = 最大专家 + 1 KB guard, 拷贝时把 guard 一起写成 0 (feature.md 4.2 / 6.5). 修复后 DeepSeek-Lite 和 Qwen3-30B 在 `-ub 9/512`、pool 0.1/0.3 下输出都正确.
- 这条也解释了旧方案里 host 内存上的失败: 上游的 host buffer 布局不受控制, 专家末尾之后同样可能是 NaN 字节.
