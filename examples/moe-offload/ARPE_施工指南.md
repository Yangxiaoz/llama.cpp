# ARPE 施工指南

> **Affinity-aware Reuse-Probability Eviction（亲和感知的重用概率驱逐）**
>
> 目标：让读者**只读本文**，就能在另一个推理系统（如 llama.cpp、自研推理框架）中完整复现 ARPE 的 decode 阶段专家缓存驱逐。
>
> 本文只讲方法与落地步骤，不依赖原仿真器的工程结构。参考实现 `cache_Laboratory_Manual/code/arpe.py` 仅用于最后的差分验证。

---

## 0. 一句话概括

当专家缓存已满、需要为当前层加载一个缺失专家时：

> **在所有“已驻留且未锁定”的专家中，驱逐“它所在层下一次被访问时被选中概率”最低的那个。**

这个概率由两部分组成：

```text
p = 历史重用概率（频率 + 近期性） + 跨层亲和预测（当前层路由 → 下一层路由）
```

- **prefill 阶段不使用 ARPE 驱逐**，沿用 LRU（或系统已有的策略）；但 prefill 的路由要用来构建先验和亲和统计。
- **decode 阶段**才使用 ARPE 做驱逐。
- 只依赖每层 router 选出的 top-k 专家 ID，**不需要门控概率，也不需要未来信息**。

---

## 1. 前置条件

在动手前，确认你的系统满足以下条件：

| 条件 | 说明 |
| --- | --- |
| 逐专家缓存 | 缓存以 `(layer, expert)` 为槽位，可单独加载/驱逐某个专家。若系统只能整层 offload，需要先增加逐专家驻留层。 |
| 知道 top-k 路由 | 每个 token、每一层都能拿到 router 选中的 K 个专家 ID。 |
| 可以锁定当前层 | 当前层本 token 需要的 K 个专家在加载/计算期间不能被驱逐。 |
| 容量足够 | 缓存容量 `C ≥ K + 1`（至少能放下当前层 K 个专家，并留出 1 个可驱逐槽位）。 |
| 明确请求边界 | ARPE 的统计量是**按请求**维护的。多请求共享缓存时，需要额外定义聚合策略（见 §11.4）。 |
| 标定数据（推荐） | 少量请求的逐层 top-k 路由，用于离线构建亲和先验 `A0`。没有也能跑，但会损失一部分收益。 |

---

## 2. 符号与状态

### 2.1 模型规模

| 符号 | 含义 |
| --- | --- |
| `L` | MoE 层数 |
| `E` | 每层路由专家数 |
| `K` | 每个 token 每层选中的专家数（top-k） |
| `C` | 缓存可容纳的专家槽位数 |

### 2.2 每个请求的状态

每个请求开始时全部重置：

| 名称 | 形状 | 初值 | 含义 |
| --- | --- | --- | --- |
| `cnt[l][e]` | `L × E` | 0 | decode 中专家 `e` 在第 `l` 层被选中的次数 |
| `seen[l]` | `L` | 0 | 第 `l` 层已完成的 decode token 数 |
| `last[l][e]` | `L × E` | `NEVER` | 最近一次被选中的 decode token 序号 |
| `prior[l][e]` | `L × E` | `K/E` | 由 prefill 路由得到的先验概率 |
| `Co[l][i][j]` | `(L-1) × E × E` | 0 | 本请求在线统计的相邻层共现次数 |
| `No[l][i]` | `(L-1) × E` | 0 | 本请求在线统计的上一层出现次数 |
| `prev_sel` | `K` | 空 | 当前 token 上一层选中的专家集合 |

全局状态（部署前算好，所有请求共享）：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `A0[l][i][j]` | `(L-1) × E × E` | 离线亲和先验：给定当前层选中 `i` 时，下一层选中 `j` 的概率的平滑估计 |

`NEVER` 是一个“从未使用”的哨兵。实现时建议用布尔数组或 `int64` 极小值表示，**不要**直接拿它做指数运算；计算近期性时显式判断。

---

## 3. 离线标定：构建亲和先验 `A0`

### 3.1 需要采集什么

用少量**标定请求**跑一遍推理，记录每个请求、每一层、每个 token 的 top-k 专家集合：

```text
trace[request][layer][token] = [e_1, e_2, ..., e_K]
```

prefill 和 decode 的 token 都要记录。标定集不要与被评测的请求重合；如果某条请求既参与标定又被评测，构建先验时应把它排除（留一法）。

### 3.2 统计相邻层共现

对每个请求、每个 token，把同一 token 在第 `l` 层和第 `l+1` 层的 top-k 配对：

```text
for l in 0 .. L-2:
    for each token t:
        S_l   = top-k of token t at layer l
        S_l1  = top-k of token t at layer l+1
        for i in S_l:
            N[l][i] += 1
            for j in S_l1:
                C[l][i][j] += 1
```

如果第 `l+1` 层记录的 prompt token 数 `n` 少于第 `l` 层（例如最后一层只记录了最后一个 prompt token），就把第 `l` 层的**最后 n 个** prompt token 与第 `l+1` 层的 n 个 prompt token 按位置对齐。正常完整日志中每层 token 数相同，逐行配对即可。

### 3.3 平滑成概率

```text
A0[l][i][j] = (C[l][i][j] + 1) / (N[l][i] + E/K)
```

- `+1` 是拉普拉斯平滑；
- `E/K` 是平滑强度，使每一行的和恰好为 `K`；
- 自检：对任意 `l, i`，`sum_j A0[l][i][j] ≈ K`。

### 3.4 没有标定数据时

- 令 `A0[l][i][j] = K/E`（均匀），此时亲和修正不提供信息；
- 仍可使用在线亲和（`Co/No`）和历史项；
- 或者直接关闭亲和项（`lam=()`），退化为“平滑频率 + 近期性”策略。

---

## 4. 每个请求的初始化

```text
cnt[l][e]  = 0
seen[l]    = 0
last[l][e] = NEVER
prior[l][e] = K / E
Co[l][i][j] = 0
No[l][i]    = 0
prev_sel    = None
```

`A0` 在启动时加载一次，所有请求共享。

---

## 5. Prefill 阶段

**驱逐策略：沿用 LRU**（或系统已有的 prefill 策略）。ARPE 只负责更新统计量，不参与 prefill 驱逐。

在每一层的 router 得到**该层全部 prompt token** 的 top-k 之后（`sel` 是 `n_l × K`，按 token 顺序）：

```text
# 1. 更新该层的 prefill 先验
pc[e] = sel 中专家 e 出现的总次数
prior[l][e] = K * (pc[e] + 1) / (sum(pc) + E)

# 2. 只有最后一个 prompt token 的专家被视为“第 -1 个 decode token”
for e in sel 的最后一行:
    last[l][e] = -1

# 3. 在线相邻层配对（第 0 层没有上一层）
if l > 0 and online_aff:
    prev = prev_layer_sel 的最后 len(sel) 行
    for (a, b) in zip(prev, sel):
        for i in a:
            No[l-1][i] += 1
            for j in b:
                Co[l-1][i][j] += 1

prev_layer_sel = sel
```

要点：

- `prior` 用**全部** prompt token 统计；`last` 只用**最后一个** prompt token 初始化。
- 第 0 层没有上一层，跳过配对；prefill 的配对只发生在**同一 token 的相邻层**之间。
- 如果 prefill 是分块执行的：累计每层的专家计数，在进入 decode 前用累计值计算 `prior`；`last` 取最后一个 chunk 的最后一个 token；在线配对按 token 位置跨 chunk 对齐。

---

## 6. Decode 阶段

### 6.1 每个 token、每一层的顺序

对 decode token `t = 0, 1, 2, ...`，依次处理每一层 `c = 0 .. L-1`：

```text
1. 拿到当前 token 在第 c 层的 top-k：S_c
2. 对 S_c 中每个不在缓存里的专家 e（按路由顺序）：
     如果缓存已满：
          victim = select_victim(c, t, resident, locked=S_c)
          驱逐 victim
     加载 e，放入空出的槽位
3. 计算该层
4. 调用 observe_decode(c, t, S_c)   # 更新 cnt / last / seen / Co / No
5. （可选）预取下一层：在 observe_decode 之后调用预取逻辑
```

**关键顺序**：按需加载的驱逐发生在 `observe_decode(c, t, S_c)` **之前**。因此计算概率时，当前 token 在第 `c` 层的数据还没有计入 `cnt/last`，`seen[c]` 仍等于 `t`。这是方法定义的一部分，不是实现细节。

### 6.2 多个缺失专家

同一层可能有多个专家缺失。每加载一个专家前调用一次 `select_victim`；概率在 `observe_decode` 之前不变，只有驻留集合在变。新加载的专家立即变为“驻留且锁定”，后续驱逐不能选它。

### 6.3 异步预取 / 预加载

如果系统会在 `observe_decode` 之后预取下一层，驱逐时使用的 token 序号要特别小心：

- 按需加载（observe 之前）：直接用当前 token `t`；
- observe 之后预取：统计数据已经计入当前 token，若用 `seen[c]` 推导，应使用 `seen[c] - 1`，结果仍是 `t`。

参考实现用一个 `observed` 标志统一处理：

```text
t_eff = seen[c] - (1 if observe_decode(c, t) 已经调用 else 0)
```

在参考实现中，按需加载传 `observed=False`，预取提交传 `observed=True`。

另外：`locked` 始终是**当前层 `c`** 的 K 个专家，即使你在预取层 `c+1`。因为亲和修正的输入是“当前 token 在当前层的路由”。

---

## 7. 计算重用概率

在 decode 第 `t` 个 token、正在处理第 `c` 层、且**尚未**调用 `observe_decode(c, t)` 时，计算所有层的分数：

### 7.1 历史项（所有层）

```text
next_tok(l) = t + 1    if l <= c
            = t        if l >  c

f(l,e) = (cnt[l][e] + W0 * prior[l][e]) / (seen[l] + W0)

p(l,e) = f(l,e) + alpha * rho^(next_tok(l) - last[l][e] - 1)
```

- 若 `last[l][e] == NEVER`，第二项取 **0**（不要直接做指数运算）。
- `next_tok - last - 1` 中的 `last` 是“上次被选中的 token 序号”。例如上次在 token 5 被选中、下次在 token 6 被选中，距离为 1，指数为 0，近期性加成是 `alpha`。
- 当前层 `l = c` 属于“本 token 已经过”的层，下次访问在下一个 token，所以 `next_tok = t+1`；尚未处理的层 `l > c` 下次访问就在本 token，所以 `next_tok = t`。
- `seen[l]` 是第 `l` 层已完成的 decode token 数；它已包含/不包含当前 token 取决于 `l` 与 `c` 的关系，直接使用当前值即可。

### 7.2 亲和项（默认只修正下一层）

只有 `c + 1 < L` 时执行：

```text
A[i][j] = (Co[c][i][j] + w_aff * A0[c][i][j]) / (No[c][i] + w_aff)

q[j] = (1/K) * sum_{i in S_c} A[i][j]

p(c+1, j) = lam[0] * q[j] + (1 - lam[0]) * p(c+1, j)
```

其中 `S_c` 是当前 token 在当前层的 K 个专家（`locked`）。

实现建议：

- 不需要真的构造完整的 `A` 矩阵。只需对 `i in S_c` 求和：

  ```text
  q[j] = (1/K) * sum_{i in S_c} (Co[c][i][j] + w_aff * A0[c][i][j]) / (No[c][i] + w_aff)
  ```

  这是 `K × E` 的工作量，而不是 `E × E`。
- 参考实现还有一步 `q = q / sum(q) * K`。由于 `A` 的每一行和都是 `K`，`q` 的和本来就是 `K`，这一步在数学上是冗余的（浮点误差范围内）。为了严格照搬可以保留；为了简洁可以省略。
- 默认 `lam = (0.6,)`，只修正下一层。参考实现支持多个 `lam`，对第 `c+d` 层使用 `lam[d-1]`，并把 `q` 依次传播过 `A[c], A[c+1], ...`。实验结论是推到第 `c+2` 层没有额外收益，**默认只用下一层**。

---

## 8. 驱逐决策

### 8.1 候选集

```text
候选 = { (l, e) | 已驻留(l,e) 且 不是当前层的锁定专家 }

当前层锁定 = S_c（当前 token 在第 c 层的 K 个专家）
```

- 锁定只针对**当前层**；其他层的专家都可以被驱逐，包括上一层、下一层和更远的层。
- 不设层配额、不设“保护下一层”之类的规则。实验表明，ARPE 有相当比例的驱逐确实发生在下一层，这是正常的。
- 正在加载中（未就绪）的条目如何处理？参考实现中它们**不算驻留候选**（不能驱逐，但占槽位）。如果你的系统允许取消预取，可以自行定义；但必须保证不会驱逐当前计算依赖的条目。

### 8.2 打分与同分裁决

```text
dist(l, c) = L                   if l == c
           = (l - c + L) % L     otherwise

score(l,e) = p(l,e) - 1e-9 * dist(l,c)
```

- 先按 `p` 最小选择；
- `1e-9 * dist` 只是**同分裁决**：`p` 相同或几乎相同时，优先驱逐距离下一次访问更远的层。当前层记为 `L`（最远），下一层为 `1`（最近）。
- 如果 `score` 仍然完全相同，选 **`(l, e)` 行优先编号最小者**（先 `l` 小，再 `e` 小）。实现时按 `l` 从 0 到 L-1、`e` 从 0 到 E-1 扫描，用严格小于 `<` 更新最优值即可。

### 8.3 没有候选怎么办

参考实现的实验容量远大于 K，因此没有触发“无候选”的情况；代码里用断言保护。移植时建议：

- 保证 `C ≥ K + 1`；
- 如果仍出现无候选（例如所有驻留条目都在加载中），定义明确的回退策略：等待、放弃预取、或退化到 LRU 驱逐一个非锁定条目；
- 不要在没有候选时随机驱逐一个当前层锁定专家，否则当前层无法完成。

---

## 9. 完整伪代码

以下是与参考实现等价的、语言无关的版本。

```text
# ---------- 全局/离线 ----------
A0 = calibrate_affinity(calibration_traces)     # (L-1, E, E)

# ---------- 每请求状态 ----------
cnt   = zeros(L, E)
seen  = zeros(L)
last  = NEVER_matrix(L, E)
prior = full(L, E, K / E)
Co    = zeros(L-1, E, E)
No    = zeros(L-1, E)
prev_sel = None

# ---------- prefill ----------
def observe_prefill(l, sel):        # sel: n_l × K, token 顺序
    if prefill_prior:
        pc = counts(sel)
        prior[l] = K * (pc + 1) / (sum(pc) + E)
    for e in sel[-1]:
        last[l][e] = -1
    if online_aff and l > 0:
        prev = prev_sel[-len(sel):]
        for a, b in zip(prev, sel):
            for i in a:
                No[l-1][i] += 1
                for j in b:
                    Co[l-1][i][j] += 1
    prev_sel = sel

# ---------- decode ----------
def observe_decode(l, t, sel):
    if online_aff and l > 0:
        for i in prev_sel:
            No[l-1][i] += 1
            for j in sel:
                Co[l-1][i][j] += 1
    for e in sel:
        cnt[l][e] += 1
        last[l][e] = t
    seen[l] += 1
    prev_sel = sel

# ---------- 重用概率 ----------
def reuse_prob(c, t, S_c):
    p = zeros(L, E)
    for l in range(L):
        next_tok = t + (1 if l <= c else 0)
        for e in range(E):
            f = (cnt[l][e] + W0 * prior[l][e]) / (seen[l] + W0)
            if last[l][e] != NEVER:
                f += alpha * rho ** (next_tok - last[l][e] - 1)
            p[l][e] = f

    if len(lam) > 0 and lam[0] > 0 and c + 1 < L:
        q = zeros(E)
        for i in S_c:
            for j in range(E):
                if online_aff:
                    A_ij = (Co[c][i][j] + w_aff * A0[c][i][j]) / (No[c][i] + w_aff)
                else:
                    A_ij = A0[c][i][j]          # 只用离线先验
                q[j] += A_ij / K
        # 可选：q = q / sum(q) * K
        for j in range(E):
            p[c+1][j] = lam[0] * q[j] + (1 - lam[0]) * p[c+1][j]

    return p

# ---------- 驱逐 ----------
def select_victim(c, t, resident, locked):
    p = reuse_prob(c, t, set_of(locked))
    best = None
    for l in range(L):
        dist = L if l == c else (l - c + L) % L
        for e in range(E):
            if not resident[l][e]:
                continue
            if l == c and locked[e]:
                continue
            score = p[l][e] - 1e-9 * dist
            if best is None or score < best.score:
                best = (l, e, score)
    if best is None:
        raise NoEvictionCandidate     # 见 §8.3：等待/回退/放弃预取
    return best

# ---------- decode 一层的主循环 ----------
def decode_layer(c, t, S_c):
    locked = bitmask(S_c)
    for e in S_c:
        if not resident[c][e]:
            while cache_is_full():
                vl, ve = select_victim(c, t, resident, locked)
                evict(vl, ve)
            load_and_place(c, e)
    compute_layer(c)
    observe_decode(c, t, S_c)
```

---

## 10. 参数与调优

### 10.1 默认参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `W0` | 40.0 | prefill 先验相对 decode 计数的等效样本数；越大越信任 prefill |
| `alpha` | 0.16 | 近期性项的幅度；上次使用距离为 1 时加 `alpha` |
| `rho` | 0.1 | 近期性项每个 token 的衰减率 |
| `lam` | `(0.6,)` | 亲和预测的混合权重；默认只修正下一层 |
| `w_aff` | 20.0 | 离线亲和先验相对在线计数的等效样本数 |
| `online_aff` | `True` | 是否使用本请求的在线相邻层计数 |
| `prefill_prior` | `True` | 是否用 prefill 路由构造 `prior` |

这些值就是参考实现和 `../src/ReuseCache.py` 的默认值，可以直接使用。

### 10.2 换模型时是否要调参

- 如果目标模型/数据与 DeepSeek-16B 差异很大，建议在少量标定请求上重新拟合 `W0/alpha/rho`：
  - 对每个 decode token、每层、每个专家构造伯努利样本：`y = 1[该专家本 token 被选中]`；
  - 特征：`n = 该专家在之前 decode token 中被选中的次数`、`s = 已处理 token 数`、`pi = prefill 先验`、`a = 距上次被选中的 token 数`；
  - 最大化 `mean[y log p + (1-y) log(1-p)]`，其中 `p = f + alpha * rho^(a-1)`，`f = (n + W0*pi)/(s + W0)`。
- `lam` 在标定集上做网格搜索（0.4–0.8），默认 0.6。只修正下一层即可。
- `w_aff` 是经验值；离线先验越可靠，可以适当增大。
- 参考实现的拟合结果：`W0≈39.8`、`alpha≈0.148–0.16`、`rho≈0.10`。似然面很平坦，默认值 `40/0.16/0.1` 足够；**不需要为了第 3 位小数纠结**。

> 说明：手册 `REPORT.md §3.7` 中写的是“加法形式 MLE：W0=39.8, α=0.148, ρ=0.103”，而代码默认是 `α=0.16`。这来自“广义形式（γ≈0.034）”的拟合结果（39.82/0.1601/0.1008）与最终取整；两者似然几乎相同。以代码默认值为准即可。

### 10.3 可以简化吗

| 简化 | 代价 | 建议 |
| --- | --- | --- |
| 关闭离线亲和（`A0=uniform`） | 手册报告平均少约 0.3pp | 没有标定数据时可以这样做 |
| 关闭在线亲和（只用 `A0`） | 报告中平均约 +2.30pp vs +2.34pp | 内存/统计开销敏感时可用 |
| 关闭 prefill 先验（`prior=K/E`） | 平均约 +2.14pp，大预算下可能为负 | 不建议 |
| 关闭近期性项（`alpha=0`） | 小预算下明显变差（约 −5pp） | 不建议 |
| 乘法历史项 | 平均约 +2.11pp | 不建议；用加法形式 |

---

## 11. 接入推理引擎：通用步骤

### 11.1 找到三个钩子点

1. **路由观测点**：每层 router 产生 top-k 之后，记录 `(request_id, layer, token_index, topk_ids)`。
2. **专家加载点**：某层需要某个专家、但缓存中没有时，在真正搬运数据之前调用 `select_victim`。
3. **层完成点**：该层计算/访问完成后，调用 `observe_decode`（decode）或 `observe_prefill`（prefill）。

### 11.2 缓存管理器的最小接口

```text
class ExpertCache:
    resident:  (L, E) bool
    slots:     (C) -> (layer, expert) or empty
    loc:       (layer, expert) -> slot index

    ensure(layer, expert_ids):
        lock = set(expert_ids)
        for e in expert_ids:
            if not resident[layer][e]:
                while free_slots == 0:
                    (vl, ve) = arpe.select_victim(layer, t, resident, lock)
                    evict(vl, ve)
                load(layer, e)          # 异步/同步由系统决定
        return lock                     # 计算期间保持锁定
```

- `resident` 只包含**已就绪**的专家；正在加载中的条目不要参与候选。
- 驱逐后要立即更新 `resident` 和 `loc`，否则后续 `victim` 会看到过时的驻留集。
- 确保被驱逐的专家当前没有被其他计算引用；必要时用引用计数或同步。

### 11.3 在 llama.cpp 中落地的建议

llama.cpp 的具体实现随版本变化，下面只给概念映射：

1. **前提**：llama.cpp 需要有逐专家的驻留/换入换出机制。如果当前只是 mmap 或整层 offload，需要先增加一个 `(layer, expert) -> slot` 的缓存层；ARPE 才能在其上做槽位级驱逐。
2. **路由观测**：在 MoE FFN / router 节点拿到每层选中的 expert ID（top-k）。可以在图执行时加一个轻量回调，把 `(layer, token, ids)` 写入一个 per-context 的观测缓冲。
3. **加载钩子**：在 expert 张量从 host/disk 进入工作缓冲区之前，调用 `ensure()`；`ensure()` 内部用 ARPE 选 victim。
4. **prefill**：保留 llama.cpp 现有的 prefill 策略（LRU 或整层策略），只在每层完成后调用 `observe_prefill`。
5. **decode**：每层计算前调用 `ensure()`，计算后调用 `observe_decode`；如果框架在层间做预取，注意 §6.3 的 `t_eff` 修正。
6. **标定**：用 5–10 条与评测集不重合的 prompt 跑一遍，记录逐层 top-k，离线计算 `A0`，以二进制 `float32` 保存并在启动时加载。
7. **线程/上下文**：统计量是 per-context 的；缓存可能是共享的。先按“单 context 串行 decode”落地，再考虑多 context 的聚合（见 §11.4）。
8. **性能**：每次驱逐扫描 `L×E` 个候选；对 `L=26–48, E=64–128` 来说远小于一次专家加载。可以在同一层内缓存 `p`（驱逐前后统计不变），避免对每个缺失专家重复计算。
9. **内存**：`A0` 是 `(L-1)×E×E` 个浮点数。DeepSeek-16B（26×64）约 0.4 MB（float32），Qwen-30B（48×128）约 3 MB；其余状态很小。

### 11.4 多请求/多序列

参考方法在**单请求**下定义和验证。多请求共享缓存时，至少要做以下决策之一：

- **单请求模式**：一次只让一个请求使用 ARPE（最简单，与参考实验一致）；
- **每请求统计 + 聚合打分**：每个请求维护自己的 `cnt/last/seen/prior/Co/No`；驱逐时对每个驻留专家计算各请求的 `p`，取 `max`（任何一个请求马上要用就保留）或加权和；
- **只保留当前请求统计**：缓存内容跨请求保留，但统计量在请求边界重置；实现简单，但对其他请求不友好。

无论哪种，都要明确：`A0` 是模型级全局先验；`Co/No` 是请求级在线统计；缓存驻留集是系统级资源。不要把不同请求的 `cnt/last` 混在一起。

---

## 12. 正确性验证清单

### 12.1 不变量（每次更新后检查）

1. `sum_j A0[l][i][j] ≈ K`（每个 `l, i`）。
2. `sum_e prior[l][e] ≈ K`（每个 `l`）。
3. `seen[l]` 等于该层已完成的 decode token 数；处理完第 `t` 个 token 后 `seen[l] = t+1`。
4. `cnt[l][e]` 不超过 `seen[l]`；`last[l][e] < seen[l]`（除非 `NEVER`）。
5. `select_victim` 返回的 `(l,e)` 必须 `resident[l][e] == True`，且不能是 `(c, locked)`。
6. 对 `NEVER` 专家，近期性项严格为 0。
7. 第 0 层 decode 不与上一个 token 的最后一层配对（检查 `Co[L-2]` 的增长时机）。

### 12.2 差分测试（推荐）

把待实现版本与 Python 参考实现 `cache_Laboratory_Manual/code/arpe.py` 放在同一组输入上，逐步比较：

```python
# 伪代码
ref = ReuseModel(L, E, K, offline_aff=(C, N))
impl = YourReuseModel(L, E, K, offline_aff=(C, N))

for l in range(L):
    ref.observe_prefill(l, prefill_sel[l])
    impl.observe_prefill(l, prefill_sel[l])
    compare(ref.prior, impl.prior)
    compare(ref.last, impl.last)
    compare(ref.Co, impl.Co); compare(ref.No, impl.No)

for t in range(T):
    for l in range(L):
        S_c = dec_sel[t, l]          # 当前 token 在当前层的 K 个专家
        # 在 observe 之前比较 reuse_prob
        compare(ref.reuse_prob(l, t, S_c), impl.reuse_prob(l, t, S_c))
        ref.observe_decode(l, t, dec_sel[t, l])
        impl.observe_decode(l, t, dec_sel[t, l])
        compare(ref.cnt, impl.cnt); compare(ref.last, impl.last)
        compare(ref.seen, impl.seen)
```

再用随机生成的 `resident`/`locked` 状态比较 `victim` 的返回值。验收标准：**统计量在浮点容差内逐元素相等（Python 参考实现之间应完全相同），victim 逐次相同**。

本仓库已经做过这项核对：

- `cache_Laboratory_Manual/code/arpe.py` 与 `../src/ReuseCache.py` 在同一条轨迹上，`prior/cnt/last/seen/Co/No/A0` 完全相同；
- 200 组随机驻留/锁定状态下，`victim` 决策完全相同；
- 手册的 `cachesim` 与完整仿真器（`Simulation.Simulator + OnDemand`）在 LRU/LFU/ARPE × 10 条日志 × 11 个预算上，命中率与时延最大绝对差为 `0.0`（`cache_Laboratory_Manual/data/original_simulator/sim_check.csv`）。

### 12.3 端到端验收

1. 先在**离线仿真**中把新实现与参考实现对齐（上面的差分测试）。
2. 接入真实系统后，用固定 seed 和固定 prompt 集做 A/B：
   - 基线：系统原有 LRU；
   - 实验：ARPE + 相同的预取/加载策略；
   - 指标：decode 专家命中率、每 token 时延、总时延、加载字节数。
3. 检查驱逐分布：ARPE 应该经常驱逐“下一层用不到”的专家，而不是总驱逐刚用过的上一层专家。若分布与 LRU 几乎一样，通常说明概率算错或调用时机不对。

### 12.4 小规模自检

用一个极小的人工状态手算一遍，确认实现没有把 `next_tok` 或 `seen` 弄反：

- 处理第 `t` 个 token 的第 `c` 层之前，`seen[c]` 应该等于 `t`；
- 当前层 `c` 的 `next_tok` 是 `t+1`；
- 下一层 `c+1` 的 `next_tok` 是 `t`；
- 刚在第 `t-1` 个 token 用过的专家，在尚未处理到的层（`l > c`）的近期性项应为 `alpha * rho^0 = alpha`；已经处理过的层（`l <= c`）应为 `alpha * rho`；
- 从未用过的专家，近期性项应为 0。

---

## 13. 常见坑

1. **在 `observe_decode` 之后才算驱逐概率**：当前 token 的数据会提前进入统计，导致 `seen/last` 偏大，分数错误。按需加载必须在 observe 之前。
2. **把 `locked` 当成所有层的专家**：`locked` 只包含当前层 `c` 的 K 个专家；其他层都可以被驱逐。
3. **只锁定命中的专家**：当前层所有 K 个专家（命中的和缺失的）都要锁定，否则可能把刚加载的专家又驱逐掉。
4. **prefill 给所有 token 设置 `last`**：只有最后一个 prompt token 的专家设 `last=-1`；其余专家保持 `NEVER`。`prior` 才用全部 prompt token。
5. **第 0 层 decode 与上一个 token 的最后一层配对**：不要跨 token 配对；在线亲和只在同一 token 的相邻层之间统计。
6. **把正在加载中的条目当作可驱逐候选**：参考实现只把已就绪的 `Memory` 专家作为候选；加载中的条目占槽位但不能被选。
7. **忘记在每次驱逐后更新驻留集**：多个缺失专家连续加载时，`resident` 必须实时更新，否则会重复选同一个 victim 或选到已驱逐的条目。
8. **以为需要门控概率**：ARPE 只用 top-k 路由，不需要 softmax 概率。
9. **使用未来信息**：ARPE 是 online 的，只用当前 token 当前层及之前的统计；不要使用下一层的真实路由。ARPE 本身不需要预测日志（预测日志是 CrossPrefetch / IoAware 等预取策略的输入）。
10. **多请求统计串味**：`cnt/last/seen/Co/No` 必须按请求隔离；`A0` 才是全局共享。
11. **容量小于 K+1**：可能没有非锁定候选。保证容量，或定义回退策略。
12. **近似实现 tie-break**：参考实现用 `p - 1e-9 * dist`，并按 `(l,e)` 行优先取最小。若你的实现用排序/堆，确保同样的次序。
13. **浮点用 float16**：概率比较对精度敏感，用 `float32` 或 `float64`；`A0` 用 `float32` 足够。

---

## 14. 参考实现与文件

| 文件 | 用途 |
| --- | --- |
| `cache_Laboratory_Manual/code/arpe.py` | 参考实现（`ReuseModel` + `count_affinity`），可直接导入并对照 |
| `cache_Laboratory_Manual/code/cachesim.py` | 同步 OnDemand 仿真器与 LRU/LFU/Belady 基线 |
| `cache_Laboratory_Manual/code/fit_params.py` | `W0/alpha/rho` 的最大似然拟合 |
| `cache_Laboratory_Manual/REPORT.md` | 方法报告、实验与结果 |
| `../src/ReuseCache.py` | 原工程中的同一实现（FastSim 适配器） |
| `../src/Strategy.py` | 完整仿真器中的集成方式（`free_mode=4`、`free_arpe`、`observed`） |

---

## 15. 移植完成定义（Definition of Done）

满足以下全部条件，才算“完美复现”：

- [ ] `A0` 的每一行和为 `K`；`prior` 每一层和为 `K`。
- [ ] 与参考实现在同一条轨迹上逐步比较，`prior/cnt/last/seen/Co/No` 在浮点容差内逐元素相等。
- [ ] 与参考实现在随机驻留状态上比较，`victim` 全部相同。
- [ ] 端到端离线仿真的 decode 命中率与时延与参考实现一致（允许浮点末位差异）。
- [ ] 真实系统中 A/B：ARPE 的 decode 命中率**平均**不低于 LRU（允许个别请求小幅波动），且时延有改善（在内存预算紧张时更明显）。
- [ ] 驱逐分布不是“总驱逐刚用过的上一层”，而是能经常命中“下一层用不到”的专家。
- [ ] 多请求场景已明确统计隔离/聚合策略，且没有跨请求状态泄漏。
