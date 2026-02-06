# Sequence Parallel (Ulysses) C++ 实现使用指南

## 概述

这是 DeepSpeed Ulysses Sequence Parallelism 的 C++ 实现，用于 libtorch 推理。

## 核心特性

✅ **All-to-All 通信** - 实现了 Sequence ↔ Head 并行的动态切换
✅ **Attention 层** - 支持 Sequence Parallelism 的 Scaled Dot-Product Attention
✅ **序列自动填充** - 自动将序列长度填充为 sp_size 的倍数
✅ **完整数据流** - Embedding → Split → Attention (with All-to-All) → AllGather
✅ **MPI 支持** - 使用 ProcessGroupMPI 进行通信（CPU 或多机）

## 编译

```bash
cd /data/workspace/hunyuan_ptm/torch_examples/cpp/distributed/build

# 配置
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# 编译
make sequence-parallel-inference
```

## 运行

### 基础运行（4 进程）

```bash
mpirun -np 4 ./sequence-parallel-inference
```

### 预期输出

```
Rank 0/4 started
Rank 1/4 started
Rank 2/4 started
Rank 3/4 started
Rank 0: TransformerSP initialized with vocab_size=320, dim=256, n_heads=4

Input tokens shape: [2, 12]
Sample tokens: ...

After split: rank 0 -> h shape: [2, 3, 256]

Rank 0 inference completed.
Output shape: [2, 12, 256]
Output sample (first token):
...

=== Sequence Parallel inference completed successfully! ===
```

## 代码结构

### 1. All-to-All 通信原语

```cpp
torch::Tensor all_to_all(
    torch::Tensor input,
    int scatter_dim,    // 切分的维度
    int gather_dim,     // 拼接的维度
    c10::intrusive_ptr<c10d::ProcessGroupMPI> pg
);
```

**功能**：将张量按 `scatter_dim` 切分成 N 份，发送给 N 个进程，然后在 `gather_dim` 上拼接接收到的数据。

**示例**：
- 输入：`[B, H, L/N, D]`，scatter_dim=2, gather_dim=1
- 输出：`[B, H/N, L, D]`

### 2. AttentionSP 类

```cpp
class AttentionSP {
public:
    torch::Tensor forward(torch::Tensor x, int64_t sp_size);
};
```

**核心逻辑**：
1. QKV 投影：`[B, L/N, dim]` → `[B, L/N, H, D]`
2. 转置：`[B, L/N, H, D]` → `[B, H, L/N, D]`
3. **All-to-All (seq→head)**：`[B, H, L/N, D]` → `[B, H/N, L, D]`
4. Attention 计算（在完整序列上）
5. **All-to-All (head→seq)**：`[B, H/N, L, D]` → `[B, H, L/N, D]`
6. 输出投影

### 3. TransformerSP 类

```cpp
class TransformerSP {
public:
    torch::Tensor forward(torch::Tensor tokens, int64_t sp_size);
};
```

**主流程**：
1. Embedding（所有 rank 相同）
2. Padding 到 sp_size 的倍数
3. 切分序列（每个 rank 取 L/N）
4. 通过 Attention layers
5. AllGather 恢复完整序列
6. 去除 padding

## 数据流可视化

以 4 个进程为例：

```
输入: [B=2, L=12] tokens (所有 rank 相同)
  ↓
Embedding: [2, 12, 256]
  ↓
Padding: [2, 12, 256] (12 已经能被 4 整除，无需 padding)
  ↓
Split by sequence:
  Rank 0: [2, 3, 256]   (tokens 0:3)
  Rank 1: [2, 3, 256]   (tokens 3:6)
  Rank 2: [2, 3, 256]   (tokens 6:9)
  Rank 3: [2, 3, 256]   (tokens 9:12)
  ↓
QKV Projection: [2, 3, 4, 64]
  ↓
Transpose: [2, 4, 3, 64]
  ↓
All-to-All (scatter_dim=2, gather_dim=1):
  Rank 0: [2, 1, 12, 64]  (head 0, 完整序列!)
  Rank 1: [2, 1, 12, 64]  (head 1, 完整序列!)
  Rank 2: [2, 1, 12, 64]  (head 2, 完整序列!)
  Rank 3: [2, 1, 12, 64]  (head 3, 完整序列!)
  ↓
Scaled Dot-Product Attention
  每个 rank 计算 1 个 head 的 attention
  ↓
All-to-All (scatter_dim=1, gather_dim=2):
  Rank 0: [2, 4, 3, 64]
  Rank 1: [2, 4, 3, 64]
  Rank 2: [2, 4, 3, 64]
  Rank 3: [2, 4, 3, 64]
  ↓
Transpose + Reshape: [2, 3, 256]
  ↓
Output Projection: [2, 3, 256]
  ↓
AllGather (gather_dim=1):
  [2, 12, 256] (完整输出)
```

## 参数配置

### 关键约束

1. **n_heads 必须能被 world_size 整除**
   - 例如：4 个进程，n_heads 必须是 4、8、12、16...
   - 原因：每个进程负责 n_heads/world_size 个 heads

2. **序列长度会自动 padding**
   - 如果 seq_len 不能被 world_size 整除，会自动填充
   - 最后会去除 padding

3. **所有 rank 的输入必须相同**
   - 在 Embedding 阶段，所有 rank 使用相同的 tokens
   - 切分发生在 Embedding 之后

## 与 Tensor Parallelism 的对比

| 维度 | Tensor Parallelism | Sequence Parallelism (本实现) |
|------|-------------------|----------------------------|
| 输入数据 | 所有 rank 相同 | 所有 rank 相同（初始），切分后不同 |
| 权重切分 | ✅ 切分权重 | ❌ 完整权重（可以组合 TP+SP） |
| 通信模式 | AllReduce | All-to-All + AllGather |
| 适用场景 | 模型太大 | 序列太长 |
| 内存节省 | 权重内存 | 激活内存 |

## 性能优化建议

### 1. 使用 NCCL（如果有 GPU）

当前实现使用 MPI (CPU)，如果有 GPU，修改为：

```cpp
// 创建 NCCL ProcessGroup
auto pg = c10d::ProcessGroupNCCL::createProcessGroupNCCL(...);

// 每个 rank 使用不同的 GPU
torch::Device device(torch::kCUDA, rank);
model.to(device);
```

### 2. 实现真正的 MPI_Alltoall

当前的 `all_to_all()` 使用多次 broadcast 模拟，效率较低。生产环境应该：

```cpp
// 直接调用 MPI_Alltoall
#include <mpi.h>

MPI_Alltoall(
    send_buffer, send_count, MPI_DATATYPE,
    recv_buffer, recv_count, MPI_DATATYPE,
    MPI_COMM_WORLD
);
```

### 3. 异步通信与计算重叠

```cpp
// 启动异步通信
auto work = pg->alltoall_base(...);

// 执行可以重叠的计算
do_some_computation();

// 等待通信完成
work->wait();
```

### 4. 组合 TP + SP

对于超大模型 + 超长序列：

```
8 GPUs 配置:
├─ Tensor Parallel (TP): 2x     ← 切分权重
└─ Sequence Parallel (SP): 4x   ← 切分序列

每个 GPU：
- 权重：1/2
- 序列：1/4
- 总内存占用：1/8
```

## 已知限制

1. **简化的 All-to-All 实现**
   - 使用多次 broadcast 模拟，不是真正的 MPI_Alltoall
   - 性能较低，仅用于演示

2. **简化的 Attention**
   - 没有实现 RoPE（Rotary Position Embedding）
   - Causal mask 使用简单的上三角矩阵

3. **单层 Transformer**
   - 只实现了一个 Attention layer
   - 实际模型需要多层 + FeedForward + LayerNorm

4. **CPU 限制**
   - 当前使用 MPI (CPU)，性能受限
   - 生产环境应使用 NCCL (GPU)

## 扩展到真实模型

要支持完整的 LLaMA 等模型，需要添加：

### 1. 多层 Transformer

```cpp
std::vector<std::shared_ptr<TransformerBlockSP>> layers_;
for (int i = 0; i < n_layers; ++i) {
    layers_.push_back(std::make_shared<TransformerBlockSP>(...));
}
```

### 2. RoPE (Rotary Position Embedding)

```cpp
torch::Tensor apply_rotary_emb(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor freqs_cis
);
```

### 3. FeedForward (SwiGLU)

```cpp
class FeedForwardSP {
    torch::Tensor w1, w2, w3;  // SwiGLU 需要 3 个权重

    torch::Tensor forward(torch::Tensor x) {
        return w2(F::silu(w1(x)) * w3(x));
    }
};
```

### 4. RMS Norm

```cpp
class RMSNorm {
    torch::Tensor weight;
    float eps;

    torch::Tensor forward(torch::Tensor x);
};
```

### 5. KV Cache (推理优化)

```cpp
class AttentionWithCache {
    std::vector<torch::Tensor> key_cache_;
    std::vector<torch::Tensor> value_cache_;

    torch::Tensor forward_with_cache(torch::Tensor x, int step);
};
```

## 测试验证

### 1. 与 Python 版本对比

```python
# Python 端生成参考输出
import torch
from llama2_model import Transformer, ModelArgs

model_args = ModelArgs(dim=256, n_layers=1, n_heads=4, vocab_size=320, sp_size=4)
model = Transformer.from_model_args(model_args)
tokens = torch.randint(0, 320, (2, 12))
output_ref = model(tokens)
torch.save(output_ref, "output_ref.pt")
```

```cpp
// C++ 端对比
auto output_ref = torch::jit::load("output_ref.pt");
auto output_cpp = model.forward(tokens, 4);
bool close = torch::allclose(output_cpp, output_ref, 1e-4, 1e-3);
std::cout << "Outputs match: " << (close ? "Yes" : "No") << std::endl;
```

### 2. 不同进程数测试

```bash
# 测试 2, 4, 8 进程
for np in 2 4 8; do
    echo "Testing with $np processes..."
    mpirun -np $np ./sequence-parallel-inference
done
```

### 3. 内存占用分析

```cpp
// 记录内存使用
auto memory_before = torch::cuda::memory_allocated(rank);
auto output = model.forward(tokens, sp_size);
auto memory_after = torch::cuda::memory_allocated(rank);
std::cout << "Memory used: " << (memory_after - memory_before) / 1e6 << " MB" << std::endl;
```

## 故障排查

### 问题 1: n_heads 不能被 world_size 整除

**错误信息**：
```
Error: n_heads (3) must be divisible by world_size (4)
```

**解决**：修改 n_heads 为 4 的倍数（4, 8, 12...）

### 问题 2: All-to-All 通信卡住

**症状**：程序在 all_to_all() 处卡住，不返回

**原因**：
- 某个 rank 提前退出
- 张量 shape 不一致
- 内存不连续

**解决**：
1. 确保所有 rank 都调用 all_to_all()
2. 打印每个 rank 的张量 shape，确保一致
3. 使用 `.contiguous()` 确保内存连续

### 问题 3: 输出结果不正确

**调试步骤**：
1. 检查每个 rank 的输入是否相同（Embedding 阶段）
2. 打印 split 后每个 rank 的序列片段
3. 检查 All-to-All 前后的 shape 变化
4. 验证 AllGather 是否正确恢复序列

## 参考资料

- **论文**: [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
- **Python 参考实现**: `/data/workspace/hunyuan_ptm/torch_examples/distributed/tensor_parallelism/`
- **Tensor Parallel 实现**: `tp_demo_infer.cpp`
- **详细指南**: `TENSOR_PARALLEL_GUIDE.md`

## 贡献与反馈

如有问题或改进建议，欢迎：
1. 查看完整指南：`TENSOR_PARALLEL_GUIDE.md`
2. 对比 Python 实现：`deepspeed_tp_example.py`
3. 提交 Issue 或 PR

---

**实现状态**: ✅ 核心功能完成，可用于学习和原型验证
**生产就绪**: ❌ 需要优化 All-to-All 实现和添加完整 Transformer 组件

**作者**: Claude (Claude-4.5-Sonnet)
**日期**: 2026-02-09
