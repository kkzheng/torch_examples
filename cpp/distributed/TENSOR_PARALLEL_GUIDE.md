# 模型并行推理使用指南

## 概述

这个指南介绍了在 libtorch (C++) 中实现大模型并行推理的两种主要方法：

1. **Tensor Parallelism (TP)** - 切分模型权重到多个设备
2. **Sequence Parallelism (SP)** - 切分序列到多个设备（特别是 DeepSpeed Ulysses）

由于 libtorch 不提供 Python 中的 `parallelize_module`、`ColwiseParallel`、`DTensor` 等高级 API，我们需要手动实现模型切分和通信。

---

# 第一部分：Tensor Parallelism (TP)

## TP 核心思路（Megatron-LM 风格）

```
Input [B, 1024]
    |
    v
FC1 (Column-wise split)
    GPU 0: [B, 1024] x [2048, 1024] -> [B, 2048]
    GPU 1: [B, 1024] x [2048, 1024] -> [B, 2048]
    (无需通信)
    |
    v
ReLU
    |
    v
FC2 (Row-wise split)
    GPU 0: [B, 2048] x [512, 2048] -> [B, 512]
    GPU 1: [B, 2048] x [512, 2048] -> [B, 512]
    |
    v
AllReduce (SUM)
    |
    v
Output [B, 512]
```

### TP 关键点

1. **Column-wise Parallel (第一层)**
   - 权重按输出维度切分：`[hidden_size, input_size]` → `[hidden_size/N, input_size]`
   - 每个进程计算部分输出
   - 无需通信（输入相同）

2. **Row-wise Parallel (第二层)**
   - 权重按输入维度切分：`[output_size, hidden_size]` → `[output_size, hidden_size/N]`
   - 每个进程计算部分结果
   - 需要 AllReduce 合并

3. **通信最小化**
   - 整个前向传播只有一次 AllReduce
   - 相比每层都通信，性能显著提升

4. **数据分布**
   - **输入：所有 rank 相同** - 每个进程处理相同的输入数据
   - **权重：切分到不同 rank** - 每个进程持有部分模型权重
   - **输出：AllReduce 合并** - 通过 AllReduce 得到完整输出

## TP 编译与运行

### 编译

```bash
cd /data/workspace/hunyuan_ptm/torch_examples/cpp/distributed/build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make tensor-parallel-inference
```

### 运行

使用 MPI 运行（不需要 GPU）：

```bash
# 4 个进程，每个进程处理模型的 1/4
mpirun -np 4 ./tensor-parallel-inference
```

输出示例：
```
Rank 0/4 started
Rank 1/4 started
Rank 2/4 started
Rank 3/4 started
Rank 0 loaded sharded weights:
  fc1_weight_shard: [1024, 1024]
  fc2_weight_shard: [512, 1024]
...
Rank 0 inference completed. Output shape: [8, 512]
```

## TP 代码结构

### TensorParallelInference 类

```cpp
class TensorParallelInference {
public:
  // 构造函数：接收 rank、world_size 和 ProcessGroup
  TensorParallelInference(int rank, int world_size,
                          c10::intrusive_ptr<c10d::ProcessGroupMPI> pg);

  // 加载模型并切分权重
  void load_and_shard_model(const std::string& model_path,
                           int64_t input_size,
                           int64_t hidden_size,
                           int64_t output_size);

  // Tensor Parallel 前向传播
  torch::Tensor forward(torch::Tensor input);

  // 移动到指定设备（CPU 或 CUDA）
  void to(torch::Device device);
};
```

### 主要步骤

1. **初始化 MPI**
   ```cpp
   auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
   ```

2. **加载并切分模型**
   - Rank 0 加载完整权重
   - 广播到所有 rank
   - 每个 rank 切分自己的部分

3. **推理**
   - 所有 rank 使用相同的输入（使用相同的随机种子）
   - 每个 rank 计算部分结果
   - AllReduce 合并最终输出

---

# 第二部分：Sequence Parallelism (SP) - DeepSpeed Ulysses

## SP 概述

Sequence Parallelism 与 Tensor Parallelism 的核心区别：

| 维度 | Tensor Parallelism (TP) | Sequence Parallelism (SP) |
|------|------------------------|--------------------------|
| **数据切分** | 输入相同，权重切分 | 输入序列切分，权重可以完整或切分 |
| **适用场景** | 模型太大放不下单卡 | 序列太长放不下单卡（如 >100K tokens） |
| **通信模式** | AllReduce | All-to-All |
| **内存瓶颈** | 模型权重 | Attention 中间激活（O(seq^2)） |

### DeepSpeed Ulysses 的核心思想

DeepSpeed Ulysses 是一种高效的 Sequence Parallelism 实现（论文：[DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)）。

**关键创新**：通过 **All-to-All** 通信，在 Attention 计算前后动态切换数据布局。

```
阶段 1: Sequence Parallel 布局
  每个 GPU: [B, L/N, H, D]  ← 序列被切分

      ↓ All-to-All (scatter_dim=1, gather_dim=2)

阶段 2: Head Parallel 布局
  每个 GPU: [B, L, H/N, D]  ← 完整序列，头被切分

      ↓ Scaled Dot-Product Attention (on full sequence!)

阶段 3: Head Parallel 输出
  每个 GPU: [B, L, H/N, D]

      ↓ All-to-All (scatter_dim=2, gather_dim=1)

阶段 4: Sequence Parallel 布局
  每个 GPU: [B, L/N, H, D]  ← 又回到序列切分
```

### 为什么需要 All-to-All？

**问题**：Attention 需要看到完整的序列来计算 Q·K^T

**解决方案**：
1. 在 QKV 投影后，使用 All-to-All 将"序列并行"转换为"头并行"
2. 每个 GPU 现在有完整序列，但只计算部分 attention heads
3. Attention 计算后，再次使用 All-to-All 转换回"序列并行"

### SP 完整数据流（4 GPU 示例）

```
输入: [B, L=4096] tokens
  ↓
Embedding: [B, L, C]
  ↓
Padding to divisible by 4: [B, L', C]  (L' ≥ L)
  ↓
Split by sequence dim
  GPU 0: [B, L'/4, C]  (tokens 0:1024)
  GPU 1: [B, L'/4, C]  (tokens 1024:2048)
  GPU 2: [B, L'/4, C]  (tokens 2048:3072)
  GPU 3: [B, L'/4, C]  (tokens 3072:4096)
  ↓
QKV Projection: [B, L'/4, n_heads=32, head_dim]
  ↓
All-to-All (dim 1→2, 维度交换)
  GPU 0: [B, L', 32/4=8, head_dim]  (heads 0:8, 完整序列!)
  GPU 1: [B, L', 8, head_dim]       (heads 8:16, 完整序列!)
  GPU 2: [B, L', 8, head_dim]       (heads 16:24, 完整序列!)
  GPU 3: [B, L', 8, head_dim]       (heads 24:32, 完整序列!)
  ↓
Scaled Dot-Product Attention (is_causal=True)
  每个 GPU 独立计算其负责的 8 个 heads
  ↓
All-to-All (dim 2→1, 维度交换回来)
  GPU 0: [B, L'/4, 32, head_dim]
  GPU 1: [B, L'/4, 32, head_dim]
  GPU 2: [B, L'/4, 32, head_dim]
  GPU 3: [B, L'/4, 32, head_dim]
  ↓
Output Projection: [B, L'/4, C]
  ↓
AllGather (合并序列维度)
  [B, L', C] → 去除 padding → [B, L, C]
```

## SP 关键代码实现（Python 参考）

### 1. All-to-All 通信原语

```python
class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scatter_dim, gather_dim, cur_group, async_op):
        """
        Args:
            tensor: 输入张量
            scatter_dim: 要切分并发送的维度
            gather_dim: 要收集并拼接的维度
            cur_group: 进程组

        返回:
            重新排列后的张量
        """
        group_size = dist.get_world_size(group=cur_group)

        # 1. 按 scatter_dim 切分成 N 份
        scatter_list = list(torch.chunk(tensor, chunks=group_size, dim=scatter_dim))

        # 2. 准备接收缓冲区
        gather_list = [torch.zeros_like(x) for x in scatter_list]

        # 3. All-to-All 通信
        # Rank i 发送 scatter_list[j] 给 Rank j
        # Rank i 从 Rank j 接收数据到 gather_list[j]
        dist.all_to_all(gather_list, scatter_list, group=cur_group, async_op=async_op)

        # 4. 按 gather_dim 拼接
        return torch.cat(gather_list, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_outputs):
        # 反向传播：scatter 和 gather 维度互换
        return (all2all(grad_outputs, ctx.gather_dim, ctx.scatter_dim,
                       ctx.cur_group, False),
                None, None, None, None)
```

### 2. Attention 中使用 All-to-All

```python
class AttentionWithSP(nn.Module):
    def forward(self, x, freqs_cis, sp_size, sp_group):
        bsz, seqlen, _ = x.shape

        # QKV projection
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention: [B, L, H, D] -> [B, H, L, D]
        xq = xq.transpose(1, 2)  # [B, n_heads, L/N, D]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if sp_size > 1:
            # 关键！All-to-All: Sequence Parallel -> Head Parallel
            # scatter_dim=2 (seq), gather_dim=1 (heads)
            xq = _All2All.apply(xq, 2, 1, sp_group, False)  # [B, H/N, L, D]
            xk = _All2All.apply(xk, 2, 1, sp_group, False)
            xv = _All2All.apply(xv, 2, 1, sp_group, False)

        # Scaled Dot-Product Attention (现在有完整序列!)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        # output: [B, H/N, L, D]

        if sp_size > 1:
            # 关键！All-to-All: Head Parallel -> Sequence Parallel
            # scatter_dim=1 (heads), gather_dim=2 (seq)
            output = _All2All.apply(output, 1, 2, sp_group, False)  # [B, H, L/N, D]

        # Transpose back: [B, H, L/N, D] -> [B, L/N, H, D]
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)

        return self.wo(output)
```

### 3. 序列切分与填充

```python
def minimal_pad_to_divisible(tensor, sp_size, dim=1, pad_value=0.0):
    """
    对序列进行最小化 padding，使长度能被 sp_size 整除

    Args:
        tensor: [B, L, C]
        sp_size: 进程数
        dim: 要 padding 的维度（默认 1 = sequence dim）

    Returns:
        padded_tensor: [B, L', C] where L' % sp_size == 0
        padding_len: 添加的 padding 长度
    """
    current_size = tensor.size(dim)
    padding_len = (sp_size - current_size % sp_size) % sp_size

    if padding_len == 0:
        return tensor, 0

    # 构造 padding 参数（F.pad 从最后一维开始）
    padding_dims = [0] * (2 * tensor.dim())
    pad_index = 2 * (tensor.dim() - dim - 1) + 1
    padding_dims[pad_index] = padding_len

    return F.pad(tensor, tuple(padding_dims), mode='constant', value=pad_value), padding_len
```

### 4. Transformer 主流程

```python
class TransformerWithSP(nn.Module):
    def forward(self, tokens):
        bsz, seqlen = tokens.shape

        # 1. Embedding (在所有 ranks 上完整计算)
        h = self.tok_embeddings(tokens)  # [B, L, C]

        if self.sp_size > 1:
            # 2. Padding 使序列长度能被 sp_size 整除
            rank_in_sp = dist.get_group_rank(self.sp_group, dist.get_rank())
            h, padding_len = minimal_pad_to_divisible(h, self.sp_size, dim=1)

            # 3. 切分序列到不同 ranks
            h = torch.chunk(h, self.sp_size, dim=1)[rank_in_sp]  # [B, L/N, C]

            # 4. 同样切分 freqs_cis
            freqs_cis = self.freqs_cis[:seqlen + padding_len]
            freqs_cis = torch.chunk(freqs_cis, self.sp_size, dim=0)[rank_in_sp]

        # 5. 通过所有 Transformer layers
        for layer in self.layers:
            if self.sp_size > 1:
                h = layer(h, freqs_cis, sp_size=self.sp_size, sp_group=self.sp_group)
            else:
                h = layer(h, freqs_cis)

        if self.sp_size > 1:
            # 6. AllGather 恢复完整序列
            h = _Allgather.apply(h, 1, self.sp_group, False)  # [B, L', C]

            # 7. 去除 padding
            if padding_len > 0:
                h = h[:, :-padding_len, :]  # [B, L, C]

        # 8. Layer Norm + Output projection
        h = self.norm(h)
        output = self.output(h).float()
        return output
```

## SP vs TP 对比总结

| 特性 | Tensor Parallelism | Sequence Parallelism (Ulysses) |
|------|-------------------|-------------------------------|
| **切分对象** | 模型权重 | 输入序列 |
| **输入数据** | 所有 rank 相同 | 每个 rank 不同（序列片段） |
| **通信原语** | AllReduce (Reduce-Scatter + AllGather) | All-to-All |
| **通信次数** | 每层 1-2 次 | Attention 前后各 1 次 |
| **内存节省** | 模型权重：1/N | 激活值：1/N |
| **计算负载** | 每个 rank 计算量相同 | 每个 rank 计算量相同 |
| **扩展性** | 受限于模型维度（如 hidden_size, n_heads） | 受限于序列长度 |
| **适用场景** | 模型太大 | 序列太长 |
| **可组合性** | 可与 SP、PP 组合 | 可与 TP、PP 组合 |

### 组合使用示例

对于超大模型 + 超长序列：

```
World Size = 64 GPUs
├─ Data Parallel (DP): 2x
├─ Pipeline Parallel (PP): 4x
├─ Tensor Parallel (TP): 4x     ← 切分模型权重
└─ Sequence Parallel (SP): 2x   ← 切分序列

每个 GPU：
- 模型权重：1/4（由 TP 切分）
- 序列长度：1/2（由 SP 切分）
- 激活内存：1/(4*2) = 1/8
```

## 在 libtorch (C++) 中实现 SP

### 挑战

1. **All-to-All 实现**
   - MPI 的 `MPI_Alltoall` 或 `MPI_Alltoallv`
   - NCCL 的 `ncclGroupStart/ncclSend/ncclRecv/ncclGroupEnd`

2. **动态 Reshape 和转置**
   - `torch::chunk()` - 切分张量
   - `torch::cat()` - 拼接张量
   - `transpose()` - 维度转置

3. **反向传播**
   - 需要实现自定义 autograd Function（类似 Python 的 `torch.autograd.Function`）
   - C++ 中需要手动管理梯度流

### 简化的 C++ All-to-All 伪代码

```cpp
torch::Tensor all_to_all_cpp(
    torch::Tensor input,
    int scatter_dim,
    int gather_dim,
    c10::intrusive_ptr<c10d::ProcessGroup> pg
) {
    int world_size = pg->getSize();
    int rank = pg->getRank();

    // 1. 切分输入
    auto scatter_list = torch::chunk(input, world_size, scatter_dim);

    // 2. 准备接收缓冲区
    std::vector<torch::Tensor> gather_list;
    for (int i = 0; i < world_size; ++i) {
        gather_list.push_back(torch::zeros_like(scatter_list[i]));
    }

    // 3. MPI All-to-All (需要转换为 MPI 数据类型)
    // 这是最复杂的部分 - 需要处理步长、数据类型等
    std::vector<torch::Tensor> send_vec, recv_vec;
    for (int i = 0; i < world_size; ++i) {
        send_vec.push_back(scatter_list[i].contiguous());
        recv_vec.push_back(gather_list[i]);
    }

    // 使用 ProcessGroup 的 alltoall 方法
    pg->alltoall(recv_vec, send_vec)->wait();

    // 4. 拼接结果
    return torch::cat(gather_list, gather_dim).contiguous();
}
```

### 实现建议

对于 C++ 实现 Sequence Parallelism：

1. **从 Python 原型开始**
   - 先在 Python 中验证算法正确性
   - 使用 Python 生成测试数据和期望输出

2. **逐步移植到 C++**
   - 先实现 all_to_all 通信原语
   - 再实现 Attention 层的 SP 版本
   - 最后集成到完整的 Transformer

3. **测试策略**
   - 与 Python 版本进行数值对比（forward pass）
   - 使用小模型和短序列进行调试
   - 逐渐增加规模验证性能

4. **性能优化**
   - 使用 NCCL 代替 MPI（GPU 上）
   - 重叠通信与计算
   - 使用异步操作（async_op=True）

---

# 第三部分：TP 扩展与性能优化

当前示例只实现了两个 Linear 层。要支持真实的大模型（如 LLaMA），需要扩展：

### 1. 支持更多层类型

```cpp
// Attention 层的 TP
class AttentionTP {
  // Q, K, V 都是 column-wise parallel
  torch::Tensor q_proj_shard, k_proj_shard, v_proj_shard;
  // Output projection 是 row-wise parallel
  torch::Tensor o_proj_shard;
};

// Layer Norm (复制到所有 rank)
class LayerNormTP {
  torch::Tensor weight, bias;  // 不切分
};
```

### 2. 从磁盘加载权重

```cpp
void load_and_shard_model(const std::string& model_path, ...) {
  // 直接从磁盘加载分片（避免广播开销）
  std::string shard_path = model_path + ".shard." + std::to_string(rank_);
  torch::load(fc1_weight_shard_, shard_path + "/fc1_weight.pt");
  torch::load(fc1_bias_shard_, shard_path + "/fc1_bias.pt");
  // ...
}
```

### 3. 支持 CUDA

```cpp
// 每个 rank 使用不同的 GPU
torch::Device device(torch::kCUDA, rank_);
tp_model.to(device);
```

---

# 第三部分：通用内容

## Python 端准备权重

可以在 Python 中训练模型并导出分片：

```python
import torch
from torch.distributed.tensor.parallel import parallelize_module

# 训练完成后，导出分片
def export_sharded_weights(model, world_size, output_dir):
    for rank in range(world_size):
        # 模拟切分逻辑
        fc1_weight_shard = model.fc1.weight[
            rank * hidden_size // world_size :
            (rank + 1) * hidden_size // world_size
        ]
        torch.save(fc1_weight_shard, f"{output_dir}/rank{rank}/fc1_weight.pt")
        # ... 保存其他层
```

---

# 第四部分：参考资料与推荐框架

## 学术论文

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
   - https://arxiv.org/abs/1909.08053
   - Tensor Parallelism 的经典论文

2. **Reducing Activation Recomputation in Large Transformer Models**
   - https://arxiv.org/abs/2205.05198
   - Megatron-LM 风格的 Sequence Parallelism

3. **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**
   - https://arxiv.org/abs/2309.14509
   - DeepSpeed Ulysses (All-to-All based SP)

4. **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**
   - https://arxiv.org/abs/2304.11277
   - FSDP (用于训练，非推理)

## 开源实现参考

### Python 实现
- **PyTorch Tensor Parallel**: `torch.distributed.tensor.parallel`
- **DeepSpeed**: https://github.com/microsoft/DeepSpeed
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM

### C++ 实现参考
- **本项目**: `tp_demo_infer.cpp` (TP 基础实现)
- **PyTorch C++ API**: libtorch distributed primitives

## 性能考虑

1. **通信开销**
   - **TP**: AllReduce 在每层之后，通信频繁
   - **SP**: All-to-All 只在 Attention，但数据量大（O(B*L*H*D)）
   - MPI AllReduce/All-to-All 在 CPU 上较慢
   - 使用 NCCL（GPU）可以显著加速（10-100x）

2. **内存节省**
   - **TP**: 每个进程只保存 1/N 的模型权重
     - 4 进程可以运行 4 倍大的模型
   - **SP**: 每个进程只保存 1/N 的序列激活
     - 4 进程可以处理 4 倍长的序列
   - 可以组合使用 TP + SP 同时节省权重和激活内存

3. **计算效率**
   - **TP**: 单个进程的计算量减少（权重少了）
   - **SP**: 单个进程的计算量减少（序列短了）
   - 总体吞吐量取决于通信/计算比
   - GPU 利用率：通信时 GPU 可能空闲（除非重叠）

4. **通信优化技巧**
   - **异步通信**: `async_op=True` + 计算与通信重叠
   - **通信融合**: 合并多个小通信为一个大通信
   - **梯度累积**: 减少通信频率（训练时）
   - **混合精度**: FP16/BF16 减少通信数据量

## 生产环境推荐框架

## 生产环境推荐框架

对于真实的大模型推理，建议使用成熟框架（除非有特殊需求）：

### 推荐框架对比

| 框架 | TP 支持 | SP 支持 | 语言 | 适用场景 |
|------|--------|--------|------|---------|
| **TensorRT-LLM** | ✅ 优秀 | ✅ 有 | C++/Python | NVIDIA GPU，性能最优 |
| **vLLM** | ✅ 优秀 | ✅ 有 | Python | 高吞吐批处理推理 |
| **DeepSpeed-Inference** | ✅ 优秀 | ✅ 优秀 (Ulysses) | Python | 灵活性高，功能全面 |
| **FasterTransformer** | ✅ 优秀 | ❌ 无 | C++/Python | 已归档，但性能仍优秀 |
| **llama.cpp** | ❌ 无 | ❌ 无 | C++ | CPU 推理，量化优化 |

### 框架特性

**TensorRT-LLM** (推荐用于生产):
**TensorRT-LLM** (推荐用于生产):
- 开箱即用的 TP/PP 支持
- 高度优化的 CUDA kernel（Flash Attention, fused ops）
- 支持多种量化（INT8, FP8, INT4）
- 最佳性能和延迟

**vLLM**:
- PagedAttention 优化 KV Cache
- Continuous batching（动态批处理）
- 高吞吐量（适合服务多用户）
- Python API 简单易用

**DeepSpeed-Inference**:
- 支持 TP + PP + ZeRO
- Ulysses Sequence Parallelism
- Kernel 注入（替换 PyTorch 原生算子）
- 与 DeepSpeed 训练无缝衔接

**FasterTransformer** (已归档但仍可用):
- 纯 C++/CUDA 实现
- 低延迟优化
- 支持 TP + PP
- 需要手动集成

### 何时使用本项目的实现？

适合以下场景：
- **学习目的**: 理解 TP/SP 的底层原理
- **特殊需求**: 需要自定义通信拓扑或算法
- **嵌入式部署**: 需要极简依赖（仅 libtorch + MPI）
- **研究原型**: 快速验证新的并行策略

不适合：
- 生产环境（除非有充分理由）
- 追求极致性能
- 需要开箱即用的功能

---

# 第五部分：故障排查与调试

## 故障排查

### 编译错误：找不到 c10d 命名空间

确保：
1. 定义了 `USE_C10D_MPI` 宏
2. 链接了正确的 libtorch 库
3. libtorch 编译时启用了 MPI 支持

### 运行时错误：MPI 未初始化

确保使用 `mpirun` 启动程序，而不是直接运行可执行文件。

### 不同 rank 的输出不一致

确保：
1. 所有 rank 使用相同的随机种子
2. 所有 rank 的输入数据相同（对于 TP）或正确切分（对于 SP）
3. 权重广播正确完成
4. 使用 `torch::allclose()` 检查数值差异

### All-to-All 通信失败

症状：`MPI_Alltoall` 或 `dist.all_to_all` 报错

解决：
1. 确保所有 rank 发送/接收的张量 shape 一致
2. 检查是否有 rank 提前退出
3. 使用 `contiguous()` 确保内存连续
4. 验证进程组（ProcessGroup）配置正确

### 内存不足（OOM）

症状：CUDA out of memory 或 CPU OOM

解决：
1. **减少 batch size**: 最直接的方法
2. **增加并行度**: 使用更多 GPU/进程
3. **Gradient checkpointing**: 训练时重计算激活（推理不适用）
4. **混合精度**: FP16/BF16 减少内存占用
5. **量化**: INT8/INT4（需要框架支持）

### 调试技巧

1. **逐步验证**:
   ```python
   # 1. 单进程验证
   python your_model.py

   # 2. 2 进程验证
   mpirun -np 2 python your_model.py

   # 3. 逐渐增加到目标进程数
   mpirun -np 4 python your_model.py
   ```

2. **数值对比**:
   ```python
   # Python 端生成参考输出
   output_ref = model(input)
   torch.save(output_ref, "output_ref.pt")

   # C++ 端对比
   auto output_cpp = tp_model.forward(input);
   auto output_ref = torch::load("output_ref.pt");
   assert(torch::allclose(output_cpp, output_ref, 1e-5, 1e-3));
   ```

3. **打印中间结果**:
   ```cpp
   if (rank == 0) {
       std::cout << "fc1 output shape: " << h.sizes() << std::endl;
       std::cout << "fc1 output sample: " << h.slice(0, 0, 1) << std::endl;
   }
   ```

4. **使用调试工具**:
   - `gdb` with MPI: `mpirun -np 4 xterm -e gdb ./program`
   - NCCL 调试: `export NCCL_DEBUG=INFO`
   - PyTorch 分布式调试: `export TORCH_DISTRIBUTED_DEBUG=DETAIL`

---

# 附录

## A. 关键概念速查表

| 术语 | 含义 | 示例 |
|------|------|------|
| **Rank** | 进程编号 (0 到 world_size-1) | Rank 0, Rank 1, ... |
| **World Size** | 总进程数 | 4 GPUs → world_size=4 |
| **Column-wise Parallel** | 按列（输出维度）切分权重 | `[H, I]` → `[H/N, I]` |
| **Row-wise Parallel** | 按行（输入维度）切分权重 | `[O, H]` → `[O, H/N]` |
| **AllReduce** | 所有进程归约（求和）并广播结果 | TP 的最后一步 |
| **All-to-All** | 所有进程互相交换数据 | SP (Ulysses) 的核心 |
| **AllGather** | 收集所有进程的数据 | SP 恢复完整序列 |
| **Broadcast** | Rank 0 广播数据到所有 Rank | 分发权重 |
| **Scatter** | 切分并分发数据 | 数据并行的输入切分 |
| **ProcessGroup** | 进程组（通信域） | MPI_COMM_WORLD |

## B. libtorch Distributed API 参考

```cpp
// ProcessGroup 创建
auto pg_mpi = c10d::ProcessGroupMPI::createProcessGroupMPI();
auto pg_nccl = c10d::ProcessGroupNCCL::createProcessGroupNCCL(...);

// 基础信息
int rank = pg->getRank();
int world_size = pg->getSize();

// 集合通信
std::vector<torch::Tensor> tensors = {tensor};

// AllReduce (求和)
pg->allreduce(tensors, c10d::ReduceOp::SUM)->wait();

// Broadcast (从 rank 0 广播)
pg->broadcast(tensors, 0)->wait();

// AllGather
std::vector<torch::Tensor> gather_list(world_size);
for (int i = 0; i < world_size; ++i) {
    gather_list[i] = torch::empty_like(tensor);
}
pg->allgather({gather_list}, {tensor})->wait();

// All-to-All (需要自己实现或使用 MPI 直接调用)
// libtorch 的 ProcessGroup 没有直接的 alltoall API
// 需要使用 MPI_Alltoall 或通过 Send/Recv 实现
```

## C. 常用命令

```bash
# 编译
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make

# MPI 运行（CPU）
mpirun -np 4 ./tensor-parallel-inference

# MPI 运行（GPU，每个 rank 使用不同 GPU）
mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 ./tensor-parallel-inference

# 调试单个 rank
mpirun -np 4 xterm -e gdb --args ./tensor-parallel-inference

# 性能分析
mpirun -np 4 nsys profile -o profile_%q{OMPI_COMM_WORLD_RANK} ./program

# 检查 MPI 安装
mpirun --version
which mpirun

# 检查 NCCL
python -c "import torch; print(torch.cuda.nccl.version())"
```

## D. 延伸阅读

### 博客文章
- [Megatron-LM 源码解读](https://zhuanlan.zhihu.com/p/366906920)
- [DeepSpeed Ulysses 详解](https://zhuanlan.zhihu.com/p/496065391)
- [PyTorch Distributed 教程](https://pytorch.org/tutorials/beginner/dist_overview.html)

### 视频教程
- [NVIDIA GTC: Megatron-LM Training](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)
- [PyTorch Conference: Distributed Training](https://www.youtube.com/watch?v=Cvdhwx-OBBo)

### 实践项目
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [vLLM GitHub](https://github.com/vllm-project/vllm)

---

**文档版本**: v2.0
**最后更新**: 2026-02-09
**贡献者**: Claude (Claude-4.5-Sonnet)

如有问题或建议，欢迎提 Issue！
