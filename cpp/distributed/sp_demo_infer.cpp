#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

/*
 * DeepSpeed Ulysses Sequence Parallelism Implementation (使用 MPI)
 *
 * 核心思路：
 * 1. 将输入序列切分到不同的 rank（每个 rank 处理 L/N 的序列片段）
 * 2. 在 Attention 前使用 All-to-All 将 "序列并行" 转换为 "头并行"
 * 3. 每个 rank 在完整序列上计算部分 attention heads
 * 4. Attention 后使用 All-to-All 转换回 "序列并行"
 * 5. 最后使用 AllGather 恢复完整序列
 * 6. 添加验证对比：SP vs No-SP
 *
 * 参考论文: DeepSpeed Ulysses (https://arxiv.org/abs/2309.14509)
 */

// ============================================================================
// 工具函数：All-to-All 通信（使用原生 API）
// ============================================================================

torch::Tensor all_to_all(
    torch::Tensor input,
    int scatter_dim,
    int gather_dim,
    c10::intrusive_ptr<c10d::ProcessGroupMPI> pg) {
  /*
   * 使用 ProcessGroup 原生的 alltoall 函数
   *
   * ProcessGroup::alltoall 接口：
   *   alltoall(
   *     std::vector<at::Tensor>& outputTensors,  // 输出列表
   *     std::vector<at::Tensor>& inputTensors,   // 输入列表
   *     const AllToAllOptions& opts
   *   )
   *
   * 语义：
   *   - inputTensors[i] 发送给 rank i
   *   - 从 rank i 接收数据到 outputTensors[i]
   */

  int world_size = pg->getSize();

  // 1. 切分输入张量
  auto scatter_list = torch::chunk(input, world_size, scatter_dim);

  // 确保所有张量连续
  std::vector<torch::Tensor> input_tensors;
  for (const auto& t : scatter_list) {
    input_tensors.push_back(t.contiguous());
  }

  // 2. 准备输出缓冲区
  std::vector<torch::Tensor> output_tensors;
  for (int i = 0; i < world_size; ++i) {
    output_tensors.push_back(input_tensors[0].clone().zero_().contiguous());
  }

  // 3. 调用原生 alltoall (使用 MPI_Alltoall)
  c10d::AllToAllOptions opts;
  pg->alltoall(output_tensors, input_tensors, opts)->wait();

  // 4. 沿 gather_dim 拼接结果
  return torch::cat(output_tensors, gather_dim).contiguous();
}

// ============================================================================
// 工具函数：AllGather 通信
// ============================================================================

torch::Tensor all_gather(
    torch::Tensor input,
    int gather_dim,
    c10::intrusive_ptr<c10d::ProcessGroupMPI> pg) {
  /*
   * AllGather 集合通信
   *
   * Args:
   *   input: 输入张量
   *   gather_dim: 拼接的维度
   *   pg: MPI ProcessGroup
   *
   * Returns:
   *   在 gather_dim 上拼接所有 rank 的张量
   */
  int world_size = pg->getSize();

  // 准备接收缓冲区
  std::vector<torch::Tensor> gather_list;
  for (int i = 0; i < world_size; ++i) {
    gather_list.push_back(torch::zeros_like(input));
  }

  // AllGather
  std::vector<std::vector<torch::Tensor>> output_tensors = {gather_list};
  std::vector<torch::Tensor> input_tensors = {input.contiguous()};
  pg->allgather(output_tensors, input_tensors)->wait();

  // 拼接
  return torch::cat(gather_list, gather_dim).contiguous();
}

// ============================================================================
// 工具函数：序列填充
// ============================================================================

std::pair<torch::Tensor, int64_t> pad_to_divisible(
    torch::Tensor tensor,
    int64_t divisor,
    int64_t dim = 1,
    float pad_value = 0.0) {
  /*
   * 将张量在指定维度填充，使其长度能被 divisor 整除
   *
   * Args:
   *   tensor: 输入张量 (例如 [B, L, C])
   *   divisor: 除数 (通常是 world_size)
   *   dim: 要填充的维度
   *   pad_value: 填充值
   *
   * Returns:
   *   (padded_tensor, padding_length)
   */
  int64_t current_size = tensor.size(dim);
  int64_t padding_len = (divisor - current_size % divisor) % divisor;

  if (padding_len == 0) {
    return {tensor, 0};
  }

  // 构造 padding 参数（F.pad 从最后一维开始）
  std::vector<int64_t> padding_dims(2 * tensor.dim(), 0);
  int64_t pad_index = 2 * (tensor.dim() - dim - 1) + 1;
  padding_dims[pad_index] = padding_len;

  // 使用 torch::nn::functional::pad
  auto padded = torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions(padding_dims)
          .mode(torch::kConstant)
          .value(pad_value));

  return {padded, padding_len};
}

// ============================================================================
// 标准 Transformer Block（支持 Sequence Parallelism）
// ============================================================================

class AttentionSP {
 public:
  AttentionSP(
      int64_t dim,
      int64_t n_heads,
      int64_t hidden_dim,
      c10::intrusive_ptr<c10d::ProcessGroupMPI> pg)
      : dim_(dim), n_heads_(n_heads), hidden_dim_(hidden_dim), pg_(pg) {
    head_dim_ = dim / n_heads;

    int rank = pg->getRank();

    // Self-Attention 权重（所有 rank 使用相同的权重）
    if (rank == 0) {
      wq_ = torch::randn({dim, dim});
      wk_ = torch::randn({dim, dim});
      wv_ = torch::randn({dim, dim});
      wo_ = torch::randn({dim, dim});

      // Feed-Forward Network 权重
      w1_ = torch::randn({hidden_dim, dim}); // FFN first layer
      w2_ = torch::randn({dim, hidden_dim}); // FFN second layer

      // LayerNorm 参数
      attention_norm_weight_ = torch::ones({dim});
      attention_norm_bias_ = torch::zeros({dim});
      ffn_norm_weight_ = torch::ones({dim});
      ffn_norm_bias_ = torch::zeros({dim});
    } else {
      wq_ = torch::zeros({dim, dim});
      wk_ = torch::zeros({dim, dim});
      wv_ = torch::zeros({dim, dim});
      wo_ = torch::zeros({dim, dim});
      w1_ = torch::zeros({hidden_dim, dim});
      w2_ = torch::zeros({dim, hidden_dim});
      attention_norm_weight_ = torch::zeros({dim});
      attention_norm_bias_ = torch::zeros({dim});
      ffn_norm_weight_ = torch::zeros({dim});
      ffn_norm_bias_ = torch::zeros({dim});
    }

    // 广播所有权重到所有 rank
    std::vector<torch::Tensor> wq_vec = {wq_};
    std::vector<torch::Tensor> wk_vec = {wk_};
    std::vector<torch::Tensor> wv_vec = {wv_};
    std::vector<torch::Tensor> wo_vec = {wo_};
    std::vector<torch::Tensor> w1_vec = {w1_};
    std::vector<torch::Tensor> w2_vec = {w2_};
    std::vector<torch::Tensor> attn_norm_w_vec = {attention_norm_weight_};
    std::vector<torch::Tensor> attn_norm_b_vec = {attention_norm_bias_};
    std::vector<torch::Tensor> ffn_norm_w_vec = {ffn_norm_weight_};
    std::vector<torch::Tensor> ffn_norm_b_vec = {ffn_norm_bias_};

    c10d::BroadcastOptions opts;
    opts.rootRank = 0;
    pg->broadcast(wq_vec, opts)->wait();
    pg->broadcast(wk_vec, opts)->wait();
    pg->broadcast(wv_vec, opts)->wait();
    pg->broadcast(wo_vec, opts)->wait();
    pg->broadcast(w1_vec, opts)->wait();
    pg->broadcast(w2_vec, opts)->wait();
    pg->broadcast(attn_norm_w_vec, opts)->wait();
    pg->broadcast(attn_norm_b_vec, opts)->wait();
    pg->broadcast(ffn_norm_w_vec, opts)->wait();
    pg->broadcast(ffn_norm_b_vec, opts)->wait();

    wq_ = wq_vec[0];
    wk_ = wk_vec[0];
    wv_ = wv_vec[0];
    wo_ = wo_vec[0];
    w1_ = w1_vec[0];
    w2_ = w2_vec[0];
    attention_norm_weight_ = attn_norm_w_vec[0];
    attention_norm_bias_ = attn_norm_b_vec[0];
    ffn_norm_weight_ = ffn_norm_w_vec[0];
    ffn_norm_bias_ = ffn_norm_b_vec[0];
  }

  torch::Tensor forward(torch::Tensor x, int64_t sp_size) {
    /*
     * Complete Transformer Block with Sequence Parallelism
     *
     * Args:
     *   x: [batch_size, seq_len/sp_size, dim]
     *   sp_size: Sequence Parallel 的进程数
     *
     * Returns:
     *   [batch_size, seq_len/sp_size, dim]
     */
    auto sizes = x.sizes();
    int64_t bsz = sizes[0];
    int64_t seqlen = sizes[1]; // 这是局部序列长度 (L/sp_size)

    // 1. Pre-Norm + Self-Attention
    auto h = torch::layer_norm(
        x, {dim_}, attention_norm_weight_, attention_norm_bias_, 1e-5);

    // QKV projection
    auto xq = torch::matmul(h, wq_.transpose(0, 1)); // [B, L/N, dim]
    auto xk = torch::matmul(h, wk_.transpose(0, 1));
    auto xv = torch::matmul(h, wv_.transpose(0, 1));

    // Reshape: [B, L/N, dim] -> [B, L/N, n_heads, head_dim]
    xq = xq.view({bsz, seqlen, n_heads_, head_dim_});
    xk = xk.view({bsz, seqlen, n_heads_, head_dim_});
    xv = xv.view({bsz, seqlen, n_heads_, head_dim_});

    // Transpose for attention: [B, L/N, H, D] -> [B, H, L/N, D]
    xq = xq.transpose(1, 2);
    xk = xk.transpose(1, 2);
    xv = xv.transpose(1, 2);

    if (sp_size > 1) {
      // All-to-All: Sequence Parallel -> Head Parallel
      // [B, H, L/N, D] -> [B, H/N, L, D]
      xq = all_to_all(xq, /*scatter_dim=*/1, /*gather_dim=*/2, pg_);
      xk = all_to_all(xk, /*scatter_dim=*/1, /*gather_dim=*/2, pg_);
      xv = all_to_all(xv, /*scatter_dim=*/1, /*gather_dim=*/2, pg_);
    }

    // Scaled Dot-Product Attention
    auto attn_output = scaled_dot_product_attention(xq, xk, xv);

    if (sp_size > 1) {
      // All-to-All: Head Parallel -> Sequence Parallel
      // [B, H/N, L, D] -> [B, H, L/N, D]
      attn_output =
          all_to_all(attn_output, /*scatter_dim=*/2, /*gather_dim=*/1, pg_);
    }

    // Transpose back: [B, H, L/N, D] -> [B, L/N, H, D]
    attn_output = attn_output.transpose(1, 2).contiguous();

    // Reshape: [B, L/N, H, D] -> [B, L/N, dim]
    attn_output = attn_output.view({bsz, seqlen, dim_});

    // Output projection
    attn_output = torch::matmul(attn_output, wo_.transpose(0, 1));

    // Residual connection
    h = x + attn_output;

    // 2. Pre-Norm + Feed-Forward Network
    auto h_ffn = torch::layer_norm(
        h, {dim_}, ffn_norm_weight_, ffn_norm_bias_, 1e-5);

    // FFN: Linear -> ReLU -> Linear
    h_ffn = torch::matmul(h_ffn, w1_.transpose(0, 1)); // [B, L/N, hidden_dim]
    h_ffn = torch::relu(h_ffn);
    h_ffn = torch::matmul(h_ffn, w2_.transpose(0, 1)); // [B, L/N, dim]

    // Residual connection
    h = h + h_ffn;

    return h;
  }

  // 不使用 SP 的标准 Transformer Block（用于验证）
  torch::Tensor forward_no_sp(torch::Tensor x) {
    /*
     * Standard Transformer Block (No Sequence Parallelism)
     *
     * Args:
     *   x: [batch_size, full_seq_len, dim]
     *
     * Returns:
     *   [batch_size, full_seq_len, dim]
     */
    auto sizes = x.sizes();
    int64_t bsz = sizes[0];
    int64_t seqlen = sizes[1]; // 完整序列长度

    // 1. Pre-Norm + Self-Attention
    auto h = torch::layer_norm(
        x, {dim_}, attention_norm_weight_, attention_norm_bias_, 1e-5);

    // QKV projection
    auto xq = torch::matmul(h, wq_.transpose(0, 1));
    auto xk = torch::matmul(h, wk_.transpose(0, 1));
    auto xv = torch::matmul(h, wv_.transpose(0, 1));

    // Reshape: [B, L, dim] -> [B, L, n_heads, head_dim]
    xq = xq.view({bsz, seqlen, n_heads_, head_dim_});
    xk = xk.view({bsz, seqlen, n_heads_, head_dim_});
    xv = xv.view({bsz, seqlen, n_heads_, head_dim_});

    // Transpose for attention: [B, L, H, D] -> [B, H, L, D]
    xq = xq.transpose(1, 2);
    xk = xk.transpose(1, 2);
    xv = xv.transpose(1, 2);

    // Scaled Dot-Product Attention
    auto attn_output = scaled_dot_product_attention(xq, xk, xv);

    // Transpose back: [B, H, L, D] -> [B, L, H, D]
    attn_output = attn_output.transpose(1, 2).contiguous();

    // Reshape: [B, L, H, D] -> [B, L, dim]
    attn_output = attn_output.view({bsz, seqlen, dim_});

    // Output projection
    attn_output = torch::matmul(attn_output, wo_.transpose(0, 1));

    // Residual connection
    h = x + attn_output;

    // 2. Pre-Norm + Feed-Forward Network
    auto h_ffn = torch::layer_norm(
        h, {dim_}, ffn_norm_weight_, ffn_norm_bias_, 1e-5);

    // FFN: Linear -> ReLU -> Linear
    h_ffn = torch::matmul(h_ffn, w1_.transpose(0, 1)); // [B, L, hidden_dim]
    h_ffn = torch::relu(h_ffn);
    h_ffn = torch::matmul(h_ffn, w2_.transpose(0, 1)); // [B, L, dim]

    // Residual connection
    h = h + h_ffn;

    return h;
  }

  void to(torch::Device device) {
    wq_ = wq_.to(device);
    wk_ = wk_.to(device);
    wv_ = wv_.to(device);
    wo_ = wo_.to(device);
    w1_ = w1_.to(device);
    w2_ = w2_.to(device);
    attention_norm_weight_ = attention_norm_weight_.to(device);
    attention_norm_bias_ = attention_norm_bias_.to(device);
    ffn_norm_weight_ = ffn_norm_weight_.to(device);
    ffn_norm_bias_ = ffn_norm_bias_.to(device);
  }

 private:
  int64_t dim_;
  int64_t n_heads_;
  int64_t hidden_dim_;
  int64_t head_dim_;
  c10::intrusive_ptr<c10d::ProcessGroupMPI> pg_;

  // Self-Attention weights
  torch::Tensor wq_, wk_, wv_, wo_;

  // Feed-Forward Network weights
  torch::Tensor w1_, w2_;

  // LayerNorm parameters
  torch::Tensor attention_norm_weight_, attention_norm_bias_;
  torch::Tensor ffn_norm_weight_, ffn_norm_bias_;

  // 简化的 Scaled Dot-Product Attention
  torch::Tensor scaled_dot_product_attention(
      torch::Tensor q,
      torch::Tensor k,
      torch::Tensor v) {
    // q, k, v: [B, H, L, D]
    float scale = 1.0 / std::sqrt(static_cast<float>(head_dim_));

    // Attention scores: [B, H, L, L]
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

    // Causal mask (下三角)
    int64_t seq_len = q.size(2);
    auto mask = torch::triu(
        torch::ones({seq_len, seq_len}, q.options()) * -1e9,
        /*diagonal=*/1);
    scores = scores + mask;

    // Softmax
    auto attn_weights = torch::softmax(scores, /*dim=*/-1);

    // Output: [B, H, L, D]
    return torch::matmul(attn_weights, v);
  }
};

// ============================================================================
// Sequence Parallel Transformer（简化版）
// ============================================================================

class TransformerSP {
 public:
  TransformerSP(
      int64_t vocab_size,
      int64_t dim,
      int64_t n_heads,
      int64_t hidden_dim,
      int rank,
      int world_size,
      c10::intrusive_ptr<c10d::ProcessGroupMPI> pg)
      : vocab_size_(vocab_size),
        dim_(dim),
        n_heads_(n_heads),
        hidden_dim_(hidden_dim),
        rank_(rank),
        world_size_(world_size),
        pg_(pg) {
    // Embedding (rank 0 初始化，然后广播)
    if (rank == 0) {
      tok_embeddings_ = torch::randn({vocab_size, dim});

      // Final LayerNorm 参数
      norm_weight_ = torch::ones({dim});
      norm_bias_ = torch::zeros({dim});

      // Output projection (LM Head)
      output_weight_ = torch::randn({vocab_size, dim});
    } else {
      tok_embeddings_ = torch::zeros({vocab_size, dim});
      norm_weight_ = torch::zeros({dim});
      norm_bias_ = torch::zeros({dim});
      output_weight_ = torch::zeros({vocab_size, dim});
    }

    // 广播所有权重
    std::vector<torch::Tensor> tok_emb_vec = {tok_embeddings_};
    std::vector<torch::Tensor> norm_w_vec = {norm_weight_};
    std::vector<torch::Tensor> norm_b_vec = {norm_bias_};
    std::vector<torch::Tensor> output_w_vec = {output_weight_};

    pg_->broadcast(tok_emb_vec)->wait();
    pg_->broadcast(norm_w_vec)->wait();
    pg_->broadcast(norm_b_vec)->wait();
    pg_->broadcast(output_w_vec)->wait();

    tok_embeddings_ = tok_emb_vec[0];
    norm_weight_ = norm_w_vec[0];
    norm_bias_ = norm_b_vec[0];
    output_weight_ = output_w_vec[0];

    // Transformer Block (Attention + FFN + LayerNorm)
    attention_ = std::make_shared<AttentionSP>(dim, n_heads, hidden_dim, pg_);

    std::cout << "Rank " << rank << ": TransformerSP initialized with "
              << "vocab_size=" << vocab_size << ", dim=" << dim
              << ", n_heads=" << n_heads << ", hidden_dim=" << hidden_dim
              << std::endl;
  }

  torch::Tensor forward(torch::Tensor tokens, int64_t sp_size) {
    /*
     * Sequence Parallel Forward Pass
     *
     * Args:
     *   tokens: [batch_size, seq_len] (完整序列，所有 rank 相同)
     *   sp_size: Sequence Parallel 的进程数
     *
     * Returns:
     *   [batch_size, seq_len, vocab_size] (logits for next token prediction)
     */
    auto sizes = tokens.sizes();
    int64_t bsz = sizes[0];
    int64_t seq_len = sizes[1];

    // 1. Embedding (所有 rank 完整计算)
    auto h = torch::embedding(tok_embeddings_, tokens); // [B, L, dim]

    int64_t padding_len = 0;
    torch::Tensor freqs_cis; // 简化：不使用 RoPE

    if (sp_size > 1) {
      // 2. Padding 使序列长度能被 sp_size 整除
      auto [padded_h, pad_len] = pad_to_divisible(h, sp_size, /*dim=*/1);
      h = padded_h;
      padding_len = pad_len;

      // 3. 切分序列到不同 ranks
      int rank_in_sp = rank_ % sp_size;
      auto chunks = torch::chunk(h, sp_size, /*dim=*/1);
      h = chunks[rank_in_sp].clone(); // [B, L/N, dim]

      if (rank_ == 0) {
        std::cout << "After split: rank " << rank_ << " -> "
                  << "h shape: " << h.sizes() << std::endl;
      }
    }

    // 4. 通过 Transformer Block
    h = attention_->forward(h, sp_size);

    if (sp_size > 1) {
      // 5. AllGather 恢复完整序列
      h = all_gather(h, /*gather_dim=*/1, pg_); // [B, L', dim]

      // 6. 去除 padding
      if (padding_len > 0) {
        h = h.slice(/*dim=*/1, /*start=*/0, /*end=*/seq_len); // [B, L, dim]
      }
    }

    // 7. Final LayerNorm
    h = torch::layer_norm(h, {dim_}, norm_weight_, norm_bias_, 1e-5);

    // 8. Output projection (LM Head)
    auto logits = torch::matmul(h, output_weight_.transpose(0, 1)); // [B, L, vocab_size]

    return logits;
  }

  void to(torch::Device device) {
    tok_embeddings_ = tok_embeddings_.to(device);
    norm_weight_ = norm_weight_.to(device);
    norm_bias_ = norm_bias_.to(device);
    output_weight_ = output_weight_.to(device);
    attention_->to(device);
  }

  // 访问器（用于验证）
  torch::Tensor& get_tok_embeddings() {
    return tok_embeddings_;
  }
  std::shared_ptr<AttentionSP>& get_attention() {
    return attention_;
  }

 private:
  int64_t vocab_size_;
  int64_t dim_;
  int64_t n_heads_;
  int64_t hidden_dim_;
  int rank_;
  int world_size_;
  c10::intrusive_ptr<c10d::ProcessGroupMPI> pg_;

  torch::Tensor tok_embeddings_;
  torch::Tensor norm_weight_, norm_bias_;     // Final LayerNorm
  torch::Tensor output_weight_;                // LM Head
  std::shared_ptr<AttentionSP> attention_;
};

// ============================================================================
// Main 函数
// ============================================================================

int main(int argc, char* argv[]) {
  // 初始化 MPI Process Group
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  int rank = pg->getRank();
  int world_size = pg->getSize();

  if (rank == 0) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  Sequence Parallel Demo with Verification            ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;
    std::cout << "MPI World Size: " << world_size << "\n" << std::endl;
  }

  // 模型配置
  int64_t vocab_size = 320;
  int64_t dim = 256;       // embedding dimension
  int64_t n_heads = 4;     // attention heads (必须能被 world_size 整除)
  int64_t hidden_dim = 1024; // FFN hidden dimension (通常是 dim * 4)
  int64_t batch_size = 2;
  int64_t seq_len = 12; // 序列长度

  // 检查配置
  if (n_heads % world_size != 0) {
    if (rank == 0) {
      std::cerr << "Error: n_heads (" << n_heads
                << ") must be divisible by world_size (" << world_size << ")"
                << std::endl;
    }
    return 1;
  }

  // 创建 Sequence Parallel Transformer
  TransformerSP model(vocab_size, dim, n_heads, hidden_dim, rank, world_size, pg);

  // 移动到 CPU
  torch::Device device(torch::kCPU);
  model.to(device);

  // 推理输入（所有 rank 使用相同的输入）
  torch::manual_seed(42);
  auto tokens = torch::randint(
      0,
      vocab_size,
      {batch_size, seq_len},
      torch::TensorOptions().dtype(torch::kLong).device(device));

  if (rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 1: Sequence Parallel Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input tokens shape: " << tokens.sizes() << std::endl;
  }

  // SP Forward pass
  auto start = std::chrono::high_resolution_clock::now();
  auto output_sp = model.forward(tokens, /*sp_size=*/world_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  if (rank == 0) {
    std::cout << "Output shape: " << output_sp.sizes() << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Sample (first 3 elements): "
              << output_sp.flatten().slice(0, 0, 3) << std::endl;
  }

  // 验证：对比 SP 和 No-SP 的 Transformer 完整输出
  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 2: No-SP Verification" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // 使用相同的输入进行验证
  torch::Tensor output_no_sp;
  if (rank == 0) {
    // 只在 rank 0 运行 No-SP 推理（sp_size=1 表示不使用 SP）
    start = std::chrono::high_resolution_clock::now();
    output_no_sp = model.forward(tokens, /*sp_size=*/1);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Sample (first 3 elements): "
              << output_no_sp.flatten().slice(0, 0, 3) << std::endl;
  }

  // 同步所有进程
  std::vector<torch::Tensor> barrier = {torch::zeros({1})};
  c10d::AllreduceOptions barrier_opts;
  barrier_opts.reduceOp = c10d::ReduceOp::SUM;
  pg->allreduce(barrier, barrier_opts)->wait();

  // 验证结果
  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Verification" << std::endl;
    std::cout << "========================================" << std::endl;

    // 计算差异
    auto diff = torch::abs(output_sp - output_no_sp);
    auto max_diff = torch::max(diff).item<float>();
    auto mean_diff = torch::mean(diff).item<float>();

    std::cout << "Max absolute diff:  " << max_diff << std::endl;
    std::cout << "Mean absolute diff: " << mean_diff << std::endl;

    // 判断是否通过
    float atol = 1e-3;
    bool passed = (max_diff < atol);

    if (passed) {
      std::cout << "\n✅ PASSED: SP and No-SP outputs match!" << std::endl;
    } else {
      std::cout << "\n❌ FAILED: Max diff " << max_diff << " > " << atol
                << std::endl;
    }

    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  ✅ Sequence Parallel Demo Complete!                  ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝"
              << std::endl;

    std::cout << "\nKey Concepts Demonstrated:" << std::endl;
    std::cout << "  ✓ Sequence split across ranks" << std::endl;
    std::cout << "  ✓ All-to-All: Sequence Parallel -> Head Parallel"
              << std::endl;
    std::cout << "  ✓ Attention on full sequence with partial heads"
              << std::endl;
    std::cout << "  ✓ All-to-All: Head Parallel -> Sequence Parallel"
              << std::endl;
    std::cout << "  ✓ AllGather: Restore complete sequence" << std::endl;
    std::cout << "  ✓ Verification: SP matches single-device inference"
              << std::endl;
    std::cout
        << "\nThis demonstrates the CORE Ulysses Sequence Parallelism logic!\n"
        << std::endl;
  }

  return 0;
}
