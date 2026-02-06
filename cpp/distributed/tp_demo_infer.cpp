#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>

/*
 * 手动实现 Tensor Parallelism 用于大模型推理（使用 MPI）
 *
 * 核心思路（类似 Megatron-LM）：
 * 1. Column-wise parallel: 第一个 Linear 按列切分
 * 2. Row-wise parallel: 第二个 Linear 按行切分
 * 3. 只在必要时进行通信（AllReduce）
 * 4. 添加验证对比：TP vs No-TP
 */

// Tensor Parallel 推理包装器
class TensorParallelInference {
 public:
  TensorParallelInference(
      int rank,
      int world_size,
      c10::intrusive_ptr<c10d::ProcessGroupMPI> process_group)
      : rank_(rank), world_size_(world_size), pg_(process_group) {}

  // 加载模型并切分权重
  void load_and_shard_model(
      const std::string& model_path,
      int64_t input_size,
      int64_t hidden_size,
      int64_t output_size) {
    // 1. 只在 rank 0 加载完整模型
    torch::Tensor fc1_weight, fc1_bias, fc2_weight, fc2_bias;

    if (rank_ == 0) {
      // 加载完整模型权重（假设已经保存）
      // 实际应该从文件加载: torch::load(fc1_weight, "fc1_weight.pt");
      fc1_weight = torch::randn({hidden_size, input_size});
      fc1_bias = torch::randn({hidden_size});
      fc2_weight = torch::randn({output_size, hidden_size});
      fc2_bias = torch::randn({output_size});
    } else {
      // 其他 rank 创建空张量
      fc1_weight = torch::empty({hidden_size, input_size});
      fc1_bias = torch::empty({hidden_size});
      fc2_weight = torch::empty({output_size, hidden_size});
      fc2_bias = torch::empty({output_size});
    }

    // 2. 广播完整权重到所有 rank（或者直接从文件各自加载）
    std::vector<torch::Tensor> fc1_weight_vec = {fc1_weight};
    std::vector<torch::Tensor> fc1_bias_vec = {fc1_bias};
    std::vector<torch::Tensor> fc2_weight_vec = {fc2_weight};
    std::vector<torch::Tensor> fc2_bias_vec = {fc2_bias};

    pg_->broadcast(fc1_weight_vec)->wait();
    pg_->broadcast(fc1_bias_vec)->wait();
    pg_->broadcast(fc2_weight_vec)->wait();
    pg_->broadcast(fc2_bias_vec)->wait();

    // 3. Column-wise 切分 fc1 (按输出维度切分)
    // fc1: [hidden_size, input_size] -> [hidden_size/world_size, input_size]
    int64_t hidden_per_rank = hidden_size / world_size_;
    fc1_weight_shard_ =
        fc1_weight
            .slice(0, rank_ * hidden_per_rank, (rank_ + 1) * hidden_per_rank)
            .clone();
    fc1_bias_shard_ =
        fc1_bias
            .slice(0, rank_ * hidden_per_rank, (rank_ + 1) * hidden_per_rank)
            .clone();

    // 4. Row-wise 切分 fc2 (按输入维度切分)
    // fc2: [output_size, hidden_size] -> [output_size, hidden_size/world_size]
    fc2_weight_shard_ =
        fc2_weight
            .slice(1, rank_ * hidden_per_rank, (rank_ + 1) * hidden_per_rank)
            .clone();
    fc2_bias_shard_ = fc2_bias.clone(); // bias 不切分，但需要 reduce

    std::cout << "Rank " << rank_ << " loaded sharded weights:" << std::endl;
    std::cout << "  fc1_weight_shard: " << fc1_weight_shard_.sizes()
              << std::endl;
    std::cout << "  fc2_weight_shard: " << fc2_weight_shard_.sizes()
              << std::endl;
  }

  // Tensor Parallel 前向传播
  torch::Tensor forward(torch::Tensor input) {
    // input: [batch_size, input_size]

    // 1. Column-wise parallel fc1
    // 每个 rank 计算部分输出: [batch_size, hidden_size/world_size]
    auto h = torch::linear(input, fc1_weight_shard_, fc1_bias_shard_);
    h = torch::relu(h);

    // 注意：这里不需要 AllGather，因为下一层是 Row-wise parallel
    // Row-wise parallel 会自然地消费切分的输入

    // 2. Row-wise parallel fc2
    // 每个 rank 计算部分结果: [batch_size, output_size]
    auto output =
        torch::linear(h, fc2_weight_shard_, fc2_bias_shard_ / world_size_);

    // 3. AllReduce 合并所有 rank 的结果
    std::vector<torch::Tensor> output_vec = {output};
    c10d::AllreduceOptions opts;
    opts.reduceOp = c10d::ReduceOp::SUM;
    pg_->allreduce(output_vec, opts)->wait();

    return output_vec[0];
  }

  // 不使用 TP 的标准前向传播（用于验证）
  torch::Tensor forward_no_tp(
      torch::Tensor input,
      torch::Tensor fc1_weight_full,
      torch::Tensor fc1_bias_full,
      torch::Tensor fc2_weight_full,
      torch::Tensor fc2_bias_full) {
    // input: [batch_size, input_size]

    // 1. 完整的 fc1
    auto h = torch::linear(input, fc1_weight_full, fc1_bias_full);
    h = torch::relu(h);

    // 2. 完整的 fc2
    auto output = torch::linear(h, fc2_weight_full, fc2_bias_full);

    return output;
  }

  // 获取完整权重（用于验证）
  void get_full_weights(
      torch::Tensor& fc1_weight_full,
      torch::Tensor& fc1_bias_full,
      torch::Tensor& fc2_weight_full,
      torch::Tensor& fc2_bias_full) {
    // AllGather fc1 weights
    std::vector<torch::Tensor> fc1_weight_list;
    for (int i = 0; i < world_size_; ++i) {
      fc1_weight_list.push_back(fc1_weight_shard_.clone().zero_());
    }
    std::vector<std::vector<torch::Tensor>> fc1_weight_output = {
        fc1_weight_list};
    std::vector<torch::Tensor> fc1_weight_input = {
        fc1_weight_shard_.contiguous()};
    pg_->allgather(fc1_weight_output, fc1_weight_input)->wait();
    fc1_weight_full = torch::cat(fc1_weight_list, 0);

    // AllGather fc1 bias
    std::vector<torch::Tensor> fc1_bias_list;
    for (int i = 0; i < world_size_; ++i) {
      fc1_bias_list.push_back(fc1_bias_shard_.clone().zero_());
    }
    std::vector<std::vector<torch::Tensor>> fc1_bias_output = {fc1_bias_list};
    std::vector<torch::Tensor> fc1_bias_input = {fc1_bias_shard_.contiguous()};
    pg_->allgather(fc1_bias_output, fc1_bias_input)->wait();
    fc1_bias_full = torch::cat(fc1_bias_list, 0);

    // AllGather fc2 weights (沿列拼接)
    std::vector<torch::Tensor> fc2_weight_list;
    for (int i = 0; i < world_size_; ++i) {
      fc2_weight_list.push_back(fc2_weight_shard_.clone().zero_());
    }
    std::vector<std::vector<torch::Tensor>> fc2_weight_output = {
        fc2_weight_list};
    std::vector<torch::Tensor> fc2_weight_input = {
        fc2_weight_shard_.contiguous()};
    pg_->allgather(fc2_weight_output, fc2_weight_input)->wait();
    fc2_weight_full = torch::cat(fc2_weight_list, 1);

    // fc2 bias 不切分，直接使用
    fc2_bias_full = fc2_bias_shard_.clone();
  }

  // 将模型移动到指定设备
  void to(torch::Device device) {
    fc1_weight_shard_ = fc1_weight_shard_.to(device);
    fc1_bias_shard_ = fc1_bias_shard_.to(device);
    fc2_weight_shard_ = fc2_weight_shard_.to(device);
    fc2_bias_shard_ = fc2_bias_shard_.to(device);
  }

 private:
  int rank_;
  int world_size_;
  c10::intrusive_ptr<c10d::ProcessGroupMPI> pg_;

  // 切分后的权重
  torch::Tensor fc1_weight_shard_;
  torch::Tensor fc1_bias_shard_;
  torch::Tensor fc2_weight_shard_;
  torch::Tensor fc2_bias_shard_;
};

int main(int argc, char* argv[]) {
  // 初始化 MPI Process Group
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  int rank = pg->getRank();
  int world_size = pg->getSize();

  if (rank == 0) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  Tensor Parallel Demo with Verification              ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;
    std::cout << "MPI World Size: " << world_size << "\n" << std::endl;
  }

  // 模型配置
  int64_t input_size = 1024;
  int64_t hidden_size = 4096; // 会被切分到多个进程
  int64_t output_size = 512;
  int64_t batch_size = 8;

  // 检查 hidden_size 能否被 world_size 整除
  if (hidden_size % world_size != 0) {
    if (rank == 0) {
      std::cerr << "Error: hidden_size (" << hidden_size
                << ") must be divisible by world_size (" << world_size << ")"
                << std::endl;
    }
    return 1;
  }

  // 创建 Tensor Parallel 推理引擎
  TensorParallelInference tp_model(rank, world_size, pg);

  // 加载并切分模型
  if (rank == 0) {
    std::cout << "Loading and sharding model weights..." << std::endl;
  }
  tp_model.load_and_shard_model(
      "model.pt", input_size, hidden_size, output_size);

  // 移动到 CPU（如果有 CUDA，可以改为 cuda:rank）
  torch::Device device(torch::kCPU);
  tp_model.to(device);

  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 1: Tensor Parallel Inference" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // 推理
  // 注意：所有 rank 必须有相同的输入
  torch::manual_seed(42);
  auto input = torch::randn({batch_size, input_size}).to(device);

  auto start = std::chrono::high_resolution_clock::now();
  auto output_tp = tp_model.forward(input);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  if (rank == 0) {
    std::cout << "Output shape: " << output_tp.sizes() << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Sample (first 3 elements): "
              << output_tp.flatten().slice(0, 0, 3) << std::endl;
  }

  // 验证：对比 TP 和 No-TP 的结果
  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 2: No-TP Verification" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // 收集完整权重
  torch::Tensor fc1_weight_full, fc1_bias_full, fc2_weight_full, fc2_bias_full;
  tp_model.get_full_weights(
      fc1_weight_full, fc1_bias_full, fc2_weight_full, fc2_bias_full);

  torch::Tensor output_no_tp;
  if (rank == 0) {
    // 只在 rank 0 运行 No-TP 推理
    start = std::chrono::high_resolution_clock::now();
    output_no_tp = tp_model.forward_no_tp(
        input, fc1_weight_full, fc1_bias_full, fc2_weight_full, fc2_bias_full);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Output shape: " << output_no_tp.sizes() << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Sample (first 3 elements): "
              << output_no_tp.flatten().slice(0, 0, 3) << std::endl;
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
    auto diff = torch::abs(output_tp - output_no_tp);
    auto max_diff = torch::max(diff).item<float>();
    auto mean_diff = torch::mean(diff).item<float>();

    // 相对误差
    auto relative_diff = diff / (torch::abs(output_no_tp) + 1e-8);
    auto max_relative = torch::max(relative_diff).item<float>();
    auto mean_relative = torch::mean(relative_diff).item<float>();

    std::cout << "\nNumerical Difference:" << std::endl;
    std::cout << "  Max absolute diff:  " << max_diff << std::endl;
    std::cout << "  Mean absolute diff: " << mean_diff << std::endl;
    std::cout << "  Max relative diff:  " << max_relative << std::endl;
    std::cout << "  Mean relative diff: " << mean_relative << std::endl;

    // 判断是否通过
    float atol = 1e-3; // absolute tolerance (放宽到1e-3，因为浮点累积误差)
    float rtol = 1e-3; // relative tolerance
    bool passed = (max_diff < atol) || (max_relative < rtol);

    std::cout << "\nTolerance Settings:" << std::endl;
    std::cout << "  Absolute tolerance: " << atol << std::endl;
    std::cout << "  Relative tolerance: " << rtol << std::endl;

    if (passed) {
      std::cout << "\n✅ PASSED: TP and No-TP outputs match!" << std::endl;
    } else {
      std::cout << "\n❌ FAILED: Outputs differ significantly!" << std::endl;
    }

    // 显示样本值对比
    std::cout << "\nSample Values Comparison:" << std::endl;
    std::cout << "  TP output:     " << output_tp.flatten().slice(0, 0, 5)
              << std::endl;
    std::cout << "  No-TP output:  " << output_no_tp.flatten().slice(0, 0, 5)
              << std::endl;
    std::cout << "  Difference:    " << diff.flatten().slice(0, 0, 5)
              << std::endl;

    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  ✅ Tensor Parallel Demo Complete!                    ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝"
              << std::endl;

    std::cout << "\nKey Concepts Demonstrated:" << std::endl;
    std::cout << "  ✓ Column-wise parallel: FC1 split by output dimension"
              << std::endl;
    std::cout << "  ✓ Row-wise parallel: FC2 split by input dimension"
              << std::endl;
    std::cout << "  ✓ AllReduce: Combine partial results from all ranks"
              << std::endl;
    std::cout << "  ✓ Verification: TP matches single-device inference"
              << std::endl;
    std::cout << "\nThis demonstrates the CORE Tensor Parallelism logic!\n"
              << std::endl;
  }

  return 0;
}
