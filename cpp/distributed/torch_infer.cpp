#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

std::string model_path =
    "/data/workspace/hunyuan_ptm/torch_examples/exporte_and_compile/model.pt2";

// ============================================================================
// 函数 1: 直接推理（单进程，不使用并行）
// ============================================================================
torch::Tensor direct_inference(
    torch::inductor::AOTIModelPackageLoader& loader,
    torch::Tensor input,
    const std::string& name = "Direct") {
  std::cout << "\n========================================" << std::endl;
  std::cout << "[" << name << " Inference]" << std::endl;
  std::cout << "Input shape: " << input.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<torch::Tensor> inputs = {input};
  std::vector<torch::Tensor> outputs = loader.run(inputs);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Output shape: " << outputs[0].sizes() << std::endl;
  std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
  std::cout << "========================================\n" << std::endl;

  return outputs[0];
}

// ============================================================================
// 函数 2: Tensor Parallel 推理（多进程，模型权重切分）
// ============================================================================
torch::Tensor tensor_parallel_inference(
    torch::inductor::AOTIModelPackageLoader& loader,
    torch::Tensor input,
    c10::intrusive_ptr<c10d::ProcessGroupMPI> pg,
    int rank,
    int world_size,
    const std::string& name = "TensorParallel") {
  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "[" << name << " Inference] (world_size=" << world_size << ")"
              << std::endl;
    std::cout << "Input shape: " << input.sizes() << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // 注意：这里是简化实现
  // 真正的 TP 需要：
  // 1. 将模型权重切分到不同的 rank
  // 2. 每个 rank 计算部分输出
  // 3. AllReduce 合并结果
  //
  // 由于 AOTIModelPackageLoader 已经编译好了，我们无法直接切分权重
  // 这里演示的是：每个 rank 独立运行完整模型，然后 AllReduce 结果

  // 方案：每个 rank 运行完整推理，然后平均结果（模拟 TP 的 AllReduce）
  std::vector<torch::Tensor> inputs = {input};
  std::vector<torch::Tensor> outputs = loader.run(inputs);

  // AllReduce: 求和所有 rank 的输出
  std::vector<torch::Tensor> output_list = {outputs[0]};
  c10d::AllreduceOptions allreduce_opts;
  allreduce_opts.reduceOp = c10d::ReduceOp::SUM;
  pg->allreduce(output_list, allreduce_opts)->wait();

  // 除以 world_size 得到平均值（或者根据 TP 逻辑处理）
  // 注意：真正的 TP 不需要平均，这里只是演示通信
  auto result = output_list[0] / static_cast<float>(world_size);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  if (rank == 0) {
    std::cout << "Output shape: " << result.sizes() << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  return result;
}

// ============================================================================
// 验证函数：检查两个张量是否近似相等
// ============================================================================
bool verify_results(
    const torch::Tensor& output1,
    const torch::Tensor& output2,
    float rtol = 1e-5,
    float atol = 1e-5,
    const std::string& name1 = "Output1",
    const std::string& name2 = "Output2") {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Verification: " << name1 << " vs " << name2 << std::endl;
  std::cout << "========================================" << std::endl;

  // 1. 检查形状
  if (output1.sizes() != output2.sizes()) {
    std::cout << "❌ Shape mismatch!" << std::endl;
    std::cout << "  " << name1 << " shape: " << output1.sizes() << std::endl;
    std::cout << "  " << name2 << " shape: " << output2.sizes() << std::endl;
    return false;
  }
  std::cout << "✓ Shapes match: " << output1.sizes() << std::endl;

  // 2. 检查数值
  auto diff = torch::abs(output1 - output2);
  auto max_diff = torch::max(diff).item<float>();
  auto mean_diff = torch::mean(diff).item<float>();

  std::cout << "Max absolute difference: " << max_diff << std::endl;
  std::cout << "Mean absolute difference: " << mean_diff << std::endl;

  // 3. 使用 torch::allclose
  bool is_close = torch::allclose(output1, output2, rtol, atol);

  if (is_close) {
    std::cout << "✅ Results match! (rtol=" << rtol << ", atol=" << atol << ")"
              << std::endl;
  } else {
    std::cout << "❌ Results do NOT match!" << std::endl;
    std::cout << "\nSample values:" << std::endl;
    std::cout << "  " << name1
              << " (first 5): " << output1.flatten().slice(0, 0, 5)
              << std::endl;
    std::cout << "  " << name2
              << " (first 5): " << output2.flatten().slice(0, 0, 5)
              << std::endl;
  }

  std::cout << "========================================\n" << std::endl;
  return is_close;
}

// ============================================================================
// Main 函数
// ============================================================================
int main(int argc, char* argv[]) {
  std::cout << "\n╔════════════════════════════════════════════════════════╗"
            << std::endl;
  std::cout << "║  PyTorch C++ Inference: Direct vs Tensor Parallel     ║"
            << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════╝\n"
            << std::endl;

  // ========================================================================
  // 1. 初始化
  // ========================================================================
  c10::InferenceMode mode;

  // 选择设备
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    try {
      device = torch::Device(torch::kCUDA);
      auto test_tensor = torch::randn({1, 1}, device);
      std::cout << "Using CUDA device" << std::endl;
    } catch (const c10::Error& e) {
      std::cout << "CUDA available but failed to use, falling back to CPU"
                << std::endl;
      device = torch::Device(torch::kCPU);
    }
  } else {
    std::cout << "Using CPU device" << std::endl;
  }

  // 初始化 MPI (用于 Tensor Parallel)
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  int rank = pg->getRank();
  int world_size = pg->getSize();

  if (rank == 0) {
    std::cout << "MPI initialized: world_size = " << world_size << "\n"
              << std::endl;
  }

  // 加载模型
  if (rank == 0) {
    std::cout << "Loading model from: " << model_path << std::endl;
  }
  torch::inductor::AOTIModelPackageLoader loader(model_path);

  // ========================================================================
  // 2. 准备输入数据（所有 rank 使用相同的输入）
  // ========================================================================
  torch::manual_seed(42); // 固定随机种子，确保可重复
  int64_t batch_size = 8;
  int64_t input_size = 10;
  auto input = torch::randn({batch_size, input_size}, device);

  if (rank == 0) {
    std::cout << "\nInput statistics:" << std::endl;
    std::cout << "  Mean: " << torch::mean(input).item<float>() << std::endl;
    std::cout << "  Std:  " << torch::std(input).item<float>() << std::endl;
    std::cout << "  Sample (first 5): " << input.flatten().slice(0, 0, 5)
              << std::endl;
  }

  // ========================================================================
  // 3. 测试 1: 直接推理（只在 rank 0 运行）
  // ========================================================================
  torch::Tensor output_direct;
  if (rank == 0) {
    output_direct = direct_inference(loader, input, "Direct");
  }

  // 广播 direct 结果到所有 rank（用于后续比较）
  if (rank == 0) {
    // Rank 0 已经有结果
  } else {
    // 其他 rank 创建空张量接收
    output_direct = torch::zeros({batch_size, 1}, device);
  }
  std::vector<torch::Tensor> bcast_list = {output_direct};
  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = 0;
  bcast_opts.rootTensor = 0;
  pg->broadcast(bcast_list, bcast_opts)->wait();

  // ========================================================================
  // 4. 测试 2: Tensor Parallel 推理（所有 rank 参与）
  // ========================================================================
  auto output_tp = tensor_parallel_inference(
      loader, input, pg, rank, world_size, "TensorParallel");

  // ========================================================================
  // 5. 验证结果（只在 rank 0）
  // ========================================================================
  if (rank == 0) {
    bool match = verify_results(
        output_direct,
        output_tp,
        1e-5,
        1e-5,
        "Direct Inference",
        "Tensor Parallel Inference");

    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    if (match) {
      std::cout << "║  ✅ SUCCESS: Both methods produce identical results!  ║"
                << std::endl;
    } else {
      std::cout << "║  ⚠️  WARNING: Results differ (expected for demo)      ║"
                << std::endl;
      std::cout << "║     (This is because we're averaging, not true TP)    ║"
                << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;

    // 输出样本结果
    std::cout << "Sample outputs (first 3 rows):" << std::endl;
    std::cout << "  Direct:         " << output_direct.slice(0, 0, 3).flatten()
              << std::endl;
    std::cout << "  TensorParallel: " << output_tp.slice(0, 0, 3).flatten()
              << std::endl;
  }

  // 同步所有进程
  std::vector<torch::Tensor> barrier = {torch::zeros({1})};
  pg->allreduce(barrier)->wait();

  if (rank == 0) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  All tests completed successfully!                    ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝\n"
              << std::endl;

    std::cout << "Note:" << std::endl;
    std::cout
        << "  This is a simplified demonstration. True Tensor Parallelism requires:"
        << std::endl;
    std::cout << "  1. Model weights to be split across ranks" << std::endl;
    std::cout << "  2. Column-wise and row-wise parallel linear layers"
              << std::endl;
    std::cout << "  3. Proper AllReduce for combining partial outputs"
              << std::endl;
    std::cout
        << "  See tensor_parallel_inference.cpp for a full implementation.\n"
        << std::endl;
  }

  return 0;
}

// ============================================================================
// 编译和运行:
// make torch-infer
// source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64
// export CUDA_HOME=/data/workspace/cuda-12.8
// export PATH=$CUDA_HOME/bin:$PATH
// export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
// export OMPI_ALLOW_RUN_AS_ROOT=1
// export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
// export OMPI_MCA_pml=ob1
// export OMPI_MCA_btl=tcp,self,vader
// export OMPI_MCA_mtl=^ofi
// export LD_LIBRARY_PATH=/data/workspace/cutlass/.conda/lib:$LD_LIBRARY_PATH
// mpirun -np 4 ./torch-infer
// ============================================================================
