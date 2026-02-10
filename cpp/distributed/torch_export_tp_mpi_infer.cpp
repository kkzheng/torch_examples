/**
 * Tensor Parallel 推理示例（使用 MPI + ExportedProgram）
 *
 * 功能：
 * - 使用 c10d::ProcessGroupMPI 初始化分布式环境
 * - 每个 rank 加载自己的切分模型（transformer_tp_rank{rank}.pt2）
 * - 执行分布式推理，在 row parallel 层后手动执行 all-reduce
 *
 * 注意：
 * - 需要先运行 Python 脚本导出切分后的模型：
 *   torchrun --nproc_per_node=2 script/inspect_exported.py --tp
 * --no-allreduce-in-graph
 * - 这会生成：transformer_tp_rank0.pt2, transformer_tp_rank1.pt2, ...
 *
 * 编译：
 *   mkdir -p build && cd build
 *   cmake .. && make torch-export-tp-mpi-infer
 *
 * 运行：
 *   export CUDA_HOME=/data/workspace/cuda-12.8
 *   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 *   mpirun -np 2 ./torch-export-tp-mpi-infer
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/nativert/ModelRunnerHandle.h>
#include <torch/torch.h>

// 模型配置
const std::string MODEL_DIR = "../model";
const std::string MODEL_NAME = "model";
const int64_t SEQ_LEN = 32;
const int64_t VOCAB_SIZE = 1000;

/**
 * 检查文件是否存在
 */
bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

/**
 * 执行 TP 推理
 *
 * All-reduce 已经内置在 FX Graph 中，会自动执行。
 */
torch::Tensor run_tp_inference(
    torch::nativert::ModelRunnerHandle& runner,
    torch::Tensor input,
    int rank) {
  std::cout << "[Rank " << rank << "] Running inference..." << std::endl;
  std::cout << "[Rank " << rank << "] Input shape: " << input.sizes()
            << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // 准备输入
  std::vector<c10::IValue> inputs;
  inputs.push_back(input);

  // 执行推理（all-reduce 已在 FX Graph 中自动执行）
  auto output_vec = runner.runWithFlatInputsAndOutputs(std::move(inputs));
  auto output = output_vec[0].toTensor();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "[Rank " << rank << "] Output shape: " << output.sizes()
            << std::endl;
  std::cout << "[Rank " << rank << "] Inference time (with all-reduce): "
            << duration.count() / 1000.0 << " ms" << std::endl;

  return output;
}

int main(int argc, char* argv[]) {
  // 1. 创建 MPI Backend
  auto backend_mpi = c10d::ProcessGroupMPI::createProcessGroupMPI();
  int rank = backend_mpi->getRank();
  int world_size = backend_mpi->getSize();

  std::cout << "[Rank " << rank << "] ProcessGroupMPI Backend created, world_size=" << world_size << std::endl;

  // 2. 创建 ProcessGroup 包装器（参考 Python 的 distributed_c10d.py:2031-2036）
  // ProcessGroup 是一个包装器，它持有 Backend
  auto pg = c10::make_intrusive<c10d::ProcessGroup>(rank, world_size);

  // 3. 将 MPI Backend 注册到 ProcessGroup 中
  pg->setBackend(
      c10::DeviceType::CPU,  // device type
      c10d::ProcessGroup::BackendType::MPI,  // backend type
      backend_mpi  // backend instance
  );

  // 4. 设置默认 backend
  pg->setDefaultBackend(c10d::ProcessGroup::BackendType::MPI);

  // 5. 注册 ProcessGroup 到 GroupRegistry，使用名称 "0"
  c10d::register_process_group("0", pg);
  std::cout << "[Rank " << rank << "] ProcessGroup registered with name '0'" << std::endl;

  try {
    // 构建模型路径
    std::string model_path =
        MODEL_DIR + "/transformer_tp_rank" + std::to_string(rank) + ".pt2";

    std::cout << "[Rank " << rank << "] Loading model: " << model_path
              << std::endl;

    // 检查文件是否存在
    if (!file_exists(model_path)) {
      std::cerr << "[Rank " << rank
                << "] Error: Model file not found: " << model_path << std::endl;
      std::cerr << "Please run: torchrun --nproc_per_node=" << world_size
                << " script/inspect_exported.py --tp" << std::endl;
      return 1;
    }

    // 加载模型
    torch::nativert::ModelRunnerHandle runner(model_path, MODEL_NAME);
    std::cout << "[Rank " << rank << "] ✓ Model loaded successfully\n"
              << std::endl;

    // 使用固定的随机种子生成相同的输入（所有 rank 使用相同输入）
    torch::manual_seed(42);
    auto input = torch::randint(0, VOCAB_SIZE, {4, SEQ_LEN}, torch::kInt64);

    // 执行 TP 推理
    auto output = run_tp_inference(runner, input, rank);

    // 打印结果
    if (rank == 0) {
      std::cout << "\n[Rank " << rank << "] Final output:" << std::endl;
      std::cout << output << std::endl;
      std::cout << "\n========================================" << std::endl;
      std::cout << "All ranks completed successfully!" << std::endl;
      std::cout << "========================================" << std::endl;
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
    return 1;
  }
}
