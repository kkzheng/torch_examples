#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

std::string model_path =
    "/data/workspace/hunyuan_ptm/torch_examples/exporte_and_compile/model.pt2";

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

int main(int argc, char* argv[]) {
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

  torch::manual_seed(42); // 固定随机种子，确保可重复
  int64_t batch_size = 8;
  int64_t input_size = 10;
  auto input = torch::randn({batch_size, input_size}, device);

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

  // 同步所有进程
  std::vector<torch::Tensor> barrier = {torch::zeros({1})};
  pg->allreduce(barrier)->wait();

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
