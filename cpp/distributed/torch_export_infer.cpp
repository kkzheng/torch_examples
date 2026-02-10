/**
 * torch.export 模型加载和推理示例
 *
 * 功能：使用 torch::nativert::ModelRunnerHandle 加载 .pt2 模型
 *
 * 关键 API：
 * - torch::nativert::ModelRunnerHandle - 加载和运行 ExportedProgram
 * - 可以加载 torch.export 导出的 .pt2 文件
 * - 支持动态 batch size
 *
 * .pt2 文件结构（ZIP 归档）：
 *   - models/model.json           # ExportedProgram 的 JSON 表示
 *   - data/weights/*.pt           # 权重文件
 *   - data/constants/*.pt         # 常量文件
 *   - archive_format              # 格式标识 "pt2"
 *   - archive_version             # 版本号
 *
 * 编译：
 *   mkdir -p build && cd build
 *   cmake .. && make torch-export-infer
 *
 * 运行：
 *   export CUDA_HOME=/data/workspace/cuda-12.8
 *   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 *   ./torch-export-infer
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <torch/nativert/ModelRunnerHandle.h>
#include <torch/torch.h>

// 模型路径
const std::string PT2_MODEL_PATH = "../model/transformer_exported.pt2";
const std::string MODEL_NAME = "model"; // 默认模型名称

/**
 * 检查文件是否存在
 */
bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

/**
 * 执行推理
 */
torch::Tensor run_inference(
    torch::nativert::ModelRunnerHandle& runner,
    torch::Tensor input) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Running inference..." << std::endl;
  std::cout << "Input shape: " << input.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // 准备输入
  std::vector<c10::IValue> inputs;
  inputs.push_back(input);

  // 执行推理
  auto output = runner.runWithFlatInputsAndOutputs(std::move(inputs));

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto output_tensor = output[0].toTensor();

  std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
  std::cout << "Inference time: " << duration.count() / 1000.0 << " ms"
            << std::endl;
  std::cout << "========================================\n" << std::endl;

  return output_tensor;
}

int main(int argc, char* argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "torch.export (.pt2) Inference Example" << std::endl;
  std::cout << "========================================\n" << std::endl;

  if (!file_exists(PT2_MODEL_PATH)) {
    std::cerr << ".pt2 file not found: " << PT2_MODEL_PATH << std::endl;
    std::cerr << "Run: python3.11 export_transformer_torch_export.py"
              << std::endl;
    return 1;
  }

  try {
    // 1. 加载模型
    std::cout << "Loading model from: " << PT2_MODEL_PATH << std::endl;
    std::cout << "Model name: " << MODEL_NAME << std::endl;

    torch::nativert::ModelRunnerHandle runner(PT2_MODEL_PATH, MODEL_NAME);
    std::cout << "✓ Model loaded successfully\n" << std::endl;

    // 2. 使用固定输入进行推理
    std::cout << "========================================" << std::endl;
    std::cout << "Testing with fixed input" << std::endl;
    std::cout << "========================================" << std::endl;

    // 固定随机种子
    torch::manual_seed(42);
    auto test_input = torch::randint(0, 1000, {4, 32}, torch::kInt64);
    auto output = run_inference(runner, test_input);

    std::cout << "\nOutput values (complete):" << std::endl;
    std::cout << output << std::endl;

    // 打印统计信息
    std::cout << "\nOutput statistics:" << std::endl;
    std::cout << "  Mean:   " << output.mean().item<float>() << std::endl;
    std::cout << "  Std:    " << output.std().item<float>() << std::endl;
    std::cout << "  Min:    " << output.min().item<float>() << std::endl;
    std::cout << "  Max:    " << output.max().item<float>() << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    std::cerr << "\nNote: torch::nativert::ModelRunnerHandle requires:"
              << std::endl;
    std::cerr << "  - Correct .pt2 archive format" << std::endl;
    std::cerr << "  - Model exported with torch.export.export()" << std::endl;
    std::cerr << "  - Saved with torch.export.save()" << std::endl;
    return 1;
  }
}
