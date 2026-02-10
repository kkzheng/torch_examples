#!/usr/bin/env python3
"""
使用 torch.export 导出 Transformer 模型

关键点：
1. 使用 torch.export.export() 而不是 TorchScript
2. 支持动态 batch size
3. 不使用 torch._inductor 编译，只做 export
4. 导出为 .pt2 格式

对比 TorchScript 方式：
- TorchScript (export_transformer_torchscript.py):
  * 使用 torch.jit.script() 或 torch.jit.trace()
  * 导出为 .pt 格式
  * 适合 C++ LibTorch 加载

- torch.export (本文件):
  * 使用 torch.export.export() - PyTorch 2.x 新特性
  * 导出为 .pt2 格式 (ExportedProgram)
  * 更严格的图捕获，无 Python 依赖
  * 支持更精确的动态形状定义
  * 可以后续用于 torch.compile 或 AOTI (不在本例中)

使用方法：
    # 设置环境变量
    export CUDA_HOME=/data/workspace/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # 运行导出
    python3.11 export_transformer_torch_export.py

    # 生成文件：./model/transformer_exported.pt2
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

output_path = "../model/transformer_exported.pt2"


class SimpleTransformer(nn.Module):
    """
    简化的 Transformer 模型
    用于演示 torch.export
    """
    def __init__(self, d_model=256, n_heads=4, vocab_size=1000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Attention Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed Forward
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

        # Layer Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Output
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len] - token ids

        # Embedding
        h = self.embedding(x)  # [batch, seq_len, d_model]

        # Attention block
        residual = h
        h = self.ln1(h)

        # Q, K, V projections
        q = self.q_proj(h)  # [batch, seq_len, d_model]
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Reshape for multi-head attention
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, n_heads, seq_len, head_dim]

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, n_heads, seq_len, seq_len]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [batch, n_heads, seq_len, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        out = self.o_proj(out)
        h = residual + out

        # Feed forward block
        residual = h
        h = self.ln2(h)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = residual + h

        # Output head (取最后一个 token)
        h = h[:, -1, :]  # [batch, d_model]
        out = self.output(h)  # [batch, 1]

        return out


def export_model():
    """使用 torch.export 导出模型"""

    device = "cpu"
    print(f"Exporting model for device: {device}")

    # 创建模型
    model = SimpleTransformer(d_model=256, n_heads=4, vocab_size=1000)
    model.to(device)
    model.eval()

    # 初始化权重
    torch.manual_seed(2025)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    model.apply(init_weights)

    print("\nModel structure:")
    print(model)
    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,} parameters")

    # 测试推理
    print("\n" + "="*60)
    print("Testing model before export...")
    print("="*60)

    with torch.no_grad():
        torch.manual_seed(42)
        test_input = torch.randint(0, 1000, (2, 32), device=device)  # [batch=2, seq_len=32]
        test_output = model(test_input)

        print(f"Input shape:  {list(test_input.shape)}")
        print(f"Output shape: {list(test_output.shape)}")
        print(f"Sample output: {test_output[:, 0].tolist()}")

    # 使用 torch.export.export 导出
    print("\n" + "="*60)
    print("Exporting with torch.export...")
    print("="*60)

    # 准备 example inputs
    example_inputs = (torch.randint(0, 1000, (8, 32), device=device),)

    # 定义动态维度 (可选)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    dynamic_shapes = {"x": {0: batch_dim}}  # 第一个维度 (batch) 是动态的

    # 导出模型
    exported_program = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes
    )

    print(f"✓ Model exported successfully")
    print(f"\nExported program info:")
    print(f"  Graph module: {type(exported_program.graph_module)}")
    print(f"  Number of graph nodes: {len(list(exported_program.graph_module.graph.nodes))}")

    # 保存导出的模型
    torch.export.save(exported_program, output_path)
    print(f"\n✓ Model saved to: {output_path}")

    # 验证加载
    print("\n" + "="*60)
    print("Verifying saved model...")
    print("="*60)

    loaded_program = torch.export.load(output_path)

    with torch.no_grad():
        torch.manual_seed(42)
        test_input = torch.randint(0, 1000, (2, 32), device=device)
        loaded_output = loaded_program.module()(test_input)

        if torch.allclose(test_output, loaded_output, rtol=1e-5, atol=1e-5):
            print("✓ Loaded model produces identical output")
            print(f"  Max diff: {(test_output - loaded_output).abs().max().item():.2e}")
        else:
            print("✗ Warning: Output mismatch")
            print(f"  Max diff: {(test_output - loaded_output).abs().max().item():.2e}")

    # 测试动态 batch size
    print("\n" + "="*60)
    print("Testing dynamic batch size...")
    print("="*60)

    for batch_size in [1, 4, 16]:
        test_input = torch.randint(0, 1000, (batch_size, 32), device=device)
        output = loaded_program.module()(test_input)
        print(f"  Batch size {batch_size:2d}: input {list(test_input.shape)} -> output {list(output.shape)}")

    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Model: SimpleTransformer")
    print(f"  d_model: 256")
    print(f"  n_heads: 4")
    print(f"  vocab_size: 1000")
    print(f"\nSaved to: {output_path}")
    print(f"Format: torch.export (ExportedProgram)")
    print(f"Dynamic shapes: batch dimension [1, 1024]")
    print(f"\nReady for inference!")
    print("="*60)

    # 打印权重信息（用于调试）
    print("\nKey layer shapes:")
    print(f"  q_proj.weight: {list(model.q_proj.weight.shape)}")
    print(f"  k_proj.weight: {list(model.k_proj.weight.shape)}")
    print(f"  v_proj.weight: {list(model.v_proj.weight.shape)}")
    print(f"  o_proj.weight: {list(model.o_proj.weight.shape)}")


def load_and_infer():
    """加载并推理导出的模型"""

    device = "cpu"
    print(f"\nLoading model for device: {device}")

    # 加载导出的模型
    loaded_program = torch.export.load(output_path)
    print(f"✓ Model loaded from: {output_path}")

    # 推理测试
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)

    with torch.no_grad():
        torch.manual_seed(999)
        test_input = torch.randint(0, 1000, (4, 32), device=device)
        output = loaded_program.module()(test_input)

        print(f"Input shape:  {list(test_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print(f"Output values:\n{output}")


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("../model", exist_ok=True)

    # 导出模型
    export_model()

    # 加载并推理
    print("\n" + "="*60)
    print("Testing load and inference...")
    print("="*60)
    load_and_infer()
