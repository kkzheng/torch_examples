#!/usr/bin/env python3
"""
导出 TorchScript 模型用于 C++ Sequence Parallelism

关键点：
1. 使用 torch.jit.script 而不是 trace（保留控制流）
2. 导出简单的 Transformer-like 模型
3. C++ 可以访问和修改每一层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

output_path = "../model/transformer_model.pt"

class SimpleTransformer(nn.Module):
    """
    简化的 Transformer 模型
    用于演示 Sequence Parallelism

    新增功能：支持在 C++ 中替换 Attention
    - get_attention_input(x): 获取 attention 输入
    - forward_with_custom_attention(x, attn_out): 使用自定义 attention 输出
    """
    def __init__(self, d_model=256, n_heads=4, seq_len=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len

        # Embedding
        self.embedding = nn.Embedding(1000, d_model)

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

    @torch.jit.export
    def get_attention_input(self, x):
        """
        获取 Attention 层的输入
        用于在 C++ 中调用自定义 SP Attention

        Args:
            x: [batch, seq_len] - token ids
        Returns:
            h: [batch, seq_len, d_model] - attention 输入 (after embedding + ln1)
            residual: [batch, seq_len, d_model] - 用于 residual connection
        """
        h = self.embedding(x)
        residual = h
        h = self.ln1(h)
        return h, residual

    @torch.jit.export
    def forward_with_custom_attention(self, attn_output, residual):
        """
        使用自定义 Attention 输出继续 forward
        用于在 C++ 中替换 Attention 层

        Args:
            attn_output: [batch, seq_len, d_model] - 自定义 attention 的输出
            residual: [batch, seq_len, d_model] - attention 之前的 residual
        Returns:
            out: [batch, 1] - 最终输出
        """
        # Residual connection after attention
        h = residual + attn_output

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
    """导出 TorchScript 模型"""

    # 创建模型
    model = SimpleTransformer(d_model=256, n_heads=4, seq_len=32)
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

    print("Model structure:")
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
        test_input = torch.randint(0, 1000, (2, 32))  # [batch=2, seq_len=32]

        # 测试 1: 标准 forward
        test_output = model(test_input)
        print(f"\n1. Standard forward:")
        print(f"   Input shape:  {list(test_input.shape)}")
        print(f"   Output shape: {list(test_output.shape)}")
        print(f"   Sample output: {test_output[:, 0].tolist()}")

        # 测试 2: 使用新的模块化方法
        print(f"\n2. Modular forward (for C++ usage):")
        attn_input, residual = model.get_attention_input(test_input)
        print(f"   Attention input shape: {list(attn_input.shape)}")
        print(f"   Residual shape: {list(residual.shape)}")

        # 手动计算 attention (模拟 C++ 中的 SP Attention)
        # 这里使用模型内部的 attention 权重来验证
        q = torch.matmul(attn_input, model.q_proj.weight.t())
        k = torch.matmul(attn_input, model.k_proj.weight.t())
        v = torch.matmul(attn_input, model.v_proj.weight.t())

        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)

        scale = 1.0 / (model.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=test_input.device), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, model.d_model)
        attn_out = torch.matmul(attn_out, model.o_proj.weight.t())

        print(f"   Manual attention output shape: {list(attn_out.shape)}")

        # 使用 forward_with_custom_attention 继续
        test_output_custom = model.forward_with_custom_attention(attn_out, residual)
        print(f"   Final output shape: {list(test_output_custom.shape)}")
        print(f"   Sample output: {test_output_custom[:, 0].tolist()}")

        # 验证两种方式输出一致
        if torch.allclose(test_output, test_output_custom, atol=1e-5):
            print(f"\n   ✓ Standard and modular forwards match!")
        else:
            max_diff = (test_output - test_output_custom).abs().max().item()
            print(f"\n   ✗ Warning: Outputs differ (max diff: {max_diff:.2e})")

    # 使用 torch.jit.script 导出
    print("\n" + "="*60)
    print("Exporting to TorchScript...")
    print("="*60)

    scripted_model = torch.jit.script(model)

    scripted_model.save(output_path)

    print(f"✓ Model saved to: {output_path}")

    # 验证加载
    print("\nVerifying saved model...")
    loaded_model = torch.jit.load(output_path)

    with torch.no_grad():
        torch.manual_seed(42)
        test_input = torch.randint(0, 1000, (2, 32))
        loaded_output = loaded_model(test_input)

        if torch.allclose(test_output, loaded_output):
            print("✓ Loaded model produces identical output")
        else:
            print("✗ Warning: Output mismatch")

    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Model: SimpleTransformer")
    print(f"  d_model: 256")
    print(f"  n_heads: 4")
    print(f"  seq_len: 32")
    print(f"  vocab_size: 1000")
    print(f"\nSaved to: {output_path}")
    print(f"Format: TorchScript (torch.jit.script)")
    print(f"\n🎯 Exported Methods for C++:")
    print(f"  1. forward(x)")
    print(f"     - Standard inference with built-in attention")
    print(f"     - Input: [batch, seq_len] token ids")
    print(f"     - Output: [batch, 1] predictions")
    print(f"\n  2. get_attention_input(x)")
    print(f"     - Get attention layer input (after embedding + ln1)")
    print(f"     - Returns: (attn_input, residual)")
    print(f"     - Use this to inject custom SP Attention")
    print(f"\n  3. forward_with_custom_attention(attn_output, residual)")
    print(f"     - Continue forward with custom attention output")
    print(f"     - Input: attention output + residual from step 2")
    print(f"     - Output: [batch, 1] predictions")
    print(f"\n💡 C++ Usage Pattern (Replace Attention):")
    print(f"  // 1. Get attention input")
    print(f"  auto result = model.get_method(\"get_attention_input\")({{tokens}});")
    print(f"  auto attn_input = result.toTuple()->elements()[0].toTensor();")
    print(f"  auto residual = result.toTuple()->elements()[1].toTensor();")
    print(f"\n  // 2. Use your custom SP Attention")
    print(f"  auto attn_output = my_sp_attention.forward(attn_input, sp_size);")
    print(f"\n  // 3. Continue model forward")
    print(f"  auto output = model.get_method(\"forward_with_custom_attention\")(")
    print(f"      {{attn_output, residual}}).toTensor();")
    print(f"\n✅ Benefits:")
    print(f"  - No need to manually rebuild full model pipeline")
    print(f"  - Model structure changes don't require C++ code changes")
    print(f"  - Only replace the attention layer, rest uses original model")
    print(f"\nReady for C++ Sequence Parallelism!")
    print("="*60)

    # 打印权重信息（用于 C++ 调试）
    print("\nKey layer shapes (for C++ reference):")
    print(f"  q_proj.weight: {list(model.q_proj.weight.shape)}")
    print(f"  k_proj.weight: {list(model.k_proj.weight.shape)}")
    print(f"  v_proj.weight: {list(model.v_proj.weight.shape)}")
    print(f"  o_proj.weight: {list(model.o_proj.weight.shape)}")

if __name__ == "__main__":
    export_model()
