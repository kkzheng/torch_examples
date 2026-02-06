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

output_path = "./model/transformer_model.pt"

class SimpleTransformer(nn.Module):
    """
    简化的 Transformer 模型
    用于演示 Sequence Parallelism
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
        test_output = model(test_input)

        print(f"Input shape:  {list(test_input.shape)}")
        print(f"Output shape: {list(test_output.shape)}")
        print(f"Sample output: {test_output[:, 0].tolist()}")

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
