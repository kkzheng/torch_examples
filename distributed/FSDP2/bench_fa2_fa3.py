import torch
from triton.testing import do_bench
import time

def benchmark_fa2_vs_fa3(
    B=4,
    H=32,
    N=4096,
    D=128,
    dtype=torch.bfloat16,
    causal=True,
    iters=50,
    warmup=10,
):
    device = "cuda"

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # -----------------------------
    # Import FA2 / FA3
    # -----------------------------
    from flash_attn.flash_attn_interface import flash_attn_func as fa2_func
    from flash_attn_interface import flash_attn_func as fa3_func

    # -----------------------------
    # CUDA events
    # -----------------------------
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def run(func, name):
        # warmup
        for _ in range(warmup):
            func(q, k, v, causal=causal)
        torch.cuda.synchronize()

        start.record()
        for _ in range(iters):
            out = func(q, k, v, causal=causal)
        end.record()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        print(f"{name:<10}: {ms:.3f} ms / iter")

    print("===== FlashAttention Benchmark =====")
    print(f"B={B}, H={H}, N={N}, D={D}, dtype={dtype}, causal={causal}")
    print("-----------------------------------")

    run(fa2_func, "FA2")
    run(fa3_func, "FA3")

def benchmark_fa2_vs_fa3_do_bench(
    B=2,
    H=32,
    N=8192,
    D=128,
    dtype=torch.bfloat16,
    causal=True,
    iters=50,
    warmup=10,
):
    device = "cuda"

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # -----------------------------
    # Import FA2 / FA3
    # -----------------------------
    from flash_attn.flash_attn_interface import flash_attn_func as fa2_func
    from flash_attn_interface import flash_attn_func as fa3_func

    # -----------------------------
    # Wrap functions
    # -----------------------------
    def run_fa2():
        return fa2_func(q, k, v, causal=causal)

    def run_fa3():
        return fa3_func(q, k, v, causal=causal)

    # -----------------------------
    # Benchmark
    # -----------------------------
    fa2_ms = do_bench(run_fa2)
    fa3_ms = do_bench(run_fa3)

    print("===== FlashAttention Benchmark (do_bench) =====")
    print(f"B={B}, H={H}, N={N}, D={D}, dtype={dtype}, causal={causal}")
    print("------------------------------------------------")
    print(f"FA2 : {fa2_ms:.3f} ms")
    print(f"FA3 : {fa3_ms:.3f} ms")

#benchmark_fa2_vs_fa3(
benchmark_fa2_vs_fa3_do_bench(
    B=2,
    H=32,
    N=8192,
    D=128,
    dtype=torch.bfloat16,
    causal=True,
    iters=100,
)
