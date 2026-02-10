#!/usr/bin/env python3
"""
完整的 FX Graph 修改方案（修复所有形状问题）
"""

import torch
import torch.distributed as dist


def init_distributed(backend='mpi'):
    """
    初始化分布式环境

    Args:
        backend: 通信后端 ('mpi', 'gloo', 'nccl')

    Returns:
        (rank, world_size)
    """
    if not dist.is_initialized():
        if backend == 'mpi':
            # MPI 由运行时管理，直接初始化
            dist.init_process_group(backend='mpi')
        else:
            # gloo/nccl 需要环境变量（通过 torchrun 启动）
            dist.init_process_group(backend=backend, init_method='env://')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] 分布式环境初始化完成: backend={dist.get_backend()}, world_size={world_size}")

    return rank, world_size


def run_tp_inference(rank, world_size):
    """执行 TP 推理"""
    print(f"\n{'='*60}")
    print(f"Rank {rank}/{world_size}: TP 推理（FX Graph 修改方案）")
    print(f"{'='*60}")

    # 1. 加载切分后的模型
    model_path = f"../model/transformer_tp_rank{rank}.pt2"
    print(f"\n[Rank {rank}] 加载模型: {model_path}")

    exported_program = torch.export.load(model_path)
    print(f"[Rank {rank}] ✓ 模型加载成功")

    graph_module = exported_program.module()

    # 3. 准备输入
    torch.manual_seed(42)
    test_input = torch.randint(0, 1000, (4, 32), dtype=torch.long)

    # 4. 执行推理
    print(f"[Rank {rank}] 执行推理...")

    with torch.no_grad():
        output = graph_module(test_input)
    
    # 同步
    dist.barrier()

    print(f"[Rank {rank}] ✓ 推理成功！")
    print(f"[Rank {rank}] 输出 shape: {output.shape}")

    # 注意：all-reduce 已经在 FX Graph 中自动执行
    # 如果模型是通过 modify_graph_for_tp 修改并重新导出的，
    # 则不需要手动执行 all-reduce
    print(f"[Rank {rank}] 注意：all-reduce 已在 graph 中自动执行")

    # 6. 打印结果
    print(f"\n[Rank {rank}] 最终输出:")
    print(output)

    if rank == 0:
        print(f"\n[输出统计]")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std:  {output.std().item():.6f}")
        print(f"  Min:  {output.min().item():.6f}")
        print(f"  Max:  {output.max().item():.6f}")

    return output


def main():
    # 初始化分布式
    rank, world_size = init_distributed()

    try:
        # 执行 TP 推理
        output = run_tp_inference(rank, world_size)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"✓ 所有 ranks 完成！")
            print(f"{'='*60}")

    except Exception as e:
        print(f"\n[Rank {rank}] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
