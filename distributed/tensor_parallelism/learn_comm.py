import torch
import torch.distributed as dist
import os
import argparse
from torch.distributed.device_mesh import init_device_mesh

from communication import minimal_pad_to_divisible, _All2All

def run_all_to_all_demo():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Rank {rank}/{world_size} starting...")

    # --- 2. 准备发送数据 ---
    # 每个进程都有一个源张量 src。
    # 假设每个进程想发送给其他进程一个小的张量。
    # 为了简化，我们让每个进程发送给所有进程相同形状的数据。
    # src_tensor 的总大小是 world_size * (每个发送块的大小)。
    # 比如，如果 world_size=2，每个进程发送一个包含自己 rank 的 2x2 张量给另一个进程。
    # 那么 src_tensor 可能是 (2 * 2 * 2) = 8 个元素。
    # 更直接的理解：src 应该是一个 (world_size, *sub_tensor_shape) 的张量
    # src[j] 是要发送给 rank j 的数据。

    # 定义每个进程要发送给其他进程的子张量大小
    # 为了演示简单，我们让所有子张量大小一样，实际应用中可以不一样。
    sub_tensor_shape = (2, 4)
    
    # 构建源张量：每个进程的源张量 src_data
    # src_data 的第一维度是 world_size，表示要发送给每个 rank 的数据块
    # 每一个子块都填充为发送方的 rank 值 + 目标 rank 值，便于观察
    src_data = torch.zeros(world_size, *sub_tensor_shape, dtype=torch.float32).cuda(rank)

    for i in range(world_size):
        # 进程 rank 准备发送给进程 i 的数据
        # 比如 rank 0 准备发送给 rank 0 的数据是 [0,0,0,0]
        # rank 0 准备发送给 rank 1 的数据是 [0,1,0,1]
        # rank 1 准备发送给 rank 0 的数据是 [1,0,1,0]
        # rank 1 准备发送给 rank 1 的数据是 [1,1,1,1]
        src_data[i, :, :] = float(rank * 10 + i) # 简单地用 rank*10 + 目标rank 填充，便于区分

    print(f"Rank {rank} source data (before all_to_all):\n{src_data}\n")

    # --- 3. 准备接收数据张量 ---
    # 接收张量 dst_data 的形状与 src_data 相同，因为我们假设发送和接收块大小一致。
    # dst_data 的第一维度是 world_size，dst_data[j] 将存放从 rank j 接收到的数据。
    dst_data = torch.empty(world_size, *sub_tensor_shape, dtype=torch.float32).cuda(rank)

    # --- 4. 执行 all_to_all_single ---
    # `src_split_sizes` 和 `dst_split_sizes` 允许发送/接收不同大小的块。
    # 如果省略，则认为所有块大小相同（src.shape[0] / world_size）。
    # 在我们的例子中，src_data 的第一维就是 world_size，所以每个块就是 src_data[i]。
    dist.all_to_all_single(dst_data, src_data)

    print(f"Rank {rank} received data (after all_to_all):\n{dst_data} {dst_data.shape}\n")

    # --- 5. 验证结果 ---
    # 验证逻辑：
    # 进程 `rank` 的 `dst_data[j]` 应该等于进程 `j` 的 `src_data[rank]`。
    # 在我们的填充逻辑中，进程 `j` 的 `src_data[rank]` 应该是 `float(j * 10 + rank)`。
    expected_data = torch.zeros(world_size, *sub_tensor_shape, dtype=torch.float32).cuda(rank)
    for j in range(world_size):
        # 进程 `rank` 从进程 `j` 接收的数据，应该是由进程 `j` 发送给进程 `rank` 的数据。
        # 进程 `j` 发送给进程 `rank` 的数据在进程 `j` 的 `src_data[rank]` 位置。
        # 根据填充规则，这个值是 `j * 10 + rank`。
        expected_data[j, :, :] = float(j * 10 + rank)
    
    # 允许浮点数误差
    if torch.allclose(dst_data, expected_data):
        print(f"Rank {rank} verification successful!")
    else:
        print(f"Rank {rank} verification FAILED!")
        print(f"Rank {rank} Expected data:\n{expected_data}\n")


    dist.destroy_process_group()
    print(f"Rank {rank} finished.")


def run_all_to_all_demo2():
    rank = int(os.environ["RANK"])
    inp = torch.arange(4) + rank * 4
    print(f"Rank {rank} inp: {inp}")
    inp = list(inp.chunk(4))
    print(f"Rank {rank} inp: {inp}")
    output = list(torch.empty([4], dtype=torch.int64).chunk(4))
    print(f"Rank {rank} output: {output}")

if __name__ == "__main__":

    # --- 1. 初始化进程组 ---
    # understand world topology
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_type = torch.accelerator.current_accelerator().type

    device_mesh = init_device_mesh(device_type, (1, world_size), mesh_dim_names=("dp","tp"))

    tp_mesh = device_mesh["tp"]
    sp_group = tp_mesh.get_group()

    if torch.distributed.get_rank() == 0:
        print(f"sp_group: {sp_group}, sp_group_size: {dist.get_world_size(sp_group)}, type: {type(sp_group)}")

    # run_all_to_all_demo()
    # run_all_to_all_demo2()
    
    inp = torch.randn(world_size, 4).cuda(rank)
    print(f"Rank {rank} inp: {inp}, inp.shape: {inp.shape}")
    inp = _All2All.apply(inp, 0, 1, sp_group, False)
    print(f"Rank {rank} out: {inp}, out.shape: {inp.shape}")
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    
# torchrun --nnodes=1 --nproc_per_node=4 learn_comm.py