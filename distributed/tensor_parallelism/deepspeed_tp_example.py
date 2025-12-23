import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate

tp_size = 2
logger = get_logger()

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])

dp_size = _world_size // tp_size
device_type = torch.accelerator.current_accelerator().type

device_mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

rank_log(_rank, logger, f"Device Mesh created: {device_mesh}")
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

def normal():
    simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=4, vocab_size=320)
    model = Transformer.from_model_args(simple_llama2_config).to(device_type)
    model.init_weights()

    rank_log(_rank, logger, f"Model {model}\n")

    lr = 3e-3
    rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    num_iterations = 1
    batch_size = 2

    for i in range(num_iterations):
        torch.manual_seed(2025)
        inp = torch.randint(320, (8, 2), device=device_type)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        rank_log(_rank, logger, f"iter {i} complete")

    rank_log(_rank, logger, "training successfully completed!")
    rank_log(_rank, logger, f"output is {output}")
    rank_log(_rank, logger, f"inp.shape is {inp.shape}, output.shape is {output.shape}")
    return output

if __name__ == "__main__":
    output = normal()
    if dist.is_initialized():
        dist.destroy_process_group()
        
        
# torchrun --nnodes=1 --nproc_per_node=4 deepspeed_tp_example.py 