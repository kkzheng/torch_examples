import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
import copy
from torch.amp import  autocast
os.environ['NCCL_ALGO']='Ring'

from torch.utils.checkpoint import checkpoint

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 2)

    def _checkpoint_forward(self, x):
        # 注意：这里的操作是模型的一部分，所以可以直接访问 self.linear1, self.linear2
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

    def forward(self, x):
        x = checkpoint(self._checkpoint_forward, x, use_reentrant=False)
        print(f'x.dtype:{x.dtype}')
        x = self.linear3(x)
        return x

def main():
    mesh_all = init_device_mesh("cuda",(1,2),mesh_dim_names=("shard","replica"))
    sp_mesh = mesh_all["replica"]
    sp_group = sp_mesh.get_group()
    rank = dist.get_rank()
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)

    model = SimpleModel()
    model_fsdp = FSDP(
        model,
        device_id=rank,
        use_orig_params=True,  # 为了能访问 .grad
        device_mesh=sp_mesh,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True),
    )
    shape = (2,2)
    # input_tensor = torch.randn(shape, generator=generator, requires_grad=True)
    datas = []
    for i in range(5):
        datas.append(torch.randn(shape, generator=generator, requires_grad=True))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_fsdp.parameters()),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # model_fsdp = model_fsdp.to(torch.bfloat16)

    for name, param in model_fsdp.named_parameters():
        print(f'name:{name};param.dtype:{param.dtype}')

    model_fsdp.train()
    activities = [ProfilerActivity.CPU]
    activities += [ProfilerActivity.CUDA]
    for i in range(1):
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_train_"+str(i)):
                data = datas[i].to(torch.cuda.current_device())
                target = torch.tensor([0, 1],dtype=torch.long,device=torch.cuda.current_device())

                with record_function("forward"):
                    optimizer.zero_grad()
                    # with autocast("cuda",dtype=torch.bfloat16,enabled=True):
                    output = model_fsdp(data)

                print(f'output.dtype:{output.dtype}')

                with record_function("backward"):
                    loss = F.cross_entropy(output,target)
                    loss.backward()
                optimizer.step()

                param_group = optimizer.param_groups[0]
                for p in param_group['params']:
                    if p.grad is not None:
                        # p.grad 是分片后的梯度，其类型由 reduce_dtype 和 FSDP 决定，
                        # 在您的配置中，reduce_scatter 后的梯度通常是 float32 (除非使用 grad_scaler 导致的 scale/unscale)
                        # 但我们这里要检查的是 m 和 v 的类型

                        # 获取当前参数的优化器状态
                        state = optimizer.state[p]

                        # 检查 m 和 v
                        if 'exp_avg' in state and 'exp_avg_sq' in state:
                            m_dtype = state['exp_avg'].dtype
                            v_dtype = state['exp_avg_sq'].dtype
                            
                            print(f"--- Optimizer State for a parameter ---")
                            print(f"Parameter dtype: {p.dtype}")
                            print(f"m (exp_avg) dtype: {m_dtype}")
                            print(f"v (exp_avg_sq) dtype: {v_dtype}")
                            
                            # 找到一个即可
                            break

        if torch.distributed.get_rank() == 0:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 test_fsdp_mixpolicy.py

'''
master的 rank需要是0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=29.162.244.14 --master_port=29508 FSDP2/test_fsdp_mixpolicy.py
'''
