import os
import torch
from model import ModelArgs, Transformer
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard

'''
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
'''

def main():
    rank = int(os.environ["LOCAL_RANK"])
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()
        device = torch.device(f"{device_type}:{rank}")
        torch.cuda.set_device(rank)
        print(f"Running on rank {rank} on device {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on device {device}")
    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)
    
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=2,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("cuda"):
        model = Transformer(model_args)
    
    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    
    # inspect_model
    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert param.dtype == torch.float32
    
    # inspect_mixed_precision
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for _ in range(1):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            output = model(x)
            loss = output.sum()
            print(f"output.shape: {output.shape}, loss: {loss}")
            loss.backward()
            optim.step()
            optim.zero_grad()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    
    main()