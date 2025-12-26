import os
import torch
from model import ModelArgs, Transformer

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
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for _ in range(1):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            loss = model(x).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

    print("Done")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    
    main()