import os
import inspect
import numpy as np
import torch
from torchvision.models.resnet import ResNet18_Weights, resnet18

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

with torch.no_grad():
    exported_program = torch.export.export(torch_model, args=example_args)
    print(f" type of exported_program: {type(exported_program)}")
    mod = from_exported_program(exported_program, keep_params_as_input=True)

mod, params = relax.frontend.detach_params(mod)
# mod.show()
print(f" type of mod: {type(mod)}")

TOTAL_TRIALS = 0  # Change to 20000 for better performance if needed

# nvidia-smi --query-gpu=name,compute_cap --format=csv
dev = tvm.cuda()
target = tvm.target.Target.from_device(dev)
print(f" target: {target}")
work_dir = "tuning_logs"

# grep -R "def static_shape_tuning" -n /apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/python
mod = relax.get_pipeline(
    "static_shape_tuning",
    target=target,
    work_dir=work_dir,
    total_trials=TOTAL_TRIALS,
)(mod)

# mod["main"].show()
print(f" type of mod: {type(mod)}")

with target:
    mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
ex = tvm.compile(mod, target=target)
print(f" type of ex: {type(ex)}")

vm = relax.VirtualMachine(ex, dev)
gpu_data = tvm.runtime.tensor(example_args[0], dev)
gpu_params = [tvm.runtime.tensor(p, dev) for p in params["main"]]
print(f" len of gpu_params: {len(gpu_params)}")
opt_out = vm["main"](gpu_data, *gpu_params)[0].numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_model = torch_model.to(device)
example_args = [arg.to(device) for arg in example_args]
with torch.no_grad():
    ori_out = torch_model(*example_args).cpu().numpy()

print(opt_out.shape)
print(ori_out.shape)
if not tvm.testing.assert_allclose(opt_out, ori_out, rtol=1e-5, atol=1e-5):
    print(opt_out[0, :10])
    print(ori_out[0, :10])
    print("The output is not close enough.")
else:
    print("The output is close enough.")
