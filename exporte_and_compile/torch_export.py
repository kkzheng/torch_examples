import os
import torch

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def export():
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model().to(device=device)
        example_inputs=(torch.randn(8, 10, device=device),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        # [Optional] Specify the first dimension of the input x as dynamic.
        exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )
        print(f"Model exported and compiled to {output_path}")

def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
    print(model(torch.randn(8, 10, device=device)))

if __name__ == "__main__":
    # export()
    load()