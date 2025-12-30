import onnxruntime.training
import torch
from torch.utils.cpp_extension import load_inline

# C++ source code for the square operation
cpp_source = """
torch::Tensor square_cpu(torch::Tensor input) {
    // Check that input is a CPU tensor
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");

    // Create output tensor with same shape and dtype as input
    torch::Tensor output = torch::empty_like(input);

    // Get data pointers
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Get total number of elements
    int64_t numel = input.numel();

    // For loop to compute square of each element
    for (int64_t i = 0; i < numel; i++) {
        output_data[i] = input_data[i] * input_data[i];
    }

    return output;
}
"""

# Load the extension inline
square_module = load_inline(
    name="square_cpu_kernel",
    cpp_sources=cpp_source,
    functions=["square_cpu"],
    verbose=True
)

# def square(x):
#     return square_module.square_cpu(x)

@torch.library.custom_op("mylib::square", mutates_args=())
def square(x: torch.Tensor) -> torch.Tensor:
    return square_module.square_cpu(x)

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@square.register_fake
def _(x):
    return x.new_empty(x.size())

@torch.compile(fullgraph=True)
def f(x):
    return square(x)

print(f(torch.tensor([1.0, 2.0, 3.0])))