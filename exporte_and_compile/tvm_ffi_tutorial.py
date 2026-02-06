from tvm_ffi.cpp import load_inline

cuda_src = r"""
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <cuda_runtime.h>

TVM_REGISTER_GLOBAL("cuda_hello").set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    printf("Hello from CUDA kernel!\\n");
});
"""

load_inline(
    name="cuda_mod",
    cpp_sources="",
    cuda_sources=cuda_src,
    extra_cuda_cflags=["--use_fast_math"],
)

import tvm
tvm.get_global_func("cuda_hello")()
