#include <dlpack/dlpack.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/vm/vm.h>

int main() {
    tvm::ffi::Module mod = tvm::ffi::Module::LoadFromFile("model.so");
    return 0;
}

/*

g++ tvm_infer.cc -I/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/include -I/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/3rdparty/dmlc-core/include -I/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/3rdparty/tvm-ffi/include -I/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include -L/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/lib -L/apdcephfs_zwfy/share_303204533/liamjhzhang/tvm/build -ltvm_runtime -o run_tvm

*/