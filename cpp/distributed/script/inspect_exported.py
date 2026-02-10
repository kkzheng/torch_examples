#!/usr/bin/env python3
"""
检查 ExportedProgram 的图结构和 Signature，并使用 torch._inductor 编译 AOT 模型

功能：
1. 加载 torch.export 导出的 .pt2 文件
2. 查看图结构 (graph)
3. 查看模型签名 (signature)
4. 查看输入输出规格 (input/output specs)
5. 查看所有节点和操作
6. 使用 torch._inductor 编译 AOT 模型
7. 对比编译前后的推理性能
8. 使用 Netron 可视化图结构
9. Tensor Parallel 支持（多 GPU 权重切分演示）

使用方法：
    # 基本检查（查看图结构和签名）
    python3.11 script/inspect_exported.py

    # 可视化图结构（生成 ONNX 文件）
    python3.11 script/inspect_exported.py --visualize

    # 性能对比（torch._inductor 编译前后）
    python3.11 script/inspect_exported.py --benchmark

    # Tensor Parallel 权重切分演示（需要使用 torchrun 启动）
    torchrun --nproc_per_node=2 script/inspect_exported.py --tp
    torchrun --nproc_per_node=4 script/inspect_exported.py --tp

    # 或使用便捷脚本
    ./script/run_tp_inference.sh 2  # 2 个进程
    ./script/run_tp_inference.sh 4  # 4 个进程

注意：
    - --tp 模式必须使用 torchrun 启动，不支持单进程
    - TP 模式会导出每个 rank 的切分模型（transformer_tp_rank0.pt2, transformer_tp_rank1.pt2, ...）
    - 可以在 C++ 中使用 ProcessGroupMPI 加载对应 rank 的模型进行分布式推理
"""

import os
import time
import argparse
import torch
import torch.distributed as dist


def init_distributed(backend='gloo'):
    """
    初始化分布式环境

    Args:
        backend: 通信后端 ('mpi', 'gloo', 'nccl')

    Returns:
        (rank, world_size)
    """
    if not dist.is_initialized():
        if backend == 'mpi':
            # MPI 由运行时管理，直接初始化
            dist.init_process_group(backend='mpi')
        else:
            # gloo/nccl 需要环境变量（通过 torchrun 启动）
            dist.init_process_group(backend=backend, init_method='env://')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] 分布式环境初始化完成: backend={dist.get_backend()}, world_size={world_size}")

    return rank, world_size


def all_reduce_hook(tensor):
    """
    All-reduce 钩子函数，用于插入到 FX Graph

    在 row-parallel 层（o_proj, fc2）之后调用，合并各 rank 的部分输出
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def modify_graph_for_tp(exported_program, tp_size: int, rank: int):
    """
    运行时修改 FX Graph 以适配 TP，并插入 all-reduce 节点

    关键修改：
    1. Q/K/V 的 view: [batch, seq, n_heads, head_dim] -> [batch, seq, n_heads//tp_size, head_dim]
    2. Attention 输出的 view: [batch, seq, d_model] -> [batch, seq, d_model//tp_size]
    3. 在 o_proj 和 fc2 后插入 all-reduce 节点（row-parallel 层需要合并部分输出）

    Args:
        exported_program: ExportedProgram 对象
        tp_size: TP 大小
        rank: 当前 rank

    Returns:
        修改后的 graph_module
    """
    print(f"\n[Rank {rank}] 修改 FX Graph 以适配 TP (tp_size={tp_size})")

    graph_module = exported_program.module()
    graph = graph_module.graph

    # 统计修改次数
    n_qkv_views = 0
    n_output_views = 0
    n_allreduce_inserted = 0

    # 第一步：修复形状
    for node in graph.nodes:
        if node.target == torch.ops.aten.view.default:
            if len(node.args) >= 2:
                shape_arg = node.args[1]
                if isinstance(shape_arg, (list, tuple)):
                    if len(shape_arg) == 4:
                        # Q/K/V 的 reshape: [batch, seq, n_heads, head_dim]
                        if shape_arg[2] == 4 and shape_arg[3] == 64:
                            new_n_heads = 4 // tp_size
                            new_shape = [shape_arg[0], shape_arg[1], new_n_heads, shape_arg[3]]
                            node.args = (node.args[0], new_shape)
                            n_qkv_views += 1
                            print(f"  [Rank {rank}] 修复 Q/K/V view: {node.name}")
                            print(f"    {list(shape_arg)} -> {new_shape}")

                    elif len(shape_arg) == 3:
                        # Attention 输出的 reshape: [batch, seq, d_model]
                        if shape_arg[2] == 256:  # d_model
                            new_d_model = 256 // tp_size
                            new_shape = [shape_arg[0], shape_arg[1], new_d_model]
                            node.args = (node.args[0], new_shape)
                            n_output_views += 1
                            print(f"  [Rank {rank}] 修复 Attention 输出 view: {node.name}")
                            print(f"    {list(shape_arg)} -> {new_shape}")

    # 第二步：在 row-parallel 层后插入 all-reduce
    # 需要找到 o_proj 和 fc2 对应的 linear 操作
    nodes_to_insert_allreduce = []

    for node in graph.nodes:
        if node.target == torch.ops.aten.linear.default:
            # 检查这个 linear 的权重参数名称
            if len(node.args) >= 2:
                weight_node = node.args[1]
                if hasattr(weight_node, 'target'):
                    param_name = str(weight_node.target)
                    # 检查是否是 row-parallel 层
                    if 'o_proj' in param_name or 'fc2' in param_name:
                        nodes_to_insert_allreduce.append((node, param_name))

    print(f"\n  [Rank {rank}] 找到 {len(nodes_to_insert_allreduce)} 个需要插入 all-reduce 的位置")

    # 插入 all-reduce 调用
    for linear_node, param_name in nodes_to_insert_allreduce:
        with graph.inserting_after(linear_node):
            # 创建 all-reduce 函数调用节点
            allreduce_node = graph.call_function(
                all_reduce_hook,
                args=(linear_node,),
            )

            # 替换所有使用 linear_node 的地方（除了 allreduce_node 本身）
            linear_node.replace_all_uses_with(allreduce_node)
            # 恢复 allreduce_node 的输入为 linear_node
            allreduce_node.args = (linear_node,)

            n_allreduce_inserted += 1
            print(f"  [Rank {rank}] 在 {param_name} 后插入 all-reduce")

    # 重新编译
    graph_module.recompile()

    print(f"  [Rank {rank}] ✓ 修改完成:")
    print(f"    - {n_qkv_views} 个 Q/K/V views")
    print(f"    - {n_output_views} 个输出 views")
    print(f"    - {n_allreduce_inserted} 个 all-reduce 节点\n")

    return graph_module


def apply_tensor_parallel(exported_program, tp_rank: int, tp_size: int):
    """
    对 ExportedProgram 应用 Tensor Parallel 切分

    策略：
    - Linear 层按列切分（Column Parallel）：将权重 [out_features, in_features]
      沿 out_features 维度切分为 tp_size 份
    - 特殊处理 Attention 的 Q/K/V/O 投影
    - FFN 的 fc1 按列切分，fc2 按行切分

    完整 TP 流程：
    1. 权重切分（本函数）
    2. FX Graph 修改（modify_graph_for_tp）

    Args:
        exported_program: ExportedProgram 对象
        tp_rank: 当前进程的 rank
        tp_size: 总 TP 进程数

    Returns:
        修改后的 exported_program（仅权重切分）
    """
    print("=" * 60)
    print(f"Applying Tensor Parallel (rank {tp_rank}/{tp_size})")
    print("=" * 60)

    state_dict = exported_program.state_dict

    # 需要切分的权重列表（按列切分）
    column_parallel_weights = [
        'q_proj.weight',  # [256, 256] -> [256/tp_size, 256]
        'k_proj.weight',
        'v_proj.weight',
        'fc1.weight',     # [1024, 256] -> [1024/tp_size, 256]
    ]

    # 需要切分的 bias（按列切分）
    column_parallel_biases = [
        'fc1.bias',       # [1024] -> [1024/tp_size]
    ]

    # 按行切分的权重
    row_parallel_weights = [
        'o_proj.weight',  # [256, 256] -> [256, 256/tp_size]
        'fc2.weight',     # [256, 1024] -> [256, 1024/tp_size]
    ]

    memory_saved = 0
    total_params = 0

    # 创建新的 state_dict 来存储切分后的参数
    new_state_dict = {}

    print("\nSharding weights:")
    for name, param in state_dict.items():
        original_size = param.numel() * param.element_size()
        total_params += param.numel()

        # 判断参数是否是 nn.Parameter
        is_parameter = isinstance(param, torch.nn.Parameter)

        # 检查是否是需要切分的权重
        is_column_parallel = any(w in name for w in column_parallel_weights)
        is_column_bias = any(b in name for b in column_parallel_biases)
        is_row_parallel = any(w in name for w in row_parallel_weights)

        if is_column_parallel:
            # 按列切分（第 0 维）
            original_shape = param.shape
            if len(original_shape) == 2:
                # Linear weight: [out_features, in_features]
                out_features = original_shape[0]
                chunk_size = out_features // tp_size
                start = tp_rank * chunk_size
                end = start + chunk_size

                # 保持为 Parameter 如果原来是 Parameter
                sharded_tensor = param[start:end, :].clone()
                if is_parameter:
                    new_state_dict[name] = torch.nn.Parameter(sharded_tensor)
                else:
                    new_state_dict[name] = sharded_tensor

                print(f"  ✂️  {name}: {list(original_shape)} -> {list(new_state_dict[name].shape)} (column parallel)")

                # 计算节省的内存
                new_size = new_state_dict[name].numel() * new_state_dict[name].element_size()
                memory_saved += (original_size - new_size)

        elif is_column_bias:
            # bias 按列切分
            original_shape = param.shape
            if len(original_shape) == 1:
                out_features = original_shape[0]
                chunk_size = out_features // tp_size
                start = tp_rank * chunk_size
                end = start + chunk_size

                sharded_tensor = param[start:end].clone()
                if is_parameter:
                    new_state_dict[name] = torch.nn.Parameter(sharded_tensor)
                else:
                    new_state_dict[name] = sharded_tensor

                print(f"  ✂️  {name}: {list(original_shape)} -> {list(new_state_dict[name].shape)} (column parallel bias)")

                new_size = new_state_dict[name].numel() * new_state_dict[name].element_size()
                memory_saved += (original_size - new_size)

        elif is_row_parallel:
            # 按行切分（第 1 维）
            original_shape = param.shape
            if len(original_shape) == 2:
                # Linear weight: [out_features, in_features]
                in_features = original_shape[1]
                chunk_size = in_features // tp_size
                start = tp_rank * chunk_size
                end = start + chunk_size

                sharded_tensor = param[:, start:end].clone()
                if is_parameter:
                    new_state_dict[name] = torch.nn.Parameter(sharded_tensor)
                else:
                    new_state_dict[name] = sharded_tensor

                print(f"  ✂️  {name}: {list(original_shape)} -> {list(new_state_dict[name].shape)} (row parallel)")

                new_size = new_state_dict[name].numel() * new_state_dict[name].element_size()
                memory_saved += (original_size - new_size)
        else:
            # 不切分的参数（embedding, layer_norm, output）
            # 保持原样（包括 Parameter 属性）
            new_state_dict[name] = param if is_parameter else param.clone()
            print(f"  📦 {name}: {list(param.shape)} (replicated)")

    # 更新 state_dict
    exported_program.state_dict.clear()
    exported_program.state_dict.update(new_state_dict)

    print(f"\n💾 Memory Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Memory saved per rank: {memory_saved / 1024 / 1024:.2f} MB")
    print(f"   Memory reduction: {memory_saved / (total_params * 4) * 100:.1f}%")
    print()

    return exported_program


def inspect_exported_program(pt2_path: str):
    """
    检查 ExportedProgram 的详细信息

    Args:
        pt2_path: .pt2 文件路径
    """
    print("=" * 60)
    print("Loading ExportedProgram")
    print("=" * 60)
    print(f"Path: {pt2_path}\n")

    # 1. 加载 ExportedProgram
    exported_program = torch.export.load(pt2_path)

    print("✓ ExportedProgram loaded successfully\n")

    # 2. 基本信息
    print("=" * 60)
    print("Basic Information")
    print("=" * 60)
    print(f"Type: {type(exported_program)}")
    print(f"Dialect: {exported_program.dialect}")
    if hasattr(exported_program, 'verifiers'):
        print(f"Verifiers: {exported_program.verifiers}")
    print()

    # 3. 获取 Graph Module
    print("=" * 60)
    print("Graph Module")
    print("=" * 60)
    graph_module = exported_program.graph_module
    print(f"Type: {type(graph_module)}")
    print(f"Graph: {type(graph_module.graph)}")
    print(f"Number of nodes: {len(list(graph_module.graph.nodes))}")
    print()

    # 4. 获取 Signature
    print("=" * 60)
    print("Signature")
    print("=" * 60)
    signature = exported_program.graph_signature
    print(f"Type: {type(signature)}")
    print()

    # 4.1 输入参数
    print("Input Specifications:")
    print(f"  - User inputs: {signature.user_inputs}")
    print(f"  - Parameters: {signature.parameters}")
    print(f"  - Buffers: {signature.buffers}")
    print()

    # 4.2 输出
    print("Output Specifications:")
    print(f"  - User outputs: {signature.user_outputs}")
    print(f"  - Buffer mutations: {signature.buffers_to_mutate}")
    print()

    # 5. 图的节点详情
    print("=" * 60)
    print("Graph Nodes (first 20)")
    print("=" * 60)
    for i, node in enumerate(graph_module.graph.nodes):
        if i >= 20:
            remaining = len(list(graph_module.graph.nodes)) - 20
            print(f"... and {remaining} more nodes")
            break

        print(f"\n{i+1}. Node: {node.name}")
        print(f"   Op: {node.op}")
        print(f"   Target: {node.target}")

        if node.op == 'call_function':
            print(f"   Function: {node.target}")

        # 输入参数
        if node.args:
            args_str = ', '.join([str(arg) if not hasattr(arg, 'name') else arg.name for arg in node.args])
            print(f"   Args: {args_str[:80]}{'...' if len(args_str) > 80 else ''}")

        # 输出形状（如果有）
        if hasattr(node, 'meta') and 'val' in node.meta:
            val = node.meta['val']
            if hasattr(val, 'shape'):
                print(f"   Output shape: {val.shape}")

    print()

    # 6. 统计操作类型
    print("=" * 60)
    print("Operation Statistics")
    print("=" * 60)

    op_counts = {}
    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target)
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

    print(f"Total unique operations: {len(op_counts)}\n")
    print("Top operations:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {op}: {count}")

    print()

    # 7. 参数和状态字典
    print("=" * 60)
    print("Parameters and Buffers")
    print("=" * 60)

    state_dict = exported_program.state_dict
    print(f"Total parameters and buffers: {len(state_dict)}\n")

    for name, tensor in list(state_dict.items())[:10]:
        print(f"  {name}: {tensor.shape} ({tensor.dtype})")

    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more")

    print()

    # 8. 输入输出规格 (TreeSpec)
    print("=" * 60)
    print("Input/Output TreeSpec")
    print("=" * 60)

    call_spec = exported_program.call_spec
    print("Input TreeSpec:")
    print(f"  {call_spec.in_spec}")
    print()
    print("Output TreeSpec:")
    print(f"  {call_spec.out_spec}")
    print()

    # 9. 动态维度约束
    print("=" * 60)
    print("Range Constraints (Dynamic Shapes)")
    print("=" * 60)

    if exported_program.range_constraints:
        for name, constraint in exported_program.range_constraints.items():
            print(f"  {name}: {constraint}")
    else:
        print("  No dynamic shape constraints")

    print()

    return exported_program


def benchmark_inference(model_fn, input_tensor, warmup=5, iterations=50, name="Model"):
    """
    性能测试

    Args:
        model_fn: 模型函数
        input_tensor: 输入张量
        warmup: 预热次数
        iterations: 测试迭代次数
        name: 模型名称

    Returns:
        平均推理时间（毫秒）
    """
    print(f"\nBenchmarking {name}...")
    print(f"  Warmup: {warmup} iterations")
    print(f"  Test: {iterations} iterations")

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_fn(input_tensor)

    # 性能测试
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model_fn(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")

    return avg_time


def visualize_graph_with_netron(exported_program, output_path="../model/graph_visualization.onnx"):
    """
    使用 Netron 可视化图结构

    方法：
    1. 将 ExportedProgram 转换为 ONNX 格式
    2. 使用 Netron 打开 ONNX 文件进行可视化

    Args:
        exported_program: ExportedProgram 对象
        output_path: ONNX 文件输出路径
    """
    print("=" * 60)
    print("Visualizing Graph with Netron")
    print("=" * 60)

    try:
        # 方式1：直接保存 graph module 为 ONNX
        print("\nMethod 1: Export to ONNX for Netron visualization")
        print(f"Output path: {output_path}")

        # 准备示例输入
        batch_size = 2
        seq_len = 32
        vocab_size = 1000
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        print(f"Dummy input shape: {dummy_input.shape}")

        # 使用 torch.onnx.export 导出
        print("\nExporting to ONNX...")
        torch.onnx.export(
            exported_program.module(),  # 模型
            dummy_input,                # 示例输入
            output_path,                # 输出路径
            export_params=True,         # 导出参数
            opset_version=17,           # ONNX opset 版本
            do_constant_folding=True,   # 常量折叠优化
            input_names=['input'],      # 输入名称
            output_names=['output'],    # 输出名称
            dynamic_axes={              # 动态维度
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✓ ONNX file saved to: {output_path}")
        print()

        # 方式2：保存 FX Graph 的文本表示
        graph_text_path = output_path.replace('.onnx', '_fx_graph.txt')
        print(f"\nMethod 2: Save FX Graph as text")
        print(f"Output path: {graph_text_path}")

        with open(graph_text_path, 'w') as f:
            # 保存图的代码表示
            f.write("=" * 80 + "\n")
            f.write("FX Graph Code Representation\n")
            f.write("=" * 80 + "\n\n")
            f.write(str(exported_program.graph_module.code))
            f.write("\n\n")

            # 保存图的结构
            f.write("=" * 80 + "\n")
            f.write("FX Graph Structure\n")
            f.write("=" * 80 + "\n\n")
            f.write(str(exported_program.graph_module.graph))
            f.write("\n\n")

            # 保存节点详情
            f.write("=" * 80 + "\n")
            f.write("Graph Nodes Detail\n")
            f.write("=" * 80 + "\n\n")

            for i, node in enumerate(exported_program.graph_module.graph.nodes):
                f.write(f"Node {i+1}: {node.name}\n")
                f.write(f"  Op:     {node.op}\n")
                f.write(f"  Target: {node.target}\n")

                if node.args:
                    f.write(f"  Args:   {node.args}\n")

                if node.kwargs:
                    f.write(f"  Kwargs: {node.kwargs}\n")

                if hasattr(node, 'meta') and 'val' in node.meta:
                    val = node.meta['val']
                    if hasattr(val, 'shape'):
                        f.write(f"  Shape:  {val.shape}\n")

                f.write("\n")

        print(f"✓ FX Graph text saved to: {graph_text_path}")
        print()

        print()
        print("=" * 60)
        print("Visualization Options Summary")
        print("=" * 60)
        print()
        print("1. ONNX format (Netron):")
        print(f"   File: {output_path}")
        print("   View online: https://netron.app (drag and drop the .onnx file)")
        print(f"   Or install: pip install netron && netron {output_path}")
        print()
        print("2. FX Graph text:")
        print(f"   File: {graph_text_path}")
        print(f"   View: cat {graph_text_path}")
        print()
        print("=" * 60)
        print()

        return output_path

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_with_inductor(exported_program):
    """
    使用 torch._inductor 编译 ExportedProgram

    Args:
        exported_program: ExportedProgram 对象

    Returns:
        compiled_model: 编译后的模型
    """
    print("=" * 60)
    print("Compiling with torch._inductor (AOT)")
    print("=" * 60)

    try:
        import torch._inductor

        print("✓ torch._inductor available")
        print("\nCompilation options:")
        print("  - mode: 'max-autotune' for best performance")
        print("  - fullgraph: True")
        print("  - dynamic: False (static shapes)")

        # 使用 torch.compile 编译
        print("\nStarting compilation...")
        start_time = time.time()

        # 方式1：使用 torch.compile
        compiled_model = torch.compile(
            exported_program.module(),
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        # 触发编译（运行一次推理）
        batch_size = 2
        seq_len = 32
        vocab_size = 1000
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            _ = compiled_model(dummy_input)

        compile_time = time.time() - start_time

        print(f"✓ Compilation completed in {compile_time:.2f} seconds")
        print()

        return compiled_model

    except ImportError:
        print("✗ torch._inductor not available")
        print("  Please ensure you have a recent version of PyTorch with inductor support")
        return None
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return None


def compare_performance(exported_program):
    """
    对比编译前后的性能

    Args:
        exported_program: ExportedProgram 对象
    """
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # 准备测试输入
    batch_size = 8
    seq_len = 32
    vocab_size = 1000

    torch.manual_seed(42)
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    print(f"\nTest configuration:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")

    # 1. 原始 ExportedProgram
    print("\n" + "-" * 60)
    print("1. Original ExportedProgram (Eager Mode)")
    print("-" * 60)

    original_model = exported_program.module()
    original_time = benchmark_inference(
        original_model,
        test_input,
        warmup=5,
        iterations=50,
        name="Original Model"
    )

    # 2. 编译后的模型
    print("\n" + "-" * 60)
    print("2. Compiled with torch._inductor")
    print("-" * 60)

    compiled_model = compile_with_inductor(exported_program)

    if compiled_model is not None:
        compiled_time = benchmark_inference(
            compiled_model,
            test_input,
            warmup=5,
            iterations=50,
            name="Compiled Model"
        )

        # 对比结果
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Original model:  {original_time:.3f} ms")
        print(f"Compiled model:  {compiled_time:.3f} ms")
        print(f"Speedup:         {original_time / compiled_time:.2f}x")
        print(f"Improvement:     {((original_time - compiled_time) / original_time * 100):.1f}%")
        print()

        # 验证输出一致性
        print("=" * 60)
        print("Verifying Output Consistency")
        print("=" * 60)

        with torch.no_grad():
            original_output = original_model(test_input)
            compiled_output = compiled_model(test_input)

        max_diff = torch.abs(original_output - compiled_output).max().item()
        print(f"Max difference: {max_diff:.2e}")

        if torch.allclose(original_output, compiled_output, atol=1e-4, rtol=1e-4):
            print("✓ Outputs match (within tolerance)")
        else:
            print("⚠️  Warning: Outputs differ")
            print(f"Original output: {original_output[:3]}")
            print(f"Compiled output: {compiled_output[:3]}")

    else:
        print("\n⚠️  Skipping performance comparison (compilation failed)")

    print()


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='Inspect ExportedProgram')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize graph with Netron (export to ONNX)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark with torch._inductor')
    parser.add_argument('--tp', action='store_true',
                        help='Enable Tensor Parallel mode')
    args = parser.parse_args()

    # 模型路径
    pt2_path = "../model/transformer_exported.pt2"

    if not os.path.exists(pt2_path):
        print(f"Error: Model file not found: {pt2_path}")
        print("Please run: python3.11 export_transformer_torch_export.py")
        return

    # Tensor Parallel 模式
    if args.tp:
        print("\n" + "=" * 60)
        print("Tensor Parallel Mode")
        print("=" * 60)

        # 初始化分布式
        rank, world_size = init_distributed()

        if rank is None:
            print("\n⚠️  Error: --tp flag requires distributed environment")
            print("Usage: torchrun --nproc_per_node=N script/inspect_exported.py --tp")
            return

        # 多进程模式
        print(f"\nRank {rank}/{world_size} loading model...")

        # 加载模型
        exported_program = torch.export.load(pt2_path)

        # 应用 TP 切分
        exported_program = apply_tensor_parallel(exported_program, rank, world_size)

        # 修改 FX Graph 以适配 TP
        graph_module = modify_graph_for_tp(exported_program, world_size, rank)

        # 准备示例输入用于重新导出
        print(f"[Rank {rank}] 准备重新导出修改后的 graph...")
        batch_size = 4
        seq_len = 32
        vocab_size = 1000
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # 使用 torch.export.export() 重新导出修改后的 graph_module
        print(f"[Rank {rank}] 重新导出为 ExportedProgram...")
        try:
            # 定义动态维度
            dynamic_shapes = {
                "x": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")}
            }

            # 重新导出
            new_exported_program = torch.export.export(
                graph_module,
                (dummy_input,),
                dynamic_shapes=dynamic_shapes
            )
            print(f"[Rank {rank}] ✓ 重新导出成功")

        except Exception as e:
            print(f"[Rank {rank}] ⚠️  动态维度导出失败，尝试静态形状导出...")
            print(f"[Rank {rank}] 错误信息: {e}")
            # 如果动态维度失败，尝试静态形状导出
            new_exported_program = torch.export.export(
                graph_module,
                (dummy_input,)
            )
            print(f"[Rank {rank}] ✓ 静态形状导出成功")

        # 导出切分后的模型
        output_dir = "../model"
        output_path = os.path.join(output_dir, f"transformer_tp_rank{rank}.pt2")

        print(f"[Rank {rank}] 保存切分模型到 {output_path}")
        torch.export.save(new_exported_program, output_path)
        print(f"[Rank {rank}] ✓ 模型保存成功\n")

        # 执行 TP 推理测试
        # 准备测试输入
        torch.manual_seed(42)
        test_input = torch.randint(0, 1000, (4, 32), dtype=torch.long)

        # 执行推理（使用重新导出的 ExportedProgram）
        print(f"[Rank {rank}] 执行推理...")
        inference_module = new_exported_program.module()

        # 检查 ProcessGroup 注册情况
        try:
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            print(f"[Rank {rank}] 默认 ProcessGroup: {default_pg}")
            print(f"[Rank {rank}] Backend: {dist.get_backend()}")
            print(f"[Rank {rank}] Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        except Exception as e:
            print(f"[Rank {rank}] 无法获取 ProcessGroup 信息: {e}")

        with torch.no_grad():
            output = inference_module(test_input)
            
        # 同步
        dist.barrier()

        print(f"[Rank {rank}] ✓ 推理成功！")
        print(f"[Rank {rank}] 输出 shape: {output.shape}\n")

        # 打印结果（all-reduce 已在 Graph 中自动执行）
        if rank == 0:
            print(f"\n[Rank {rank}] 最终输出:")
            print(output)
            print(f"\n[输出统计]")
            print(f"  Mean: {output.mean().item():.6f}")
            print(f"  Std:  {output.std().item():.6f}")
            print(f"  Min:  {output.min().item():.6f}")
            print(f"  Max:  {output.max().item():.6f}")

            # 显示期望输出用于对比
            print(f"\n[期望输出 - 完整模型]")
            print("tensor([[ 0.7054],")
            print("        [ 0.7020],")
            print("        [-0.1992],")
            print("        [ 1.1340]])")

        # 清理
        dist.destroy_process_group()

        return

    # 检查 ExportedProgram
    exported_program = inspect_exported_program(pt2_path)

    # 可视化图结构（可选）
    if args.visualize:
        visualize_graph_with_netron(exported_program)

    # 性能对比（可选）
    if args.benchmark:
        compare_performance(exported_program)

if __name__ == "__main__":
    main()
