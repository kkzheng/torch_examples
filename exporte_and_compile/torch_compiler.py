import torch
import os
import sys
import onnxruntime.training

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TORCH_COMPILE_DEBUG_DIR"] = "./debug"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "graph_breaks"
# os.environ["TORCH_LOGS"] = "guards"
# os.environ["TORCH_LOGS"] = "recompiles"
# os.environ["TORCH_LOGS"] = "dynamic"
# os.environ["TORCH_LOGS"] = "trace_bytecode,trace_source,graph_code"

torch._logging.set_logs(graph_breaks=True)
os.system("rm -rf ./torch_compile_debug")

print(f"supported backend: {torch.compiler.list_backends()}")
# ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

# print(f"compile mode: {torch._inductor.list_mode_options()}")
'''
{'default': {},
 'reduce-overhead': {'triton.cudagraphs': True},
 'max-autotune-no-cudagraphs': {'max_autotune': True,
  'coordinate_descent_tuning': True},
 'max-autotune': {'max_autotune': True,
  'triton.cudagraphs': True,
  'coordinate_descent_tuning': True}}
'''
# print(f"compile options: {torch._inductor.list_options()}")
'''
['TYPE_CHECKING',
 'inplace_padding',
 'can_inplace_pad_graph_input',
 'enable_auto_functionalized_v2',
 'debug',
 'disable_progress',
 'verbose_progress',
 'fx_graph_cache',
 'fx_graph_remote_cache',
 'bundle_triton_into_fx_graph_cache',
 'autotune_local_cache',
 'autotune_remote_cache',
 'bundled_autotune_remote_cache',
 'force_disable_caches',
 'sleep_sec_TESTING_ONLY',
 'custom_op_default_layout_constraint',
 'triton_kernel_default_layout_constraint',
 'cpp_wrapper',
 'online_softmax',
 'dce',
 'static_weight_shapes',
 'size_asserts',
 'nan_asserts',
 'scalar_asserts',
 'pick_loop_orders',
 'inplace_buffers',
 'allow_buffer_reuse',
 'memory_planning',
 'use_fast_math',
 'memory_pool',
 'benchmark_harness',
 'epilogue_fusion',
 'prologue_fusion',
 'epilogue_fusion_first',
 'pattern_matcher',
 'b2b_gemm_pass',
 'post_grad_custom_pre_pass',
 'post_grad_custom_post_pass',
 'joint_custom_pre_pass',
 'joint_custom_post_pass',
 'pre_grad_custom_pass',
 '_pre_fusion_custom_pass',
 'split_cat_fx_passes',
 'efficient_conv_bn_eval_fx_passes',
 'is_predispatch',
 'group_fusion',
 'batch_fusion',
 'pre_grad_fusion_options',
 'post_grad_fusion_options',
 'reorder_for_locality',
 'dynamic_scale_rblock',
 'force_fuse_int_mm_with_mul',
 'use_mixed_mm',
 'fx_passes_numeric_check',
 'mixed_mm_choice',
 'reorder_for_compute_comm_overlap',
 'reorder_for_compute_comm_overlap_passes',
 'reorder_for_peak_memory',
 'estimate_op_runtime',
 'intra_node_bw',
 'inter_node_bw',
 'use_experimental_benchmarker',
 'max_autotune',
 'max_autotune_pointwise',
 'max_autotune_gemm',
 'autotune_num_choices_displayed',
 'graph_partition',
 'force_same_precision',
 'max_autotune_gemm_backends',
 'max_autotune_conv_backends',
 'max_autotune_gemm_search_space',
 'autotune_fallback_to_aten',
 'unbacked_symint_fallback',
 'search_autotune_cache',
 'save_args',
 'autotune_in_subproc',
 'max_autotune_subproc_result_timeout_seconds',
 'max_autotune_subproc_graceful_timeout_seconds',
 'max_autotune_subproc_terminate_timeout_seconds',
 'autotune_multi_device',
 'coordinate_descent_tuning',
 'coordinate_descent_check_all_directions',
 'coordinate_descent_search_radius',
 'autoheuristic_collect',
 'autoheuristic_use',
 'autoheuristic_log_path',
 'layout_opt_default',
 'layout_optimization',
 'force_layout_optimization',
 'keep_output_stride',
 'warn_mix_layout',
 'realize_reads_threshold',
 'realize_opcount_threshold',
 'realize_acc_reads_threshold',
 'fallback_random',
 'implicit_fallbacks',
 'aggressive_fusion',
 'debug_fusion',
 'benchmark_fusion',
 'enabled_metric_tables',
 'loop_ordering_after_fusion',
 'score_fusion_memory_threshold',
 'benchmark_epilogue_fusion',
 'max_epilogue_benchmarked_choices',
 'max_fusion_size',
 'max_pointwise_cat_inputs',
 'force_pointwise_cat',
 'unroll_reductions_threshold',
 'comment_origin',
 'conv_1x1_as_mm',
 'split_reductions',
 'benchmark_kernel',
 'constant_and_index_propagation',
 'always_keep_tensor_constants',
 'assert_indirect_indexing',
 'compute_all_bounds',
 'combo_kernels',
 'benchmark_combo_kernel',
 'combo_kernels_autotune',
 'combo_kernel_allow_mixed_sizes',
 'combo_kernel_foreach_dynamic_shapes',
 'joint_graph_constant_folding',
 'debug_index_asserts',
 'emulate_precision_casts',
 'is_nightly_or_source',
 'developer_warnings',
 'optimize_scatter_upon_const_tensor',
 'add_pre_grad_passes',
 'remove_pre_grad_passes',
 'worker_start_method',
 '_fuse_ddp_communication',
 '_fuse_ddp_bucket_size',
 '_fuse_ddp_communication_passes',
 '_micro_pipeline_tp',
 '_collective.auto_select',
 '_collective.one_shot_all_reduce_threshold_bytes',
 'compile_threads',
 'global_cache_dir',
 'kernel_name_max_ops',
 'shape_padding',
 'comprehensive_padding',
 'pad_channels_last',
 'disable_padding_cpu',
 'padding_alignment_bytes',
 'padding_stride_threshold',
 'pad_outputs',
 'bw_outputs_user_visible',
 'force_shape_pad',
 'permute_fusion',
 'profiler_mark_wrapper_call',
 'generate_intermediate_hooks',
 'debug_ir_traceback',
 '_raise_error_for_testing',
 '_profile_var',
 'profile_bandwidth',
 'profile_bandwidth_regex',
 'profile_bandwidth_output',
 'profile_bandwidth_with_do_bench_using_profiling',
 'disable_cpp_codegen',
 'freezing',
 'freezing_discard_parameters',
 'decompose_mem_bound_mm',
 'assume_aligned_inputs',
 'unsafe_ignore_unsupported_triton_autotune_args',
 'check_stack_no_cycles_TESTING_ONLY',
 'always_complex_memory_overlap_TESTING_ONLY',
 'enable_linear_binary_folding',
 'annotate_training',
 'cpp.threads',
 'cpp.no_redundant_loops',
 'cpp.dynamic_threads',
 'cpp.simdlen',
 'cpp.min_chunk_size',
 'cpp.cxx',
 'cpp.enable_kernel_profile',
 'cpp.weight_prepack',
 'cpp.inject_relu_bug_TESTING_ONLY',
 'cpp.inject_log1p_bug_TESTING_ONLY',
 'cpp.vec_isa_ok',
 'cpp.descriptive_names',
 'cpp.max_horizontal_fusion_size',
 'cpp.fallback_scatter_reduce_sum',
 'cpp.enable_unsafe_math_opt_flag',
 'cpp.enable_floating_point_contract_flag',
 'cpp.enable_tiling_heuristics',
 'cpp.enable_grouped_gemm_template',
 'cpp.gemm_max_k_slices',
 'cpp.gemm_cache_blocking',
 'cpp.gemm_thread_factors',
 'cpp.enable_loop_tail_vec',
 'cpp.enable_concat_linear',
 'triton.cudagraphs',
 'triton.cudagraph_trees',
 'triton.cudagraph_skip_dynamic_graphs',
 'triton.slow_path_cudagraph_asserts',
 'triton.cudagraph_trees_history_recording',
 'triton.cudagraph_support_input_mutation',
 'triton.cudagraph_unexpected_rerecord_limit',
 'triton.cudagraph_dynamic_shape_warn_limit',
 'triton.force_cudagraph_sync',
 'triton.force_cudagraphs_warmup',
 'triton.fast_path_cudagraph_asserts',
 'triton.skip_cudagraph_warmup',
 'triton.debug_sync_graph',
 'triton.debug_sync_kernel',
 'triton.dense_indexing',
 'triton.max_tiles',
 'triton.prefer_nd_tiling',
 'triton.autotune_pointwise',
 'triton.autotune_cublasLt',
 'triton.autotune_at_compile_time',
 'triton.tile_reductions',
 'triton.tiling_prevents_pointwise_fusion',
 'triton.tiling_prevents_reduction_fusion',
 'triton.unique_kernel_names',
 'triton.unique_user_kernel_names',
 'triton.descriptive_names',
 'triton.persistent_reductions',
 'triton.cooperative_reductions',
 'triton.force_cooperative_reductions',
 'triton.multi_kernel',
 'triton.divisible_by_16',
 'triton.min_split_scan_rblock',
 'triton.store_cubin',
 'triton.spill_threshold',
 'triton.use_block_ptr',
 'triton.inject_relu_bug_TESTING_ONLY',
 'triton.codegen_upcast_to_fp32',
 'triton.enable_persistent_tma_matmul',
 'triton.skip_l1_cache',
 'triton.disallow_failing_autotune_kernels_TESTING_ONLY',
 'aot_inductor.output_path',
 'aot_inductor.debug_compile',
 'aot_inductor.compile_wrapper_with_O0',
 'aot_inductor.debug_intermediate_value_printer',
 'aot_inductor.filtered_kernel_names',
 'aot_inductor.serialized_in_spec',
 'aot_inductor.serialized_out_spec',
 'aot_inductor.use_runtime_constant_folding',
 'aot_inductor.force_mmap_weights',
 'aot_inductor.package',
 'aot_inductor.package_cpp_only',
 'aot_inductor.metadata',
 'aot_inductor.raise_error_on_ignored_optimization',
 'aot_inductor.dump_aoti_minifier',
 'aot_inductor.repro_level',
 'aot_inductor.presets',
 'aot_inductor.allow_stack_allocation',
 'aot_inductor.use_minimal_arrayref_interface',
 'aot_inductor.package_constants_in_so',
 'cuda.arch',
 'cuda.version',
 'cuda.compile_opt_level',
 'cuda.enable_cuda_lto',
 'cuda.enable_ptxas_info',
 'cuda.enable_debug_info',
 'cuda.use_fast_math',
 'cuda.cutlass_dir',
 'cuda.cutlass_max_profiling_configs',
 'cuda.cutlass_max_profiling_swizzle_options',
 'cuda.cuda_cxx',
 'cuda.cutlass_backend_min_gemm_size',
 'cuda.generate_test_runner',
 'cuda.cutlass_op_allowlist_regex',
 'cuda.cutlass_op_denylist_regex',
 'cuda.cutlass_instantiation_level',
 'rocm.arch',
 'rocm.ck_supported_arch',
 'rocm.compile_opt_level',
 'rocm.is_debug',
 'rocm.save_temps',
 'rocm.use_fast_math',
 'rocm.flush_denormals',
 'rocm.print_kernel_resource_usage',
 'rocm.rocm_home',
 'rocm.ck_dir',
 'rocm.generate_test_runner',
 'rocm.n_max_profiling_configs',
 'rocm.use_preselected_instances',
 'rocm.kBatch_sweep',
 'rocm.split_k_threshold',
 'cpu_backend',
 'cuda_backend',
 'halide.cpu_target',
 'halide.gpu_target',
 'halide.scheduler_cuda',
 'halide.scheduler_cpu',
 'halide.asserts',
 'halide.debug',
 'halide.scan_kernels',
 'trace.enabled',
 'trace.save_real_tensors',
 'trace.debug_dir',
 'trace.debug_log',
 'trace.info_log',
 'trace.fx_graph',
 'trace.fx_graph_transformed',
 'trace.ir_pre_fusion',
 'trace.ir_post_fusion',
 'trace.output_code',
 'trace.graph_diagram',
 'trace.draw_orig_fx_graph',
 'trace.dot_graph_shape',
 'trace.log_url_for_graph_xform',
 'trace.compile_profile',
 'trace.upload_tar',
 'trace.log_autotuning_results',
 'trace.log_inductor_triton_kernel_to_post_grad_node_info',
 '_save_config_ignore',
 '_cache_config_ignore_prefix',
 'external_matmul',
 'test_configs.force_extern_kernel_in_multi_template',
 'test_configs.max_mm_configs',
 'test_configs.runtime_triton_dtype_assert',
 'test_configs.autotune_choice_name_regex',
 'test_configs.autotune_choice_desc_regex',
 'test_configs.graphsafe_rng_func_ignores_fallback_random']
'''

def fn1():
    backend = "inductor"
    if torch.onnx.is_onnxrt_backend_supported():
        # build onnxruntime-training
        # v1.23.2 
        # ./build.sh --config RelWithDebInfo --enable_training --build_wheel --use_cuda  --parallel  --cuda_home=/usr/local/cuda --cudnn_home=/usr/local/cuda  --allow_running_as_root --skip_tests --cuda_version=12.8
        # onnxruntime 和 onnxruntime-training 不能同时安装
        backend = "onnxrt"
    # backend = "inductor"
    print(f"backend: {backend}")
    
    inp = torch.randn(10000).to("cuda:0")
    def fn1_compile(x):
        a = torch.cos(x)
        b = torch.sin(a)
        return b
    compiled_fn1 = torch.compile(fn1_compile, backend=backend)
    compiled_fn1(inp)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # Downloading: "https://github.com/pytorch/vision/zipball/v0.10.0" to /root/.cache/torch/hub/v0.10.0.zip
    # Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    model = model.to("cuda:0")
    opt_model = torch.compile(model, backend=backend)
    opt_output = opt_model(torch.randn(1, 3, 64, 64).to("cuda:0"))
    print(opt_output.shape)

@torch.compile
def fn_old(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

@torch.compile
def fn_new(x):
    y = x.sum()
    return torch.cond(
        y > 0,
        lambda x: x + y,
        lambda x: x - y,
        (x,),
    )

# ref: https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_true.html
def g(x):
    y = x.sin()
    torch._dynamo.graph_break()
    z = y.cos()
    return z

@torch.compile(fullgraph=True)
def f(xs):
    w = xs.sin()
    return torch._dynamo.nonstrict_trace(g)(w)

if __name__ == "__main__":
    # fn1()
    torch._dynamo.config.capture_scalar_outputs = True
    # print(fn_new(torch.ones(3, 3)))
    
    xs = torch.tensor(1.)
    out = f(xs)
    print(out)

# python -m torch.utils.bottleneck torch_compiler.py

# pip install tlparse
# TORCH_TRACE="/apdcephfs_zwfy/share_303204533/liamjhzhang/tracedir" python torch_compiler.py 
# tlparse /apdcephfs_zwfy/share_303204533/liamjhzhang/tracedir/dedicated_log_torch_trace_d7gcw_i0.log