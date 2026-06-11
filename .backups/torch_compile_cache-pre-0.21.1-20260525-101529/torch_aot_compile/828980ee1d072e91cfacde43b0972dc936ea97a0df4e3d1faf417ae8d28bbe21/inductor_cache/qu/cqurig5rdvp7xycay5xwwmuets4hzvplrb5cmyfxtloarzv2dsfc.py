r"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/og/cogjan6qk4fjmcoaackquvgippbjyvikqwcwzu77ieqq7ogorr6q.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(out_ptr0, out_ptr1, out_ptr2, xnumel_0, xnumel_1, xnumel_2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = xindex
        tmp0 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = xindex
        tmp1 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp2 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr2 + (x2), tmp2, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_2 = rand_strided((8192, 272), (272, 1), device='cuda:0', dtype=torch.int32)
    return arg_0, arg_1, arg_2, 786432, 655360, 2228224,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/nm/cnm2c7a4a27as7li6sna6fhul7cmcc6vnebtsyhua35nv4nih7im.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/hv/chvxlvntgfbgn3z6ojxr7ev4wssssv5h7muqftxl2o7cropm5e34.py
# Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add => add_22
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_2
#   mean => mean
#   mul_1 => mul_32
#   mul_2 => mul_35
#   mul_3 => mul_40
#   pow_1 => pow_1
#   rearrange => view_3
#   reshape => view
#   reshape_1 => view_1
#   rsqrt => rsqrt
#   scaled_fp4_quant_out => scaled_fp4_quant_out_2
#   silu => add_35, div, exp, neg
#   to => convert_element_type_3
#   zeros => full_default
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 48, 128][6144, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "f32[48*s18, 1][1, 48*s18]cuda:0" = PlaceHolder[target=buf0]
#   %arg3_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg2_1 : Tensor "bf16[s18, 48, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_32 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %mul_35 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %convert_element_type_1), kwargs = {})
#   %view_1 : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [%mul_6, 128]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_35), kwargs = {})
#   %mul_40 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %div), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [%arg5_1, 6144]), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 96][96, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 96], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out_2 : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%view_3, %arg7_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf2
triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 404226304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((((x0 + 6144*x1) // 128) % (48*ks0))) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((((x0 + 6144*x1) // 128) % (48*ks0))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + ((x2 % 128)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (128*((((((x0 + 6144*x1) // 128) % (48*ks0))) % 48)) + 16384*(((((((x0 + 6144*x1) // 128) % (48*ks0))) // 48) % ks0)) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 128.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = -tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.full([1], 1.0, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = (tmp13 / tmp17)
    tmp19 = tmp11 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/f6/cf6d2mabddlfm73oefktuogflrkeppm2t7zhxms777mogbmdfpiq.py
# Topologically Sorted Source Nodes: [add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add_1 => add_82
#   add_2 => add_83
#   float_4 => convert_element_type_4
#   rms_norm_default => add_tensor_3, convert_element_type_default_6, convert_element_type_default_7, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
#   scaled_fp4_quant_out_1 => scaled_fp4_quant_out_1
#   zeros_1 => full_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg13_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf11 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf11]
#   %arg12_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_83 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg13_1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.float32), kwargs = {})
#   %add_82 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 1.0), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_83, torch.float32), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_6, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %rsqrt_default_3), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_6, %add_82), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out_1 : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_7, %arg14_1, True), kwargs = {output: %empty_1, output_scale: %full_default_1})
#   return %buf11,%buf13
triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 335554560}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 5120.0, tl.float32)
        tmp13 = (tmp6 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 1.0, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp17 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 5120*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/uu/cuu6crmv3ldxhlaemj4vi7j6bsip24oqnpmfl257wjbi4uhy33fn.py
# Topologically Sorted Source Nodes: [add_2, add_4, float_5, add_3, rms_norm_default_1, zeros_3, scaled_fp4_quant_out_3], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant, aten.copy_]
# Source node to ATen node mapping:
#   add_2 => add_83
#   add_3 => add_145
#   add_4 => add_146
#   float_5 => convert_element_type_7
#   rms_norm_default_1 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   scaled_fp4_quant_out_3 => scaled_fp4_quant_out
#   zeros_3 => full_default_3
# Graph fragment:
#   %flashinfer_mm_fp4_2 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4_2]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg13_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_146 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=add_146]
#   %buf33 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf33]
#   %arg22_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %add_83 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg13_1), kwargs = {})
#   %add_146 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4_2, %add_83), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.float32), kwargs = {})
#   %add_145 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_146, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_145), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.bfloat16), kwargs = {})
#   %full_default_3 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_5, %arg23_1, True), kwargs = {output: %empty_4, output_scale: %full_default_3})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %flashinfer_mm_fp4), kwargs = {})
#   return %add_146,%buf56,%buf33,%buf35
triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 754984960}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp4, r0_mask & xmask)
        tl.store(out_ptr0 + (r0_1 + 5120*x0), tmp1, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 5120.0, tl.float32)
        tmp13 = (tmp8 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 1.0, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp17 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/lw/clwu6pwz6gwfcpgey3m7vethx4zdjp3ecehtxv4qpp5l5i7jx3bf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*i32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = xindex
        tmp0 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = (xindex % 32)
        x2 = xindex // 32
        x3 = xindex
        tmp30 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
        tmp1 = x1
        tmp2 = tl.full([1], 2, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1], 30, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = (((-2) + x1) % 3)
        tmp7 = tl.full([1], 0, tl.int64)
        tmp8 = tmp6 == tmp7
        tmp9 = tmp3 & tmp5
        tmp10 = tmp9 & tmp8
        tmp11 = tl.load(in_ptr0 + (x2 + 2*ks0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tl.broadcast_to(tmp15, [XBLOCK])) & (tl.broadcast_to(tmp15, [XBLOCK]) < 1048576)) | ~(tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp15, [XBLOCK]) < 1048576")
        tmp17 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x1,  3)) + 64*tmp15), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.full([1], 1, tl.int64)
        tmp19 = tmp1 >= tmp18
        tmp20 = (((-1) + x1) % 3)
        tmp21 = tmp20 == tmp7
        tmp22 = tmp19 & tmp21
        tmp23 = tl.load(in_ptr0 + (ks0 + x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp23 < 0
        tmp27 = tl.where(tmp26, tmp25, tmp23)
        tl.device_assert(((0 <= tl.broadcast_to(tmp27, [XBLOCK])) & (tl.broadcast_to(tmp27, [XBLOCK]) < 1048576)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp27, [XBLOCK]) < 1048576")
        tmp29 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x1,  3)) + 64*tmp27), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp32 = tmp30 + tmp31
        tmp33 = tmp30 < 0
        tmp34 = tl.where(tmp33, tmp32, tmp30)
        tl.device_assert(((0 <= tmp34) & (tmp34 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp34 < 1048576")
        tmp36 = tl.load(in_ptr1 + (x1 + 64*tmp34), xmask).to(tl.float32)
        tmp37 = tl.where(tmp22, tmp29, tmp36)
        tmp38 = tl.where(tmp10, tmp17, tmp37)
        tmp39 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x1,  3)) + 64*tmp15), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x1,  3)) + 64*tmp27), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr1 + (32 + x1 + 64*tmp34), xmask).to(tl.float32)
        tmp42 = tl.where(tmp22, tmp40, tmp41)
        tmp43 = tl.where(tmp10, tmp39, tmp42)
        tl.store(out_ptr1 + (x3), tmp38, xmask)
        tl.store(out_ptr2 + (x3), tmp43, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = 8193
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 655360, 262144,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/tb/ctbaokvaufamm4leyc3gxoilr7mg52apewvf2xrnveceyfkeyjwf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + 512*(x0 // 256) + 14336*x1 + ((x0 % 256))), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/ey/cey25domin7kyivzpzycs6gsv3wawz3qgqbg2i3brjkmakmpegvu.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_7 = async_compile.triton('triton_red_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_7', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_7(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 24)
        x1 = xindex // 24
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 14336*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = (xindex % 4)
        x5 = xindex // 4
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (12288 + r0_6 + 256*x4 + 14336*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x7), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 24, 1), (24, 1, 196608), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 4, 1), (4, 1, 32768), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 196608, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/66/c66ppvitbpcr3bfxuqni2se6ixjoowizvtcbnqs5gesu7byv7s5a.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 24)
        x2 = xindex // 1536
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([1], 256.0, tl.float32)
        tmp9 = (tmp7 / tmp8)
        tmp10 = tl.full([1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr0 + (32 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp12
        tmp25 = tl.load(in_ptr2 + (32 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 + tmp16
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp21 - tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp4, tmp32, tmp33)
        tmp35 = tmp0 >= tmp3
        tmp36 = tl.full([1], 64, tl.int64)
        tmp37 = tmp0 < tmp36
        tmp38 = tl.load(in_ptr0 + (32 + 512*x1 + 14336*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (x3), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full([1], 256.0, tl.float32)
        tmp42 = (tmp40 / tmp41)
        tmp43 = tl.full([1], 1e-06, tl.float32)
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp39 * tmp45
        tmp47 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tl.full([1], 1.0, tl.float32)
        tmp50 = tmp48 + tmp49
        tmp51 = tmp46 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56 * tmp45
        tmp58 = tl.load(in_ptr2 + ((-32) + x0), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59 + tmp49
        tmp61 = tmp57 * tmp60
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp62 * tmp63
        tmp65 = tmp54 + tmp64
        tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
        tmp67 = tl.where(tmp35, tmp65, tmp66)
        tmp68 = tl.where(tmp4, tmp34, tmp67)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp68, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 24)
        x6 = xindex // 4608
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 14336*x6), xmask).to(tl.float32)
        tmp71 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp70 = tmp69.to(tl.float32)
        tmp72 = tl.full([1], 256.0, tl.float32)
        tmp73 = (tmp71 / tmp72)
        tmp74 = tl.full([1], 1e-06, tl.float32)
        tmp75 = tmp73 + tmp74
        tmp76 = libdevice.rsqrt(tmp75)
        tmp77 = tmp70 * tmp76
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tl.full([1], 1.0, tl.float32)
        tmp81 = tmp79 + tmp80
        tmp82 = tmp77 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp83, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x10 = xindex // 256
        x9 = ((xindex // 64) % 4)
        x11 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (12288 + 256*x9 + 14336*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp90 = tmp89.to(tl.float32)
        tmp91 = tl.load(in_ptr5 + (x11), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.full([1], 256.0, tl.float32)
        tmp93 = (tmp91 / tmp92)
        tmp94 = tl.full([1], 1e-06, tl.float32)
        tmp95 = tmp93 + tmp94
        tmp96 = libdevice.rsqrt(tmp95)
        tmp97 = tmp90 * tmp96
        tmp98 = tl.load(in_ptr6 + (x8), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tmp98.to(tl.float32)
        tmp100 = tl.full([1], 1.0, tl.float32)
        tmp101 = tmp99 + tmp100
        tmp102 = tmp97 * tmp101
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.load(in_ptr3 + (32*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp105 = tmp103 * tmp104
        tmp106 = tl.load(in_ptr0 + (12320 + 256*x9 + 14336*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107 * tmp96
        tmp109 = tl.load(in_ptr6 + (32 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp110 + tmp100
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tl.load(in_ptr4 + (32*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp115 = tmp113 * tmp114
        tmp116 = tmp105 - tmp115
        tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
        tmp118 = tl.where(tmp88, tmp116, tmp117)
        tmp119 = tmp84 >= tmp87
        tmp120 = tl.full([1], 64, tl.int64)
        tmp121 = tmp84 < tmp120
        tmp122 = tl.load(in_ptr0 + (12320 + 256*x9 + 14336*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp123 = tmp122.to(tl.float32)
        tmp124 = tl.load(in_ptr5 + (x11), tmp119 & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tl.full([1], 256.0, tl.float32)
        tmp126 = (tmp124 / tmp125)
        tmp127 = tl.full([1], 1e-06, tl.float32)
        tmp128 = tmp126 + tmp127
        tmp129 = libdevice.rsqrt(tmp128)
        tmp130 = tmp123 * tmp129
        tmp131 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp132 = tmp131.to(tl.float32)
        tmp133 = tl.full([1], 1.0, tl.float32)
        tmp134 = tmp132 + tmp133
        tmp135 = tmp130 * tmp134
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.load(in_ptr3 + (32*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp138 = tmp136 * tmp137
        tmp139 = tl.load(in_ptr0 + (12288 + 256*x9 + 14336*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tmp140 * tmp129
        tmp142 = tl.load(in_ptr6 + ((-32) + x8), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp143 = tmp142.to(tl.float32)
        tmp144 = tmp143 + tmp133
        tmp145 = tmp141 * tmp144
        tmp146 = tmp145.to(tl.float32)
        tmp147 = tl.load(in_ptr4 + (32*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp148 = tmp146 * tmp147
        tmp149 = tmp138 + tmp148
        tmp150 = tl.full(tmp149.shape, 0.0, tmp149.dtype)
        tmp151 = tl.where(tmp119, tmp149, tmp150)
        tmp152 = tl.where(tmp88, tmp118, tmp151)
        tl.store(out_ptr2 + (x8 + 256*x11), tmp152, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x12 = (xindex % 192)
        x13 = ((xindex // 192) % 4)
        x14 = xindex // 768
        x15 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (12352 + x12 + 256*x13 + 14336*x14), xmask).to(tl.float32)
        tmp155 = tl.load(in_ptr5 + (x15), xmask, eviction_policy='evict_last')
        tmp162 = tl.load(in_ptr6 + (64 + x12), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp154 = tmp153.to(tl.float32)
        tmp156 = tl.full([1], 256.0, tl.float32)
        tmp157 = (tmp155 / tmp156)
        tmp158 = tl.full([1], 1e-06, tl.float32)
        tmp159 = tmp157 + tmp158
        tmp160 = libdevice.rsqrt(tmp159)
        tmp161 = tmp154 * tmp160
        tmp163 = tmp162.to(tl.float32)
        tmp164 = tl.full([1], 1.0, tl.float32)
        tmp165 = tmp163 + tmp164
        tmp166 = tmp161 * tmp165
        tmp167 = tmp166.to(tl.float32)
        tl.store(out_ptr3 + (x12 + 256*x15), tmp167, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 24, 1), (24, 1, 196608), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 4, 1), (4, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 24, 64), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 24, 192), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 4, 64), (1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 4, 192), (1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 12582912, 37748736, 2097152, 6291456,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    stream0 = get_raw_stream(0)
stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 96), (96, 1), 'cuda:0', torch.int32, 0, (8192, 96))
buf14 = generate_example_value((8192, 80), (80, 1), 'cuda:0', torch.int32, 0, (8192, 80))
buf23 = generate_example_value((8192, 272), (272, 1), 'cuda:0', torch.int32, 0, (8192, 272))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_0.run(buf3, buf14, buf23, 786432, 655360, 2228224, stream=stream0)
del buf3, buf14, buf23

stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192, 48, 128), (6144, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 48, 128))
buf0 = generate_example_value((393216, 1), (1, 393216), 'cuda:0', torch.float32, 0, (393216, 1))
with torch.cuda._DeviceGuard(0):
    triton_per_fused_1.run(arg0_1, buf0, 393216, 128, stream=stream0)

stream0 = get_raw_stream(0)
arg3_1 = generate_example_value((128,), (1,), 'cuda:0', torch.bfloat16, 0, (128,))
arg2_1 = generate_example_value((8192, 48, 128), (16384, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 48, 128))
buf2 = generate_example_value((8192, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6144))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2.run(arg0_1, buf0, arg3_1, arg2_1, buf2, 8192, 50331648, stream=stream0)
del arg0_1, buf0, arg3_1, arg2_1, buf2

stream0 = get_raw_stream(0)
buf10 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg13_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg12_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
buf13 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3.run(buf10, arg13_1, arg12_1, buf13, 8192, 5120, stream=stream0)
del arg12_1, buf13

stream0 = get_raw_stream(0)
buf32 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg22_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
arg11_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
buf35 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4.run(buf32, buf10, arg13_1, arg22_1, arg11_1, buf35, 8192, 5120, stream=stream0)
del buf10, arg13_1, buf32, arg22_1, arg11_1, buf35

stream0 = get_raw_stream(0)
arg30_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:0', torch.int64, 0, (3, 8192))
arg29_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (1048576, 64))
buf36 = generate_example_value((8192, 80), (80, 1), 'cuda:0', torch.int32, 0, (8192, 80))
buf46 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
buf47 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_5.run(arg30_1, arg29_1, buf36, buf46, buf47, 8193, 655360, 262144, stream=stream0)
del arg30_1, arg29_1, buf36

stream0 = get_raw_stream(0)
buf43 = generate_example_value((8192, 14336), (14336, 1), 'cuda:0', torch.bfloat16, 0, (8192, 14336))
buf44 = generate_example_value((8192, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6144))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_6.run(buf43, buf44, 50331648, stream=stream0)
del buf44

stream0 = get_raw_stream(0)
buf45 = generate_example_value((8192, 24, 1), (24, 1, 196608), 'cuda:0', torch.float32, 0, (8192, 24, 1))
buf51 = generate_example_value((8192, 4, 1), (4, 1, 32768), 'cuda:0', torch.float32, 0, (8192, 4, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_7.run(buf43, buf45, buf51, 196608, 32768, stream=stream0)

stream0 = get_raw_stream(0)
arg27_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg28_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
buf48 = generate_example_value((8192, 24, 64), (6144, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 64))
buf49 = generate_example_value((8192, 24, 192), (6144, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 192))
buf52 = generate_example_value((8192, 4, 64), (1024, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4, 64))
buf53 = generate_example_value((8192, 4, 192), (1024, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4, 192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_8.run(buf43, buf45, arg27_1, buf46, buf47, buf51, arg28_1, buf48, buf49, buf52, buf53, 12582912, 37748736, 2097152, 6291456, stream=stream0)
del buf46, buf47, buf43, buf45, buf51, arg27_1, arg28_1, buf48, buf49, buf52, buf53

"""
# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/og/cogjan6qk4fjmcoaackquvgippbjyvikqwcwzu77ieqq7ogorr6q.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(out_ptr0, out_ptr1, out_ptr2, xnumel_0, xnumel_1, xnumel_2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = xindex
        tmp0 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = xindex
        tmp1 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp2 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr2 + (x2), tmp2, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_2 = rand_strided((8192, 272), (272, 1), device='cuda:0', dtype=torch.int32)
    return arg_0, arg_1, arg_2, 786432, 655360, 2228224,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/nm/cnm2c7a4a27as7li6sna6fhul7cmcc6vnebtsyhua35nv4nih7im.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/hv/chvxlvntgfbgn3z6ojxr7ev4wssssv5h7muqftxl2o7cropm5e34.py
# Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add => add_22
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_2
#   mean => mean
#   mul_1 => mul_32
#   mul_2 => mul_35
#   mul_3 => mul_40
#   pow_1 => pow_1
#   rearrange => view_3
#   reshape => view
#   reshape_1 => view_1
#   rsqrt => rsqrt
#   scaled_fp4_quant_out => scaled_fp4_quant_out_2
#   silu => add_35, div, exp, neg
#   to => convert_element_type_3
#   zeros => full_default
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 48, 128][6144, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "f32[48*s18, 1][1, 48*s18]cuda:0" = PlaceHolder[target=buf0]
#   %arg3_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg2_1 : Tensor "bf16[s18, 48, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[48*s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_32 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %mul_35 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %convert_element_type_1), kwargs = {})
#   %view_1 : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [%mul_6, 128]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_35), kwargs = {})
#   %mul_40 : Tensor "f32[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %div), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[48*s18, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [%arg5_1, 6144]), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 96][96, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 96], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out_2 : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%view_3, %arg7_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf2
triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 404226304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((((x0 + 6144*x1) // 128) % (48*ks0))) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((((x0 + 6144*x1) // 128) % (48*ks0))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + ((x2 % 128)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (128*((((((x0 + 6144*x1) // 128) % (48*ks0))) % 48)) + 16384*(((((((x0 + 6144*x1) // 128) % (48*ks0))) // 48) % ks0)) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 128.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = -tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.full([1], 1.0, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = (tmp13 / tmp17)
    tmp19 = tmp11 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/f6/cf6d2mabddlfm73oefktuogflrkeppm2t7zhxms777mogbmdfpiq.py
# Topologically Sorted Source Nodes: [add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add_1 => add_82
#   add_2 => add_83
#   float_4 => convert_element_type_4
#   rms_norm_default => add_tensor_3, convert_element_type_default_6, convert_element_type_default_7, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
#   scaled_fp4_quant_out_1 => scaled_fp4_quant_out_1
#   zeros_1 => full_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg13_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf11 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf11]
#   %arg12_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_83 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg13_1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.float32), kwargs = {})
#   %add_82 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 1.0), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_83, torch.float32), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_6, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %rsqrt_default_3), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_6, %add_82), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out_1 : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_7, %arg14_1, True), kwargs = {output: %empty_1, output_scale: %full_default_1})
#   return %buf11,%buf13
triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 335554560}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 5120.0, tl.float32)
        tmp13 = (tmp6 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 1.0, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp17 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 5120*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/uu/cuu6crmv3ldxhlaemj4vi7j6bsip24oqnpmfl257wjbi4uhy33fn.py
# Topologically Sorted Source Nodes: [add_2, add_4, float_5, add_3, rms_norm_default_1, zeros_3, scaled_fp4_quant_out_3], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant, aten.copy_]
# Source node to ATen node mapping:
#   add_2 => add_83
#   add_3 => add_145
#   add_4 => add_146
#   float_5 => convert_element_type_7
#   rms_norm_default_1 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   scaled_fp4_quant_out_3 => scaled_fp4_quant_out
#   zeros_3 => full_default_3
# Graph fragment:
#   %flashinfer_mm_fp4_2 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4_2]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg13_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_146 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=add_146]
#   %buf33 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf33]
#   %arg22_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %add_83 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg13_1), kwargs = {})
#   %add_146 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4_2, %add_83), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.float32), kwargs = {})
#   %add_145 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_146, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_145), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.bfloat16), kwargs = {})
#   %full_default_3 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg6_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_5, %arg23_1, True), kwargs = {output: %empty_4, output_scale: %full_default_3})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %flashinfer_mm_fp4), kwargs = {})
#   return %add_146,%buf56,%buf33,%buf35
triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 754984960}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp4, r0_mask & xmask)
        tl.store(out_ptr0 + (r0_1 + 5120*x0), tmp1, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 5120.0, tl.float32)
        tmp13 = (tmp8 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 1.0, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp17 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/lw/clwu6pwz6gwfcpgey3m7vethx4zdjp3ecehtxv4qpp5l5i7jx3bf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*i32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = xindex
        tmp0 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = (xindex % 32)
        x2 = xindex // 32
        x3 = xindex
        tmp30 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
        tmp1 = x1
        tmp2 = tl.full([1], 2, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1], 30, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = (((-2) + x1) % 3)
        tmp7 = tl.full([1], 0, tl.int64)
        tmp8 = tmp6 == tmp7
        tmp9 = tmp3 & tmp5
        tmp10 = tmp9 & tmp8
        tmp11 = tl.load(in_ptr0 + (x2 + 2*ks0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tl.broadcast_to(tmp15, [XBLOCK])) & (tl.broadcast_to(tmp15, [XBLOCK]) < 1048576)) | ~(tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp15, [XBLOCK]) < 1048576")
        tmp17 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x1,  3)) + 64*tmp15), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.full([1], 1, tl.int64)
        tmp19 = tmp1 >= tmp18
        tmp20 = (((-1) + x1) % 3)
        tmp21 = tmp20 == tmp7
        tmp22 = tmp19 & tmp21
        tmp23 = tl.load(in_ptr0 + (ks0 + x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp23 < 0
        tmp27 = tl.where(tmp26, tmp25, tmp23)
        tl.device_assert(((0 <= tl.broadcast_to(tmp27, [XBLOCK])) & (tl.broadcast_to(tmp27, [XBLOCK]) < 1048576)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp27, [XBLOCK]) < 1048576")
        tmp29 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x1,  3)) + 64*tmp27), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.full([XBLOCK], 1048576, tl.int32)
        tmp32 = tmp30 + tmp31
        tmp33 = tmp30 < 0
        tmp34 = tl.where(tmp33, tmp32, tmp30)
        tl.device_assert(((0 <= tmp34) & (tmp34 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp34 < 1048576")
        tmp36 = tl.load(in_ptr1 + (x1 + 64*tmp34), xmask).to(tl.float32)
        tmp37 = tl.where(tmp22, tmp29, tmp36)
        tmp38 = tl.where(tmp10, tmp17, tmp37)
        tmp39 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x1,  3)) + 64*tmp15), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x1,  3)) + 64*tmp27), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr1 + (32 + x1 + 64*tmp34), xmask).to(tl.float32)
        tmp42 = tl.where(tmp22, tmp40, tmp41)
        tmp43 = tl.where(tmp10, tmp39, tmp42)
        tl.store(out_ptr1 + (x3), tmp38, xmask)
        tl.store(out_ptr2 + (x3), tmp43, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = 8193
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 655360, 262144,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/tb/ctbaokvaufamm4leyc3gxoilr7mg52apewvf2xrnveceyfkeyjwf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + 512*(x0 // 256) + 14336*x1 + ((x0 % 256))), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/ey/cey25domin7kyivzpzycs6gsv3wawz3qgqbg2i3brjkmakmpegvu.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_7 = async_compile.triton('triton_red_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_7', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_7(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 24)
        x1 = xindex // 24
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 14336*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = (xindex % 4)
        x5 = xindex // 4
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (12288 + r0_6 + 256*x4 + 14336*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x7), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 24, 1), (24, 1, 196608), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 4, 1), (4, 1, 32768), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 196608, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/828980ee1d072e91cfacde43b0972dc936ea97a0df4e3d1faf417ae8d28bbe21/inductor_cache/66/c66ppvitbpcr3bfxuqni2se6ixjoowizvtcbnqs5gesu7byv7s5a.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 24)
        x2 = xindex // 1536
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([1], 256.0, tl.float32)
        tmp9 = (tmp7 / tmp8)
        tmp10 = tl.full([1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr0 + (32 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp12
        tmp25 = tl.load(in_ptr2 + (32 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 + tmp16
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp21 - tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp4, tmp32, tmp33)
        tmp35 = tmp0 >= tmp3
        tmp36 = tl.full([1], 64, tl.int64)
        tmp37 = tmp0 < tmp36
        tmp38 = tl.load(in_ptr0 + (32 + 512*x1 + 14336*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tl.load(in_ptr1 + (x3), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full([1], 256.0, tl.float32)
        tmp42 = (tmp40 / tmp41)
        tmp43 = tl.full([1], 1e-06, tl.float32)
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp39 * tmp45
        tmp47 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tl.full([1], 1.0, tl.float32)
        tmp50 = tmp48 + tmp49
        tmp51 = tmp46 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56 * tmp45
        tmp58 = tl.load(in_ptr2 + ((-32) + x0), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59 + tmp49
        tmp61 = tmp57 * tmp60
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp62 * tmp63
        tmp65 = tmp54 + tmp64
        tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
        tmp67 = tl.where(tmp35, tmp65, tmp66)
        tmp68 = tl.where(tmp4, tmp34, tmp67)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp68, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 24)
        x6 = xindex // 4608
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 14336*x6), xmask).to(tl.float32)
        tmp71 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp70 = tmp69.to(tl.float32)
        tmp72 = tl.full([1], 256.0, tl.float32)
        tmp73 = (tmp71 / tmp72)
        tmp74 = tl.full([1], 1e-06, tl.float32)
        tmp75 = tmp73 + tmp74
        tmp76 = libdevice.rsqrt(tmp75)
        tmp77 = tmp70 * tmp76
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tl.full([1], 1.0, tl.float32)
        tmp81 = tmp79 + tmp80
        tmp82 = tmp77 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp83, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x10 = xindex // 256
        x9 = ((xindex // 64) % 4)
        x11 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (12288 + 256*x9 + 14336*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp90 = tmp89.to(tl.float32)
        tmp91 = tl.load(in_ptr5 + (x11), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.full([1], 256.0, tl.float32)
        tmp93 = (tmp91 / tmp92)
        tmp94 = tl.full([1], 1e-06, tl.float32)
        tmp95 = tmp93 + tmp94
        tmp96 = libdevice.rsqrt(tmp95)
        tmp97 = tmp90 * tmp96
        tmp98 = tl.load(in_ptr6 + (x8), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tmp98.to(tl.float32)
        tmp100 = tl.full([1], 1.0, tl.float32)
        tmp101 = tmp99 + tmp100
        tmp102 = tmp97 * tmp101
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.load(in_ptr3 + (32*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp105 = tmp103 * tmp104
        tmp106 = tl.load(in_ptr0 + (12320 + 256*x9 + 14336*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tmp107 * tmp96
        tmp109 = tl.load(in_ptr6 + (32 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp110 + tmp100
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tl.load(in_ptr4 + (32*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp115 = tmp113 * tmp114
        tmp116 = tmp105 - tmp115
        tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
        tmp118 = tl.where(tmp88, tmp116, tmp117)
        tmp119 = tmp84 >= tmp87
        tmp120 = tl.full([1], 64, tl.int64)
        tmp121 = tmp84 < tmp120
        tmp122 = tl.load(in_ptr0 + (12320 + 256*x9 + 14336*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp123 = tmp122.to(tl.float32)
        tmp124 = tl.load(in_ptr5 + (x11), tmp119 & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tl.full([1], 256.0, tl.float32)
        tmp126 = (tmp124 / tmp125)
        tmp127 = tl.full([1], 1e-06, tl.float32)
        tmp128 = tmp126 + tmp127
        tmp129 = libdevice.rsqrt(tmp128)
        tmp130 = tmp123 * tmp129
        tmp131 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp132 = tmp131.to(tl.float32)
        tmp133 = tl.full([1], 1.0, tl.float32)
        tmp134 = tmp132 + tmp133
        tmp135 = tmp130 * tmp134
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.load(in_ptr3 + (32*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp138 = tmp136 * tmp137
        tmp139 = tl.load(in_ptr0 + (12288 + 256*x9 + 14336*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tmp140 * tmp129
        tmp142 = tl.load(in_ptr6 + ((-32) + x8), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp143 = tmp142.to(tl.float32)
        tmp144 = tmp143 + tmp133
        tmp145 = tmp141 * tmp144
        tmp146 = tmp145.to(tl.float32)
        tmp147 = tl.load(in_ptr4 + (32*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp148 = tmp146 * tmp147
        tmp149 = tmp138 + tmp148
        tmp150 = tl.full(tmp149.shape, 0.0, tmp149.dtype)
        tmp151 = tl.where(tmp119, tmp149, tmp150)
        tmp152 = tl.where(tmp88, tmp118, tmp151)
        tl.store(out_ptr2 + (x8 + 256*x11), tmp152, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x12 = (xindex % 192)
        x13 = ((xindex // 192) % 4)
        x14 = xindex // 768
        x15 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (12352 + x12 + 256*x13 + 14336*x14), xmask).to(tl.float32)
        tmp155 = tl.load(in_ptr5 + (x15), xmask, eviction_policy='evict_last')
        tmp162 = tl.load(in_ptr6 + (64 + x12), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp154 = tmp153.to(tl.float32)
        tmp156 = tl.full([1], 256.0, tl.float32)
        tmp157 = (tmp155 / tmp156)
        tmp158 = tl.full([1], 1e-06, tl.float32)
        tmp159 = tmp157 + tmp158
        tmp160 = libdevice.rsqrt(tmp159)
        tmp161 = tmp154 * tmp160
        tmp163 = tmp162.to(tl.float32)
        tmp164 = tl.full([1], 1.0, tl.float32)
        tmp165 = tmp163 + tmp164
        tmp166 = tmp161 * tmp165
        tmp167 = tmp166.to(tl.float32)
        tl.store(out_ptr3 + (x12 + 256*x15), tmp167, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 24, 1), (24, 1, 196608), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 4, 1), (4, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 24, 64), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 24, 192), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 4, 64), (1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 4, 192), (1024, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 12582912, 37748736, 2097152, 6291456,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
        args.clear()
        s72 = arg1_1
        s18 = arg5_1
        s7 = arg31_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((128*((127 + s18) // 128), 96), (96, 1), torch.int32)
            buf14 = empty_strided_cuda((128*((127 + s18) // 128), 80), (80, 1), torch.int32)
            buf23 = empty_strided_cuda((128*((127 + s18) // 128), 272), (272, 1), torch.int32)
            buf0 = empty_strided_cuda((48*s18, 1), (1, 48*s18), torch.float32)
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out, add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1, zeros_2], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant, vllm_ir.rms_norm]
            triton_poi_fused_0_xnumel_0 = 12288*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_1 = 10240*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_2 = 34816*((127 + s18) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(buf3, buf14, buf23, triton_poi_fused_0_xnumel_0, triton_poi_fused_0_xnumel_1, triton_poi_fused_0_xnumel_2, stream=stream0)
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out, add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1, zeros_2], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant, vllm_ir.rms_norm]
            triton_per_fused_1_xnumel = 48*s18
            stream0 = get_raw_stream(0)
            triton_per_fused_1.run(arg0_1, buf0, triton_per_fused_1_xnumel, 128, stream=stream0)
            buf1 = empty_strided_cuda((s18, 3072), (3072, 1), torch.uint8)
            buf2 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant]
            triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2_xnumel = 6144*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2.run(arg0_1, buf0, arg3_1, arg2_1, buf2, s18, triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2_xnumel, stream=stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            del buf0
            # Topologically Sorted Source Nodes: [reshape, float_1, pow_1, mean, add, rsqrt, mul_1, float_2, mul_2, reshape_1, float_3, silu, mul_3, to, rearrange, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._unsafe_view, aten.silu, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf2, arg7_1, True, output=buf1, output_scale=buf3)
            del arg7_1
            del buf2
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default], Original ATen: [aten.view]
            buf7 = torch.ops.aten.view.dtype(buf3, torch.float8_e4m3fn)
            buf8 = buf7
            # Topologically Sorted Source Nodes: [t, flashinfer_mm_fp4_default, view_2, t_1], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf9 = torch.ops.vllm.flashinfer_mm_fp4.default(buf1, reinterpret_tensor(arg9_1, (3072, 5120), (1, 3072), 0), aten.view.dtype(buf8, torch.uint8), aten.view.dtype(reinterpret_tensor(arg8_1, (384, 5120), (1, 384), 0), torch.uint8), arg10_1, torch.bfloat16, False, 'cutlass')
            del arg10_1
            del arg8_1
            del arg9_1
            del buf1
            del buf3
            del buf7
            del buf8
            buf10 = buf9
            del buf9
            buf13 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3.run(buf10, arg13_1, arg12_1, buf13, s18, 5120, stream=stream0)
            del arg12_1
            buf12 = empty_strided_cuda((s18, 2560), (2560, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [add_2, float_4, add_1, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf13, arg14_1, True, output=buf12, output_scale=buf14)
            del arg14_1
            del buf13
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_1], Original ATen: [aten.view]
            buf18 = torch.ops.aten.view.dtype(buf14, torch.float8_e4m3fn)
            buf19 = buf18
            # Topologically Sorted Source Nodes: [t_2, flashinfer_mm_fp4_default_1, view_6, t_3], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf20 = torch.ops.vllm.flashinfer_mm_fp4.default(buf12, reinterpret_tensor(arg16_1, (2560, 34816), (1, 2560), 0), aten.view.dtype(buf19, torch.uint8), aten.view.dtype(reinterpret_tensor(arg15_1, (320, 34816), (1, 320), 0), torch.uint8), arg17_1, torch.bfloat16, False, 'cutlass')
            del arg15_1
            del arg16_1
            del arg17_1
            del buf12
            del buf14
            del buf18
            del buf19
            buf21 = buf20
            del buf20
            buf22 = empty_strided_cuda((s18, 8704), (8704, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [zeros_2], Original ATen: [aten.zeros]
            torch.ops._C.silu_and_mul_nvfp4_quant.default(buf22, buf23, buf21, arg18_1)
            del arg18_1
            del buf21
            buf27 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_2], Original ATen: [aten.view]
            buf28 = torch.ops.aten.view.dtype(buf23, torch.float8_e4m3fn)
            buf29 = buf28
            # Topologically Sorted Source Nodes: [t_4, flashinfer_mm_fp4_default_2, view_10, t_5], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf30 = torch.ops.vllm.flashinfer_mm_fp4.default(buf22, reinterpret_tensor(arg20_1, (8704, 5120), (1, 8704), 0), aten.view.dtype(buf29, torch.uint8), aten.view.dtype(reinterpret_tensor(arg19_1, (1088, 5120), (1, 1088), 0), torch.uint8), arg21_1, torch.bfloat16, False, 'cutlass')
            del arg19_1
            del arg20_1
            del arg21_1
            del buf22
            del buf23
            del buf28
            del buf29
            buf31 = buf30
            del buf30
            buf32 = buf31; del buf31  # reuse
            buf35 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_2, add_4, float_5, add_3, rms_norm_default_1, zeros_3, scaled_fp4_quant_out_3], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_copy__rms_norm_scaled_fp4_quant_zeros_4.run(buf32, buf10, arg13_1, arg22_1, arg11_1, buf35, s18, 5120, stream=stream0)
            del arg11_1
            del arg13_1
            del arg22_1
            del buf10
            buf34 = empty_strided_cuda((s18, 2560), (2560, 1), torch.uint8)
            buf36 = empty_strided_cuda((128*((127 + s18) // 128), 80), (80, 1), torch.int32)
            buf46 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            buf47 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_5, add_3, rms_norm_default_1, zeros_3, scaled_fp4_quant_out_3], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            triton_poi_fused_5_xnumel_0 = 10240*((127 + s18) // 128)
            triton_poi_fused_5_xnumel_1 = 32*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_5.run(arg30_1, arg29_1, buf36, buf46, buf47, s7, triton_poi_fused_5_xnumel_0, triton_poi_fused_5_xnumel_1, stream=stream0)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [float_5, add_3, rms_norm_default_1, zeros_3, scaled_fp4_quant_out_3], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf35, arg23_1, True, output=buf34, output_scale=buf36)
            del arg23_1
            del buf35
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_3], Original ATen: [aten.view]
            buf40 = torch.ops.aten.view.dtype(buf36, torch.float8_e4m3fn)
            buf41 = buf40
            # Topologically Sorted Source Nodes: [t_6, flashinfer_mm_fp4_default_3, view_14, t_7], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf42 = torch.ops.vllm.flashinfer_mm_fp4.default(buf34, reinterpret_tensor(arg25_1, (2560, 14336), (1, 2560), 0), aten.view.dtype(buf41, torch.uint8), aten.view.dtype(reinterpret_tensor(arg24_1, (320, 14336), (1, 320), 0), torch.uint8), arg26_1, torch.bfloat16, False, 'cutlass')
            del arg24_1
            del arg25_1
            del arg26_1
            del buf34
            del buf36
            del buf40
            del buf41
            buf43 = buf42
            del buf42
            buf44 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
            buf45 = empty_strided_cuda((s18, 24, 1), (24, 1, 24*s18), torch.float32)
            buf51 = empty_strided_cuda((s18, 4, 1), (4, 1, 4*s18), torch.float32)
            # Topologically Sorted Source Nodes: [split, view_16, chunk, reshape_8, rms_norm_default_2, view_19, rms_norm_default_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._unsafe_view, vllm_ir.rms_norm]
            triton_poi_fused_6_xnumel = 6144*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_6.run(buf43, buf44, triton_poi_fused_6_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [split, view_16, chunk, reshape_8, rms_norm_default_2, view_19, rms_norm_default_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._unsafe_view, vllm_ir.rms_norm]
            triton_red_fused_7_xnumel_0 = 24*s18
            triton_red_fused_7_xnumel_1 = 4*s18
            stream0 = get_raw_stream(0)
            triton_red_fused_7.run(buf43, buf45, buf51, triton_red_fused_7_xnumel_0, triton_red_fused_7_xnumel_1, stream=stream0)
            buf50 = empty_strided_cuda((s18, 24, 256), (6144, 256, 1), torch.bfloat16)
            buf48 = reinterpret_tensor(buf50, (s18, 24, 64), (6144, 256, 1), 0)  # alias
            buf49 = reinterpret_tensor(buf50, (s18, 24, 192), (6144, 256, 1), 64)  # alias
            buf54 = empty_strided_cuda((s18, 4, 256), (1024, 256, 1), torch.bfloat16)
            buf52 = reinterpret_tensor(buf54, (s18, 4, 64), (1024, 256, 1), 0)  # alias
            buf53 = reinterpret_tensor(buf54, (s18, 4, 192), (1024, 256, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, view_16, chunk, float_6, add_5, rms_norm_default_2, getitem_16, chunk_2, mul_5, mul_6, sub, mul_7, mul_8, add_7, cat, getitem_17, cat_1, view_19, float_7, add_6, rms_norm_default_3, getitem_20, chunk_3, mul_9, mul_10, sub_1, mul_11, mul_12, add_8, cat_2, getitem_21, cat_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._to_copy, aten.add, vllm_ir.rms_norm, aten.slice, aten.unsqueeze, aten.mul, aten.sub, aten.cat]
            triton_poi_fused_8_xnumel_0 = 1536*s18
            triton_poi_fused_8_xnumel_1 = 4608*s18
            triton_poi_fused_8_xnumel_2 = 256*s18
            triton_poi_fused_8_xnumel_3 = 768*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_8.run(buf43, buf45, arg27_1, buf46, buf47, buf51, arg28_1, buf48, buf49, buf52, buf53, triton_poi_fused_8_xnumel_0, triton_poi_fused_8_xnumel_1, triton_poi_fused_8_xnumel_2, triton_poi_fused_8_xnumel_3, stream=stream0)
            del arg27_1
            del arg28_1
            del buf45
            del buf46
            del buf47
            del buf51
            buf55 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
        return (buf54, reinterpret_tensor(buf43, (s18, 4, 256), (14336, 256, 1), 13312), buf50, reinterpret_tensor(buf55, (s18, 24, 256), (6144, 256, 1), 0), buf44, buf27, buf32, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 48, 128), (6144, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 48, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = 8192
    arg5_1 = 8192
    arg6_1 = 8192
    arg7_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((5120, 384), (384, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg9_1 = rand_strided((5120, 3072), (3072, 1), device='cuda:0', dtype=torch.uint8)
    arg10_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((34816, 320), (320, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg16_1 = rand_strided((34816, 2560), (2560, 1), device='cuda:0', dtype=torch.uint8)
    arg17_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((5120, 1088), (1088, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg20_1 = rand_strided((5120, 8704), (8704, 1), device='cuda:0', dtype=torch.uint8)
    arg21_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((14336, 320), (320, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg25_1 = rand_strided((14336, 2560), (2560, 1), device='cuda:0', dtype=torch.uint8)
    arg26_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg31_1 = 8193
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
