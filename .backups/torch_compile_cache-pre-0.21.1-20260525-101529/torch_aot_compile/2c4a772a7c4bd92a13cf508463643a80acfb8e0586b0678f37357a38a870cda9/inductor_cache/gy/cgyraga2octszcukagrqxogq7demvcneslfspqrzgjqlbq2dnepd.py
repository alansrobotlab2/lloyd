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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/sy/csya56nsjiwextf2ojp2wuat7wv6mzlkl7ynysaikyqjcd7obxdc.py
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
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tl.store(out_ptr0 + (x0), tmp3, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = xindex
        tmp4 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp4, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp5 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr2 + (x2), tmp5, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x3 = xindex
        tmp6 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr3 + (x3), tmp6, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 24, 256), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.int32)
    arg_4 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_5 = rand_strided((8192, 272), (272, 1), device='cuda:0', dtype=torch.int32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 50331648, 786432, 655360, 2228224,


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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/oy/coykphlilwvmfilbfc7drl72f3hjhs4oobfl6spnbgpviwmvxvsa.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add => add_39
#   add_1 => add_40
#   float_1 => convert_element_type
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   scaled_fp4_quant_out_1 => scaled_fp4_quant_out
#   zeros_1 => full_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg10_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_40 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %convert_element_type : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_39 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_39), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_3, %arg12_1, True), kwargs = {output: %empty_1, output_scale: %full_default_1})
#   return %buf10,%buf12
triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 335554560}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/r6/cr63rkmgsiqnrhwf734a3b26ve6ozp4iesjj32cztmmu2m6ifbwg.py
# Topologically Sorted Source Nodes: [add_1, add_3, float_2, add_2, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add_1 => add_40
#   add_2 => add_102
#   add_3 => add_103
#   float_2 => convert_element_type_3
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %flashinfer_mm_fp4_2 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4_2]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf30 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf30]
#   %arg20_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_40 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %add_103 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4_2, %add_40), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.float32), kwargs = {})
#   %add_102 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_102), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %flashinfer_mm_fp4), kwargs = {})
#   return %buf30,%convert_element_type_default_1,%buf32
triton_red_fused__to_copy_add_copy__rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 587212800}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 5120.0, tl.float32)
        tmp17 = (tmp8 / tmp16)
        tmp18 = tl.full([1, 1], 1e-06, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.full([1, 1], 1.0, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp21 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp27, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_1 + 5120*x0), tmp11, r0_mask & xmask)
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
arg0_1 = generate_example_value((8192, 24, 256), (6144, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 256))
arg2_1 = generate_example_value((8192, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6144))
buf1 = generate_example_value((8192, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6144))
buf2 = generate_example_value((8192, 96), (96, 1), 'cuda:0', torch.int32, 0, (8192, 96))
buf13 = generate_example_value((8192, 80), (80, 1), 'cuda:0', torch.int32, 0, (8192, 80))
buf22 = generate_example_value((8192, 272), (272, 1), 'cuda:0', torch.int32, 0, (8192, 272))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_0.run(arg0_1, arg2_1, buf1, buf2, buf13, buf22, 50331648, 786432, 655360, 2228224, stream=stream0)
del arg0_1, arg2_1, buf1, buf2, buf13, buf22

stream0 = get_raw_stream(0)
buf9 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg11_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg10_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
buf12 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1.run(buf9, arg11_1, arg10_1, buf12, 8192, 5120, stream=stream0)
del arg10_1, buf12

stream0 = get_raw_stream(0)
buf31 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg20_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
arg9_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_copy__rms_norm_2.run(buf31, buf9, arg11_1, arg20_1, arg9_1, 8192, 5120, stream=stream0)
del buf9, arg11_1, buf31, arg20_1, arg9_1

"""
# AOT ID: ['64_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/sy/csya56nsjiwextf2ojp2wuat7wv6mzlkl7ynysaikyqjcd7obxdc.py
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
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tl.store(out_ptr0 + (x0), tmp3, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x1 = xindex
        tmp4 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp4, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp5 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr2 + (x2), tmp5, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x3 = xindex
        tmp6 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr3 + (x3), tmp6, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 24, 256), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 96), (96, 1), device='cuda:0', dtype=torch.int32)
    arg_4 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_5 = rand_strided((8192, 272), (272, 1), device='cuda:0', dtype=torch.int32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 50331648, 786432, 655360, 2228224,


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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/oy/coykphlilwvmfilbfc7drl72f3hjhs4oobfl6spnbgpviwmvxvsa.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   add => add_39
#   add_1 => add_40
#   float_1 => convert_element_type
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   scaled_fp4_quant_out_1 => scaled_fp4_quant_out
#   zeros_1 => full_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg10_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_40 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %convert_element_type : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_39 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_39), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "i32[128*(((s18 + 127)//128)), 80][80, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 80], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%convert_element_type_default_3, %arg12_1, True), kwargs = {output: %empty_1, output_scale: %full_default_1})
#   return %buf10,%buf12
triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 335554560}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2c4a772a7c4bd92a13cf508463643a80acfb8e0586b0678f37357a38a870cda9/inductor_cache/r6/cr63rkmgsiqnrhwf734a3b26ve6ozp4iesjj32cztmmu2m6ifbwg.py
# Topologically Sorted Source Nodes: [add_1, add_3, float_2, add_2, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add_1 => add_40
#   add_2 => add_102
#   add_3 => add_103
#   float_2 => convert_element_type_3
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %flashinfer_mm_fp4_2 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4_2]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf30 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf30]
#   %arg20_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_40 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %add_103 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4_2, %add_40), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.float32), kwargs = {})
#   %add_102 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_102), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %flashinfer_mm_fp4), kwargs = {})
#   return %buf30,%convert_element_type_default_1,%buf32
triton_red_fused__to_copy_add_copy__rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 587212800}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 5120.0, tl.float32)
        tmp17 = (tmp8 / tmp16)
        tmp18 = tl.full([1, 1], 1e-06, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.full([1, 1], 1.0, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp21 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp27, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_1 + 5120*x0), tmp11, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
        args.clear()
        s72 = arg1_1
        s18 = arg3_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s18, 3072), (3072, 1), torch.uint8)
            buf1 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((128*((127 + s18) // 128), 96), (96, 1), torch.int32)
            buf13 = empty_strided_cuda((128*((127 + s18) // 128), 80), (80, 1), torch.int32)
            buf22 = empty_strided_cuda((128*((127 + s18) // 128), 272), (272, 1), torch.int32)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul_1, zeros, scaled_fp4_quant_out, add_1, float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1, zeros_2], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant, aten.add, aten._to_copy, vllm_ir.rms_norm]
            triton_poi_fused_0_xnumel_0 = 6144*s18
            triton_poi_fused_0_xnumel_1 = 12288*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_2 = 10240*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_3 = 34816*((127 + s18) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg0_1, arg2_1, buf1, buf2, buf13, buf22, triton_poi_fused_0_xnumel_0, triton_poi_fused_0_xnumel_1, triton_poi_fused_0_xnumel_2, triton_poi_fused_0_xnumel_3, stream=stream0)
            del arg0_1
            del arg2_1
            # Topologically Sorted Source Nodes: [view, sigmoid, mul_1, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf1, arg5_1, True, output=buf0, output_scale=buf2)
            del arg5_1
            del buf1
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default], Original ATen: [aten.view]
            buf6 = torch.ops.aten.view.dtype(buf2, torch.float8_e4m3fn)
            buf7 = buf6
            # Topologically Sorted Source Nodes: [t, flashinfer_mm_fp4_default, view_3, t_1], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf8 = torch.ops.vllm.flashinfer_mm_fp4.default(buf0, reinterpret_tensor(arg7_1, (3072, 5120), (1, 3072), 0), aten.view.dtype(buf7, torch.uint8), aten.view.dtype(reinterpret_tensor(arg6_1, (384, 5120), (1, 384), 0), torch.uint8), arg8_1, torch.bfloat16, False, 'cutlass')
            del arg6_1
            del arg7_1
            del arg8_1
            del buf0
            del buf2
            del buf6
            del buf7
            buf9 = buf8
            del buf8
            buf12 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_1.run(buf9, arg11_1, arg10_1, buf12, s18, 5120, stream=stream0)
            del arg10_1
            buf11 = empty_strided_cuda((s18, 2560), (2560, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf12, arg12_1, True, output=buf11, output_scale=buf13)
            del arg12_1
            del buf12
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_1], Original ATen: [aten.view]
            buf17 = torch.ops.aten.view.dtype(buf13, torch.float8_e4m3fn)
            buf18 = buf17
            # Topologically Sorted Source Nodes: [t_2, flashinfer_mm_fp4_default_1, view_7, t_3], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf19 = torch.ops.vllm.flashinfer_mm_fp4.default(buf11, reinterpret_tensor(arg14_1, (2560, 34816), (1, 2560), 0), aten.view.dtype(buf18, torch.uint8), aten.view.dtype(reinterpret_tensor(arg13_1, (320, 34816), (1, 320), 0), torch.uint8), arg15_1, torch.bfloat16, False, 'cutlass')
            del arg13_1
            del arg14_1
            del arg15_1
            del buf11
            del buf13
            del buf17
            del buf18
            buf20 = buf19
            del buf19
            buf21 = empty_strided_cuda((s18, 8704), (8704, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [zeros_2], Original ATen: [aten.zeros]
            torch.ops._C.silu_and_mul_nvfp4_quant.default(buf21, buf22, buf20, arg16_1)
            del arg16_1
            del buf20
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_2], Original ATen: [aten.view]
            buf26 = torch.ops.aten.view.dtype(buf22, torch.float8_e4m3fn)
            buf27 = buf26
            # Topologically Sorted Source Nodes: [t_4, flashinfer_mm_fp4_default_2, view_11, t_5], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf28 = torch.ops.vllm.flashinfer_mm_fp4.default(buf21, reinterpret_tensor(arg18_1, (8704, 5120), (1, 8704), 0), aten.view.dtype(buf27, torch.uint8), aten.view.dtype(reinterpret_tensor(arg17_1, (1088, 5120), (1, 1088), 0), torch.uint8), arg19_1, torch.bfloat16, False, 'cutlass')
            del arg17_1
            del arg18_1
            del arg19_1
            del buf21
            del buf22
            del buf26
            del buf27
            buf29 = buf28
            del buf28
            buf31 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [add_1, add_3, float_2, add_2, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_copy__rms_norm_2.run(buf31, buf9, arg11_1, arg20_1, arg9_1, s18, 5120, stream=stream0)
            del arg11_1
            del arg20_1
            del arg9_1
            del buf9
        return (buf31, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 24, 256), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = 8192
    arg4_1 = 8192
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((5120, 384), (384, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg7_1 = rand_strided((5120, 3072), (3072, 1), device='cuda:0', dtype=torch.uint8)
    arg8_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((34816, 320), (320, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg14_1 = rand_strided((34816, 2560), (2560, 1), device='cuda:0', dtype=torch.uint8)
    arg15_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((5120, 1088), (1088, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg18_1 = rand_strided((5120, 8704), (8704, 1), device='cuda:0', dtype=torch.uint8)
    arg19_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
