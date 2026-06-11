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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/uy/cuyndbyxlcj6vko32uzt3buvfk6qcylv5yefrpw3pu2ep3l2tjai.py
# Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_1, marlin_gemm_1], Original ATen: [vllm_ir.rms_norm, aten.add, _C.marlin_gemm]
# Source node to ATen node mapping:
#   add => add_15
#   marlin_gemm_1 => marlin_gemm_1
#   rms_norm_default => add_tensor_5, convert_element_type_default_10, convert_element_type_default_11, mean_dim_5, mul_tensor_10, mul_tensor_11, pow_tensor_scalar_5, rsqrt_default_5
#   rms_norm_default_1 => add_tensor_4, convert_element_type_default_8, convert_element_type_default_9, mean_dim_4, mul_tensor_8, mul_tensor_9, pow_tensor_scalar_4, rsqrt_default_4
# Graph fragment:
#   %marlin_gemm : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf3 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf3]
#   %arg8_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_default_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_5 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_10, 2), kwargs = {})
#   %mean_dim_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_5, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_5, 1e-06), kwargs = {})
#   %rsqrt_default_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_10, %rsqrt_default_5), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_10, torch.bfloat16), kwargs = {})
#   %mul_tensor_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg6_1), kwargs = {})
#   %add_15 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_11, %arg7_1), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %pow_tensor_scalar_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_8, 2), kwargs = {})
#   %mean_dim_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_4, [-1], True), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_4, 1e-06), kwargs = {})
#   %rsqrt_default_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_tensor_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, %rsqrt_default_4), kwargs = {})
#   %convert_element_type_default_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_8, torch.bfloat16), kwargs = {})
#   %mul_tensor_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_9, %arg8_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "bf16[s72, 20480][20480, 1]cuda:0"[num_users=2] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_9, None, %arg9_1, None, %arg10_1, None, %arg11_1, None, None, None, %arg12_1, 562949953487106, %arg1_1, 20480, 2560, True, False, True, False), kwargs = {})
#   return %buf2,%buf3,%buf4
triton_red_fused_add_marlin_gemm_rms_norm_0 = async_compile.triton('triton_red_fused_add_marlin_gemm_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_marlin_gemm_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 65536, 'r0_': 167782400}}
)
@triton.jit
def triton_red_fused_add_marlin_gemm_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 2560.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.full([1, 1], 2560.0, tl.float32)
        tmp27 = (tmp4 / tmp26)
        tmp28 = tl.full([1, 1], 1e-06, tl.float32)
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = (tmp22 / tmp26)
        tmp39 = tmp38 + tmp28
        tmp40 = libdevice.rsqrt(tmp39)
        tmp41 = tmp37 * tmp40
        tmp42 = tmp41.to(tl.float32)
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp44, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/r4/cr46b6ry43poqe2aeruvi37snmhdfelpwrmqufjbojcztm3dawzr.py
# Topologically Sorted Source Nodes: [getitem, gelu, getitem_1, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.gelu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   gelu => add_34, add_35, convert_element_type, convert_element_type_1, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, tanh
#   getitem => slice_1
#   getitem_1 => slice_2
#   marlin_gemm_2 => marlin_gemm_2
#   mul => mul_44
# Graph fragment:
#   %marlin_gemm_1 : Tensor "bf16[s72, 20480][20480, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "bf16[s72, 10240][20480, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 10240), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %mul_38 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_34 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %convert_element_type), kwargs = {})
#   %mul_35 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %convert_element_type), kwargs = {})
#   %mul_36 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, 0.044715), kwargs = {})
#   %add_34 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %mul_36), kwargs = {})
#   %mul_37 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_37,), kwargs = {})
#   %add_35 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_39 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %add_35), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 10240][20480, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 10240, 9223372036854775807), kwargs = {})
#   %mul_44 : Tensor "bf16[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   %marlin_gemm_2 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_44, None, %arg13_1, None, %arg14_1, None, %arg15_1, None, None, None, %arg16_1, 562949953487106, %arg1_1, 2560, 10240, True, False, True, False), kwargs = {})
#   return %buf7
triton_poi_fused_gelu_marlin_gemm_mul_slice_1 = async_compile.triton('triton_poi_fused_gelu_marlin_gemm_mul_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_marlin_gemm_mul_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 671088640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_marlin_gemm_mul_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10240)
    x1 = xindex // 10240
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 20480*x1), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (10240 + x0 + 20480*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/5q/c5q4xamzhgg3hggnx7v42b7ylyi6zt5gp3dbf3yjvkdwih2ykusj.py
# Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_2, add_1], Original ATen: [vllm_ir.rms_norm, aten.add]
# Source node to ATen node mapping:
#   add => add_15
#   add_1 => add_57
#   rms_norm_default => add_tensor_5, convert_element_type_default_10, convert_element_type_default_11, mean_dim_5, mul_tensor_10, mul_tensor_11, pow_tensor_scalar_5, rsqrt_default_5
#   rms_norm_default_2 => add_tensor_3, convert_element_type_default_6, convert_element_type_default_7, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
# Graph fragment:
#   %marlin_gemm_2 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %buf10 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf10]
#   %arg17_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %marlin_gemm : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_default_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_5 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_10, 2), kwargs = {})
#   %mean_dim_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_5, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_5, 1e-06), kwargs = {})
#   %rsqrt_default_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_10, %rsqrt_default_5), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_10, torch.bfloat16), kwargs = {})
#   %mul_tensor_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg6_1), kwargs = {})
#   %add_15 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_11, %arg7_1), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_6, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %rsqrt_default_3), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_6, torch.bfloat16), kwargs = {})
#   %mul_tensor_7 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg17_1), kwargs = {})
#   %add_57 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_7, %add_15), kwargs = {})
#   return %buf10,%add_57
triton_red_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_add_rms_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_rms_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 32768, 'r0_': 209725440}}
)
@triton.jit
def triton_red_fused_add_rms_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 2560.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp20 = (tmp19 / tmp8)
        tmp21 = tmp20 + tmp10
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp18 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp16 + tmp28
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/fk/cfkqi4x7lsbwshubbvn3dpmf4mtslwl7q7ozq3c6j7apkjrixhuh.py
# Topologically Sorted Source Nodes: [gelu_1, mul_1, marlin_gemm_4], Original ATen: [aten.gelu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   gelu_1 => add_70, add_71, convert_element_type_2, convert_element_type_3, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, tanh_1
#   marlin_gemm_4 => marlin_gemm_4
#   mul_1 => mul_83
# Graph fragment:
#   %marlin_gemm_3 : Tensor "bf16[s72, 256][256, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg22_1 : Tensor "bf16[s72, 256][10752, 1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_2 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_3, torch.float32), kwargs = {})
#   %mul_79 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_75 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_2), kwargs = {})
#   %mul_76 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %convert_element_type_2), kwargs = {})
#   %mul_77 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, 0.044715), kwargs = {})
#   %add_70 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %mul_77), kwargs = {})
#   %mul_78 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, 0.7978845608028654), kwargs = {})
#   %tanh_1 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_78,), kwargs = {})
#   %add_71 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_1, 1), kwargs = {})
#   %mul_80 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %add_71), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_80, torch.bfloat16), kwargs = {})
#   %mul_83 : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %arg22_1), kwargs = {})
#   %marlin_gemm_4 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_83, None, %arg23_1, None, %arg24_1, None, %arg25_1, None, None, None, %arg26_1, 562949953487106, %arg1_1, 2560, 256, True, False, True, False), kwargs = {})
#   return %buf14
triton_poi_fused_gelu_marlin_gemm_mul_3 = async_compile.triton('triton_poi_fused_gelu_marlin_gemm_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_marlin_gemm_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_marlin_gemm_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (x0 + 10752*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/zw/czwar5aggrjyuk4exursig6i7fh42pj42cjsy4acpmbzkss2w7gd.py
# Topologically Sorted Source Nodes: [rms_norm_default_3, add_2, mul_2, rms_norm_default_4, marlin_gemm_5], Original ATen: [vllm_ir.rms_norm, aten.add, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   add_2 => add_90
#   marlin_gemm_5 => marlin_gemm_5
#   mul_2 => mul_102
#   rms_norm_default_3 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   rms_norm_default_4 => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %marlin_gemm_4 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm_4]
#   %add_57 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=add_57]
#   %buf17 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf17]
#   %arg27_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "bf16[1][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %mul_102 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=mul_102]
#   %buf19 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf19]
#   %arg29_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_4, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg27_1), kwargs = {})
#   %add_90 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %mul_tensor_5), kwargs = {})
#   %mul_102 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_90, %arg28_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_102, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.bfloat16), kwargs = {})
#   %mul_tensor_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg29_1), kwargs = {})
#   %marlin_gemm_5 : Tensor "bf16[s72, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_3, None, %arg30_1, None, %arg31_1, None, %arg32_1, None, None, None, %arg33_1, 562949953487106, %arg1_1, 3072, 2560, True, False, True, False), kwargs = {})
#   return %buf17,%mul_102,%buf19,%buf20
triton_red_fused_add_marlin_gemm_mul_rms_norm_4 = async_compile.triton('triton_red_fused_add_marlin_gemm_mul_rms_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_marlin_gemm_mul_rms_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 251668480}}
)
@triton.jit
def triton_red_fused_add_marlin_gemm_mul_rms_norm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [1, 1])
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full([1, 1], 2560.0, tl.float32)
        tmp10 = (tmp4 / tmp9)
        tmp11 = tl.full([1, 1], 1e-06, tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.rsqrt(tmp12)
        tmp14 = tmp8 * tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 * tmp16
        tmp18 = tmp6 + tmp17
        tmp21 = tmp18 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp21, r0_mask & xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.full([1, 1], 2560.0, tl.float32)
        tmp30 = (tmp25 / tmp29)
        tmp31 = tl.full([1, 1], 1e-06, tl.float32)
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp35 * tmp36
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp37, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/6h/c6h2q3g4hjjc3f3puypy77xfjlvdttujfkp74lpilq2dvkyrw3m6.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten, rms_norm_default_5, chunk_1, unsqueeze, mul_3, unsqueeze_1, mul_4, sub, mul_5, mul_6, add_3, cat, reshape_12, to_4, reciprocal, mul_11, clamp, to_6], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat, aten._to_copy, aten.reciprocal, aten.clamp]
# Source node to ATen node mapping:
#   add_3 => add_184
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   clamp => clamp_max, clamp_min
#   index_select => index
#   mul_11 => mul_223
#   mul_3 => mul_156
#   mul_4 => mul_159
#   mul_5 => mul_164
#   mul_6 => mul_167
#   reciprocal => reciprocal
#   reshape_12 => view_16
#   rms_norm_default_5 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   split => split_with_sizes
#   sub => sub_52
#   to_4 => convert_element_type_4
#   to_6 => convert_element_type_5
#   unflatten => view_13
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %marlin_gemm_5 : Tensor "bf16[s72, 3072][3072, 1]cuda:0" = PlaceHolder[target=marlin_gemm_5]
#   %buf23 : Tensor "f32[s72, 8, 1][8, 1, 8*s72]cuda:0" = PlaceHolder[target=buf23]
#   %arg34_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "bf16[131072, 256][256, 1]cuda:0" = PlaceHolder[target=arg36_1]
#   %cat : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=cat]
#   %arg37_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=arg37_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_5, [2048, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg36_1, [%arg35_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 128, -1), kwargs = {})
#   %view_13 : Tensor "bf16[s72, 8, 256][3072, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg1_1, 8, 256]), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg34_1), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_1, 128, -1), kwargs = {})
#   %unsqueeze : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_156 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_159 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_52 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_156, %mul_159), kwargs = {})
#   %mul_164 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_167 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_184 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %mul_167), kwargs = {})
#   %cat : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_52, %add_184], -1), kwargs = {})
#   %view_16 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [%arg1_1, 2048]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %reciprocal : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%arg37_1,), kwargs = {})
#   %mul_223 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %reciprocal), kwargs = {})
#   %clamp_min : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_223, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f8e4m3fn[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   return %buf23,%cat,%convert_element_type_5
triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5 = async_compile.triton('triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 134219264}}
)
@triton.jit
def triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 3072*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp76 = tl.load(in_ptr4 + (0))
    tmp77 = tl.broadcast_to(tmp76, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp6 = r0_2
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tmp6 >= tmp7
        tmp9 = tl.full([1, 1], 128, tl.int64)
        tmp10 = tmp6 < tmp9
        tmp11 = tl.load(in_ptr0 + (256*x0 + 3072*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 256.0, tl.float32)
        tmp14 = (tmp4 / tmp13)
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.full([1, 1], 131072, tl.int32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp22 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp22)
        tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 131072)) | ~(r0_mask & tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 131072")
        tmp28 = tl.load(in_ptr3 + (tl.broadcast_to(256*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tmp21 * tmp28
        tmp30 = tl.load(in_ptr0 + (128 + 256*x0 + 3072*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 * tmp17
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.load(in_ptr1 + (tl.broadcast_to(128 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tmp33 * tmp34
        tmp36 = tl.load(in_ptr3 + (tl.broadcast_to(128 + 256*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tmp35 * tmp36
        tmp38 = tmp29 - tmp37
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp10, tmp38, tmp39)
        tmp41 = tmp6 >= tmp9
        tmp42 = tl.full([1, 1], 256, tl.int64)
        tmp43 = tmp6 < tmp42
        tmp44 = tl.load(in_ptr0 + (128 + 256*x0 + 3072*x1 + ((-128) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tl.full([1, 1], 256.0, tl.float32)
        tmp47 = (tmp4 / tmp46)
        tmp48 = tl.full([1, 1], 1e-06, tl.float32)
        tmp49 = tmp47 + tmp48
        tmp50 = libdevice.rsqrt(tmp49)
        tmp51 = tmp45 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr1 + (tl.broadcast_to(128 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.full([1, 1], 131072, tl.int32)
        tmp57 = tmp55 + tmp56
        tmp58 = tmp55 < 0
        tmp59 = tl.where(tmp58, tmp57, tmp55)
        tl.device_assert(((0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 131072)) | ~(r0_mask & tmp41 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 131072")
        tmp61 = tl.load(in_ptr3 + (tl.broadcast_to(256*tmp59 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp62 = tmp54 * tmp61
        tmp63 = tl.load(in_ptr0 + (256*x0 + 3072*x1 + ((-128) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp64 * tmp50
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (tl.broadcast_to((-128) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp68 = tmp66 * tmp67
        tmp69 = tl.load(in_ptr3 + (tl.broadcast_to(128 + 256*tmp59 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp70 = tmp68 * tmp69
        tmp71 = tmp62 + tmp70
        tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
        tmp73 = tl.where(tmp41, tmp71, tmp72)
        tmp74 = tl.where(tmp10, tmp40, tmp73)
        tmp75 = tmp74.to(tl.float32)
        tmp78 = tl.full([1, 1], 1, tl.int32)
        tmp79 = (tmp78 / tmp77)
        tmp80 = tmp75 * tmp79
        tmp81 = tl.full([1, 1], -448.0, tl.float32)
        tmp82 = triton_helpers.maximum(tmp80, tmp81)
        tmp83 = tl.full([1, 1], 448.0, tl.float32)
        tmp84 = triton_helpers.minimum(tmp82, tmp83)
        tmp85 = tmp84.to(tl.float8e4nv)
        tl.store(out_ptr2 + (r0_2 + 256*x3), tmp85, r0_mask & xmask)
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
buf1 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
arg6_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
arg7_1 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
arg8_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
buf2 = generate_example_value((8192, 1), (1, 8192), 'cuda:0', torch.float32, 0, (8192, 1))
buf4 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_add_marlin_gemm_rms_norm_0.run(buf1, arg6_1, arg7_1, arg8_1, buf2, buf4, 8192, 2560, stream=stream0)
del arg8_1, buf4

stream0 = get_raw_stream(0)
buf6 = generate_example_value((8192, 20480), (20480, 1), 'cuda:0', torch.bfloat16, 0, (8192, 20480))
buf7 = generate_example_value((8192, 10240), (10240, 1), 'cuda:0', torch.bfloat16, 0, (8192, 10240))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_gelu_marlin_gemm_mul_slice_1.run(buf6, buf7, 83886080, stream=stream0)
del buf6, buf7

stream0 = get_raw_stream(0)
buf11 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
arg17_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_add_rms_norm_2.run(buf11, arg17_1, buf1, buf2, arg6_1, arg7_1, 8192, 2560, stream=stream0)
del buf1, arg6_1, arg7_1, buf2, buf11, arg17_1

stream0 = get_raw_stream(0)
buf14 = generate_example_value((8192, 256), (256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 256))
arg22_1 = generate_example_value((8192, 256), (10752, 1), 'cuda:0', torch.bfloat16, 0, (8192, 256))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_gelu_marlin_gemm_mul_3.run(buf14, arg22_1, 2097152, stream=stream0)
del buf14, arg22_1

stream0 = get_raw_stream(0)
buf18 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf16 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
arg27_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
arg28_1 = generate_example_value((1,), (1,), 'cuda:0', torch.bfloat16, 0, (1,))
arg29_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
buf20 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_add_marlin_gemm_mul_rms_norm_4.run(buf18, buf16, arg27_1, arg28_1, arg29_1, buf20, 8192, 2560, stream=stream0)
del buf18, buf16, arg27_1, arg28_1, arg29_1, buf20

stream0 = get_raw_stream(0)
buf22 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg34_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg35_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg36_1 = generate_example_value((131072, 256), (256, 1), 'cuda:0', torch.bfloat16, 0, (131072, 256))
arg37_1 = generate_example_value((), (), 'cuda:0', torch.float32, 0, ())
buf25 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.float8_e4m3fn, 0, (8192, 2048))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5.run(buf22, arg34_1, arg35_1, arg36_1, arg37_1, buf25, 65536, 256, stream=stream0)
del buf22, arg34_1, arg35_1, arg36_1, arg37_1, buf25

"""
# AOT ID: ['24_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/uy/cuyndbyxlcj6vko32uzt3buvfk6qcylv5yefrpw3pu2ep3l2tjai.py
# Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_1, marlin_gemm_1], Original ATen: [vllm_ir.rms_norm, aten.add, _C.marlin_gemm]
# Source node to ATen node mapping:
#   add => add_15
#   marlin_gemm_1 => marlin_gemm_1
#   rms_norm_default => add_tensor_5, convert_element_type_default_10, convert_element_type_default_11, mean_dim_5, mul_tensor_10, mul_tensor_11, pow_tensor_scalar_5, rsqrt_default_5
#   rms_norm_default_1 => add_tensor_4, convert_element_type_default_8, convert_element_type_default_9, mean_dim_4, mul_tensor_8, mul_tensor_9, pow_tensor_scalar_4, rsqrt_default_4
# Graph fragment:
#   %marlin_gemm : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf3 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf3]
#   %arg8_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_default_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_5 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_10, 2), kwargs = {})
#   %mean_dim_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_5, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_5, 1e-06), kwargs = {})
#   %rsqrt_default_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_10, %rsqrt_default_5), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_10, torch.bfloat16), kwargs = {})
#   %mul_tensor_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg6_1), kwargs = {})
#   %add_15 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_11, %arg7_1), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %pow_tensor_scalar_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_8, 2), kwargs = {})
#   %mean_dim_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_4, [-1], True), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_4, 1e-06), kwargs = {})
#   %rsqrt_default_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_tensor_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, %rsqrt_default_4), kwargs = {})
#   %convert_element_type_default_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_8, torch.bfloat16), kwargs = {})
#   %mul_tensor_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_9, %arg8_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "bf16[s72, 20480][20480, 1]cuda:0"[num_users=2] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_9, None, %arg9_1, None, %arg10_1, None, %arg11_1, None, None, None, %arg12_1, 562949953487106, %arg1_1, 20480, 2560, True, False, True, False), kwargs = {})
#   return %buf2,%buf3,%buf4
triton_red_fused_add_marlin_gemm_rms_norm_0 = async_compile.triton('triton_red_fused_add_marlin_gemm_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_marlin_gemm_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 65536, 'r0_': 167782400}}
)
@triton.jit
def triton_red_fused_add_marlin_gemm_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 2560.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.full([1, 1], 2560.0, tl.float32)
        tmp27 = (tmp4 / tmp26)
        tmp28 = tl.full([1, 1], 1e-06, tl.float32)
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = (tmp22 / tmp26)
        tmp39 = tmp38 + tmp28
        tmp40 = libdevice.rsqrt(tmp39)
        tmp41 = tmp37 * tmp40
        tmp42 = tmp41.to(tl.float32)
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp44, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/r4/cr46b6ry43poqe2aeruvi37snmhdfelpwrmqufjbojcztm3dawzr.py
# Topologically Sorted Source Nodes: [getitem, gelu, getitem_1, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.gelu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   gelu => add_34, add_35, convert_element_type, convert_element_type_1, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, tanh
#   getitem => slice_1
#   getitem_1 => slice_2
#   marlin_gemm_2 => marlin_gemm_2
#   mul => mul_44
# Graph fragment:
#   %marlin_gemm_1 : Tensor "bf16[s72, 20480][20480, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "bf16[s72, 10240][20480, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 10240), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %mul_38 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_34 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %convert_element_type), kwargs = {})
#   %mul_35 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %convert_element_type), kwargs = {})
#   %mul_36 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, 0.044715), kwargs = {})
#   %add_34 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %mul_36), kwargs = {})
#   %mul_37 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_37,), kwargs = {})
#   %add_35 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_39 : Tensor "f32[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %add_35), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 10240][20480, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 10240, 9223372036854775807), kwargs = {})
#   %mul_44 : Tensor "bf16[s72, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   %marlin_gemm_2 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_44, None, %arg13_1, None, %arg14_1, None, %arg15_1, None, None, None, %arg16_1, 562949953487106, %arg1_1, 2560, 10240, True, False, True, False), kwargs = {})
#   return %buf7
triton_poi_fused_gelu_marlin_gemm_mul_slice_1 = async_compile.triton('triton_poi_fused_gelu_marlin_gemm_mul_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_marlin_gemm_mul_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 671088640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_marlin_gemm_mul_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10240)
    x1 = xindex // 10240
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 20480*x1), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (10240 + x0 + 20480*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/5q/c5q4xamzhgg3hggnx7v42b7ylyi6zt5gp3dbf3yjvkdwih2ykusj.py
# Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_2, add_1], Original ATen: [vllm_ir.rms_norm, aten.add]
# Source node to ATen node mapping:
#   add => add_15
#   add_1 => add_57
#   rms_norm_default => add_tensor_5, convert_element_type_default_10, convert_element_type_default_11, mean_dim_5, mul_tensor_10, mul_tensor_11, pow_tensor_scalar_5, rsqrt_default_5
#   rms_norm_default_2 => add_tensor_3, convert_element_type_default_6, convert_element_type_default_7, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
# Graph fragment:
#   %marlin_gemm_2 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %buf10 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf10]
#   %arg17_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %marlin_gemm : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_default_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_5 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_10, 2), kwargs = {})
#   %mean_dim_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_5, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_5, 1e-06), kwargs = {})
#   %rsqrt_default_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_10, %rsqrt_default_5), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_10, torch.bfloat16), kwargs = {})
#   %mul_tensor_11 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg6_1), kwargs = {})
#   %add_15 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_11, %arg7_1), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_6, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %rsqrt_default_3), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_6, torch.bfloat16), kwargs = {})
#   %mul_tensor_7 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg17_1), kwargs = {})
#   %add_57 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor_7, %add_15), kwargs = {})
#   return %buf10,%add_57
triton_red_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_add_rms_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_rms_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 32768, 'r0_': 209725440}}
)
@triton.jit
def triton_red_fused_add_rms_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 2560.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp20 = (tmp19 / tmp8)
        tmp21 = tmp20 + tmp10
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp18 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp16 + tmp28
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/fk/cfkqi4x7lsbwshubbvn3dpmf4mtslwl7q7ozq3c6j7apkjrixhuh.py
# Topologically Sorted Source Nodes: [gelu_1, mul_1, marlin_gemm_4], Original ATen: [aten.gelu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   gelu_1 => add_70, add_71, convert_element_type_2, convert_element_type_3, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, tanh_1
#   marlin_gemm_4 => marlin_gemm_4
#   mul_1 => mul_83
# Graph fragment:
#   %marlin_gemm_3 : Tensor "bf16[s72, 256][256, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg22_1 : Tensor "bf16[s72, 256][10752, 1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_2 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_3, torch.float32), kwargs = {})
#   %mul_79 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_75 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_2), kwargs = {})
#   %mul_76 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %convert_element_type_2), kwargs = {})
#   %mul_77 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, 0.044715), kwargs = {})
#   %add_70 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %mul_77), kwargs = {})
#   %mul_78 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, 0.7978845608028654), kwargs = {})
#   %tanh_1 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_78,), kwargs = {})
#   %add_71 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_1, 1), kwargs = {})
#   %mul_80 : Tensor "f32[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %add_71), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_80, torch.bfloat16), kwargs = {})
#   %mul_83 : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %arg22_1), kwargs = {})
#   %marlin_gemm_4 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_83, None, %arg23_1, None, %arg24_1, None, %arg25_1, None, None, None, %arg26_1, 562949953487106, %arg1_1, 2560, 256, True, False, True, False), kwargs = {})
#   return %buf14
triton_poi_fused_gelu_marlin_gemm_mul_3 = async_compile.triton('triton_poi_fused_gelu_marlin_gemm_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_marlin_gemm_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_marlin_gemm_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (x0 + 10752*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/zw/czwar5aggrjyuk4exursig6i7fh42pj42cjsy4acpmbzkss2w7gd.py
# Topologically Sorted Source Nodes: [rms_norm_default_3, add_2, mul_2, rms_norm_default_4, marlin_gemm_5], Original ATen: [vllm_ir.rms_norm, aten.add, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   add_2 => add_90
#   marlin_gemm_5 => marlin_gemm_5
#   mul_2 => mul_102
#   rms_norm_default_3 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
#   rms_norm_default_4 => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %marlin_gemm_4 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=marlin_gemm_4]
#   %add_57 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=add_57]
#   %buf17 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf17]
#   %arg27_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "bf16[1][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %mul_102 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=mul_102]
#   %buf19 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf19]
#   %arg29_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_4, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg27_1), kwargs = {})
#   %add_90 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %mul_tensor_5), kwargs = {})
#   %mul_102 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_90, %arg28_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_102, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.bfloat16), kwargs = {})
#   %mul_tensor_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg29_1), kwargs = {})
#   %marlin_gemm_5 : Tensor "bf16[s72, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_3, None, %arg30_1, None, %arg31_1, None, %arg32_1, None, None, None, %arg33_1, 562949953487106, %arg1_1, 3072, 2560, True, False, True, False), kwargs = {})
#   return %buf17,%mul_102,%buf19,%buf20
triton_red_fused_add_marlin_gemm_mul_rms_norm_4 = async_compile.triton('triton_red_fused_add_marlin_gemm_mul_rms_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_marlin_gemm_mul_rms_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 251668480}}
)
@triton.jit
def triton_red_fused_add_marlin_gemm_mul_rms_norm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [1, 1])
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full([1, 1], 2560.0, tl.float32)
        tmp10 = (tmp4 / tmp9)
        tmp11 = tl.full([1, 1], 1e-06, tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.rsqrt(tmp12)
        tmp14 = tmp8 * tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 * tmp16
        tmp18 = tmp6 + tmp17
        tmp21 = tmp18 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp21, r0_mask & xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.full([1, 1], 2560.0, tl.float32)
        tmp30 = (tmp25 / tmp29)
        tmp31 = tl.full([1, 1], 1e-06, tl.float32)
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp35 * tmp36
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp37, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/6h/c6h2q3g4hjjc3f3puypy77xfjlvdttujfkp74lpilq2dvkyrw3m6.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten, rms_norm_default_5, chunk_1, unsqueeze, mul_3, unsqueeze_1, mul_4, sub, mul_5, mul_6, add_3, cat, reshape_12, to_4, reciprocal, mul_11, clamp, to_6], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat, aten._to_copy, aten.reciprocal, aten.clamp]
# Source node to ATen node mapping:
#   add_3 => add_184
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   clamp => clamp_max, clamp_min
#   index_select => index
#   mul_11 => mul_223
#   mul_3 => mul_156
#   mul_4 => mul_159
#   mul_5 => mul_164
#   mul_6 => mul_167
#   reciprocal => reciprocal
#   reshape_12 => view_16
#   rms_norm_default_5 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   split => split_with_sizes
#   sub => sub_52
#   to_4 => convert_element_type_4
#   to_6 => convert_element_type_5
#   unflatten => view_13
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %marlin_gemm_5 : Tensor "bf16[s72, 3072][3072, 1]cuda:0" = PlaceHolder[target=marlin_gemm_5]
#   %buf23 : Tensor "f32[s72, 8, 1][8, 1, 8*s72]cuda:0" = PlaceHolder[target=buf23]
#   %arg34_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "bf16[131072, 256][256, 1]cuda:0" = PlaceHolder[target=arg36_1]
#   %cat : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=cat]
#   %arg37_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=arg37_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_5, [2048, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg36_1, [%arg35_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 128, -1), kwargs = {})
#   %view_13 : Tensor "bf16[s72, 8, 256][3072, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg1_1, 8, 256]), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg34_1), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_1, 128, -1), kwargs = {})
#   %unsqueeze : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_156 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_159 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_52 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_156, %mul_159), kwargs = {})
#   %mul_164 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_167 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_184 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %mul_167), kwargs = {})
#   %cat : Tensor "bf16[s72, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_52, %add_184], -1), kwargs = {})
#   %view_16 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [%arg1_1, 2048]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %reciprocal : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%arg37_1,), kwargs = {})
#   %mul_223 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %reciprocal), kwargs = {})
#   %clamp_min : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_223, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f8e4m3fn[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   return %buf23,%cat,%convert_element_type_5
triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5 = async_compile.triton('triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 134219264}}
)
@triton.jit
def triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 256*x0 + 3072*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp76 = tl.load(in_ptr4 + (0))
    tmp77 = tl.broadcast_to(tmp76, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp6 = r0_2
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tmp6 >= tmp7
        tmp9 = tl.full([1, 1], 128, tl.int64)
        tmp10 = tmp6 < tmp9
        tmp11 = tl.load(in_ptr0 + (256*x0 + 3072*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 256.0, tl.float32)
        tmp14 = (tmp4 / tmp13)
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.full([1, 1], 131072, tl.int32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp22 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp22)
        tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 131072)) | ~(r0_mask & tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 131072")
        tmp28 = tl.load(in_ptr3 + (tl.broadcast_to(256*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tmp21 * tmp28
        tmp30 = tl.load(in_ptr0 + (128 + 256*x0 + 3072*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 * tmp17
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.load(in_ptr1 + (tl.broadcast_to(128 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tmp33 * tmp34
        tmp36 = tl.load(in_ptr3 + (tl.broadcast_to(128 + 256*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tmp35 * tmp36
        tmp38 = tmp29 - tmp37
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp10, tmp38, tmp39)
        tmp41 = tmp6 >= tmp9
        tmp42 = tl.full([1, 1], 256, tl.int64)
        tmp43 = tmp6 < tmp42
        tmp44 = tl.load(in_ptr0 + (128 + 256*x0 + 3072*x1 + ((-128) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tl.full([1, 1], 256.0, tl.float32)
        tmp47 = (tmp4 / tmp46)
        tmp48 = tl.full([1, 1], 1e-06, tl.float32)
        tmp49 = tmp47 + tmp48
        tmp50 = libdevice.rsqrt(tmp49)
        tmp51 = tmp45 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp53 = tl.load(in_ptr1 + (tl.broadcast_to(128 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.full([1, 1], 131072, tl.int32)
        tmp57 = tmp55 + tmp56
        tmp58 = tmp55 < 0
        tmp59 = tl.where(tmp58, tmp57, tmp55)
        tl.device_assert(((0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 131072)) | ~(r0_mask & tmp41 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 131072")
        tmp61 = tl.load(in_ptr3 + (tl.broadcast_to(256*tmp59 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp62 = tmp54 * tmp61
        tmp63 = tl.load(in_ptr0 + (256*x0 + 3072*x1 + ((-128) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp64 * tmp50
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (tl.broadcast_to((-128) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp68 = tmp66 * tmp67
        tmp69 = tl.load(in_ptr3 + (tl.broadcast_to(128 + 256*tmp59 + ((-128) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp70 = tmp68 * tmp69
        tmp71 = tmp62 + tmp70
        tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
        tmp73 = tl.where(tmp41, tmp71, tmp72)
        tmp74 = tl.where(tmp10, tmp40, tmp73)
        tmp75 = tmp74.to(tl.float32)
        tmp78 = tl.full([1, 1], 1, tl.int32)
        tmp79 = (tmp78 / tmp77)
        tmp80 = tmp75 * tmp79
        tmp81 = tl.full([1, 1], -448.0, tl.float32)
        tmp82 = triton_helpers.maximum(tmp80, tmp81)
        tmp83 = tl.full([1, 1], 448.0, tl.float32)
        tmp84 = triton_helpers.minimum(tmp82, tmp83)
        tmp85 = tmp84.to(tl.float8e4nv)
        tl.store(out_ptr2 + (r0_2 + 256*x3), tmp85, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1 = args
        args.clear()
        s72 = arg1_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [view, marlin_gemm], Original ATen: [aten.view, _C.marlin_gemm]
            buf0 = torch.ops._C.marlin_gemm.default(reinterpret_tensor(arg0_1, (s72, 4096), (4096, 1), 0), None, arg2_1, None, arg3_1, None, arg4_1, None, None, None, arg5_1, 562949953487106, s72, 2560, 4096, True, False, True, False)
            del arg0_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf1 = buf0
            del buf0
            buf2 = empty_strided_cuda((s72, 1), (1, s72), torch.float32)
            buf4 = empty_strided_cuda((s72, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_1, marlin_gemm_1], Original ATen: [vllm_ir.rms_norm, aten.add, _C.marlin_gemm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_marlin_gemm_rms_norm_0.run(buf1, arg6_1, arg7_1, arg8_1, buf2, buf4, s72, 2560, stream=stream0)
            del arg8_1
            # Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_1, marlin_gemm_1], Original ATen: [vllm_ir.rms_norm, aten.add, _C.marlin_gemm]
            buf5 = torch.ops._C.marlin_gemm.default(buf4, None, arg9_1, None, arg10_1, None, arg11_1, None, None, None, arg12_1, 562949953487106, s72, 20480, 2560, True, False, True, False)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg9_1
            del buf4
            buf6 = buf5
            del buf5
            buf7 = empty_strided_cuda((s72, 10240), (10240, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, gelu, getitem_1, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.gelu, aten.mul, _C.marlin_gemm]
            triton_poi_fused_gelu_marlin_gemm_mul_slice_1_xnumel = 10240*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_marlin_gemm_mul_slice_1.run(buf6, buf7, triton_poi_fused_gelu_marlin_gemm_mul_slice_1_xnumel, stream=stream0)
            del buf6
            # Topologically Sorted Source Nodes: [getitem, gelu, getitem_1, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.gelu, aten.mul, _C.marlin_gemm]
            buf8 = torch.ops._C.marlin_gemm.default(buf7, None, arg13_1, None, arg14_1, None, arg15_1, None, None, None, arg16_1, 562949953487106, s72, 2560, 10240, True, False, True, False)
            del arg13_1
            del arg14_1
            del arg15_1
            del arg16_1
            del buf7
            buf9 = buf8
            del buf8
            buf11 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_default, add, rms_norm_default_2, add_1], Original ATen: [vllm_ir.rms_norm, aten.add]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_rms_norm_2.run(buf11, arg17_1, buf1, buf2, arg6_1, arg7_1, s72, 2560, stream=stream0)
            del arg17_1
            del arg6_1
            del arg7_1
            del buf2
            # Topologically Sorted Source Nodes: [marlin_gemm_3], Original ATen: [_C.marlin_gemm]
            buf12 = torch.ops._C.marlin_gemm.default(buf11, None, arg18_1, None, arg19_1, None, arg20_1, None, None, None, arg21_1, 562949953487106, s72, 256, 2560, True, False, True, False)
            del arg18_1
            del arg19_1
            del arg20_1
            del arg21_1
            buf13 = buf12
            del buf12
            buf14 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [gelu_1, mul_1, marlin_gemm_4], Original ATen: [aten.gelu, aten.mul, _C.marlin_gemm]
            triton_poi_fused_gelu_marlin_gemm_mul_3_xnumel = 256*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_marlin_gemm_mul_3.run(buf14, arg22_1, triton_poi_fused_gelu_marlin_gemm_mul_3_xnumel, stream=stream0)
            del arg22_1
            # Topologically Sorted Source Nodes: [gelu_1, mul_1, marlin_gemm_4], Original ATen: [aten.gelu, aten.mul, _C.marlin_gemm]
            buf15 = torch.ops._C.marlin_gemm.default(buf14, None, arg23_1, None, arg24_1, None, arg25_1, None, None, None, arg26_1, 562949953487106, s72, 2560, 256, True, False, True, False)
            del arg23_1
            del arg24_1
            del arg25_1
            del arg26_1
            del buf14
            buf16 = buf15
            del buf15
            buf18 = buf11; del buf11  # reuse
            buf20 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_default_3, add_2, mul_2, rms_norm_default_4, marlin_gemm_5], Original ATen: [vllm_ir.rms_norm, aten.add, aten.mul, _C.marlin_gemm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_marlin_gemm_mul_rms_norm_4.run(buf18, buf16, arg27_1, arg28_1, arg29_1, buf20, s72, 2560, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del buf16
            # Topologically Sorted Source Nodes: [rms_norm_default_4, marlin_gemm_5], Original ATen: [vllm_ir.rms_norm, _C.marlin_gemm]
            buf21 = torch.ops._C.marlin_gemm.default(buf20, None, arg30_1, None, arg31_1, None, arg32_1, None, None, None, arg33_1, 562949953487106, s72, 3072, 2560, True, False, True, False)
            del arg30_1
            del arg31_1
            del arg32_1
            del arg33_1
            del buf20
            buf22 = buf21
            del buf21
            buf25 = empty_strided_cuda((s72, 2048), (2048, 1), torch.float8_e4m3fn)
            # Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten, rms_norm_default_5, chunk_1, unsqueeze, mul_3, unsqueeze_1, mul_4, sub, mul_5, mul_6, add_3, cat, reshape_12, to_4, reciprocal, mul_11, clamp, to_6], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat, aten._to_copy, aten.reciprocal, aten.clamp]
            triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5_xnumel = 8*s72
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5.run(buf22, arg34_1, arg35_1, arg36_1, arg37_1, buf25, triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_5_xnumel, 256, stream=stream0)
            del arg34_1
            del arg35_1
            del arg36_1
            del arg37_1
            buf26 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
        return (reinterpret_tensor(buf25, (s72, 8, 256), (2048, 256, 1), 0), reinterpret_tensor(buf22, (s72, 2, 256), (3072, 256, 1), 2048), reinterpret_tensor(buf22, (s72, 2, 256), (3072, 256, 1), 2560), reinterpret_tensor(buf26, (s72, 8, 256), (2048, 256, 1), 0), buf18, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 8, 512), (4096, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((256, 5120), (5120, 1), device='cuda:0', dtype=torch.int32)
    arg3_1 = rand_strided((256, 2560), (2560, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg4_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg6_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((160, 40960), (40960, 1), device='cuda:0', dtype=torch.int32)
    arg10_1 = rand_strided((160, 20480), (20480, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg11_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg13_1 = rand_strided((640, 5120), (5120, 1), device='cuda:0', dtype=torch.int32)
    arg14_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg15_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg17_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((160, 512), (512, 1), device='cuda:0', dtype=torch.int32)
    arg19_1 = rand_strided((160, 256), (256, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg20_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg22_1 = rand_strided((8192, 256), (10752, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((16, 5120), (5120, 1), device='cuda:0', dtype=torch.int32)
    arg24_1 = rand_strided((16, 2560), (2560, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg25_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg27_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((160, 6144), (6144, 1), device='cuda:0', dtype=torch.int32)
    arg31_1 = rand_strided((160, 3072), (3072, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg32_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg36_1 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
