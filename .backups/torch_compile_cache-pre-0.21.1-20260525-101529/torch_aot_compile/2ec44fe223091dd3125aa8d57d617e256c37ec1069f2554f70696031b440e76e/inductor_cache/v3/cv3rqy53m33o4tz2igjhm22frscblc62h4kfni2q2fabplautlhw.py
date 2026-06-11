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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/yq/cyq4cptiwlskbaqfmrqniy76p4f3nmk4uhuw2ohw7qq2g45zxa4m.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 24, 256][6144, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 6144][6144, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 6144]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   return %mul_4
triton_poi_fused_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/bt/cbtcilksj6a7vhvbnntief72uctvzlqfnodwiltjzaxqafu2c62r.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add => add_21
#   add_1 => add_22
#   float_1 => convert_element_type_2
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %mm : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg6_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg5_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_22 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %arg6_1), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_21 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_21), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %mm), kwargs = {})
#   return %buf2,%convert_element_type_default_3,%add_22,%buf10
triton_red_fused__to_copy_add_copy__rms_norm_1 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_1', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 671098880}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp10, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 5120*x0), tmp8, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/ez/cezuwdmcg37t7u5n3cswsprve7d5ewdamzktb7oq6rb6qtxcfnic.py
# Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem => slice_1
#   getitem_1 => slice_2
#   mul_1 => mul_30
#   silu => add_35, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 34816][34816, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 17408][34816, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 17408), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_35), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 17408][34816, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 17408, 9223372036854775807), kwargs = {})
#   %mul_30 : Tensor "bf16[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_30
triton_poi_fused_mul_silu_slice_2 = async_compile.triton('triton_poi_fused_mul_silu_slice_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1140850688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 17408)
    x1 = xindex // 17408
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 34816*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (17408 + x0 + 34816*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/6f/c6flbbu4ulb643klv2j2wvfknyergqwiecw6oci5froapxm4kf6g.py
# Topologically Sorted Source Nodes: [float_2, add_2, rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_48
#   float_2 => convert_element_type_9
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %addmm_default : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=addmm_default]
#   %buf8 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf8]
#   %arg9_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type_9 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_48 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_default, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_48), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf8,%convert_element_type_default_1
triton_red_fused__to_copy_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 251668480}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 5120.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp19, r0_mask & xmask)
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
buf0 = generate_example_value((8192, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6144))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, 50331648, stream=stream0)
del arg0_1, arg2_1, buf0

stream0 = get_raw_stream(0)
buf1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg6_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg5_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
buf3 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
buf6 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg4_1 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_copy__rms_norm_1.run(buf1, arg6_1, arg5_1, buf3, buf6, arg4_1, 8192, 5120, stream=stream0)
del buf1, arg6_1, arg5_1, buf3, buf6, arg4_1

stream0 = get_raw_stream(0)
buf4 = generate_example_value((8192, 34816), (34816, 1), 'cuda:0', torch.bfloat16, 0, (8192, 34816))
buf5 = generate_example_value((8192, 17408), (17408, 1), 'cuda:0', torch.bfloat16, 0, (8192, 17408))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_2.run(buf4, buf5, 142606336, stream=stream0)
del buf4, buf5

stream0 = get_raw_stream(0)
buf9 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
arg9_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_rms_norm_3.run(buf9, arg9_1, 8192, 5120, stream=stream0)
del buf9, arg9_1

"""
# AOT ID: ['66_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/yq/cyq4cptiwlskbaqfmrqniy76p4f3nmk4uhuw2ohw7qq2g45zxa4m.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 24, 256][6144, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 6144][6144, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 6144]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   return %mul_4
triton_poi_fused_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/bt/cbtcilksj6a7vhvbnntief72uctvzlqfnodwiltjzaxqafu2c62r.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add => add_21
#   add_1 => add_22
#   float_1 => convert_element_type_2
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %mm : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg6_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg5_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_22 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %arg6_1), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_21 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_21), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %mm), kwargs = {})
#   return %buf2,%convert_element_type_default_3,%add_22,%buf10
triton_red_fused__to_copy_add_copy__rms_norm_1 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_1', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 671098880}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp10, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 5120*x0), tmp8, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/ez/cezuwdmcg37t7u5n3cswsprve7d5ewdamzktb7oq6rb6qtxcfnic.py
# Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem => slice_1
#   getitem_1 => slice_2
#   mul_1 => mul_30
#   silu => add_35, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 34816][34816, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 17408][34816, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 17408), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_35 : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_35), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 17408][34816, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 17408, 9223372036854775807), kwargs = {})
#   %mul_30 : Tensor "bf16[s18, 17408][17408, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_30
triton_poi_fused_mul_silu_slice_2 = async_compile.triton('triton_poi_fused_mul_silu_slice_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1140850688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 17408)
    x1 = xindex // 17408
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 34816*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (17408 + x0 + 34816*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/2ec44fe223091dd3125aa8d57d617e256c37ec1069f2554f70696031b440e76e/inductor_cache/6f/c6flbbu4ulb643klv2j2wvfknyergqwiecw6oci5froapxm4kf6g.py
# Topologically Sorted Source Nodes: [float_2, add_2, rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_48
#   float_2 => convert_element_type_9
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %addmm_default : Tensor "bf16[s18, 5120][5120, 1]cuda:0" = PlaceHolder[target=addmm_default]
#   %buf8 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf8]
#   %arg9_1 : Tensor "bf16[5120][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type_9 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_48 : Tensor "f32[5120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_default, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_48), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 5120][5120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf8,%convert_element_type_default_1
triton_red_fused__to_copy_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 251668480}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 5120.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 5120*x0), tmp19, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s72 = arg1_1
        arg0_1_size = arg0_1.size()
        s18 = arg0_1_size[0]
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
            triton_poi_fused_mul_sigmoid_view_0_xnumel = 6144*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, triton_poi_fused_mul_sigmoid_view_0_xnumel, stream=stream0)
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf0, reinterpret_tensor(arg3_1, (6144, 5120), (1, 6144), 0), out=buf1)
            del arg3_1
            del buf0
            buf3 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            buf6 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_copy__rms_norm_1.run(buf1, arg6_1, arg5_1, buf3, buf6, arg4_1, s18, 5120, stream=stream0)
            del arg4_1
            del arg5_1
            del arg6_1
            del buf1
            buf4 = empty_strided_cuda((s18, 34816), (34816, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, linear_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf3, reinterpret_tensor(arg7_1, (5120, 34816), (1, 5120), 0), out=buf4)
            del arg7_1
            del buf3
            buf5 = empty_strided_cuda((s18, 17408), (17408, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_2_xnumel = 17408*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_2.run(buf4, buf5, triton_poi_fused_mul_silu_slice_2_xnumel, stream=stream0)
            del buf4
            buf7 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, getitem, silu, getitem_1, mul_1, linear_2], Original ATen: [aten.add, aten.slice, aten.silu, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(buf6, buf5, reinterpret_tensor(arg8_1, (17408, 5120), (1, 17408), 0), alpha=1, beta=1, out=buf7)
            del arg8_1
            del buf5
            del buf6
            buf9 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [float_2, add_2, rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_rms_norm_3.run(buf9, arg9_1, s18, 5120, stream=stream0)
            del arg9_1
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 24, 256), (6144, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((5120, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((34816, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((5120, 17408), (17408, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
