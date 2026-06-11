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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/kh/ckhcqc7v52cjf4vu5a7g6k5boz47zo4rxhgtky7ufyimlujft4mu.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   mul => mul_4
#   scaled_fp4_quant_out => scaled_fp4_quant_out
#   sigmoid => sigmoid
#   view => view
#   zeros => full_default
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 32, 256][8192, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 8192]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 128], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%mul_4, %arg5_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf1
triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0 = async_compile.triton('triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 536870912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/z7/cz7aa4pzqozarph4a5zxwfboyt574mrwm63rb4ianuwnhn5qhyly.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   mul => mul_4
#   scaled_fp4_quant_out => scaled_fp4_quant_out
#   sigmoid => sigmoid
#   view => view
#   zeros => full_default
# Graph fragment:
#   %view : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 8192]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 128], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%mul_4, %arg5_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf2
triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1 = async_compile.triton('triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/yi/cyi5ldaou5cdozxczo2vli74ous3tz23zyncjl27ihrhxk2wrntc.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, moe_forward_shared], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add => add_39
#   add_1 => add_40
#   float_1 => convert_element_type
#   moe_forward_shared => moe_forward_shared
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg10_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_40 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %convert_element_type : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_39 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_39), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_3, %convert_element_type_default_3, %convert_element_type_default_3, None, %arg12_1), kwargs = {})
#   return %buf10,%buf11,%buf12,%buf13
triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 402659328}}
)
@triton.jit
def triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp8 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 3072.0, tl.float32)
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
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/s3/cs3r3unglkhimmqrv4dy6vi5jqs5do7wipi43z7rksgppxmd5jy6.py
# Topologically Sorted Source Nodes: [add_1, add_2, add_4, float_2, add_3, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add_1 => add_40
#   add_2 => add_56
#   add_3 => add_66
#   add_4 => add_67
#   float_2 => convert_element_type_1
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %getitem_3 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_67 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=add_67]
#   %buf19 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf19]
#   %arg13_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_40 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %add_56 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_3, %getitem_4), kwargs = {})
#   %add_67 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %add_40), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.float32), kwargs = {})
#   %add_66 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_66), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %flashinfer_mm_fp4), kwargs = {})
#   return %add_67,%buf24,%buf19,%convert_element_type_default_1
triton_red_fused__to_copy_add_copy__rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_3', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 503322624}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp6, r0_mask & xmask)
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp3, r0_mask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.full([1, 1], 3072.0, tl.float32)
        tmp15 = (tmp10 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 3072*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/uh/cuh5uiqoqfht5gdb2ss65ro6j2sdueocygit4zudixnwsu44uams.py
# Topologically Sorted Source Nodes: [zeros_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros_1 => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "bf16[s18, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 64, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default_1
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 268435456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
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
arg0_1 = generate_example_value((8192, 32, 256), (8192, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32, 256))
arg2_1 = generate_example_value((8192, 8192), (8192, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8192))
buf1 = generate_example_value((8192, 8192), (8192, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0.run(arg0_1, arg2_1, buf1, 67108864, stream=stream0)
del arg0_1, arg2_1, buf1

stream0 = get_raw_stream(0)
buf2 = generate_example_value((8192, 128), (128, 1), 'cuda:0', torch.int32, 0, (8192, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1.run(buf2, 1048576, stream=stream0)
del buf2

stream0 = get_raw_stream(0)
buf9 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg11_1 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg10_1 = generate_example_value((3072,), (1,), 'cuda:0', torch.bfloat16, 0, (3072,))
buf11 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf12 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf13 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2.run(buf9, arg11_1, arg10_1, buf11, buf12, buf13, 8192, 3072, stream=stream0)
del arg10_1, buf11, buf12, buf13

stream0 = get_raw_stream(0)
buf18 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf16 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg13_1 = generate_example_value((3072,), (1,), 'cuda:0', torch.bfloat16, 0, (3072,))
arg9_1 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf20 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_copy__rms_norm_3.run(buf18, buf16, buf9, arg11_1, arg13_1, arg9_1, buf20, 8192, 3072, stream=stream0)
del buf9, arg11_1, buf18, buf16, arg13_1, arg9_1, buf20

stream0 = get_raw_stream(0)
buf23 = generate_example_value((8192, 64, 128), (8192, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 64, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_4.run(buf23, 67108864, stream=stream0)
del buf23

"""
# AOT ID: ['4_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/kh/ckhcqc7v52cjf4vu5a7g6k5boz47zo4rxhgtky7ufyimlujft4mu.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   mul => mul_4
#   scaled_fp4_quant_out => scaled_fp4_quant_out
#   sigmoid => sigmoid
#   view => view
#   zeros => full_default
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 32, 256][8192, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 8192]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 128], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%mul_4, %arg5_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf1
triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0 = async_compile.triton('triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 536870912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/z7/cz7aa4pzqozarph4a5zxwfboyt574mrwm63rb4ianuwnhn5qhyly.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
# Source node to ATen node mapping:
#   mul => mul_4
#   scaled_fp4_quant_out => scaled_fp4_quant_out
#   sigmoid => sigmoid
#   view => view
#   zeros => full_default
# Graph fragment:
#   %view : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 8192]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %full_default : Tensor "i32[128*(((s18 + 127)//128)), 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg4_1, 128], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scaled_fp4_quant_out : [num_users=0] = call_function[target=torch.ops._C.scaled_fp4_quant.out](args = (%mul_4, %arg5_1, True), kwargs = {output: %empty, output_scale: %full_default})
#   return %buf2
triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1 = async_compile.triton('triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/yi/cyi5ldaou5cdozxczo2vli74ous3tz23zyncjl27ihrhxk2wrntc.py
# Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, moe_forward_shared], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, vllm.moe_forward_shared]
# Source node to ATen node mapping:
#   add => add_39
#   add_1 => add_40
#   float_1 => convert_element_type
#   moe_forward_shared => moe_forward_shared
#   rms_norm_default => add_tensor_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg10_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_40 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %convert_element_type : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_39 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.float32), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_39), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %moe_forward_shared : [num_users=2] = call_function[target=torch.ops.vllm.moe_forward_shared.default](args = (%convert_element_type_default_3, %convert_element_type_default_3, %convert_element_type_default_3, None, %arg12_1), kwargs = {})
#   return %buf10,%buf11,%buf12,%buf13
triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 402659328}}
)
@triton.jit
def triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp8 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 3072.0, tl.float32)
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
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/s3/cs3r3unglkhimmqrv4dy6vi5jqs5do7wipi43z7rksgppxmd5jy6.py
# Topologically Sorted Source Nodes: [add_1, add_2, add_4, float_2, add_3, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
# Source node to ATen node mapping:
#   add_1 => add_40
#   add_2 => add_56
#   add_3 => add_66
#   add_4 => add_67
#   float_2 => convert_element_type_1
#   rms_norm_default_1 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %getitem_3 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %flashinfer_mm_fp4 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=flashinfer_mm_fp4]
#   %arg11_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %copy_ : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=copy_]
#   %add_67 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=add_67]
#   %buf19 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf19]
#   %arg13_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_40 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%flashinfer_mm_fp4, %arg11_1), kwargs = {})
#   %add_56 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_3, %getitem_4), kwargs = {})
#   %add_67 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %add_40), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.float32), kwargs = {})
#   %add_66 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_66), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %flashinfer_mm_fp4), kwargs = {})
#   return %add_67,%buf24,%buf19,%convert_element_type_default_1
triton_red_fused__to_copy_add_copy__rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_copy__rms_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_copy__rms_norm_3', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 503322624}}
)
@triton.jit
def triton_red_fused__to_copy_add_copy__rms_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp6, r0_mask & xmask)
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp3, r0_mask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.full([1, 1], 3072.0, tl.float32)
        tmp15 = (tmp10 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 3072*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/e1a2297d703f1a5805caf90a2a32407b01fe3df3037936aa29b37648a021ba29/inductor_cache/uh/cuh5uiqoqfht5gdb2ss65ro6j2sdueocygit4zudixnwsu44uams.py
# Topologically Sorted Source Nodes: [zeros_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros_1 => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "bf16[s18, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 64, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default_1
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 268435456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg3_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s18, 4096), (4096, 1), torch.uint8)
            buf1 = empty_strided_cuda((s18, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
            triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0_xnumel = 8192*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0.run(arg0_1, arg2_1, buf1, triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_0_xnumel, stream=stream0)
            del arg0_1
            del arg2_1
            buf2 = empty_strided_cuda((128*((127 + s18) // 128), 128), (128, 1), torch.int32)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
            triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1_xnumel = 16384*((127 + s18) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1.run(buf2, triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view_zeros_1_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, zeros, scaled_fp4_quant_out], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf1, arg5_1, True, output=buf0, output_scale=buf2)
            del arg5_1
            del buf1
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default], Original ATen: [aten.view]
            buf6 = torch.ops.aten.view.dtype(buf2, torch.float8_e4m3fn)
            buf7 = buf6
            # Topologically Sorted Source Nodes: [t, flashinfer_mm_fp4_default, view_3, t_1], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf8 = torch.ops.vllm.flashinfer_mm_fp4.default(buf0, reinterpret_tensor(arg7_1, (4096, 3072), (1, 4096), 0), aten.view.dtype(buf7, torch.uint8), aten.view.dtype(reinterpret_tensor(arg6_1, (512, 3072), (1, 512), 0), torch.uint8), arg8_1, torch.bfloat16, False, 'cutlass')
            del arg6_1
            del arg7_1
            del arg8_1
            del buf0
            del buf2
            del buf6
            del buf7
            buf9 = buf8
            del buf8
            buf11 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            buf13 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, moe_forward_shared], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, vllm.moe_forward_shared]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_2.run(buf9, arg11_1, arg10_1, buf11, buf12, buf13, s18, 3072, stream=stream0)
            del arg10_1
            # Topologically Sorted Source Nodes: [add_1, float_1, add, rms_norm_default, moe_forward_shared], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, vllm.moe_forward_shared]
            buf14 = torch.ops.vllm.moe_forward_shared.default(buf11, buf12, buf13, None, arg12_1)
            del arg12_1
            del buf11
            del buf12
            del buf13
            buf15 = buf14[0]
            buf16 = buf14[1]
            del buf14
            buf17 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            buf18 = buf15; del buf15  # reuse
            buf20 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_1, add_2, add_4, float_2, add_3, rms_norm_default_1], Original ATen: [aten.add, aten._to_copy, vllm_ir.rms_norm, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_copy__rms_norm_3.run(buf18, buf16, buf9, arg11_1, arg13_1, arg9_1, buf20, s18, 3072, stream=stream0)
            del arg11_1
            del arg13_1
            del arg9_1
            del buf16
            del buf9
            buf21 = empty_strided_cuda((s18, 20480), (20480, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf20, reinterpret_tensor(arg14_1, (3072, 20480), (1, 3072), 0), out=buf21)
            del arg14_1
            buf22 = empty_strided_cuda((s18, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf20, reinterpret_tensor(arg15_1, (3072, 128), (1, 3072), 0), out=buf22)
            del arg15_1
            del buf20
            buf23 = empty_strided_cuda((s18, 64, 128), (8192, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [zeros_1], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_4_xnumel = 8192*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_4.run(buf23, triton_poi_fused_zeros_4_xnumel, stream=stream0)
        return (reinterpret_tensor(buf21, (s18, 12288), (20480, 1), 0), reinterpret_tensor(buf22, (s18, 64), (128, 1), 0), reinterpret_tensor(buf22, (s18, 64), (128, 1), 64), buf23, reinterpret_tensor(buf21, (s18, 64, 128), (20480, 128, 1), 12288), s18, buf17, buf18, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 32, 256), (8192, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = 8192
    arg4_1 = 8192
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((3072, 512), (512, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg7_1 = rand_strided((3072, 4096), (4096, 1), device='cuda:0', dtype=torch.uint8)
    arg8_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    import pickle
    global arg12_1
    arg12_1 = pickle.loads(b'\x80\x04\x95c\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c)language_model.model.layers.3.mlp.experts\x94sb.')
    arg13_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((20480, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
