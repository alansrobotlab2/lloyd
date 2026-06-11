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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/na/cnaig32xnj6ibtv74ed62rlbzrtjdowlou5qzwyswilt55mcwbez.py
# Topologically Sorted Source Nodes: [long, embedding, mul, rms_norm_default_1, marlin_gemm_1], Original ATen: [aten._to_copy, aten.embedding, aten.mul, vllm_ir.rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   marlin_gemm_1 => marlin_gemm_1
#   mul => mul_3
#   rms_norm_default_1 => add_tensor_4, convert_element_type_default_8, convert_element_type_default_9, mean_dim_4, mul_tensor_7, mul_tensor_8, pow_tensor_scalar_4, rsqrt_default_4
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[262144, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=arg3_1]
#   %mul_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=mul_3]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg13_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %mul_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %arg3_1), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.float32), kwargs = {})
#   %pow_tensor_scalar_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_8, 2), kwargs = {})
#   %mean_dim_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_4, [-1], True), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_4, 1e-06), kwargs = {})
#   %rsqrt_default_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, %rsqrt_default_4), kwargs = {})
#   %convert_element_type_default_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.bfloat16), kwargs = {})
#   %mul_tensor_8 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_9, %arg13_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "bf16[s72, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_8, None, %arg14_1, None, %arg15_1, None, %arg16_1, None, None, None, %arg17_1, 562949953487106, %arg1_1, 3072, 2560, True, False, True, False), kwargs = {})
#   return %mul_3,%buf1,%buf2
triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 167777280}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [1, 1])
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 262144, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 262144)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 262144")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 2560*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tl.store(out_ptr0 + (r0_1 + 2560*x0), tmp10, r0_mask & xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full([1, 1], 2560.0, tl.float32)
        tmp19 = (tmp14 / tmp18)
        tmp20 = tl.full([1, 1], 1e-06, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/gq/cgqvsamnnw7etrqt55y66ot4w5iz4kbemazkremq6rdb3qj77fgz.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*i32', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'out_ptr5': '*bf16', 'xnumel_0': 'i64', 'xnumel_1': 'i64', 'xnumel_2': 'i64', 'xnumel_3': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': None, 'xnumel_3': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr5, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
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
        tmp78 = tl.where(xmask, tmp77, 0.0)
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
            tmp79 = tl.full([1, 1], 1, tl.int32)
            tmp80 = (tmp79 / tmp78)
            tmp81 = tmp75 * tmp80
            tmp82 = tl.full([1, 1], -448.0, tl.float32)
            tmp83 = triton_helpers.maximum(tmp81, tmp82)
            tmp84 = tl.full([1, 1], 448.0, tl.float32)
            tmp85 = triton_helpers.minimum(tmp83, tmp84)
            tmp86 = tmp85.to(tl.float8e4nv)
            tl.store(out_ptr2 + (r0_2 + 256*x3), tmp86, r0_mask & xmask)
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
        x4 = (xindex % 2)
        x5 = xindex // 2
        _tmp91 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp87 = tl.load(in_ptr0 + (2048 + r0_6 + 256*x4 + 3072*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp88 = tmp87.to(tl.float32)
            tmp89 = tmp88 * tmp88
            tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
            tmp92 = _tmp91 + tmp90
            _tmp91 = tl.where(r0_mask & xmask, tmp92, _tmp91)
        tmp91 = tl.sum(_tmp91, 1)[:, None]
        tl.store(out_ptr3 + (x7), tmp91, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_2
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x8 = (xindex % 2)
        x9 = xindex // 2
        _tmp97 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x11 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_10 = r0_index
            tmp93 = tl.load(in_ptr0 + (2560 + r0_10 + 256*x8 + 3072*x9), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp94 = tmp93.to(tl.float32)
            tmp95 = tmp94 * tmp94
            tmp96 = tl.broadcast_to(tmp95, [XBLOCK, R0_BLOCK])
            tmp98 = _tmp97 + tmp96
            _tmp97 = tl.where(r0_mask & xmask, tmp98, _tmp97)
        tmp97 = tl.sum(_tmp97, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_10 = r0_index
            tmp99 = tl.load(in_ptr0 + (2560 + r0_10 + 256*x8 + 3072*x9), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp100 = tmp99.to(tl.float32)
            tmp101 = tl.full([1, 1], 256.0, tl.float32)
            tmp102 = (tmp97 / tmp101)
            tmp103 = tl.full([1, 1], 1e-06, tl.float32)
            tmp104 = tmp102 + tmp103
            tmp105 = libdevice.rsqrt(tmp104)
            tmp106 = tmp100 * tmp105
            tmp107 = tmp106.to(tl.float32)
            tl.store(out_ptr5 + (r0_10 + 256*x11), tmp107, r0_mask & xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 256
        R0_BLOCK_3: tl.constexpr = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_3
        xoffset = pid_offset.to(tl.int64) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
        xmask = xindex < xnumel_3
        r0_index = tl.arange(0, R0_BLOCK_3)[None, :].to(tl.int64)
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_13 = r0_index
        x12 = xindex
        x15 = xindex // 42
        x14 = (xindex % 42)
        tmp108 = tl.load(in_out_ptr0 + (r0_13 + 256*x12), r0_mask & xmask, other=0.0).to(tl.float32)
        tmp109 = tl.load(in_ptr5 + (0)).to(tl.float32)
        tmp110 = tl.broadcast_to(tmp109, [1, 1])
        tmp111 = tl.where(xmask, tmp110, 0.0)
        tmp126 = tl.load(in_ptr6 + (r0_13), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp128 = tl.load(in_ptr7 + (x15), xmask, eviction_policy='evict_last')
        tmp141 = tl.load(in_ptr9 + (0)).to(tl.float32)
        tmp142 = tl.broadcast_to(tmp141, [1, 1])
        tmp143 = tl.where(xmask, tmp142, 0.0)
        tmp146 = tl.load(in_ptr10 + (0)).to(tl.float32)
        tmp147 = tl.broadcast_to(tmp146, [1, 1])
        tmp148 = tl.where(xmask, tmp147, 0.0)
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tmp113 * tmp113
        tmp115 = tl.broadcast_to(tmp114, [XBLOCK, R0_BLOCK_3])
        tmp117 = tl.where(r0_mask & xmask, tmp115, 0)
        tmp118 = tl.sum(tmp117, 1)[:, None].to(tl.float32)
        tmp119 = tl.full([1, 1], 256.0, tl.float32)
        tmp120 = (tmp118 / tmp119)
        tmp121 = tl.full([1, 1], 1e-06, tl.float32)
        tmp122 = tmp120 + tmp121
        tmp123 = libdevice.rsqrt(tmp122)
        tmp124 = tmp113 * tmp123
        tmp125 = tmp124.to(tl.float32)
        tmp127 = tmp125 * tmp126
        tmp129 = tl.full([1, 1], 0, tl.int32)
        tmp130 = tmp128 >= tmp129
        tmp131 = tl.full([1, 1], 262144, tl.int32)
        tmp132 = tmp128 < tmp131
        tmp133 = tmp130 & tmp132
        tmp134 = tl.where(tmp133, tmp128, tmp129)
        tmp135 = tmp134.to(tl.int64)
        tmp136 = tmp135 + tmp131
        tmp137 = tmp135 < 0
        tmp138 = tl.where(tmp137, tmp136, tmp135)
        tl.device_assert(((0 <= tmp138) & (tmp138 < 262144)) | ~(xmask), "index out of bounds: 0 <= tmp138 < 262144")
        tmp140 = tl.load(in_ptr8 + (r0_13 + 256*x14 + 10752*tmp138), r0_mask & xmask, other=0.0).to(tl.float32)
        tmp144 = tmp140 * tmp143
        tmp145 = tmp127 + tmp144
        tmp149 = tmp145 * tmp148
        tl.store(in_out_ptr0 + (r0_13 + 256*x12), tmp149, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 42, 256), (10752, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_9 = rand_strided((262144, 10752), (10752, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_11 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_12 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg_13 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_14 = rand_strided((8192, 2, 256), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, 65536, 16384, 16384, 344064,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/3y/c3yypkldwxteiogpsdsras6xic6ctfou5nn5nhw4wtbhrqpetve7.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten_1, rms_norm_default_3, chunk_2, unsqueeze_2, mul_8, unsqueeze_3, mul_9, sub_1, mul_10, mul_11, add_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_226
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_10 => mul_156
#   mul_11 => mul_159
#   mul_8 => mul_148
#   mul_9 => mul_151
#   rms_norm_default_3 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_3, mul_tensor_4, pow_tensor_scalar_2, rsqrt_default_2
#   split => split_with_sizes
#   sub_1 => sub_64
#   unflatten_1 => view_8
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
# Graph fragment:
#   %marlin_gemm_1 : Tensor "bf16[s72, 3072][3072, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %buf6 : Tensor "f32[s72, 2, 1][2, 1, 2*s72]cuda:0" = PlaceHolder[target=buf6]
#   %arg19_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "bf16[131072, 256][256, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_1, [2048, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg21_1, [%arg20_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 128, -1), kwargs = {})
#   %view_8 : Tensor "bf16[s72, 2, 256][3072, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, 2, 256]), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %mul_tensor_4 : Tensor "bf16[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg19_1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_4, 128, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_148 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_151 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_64 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_148, %mul_151), kwargs = {})
#   %mul_156 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_159 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_226 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %mul_159), kwargs = {})
#   return %sub_64,%add_226
triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25166336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2)
    x2 = xindex // 256
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 256*x1 + 3072*x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2176 + x0 + 256*x1 + 3072*x2), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 256.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tl.full([XBLOCK], 131072, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 131072)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 131072")
    tmp18 = tl.load(in_ptr4 + (x0 + 256*tmp16), xmask).to(tl.float32)
    tmp19 = tmp11 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.load(in_ptr4 + (128 + x0 + 256*tmp16), xmask).to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 - tmp27
    tmp29 = tmp25 * tmp18
    tmp30 = tmp11 * tmp26
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x0 + 256*x3), tmp28, xmask)
    tl.store(out_ptr1 + (x0 + 256*x3), tmp31, xmask)
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
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((262144, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (262144, 2560))
arg3_1 = generate_example_value((), (), 'cuda:0', torch.bfloat16, 0, ())
arg13_1 = generate_example_value((2560,), (1,), 'cuda:0', torch.bfloat16, 0, (2560,))
buf0 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf2 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0.run(arg0_1, arg2_1, arg3_1, arg13_1, buf0, buf2, 8192, 2560, stream=stream0)
del arg2_1, arg3_1, arg13_1, buf0, buf2

stream0 = get_raw_stream(0)
buf18 = generate_example_value((8192, 42, 256), (10752, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 42, 256))
buf4 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg18_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg20_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg21_1 = generate_example_value((131072, 256), (256, 1), 'cuda:0', torch.bfloat16, 0, (131072, 256))
arg22_1 = generate_example_value((), (), 'cuda:0', torch.float32, 0, ())
arg10_1 = generate_example_value((), (), 'cuda:0', torch.bfloat16, 0, ())
arg11_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg4_1 = generate_example_value((262144, 10752), (10752, 1), 'cuda:0', torch.bfloat16, 0, (262144, 10752))
arg5_1 = generate_example_value((), (), 'cuda:0', torch.bfloat16, 0, ())
arg12_1 = generate_example_value((), (), 'cuda:0', torch.bfloat16, 0, ())
buf13 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.float8_e4m3fn, 0, (8192, 2048))
buf6 = generate_example_value((8192, 2, 1), (2, 1, 16384), 'cuda:0', torch.float32, 0, (8192, 2, 1))
buf11 = generate_example_value((8192, 2, 256), (512, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2, 256))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_1.run(buf18, buf4, arg18_1, arg20_1, arg21_1, arg22_1, arg10_1, arg11_1, arg0_1, arg4_1, arg5_1, arg12_1, buf13, buf6, buf11, 65536, 16384, 16384, 344064, stream=stream0)
del arg0_1, buf18, arg18_1, arg22_1, arg10_1, arg11_1, arg4_1, arg5_1, arg12_1, buf13, buf11

stream0 = get_raw_stream(0)
arg19_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
buf7 = generate_example_value((8192, 2, 128), (512, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2, 128))
buf8 = generate_example_value((8192, 2, 128), (512, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2.run(buf4, buf6, arg19_1, arg20_1, arg21_1, buf7, buf8, 2097152, stream=stream0)
del buf4, arg20_1, arg21_1, buf6, arg19_1, buf7, buf8

"""
# AOT ID: ['0_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/na/cnaig32xnj6ibtv74ed62rlbzrtjdowlou5qzwyswilt55mcwbez.py
# Topologically Sorted Source Nodes: [long, embedding, mul, rms_norm_default_1, marlin_gemm_1], Original ATen: [aten._to_copy, aten.embedding, aten.mul, vllm_ir.rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   marlin_gemm_1 => marlin_gemm_1
#   mul => mul_3
#   rms_norm_default_1 => add_tensor_4, convert_element_type_default_8, convert_element_type_default_9, mean_dim_4, mul_tensor_7, mul_tensor_8, pow_tensor_scalar_4, rsqrt_default_4
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[262144, 2560][2560, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=arg3_1]
#   %mul_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0" = PlaceHolder[target=mul_3]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg13_1 : Tensor "bf16[2560][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %mul_3 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %arg3_1), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.float32), kwargs = {})
#   %pow_tensor_scalar_4 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_8, 2), kwargs = {})
#   %mean_dim_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_4, [-1], True), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_4, 1e-06), kwargs = {})
#   %rsqrt_default_4 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_tensor_7 : Tensor "f32[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, %rsqrt_default_4), kwargs = {})
#   %convert_element_type_default_9 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.bfloat16), kwargs = {})
#   %mul_tensor_8 : Tensor "bf16[s72, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_9, %arg13_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "bf16[s72, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_8, None, %arg14_1, None, %arg15_1, None, %arg16_1, None, None, None, %arg17_1, 562949953487106, %arg1_1, 3072, 2560, True, False, True, False), kwargs = {})
#   return %mul_3,%buf1,%buf2
triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 167777280}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [1, 1])
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 262144, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 262144)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 262144")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 2560*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tl.store(out_ptr0 + (r0_1 + 2560*x0), tmp10, r0_mask & xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full([1, 1], 2560.0, tl.float32)
        tmp19 = (tmp14 / tmp18)
        tmp20 = tl.full([1, 1], 1e-06, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tl.store(out_ptr2 + (r0_1 + 2560*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/gq/cgqvsamnnw7etrqt55y66ot4w5iz4kbemazkremq6rdb3qj77fgz.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*i32', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'out_ptr5': '*bf16', 'xnumel_0': 'i64', 'xnumel_1': 'i64', 'xnumel_2': 'i64', 'xnumel_3': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': None, 'xnumel_3': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr5, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
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
        tmp78 = tl.where(xmask, tmp77, 0.0)
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
            tmp79 = tl.full([1, 1], 1, tl.int32)
            tmp80 = (tmp79 / tmp78)
            tmp81 = tmp75 * tmp80
            tmp82 = tl.full([1, 1], -448.0, tl.float32)
            tmp83 = triton_helpers.maximum(tmp81, tmp82)
            tmp84 = tl.full([1, 1], 448.0, tl.float32)
            tmp85 = triton_helpers.minimum(tmp83, tmp84)
            tmp86 = tmp85.to(tl.float8e4nv)
            tl.store(out_ptr2 + (r0_2 + 256*x3), tmp86, r0_mask & xmask)
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
        x4 = (xindex % 2)
        x5 = xindex // 2
        _tmp91 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp87 = tl.load(in_ptr0 + (2048 + r0_6 + 256*x4 + 3072*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp88 = tmp87.to(tl.float32)
            tmp89 = tmp88 * tmp88
            tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
            tmp92 = _tmp91 + tmp90
            _tmp91 = tl.where(r0_mask & xmask, tmp92, _tmp91)
        tmp91 = tl.sum(_tmp91, 1)[:, None]
        tl.store(out_ptr3 + (x7), tmp91, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_2
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x8 = (xindex % 2)
        x9 = xindex // 2
        _tmp97 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x11 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_10 = r0_index
            tmp93 = tl.load(in_ptr0 + (2560 + r0_10 + 256*x8 + 3072*x9), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp94 = tmp93.to(tl.float32)
            tmp95 = tmp94 * tmp94
            tmp96 = tl.broadcast_to(tmp95, [XBLOCK, R0_BLOCK])
            tmp98 = _tmp97 + tmp96
            _tmp97 = tl.where(r0_mask & xmask, tmp98, _tmp97)
        tmp97 = tl.sum(_tmp97, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_10 = r0_index
            tmp99 = tl.load(in_ptr0 + (2560 + r0_10 + 256*x8 + 3072*x9), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp100 = tmp99.to(tl.float32)
            tmp101 = tl.full([1, 1], 256.0, tl.float32)
            tmp102 = (tmp97 / tmp101)
            tmp103 = tl.full([1, 1], 1e-06, tl.float32)
            tmp104 = tmp102 + tmp103
            tmp105 = libdevice.rsqrt(tmp104)
            tmp106 = tmp100 * tmp105
            tmp107 = tmp106.to(tl.float32)
            tl.store(out_ptr5 + (r0_10 + 256*x11), tmp107, r0_mask & xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 256
        R0_BLOCK_3: tl.constexpr = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_3
        xoffset = pid_offset.to(tl.int64) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
        xmask = xindex < xnumel_3
        r0_index = tl.arange(0, R0_BLOCK_3)[None, :].to(tl.int64)
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_13 = r0_index
        x12 = xindex
        x15 = xindex // 42
        x14 = (xindex % 42)
        tmp108 = tl.load(in_out_ptr0 + (r0_13 + 256*x12), r0_mask & xmask, other=0.0).to(tl.float32)
        tmp109 = tl.load(in_ptr5 + (0)).to(tl.float32)
        tmp110 = tl.broadcast_to(tmp109, [1, 1])
        tmp111 = tl.where(xmask, tmp110, 0.0)
        tmp126 = tl.load(in_ptr6 + (r0_13), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp128 = tl.load(in_ptr7 + (x15), xmask, eviction_policy='evict_last')
        tmp141 = tl.load(in_ptr9 + (0)).to(tl.float32)
        tmp142 = tl.broadcast_to(tmp141, [1, 1])
        tmp143 = tl.where(xmask, tmp142, 0.0)
        tmp146 = tl.load(in_ptr10 + (0)).to(tl.float32)
        tmp147 = tl.broadcast_to(tmp146, [1, 1])
        tmp148 = tl.where(xmask, tmp147, 0.0)
        tmp112 = tmp108 * tmp111
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tmp113 * tmp113
        tmp115 = tl.broadcast_to(tmp114, [XBLOCK, R0_BLOCK_3])
        tmp117 = tl.where(r0_mask & xmask, tmp115, 0)
        tmp118 = tl.sum(tmp117, 1)[:, None].to(tl.float32)
        tmp119 = tl.full([1, 1], 256.0, tl.float32)
        tmp120 = (tmp118 / tmp119)
        tmp121 = tl.full([1, 1], 1e-06, tl.float32)
        tmp122 = tmp120 + tmp121
        tmp123 = libdevice.rsqrt(tmp122)
        tmp124 = tmp113 * tmp123
        tmp125 = tmp124.to(tl.float32)
        tmp127 = tmp125 * tmp126
        tmp129 = tl.full([1, 1], 0, tl.int32)
        tmp130 = tmp128 >= tmp129
        tmp131 = tl.full([1, 1], 262144, tl.int32)
        tmp132 = tmp128 < tmp131
        tmp133 = tmp130 & tmp132
        tmp134 = tl.where(tmp133, tmp128, tmp129)
        tmp135 = tmp134.to(tl.int64)
        tmp136 = tmp135 + tmp131
        tmp137 = tmp135 < 0
        tmp138 = tl.where(tmp137, tmp136, tmp135)
        tl.device_assert(((0 <= tmp138) & (tmp138 < 262144)) | ~(xmask), "index out of bounds: 0 <= tmp138 < 262144")
        tmp140 = tl.load(in_ptr8 + (r0_13 + 256*x14 + 10752*tmp138), r0_mask & xmask, other=0.0).to(tl.float32)
        tmp144 = tmp140 * tmp143
        tmp145 = tmp127 + tmp144
        tmp149 = tmp145 * tmp148
        tl.store(in_out_ptr0 + (r0_13 + 256*x12), tmp149, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 42, 256), (10752, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_9 = rand_strided((262144, 10752), (10752, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_11 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_12 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg_13 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_14 = rand_strided((8192, 2, 256), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, 65536, 16384, 16384, 344064,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/803e77ed123de1bee735b32030bea3b783b56618fe379ceac6bbbe3288eb3700/inductor_cache/3y/c3yypkldwxteiogpsdsras6xic6ctfou5nn5nhw4wtbhrqpetve7.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten_1, rms_norm_default_3, chunk_2, unsqueeze_2, mul_8, unsqueeze_3, mul_9, sub_1, mul_10, mul_11, add_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_226
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_10 => mul_156
#   mul_11 => mul_159
#   mul_8 => mul_148
#   mul_9 => mul_151
#   rms_norm_default_3 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_3, mul_tensor_4, pow_tensor_scalar_2, rsqrt_default_2
#   split => split_with_sizes
#   sub_1 => sub_64
#   unflatten_1 => view_8
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
# Graph fragment:
#   %marlin_gemm_1 : Tensor "bf16[s72, 3072][3072, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %buf6 : Tensor "f32[s72, 2, 1][2, 1, 2*s72]cuda:0" = PlaceHolder[target=buf6]
#   %arg19_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "bf16[131072, 256][256, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_1, [2048, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg21_1, [%arg20_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 128, -1), kwargs = {})
#   %view_8 : Tensor "bf16[s72, 2, 256][3072, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, 2, 256]), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   %mul_tensor_4 : Tensor "bf16[s72, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg19_1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_4, 128, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_148 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s72, 1, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_151 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_64 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_148, %mul_151), kwargs = {})
#   %mul_156 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_159 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_226 : Tensor "bf16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %mul_159), kwargs = {})
#   return %sub_64,%add_226
triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25166336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2)
    x2 = xindex // 256
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 256*x1 + 3072*x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2176 + x0 + 256*x1 + 3072*x2), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 256.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tl.full([XBLOCK], 131072, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 131072)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 131072")
    tmp18 = tl.load(in_ptr4 + (x0 + 256*tmp16), xmask).to(tl.float32)
    tmp19 = tmp11 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.load(in_ptr4 + (128 + x0 + 256*tmp16), xmask).to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 - tmp27
    tmp29 = tmp25 * tmp18
    tmp30 = tmp11 * tmp26
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x0 + 256*x3), tmp28, xmask)
    tl.store(out_ptr1 + (x0 + 256*x3), tmp31, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1 = args
        args.clear()
        s72 = arg1_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s72, 2560), (2560, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((s72, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [long, embedding, mul, rms_norm_default_1, marlin_gemm_1], Original ATen: [aten._to_copy, aten.embedding, aten.mul, vllm_ir.rms_norm, _C.marlin_gemm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_embedding_marlin_gemm_mul_rms_norm_0.run(arg0_1, arg2_1, arg3_1, arg13_1, buf0, buf2, s72, 2560, stream=stream0)
            del arg13_1
            del arg2_1
            del arg3_1
            # Topologically Sorted Source Nodes: [rms_norm_default_1, marlin_gemm_1], Original ATen: [vllm_ir.rms_norm, _C.marlin_gemm]
            buf3 = torch.ops._C.marlin_gemm.default(buf2, None, arg14_1, None, arg15_1, None, arg16_1, None, None, None, arg17_1, 562949953487106, s72, 3072, 2560, True, False, True, False)
            del arg14_1
            del arg15_1
            del arg16_1
            del arg17_1
            del buf2
            buf4 = buf3
            del buf3
            # Topologically Sorted Source Nodes: [marlin_gemm], Original ATen: [_C.marlin_gemm]
            buf15 = torch.ops._C.marlin_gemm.default(buf0, None, arg6_1, None, arg7_1, None, arg8_1, None, None, None, arg9_1, 562949953487106, s72, 10752, 2560, True, False, True, False)
            del arg6_1
            del arg7_1
            del arg8_1
            del arg9_1
            buf16 = buf15
            del buf15
            buf13 = empty_strided_cuda((s72, 2048), (2048, 1), torch.float8_e4m3fn)
            buf6 = empty_strided_cuda((s72, 2, 1), (2, 1, 2*s72), torch.float32)
            buf11 = empty_strided_cuda((s72, 2, 256), (512, 256, 1), torch.bfloat16)
            buf18 = reinterpret_tensor(buf16, (s72, 42, 256), (10752, 256, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [split, unflatten_1, rms_norm_default_3], Original ATen: [aten.split_with_sizes, aten.view, vllm_ir.rms_norm]
            triton_red_fused_1_xnumel_0 = 8*s72
            triton_red_fused_1_xnumel_1 = 2*s72
            triton_red_fused_1_xnumel_2 = 2*s72
            triton_red_fused_1_xnumel_3 = 42*s72
            stream0 = get_raw_stream(0)
            triton_red_fused_1.run(buf18, buf4, arg18_1, arg20_1, arg21_1, arg22_1, arg10_1, arg11_1, arg0_1, arg4_1, arg5_1, arg12_1, buf13, buf6, buf11, triton_red_fused_1_xnumel_0, triton_red_fused_1_xnumel_1, triton_red_fused_1_xnumel_2, triton_red_fused_1_xnumel_3, stream=stream0)
            del arg0_1
            del arg10_1
            del arg11_1
            del arg12_1
            del arg18_1
            del arg22_1
            del arg4_1
            del arg5_1
            buf9 = empty_strided_cuda((s72, 2, 256), (512, 256, 1), torch.bfloat16)
            buf7 = reinterpret_tensor(buf9, (s72, 2, 128), (512, 256, 1), 0)  # alias
            buf8 = reinterpret_tensor(buf9, (s72, 2, 128), (512, 256, 1), 128)  # alias
            # Topologically Sorted Source Nodes: [split, index_select, chunk, unflatten_1, rms_norm_default_3, chunk_2, unsqueeze_2, mul_8, unsqueeze_3, mul_9, sub_1, mul_10, mul_11, add_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
            triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 256*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2.run(buf4, buf6, arg19_1, arg20_1, arg21_1, buf7, buf8, triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=stream0)
            del arg19_1
            del arg20_1
            del arg21_1
            del buf4
            del buf6
            buf14 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
        return (buf9, buf11, reinterpret_tensor(buf13, (s72, 8, 256), (2048, 256, 1), 0), reinterpret_tensor(buf14, (s72, 8, 256), (2048, 256, 1), 0), buf0, reinterpret_tensor(buf18, (s72, 256), (10752, 1), 0), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 256), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 512), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 768), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 1024), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 1280), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 1536), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 1792), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 2048), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 2304), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 2560), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 2816), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 3072), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 3328), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 3584), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 3840), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 4096), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 4352), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 4608), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 4864), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 5120), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 5376), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 5632), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 5888), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 6144), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 6400), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 6656), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 6912), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 7168), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 7424), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 7680), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 7936), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 8192), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 8448), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 8704), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 8960), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 9216), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 9472), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 9728), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 9984), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 10240), reinterpret_tensor(buf18, (s72, 256), (10752, 1), 10496), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((262144, 10752), (10752, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((160, 21504), (21504, 1), device='cuda:0', dtype=torch.int32)
    arg7_1 = rand_strided((160, 10752), (10752, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg8_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg10_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((160, 6144), (6144, 1), device='cuda:0', dtype=torch.int32)
    arg15_1 = rand_strided((160, 3072), (3072, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg16_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((170, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg21_1 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
