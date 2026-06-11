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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/mv/cmvwxqu4embtzkfmdmhbmtur55uxvsqcesfe7gqrsmdsg4qhqvse.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp29 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (x1 + 2*ks0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tl.broadcast_to(tmp14, [XBLOCK])) & (tl.broadcast_to(tmp14, [XBLOCK]) < 1048576)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp14, [XBLOCK]) < 1048576")
    tmp16 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp0 >= tmp17
    tmp19 = (((-1) + x0) % 3)
    tmp20 = tmp19 == tmp6
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (ks0 + x1), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 1048576)) | ~(tmp21 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 1048576")
    tmp28 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1048576")
    tmp35 = tl.load(in_ptr1 + (x0 + 64*tmp33), xmask).to(tl.float32)
    tmp36 = tl.where(tmp21, tmp28, tmp35)
    tmp37 = tl.where(tmp9, tmp16, tmp36)
    tmp38 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (32 + x0 + 64*tmp33), xmask).to(tl.float32)
    tmp41 = tl.where(tmp21, tmp39, tmp40)
    tmp42 = tl.where(tmp9, tmp38, tmp41)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/ou/cou4ordscvisrypi4aflm7wkeovxk5f234vaeqrzwyd73g2gpur3.py
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
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 3072
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
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
            tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
            tmp6 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tl.full([1, 1], 3072.0, tl.float32)
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
            tl.store(out_ptr1 + (r0_1 + 6144*x0), tmp19, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 3072
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x2 = xindex
        _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp20 = tl.load(in_ptr2 + (r0_3 + 3072*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp21 = tmp20.to(tl.float32)
            tmp22 = tmp21 * tmp21
            tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
            tmp25 = _tmp24 + tmp23
            _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp24 = tl.sum(_tmp24, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp26 = tl.load(in_ptr2 + (r0_3 + 3072*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp34 = tl.load(in_ptr3 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp27 = tmp26.to(tl.float32)
            tmp28 = tl.full([1, 1], 3072.0, tl.float32)
            tmp29 = (tmp24 / tmp28)
            tmp30 = tl.full([1, 1], 1e-06, tl.float32)
            tmp31 = tmp29 + tmp30
            tmp32 = libdevice.rsqrt(tmp31)
            tmp33 = tmp27 * tmp32
            tmp35 = tmp34.to(tl.float32)
            tmp36 = tl.full([1, 1], 1.0, tl.float32)
            tmp37 = tmp35 + tmp36
            tmp38 = tmp33 * tmp37
            tmp39 = tmp38.to(tl.float32)
            tl.store(out_ptr3 + (r0_3 + 6144*x2), tmp39, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 3072), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 3072), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8192, 8192,


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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/ef/cefejsrxq5gxagjlvk7kqpne4ar6imuglzmrnimupzrnkvokaxsu.py
# Topologically Sorted Source Nodes: [float_3, add_2, rms_norm_default_2], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_14
#   float_3 => convert_element_type_4
#   rms_norm_default_2 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %mm : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf6 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf6]
#   %arg6_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_4 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_14 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_14), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.bfloat16), kwargs = {})
#   return %buf6,%convert_element_type_default_5
triton_red_fused__to_copy_add_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 151001088}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp6 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 3072.0, tl.float32)
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
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/vy/cvyomy46u6r73lnpmk7e6fhc7ihuhdunfn7ynygappi2vtcbfzfd.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + 512*(x0 // 256) + 17408*x1 + ((x0 % 256))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/dh/cdhmrslsybs6p33jrpdttc27aokenekjshpmnznxylvueea26sms.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_4 = async_compile.triton('triton_red_fused_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_4', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_4(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        x0 = (xindex % 32)
        x1 = xindex // 32
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 17408*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        x4 = (xindex % 2)
        x5 = xindex // 2
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (16384 + r0_6 + 256*x4 + 17408*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    arg_0 = rand_strided((8192, 17408), (17408, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 32, 1), (32, 1, 262144), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 262144, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/jz/cjzt5eeujajadjzcrqedmjy22jqo6uq7x2zd2cnnynp7re7m3smj.py
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
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        x1 = ((xindex // 64) % 2)
        x2 = xindex // 128
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (16384 + 256*x1 + 17408*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp22 = tl.load(in_ptr0 + (16416 + 256*x1 + 17408*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp38 = tl.load(in_ptr0 + (16416 + 256*x1 + 17408*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp55 = tl.load(in_ptr0 + (16384 + 256*x1 + 17408*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        x5 = ((xindex // 192) % 2)
        x6 = xindex // 384
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (16448 + x4 + 256*x5 + 17408*x6), xmask).to(tl.float32)
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
        x10 = xindex // 2048
        x9 = ((xindex // 64) % 32)
        x11 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (512*x9 + 17408*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp106 = tl.load(in_ptr0 + (32 + 512*x9 + 17408*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp122 = tl.load(in_ptr0 + (32 + 512*x9 + 17408*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp139 = tl.load(in_ptr0 + (512*x9 + 17408*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        x13 = ((xindex // 192) % 32)
        x14 = xindex // 6144
        x15 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (64 + x12 + 512*x13 + 17408*x14), xmask).to(tl.float32)
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
    arg_0 = rand_strided((8192, 17408), (17408, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 32, 1), (32, 1, 262144), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 2, 64), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 2, 192), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 32, 64), (8192, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 32, 192), (8192, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1048576, 3145728, 16777216, 50331648,


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

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    stream0 = get_raw_stream(0)
stream0 = get_raw_stream(0)
arg12_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:0', torch.int64, 0, (3, 8192))
arg11_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (1048576, 64))
buf11 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
buf12 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_0.run(arg12_1, arg11_1, buf11, buf12, 8193, 262144, stream=stream0)
del arg12_1, arg11_1

stream0 = get_raw_stream(0)
arg1_1 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg0_1 = generate_example_value((3072,), (1,), 'cuda:0', torch.bfloat16, 0, (3072,))
arg4_1 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg3_1 = generate_example_value((3072,), (1,), 'cuda:0', torch.bfloat16, 0, (3072,))
buf2 = generate_example_value((8192, 3072), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf3 = generate_example_value((8192, 3072), (6144, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_1.run(arg1_1, arg0_1, arg4_1, arg3_1, buf2, buf3, 8192, 8192, stream=stream0)
del arg1_1, arg0_1, arg4_1, arg3_1, buf2, buf3

stream0 = get_raw_stream(0)
buf5 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
arg6_1 = generate_example_value((3072,), (1,), 'cuda:0', torch.bfloat16, 0, (3072,))
buf7 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_rms_norm_2.run(buf5, arg6_1, buf7, 8192, 3072, stream=stream0)
del buf5, arg6_1, buf7

stream0 = get_raw_stream(0)
buf8 = generate_example_value((8192, 17408), (17408, 1), 'cuda:0', torch.bfloat16, 0, (8192, 17408))
buf20 = generate_example_value((8192, 8192), (8192, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_3.run(buf8, buf20, 67108864, stream=stream0)
del buf20

stream0 = get_raw_stream(0)
buf9 = generate_example_value((8192, 32, 1), (32, 1, 262144), 'cuda:0', torch.float32, 0, (8192, 32, 1))
buf10 = generate_example_value((8192, 2, 1), (2, 1, 16384), 'cuda:0', torch.float32, 0, (8192, 2, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_4.run(buf8, buf9, buf10, 262144, 16384, stream=stream0)

stream0 = get_raw_stream(0)
arg10_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg9_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
buf13 = generate_example_value((8192, 2, 64), (512, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2, 64))
buf14 = generate_example_value((8192, 2, 192), (512, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2, 192))
buf16 = generate_example_value((8192, 32, 64), (8192, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32, 64))
buf17 = generate_example_value((8192, 32, 192), (8192, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32, 192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_5.run(buf8, buf10, arg10_1, buf11, buf12, buf9, arg9_1, buf13, buf14, buf16, buf17, 1048576, 3145728, 16777216, 50331648, stream=stream0)
del buf11, buf12, buf8, buf9, buf10, arg10_1, arg9_1, buf13, buf14, buf16, buf17

"""
# AOT ID: ['50_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/mv/cmvwxqu4embtzkfmdmhbmtur55uxvsqcesfe7gqrsmdsg4qhqvse.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp29 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (x1 + 2*ks0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tl.broadcast_to(tmp14, [XBLOCK])) & (tl.broadcast_to(tmp14, [XBLOCK]) < 1048576)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp14, [XBLOCK]) < 1048576")
    tmp16 = tl.load(in_ptr1 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp0 >= tmp17
    tmp19 = (((-1) + x0) % 3)
    tmp20 = tmp19 == tmp6
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (ks0 + x1), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 1048576)) | ~(tmp21 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 1048576")
    tmp28 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1048576")
    tmp35 = tl.load(in_ptr1 + (x0 + 64*tmp33), xmask).to(tl.float32)
    tmp36 = tl.where(tmp21, tmp28, tmp35)
    tmp37 = tl.where(tmp9, tmp16, tmp36)
    tmp38 = tl.load(in_ptr1 + (34 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3)) + 64*tmp14), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (33 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3)) + 64*tmp26), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (32 + x0 + 64*tmp33), xmask).to(tl.float32)
    tmp41 = tl.where(tmp21, tmp39, tmp40)
    tmp42 = tl.where(tmp9, tmp38, tmp41)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/ou/cou4ordscvisrypi4aflm7wkeovxk5f234vaeqrzwyd73g2gpur3.py
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
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 3072
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
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
            tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
            tmp6 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tl.full([1, 1], 3072.0, tl.float32)
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
            tl.store(out_ptr1 + (r0_1 + 6144*x0), tmp19, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 3072
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x2 = xindex
        _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp20 = tl.load(in_ptr2 + (r0_3 + 3072*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp21 = tmp20.to(tl.float32)
            tmp22 = tmp21 * tmp21
            tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
            tmp25 = _tmp24 + tmp23
            _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp24 = tl.sum(_tmp24, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp26 = tl.load(in_ptr2 + (r0_3 + 3072*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp34 = tl.load(in_ptr3 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp27 = tmp26.to(tl.float32)
            tmp28 = tl.full([1, 1], 3072.0, tl.float32)
            tmp29 = (tmp24 / tmp28)
            tmp30 = tl.full([1, 1], 1e-06, tl.float32)
            tmp31 = tmp29 + tmp30
            tmp32 = libdevice.rsqrt(tmp31)
            tmp33 = tmp27 * tmp32
            tmp35 = tmp34.to(tl.float32)
            tmp36 = tl.full([1, 1], 1.0, tl.float32)
            tmp37 = tmp35 + tmp36
            tmp38 = tmp33 * tmp37
            tmp39 = tmp38.to(tl.float32)
            tl.store(out_ptr3 + (r0_3 + 6144*x2), tmp39, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 3072), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 3072), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8192, 8192,


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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/ef/cefejsrxq5gxagjlvk7kqpne4ar6imuglzmrnimupzrnkvokaxsu.py
# Topologically Sorted Source Nodes: [float_3, add_2, rms_norm_default_2], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add_2 => add_14
#   float_3 => convert_element_type_4
#   rms_norm_default_2 => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %mm : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf6 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf6]
#   %arg6_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_4 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_14 : Tensor "f32[3072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %mul_tensor_5 : Tensor "f32[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_4, %add_14), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_5, torch.bfloat16), kwargs = {})
#   return %buf6,%convert_element_type_default_5
triton_red_fused__to_copy_add_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 151001088}}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3072
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp6 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 3072.0, tl.float32)
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
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/vy/cvyomy46u6r73lnpmk7e6fhc7ihuhdunfn7ynygappi2vtcbfzfd.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + 512*(x0 // 256) + 17408*x1 + ((x0 % 256))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/dh/cdhmrslsybs6p33jrpdttc27aokenekjshpmnznxylvueea26sms.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_4 = async_compile.triton('triton_red_fused_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_4', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_4(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        x0 = (xindex % 32)
        x1 = xindex // 32
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 17408*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        x4 = (xindex % 2)
        x5 = xindex // 2
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (16384 + r0_6 + 256*x4 + 17408*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    arg_0 = rand_strided((8192, 17408), (17408, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 32, 1), (32, 1, 262144), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 262144, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/3ea3ded798589d414a2db0952879f2f711fe7d36f0f871ae1657e3b9c4e14cad/inductor_cache/jz/cjzt5eeujajadjzcrqedmjy22jqo6uq7x2zd2cnnynp7re7m3smj.py
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
    size_hints={'x': 67108864}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'backend_hash': '5E5AC554C8727C61196B79ADC8C935F80F9CE54B410153CD7C6D0C6B4179CF50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        x1 = ((xindex // 64) % 2)
        x2 = xindex // 128
        x3 = xindex // 64
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (16384 + 256*x1 + 17408*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp22 = tl.load(in_ptr0 + (16416 + 256*x1 + 17408*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp38 = tl.load(in_ptr0 + (16416 + 256*x1 + 17408*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp55 = tl.load(in_ptr0 + (16384 + 256*x1 + 17408*x2 + ((-32) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        x5 = ((xindex // 192) % 2)
        x6 = xindex // 384
        x7 = xindex // 192
        tmp69 = tl.load(in_ptr0 + (16448 + x4 + 256*x5 + 17408*x6), xmask).to(tl.float32)
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
        x10 = xindex // 2048
        x9 = ((xindex // 64) % 32)
        x11 = xindex // 64
        tmp84 = x8
        tmp85 = tl.full([1], 0, tl.int64)
        tmp86 = tmp84 >= tmp85
        tmp87 = tl.full([1], 32, tl.int64)
        tmp88 = tmp84 < tmp87
        tmp89 = tl.load(in_ptr0 + (512*x9 + 17408*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp106 = tl.load(in_ptr0 + (32 + 512*x9 + 17408*x10 + (x8)), tmp88 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp122 = tl.load(in_ptr0 + (32 + 512*x9 + 17408*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp139 = tl.load(in_ptr0 + (512*x9 + 17408*x10 + ((-32) + x8)), tmp119 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        x13 = ((xindex // 192) % 32)
        x14 = xindex // 6144
        x15 = xindex // 192
        tmp153 = tl.load(in_ptr0 + (64 + x12 + 512*x13 + 17408*x14), xmask).to(tl.float32)
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
    arg_0 = rand_strided((8192, 17408), (17408, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 32, 1), (32, 1, 262144), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 2, 64), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 2, 192), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 32, 64), (8192, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 32, 192), (8192, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1048576, 3145728, 16777216, 50331648,


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
        args.clear()
        s59 = arg2_1
        s18 = arg8_1
        s7 = arg13_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf11 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            buf4 = empty_strided_cuda((s18, 6144), (6144, 1), torch.bfloat16)
            buf2 = reinterpret_tensor(buf4, (s18, 3072), (6144, 1), 0)  # alias
            buf3 = reinterpret_tensor(buf4, (s18, 3072), (6144, 1), 3072)  # alias
            # Unsorted Source Nodes: [], Original ATen: []
            triton_poi_fused_0_xnumel = 32*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg12_1, arg11_1, buf11, buf12, s7, triton_poi_fused_0_xnumel, stream=stream0)
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_red_fused_1.run(arg1_1, arg0_1, arg4_1, arg3_1, buf2, buf3, s18, s18, stream=stream0)
            del arg0_1
            del arg11_1
            del arg12_1
            del arg1_1
            del arg3_1
            del arg4_1
            del buf2
            del buf3
            buf5 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf4, reinterpret_tensor(arg5_1, (6144, 3072), (1, 6144), 0), out=buf5)
            del arg5_1
            del buf4
            buf7 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_3, add_2, rms_norm_default_2], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_rms_norm_2.run(buf5, arg6_1, buf7, s18, 3072, stream=stream0)
            del arg6_1
            buf8 = empty_strided_cuda((s18, 17408), (17408, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_3, add_2, rms_norm_default_2, linear_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf7, reinterpret_tensor(arg7_1, (3072, 17408), (1, 3072), 0), out=buf8)
            del arg7_1
            del buf7
            buf20 = empty_strided_cuda((s18, 8192), (8192, 1), torch.bfloat16)
            buf9 = empty_strided_cuda((s18, 32, 1), (32, 1, 32*s18), torch.float32)
            buf10 = empty_strided_cuda((s18, 2, 1), (2, 1, 2*s18), torch.float32)
            # Topologically Sorted Source Nodes: [split, view, chunk, rms_norm_default_3, view_3, rms_norm_default_4, reshape_1], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten._unsafe_view]
            triton_poi_fused_3_xnumel = 8192*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_3.run(buf8, buf20, triton_poi_fused_3_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [split, view, chunk, rms_norm_default_3, view_3, rms_norm_default_4, reshape_1], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten._unsafe_view]
            triton_red_fused_4_xnumel_0 = 32*s18
            triton_red_fused_4_xnumel_1 = 2*s18
            stream0 = get_raw_stream(0)
            triton_red_fused_4.run(buf8, buf9, buf10, triton_red_fused_4_xnumel_0, triton_red_fused_4_xnumel_1, stream=stream0)
            buf15 = empty_strided_cuda((s18, 2, 256), (512, 256, 1), torch.bfloat16)
            buf13 = reinterpret_tensor(buf15, (s18, 2, 64), (512, 256, 1), 0)  # alias
            buf14 = reinterpret_tensor(buf15, (s18, 2, 192), (512, 256, 1), 64)  # alias
            buf18 = empty_strided_cuda((s18, 32, 256), (8192, 256, 1), torch.bfloat16)
            buf16 = reinterpret_tensor(buf18, (s18, 32, 64), (8192, 256, 1), 0)  # alias
            buf17 = reinterpret_tensor(buf18, (s18, 32, 192), (8192, 256, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, view, chunk, float_4, add_3, rms_norm_default_3, getitem_14, chunk_2, view_3, float_5, add_4, rms_norm_default_4, getitem_18, chunk_3, mul_4, mul_5, sub_1, mul_6, mul_7, add_6, cat_3, getitem_19, cat_4, mul, mul_1, sub, mul_2, mul_3, add_5, cat_1, getitem_15, cat_2], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._to_copy, aten.add, vllm_ir.rms_norm, aten.slice, aten.unsqueeze, aten.mul, aten.sub, aten.cat]
            triton_poi_fused_5_xnumel_0 = 128*s18
            triton_poi_fused_5_xnumel_1 = 384*s18
            triton_poi_fused_5_xnumel_2 = 2048*s18
            triton_poi_fused_5_xnumel_3 = 6144*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_5.run(buf8, buf10, arg10_1, buf11, buf12, buf9, arg9_1, buf13, buf14, buf16, buf17, triton_poi_fused_5_xnumel_0, triton_poi_fused_5_xnumel_1, triton_poi_fused_5_xnumel_2, triton_poi_fused_5_xnumel_3, stream=stream0)
            del arg10_1
            del arg9_1
            del buf10
            del buf11
            del buf12
            del buf9
            buf19 = empty_strided_cuda((s18, 8192), (8192, 1), torch.bfloat16)
            buf21 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
        return (buf15, reinterpret_tensor(buf8, (s18, 2, 256), (17408, 256, 1), 16896), buf18, reinterpret_tensor(buf19, (s18, 32, 256), (8192, 256, 1), 0), buf20, buf21, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = 8192
    arg3_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((3072, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((17408, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = 8192
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg13_1 = 8193
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
