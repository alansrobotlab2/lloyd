
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
