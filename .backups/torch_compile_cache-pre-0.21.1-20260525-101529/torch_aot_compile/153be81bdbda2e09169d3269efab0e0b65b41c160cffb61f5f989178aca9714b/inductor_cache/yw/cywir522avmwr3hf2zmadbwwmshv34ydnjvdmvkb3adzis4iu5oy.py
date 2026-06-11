
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'kernel_num_gb': 0.52851559, 'kernel_flop': 0}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 42
    x2 = (xindex % 42)
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [1, 1])
    tmp17 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [1, 1])
    tmp36 = tl.load(in_ptr5 + (0)).to(tl.float32)
    tmp37 = tl.broadcast_to(tmp36, [1, 1])
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 256.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-06, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp20 = tl.full([1, 1], 0, tl.int32)
    tmp21 = tmp19 >= tmp20
    tmp22 = tl.full([1, 1], 262144, tl.int32)
    tmp23 = tmp19 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.where(tmp24, tmp19, tmp20)
    tmp26 = tmp25.to(tl.int64)
    tmp27 = tmp26 + tmp22
    tmp28 = tmp26 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp26)
    tl.device_assert(((0 <= tmp29) & (tmp29 < 262144)) | ~(xmask), "index out of bounds: 0 <= tmp29 < 262144")
    tmp31 = tl.load(in_ptr3 + (r0_1 + 256*x2 + 10752*tmp29), xmask, other=0.0).to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp35 = tmp18 + tmp34
    tmp38 = tmp35 * tmp37
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp38, xmask)


def get_args():
    arg_0 = rand_strided((8192, 42, 256), (10752, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_4 = rand_strided((262144, 10752), (10752, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 344064, 256,


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
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.52851559
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
