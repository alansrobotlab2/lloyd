
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'kernel_num_gb': 0.318833156, 'kernel_flop': 0}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


def get_args():
    arg_0 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 256,


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
    num_gb = 0.318833156
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
