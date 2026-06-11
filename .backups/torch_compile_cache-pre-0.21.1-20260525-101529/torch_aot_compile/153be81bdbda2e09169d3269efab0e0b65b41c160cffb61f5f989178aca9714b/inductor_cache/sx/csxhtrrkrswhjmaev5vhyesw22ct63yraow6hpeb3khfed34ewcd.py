
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'out_ptr5': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}, 'kernel_name': 'triton_red_fused_5', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr5, xnumel_0, xnumel_1, xnumel_2, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
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
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((131072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg_6 = rand_strided((8192, 2, 1), (2, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((8192, 2, 256), (512, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 65536, 16384, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
