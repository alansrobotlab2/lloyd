
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
