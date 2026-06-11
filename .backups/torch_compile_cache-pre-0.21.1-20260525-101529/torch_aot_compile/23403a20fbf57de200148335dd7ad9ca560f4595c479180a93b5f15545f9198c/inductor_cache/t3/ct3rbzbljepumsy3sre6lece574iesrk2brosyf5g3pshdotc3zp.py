
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 5120
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp1 = tmp0.to(tl.int64)
            tmp2 = tl.full([1, 1], 248320, tl.int32)
            tmp3 = tmp1 + tmp2
            tmp4 = tmp1 < 0
            tmp5 = tl.where(tmp4, tmp3, tmp1)
            tl.device_assert(((0 <= tmp5) & (tmp5 < 248320)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 248320")
            tmp7 = tl.load(in_ptr1 + (r0_1 + 5120*tmp5), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp8 = tmp7.to(tl.float32)
            tmp9 = tmp8 * tmp8
            tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
            tmp12 = _tmp11 + tmp10
            _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp11 = tl.sum(_tmp11, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp13 = tmp0.to(tl.int64)
            tmp14 = tl.full([1, 1], 248320, tl.int32)
            tmp15 = tmp13 + tmp14
            tmp16 = tmp13 < 0
            tmp17 = tl.where(tmp16, tmp15, tmp13)
            tl.device_assert(((0 <= tmp17) & (tmp17 < 248320)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 248320")
            tmp19 = tl.load(in_ptr1 + (r0_1 + 5120*tmp17), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp20 = tmp19.to(tl.float32)
            tmp21 = tl.full([1, 1], 5120.0, tl.float32)
            tmp22 = (tmp11 / tmp21)
            tmp23 = tl.full([1, 1], 1e-06, tl.float32)
            tmp24 = tmp22 + tmp23
            tmp25 = libdevice.rsqrt(tmp24)
            tmp26 = tmp20 * tmp25
            tmp28 = tmp27.to(tl.float32)
            tmp29 = tl.full([1, 1], 1.0, tl.float32)
            tmp30 = tmp28 + tmp29
            tmp31 = tmp26 * tmp30
            tmp32 = tmp31.to(tl.float32)
            tl.store(out_ptr1 + (r0_1 + 10240*x0), tmp32, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 5120
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x2 = xindex
        _tmp37 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp33 = tl.load(in_ptr3 + (r0_3 + 5120*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp34 = tmp33.to(tl.float32)
            tmp35 = tmp34 * tmp34
            tmp36 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
            tmp38 = _tmp37 + tmp36
            _tmp37 = tl.where(r0_mask & xmask, tmp38, _tmp37)
        tmp37 = tl.sum(_tmp37, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp39 = tl.load(in_ptr3 + (r0_3 + 5120*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp47 = tl.load(in_ptr4 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp40 = tmp39.to(tl.float32)
            tmp41 = tl.full([1, 1], 5120.0, tl.float32)
            tmp42 = (tmp37 / tmp41)
            tmp43 = tl.full([1, 1], 1e-06, tl.float32)
            tmp44 = tmp42 + tmp43
            tmp45 = libdevice.rsqrt(tmp44)
            tmp46 = tmp40 * tmp45
            tmp48 = tmp47.to(tl.float32)
            tmp49 = tl.full([1, 1], 1.0, tl.float32)
            tmp50 = tmp48 + tmp49
            tmp51 = tmp46 * tmp50
            tmp52 = tmp51.to(tl.float32)
            tl.store(out_ptr3 + (r0_3 + 10240*x2), tmp52, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((248320, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((5120,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((5120,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 5120), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((8192, 5120), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8192, 8192,


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
