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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/91717618ef16106b41f25b4dc1eae4ef4a04701d8036bd632f4f95983ccc2721/inductor_cache/ti/cti5rkebmvesd2pzdkqufx4nui27mn67svrxaeoiaoodct27cofh.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(out_ptr0, out_ptr1, out_ptr2, xnumel_0, xnumel_1, xnumel_2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
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
        x1 = xindex
        tmp1 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp2 = tl.full([1], 0.0, tl.float32)
        tl.store(out_ptr2 + (x2), tmp2, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_2 = rand_strided((8192, 48, 128), (6144, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, 655360, 655360, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/91717618ef16106b41f25b4dc1eae4ef4a04701d8036bd632f4f95983ccc2721/inductor_cache/qd/cqdvijami4yx4v2ip4v64hpn2a6qam5dd7sugukt5ikoodtvhxwy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
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
        tmp7 = tl.load(in_ptr1 + (r0_1 + 5120*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 5120*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 5120.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp26, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 5120*x0), tmp26, r0_mask & xmask)
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
buf4 = generate_example_value((8192, 80), (80, 1), 'cuda:0', torch.int32, 0, (8192, 80))
buf10 = generate_example_value((8192, 80), (80, 1), 'cuda:0', torch.int32, 0, (8192, 80))
buf22 = generate_example_value((8192, 48, 128), (6144, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 48, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_0.run(buf4, buf10, buf22, 655360, 655360, 50331648, stream=stream0)
del buf4, buf10, buf22

stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((248320, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (248320, 5120))
arg3_1 = generate_example_value((5120,), (1,), 'cuda:0', torch.bfloat16, 0, (5120,))
buf0 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
buf3 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
buf9 = generate_example_value((8192, 5120), (5120, 1), 'cuda:0', torch.bfloat16, 0, (8192, 5120))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_1.run(arg0_1, arg2_1, arg3_1, buf0, buf3, buf9, 8192, 5120, stream=stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf3, buf9

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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/91717618ef16106b41f25b4dc1eae4ef4a04701d8036bd632f4f95983ccc2721/inductor_cache/ti/cti5rkebmvesd2pzdkqufx4nui27mn67svrxaeoiaoodct27cofh.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(out_ptr0, out_ptr1, out_ptr2, xnumel_0, xnumel_1, xnumel_2, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
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
        x1 = xindex
        tmp1 = tl.full([1], 0, tl.int32)
        tl.store(out_ptr1 + (x1), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x2 = xindex
        tmp2 = tl.full([1], 0.0, tl.float32)
        tl.store(out_ptr2 + (x2), tmp2, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((8192, 80), (80, 1), device='cuda:0', dtype=torch.int32)
    arg_2 = rand_strided((8192, 48, 128), (6144, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, 655360, 655360, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/91717618ef16106b41f25b4dc1eae4ef4a04701d8036bd632f4f95983ccc2721/inductor_cache/qd/cqdvijami4yx4v2ip4v64hpn2a6qam5dd7sugukt5ikoodtvhxwy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
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
        tmp7 = tl.load(in_ptr1 + (r0_1 + 5120*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 5120*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 5120*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 5120.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 5120*x0), tmp26, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 5120*x0), tmp26, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        s72 = arg1_1
        s18 = arg4_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf4 = empty_strided_cuda((128*((127 + s18) // 128), 80), (80, 1), torch.int32)
            buf10 = empty_strided_cuda((128*((127 + s18) // 128), 80), (80, 1), torch.int32)
            buf22 = empty_strided_cuda((s18, 48, 128), (6144, 128, 1), torch.bfloat16)
            buf0 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            buf9 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, add, rms_norm_default, zeros, scaled_fp4_quant_out, zeros_1, scaled_fp4_quant_out_1, zeros_2], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            triton_poi_fused_0_xnumel_0 = 10240*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_1 = 10240*((127 + s18) // 128)
            triton_poi_fused_0_xnumel_2 = 6144*s18
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(buf4, buf10, buf22, triton_poi_fused_0_xnumel_0, triton_poi_fused_0_xnumel_1, triton_poi_fused_0_xnumel_2, stream=stream0)
            # Topologically Sorted Source Nodes: [float_1, add, rms_norm_default, zeros, scaled_fp4_quant_out, zeros_1, scaled_fp4_quant_out_1, zeros_2], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            stream0 = get_raw_stream(0)
            triton_red_fused_1.run(arg0_1, arg2_1, arg3_1, buf0, buf3, buf9, s18, 5120, stream=stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            buf2 = empty_strided_cuda((s18, 2560), (2560, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [float_1, add, rms_norm_default, zeros, scaled_fp4_quant_out], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf3, arg5_1, True, output=buf2, output_scale=buf4)
            del arg5_1
            del buf3
            buf8 = empty_strided_cuda((s18, 2560), (2560, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [float_1, add, rms_norm_default, zeros_1, scaled_fp4_quant_out_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, aten.zeros, _C.scaled_fp4_quant]
            torch.ops._C.scaled_fp4_quant.out(buf9, arg9_1, True, output=buf8, output_scale=buf10)
            del arg9_1
            del buf9
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default], Original ATen: [aten.view]
            buf14 = torch.ops.aten.view.dtype(buf4, torch.float8_e4m3fn)
            buf15 = buf14
            # Topologically Sorted Source Nodes: [t, flashinfer_mm_fp4_default, view_2, t_1], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf16 = torch.ops.vllm.flashinfer_mm_fp4.default(buf2, reinterpret_tensor(arg7_1, (2560, 16384), (1, 2560), 0), aten.view.dtype(buf15, torch.uint8), aten.view.dtype(reinterpret_tensor(arg6_1, (320, 16384), (1, 320), 0), torch.uint8), arg8_1, torch.bfloat16, False, 'cutlass')
            del arg6_1
            del arg7_1
            del arg8_1
            del buf14
            del buf15
            del buf2
            del buf4
            buf17 = buf16
            del buf16
            # Topologically Sorted Source Nodes: [flashinfer_mm_fp4_default_1], Original ATen: [aten.view]
            buf18 = torch.ops.aten.view.dtype(buf10, torch.float8_e4m3fn)
            buf19 = buf18
            # Topologically Sorted Source Nodes: [t_2, flashinfer_mm_fp4_default_1, view_6, t_3], Original ATen: [aten.t, aten.view, vllm.flashinfer_mm_fp4]
            buf20 = torch.ops.vllm.flashinfer_mm_fp4.default(buf8, reinterpret_tensor(arg11_1, (2560, 96), (1, 2560), 0), aten.view.dtype(buf19, torch.uint8), aten.view.dtype(reinterpret_tensor(arg10_1, (320, 128), (1, 320), 0), torch.uint8), arg12_1, torch.bfloat16, False, 'cutlass')
            del arg10_1
            del arg11_1
            del arg12_1
            del buf10
            del buf18
            del buf19
            del buf8
            buf21 = buf20
            del buf20
            buf23 = empty_strided_cuda((s18, 5120), (5120, 1), torch.bfloat16)
        return (reinterpret_tensor(buf17, (s18, 10240), (16384, 1), 0), reinterpret_tensor(buf21, (s18, 48), (96, 1), 0), reinterpret_tensor(buf21, (s18, 48), (96, 1), 48), buf22, reinterpret_tensor(buf17, (s18, 48, 128), (16384, 128, 1), 10240), s18, 128*((127 + s18) // 128), buf23, buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((248320, 5120), (5120, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = 8192
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16384, 320), (320, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg7_1 = rand_strided((16384, 2560), (2560, 1), device='cuda:0', dtype=torch.uint8)
    arg8_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 320), (320, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg11_1 = rand_strided((96, 2560), (2560, 1), device='cuda:0', dtype=torch.uint8)
    arg12_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
