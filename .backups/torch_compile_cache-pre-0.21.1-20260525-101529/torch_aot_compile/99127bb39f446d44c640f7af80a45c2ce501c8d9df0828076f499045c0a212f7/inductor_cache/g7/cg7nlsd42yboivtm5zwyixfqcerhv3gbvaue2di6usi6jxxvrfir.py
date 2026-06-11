# AOT ID: ['69_inference']
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


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/99127bb39f446d44c640f7af80a45c2ce501c8d9df0828076f499045c0a212f7/inductor_cache/hs/chs3l7ioat7ipj2pohimwdg4l45mz47wlev4pykezp7rgbkcxcle.py
# Topologically Sorted Source Nodes: [ge, gather_indices, prev_drafts, gt, participating, prev_computed, valid_counts, corrected, where, where_1], Original ATen: [aten.ge, aten.clamp, aten.index, aten.gt, aten.bitwise_and, aten.add, aten.where, aten.copy_]
# Source node to ATen node mapping:
#   corrected => add_14
#   gather_indices => clamp_min
#   ge => ge
#   gt => gt
#   participating => bitwise_and
#   prev_computed => index_1
#   prev_drafts => index_2
#   valid_counts => index
#   where => where
#   where_1 => where_1
# Graph fragment:
#   %arg1_1 : Tensor "i64[s34][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg6_1 : Tensor "i32[s13][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg5_1 : Tensor "i32[s13][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg3_1 : Tensor "i32[s43][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg7_1 : Tensor "i32[s34][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %copy__1 : Tensor "i32[s34][1]cuda:0" = PlaceHolder[target=copy__1]
#   %where_1 : Tensor "i32[s34][1]cuda:0" = PlaceHolder[target=where_1]
#   %ge : Tensor "b8[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %clamp_min : Tensor "i64[s34][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg1_1, 0), kwargs = {})
#   %index_2 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg6_1, [%clamp_min]), kwargs = {})
#   %gt : Tensor "b8[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%index_2, 0), kwargs = {})
#   %bitwise_and : Tensor "b8[s34][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt), kwargs = {})
#   %index_1 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg5_1, [%clamp_min]), kwargs = {})
#   %index : Tensor "i32[s34][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg3_1, [%clamp_min]), kwargs = {})
#   %add_14 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_1, %index), kwargs = {})
#   %where : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %add_14, %arg7_1), kwargs = {})
#   %where_1 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %index, %arg9_1), kwargs = {})
#   %copy__1 : Tensor "i32[s34][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %where_1), kwargs = {})
#   return %where,%where_1,%buf5
triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0 = async_compile.triton('triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i32', 'in_ptr2': '*i32', 'in_ptr3': '*i32', 'in_ptr4': '*i32', 'in_ptr5': '*i32', 'out_ptr0': '*i32', 'out_ptr2': '*i32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0', 'mutated_arg_names': ['in_ptr5', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp20 = tl.load(in_ptr4 + (x0), xmask)
    tmp22 = tl.load(in_ptr5 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = ks0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp7 < ks0")
    tmp9 = tl.load(in_ptr1 + (tmp7), xmask, eviction_policy='evict_last')
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp9 > tmp10
    tmp12 = tmp2 & tmp11
    tmp13 = tl.load(in_ptr2 + (tmp7), xmask, eviction_policy='evict_last')
    tmp14 = ks1
    tmp15 = tmp3 + tmp14
    tmp16 = tl.where(tmp6, tmp15, tmp3)
    tl.device_assert(((0 <= tmp16) & (tmp16 < ks1)) | ~(xmask), "index out of bounds: 0 <= tmp16 < ks1")
    tmp18 = tl.load(in_ptr3 + (tmp16), xmask, eviction_policy='evict_last')
    tmp19 = tmp13 + tmp18
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp23 = tl.where(tmp12, tmp18, tmp22)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: /home/alansrobotlab/.cache/vllm/torch_compile_cache/torch_aot_compile/99127bb39f446d44c640f7af80a45c2ce501c8d9df0828076f499045c0a212f7/inductor_cache/qd/cqdljunmdeddev4fo7fopzaiewtcc4ujhpmg34brdd4vpj4i227v.py
# Topologically Sorted Source Nodes: [ge, gather_indices, prev_drafts, gt, participating, prev_computed, valid_counts, corrected, where], Original ATen: [aten.ge, aten.clamp, aten.index, aten.gt, aten.bitwise_and, aten.add, aten.where]
# Source node to ATen node mapping:
#   corrected => add_14
#   gather_indices => clamp_min
#   ge => ge
#   gt => gt
#   participating => bitwise_and
#   prev_computed => index_1
#   prev_drafts => index_2
#   valid_counts => index
#   where => where
# Graph fragment:
#   %where : Tensor "i32[s34][1]cuda:0" = PlaceHolder[target=where]
#   %arg5_1 : Tensor "i32[s13][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %ge : Tensor "b8[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %clamp_min : Tensor "i64[s34][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg1_1, 0), kwargs = {})
#   %index_2 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg6_1, [%clamp_min]), kwargs = {})
#   %gt : Tensor "b8[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%index_2, 0), kwargs = {})
#   %bitwise_and : Tensor "b8[s34][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt), kwargs = {})
#   %index_1 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg5_1, [%clamp_min]), kwargs = {})
#   %index : Tensor "i32[s34][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg3_1, [%clamp_min]), kwargs = {})
#   %add_14 : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_1, %index), kwargs = {})
#   %where : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %add_14, %arg7_1), kwargs = {})
#   %slice_tensor : Tensor "i32[s34][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg5_1, 0, 0, %arg8_1), kwargs = {})
#   %copy__default : Tensor "i32[s34][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%slice_tensor, %where), kwargs = {})
#   return %buf3
triton_poi_fused_add_bitwise_and_clamp_ge_gt_index_where_1 = async_compile.triton('triton_poi_fused_add_bitwise_and_clamp_ge_gt_index_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_clamp_ge_gt_index_where_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_clamp_ge_gt_index_where_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s57 = arg0_1
        s43 = arg2_1
        s13 = arg4_1
        s34 = arg8_1
        assert_size_stride(arg1_1, (s34, ), (1, ))
        assert_size_stride(arg3_1, (s43, ), (1, ))
        assert_size_stride(arg5_1, (s13, ), (1, ))
        assert_size_stride(arg6_1, (s13, ), (1, ))
        assert_size_stride(arg7_1, (s34, ), (1, ))
        assert_size_stride(arg9_1, (s34, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((s34, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [ge, gather_indices, prev_drafts, gt, participating, prev_computed, valid_counts, corrected, where, where_1], Original ATen: [aten.ge, aten.clamp, aten.index, aten.gt, aten.bitwise_and, aten.add, aten.where, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0.run(arg1_1, arg6_1, arg5_1, arg3_1, arg7_1, arg9_1, buf2, arg9_1, s13, s43, s34, stream=stream0)
            del arg1_1
            del arg3_1
            del arg6_1
            del arg7_1
            del arg9_1
            # Topologically Sorted Source Nodes: [ge, gather_indices, prev_drafts, gt, participating, prev_computed, valid_counts, corrected, where], Original ATen: [aten.ge, aten.clamp, aten.index, aten.gt, aten.bitwise_and, aten.add, aten.where]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_bitwise_and_clamp_ge_gt_index_where_1.run(buf2, arg5_1, s34, stream=stream0)
            del arg5_1
            del buf2
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 2
    arg1_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg2_1 = 2
    arg3_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg4_1 = 8
    arg5_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg6_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg7_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg8_1 = 2
    arg9_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
