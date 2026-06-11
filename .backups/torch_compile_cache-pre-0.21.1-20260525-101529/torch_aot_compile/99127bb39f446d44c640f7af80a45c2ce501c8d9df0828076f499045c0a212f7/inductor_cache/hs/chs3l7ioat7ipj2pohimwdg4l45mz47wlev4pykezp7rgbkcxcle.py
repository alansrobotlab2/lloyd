
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
