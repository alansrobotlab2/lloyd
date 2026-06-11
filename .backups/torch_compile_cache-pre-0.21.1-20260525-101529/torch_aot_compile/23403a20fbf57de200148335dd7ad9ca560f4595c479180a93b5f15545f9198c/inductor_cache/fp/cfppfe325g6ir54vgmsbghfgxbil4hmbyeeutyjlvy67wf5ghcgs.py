
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i32', 'in_ptr2': '*i32', 'in_ptr3': '*i32', 'in_ptr4': '*i32', 'in_ptr5': '*i32', 'out_ptr0': '*i32', 'out_ptr2': '*i32', 'ks0': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0', 'mutated_arg_names': ['in_ptr5', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr4 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp25 = tl.load(in_ptr5 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = ks0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < ks0), "index out of bounds: 0 <= tmp8 < ks0")
    tmp10 = tl.load(in_ptr1 + (tmp8), None, eviction_policy='evict_last')
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 > tmp11
    tmp13 = tmp3 & tmp12
    tmp14 = tl.load(in_ptr2 + (tmp8), None, eviction_policy='evict_last')
    tmp15 = tl.full([XBLOCK], 1, tl.int32)
    tmp16 = tmp4 + tmp15
    tmp17 = tl.where(tmp7, tmp16, tmp4)
    tl.device_assert((0 <= tmp17) & (tmp17 < 1), "index out of bounds: 0 <= tmp17 < 1")
    tmp21 = tmp14 + tmp20
    tmp24 = tl.where(tmp13, tmp21, tmp23)
    tmp27 = tl.where(tmp13, tmp20, tmp26)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp24, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp27, None)
