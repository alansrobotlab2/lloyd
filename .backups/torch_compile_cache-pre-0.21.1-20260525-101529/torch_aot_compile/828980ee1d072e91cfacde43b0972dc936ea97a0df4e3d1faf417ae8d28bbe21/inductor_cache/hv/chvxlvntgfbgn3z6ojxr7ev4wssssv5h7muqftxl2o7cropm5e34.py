
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '228E29CD04A1C0E45A497D2BB702521B2A73AF056F742B69E679ABE0C11C5D1C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 404226304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((((x0 + 6144*x1) // 128) % (48*ks0))) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((((x0 + 6144*x1) // 128) % (48*ks0))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + ((x2 % 128)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (128*((((((x0 + 6144*x1) // 128) % (48*ks0))) % 48)) + 16384*(((((((x0 + 6144*x1) // 128) % (48*ks0))) // 48) % ks0)) + ((x0 % 128))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 128.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = -tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.full([1], 1.0, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = (tmp13 / tmp17)
    tmp19 = tmp11 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
