#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
