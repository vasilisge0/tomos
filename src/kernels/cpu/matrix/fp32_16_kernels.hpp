#ifndef TOMOS_FP32_16_CPU_KERNELS_HPP
#define TOMOS_FP32_16_CPU_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#include "dimensions.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cpu {

// Splits a plain float32 dense matrix into the wh/wr halves of the
// NumericType::FP32_16 representation (see fp32_16_pack.hpp).
void split_dense_fp32_16(dim<2> size, const float* src, size_t ld_src,
                         uint16_t* wh, uint16_t* wr, size_t ld_dst);

// Reconstructs a plain float32 dense matrix from the wh/wr halves.
void combine_dense_fp32_16(dim<2> size, const uint16_t* wh, const uint16_t* wr,
                           size_t ld_src, float* dst, size_t ld_dst);

// y = alpha * op(A) * x + beta * y, where A is stored as wh/wr (see
// fp32_16_pack.hpp) and combined element-by-element for ~fp32 accuracy.
void mv_fp32_16_accurate(TransposeType trans, double alpha, dim<2> size,
                         const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                         const float* x, size_t ld_x, double beta, float* y,
                         size_t ld_y);

// C = alpha * op(A) * op(B) + beta * C, where A is stored as wh/wr.
void mm_fp32_16_accurate(TransposeType transA, TransposeType transB,
                         double alpha, dim<2> size_A, const uint16_t* wh,
                         const uint16_t* wr, size_t ld_a, dim<2> size_B,
                         const float* B, size_t ld_b, double beta, float* C,
                         size_t ld_c);

// Same as mv_fp32_16_accurate, but using only wh (fp32_16_half_only decode,
// see fp32_16_pack.hpp): the dot product accumulates in double, giving
// fp16-level per-element accuracy but avoiding fp16 accumulation error.
void mv_fp32_16_fast(TransposeType trans, double alpha, dim<2> size,
                     const uint16_t* wh, size_t ld_a, const float* x,
                     size_t ld_x, double beta, float* y, size_t ld_y);

// Same as mm_fp32_16_accurate, but using only wh, accumulating in double.
void mm_fp32_16_fast(TransposeType transA, TransposeType transB, double alpha,
                     dim<2> size_A, const uint16_t* wh, size_t ld_a,
                     dim<2> size_B, const float* B, size_t ld_b, double beta,
                     float* C, size_t ld_c);

// Same as mv_fp32_16_fast, but the entire reduction (multiply and
// accumulate) happens in __half arithmetic, not promoted to a wider type --
// genuine fp16-only compute, only widened to float for the final alpha/beta
// scale-and-store.
void mv_fp32_16_pure_half(TransposeType trans, double alpha, dim<2> size,
                          const uint16_t* wh, size_t ld_a, const float* x,
                          size_t ld_x, double beta, float* y, size_t ld_y);

// Same as mm_fp32_16_fast, but the entire reduction happens in __half
// arithmetic (see mv_fp32_16_pure_half).
void mm_fp32_16_pure_half(TransposeType transA, TransposeType transB,
                          double alpha, dim<2> size_A, const uint16_t* wh,
                          size_t ld_a, dim<2> size_B, const float* B,
                          size_t ld_b, double beta, float* C, size_t ld_c);

}  // namespace cpu
}  // namespace tomos

#endif  // TOMOS_FP32_16_CPU_KERNELS_HPP
