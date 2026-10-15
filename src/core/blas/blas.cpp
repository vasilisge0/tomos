#include <type_traits>

#include "blas.hpp"
#include "dense_registries.hpp"
#include "fp32_16_registries.hpp"
#include "matrix.hpp"
#include "sparse_registries.hpp"

namespace tomos {

template <typename value_t>
void set_values(value_t value, Dense& target) {
    auto fn = dispatch::Dense::get().lookup_set_values(target.get_device(),
                                                       target.get_value_type());
    fn(static_cast<double>(value), target);
}

template void set_values<double>(double value, Dense& target);
template void set_values<float>(float value, Dense& target);
template void set_values<__half>(__half value, Dense& target);
template void set_values<__nv_bfloat16>(__nv_bfloat16 value, Dense& target);

Status mv_apply(Device dev, TransposeType transpose, double alpha, Dense& A,
                Dense& x, double beta, Dense& y, NumericType compute_type,
                float* workspace) {
    auto fn = dispatch::Dense::get().lookup_mv(
        dev, A.get_value_type(), x.get_value_type(), y.get_value_type(),
        compute_type);
    fn(transpose, alpha, A, x, beta, y, workspace);
    return SUCCESS;
}

Status mv_apply(Device dev, TransposeType transpose, double alpha, Dense& A,
                Dense& x, double beta, Dense& y) {
    // compute_type == A.get_value_type() means "no override": for every
    // native FP storage type this reproduces the exact prior behavior
    // (including the CUDA FP32/FP32/FP32 case, which now resolves to the
    // tuned mv_fp32_tiled kernel instead of cublasSgemv -- see
    // dense_registries.cpp's lookup_mv registration -- this is "TOMOS
    // FP32" in the GEVM plots). NumericType::FP32_16-stored A has no
    // compute_type == FP32_16 slot registered (FP32_16 isn't itself a
    // single arithmetic C++ type an mv kernel can compute in) -- default
    // it to the FP16 ("pure_half") tier instead of throwing, since that's
    // the tuned, cuBLAS-competitive kernel the plots label "TOMOS FP16"
    // (see dense_registries.cpp's FP32_16 delegation block / Fp32_16::
    // lookup_mv_pure_half), rather than falling through to any generic/
    // naive kernel. Callers that need the OTHER FP32_16 compute tier
    // (FP32, "accurate"/hybrid -- Fp32_16::lookup_mv_accurate) must use the
    // explicit-compute_type overload above instead of this one.
    NumericType a_type = A.get_value_type();
    NumericType compute_type =
        (a_type == NumericType::FP32_16) ? NumericType::FP16 : a_type;
    return mv_apply(dev, transpose, alpha, A, x, beta, y, compute_type);
}

template <typename value_t>
Status mv_apply(Device dev, TransposeType transpose, double alpha, Dense& A,
                value_t* x, double beta, value_t* y) {
    auto fn = dispatch::Dense::get().lookup_mv(dev, A.get_value_type(), x, y);
    fn(transpose, alpha, A, x, beta, y);
    return SUCCESS;
}

// Variant overloads

Status mv_apply(Device dev, TransposeType transpose, double alpha,
                std::variant<Dense*, Sparse*>& A, Dense& x, double beta, Dense& y) {
    std::visit([&](auto* m) { mv_apply(dev, transpose, alpha, *m, x, beta, y); }, A);
    return SUCCESS;
}

Status mv_apply(Device dev, TransposeType transpose, double alpha,
                std::variant<Dense*, Sparse*>& A, Dense& x, double beta,
                Dense& y, NumericType compute_type, float* workspace) {
    std::visit([&](auto* m) {
        using T = std::decay_t<decltype(*m)>;
        if constexpr (std::is_same_v<T, Dense>) {
            mv_apply(dev, transpose, alpha, *m, x, beta, y, compute_type, workspace);
        } else {
            // Sparse has no compute_type axis (see sparse_registries.cpp's
            // lookup_mv) -- fall back to the plain overload.
            mv_apply(dev, transpose, alpha, *m, x, beta, y);
        }
    }, A);
    return SUCCESS;
}

void precond_apply(Device dev, TransposeType transpose, double alpha,
                   std::variant<Dense*, Sparse*, std::monostate>& P,
                   Dense& x, double beta, Dense& y) {
    std::visit([&](auto& p) {
        using T = std::decay_t<decltype(p)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
            mv_apply(dev, transpose, alpha, *p, x, beta, y);
    }, P);
}

Status mm_apply(Device dev, TransposeType transA, TransposeType transB,
                double alpha, Dense& A, Dense& B, double beta, Dense& C,
                bool use_workspace) {
    auto fn = dispatch::Dense::get().lookup_mm(
        dev, A.get_value_type(), B.get_value_type(), C.get_value_type(),
        use_workspace);
    fn(transA, transB, alpha, A, B, beta, C);
    return SUCCESS;
}

// Sparse API

Status mv_apply(Device dev, TransposeType transpose, double alpha, Sparse& A,
                Dense& x, double beta, Dense& y) {
    auto fn = dispatch::Sparse::get().lookup_mv(
        dev, A.get_value_type(), x.get_value_type(), y.get_value_type(),
        A.get_index_type());
    fn(transpose, alpha, A, x, beta, y);
    return SUCCESS;
}

template <typename value_t>
void set_values(value_t value, Sparse& target) {
    auto fn = dispatch::Sparse::get().lookup_set_values(
        target.get_device(), target.get_value_type());
    fn(static_cast<double>(value), target);
}

template void set_values<double>(double value, Sparse& target);
template void set_values<float>(float value, Sparse& target);
template void set_values<__half>(__half value, Sparse& target);
template void set_values<__nv_bfloat16>(__nv_bfloat16 value, Sparse& target);

// Split-precision (NumericType::FP32_16) GEMV/GEMM API.

Status split_fp32_16(Dense& source, Dense& target) {
    auto fn = dispatch::Fp32_16::get().lookup_split(target.get_device());
    fn(source, target);
    return SUCCESS;
}

Status combine_fp32_16(Dense& source, Dense& target) {
    auto fn = dispatch::Fp32_16::get().lookup_combine(source.get_device());
    fn(source, target);
    return SUCCESS;
}

Status mv_apply_fp32_16_pure_half(Device dev, TransposeType transpose,
                                  double alpha, Dense& A, Dense& x,
                                  double beta, Dense& y) {
    auto fn = dispatch::Fp32_16::get().lookup_mv_pure_half(dev);
    fn(transpose, alpha, A, x, beta, y, nullptr);
    return SUCCESS;
}

Status mv_apply_fp32_16_fast(Device dev, TransposeType transpose,
                             double alpha, Dense& A, Dense& x, double beta,
                             Dense& y) {
    auto fn = dispatch::Fp32_16::get().lookup_mv_fast(dev);
    fn(transpose, alpha, A, x, beta, y, nullptr);
    return SUCCESS;
}

Status mv_apply_fp32_16_ext(Device dev, TransposeType transpose, double alpha,
                            Dense& A, Dense& x, double beta, Dense& y) {
    auto fn = dispatch::Fp32_16::get().lookup_mv_accurate(dev);
    fn(transpose, alpha, A, x, beta, y, nullptr);
    return SUCCESS;
}

Status mm_apply_fp32_16_pure_half(Device dev, TransposeType transA,
                                  TransposeType transB, double alpha,
                                  Dense& A, Dense& B, double beta, Dense& C) {
    auto fn = dispatch::Fp32_16::get().lookup_mm_pure_half(dev);
    fn(transA, transB, alpha, A, B, beta, C);
    return SUCCESS;
}

Status mm_apply_fp32_16_fast(Device dev, TransposeType transA,
                             TransposeType transB, double alpha, Dense& A,
                             Dense& B, double beta, Dense& C) {
    auto fn = dispatch::Fp32_16::get().lookup_mm_fast(dev);
    fn(transA, transB, alpha, A, B, beta, C);
    return SUCCESS;
}

Status mm_apply_fp32_16_ext(Device dev, TransposeType transA,
                            TransposeType transB, double alpha, Dense& A,
                            Dense& B, double beta, Dense& C) {
    auto fn = dispatch::Fp32_16::get().lookup_mm_accurate(dev);
    fn(transA, transB, alpha, A, B, beta, C);
    return SUCCESS;
}

}  // namespace tomos