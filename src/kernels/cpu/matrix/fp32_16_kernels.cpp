#include "fp32_16_kernels.hpp"

#include "fp32_16_pack.hpp"

namespace tomos {
namespace cpu {

void split_dense_fp32_16(dim<2> size, const float* src, size_t ld_src,
                         uint16_t* wh, uint16_t* wr, size_t ld_dst) {
    for (size_t j = 0; j < size[1]; ++j) {
        for (size_t i = 0; i < size[0]; ++i) {
            fp32_16_split(src[i + j * ld_src], wh[i + j * ld_dst],
                          wr[i + j * ld_dst]);
        }
    }
}

void combine_dense_fp32_16(dim<2> size, const uint16_t* wh, const uint16_t* wr,
                           size_t ld_src, float* dst, size_t ld_dst) {
    for (size_t j = 0; j < size[1]; ++j) {
        for (size_t i = 0; i < size[0]; ++i) {
            dst[i + j * ld_dst] =
                fp32_16_combine(wh[i + j * ld_src], wr[i + j * ld_src]);
        }
    }
}

void mv_fp32_16_accurate(TransposeType trans, double alpha, dim<2> size,
                         const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                         const float* x, size_t ld_x, double beta, float* y,
                         size_t ld_y) {
    size_t m = size[0];
    size_t n = size[1];
    bool y_is_read = (beta != 0.0);
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;

    for (size_t i = 0; i < rows_op; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < cols_op; ++j) {
            size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
            size_t idx = a_row + a_col * ld_a;
            acc += static_cast<double>(fp32_16_combine(wh[idx], wr[idx])) *
                   static_cast<double>(x[j]);
        }
        // x/y are single-column vectors: their lead dimension (ld_x/ld_y)
        // does not enter element addressing, matching cpu::mv_strided's own
        // plain x[j]/y[i] indexing.
        y[i] = y_is_read
                   ? static_cast<float>(alpha * acc +
                                        beta * static_cast<double>(y[i]))
                   : static_cast<float>(alpha * acc);
    }
}

void mm_fp32_16_accurate(TransposeType transA, TransposeType transB,
                         double alpha, dim<2> size_A, const uint16_t* wh,
                         const uint16_t* wr, size_t ld_a, dim<2> size_B,
                         const float* B, size_t ld_b, double beta, float* C,
                         size_t ld_c) {
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            C[i + j * ld_c] =
                static_cast<float>(beta * static_cast<double>(C[i + j * ld_c]));

    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            double b_val = (transB == TransposeType::NO_TRANSPOSE)
                              ? static_cast<double>(B[l + j * ld_b])
                              : static_cast<double>(B[j + l * ld_b]);
            double scale = alpha * b_val;
            for (size_t i = 0; i < m; ++i) {
                size_t a_row = (transA == TransposeType::NO_TRANSPOSE) ? i : l;
                size_t a_col = (transA == TransposeType::NO_TRANSPOSE) ? l : i;
                size_t idx = a_row + a_col * ld_a;
                double a_val =
                    static_cast<double>(fp32_16_combine(wh[idx], wr[idx]));
                C[i + j * ld_c] = static_cast<float>(
                    static_cast<double>(C[i + j * ld_c]) + scale * a_val);
            }
        }
    }
}

void mv_fp32_16_fast(TransposeType trans, double alpha, dim<2> size,
                     const uint16_t* wh, size_t ld_a, const float* x,
                     size_t ld_x, double beta, float* y, size_t ld_y) {
    size_t m = size[0];
    size_t n = size[1];
    bool y_is_read = (beta != 0.0);
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;

    for (size_t i = 0; i < rows_op; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < cols_op; ++j) {
            size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
            size_t idx = a_row + a_col * ld_a;
            acc += static_cast<double>(fp32_16_half_only(wh[idx])) *
                   static_cast<double>(x[j]);
        }
        y[i] = y_is_read
                   ? static_cast<float>(alpha * acc +
                                        beta * static_cast<double>(y[i]))
                   : static_cast<float>(alpha * acc);
    }
}

void mm_fp32_16_fast(TransposeType transA, TransposeType transB, double alpha,
                     dim<2> size_A, const uint16_t* wh, size_t ld_a,
                     dim<2> size_B, const float* B, size_t ld_b, double beta,
                     float* C, size_t ld_c) {
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            C[i + j * ld_c] =
                static_cast<float>(beta * static_cast<double>(C[i + j * ld_c]));

    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            double b_val = (transB == TransposeType::NO_TRANSPOSE)
                              ? static_cast<double>(B[l + j * ld_b])
                              : static_cast<double>(B[j + l * ld_b]);
            double scale = alpha * b_val;
            for (size_t i = 0; i < m; ++i) {
                size_t a_row = (transA == TransposeType::NO_TRANSPOSE) ? i : l;
                size_t a_col = (transA == TransposeType::NO_TRANSPOSE) ? l : i;
                size_t idx = a_row + a_col * ld_a;
                double a_val = static_cast<double>(fp32_16_half_only(wh[idx]));
                C[i + j * ld_c] = static_cast<float>(
                    static_cast<double>(C[i + j * ld_c]) + scale * a_val);
            }
        }
    }
}

void mv_fp32_16_pure_half(TransposeType trans, double alpha, dim<2> size,
                          const uint16_t* wh, size_t ld_a, const float* x,
                          size_t ld_x, double beta, float* y, size_t ld_y) {
    size_t m = size[0];
    size_t n = size[1];
    bool y_is_read = (beta != 0.0);
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    const __half h_zero = static_cast<__half>(0.0f);

    for (size_t i = 0; i < rows_op; ++i) {
        __half acc = h_zero;
        for (size_t j = 0; j < cols_op; ++j) {
            size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
            size_t idx = a_row + a_col * ld_a;
            __half a_val = fp32_16_half_only(wh[idx]);
            __half x_val = static_cast<__half>(x[j]);
            acc = acc + a_val * x_val;
        }
        double acc_d = static_cast<double>(static_cast<float>(acc));
        y[i] = y_is_read
                   ? static_cast<float>(alpha * acc_d +
                                        beta * static_cast<double>(y[i]))
                   : static_cast<float>(alpha * acc_d);
    }
}

void mm_fp32_16_pure_half(TransposeType transA, TransposeType transB,
                          double alpha, dim<2> size_A, const uint16_t* wh,
                          size_t ld_a, dim<2> size_B, const float* B,
                          size_t ld_b, double beta, float* C, size_t ld_c) {
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];
    const __half h_zero = static_cast<__half>(0.0f);

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            __half acc = h_zero;
            for (size_t l = 0; l < k; ++l) {
                size_t a_row = (transA == TransposeType::NO_TRANSPOSE) ? i : l;
                size_t a_col = (transA == TransposeType::NO_TRANSPOSE) ? l : i;
                size_t idx = a_row + a_col * ld_a;
                __half a_val = fp32_16_half_only(wh[idx]);
                float b_f = (transB == TransposeType::NO_TRANSPOSE)
                              ? B[l + j * ld_b]
                              : B[j + l * ld_b];
                __half b_val = static_cast<__half>(b_f);
                acc = acc + a_val * b_val;
            }
            double acc_d = static_cast<double>(static_cast<float>(acc));
            C[i + j * ld_c] = static_cast<float>(
                alpha * acc_d + beta * static_cast<double>(C[i + j * ld_c]));
        }
    }
}

}  // namespace cpu
}  // namespace tomos
