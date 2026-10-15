#ifndef TOMOS_DENSE_REGISTRIES_HPP
#define TOMOS_DENSE_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "storage_registry_base.hpp"
#include "transpose.hpp"

namespace tomos {
namespace dispatch {

// Type-erased function pointer types for registry entries.
using initialize_dense_fn = void (*)(dim<2> size, DenseStorage& source);
using free_dense_fn = void (*)(DenseStorage& source);
using read_dense_fn = void (*)(char* filename, DenseStorage& source);
using copy_dense_fn = void (*)(dim<2> size, DenseStorage& source,
                               DenseStorage& target);
using set_values_dense_fn = void (*)(double val, tomos::Dense& target);
// workspace: caller-owned scratch buffer (float elements, device memory),
// used only by kernels that need one (currently: the FP32_16 "pure_half"
// TRANSPOSE kernel -- see fp32_16_registries.cpp's mv_pure_half_impl).
// nullptr means "no caller-provided workspace" -- implementations that need
// one but weren't given one fall back to their own internal allocation;
// implementations that never need one simply ignore the parameter. Callers
// that own a persistent workspace (e.g. dispatch::lsqr_dense_impl, via
// Workspace::mv_transpose_scratch_) pass it explicitly so no allocation
// happens inside a per-iteration hot loop.
using mv_dense_fn = void (*)(TransposeType trans, double alpha, tomos::Dense& A,
                              tomos::Dense& x, double beta, tomos::Dense& y,
                              float* workspace);
using mm_dense_fn = void (*)(TransposeType transA, TransposeType transB,
                              double alpha, tomos::Dense& A, tomos::Dense& B,
                              double beta, tomos::Dense& C);

struct Dense {
    // Singleton instance accessor.
    static const Dense& get();

    // Lookup functions used for dispatching; throw if no matching function is
    // found.
    initialize_dense_fn lookup_initialize(MatrixFormat format, Device dev,
                                          NumericType numeric_type) const;

    free_dense_fn lookup_free(MatrixFormat format, Device dev,
                              NumericType numeric_type) const;

    read_dense_fn lookup_read(MatrixFormat format, Device dev,
                              NumericType numeric_type) const;

    copy_dense_fn lookup_copy(MatrixFormat source_format,
                              MatrixFormat target_format, Device source_dev,
                              Device target_dev, NumericType source_type,
                              NumericType target_type) const;

    set_values_dense_fn lookup_set_values(Device dev,
                                          NumericType numeric_type) const;

    // compute_type selects the STORE-format-vs-COMPUTE-format axis: type_A
    // gives A's own storage format (native FP32/FP16/BF16, or the packed
    // NumericType::FP32_16 format), and compute_type gives the precision the
    // multiply-accumulate itself should run at where a real choice exists
    // (currently only NumericType::FP32_16 A against FP32 x/y: compute_type
    // == FP16 routes to dispatch::Fp32_16's "pure_half" -- __half-only
    // compute -- tier, compute_type == FP32 routes to its "accurate" --
    // wh+wr reconstructed, float compute -- tier; see dense_registries.cpp).
    // For storage formats with only one real compute choice (every native FP
    // triple), only compute_type == type_A is registered -- e.g. requesting
    // compute_type == FP16 for a native FP32 A throws, same "nothing
    // registered" contract every other lookup already uses. Native FP32/
    // FP32/FP32 on CUDA is special-cased to the tuned mv_fp32_tiled kernel
    // (the "TOMOS FP32" series in the GEVM plots) instead of the generic
    // mv_strided/cublasSgemv path every other native triple still uses.
    mv_dense_fn lookup_mv(Device dev, NumericType type_A, NumericType type_x,
                          NumericType type_y, NumericType compute_type) const;

    // use_workspace picks between two real, independently-tuned kernel
    // families where both exist (currently only NumericType::FP32_16 A
    // against FP32 B/C: no-workspace routes to dispatch::Fp32_16's own
    // "accurate" tier, workspace routes to its "workspace" -- decode-into-
    // scratch-then-native-GEMM -- tier, see dense_registries.cpp). For type
    // combinations with only one real implementation (every native FP
    // triple, via the plain mm_strided kernel), only use_workspace=false is
    // registered -- requesting use_workspace=true throws, since there is no
    // second, genuinely different kernel to return there.
    mm_dense_fn lookup_mm(Device dev, NumericType type_A, NumericType type_B,
                          NumericType type_C, bool use_workspace) const;

private:
    Dense();

    // Encodings that map dispatch arguments to functions stored in dispatch
    // tables.
    static constexpr std::size_t encode(MatrixFormat format, Device dev,
                                        NumericType numeric_type);

    static constexpr std::size_t encode_copy(MatrixFormat source_format,
                                             MatrixFormat target_format,
                                             Device source_dev,
                                             Device target_dev,
                                             NumericType source_type,
                                             NumericType target_type);

    // Number of items in dispatch tables (by function).
    static constexpr std::size_t encode_set_values(Device dev,
                                                   NumericType numeric_type);

    static constexpr std::size_t encode_mv(Device dev, NumericType type_A,
                                           NumericType type_x,
                                           NumericType type_y,
                                           NumericType compute_type);

    static constexpr std::size_t encode_mm(Device dev, NumericType type_A,
                                           NumericType type_B,
                                           NumericType type_C,
                                           bool use_workspace);

    static constexpr std::size_t N = num_registry_formats *
                                     num_registry_devices *
                                     num_registry_numeric_types;

    static constexpr std::size_t N_set_values =
        num_registry_devices * num_registry_numeric_types;

    static constexpr std::size_t N_copy =
        num_registry_formats * num_registry_formats * num_registry_devices *
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types;

    // *num_registry_numeric_types for the compute_type axis -- see
    // lookup_mv's own comment.
    static constexpr std::size_t N_mv =
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_numeric_types *
        num_registry_numeric_types;

    // *2 for the use_workspace bit -- see lookup_mm's own comment.
    static constexpr std::size_t N_mm =
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_numeric_types * 2;

    // Dispatch tables for functions.
    std::array<initialize_dense_fn, N> initialize_{};
    std::array<free_dense_fn, N> free_{};
    std::array<read_dense_fn, N> read_{};
    std::array<copy_dense_fn, N_copy> copy_{};
    std::array<set_values_dense_fn, N_set_values> set_values_{};
    std::array<mv_dense_fn, N_mv> mv_{};
    std::array<mm_dense_fn, N_mm> mm_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_DENSE_REGISTRIES_HPP