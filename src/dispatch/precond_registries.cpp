#include "precond_registries.hpp"

#include <stdexcept>
#include <vector>

#if defined USE_OPENBLAS
#include "cblas.h"
#endif

namespace tomos {
namespace dispatch {

// Upper-triangular solve: rhs ← op(P)^{-1} * rhs (in-place, column-major).
//
// For double and float on CPU, this delegates to cblas_dtrsv / cblas_strsv
// when OpenBLAS is available.  For __half and any type without a native BLAS
// specialization the solve is performed in double precision via a hand-written
// back-substitution loop.
template <Device dev, typename value_t>
void apply_trsv(TransposeType trans, tomos::Dense& precond, tomos::Dense& rhs) {
    auto vP = as_strided_view<value_t>(precond);
    auto vR = as_strided_view<value_t>(rhs);
    size_t n = vP.size_[0];

    if constexpr (dev == Device::CPU) {
#if defined USE_OPENBLAS
        if constexpr (std::is_same_v<value_t, double>) {
            CBLAS_TRANSPOSE ct = (trans == TransposeType::NO_TRANSPOSE)
                                     ? CblasNoTrans
                                     : CblasTrans;
            cblas_dtrsv(CblasColMajor, CblasUpper, ct, CblasNonUnit, n,
                        vP.values_, vP.lead_dim_, vR.values_, 1);
            return;
        }
        if constexpr (std::is_same_v<value_t, float>) {
            CBLAS_TRANSPOSE ct = (trans == TransposeType::NO_TRANSPOSE)
                                     ? CblasNoTrans
                                     : CblasTrans;
            cblas_strsv(CblasColMajor, CblasUpper, ct, CblasNonUnit, n,
                        vP.values_, vP.lead_dim_, vR.values_, 1);
            return;
        }
#endif
        // Generic fallback: promote to double, back-substitute, cast back.
        std::vector<double> P(n * n), r(n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                P[i + j * n] =
                    static_cast<double>(vP.values_[i + j * vP.lead_dim_]);
        for (size_t i = 0; i < n; ++i)
            r[i] = static_cast<double>(vR.values_[i]);

        if (trans == TransposeType::NO_TRANSPOSE) {
            // Solve U * r = b  (back-substitution).
            for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j)
                    r[i] -= P[i + j * n] * r[j];
                r[i] /= P[i + i * n];
            }
        } else {
            // Solve U^T * r = b  (forward-substitution).
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) r[i] -= P[j + i * n] * r[j];
                r[i] /= P[i + i * n];
            }
        }

        for (size_t i = 0; i < n; ++i)
            vR.values_[i] = static_cast<value_t>(r[i]);
    }
    // CUDA path omitted: apply_trsv requires a cublasHandle_t from Context.
}

const Precond& Precond::get() {
    static const Precond instance;
    return instance;
}

apply_precond_fn Precond::lookup_apply(Device dev,
                                       NumericType numeric_type) const {
    auto fn = apply_[encode_apply(dev, numeric_type)];
    if (!fn)
        throw std::runtime_error(
            "No precond kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

constexpr std::size_t Precond::encode_apply(Device dev,
                                            NumericType numeric_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(numeric_type);
}

Precond::Precond() {
    apply_.fill(nullptr);

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto fp_tag) {
            using FP = typename decltype(fp_tag)::type;
            apply_[encode_apply(dev, numeric_type_of<FP>)] =
                &apply_trsv<dev, FP>;
        });
    });
}

void apply_precond(TransposeType trans, Dense& precond, Dense& rhs) {
    auto dev = precond.get_device();
    auto fn =
        dispatch::Precond::get().lookup_apply(dev, precond.get_value_type());
    fn(trans, precond, rhs);
}

template <Device device, typename value_t>
void apply_precond(TransposeType trans, Dense& precond, value_t* rhs) {
    auto dev = precond.get_device();
    if (dev != device)
        throw std::invalid_argument(
            "apply_precond: precond and rhs must be on the same device.");
    if (precond.get_value_type() != numeric_type_of<value_t>)
        throw std::invalid_argument(
            "apply_precond: precond and rhs must have the same numeric type.");
}

}  // namespace dispatch
}  // namespace tomos