// CPU (Device::CPU) implementations of blas.hpp's statically-dispatched
// vector BLAS-1 primitives (norm2/scale/axpy). These are currently uncalled
// anywhere in the codebase (real LSQR/precond code paths go through
// mv_apply/precond_apply and their own dispatch::* registries instead), but
// that's a separate question from whether this file honors blas.hpp's
// declared contract -- it now does, for every declared function, matching
// each signature exactly. The LAPACK-based geqrf2/trtri/extract_upper_triang
// live in lapack.cpp instead: mixing <cblas.h> and <lapacke.h> in the same
// translation unit causes conflicting Fortran-symbol redeclarations (their
// bundled headers disagree on integer typedefs depending on include order),
// so -- matching how the rest of the codebase already keeps them apart
// (precond_registries.cpp: cblas only; sketch_qr.cpp: lapacke only) -- this
// file only ever includes cblas.h.

#if defined USE_OPENBLAS
#include "cblas.h"
typedef blasint BLAS_INT;
#endif

#include <cmath>
#include <type_traits>

#include "blas.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {

namespace {

// float-promoted generic fallback for types CBLAS doesn't natively support
// (__half, __nv_bfloat16) -- same approach as dispatch::apply_trsv's own
// "generic fallback" (precond_registries.cpp).
template <typename value_t>
double to_double(value_t v) {
    return static_cast<double>(v);
}

}  // namespace

// ---------------------------------------------------------------------------
// norm2
// ---------------------------------------------------------------------------

template <Device device, typename value_t>
Status norm2(size_t num_elems, value_t* vec, size_t inc, value_t* norm) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    if constexpr (std::is_same_v<value_t, double>) {
#if defined USE_OPENBLAS
        *norm = cblas_dnrm2((BLAS_INT)num_elems, vec, (BLAS_INT)inc);
        return SUCCESS;
#endif
    } else if constexpr (std::is_same_v<value_t, float>) {
#if defined USE_OPENBLAS
        *norm = cblas_snrm2((BLAS_INT)num_elems, vec, (BLAS_INT)inc);
        return SUCCESS;
#endif
    }
    // Generic fallback (also used for __half/__nv_bfloat16, which CBLAS has
    // no routine for at all).
    double acc = 0.0;
    for (size_t i = 0; i < num_elems; ++i) {
        double v = to_double(vec[i * inc]);
        acc += v * v;
    }
    *norm = static_cast<value_t>(std::sqrt(acc));
    return SUCCESS;
}

template Status norm2<Device::CPU, double>(size_t, double*, size_t, double*);
template Status norm2<Device::CPU, float>(size_t, float*, size_t, float*);
template Status norm2<Device::CPU, __half>(size_t, __half*, size_t, __half*);
template Status norm2<Device::CPU, __nv_bfloat16>(size_t, __nv_bfloat16*,
                                                  size_t, __nv_bfloat16*);

// ---------------------------------------------------------------------------
// scale: vec <- alpha * vec
// ---------------------------------------------------------------------------

template <Device device, typename value_t>
Status scale(size_t num_elems, value_t alpha, value_t* vec, size_t inc) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    if constexpr (std::is_same_v<value_t, double>) {
#if defined USE_OPENBLAS
        cblas_dscal((BLAS_INT)num_elems, alpha, vec, (BLAS_INT)inc);
        return SUCCESS;
#endif
    } else if constexpr (std::is_same_v<value_t, float>) {
#if defined USE_OPENBLAS
        cblas_sscal((BLAS_INT)num_elems, alpha, vec, (BLAS_INT)inc);
        return SUCCESS;
#endif
    }
    double a = to_double(alpha);
    for (size_t i = 0; i < num_elems; ++i)
        vec[i * inc] = static_cast<value_t>(to_double(vec[i * inc]) * a);
    return SUCCESS;
}

template Status scale<Device::CPU, double>(size_t, double, double*, size_t);
template Status scale<Device::CPU, float>(size_t, float, float*, size_t);
template Status scale<Device::CPU, __half>(size_t, __half, __half*, size_t);
template Status scale<Device::CPU, __nv_bfloat16>(size_t, __nv_bfloat16,
                                                  __nv_bfloat16*, size_t);

// ---------------------------------------------------------------------------
// axpy: y <- alpha * x + y
// ---------------------------------------------------------------------------

template <Device device, typename value_type>
Status axpy(size_t num_elems, value_type alpha, value_type* x, size_t incx,
           value_type* y, size_t incy) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    if constexpr (std::is_same_v<value_type, double>) {
#if defined USE_OPENBLAS
        cblas_daxpy((BLAS_INT)num_elems, alpha, x, (BLAS_INT)incx, y,
                   (BLAS_INT)incy);
        return SUCCESS;
#endif
    } else if constexpr (std::is_same_v<value_type, float>) {
#if defined USE_OPENBLAS
        cblas_saxpy((BLAS_INT)num_elems, alpha, x, (BLAS_INT)incx, y,
                   (BLAS_INT)incy);
        return SUCCESS;
#endif
    }
    double a = to_double(alpha);
    for (size_t i = 0; i < num_elems; ++i) {
        double v = to_double(x[i * incx]) * a + to_double(y[i * incy]);
        y[i * incy] = static_cast<value_type>(v);
    }
    return SUCCESS;
}

template Status axpy<Device::CPU, double>(size_t, double, double*, size_t,
                                         double*, size_t);
template Status axpy<Device::CPU, float>(size_t, float, float*, size_t, float*,
                                        size_t);
template Status axpy<Device::CPU, __half>(size_t, __half, __half*, size_t,
                                         __half*, size_t);
template Status axpy<Device::CPU, __nv_bfloat16>(size_t, __nv_bfloat16,
                                                 __nv_bfloat16*, size_t,
                                                 __nv_bfloat16*, size_t);

}  // namespace tomos
