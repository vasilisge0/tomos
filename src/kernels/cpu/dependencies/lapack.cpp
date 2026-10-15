// CPU (Device::CPU) implementations of blas.hpp's statically-dispatched
// LAPACK primitives (geqrf2/trtri/extract_upper_triang). Kept separate from
// blas.cpp: mixing <cblas.h> and <lapacke.h> in one translation unit causes
// conflicting Fortran-symbol redeclarations, so this file only ever includes
// lapacke.h -- see blas.cpp's own header comment for the full explanation.
// These are currently uncalled anywhere in the codebase, but that's a
// separate question from whether this file honors blas.hpp's declared
// contract -- it now does, for every declared function, matching each
// signature exactly.

#include <lapacke.h>

#include <stdexcept>
#include <type_traits>

#include "blas.hpp"
#include "context.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {

// ---------------------------------------------------------------------------
// geqrf2: QR factorization (in-place, column-major -- Householder reflectors
// below the diagonal, R in the upper triangle, exactly as generate_sketch_qr_
// precond_cpu (sketch_qr.cpp) already relies on for its own, separate LAPACKE
// call). Delegates to LAPACKE's _work variant, which takes the same explicit
// work/lwork buffer this declared signature expects, unlike the simpler
// LAPACKE_?geqrf wrapper sketch_qr.cpp itself uses. No half-precision
// support: LAPACK has no half-precision QR routine (see sketch_qr.cpp's own
// comment on this), so only double/float are instantiated.
// ---------------------------------------------------------------------------

template <Device device, typename value_t>
Status geqrf2(int m, int n, value_t* A, int lda, value_t* tau, value_t* work,
             int lwork, int* info) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    if constexpr (std::is_same_v<value_t, double>) {
        *info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, lda, tau, work,
                                    lwork);
    } else if constexpr (std::is_same_v<value_t, float>) {
        *info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, A, lda, tau, work,
                                    lwork);
    } else {
        throw std::invalid_argument(
            "geqrf2: no LAPACK QR routine exists for this type (only double/"
            "float are supported).");
    }
    return (*info == 0) ? SUCCESS : FAILURE;
}

template Status geqrf2<Device::CPU, double>(int, int, double*, int, double*,
                                           double*, int, int*);
template Status geqrf2<Device::CPU, float>(int, int, float*, int, float*,
                                          float*, int, int*);

// ---------------------------------------------------------------------------
// trtri: invA <- op(A)^{-1}, triangular inversion. Unlike LAPACKE_?trtri
// (in-place only), this declared signature keeps A and invA as separate
// buffers with independent leading dimensions, so invA is first filled with
// A's own n x n block (only the requested triangle needs to be valid on
// input/is meaningful on output, but the full block is copied for
// simplicity), then inverted in place via LAPACKE -- same routine sketch_qr.
// cpp's own trtri_lapacke helper uses.
// ---------------------------------------------------------------------------

template <Device device, typename value_t>
Status trtri(char uplo, char diag, int n, value_t* A, int lda, value_t* invA,
            size_t ldiv, int* info) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            invA[i + j * static_cast<int>(ldiv)] = A[i + j * lda];

    if constexpr (std::is_same_v<value_t, double>) {
        *info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, uplo, diag, n, invA,
                              static_cast<int>(ldiv));
    } else if constexpr (std::is_same_v<value_t, float>) {
        *info = LAPACKE_strtri(LAPACK_COL_MAJOR, uplo, diag, n, invA,
                              static_cast<int>(ldiv));
    } else {
        throw std::invalid_argument(
            "trtri: no LAPACK triangular-inverse routine exists for this "
            "type (only double/float are supported).");
    }
    return (*info == 0) ? SUCCESS : FAILURE;
}

template Status trtri<Device::CPU, double>(char, char, int, double*, int,
                                          double*, size_t, int*);
template Status trtri<Device::CPU, float>(char, char, int, float*, int, float*,
                                         size_t, int*);

// ---------------------------------------------------------------------------
// extract_upper_triang: zero out A's strictly-lower triangle in place,
// leaving only its upper triangle (including the diagonal). Exactly the
// operation generate_sketch_qr_precond_cpu (sketch_qr.cpp) already performs
// by hand after geqrf, to strip the Householder reflectors packed below the
// diagonal. context is accepted (matching the declared signature -- likely
// reserved for a future CUDA counterpart needing a stream/handle) but unused
// on CPU.
// ---------------------------------------------------------------------------

template <Device device, typename value_t>
Status extract_upper_triang(Context& context, int n, value_t* A, int lda) {
    static_assert(device == Device::CPU, "only Device::CPU is implemented here.");
    (void)context;
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i) A[i + j * lda] = value_t(0);
    return SUCCESS;
}

template Status extract_upper_triang<Device::CPU, double>(Context&, int,
                                                          double*, int);
template Status extract_upper_triang<Device::CPU, float>(Context&, int, float*,
                                                         int);
template Status extract_upper_triang<Device::CPU, __half>(Context&, int,
                                                          __half*, int);
template Status extract_upper_triang<Device::CPU, __nv_bfloat16>(Context&, int,
                                                                 __nv_bfloat16*,
                                                                 int);

}  // namespace tomos
