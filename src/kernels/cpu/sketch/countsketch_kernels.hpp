#ifndef TOMOS_COUNTSKETCH_CPU_KERNELS_HPP
#define TOMOS_COUNTSKETCH_CPU_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#include "dimensions.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace cpu {

template <typename value_t>
Status generate_countsketch_csr_values(size_t nelems, value_t* values);

template <typename index_t>
Status generate_countsketch_csr_idxs(Sparse& mtx);

// Apply a CountSketch S in-place via on-the-fly hashing: Y = S * X.
// X: n x B col-major (ldx >= n), Y: m x B col-major (ldy >= m), pre-zeroed.
// Uses the same splitmix64 hash and fixed seeds as the CUDA implementation.
template <typename value_t>
void countsketch_apply_impl(dim<2> size, const value_t* X, size_t ldx,
                            dim<2> size_y, value_t* Y, size_t ldy);

}  // namespace cpu
}  // namespace tomos

#endif  // TOMOS_COUNTSKETCH_CPU_KERNELS_HPP
