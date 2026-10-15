#include "countsketch_kernels.hpp"

#include <cstdint>
#include <stdexcept>

#include "matrix.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace cpu {

template <typename value_t>
Status generate_countsketch_csr_values(size_t nelems, value_t* values) {
    for (size_t i = 0; i < nelems; ++i) {
        values[i] = ((rand() % 2) == 0) ? -1.0 : 1.0;
    }
    return SUCCESS;
}

template Status generate_countsketch_csr_values<double>(size_t nelems,
                                                        double* values);
template Status generate_countsketch_csr_values<float>(size_t nelems,
                                                       float* values);
template Status generate_countsketch_csr_values<__half>(size_t nelems,
                                                        half* values);
template Status generate_countsketch_csr_values<__nv_bfloat16>(size_t nelems,
                                                        __nv_bfloat16* values);

// nnz_per_row == 1 only (matches the CUDA implementation): each row (bucket)
// picks exactly one random column in [0, ncols) to represent it, so
// row_ptrs is the trivial {0, 1, 2, ..., nrows} and nnz must equal nrows.
template <typename index_t>
Status generate_countsketch_csr_idxs(Sparse& mtx) {
    CsrSparsityView<index_t> view(mtx);
    size_t nrows = view.size_[0];
    size_t ncols = view.size_[1];
    if (view.nnz_ != nrows)
        throw std::invalid_argument(
            "generate_countsketch_csr_idxs: nnz_per_row == 1 only; the "
            "sketch's nnz must equal its number of rows.");

    index_t* ptrs = view.row_ptrs_;
    index_t* idxs = view.col_idxs_;
    srand(time(0));

    for (size_t i = 0; i < nrows; ++i) {
        ptrs[i] = static_cast<index_t>(i);
        idxs[i] = static_cast<index_t>(rand() % ncols);
    }
    ptrs[nrows] = static_cast<index_t>(nrows);
    return SUCCESS;
}

template Status generate_countsketch_csr_idxs<int>(Sparse& mtx);
template Status generate_countsketch_csr_idxs<long>(Sparse& mtx);

// Same fixed seeds as the CUDA implementation.
static constexpr uint64_t CS_BUCKET_SEED = 0x12345678abcdef00ull;
static constexpr uint64_t CS_SIGN_SEED   = 0xf00dd00d42babe55ull;

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

template <typename value_t>
void countsketch_apply_impl(dim<2> size, const value_t* X, size_t ldx,
                            dim<2> size_y, value_t* Y, size_t ldy) {
    const int n = static_cast<int>(size[0]);    // input rows
    const int B = static_cast<int>(size[1]);    // columns (batch)
    const int m = static_cast<int>(size_y[0]);  // sketch rows (buckets)

    for (int i = 0; i < n; ++i) {
        uint64_t hb = splitmix64(static_cast<uint64_t>(i) ^ CS_BUCKET_SEED);
        uint64_t hs = splitmix64(static_cast<uint64_t>(i) ^ CS_SIGN_SEED);
        int bucket = static_cast<int>(hb % static_cast<uint64_t>(m));
        value_t sign = (hs & 1ull) ? value_t(1) : value_t(-1);
        for (int j = 0; j < B; ++j)
            Y[bucket + j * static_cast<int>(ldy)] +=
                sign * X[i + j * static_cast<int>(ldx)];
    }
}

template void countsketch_apply_impl<float>(dim<2>, const float*, size_t,
                                            dim<2>, float*, size_t);
template void countsketch_apply_impl<double>(dim<2>, const double*, size_t,
                                             dim<2>, double*, size_t);

}  // namespace cpu
}  // namespace tomos
