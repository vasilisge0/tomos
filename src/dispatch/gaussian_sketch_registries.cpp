#include "gaussian_sketch_registries.hpp"

#include "countsketch_kernels.hpp"
#include "gaussian_sketch_kernels.hpp"

#if USE_CUDA
#include "countsketch_cuda_kernels.cuh"
#include "gaussian_sketch_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

namespace {

template <Device dev, typename value_t>
Status create_gaussian_sketch_dense_impl(Dense& S) {
    auto view = as_strided_view<value_t>(S);
    if constexpr (dev == Device::CPU) {
        return cpu::generate_gaussian_values<value_t>(view.size_, view.values_,
                                                       view.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::generate_gaussian_values<value_t>(view.size_, view.values_,
                                                view.lead_dim_);
        return SUCCESS;
    }
#endif
}

// Structure (nnz_per_row == 1, one random column per row) is identical to
// generate_count_sketch's; only the nonzero values differ (N(0,1) vs +-1), so
// the sparsity pattern generator is reused directly.
template <Device dev, typename value_t, typename index_t>
Status create_gaussian_sketch_sparse_impl(Sparse& S) {
    if constexpr (dev == Device::CPU) {
        cpu::generate_countsketch_csr_idxs<index_t>(S);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::generate_countsketch_csr_idxs<index_t>(S);
    }
#endif

    CsrStorage& storage = std::get<CsrStorage>(S.get_storage());
    value_t* values = std::get<value_t*>(storage.values_);
    dim<2> vsize(storage.nnz_, 1);
    if constexpr (dev == Device::CPU) {
        return cpu::generate_gaussian_values<value_t>(vsize, values,
                                                       storage.nnz_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::generate_gaussian_values<value_t>(vsize, values, storage.nnz_);
        return SUCCESS;
    }
#endif
}

}  // namespace

const GaussianSketch& GaussianSketch::get() {
    static const GaussianSketch instance;
    return instance;
}

create_gaussian_sketch_dense_fn GaussianSketch::lookup_create_dense(
    Device dev, NumericType type) const {
    auto fn = dense_[encode_dense(dev, type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for gaussian sketch (dense). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

create_gaussian_sketch_sparse_fn GaussianSketch::lookup_create_sparse(
    Device dev, NumericType type, IntType index_type) const {
    auto fn = sparse_[encode_sparse(dev, type, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for gaussian sketch (sparse). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

constexpr std::size_t GaussianSketch::encode_dense(Device dev,
                                                   NumericType type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(type);
}

constexpr std::size_t GaussianSketch::encode_sparse(Device dev,
                                                    NumericType type,
                                                    IntType index_type) {
    return (static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(type)) *
               num_registry_index_types +
           static_cast<std::size_t>(index_type);
}

GaussianSketch::GaussianSketch() {
    dense_.fill(nullptr);
    sparse_.fill(nullptr);

    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        dense_[encode_dense(Device::CPU, numeric_type_of<T>)] =
            &create_gaussian_sketch_dense_impl<Device::CPU, T>;
    });
    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        for_each_value(IndexTypes{}, [&](auto idx_tag) {
            constexpr IntType idx = decltype(idx_tag)::value;
            using IdxT = to_concrete_type<idx>;
            sparse_[encode_sparse(Device::CPU, numeric_type_of<T>, idx)] =
                &create_gaussian_sketch_sparse_impl<Device::CPU, T, IdxT>;
        });
    });

#if USE_CUDA
    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        dense_[encode_dense(Device::CUDA, numeric_type_of<T>)] =
            &create_gaussian_sketch_dense_impl<Device::CUDA, T>;
    });
    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        for_each_value(IndexTypes{}, [&](auto idx_tag) {
            constexpr IntType idx = decltype(idx_tag)::value;
            using IdxT = to_concrete_type<idx>;
            sparse_[encode_sparse(Device::CUDA, numeric_type_of<T>, idx)] =
                &create_gaussian_sketch_sparse_impl<Device::CUDA, T, IdxT>;
        });
    });
#endif
}

}  // namespace dispatch
}  // namespace tomos
