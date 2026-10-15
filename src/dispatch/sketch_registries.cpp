#include "sketch_registries.hpp"

#include "countsketch_kernels.hpp"
#include "mm_cpu_kernels.hpp"

#if USE_CUDA
#include "countsketch_cuda_kernels.cuh"
#include "mm_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

namespace {

// cpu::generate_countsketch_csr_values operates on a raw (nelems, values*)
// pair rather than a Sparse&, so adapt it to the Sparse&-based registry
// signature (the CUDA kernel already takes Sparse& directly).
template <typename value_t>
Status generate_count_sketch_values_cpu(Sparse& S) {
    CsrStorage& storage = std::get<CsrStorage>(S.get_storage());
    value_t* values = std::get<value_t*>(storage.values_);
    return cpu::generate_countsketch_csr_values<value_t>(storage.nnz_, values);
}

// target = sketch (CSR) * source (dense).
template <Device dev, typename a_t, typename b_t, typename c_t,
         typename index_t>
void apply_count_sketch_dense(Sparse& sketch, Dense& source, Dense& target) {
    auto vS = as_csr_view<a_t, index_t>(sketch);
    auto vX = as_strided_view<b_t>(source);
    auto vY = as_strided_view<c_t>(target);
    if constexpr (dev == Device::CPU) {
        cpu::mm_csr<a_t, b_t, c_t, index_t>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            vS.size_, vS.nnz_, vS.values_, vS.row_ptrs_, vS.col_idxs_,
            vX.size_, vX.values_, vX.lead_dim_, 0.0, vY.values_,
            vY.lead_dim_);
    }
    // CUDA path omitted: mm_csr requires a cusparseHandle_t from Context.
}

// target = sketch (CSR) * source (CSR), materialized as dense.
template <Device dev, typename a_t, typename b_t, typename c_t,
         typename index_t>
void apply_count_sketch_sparse(Sparse& sketch, Sparse& source, Dense& target) {
    auto vS = as_csr_view<a_t, index_t>(sketch);
    auto vB = as_csr_view<b_t, index_t>(source);
    auto vY = as_strided_view<c_t>(target);
    if constexpr (dev == Device::CPU) {
        cpu::mm_csr_csr<a_t, b_t, c_t, index_t>(
            1.0, vS.size_, vS.row_ptrs_, vS.col_idxs_, vS.values_, vB.size_,
            vB.row_ptrs_, vB.col_idxs_, vB.values_, 0.0, vY.values_,
            vY.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mm_csr_csr<a_t, b_t, c_t, index_t>(
            1.0, vS.size_, vS.row_ptrs_, vS.col_idxs_, vS.values_, vB.size_,
            vB.row_ptrs_, vB.col_idxs_, vB.values_, 0.0, vY.values_,
            vY.lead_dim_);
    }
#endif
}

}  // namespace

const Sketch& Sketch::get() {
    static const Sketch instance;
    return instance;
}

generate_count_sketch_values_fn Sketch::lookup_generate_count_sketch_values(
    Device dev, NumericType type) const {
    auto fn = generate_values_[encode_values(dev, type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for count sketch generate_values. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

generate_count_sketch_idxs_fn Sketch::lookup_generate_count_sketch_idxs(
    Device dev, IntType type) const {
    auto fn = generate_idxs_[encode_idxs(dev, type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for count sketch generate_idxs. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

apply_count_sketch_dense_fn Sketch::lookup_apply_count_sketch_dense(
    Device dev, NumericType sketch_type, NumericType source_type,
    NumericType target_type, IntType index_type) const {
    auto fn = apply_dense_[encode_apply(dev, sketch_type, source_type,
                                        target_type, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for count sketch apply (dense source). "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

apply_count_sketch_sparse_fn Sketch::lookup_apply_count_sketch_sparse(
    Device dev, NumericType sketch_type, NumericType source_type,
    NumericType target_type, IntType index_type) const {
    auto fn = apply_sparse_[encode_apply(dev, sketch_type, source_type,
                                         target_type, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for count sketch apply (sparse source). "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

constexpr std::size_t Sketch::encode_values(Device dev, NumericType type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(type);
}

constexpr std::size_t Sketch::encode_idxs(Device dev, IntType type) {
    return static_cast<std::size_t>(dev) * num_registry_index_types +
           static_cast<std::size_t>(type);
}

constexpr std::size_t Sketch::encode_apply(Device dev,
                                           NumericType sketch_type,
                                           NumericType source_type,
                                           NumericType target_type,
                                           IntType index_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_index_types +
           static_cast<std::size_t>(sketch_type) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_index_types +
           static_cast<std::size_t>(source_type) * num_registry_numeric_types *
               num_registry_index_types +
           static_cast<std::size_t>(target_type) * num_registry_index_types +
           static_cast<std::size_t>(index_type);
}

Sketch::Sketch() {
    generate_values_.fill(nullptr);
    generate_idxs_.fill(nullptr);
    apply_dense_.fill(nullptr);
    apply_sparse_.fill(nullptr);

    // CPU entries.
    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        generate_values_[encode_values(Device::CPU, numeric_type_of<T>)] =
            &generate_count_sketch_values_cpu<T>;
    });
    for_each_value(IndexTypes{}, [&](auto idx_tag) {
        constexpr IntType idx = decltype(idx_tag)::value;
        using IdxT = to_concrete_type<idx>;
        generate_idxs_[encode_idxs(Device::CPU, idx)] =
            &cpu::generate_countsketch_csr_idxs<IdxT>;
    });

    for_each_type(FPTypes{}, [&](auto a_tag) {
        using AT = typename decltype(a_tag)::type;
        for_each_type(FPTypes{}, [&](auto b_tag) {
            using BT = typename decltype(b_tag)::type;
            for_each_type(FPTypes{}, [&](auto c_tag) {
                using CT = typename decltype(c_tag)::type;
                for_each_value(IndexTypes{}, [&](auto idx_tag) {
                    constexpr IntType idx = decltype(idx_tag)::value;
                    using IdxT = to_concrete_type<idx>;
                    apply_dense_[encode_apply(Device::CPU, numeric_type_of<AT>,
                                              numeric_type_of<BT>,
                                              numeric_type_of<CT>, idx)] =
                        &apply_count_sketch_dense<Device::CPU, AT, BT, CT,
                                                  IdxT>;
                    apply_sparse_[encode_apply(
                        Device::CPU, numeric_type_of<AT>, numeric_type_of<BT>,
                        numeric_type_of<CT>, idx)] =
                        &apply_count_sketch_sparse<Device::CPU, AT, BT, CT,
                                                   IdxT>;
                });
            });
        });
    });
    // CUDA path for the dense-source overload omitted: mm_csr requires a
    // cusparseHandle_t from Context.

#if USE_CUDA
    // CUDA entries.
    for_each_type(FPTypes{}, [&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        generate_values_[encode_values(Device::CUDA, numeric_type_of<T>)] =
            &cuda::generate_countsketch_csr_values<T>;
    });
    for_each_value(IndexTypes{}, [&](auto idx_tag) {
        constexpr IntType idx = decltype(idx_tag)::value;
        using IdxT = to_concrete_type<idx>;
        generate_idxs_[encode_idxs(Device::CUDA, idx)] =
            &cuda::generate_countsketch_csr_idxs<IdxT>;
    });

    for_each_type(FPTypes{}, [&](auto a_tag) {
        using AT = typename decltype(a_tag)::type;
        for_each_type(FPTypes{}, [&](auto b_tag) {
            using BT = typename decltype(b_tag)::type;
            for_each_type(FPTypes{}, [&](auto c_tag) {
                using CT = typename decltype(c_tag)::type;
                for_each_value(IndexTypes{}, [&](auto idx_tag) {
                    constexpr IntType idx = decltype(idx_tag)::value;
                    using IdxT = to_concrete_type<idx>;
                    apply_sparse_[encode_apply(
                        Device::CUDA, numeric_type_of<AT>,
                        numeric_type_of<BT>, numeric_type_of<CT>, idx)] =
                        &apply_count_sketch_sparse<Device::CUDA, AT, BT, CT,
                                                   IdxT>;
                });
            });
        });
    });
#endif
}

}  // namespace dispatch
}  // namespace tomos
