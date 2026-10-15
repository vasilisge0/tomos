#include "sparse_registries.hpp"

#include "matrix_cpu_kernels.hpp"
#include "set_values_cpu_kernels.hpp"
#include "sparse_storage.hpp"

#if USE_CUDA
#include "context.hpp"
#include "copy_cuda_kernels.cuh"
#include "mv_cuda_kernels.cuh"
#include "set_values_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

template <Device dev, typename value_t>
void set_values_csr(double val, tomos::Sparse& target) {
    CsrStorage& S = std::get<CsrStorage>(target.get_storage());
    value_t* values = std::get<value_t*>(S.values_);
    if constexpr (dev == Device::CPU) {
        cpu::set_values_impl<value_t>(static_cast<value_t>(val), {S.nnz_, 1},
                                      values, 1);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::set_values_impl<value_t>(static_cast<value_t>(val), {S.nnz_, 1},
                                       values, 1);
    }
#endif
}

template <Device dev, typename a_t, typename x_t, typename y_t,
          typename index_t>
void mv_csr(TransposeType trans, double alpha, tomos::Sparse& A, tomos::Dense& x,
            double beta, tomos::Dense& y) {
    auto vA = as_csr_view<a_t, index_t>(A);
    auto vx = as_strided_view<x_t>(x);
    auto vy = as_strided_view<y_t>(y);
    if constexpr (dev == Device::CPU) {
        cpu::mv_csr<a_t, x_t, y_t, index_t>(trans, alpha, vA.size_, vA.nnz_,
                                             vA.values_, vA.row_ptrs_,
                                             vA.col_idxs_, vx.values_,
                                             vx.lead_dim_, beta, vy.values_,
                                             vy.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mv_csr<a_t, x_t, y_t, index_t>(trans, alpha, vA.size_, vA.nnz_,
                                             vA.values_, vA.row_ptrs_,
                                             vA.col_idxs_, vx.values_,
                                             vx.lead_dim_, beta, vy.values_,
                                             vy.lead_dim_,
                                             cuda_cusparse_handle());
    }
#endif
}

const Sparse& Sparse::get() {
    static const Sparse instance;
    return instance;
}

initialize_sparse_fn Sparse::lookup_initialize(MatrixFormat format, Device dev,
                                               NumericType numeric_type,
                                               IntType index_type) const {
    auto fn = initialize_[encode(format, dev, numeric_type, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

free_sparse_fn Sparse::lookup_free(MatrixFormat format, Device dev,
                                   NumericType numeric_type,
                                   IntType index_type) const {
    auto fn = free_[encode(format, dev, numeric_type, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

set_values_sparse_fn Sparse::lookup_set_values(Device dev,
                                               NumericType numeric_type) const {
    auto fn = set_values_[encode_set_values(dev, numeric_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mv_sparse_fn Sparse::lookup_mv(Device dev, NumericType type_A,
                               NumericType type_x, NumericType type_y,
                               IntType index_type) const {
    auto fn = mv_[encode_mv(dev, type_A, type_x, type_y, index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

copy_sparse_fn Sparse::lookup_copy(MatrixFormat source_format,
                                   MatrixFormat target_format,
                                   Device source_dev, Device target_dev,
                                   NumericType source_numeric_type,
                                   NumericType target_numeric_type,
                                   IntType source_index_type,
                                   IntType target_index_type) const {
    auto fn =
        copy_[encode_copy(source_format, target_format, source_dev, target_dev,
                          source_numeric_type, target_numeric_type,
                          source_index_type, target_index_type)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

constexpr std::size_t Sparse::encode(MatrixFormat format, Device dev,
                                     NumericType numeric_type,
                                     IntType index_type) {
    return static_cast<std::size_t>(format) * num_registry_devices *
               num_registry_numeric_types * num_registry_index_types +
           static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_index_types +
           static_cast<std::size_t>(numeric_type) * num_registry_index_types +
           static_cast<std::size_t>(index_type);
}

constexpr std::size_t Sparse::encode_set_values(Device dev,
                                                NumericType numeric_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(numeric_type);
}

constexpr std::size_t Sparse::encode_mv(Device dev, NumericType type_A,
                                        NumericType type_x, NumericType type_y,
                                        IntType index_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_index_types +
           static_cast<std::size_t>(type_A) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_index_types +
           static_cast<std::size_t>(type_x) * num_registry_numeric_types *
               num_registry_index_types +
           static_cast<std::size_t>(type_y) * num_registry_index_types +
           static_cast<std::size_t>(index_type);
}

constexpr std::size_t Sparse::encode_copy(MatrixFormat source_format,
                                          MatrixFormat target_format,
                                          Device source_dev, Device target_dev,
                                          NumericType source_numeric_type,
                                          NumericType target_numeric_type,
                                          IntType source_index_type,
                                          IntType target_index_type) {
    // Mixed-radix index; source_format/source_dev must each carry one more
    // factor of their own dimension's range than target_format/target_dev,
    // otherwise e.g. (source_dev=CPU, target_dev=CUDA) and
    // (source_dev=CUDA, target_dev=CPU) collide on the same index --
    // confirmed via a real CopyTests.TestCopyCsr failure where the CPU->CUDA
    // and CUDA->CPU CSR copy registrations silently overwrote each other.
    return static_cast<std::size_t>(source_format) *
               static_cast<std::size_t>(MatrixFormat::Count) *
               num_registry_devices * num_registry_devices *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_index_types * num_registry_index_types +
           static_cast<std::size_t>(target_format) * num_registry_devices *
               num_registry_devices * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_index_types *
               num_registry_index_types +
           static_cast<std::size_t>(source_dev) * num_registry_devices *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_index_types * num_registry_index_types +
           static_cast<std::size_t>(target_dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_index_types *
               num_registry_index_types +
           static_cast<std::size_t>(source_numeric_type) *
               num_registry_numeric_types * num_registry_index_types *
               num_registry_index_types +
           static_cast<std::size_t>(target_numeric_type) *
               num_registry_index_types * num_registry_index_types +
           static_cast<std::size_t>(source_index_type) *
               num_registry_index_types +
           static_cast<std::size_t>(target_index_type);
}

Sparse::Sparse() {
    initialize_.fill(nullptr);
    free_.fill(nullptr);
    copy_.fill(nullptr);
    set_values_.fill(nullptr);
    mv_.fill(nullptr);

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto fp_tag) {
            using FP = typename decltype(fp_tag)::type;
            for_each_value(IndexTypes{}, [&](auto idx_tag) {
                constexpr IntType idx = decltype(idx_tag)::value;
                using IdxT = to_concrete_type<idx>;
                initialize_[encode(MatrixFormat::CSR, dev, numeric_type_of<FP>,
                                   idx)] =
                    &initialize_csr_storage<dev, FP, IdxT>;
                free_[encode(MatrixFormat::CSR, dev, numeric_type_of<FP>,
                             idx)] = &free_csr_storage<dev, FP, IdxT>;
            });
        });
    });

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto fp_tag) {
            using FP = typename decltype(fp_tag)::type;
            set_values_[encode_set_values(dev, numeric_type_of<FP>)] =
                &set_values_csr<dev, FP>;
        });
    });

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto a_tag) {
            using AT = typename decltype(a_tag)::type;
            for_each_type(FPTypes{}, [&](auto x_tag) {
                using XT = typename decltype(x_tag)::type;
                for_each_type(FPTypes{}, [&](auto y_tag) {
                    using YT = typename decltype(y_tag)::type;
                    for_each_value(IndexTypes{}, [&](auto idx_tag) {
                        constexpr IntType idx = decltype(idx_tag)::value;
                        using IdxT = to_concrete_type<idx>;
                        mv_[encode_mv(dev, numeric_type_of<AT>,
                                      numeric_type_of<XT>, numeric_type_of<YT>,
                                      idx)] =
                            &mv_csr<dev, AT, XT, YT, IdxT>;
                    });
                });
            });
        });
    });

    // CPU → CPU copy -- registered unconditionally (unlike the CUDA
    // directions below), mirroring Dense's own CPU->CPU copy_ registration.
    // copy_csr_storage<Device::CPU, Device::CPU, ...> (sparse_storage.hpp)
    // already existed and worked; it just was never wired into this
    // registry, so lookup_copy() for two Device::CPU Sparse objects always
    // threw "No kernel registered" (confirmed via CopyTests.TestCopyCsr).
    for_each_type(FPTypes{}, [&](auto src_tag) {
        using SrcT = typename decltype(src_tag)::type;
        for_each_type(FPTypes{}, [&](auto tgt_tag) {
            using TgtT = typename decltype(tgt_tag)::type;
            for_each_value(IndexTypes{}, [&](auto idx_tag) {
                constexpr IntType idx = decltype(idx_tag)::value;
                using IdxT = to_concrete_type<idx>;
                copy_[encode_copy(MatrixFormat::CSR, MatrixFormat::CSR,
                                  Device::CPU, Device::CPU,
                                  numeric_type_of<SrcT>, numeric_type_of<TgtT>,
                                  idx, idx)] =
                    &copy_csr_storage<Device::CPU, Device::CPU, SrcT, TgtT,
                                      IdxT, IdxT>;
            });
        });
    });

#if USE_CUDA
    // CPU → CUDA copy
    for_each_type(FPTypes{}, [&](auto fp_tag) {
        using FP = typename decltype(fp_tag)::type;
        for_each_value(IndexTypes{}, [&](auto idx_tag) {
            constexpr IntType idx = decltype(idx_tag)::value;
            using IdxT = to_concrete_type<idx>;
            copy_[encode_copy(MatrixFormat::CSR, MatrixFormat::CSR, Device::CPU,
                              Device::CUDA, numeric_type_of<FP>,
                              numeric_type_of<FP>, idx, idx)] =
                &cuda::csr_copy_to_device<FP, IdxT>;
        });
    });

    // CUDA → CPU copy
    for_each_type(FPTypes{}, [&](auto fp_tag) {
        using FP = typename decltype(fp_tag)::type;
        for_each_value(IndexTypes{}, [&](auto idx_tag) {
            constexpr IntType idx = decltype(idx_tag)::value;
            using IdxT = to_concrete_type<idx>;
            copy_[encode_copy(MatrixFormat::CSR, MatrixFormat::CSR,
                              Device::CUDA, Device::CPU, numeric_type_of<FP>,
                              numeric_type_of<FP>, idx, idx)] =
                &cuda::csr_copy_from_device<FP, IdxT>;
        });
    });
#endif

}  // Sparse constructor

}  // namespace dispatch
}  // namespace tomos