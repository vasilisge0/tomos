#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <string>

#include "context.hpp"
#include "dense_storage.hpp"
#include "dense_storage_cpu_kernels.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "matrix.hpp"
#include "mmio.h"
#include "precision.hpp"
#include "status.hpp"
#include "type_macros.hpp"

namespace tomos {

template <Device device, typename value_t>
void initialize_strided_storage(dim<2> size, DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    size_t non_lead_dim = (S.layout_ == Layout::ROW_MAJOR) ? size[0] : size[1];
    value_t* t;
    malloc<device, value_t>(S.lead_dim_ * non_lead_dim, &t);
    S.values_ = t;
}

INSTANTIATE_ALL_DEVICES_AND_TYPES(initialize_strided_storage,
                                  (dim<2>, DenseStorage&))

template <Device device, typename value_t>
void free_strided_storage(DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    value_t* v = std::get<value_t*>(S.values_);
    if (v != nullptr) free<device, value_t>(v);
}

INSTANTIATE_ALL_DEVICES_AND_TYPES(free_strided_storage, (DenseStorage&))

// NOT USED. Zero callers outside the explicit instantiations below --
// Dense::copy_from routes through copy_strided_storage (below) instead, via
// dispatch::copy_dense_storage / the Dense registry's copy table. Kept only
// because it's still explicitly instantiated; safe to remove if that ever
// changes.
template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_data(dim<2> size, value_in_t* source, size_t ld_source,
                       value_out_t* target, size_t ld_target) {
    if constexpr (device_source == Device::CPU &&
                  device_target == Device::CPU) {
        cpu::strided_copy_kernel(size, source, ld_source, target, ld_target);
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}

#define INST_COPY_DATA(ds, dt, vi, vo)                                        \
    template void copy_strided_data<ds, dt, vi, vo>(dim<2>, vi*, size_t, vo*, \
                                                    size_t);

#define INST_COPY_DATA_VO(ds, dt, vi)  \
    INST_COPY_DATA(ds, dt, vi, double) \
    INST_COPY_DATA(ds, dt, vi, float)  \
    INST_COPY_DATA(ds, dt, vi, __half)

#define INST_COPY_DATA_DEV(ds, dt)    \
    INST_COPY_DATA_VO(ds, dt, double) \
    INST_COPY_DATA_VO(ds, dt, float)  \
    INST_COPY_DATA_VO(ds, dt, __half)

INST_COPY_DATA_DEV(Device::CPU, Device::CPU)
#if USE_CUDA
INST_COPY_DATA_DEV(Device::CPU, Device::CUDA)
INST_COPY_DATA_DEV(Device::CUDA, Device::CPU)
INST_COPY_DATA_DEV(Device::CUDA, Device::CUDA)
#endif

#undef INST_COPY_DATA
#undef INST_COPY_DATA_VO
#undef INST_COPY_DATA_DEV

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_storage(dim<2> size, DenseStorage& source,
                          DenseStorage& target) {
    StridedStorage& S_source = std::get<StridedStorage>(source);
    StridedStorage& S_target = std::get<StridedStorage>(target);
    if constexpr (device_source == Device::CPU &&
                  device_target == Device::CPU) {
        cpu::strided_copy_kernel(
            size, std::get<value_in_t*>(S_source.values_), S_source.lead_dim_,
            std::get<value_out_t*>(S_target.values_), S_target.lead_dim_);
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}  //  copy_strided_storage

INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS(copy_strided_storage,
                                      (dim<2>, DenseStorage&, DenseStorage&))

// NOT USED, AND BROKEN: never called, never explicitly instantiated, and its
// signature (value_in_t* source, value_out_t& target) doesn't even match its
// own body, which treats source/target as DenseStorage& via
// std::get<StridedStorage>(...). Would fail to compile if anything ever tried
// to instantiate it. Left as-is; likely an abandoned edit of
// copy_strided_storage above.
template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_data(dim<2> size, value_in_t* source, value_out_t& target) {
    StridedStorage& S_source = std::get<StridedStorage>(source);
    StridedStorage& S_target = std::get<StridedStorage>(target);
    if constexpr (device_source == Device::CPU &&
                  device_target == Device::CPU) {
        cpu::strided_copy_kernel(
            size, std::get<value_in_t*>(S_source.values_), S_source.lead_dim_,
            std::get<value_out_t*>(S_target.values_), S_target.lead_dim_);
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}  //  copy_strided_storage

// void write_mtx(Dense& matrix, std::string& filename, dim<2> col_subset) {}

namespace cpu {
template <typename value_type>
void read_strided_storage(char* filename, DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    dim<2> size = S.size_;
    value_type* values = std::get<value_type*>(S.values_);
    MM_typecode matcode;
    int ld = S.lead_dim_;
    int m = (S.layout_ == Layout::COL_MAJOR) ? (int)size[0] : (int)size[1];
    int n = (S.layout_ == Layout::COL_MAJOR) ? (int)size[1] : (int)size[0];
    FILE* file_handle = fopen(filename, "r");

    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, &m, &n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if constexpr (std::is_same<value_type, double>::value) {
                fscanf(file_handle, "%lf ", &values[i * ld + j]);
            } else if constexpr (std::is_same<value_type, float>::value) {
                fscanf(file_handle, "%f ", &values[i * ld + j]);
            } else if constexpr (std::is_same<value_type, __half>::value) {
                float temp;
                fscanf(file_handle, "%f ", &temp);
                values[i * ld + j] = (__half)temp;
            } else if constexpr (std::is_same<value_type, __nv_bfloat16>::value) {
                float temp;
                fscanf(file_handle, "%f ", &temp);
                values[i * ld + j] = (__nv_bfloat16)temp;
            } else {
                fclose(file_handle);
                throw std::invalid_argument(
                    "Unsupported value type for reading MTX files.");
            }
        }
    }
    fclose(file_handle);
}

#define INST_READ_STRIDED(value_t) \
    template void read_strided_storage<value_t>(char*, DenseStorage&);
FOR_ALL_FP_SINGLES(INST_READ_STRIDED)
#undef INST_READ_STRIDED

template <typename value_t>
void write_strided_storage(const char* filename, DenseStorage& source) {
    MM_typecode matcode;
    StridedStorage& S = std::get<StridedStorage>(source);
    value_t* values = std::get<value_t*>(S.values_);
    dim<2> size = S.size_;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    int ld = static_cast<int>(S.lead_dim_);
    int m = (S.layout_ == Layout::COL_MAJOR) ? (int)size[0] : (int)size[1];
    int n = (S.layout_ == Layout::COL_MAJOR) ? (int)size[1] : (int)size[0];

    FILE* file_handle = fopen(filename, "w");
    if (file_handle == nullptr) {
        std::cerr << "In \"" << filename << "\"" << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    mm_write_banner(file_handle, matcode);
    mm_write_mtx_array_size(file_handle, static_cast<int>(size[0]),
                            static_cast<int>(size[1]));
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            if (isnan(values[i + j * ld])) {
                fclose(file_handle);
                fflush(file_handle);
                throw std::runtime_error(
                    "A nan value has been encountered during a write operation.");
            }
            if constexpr (std::is_same<value_t, double>::value) {
                fprintf(file_handle, "%1.16e\n", values[i + j * ld]);
            } else if constexpr (std::is_same<value_t, float>::value) {
                fprintf(file_handle, "%1.16e\n", values[i + j * ld]);
            } else if constexpr (std::is_same<value_t, __half>::value) {
                uint16_t temp;
                memcpy(&temp, &values[i + j * ld], sizeof(uint16_t));
                fwrite(&temp, sizeof(uint16_t), 1, file_handle);
            } else if constexpr (std::is_same<value_t, __nv_bfloat16>::value) {
                uint16_t temp;
                memcpy(&temp, &values[i + j * ld], sizeof(uint16_t));
                fwrite(&temp, sizeof(uint16_t), 1, file_handle);
            } else {
                fclose(file_handle);
                fflush(file_handle);
                throw std::invalid_argument(
                    "Unsupported value type for writing MTX files.");
            }
        }
    }
    fclose(file_handle);
    fflush(file_handle);
}

template <typename value_t>
void write_strided_storage(const char* filename, DenseStorage& source,
                           dim<2> col_subset) {
    MM_typecode matcode;
    StridedStorage& S = std::get<StridedStorage>(source);
    value_t* values = std::get<value_t*>(S.values_);
    dim<2> size = S.size_;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    int ld = static_cast<int>(S.lead_dim_);
    int m = (S.layout_ == Layout::COL_MAJOR) ? (int)size[0] : (int)size[1];
    int n = (S.layout_ == Layout::COL_MAJOR) ? (int)size[1] : (int)size[0];

    FILE* file_handle = fopen(filename, "w");
    if (file_handle == nullptr) {
        std::cerr << "In \"" << filename << "\"" << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    mm_write_banner(file_handle, matcode);
    mm_write_mtx_array_size(file_handle, static_cast<int>(size[0]),
                            static_cast<int>(size[1]));
    for (size_t j = col_subset[0]; j < col_subset[1]; ++j) {
        for (size_t i = 0; i < m; ++i) {
            if (isnan(values[i + j * ld])) {
                fclose(file_handle);
                fflush(file_handle);
                throw std::runtime_error(
                    "A nan value has been encountered during a write operation.");
            }
            if constexpr (std::is_same<value_t, double>::value) {
                fprintf(file_handle, "%1.16e\n", values[i + j * ld]);
            } else if constexpr (std::is_same<value_t, float>::value) {
                fprintf(file_handle, "%1.16e\n", values[i + j * ld]);
            } else if constexpr (std::is_same<value_t, __half>::value) {
                uint16_t temp;
                memcpy(&temp, &values[i + j * ld], sizeof(uint16_t));
                fwrite(&temp, sizeof(uint16_t), 1, file_handle);
            } else if constexpr (std::is_same<value_t, __nv_bfloat16>::value) {
                uint16_t temp;
                memcpy(&temp, &values[i + j * ld], sizeof(uint16_t));
                fwrite(&temp, sizeof(uint16_t), 1, file_handle);
            } else {
                fclose(file_handle);
                fflush(file_handle);
                throw std::invalid_argument(
                    "Unsupported value type for writing MTX files.");
            }
        }
    }
    fclose(file_handle);
    fflush(file_handle);
}

}  // namespace cpu

namespace cuda {

template <typename value_t>
void read_strided_storage(char* filename_mtx, DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    dim<2> size = S.size_;
    value_t* values = S.values_;
    Context context = tomos::Context(Device::CPU);
    if constexpr (std::is_same<value_t, double>::value) {
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP64, size);
        cpu::read_strided_storage<double>(filename_mtx, (tmp).storage_);
        StridedView<double> V = StridedView<double>(tmp);

        size_t num_elems = size[0] * size[1];
        cudaMemcpy(values, V.values_, num_elems * sizeof(double),
                   cudaMemcpyHostToDevice);
    } else if constexpr (std::is_same<value_t, float>::value) {
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP32, size);
        cpu::read_strided_storage<float>(filename_mtx, (tmp).storage_);
        StridedView<float> V = StridedView<float>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(values, V.values_, num_elems * sizeof(float),
                   cudaMemcpyHostToDevice);
    } else if constexpr (std::is_same<value_t, __half>::value) {
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP16, size);
        cpu::read_strided_storage<__half>(filename_mtx, (tmp).storage_);
        StridedView<__half> V = StridedView<__half>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(values, V.values_, num_elems * sizeof(__half),
                   cudaMemcpyHostToDevice);
    } else if constexpr (std::is_same<value_t, __nv_bfloat16>::value) {
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::BF16, size);
        cpu::read_strided_storage<__nv_bfloat16>(filename_mtx, (tmp).storage_);
        StridedView<__nv_bfloat16> V = StridedView<__nv_bfloat16>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(values, V.values_, num_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
    }
}

template <typename value_t>
void write_strided_storage(char* filename_mtx, DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    dim<2> size = S.size_;
    value_t* values = S.values_;
    Context context = tomos::Context(Device::CPU);
    if constexpr (std::is_same<value_t, double>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP64, size);

        // copy over to CPU memory
        StridedView<double> V = StridedView<double>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<double>(filename_mtx, (tmp).storage_);

    } else if constexpr (std::is_same<value_t, float>::value) {
        // create temporary dense strided storage.
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP32, size);

        // copy over to CPU memory
        StridedView<float> V = StridedView<float>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<float>(filename_mtx, (tmp).storage_);
    } else if constexpr (std::is_same<value_t, __half>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP16, size);

        // copy over to CPU memory
        StridedView<__half> V = StridedView<__half>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(__half),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<__half>(filename_mtx, (tmp).storage_);
    } else if constexpr (std::is_same<value_t, __nv_bfloat16>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::BF16, size);

        // copy over to CPU memory
        StridedView<__nv_bfloat16> V = StridedView<__nv_bfloat16>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<__nv_bfloat16>(filename_mtx, (tmp).storage_);
    }
}

template <typename value_t>
void write_strided_storage(char* filename_mtx, DenseStorage& source,
                           dim<2> col_subset) {
    StridedStorage& S = std::get<StridedStorage>(source);
    dim<2> size = S.size_;
    value_t* values = S.values_;
    Context context = tomos::Context(Device::CPU);
    if constexpr (std::is_same<value_t, double>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP64, size);

        // copy over to CPU memory
        StridedView<double> V = StridedView<double>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<double>(filename_mtx, (tmp).storage_,
                                           col_subset);

    } else if constexpr (std::is_same<value_t, float>::value) {
        // create temporary dense strided storage.
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP32, size);

        // copy over to CPU memory
        StridedView<float> V = StridedView<float>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<float>(filename_mtx, (tmp).storage_,
                                          col_subset);
    } else if constexpr (std::is_same<value_t, __half>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::FP16, size);

        // copy over to CPU memory
        StridedView<__half> V = StridedView<__half>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(__half),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<__half>(filename_mtx, tmp.storage_,
                                           col_subset);
    } else if constexpr (std::is_same<value_t, __nv_bfloat16>::value) {
        // create temporary dense strided storage
        auto tmp = Dense::create_strided(context, tomos::Layout::COL_MAJOR,
                                         tomos::NumericType::BF16, size);

        // copy over to CPU memory
        StridedView<__nv_bfloat16> V = StridedView<__nv_bfloat16>(tmp);
        size_t num_elems = size[0] * size[1];
        cudaMemcpy(V.values_, values, num_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);

        // write to file
        cpu::write_strided_storage<__nv_bfloat16>(filename_mtx, tmp.storage_,
                                                   col_subset);
    }
}

}  // namespace cuda

}  // namespace tomos
