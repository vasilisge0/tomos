#include "io.hpp"

#include <cstdio>
#include <stdexcept>
#include <type_traits>

#include "mmio.h"

namespace tomos {

dim<2> read_mtx_size(const std::string& filename) {
    FILE* file_handle = std::fopen(filename.c_str(), "r");
    if (!file_handle)
        throw std::runtime_error("read_mtx_size: could not open '" + filename +
                                 "'");
    MM_typecode matcode;
    if (mm_read_banner(file_handle, &matcode) != 0) {
        std::fclose(file_handle);
        throw std::runtime_error("read_mtx_size: '" + filename +
                                 "' is not a valid MatrixMarket file");
    }
    int m = 0, n = 0;
    if (mm_read_mtx_array_size(file_handle, &m, &n) != 0) {
        std::fclose(file_handle);
        throw std::runtime_error("read_mtx_size: could not read array size "
                                 "from '" +
                                 filename + "'");
    }
    std::fclose(file_handle);
    return dim<2>{static_cast<size_t>(m), static_cast<size_t>(n)};
}

template <typename value_t>
void read_mtx_values(const std::string& filename, dim<2> size, Layout layout,
                     value_t* values, size_t ld) {
    FILE* file_handle = std::fopen(filename.c_str(), "r");
    if (!file_handle)
        throw std::runtime_error("read_mtx_values: could not open '" +
                                 filename + "'");
    MM_typecode matcode;
    if (mm_read_banner(file_handle, &matcode) != 0) {
        std::fclose(file_handle);
        throw std::runtime_error("read_mtx_values: '" + filename +
                                 "' is not a valid MatrixMarket file");
    }
    int file_rows = 0, file_cols = 0;
    if (mm_read_mtx_array_size(file_handle, &file_rows, &file_cols) != 0) {
        std::fclose(file_handle);
        throw std::runtime_error("read_mtx_values: could not read array size "
                                 "from '" +
                                 filename + "'");
    }
    if (static_cast<size_t>(file_rows) != size[0] ||
        static_cast<size_t>(file_cols) != size[1]) {
        std::fclose(file_handle);
        throw std::runtime_error(
            "read_mtx_values: '" + filename + "' declares " +
            std::to_string(file_rows) + "x" + std::to_string(file_cols) +
            ", expected " + std::to_string(size[0]) + "x" +
            std::to_string(size[1]));
    }

    // The MatrixMarket dense array format always lists values column-major
    // (all of column 0, then column 1, ...), regardless of the target
    // buffer's own layout -- so the file itself is always walked with
    // columns on the outer loop and rows on the inner loop. Only the
    // DESTINATION index changes with layout: row + col*ld for a
    // column-major buffer, row*ld + col for a row-major one.
    for (size_t col = 0; col < size[1]; ++col) {
        for (size_t row = 0; row < size[0]; ++row) {
            size_t idx = (layout == Layout::COL_MAJOR) ? (row + col * ld)
                                                        : (row * ld + col);
            bool ok;
            if constexpr (std::is_same_v<value_t, double>) {
                ok = std::fscanf(file_handle, "%lf ", &values[idx]) == 1;
            } else if constexpr (std::is_same_v<value_t, float>) {
                ok = std::fscanf(file_handle, "%f ", &values[idx]) == 1;
            } else {
                float temp = 0.0f;
                ok = std::fscanf(file_handle, "%f ", &temp) == 1;
                values[idx] = static_cast<value_t>(temp);
            }
            if (!ok) {
                std::fclose(file_handle);
                throw std::runtime_error("read_mtx_values: '" + filename +
                                         "' has truncated/malformed data");
            }
        }
    }
    std::fclose(file_handle);
}

template void read_mtx_values<double>(const std::string&, dim<2>, Layout,
                                      double*, size_t);
template void read_mtx_values<float>(const std::string&, dim<2>, Layout,
                                     float*, size_t);
template void read_mtx_values<__half>(const std::string&, dim<2>, Layout,
                                      __half*, size_t);
template void read_mtx_values<__nv_bfloat16>(const std::string&, dim<2>,
                                             Layout, __nv_bfloat16*, size_t);

}  // namespace tomos
