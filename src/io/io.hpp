#ifndef TOMOS_IO_HPP
#define TOMOS_IO_HPP

#include <string>

#include "device.hpp"
#include "dimensions.hpp"
#include "precision.hpp"

namespace tomos {

// Reads just the array-size line of a MatrixMarket dense array file (the
// "%%MatrixMarket matrix array ..." banner plus its size line), without
// reading any of the actual values. Used to size a Dense's storage before
// allocating it. Throws std::runtime_error if the file can't be opened or
// isn't a valid MatrixMarket dense array file.
dim<2> read_mtx_size(const std::string& filename);

// Reads a MatrixMarket dense array file's values into an already-allocated
// buffer of dim<2> size, at the given layout and leading dimension. The file
// itself is always stored column-major (the MatrixMarket array format), so
// this always walks it column-major regardless of the target layout, and
// only the destination index (governed by layout/ld) changes accordingly --
// see the .cpp for the exact indexing. Throws std::runtime_error if the
// file's own declared size doesn't match `size`, can't be opened, or isn't a
// valid MatrixMarket dense array file.
template <typename value_t>
void read_mtx_values(const std::string& filename, dim<2> size, Layout layout,
                     value_t* values, size_t ld);

}  // namespace tomos

#endif  // TOMOS_IO_HPP
