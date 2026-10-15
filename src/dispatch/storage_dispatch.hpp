#ifndef TOMOS_STORAGE_DISPATCH_HPP
#define TOMOS_STORAGE_DISPATCH_HPP

#include <string>

#include "dense_storage.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"

namespace tomos {
namespace dispatch {

NumericType get_numeric_type(DenseStorage& source);
NumericType get_numeric_type(SparseStorage& source);

IntType get_index_type(SparseStorage& source);

dim<2> get_size(DenseStorage& source);
dim<2> get_size(SparseStorage& source);

void initialize_dense_storage(dim<2> size, Device dev, DenseStorage& source);
void free_dense_storage(Device dev, DenseStorage& source);
void copy_dense_storage(dim<2> size, Device source_dev, Device target_dev,
                        DenseStorage& source, DenseStorage& target);
void read_dense_storage(Device device, DenseStorage& source,
                        std::string& filename);
void read_write_storage(Device device, DenseStorage& source,
                        std::string& filename);

void initialize_sparse_storage(Device dev, SparseStorage& source);
void free_sparse_storage(Device dev, SparseStorage& source);
void copy_sparse_storage(dim<2> size, Device source_dev, Device target_dev,
                         SparseStorage& source, SparseStorage& target);

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_STORAGE_DISPATCH_HPP
