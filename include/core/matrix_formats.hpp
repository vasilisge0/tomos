#ifndef TOMOS_MATRIX_FORMATS_HPP
#define TOMOS_MATRIX_FORMATS_HPP

#include "precision.hpp"

enum class DenseFormat : std::size_t { STRIDED = 0, Count };
enum class SparseFormat : std::size_t { CSR = 1, Count };
enum class MatrixFormat : std::size_t { STRIDED = 0, CSR = 1, Count };

#endif  // TOMOS_MATRIX_FORMATS_HPP