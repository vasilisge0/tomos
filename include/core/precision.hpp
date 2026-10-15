#ifndef TOMOS_PRECISIONS_HPP
#define TOMOS_PRECISIONS_HPP

#include <cuda_fp16.h>
#include <cstdint>
#include <variant>

#include "malloc.hpp"

#if USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#elif defined(__clang__) || defined(__GNUC__)
// using __half = uint16_t;
#else
#error No NumericType::FP16 type supported by the compiler
#endif

namespace tomos {

using half = __half;
using bf16 = __nv_bfloat16;
using FloatPtrVariant = std::variant<double*, float*, __half*, __nv_bfloat16*>;
using IntPtrVariant = std::variant<int*, long int*>;
// Alternative order MUST match NumericType's own enum values exactly
// (double*=FP64=0, float*=FP32=1, __half*=FP16=2, uint32_t*=FP32_16=3,
// __nv_bfloat16*=BF16=4) -- dispatch::get_numeric_type()
// (src/dispatch/storage_dispatch.cpp) recovers a DenseStorage/SparseStorage's
// NumericType via static_cast<NumericType>(values_.index()), so a reordering
// here silently desyncs every FP32_16/BF16 Dense or Sparse ever built
// through Dense::create_strided/Sparse::create_csr (std::get<T> inside the
// wrong dispatch::Dense::lookup_initialize/_free/_copy entry then throws
// std::bad_variant_access) -- Dense::wrap_strided's own NumericPtrVariant
// construction (matrix.cpp's wrap_ptr, a type-keyed switch) is unaffected
// either way, which is why this went unnoticed until FP32_16 storage was
// first built via create_strided rather than wrap_strided.
using NumericPtrVariant =
    std::variant<double*, float*, __half*, uint32_t*, __nv_bfloat16*>;

enum class MatrixType : std::size_t { STRIDED = 0, CSR = 1, Count };
enum class IntType : std::size_t { INT32 = 0, INT64 = 1, Count };
enum class NumericType : std::size_t {
    FP64 = 0,
    FP32 = 1,
    FP16 = 2,
    FP32_16 = 3,
    BF16 = 4,
    Count
};
enum class Layout : std::size_t { ROW_MAJOR, COL_MAJOR };

template <typename T>
struct NumericTypeTrait;
template <>
struct NumericTypeTrait<double> {
    static constexpr NumericType value = NumericType::FP64;
};
template <>
struct NumericTypeTrait<float> {
    static constexpr NumericType value = NumericType::FP32;
};
template <>
struct NumericTypeTrait<__half> {
    static constexpr NumericType value = NumericType::FP16;
};
template <>
struct NumericTypeTrait<uint32_t> {
    static constexpr NumericType value = NumericType::FP32_16;
};
template <>
struct NumericTypeTrait<__nv_bfloat16> {
    static constexpr NumericType value = NumericType::BF16;
};
template <typename T>
constexpr NumericType numeric_type_of = NumericTypeTrait<T>::value;

template <NumericType N> struct NumericCppType;
template <> struct NumericCppType<NumericType::FP64> { using type = double; };
template <> struct NumericCppType<NumericType::FP32> { using type = float; };
template <> struct NumericCppType<NumericType::FP16> { using type = __half; };
template <> struct NumericCppType<NumericType::FP32_16> { using type = uint32_t; };
template <> struct NumericCppType<NumericType::BF16> { using type = __nv_bfloat16; };
template <NumericType N>
using numeric_cpp_type = typename NumericCppType<N>::type;

}  // namespace tomos

#endif  // TOMOS_PRECISIONS_HPP
