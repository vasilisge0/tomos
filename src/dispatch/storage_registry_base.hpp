#ifndef TOMOS_STORAGE_REGISTRY_BASE_HPP
#define TOMOS_STORAGE_REGISTRY_BASE_HPP

#include <type_traits>

#include "device.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"

namespace tomos {
namespace dispatch {

static constexpr std::size_t num_registry_formats =
    static_cast<std::size_t>(DenseFormat::Count);

static constexpr std::size_t num_registry_devices =
    static_cast<std::size_t>(Device::Count);

static constexpr std::size_t num_registry_numeric_types =
    static_cast<std::size_t>(NumericType::Count);

static constexpr std::size_t num_registry_index_types =
    static_cast<std::size_t>(IntType::Count);

// ---------------------------------------------------------------------------
// Type-list and value-list utilities for registry population.
//
// for_each_type  — iterates TypeList<Ts...>, calls f(Tag<T>{}) per type.
//   Usage: for_each_type(FPTypes{}, [&](auto tag) {
//              using T = typename decltype(tag)::type; ... });
//
// for_each_value — iterates ValueList<Vs...>, calls f(integral_constant) per
//                  value. Access the enum value with decltype(tag)::value.
//   Usage: for_each_value(ActiveDevices{}, [&](auto tag) {
//              constexpr Device dev = decltype(tag)::value; ... });
// ---------------------------------------------------------------------------
template <typename T>
struct Tag {
    using type = T;
};

template <typename... Ts>
struct TypeList {};

template <typename F, typename... Ts>
constexpr void for_each_type(TypeList<Ts...>, F f) {
    (f(Tag<Ts>{}), ...);
}

template <auto... Vs>
struct ValueList {};

template <typename F, auto... Vs>
constexpr void for_each_value(ValueList<Vs...>, F f) {
    (f(std::integral_constant<decltype(Vs), Vs>{}), ...);
}

// Mapping IntType enum → C++ index type (analogous to NumericTypeTrait).
template <IntType I>
struct IndexTypeTrait;
template <>
struct IndexTypeTrait<IntType::INT32> {
    using type = int32_t;
};
template <>
struct IndexTypeTrait<IntType::INT64> {
    using type = int64_t;
};
template <IntType I>
using to_concrete_type = typename IndexTypeTrait<I>::type;

// Pre-defined lists covering the supported FP types, devices, and index types.
using FPTypes = TypeList<double, float, __half, __nv_bfloat16>;
#if USE_CUDA
using ActiveDevices = ValueList<Device::CPU, Device::CUDA>;
#else
using ActiveDevices = ValueList<Device::CPU>;
#endif
using IndexTypes = ValueList<IntType::INT32, IntType::INT64>;

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_STORAGE_REGISTRY_BASE_HPP
