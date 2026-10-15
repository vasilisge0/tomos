#ifndef TOMOS_MALLOC_HPP
#define TOMOS_MALLOC_HPP

#include "device.hpp"
#include "status.hpp"

namespace tomos {

template <tomos::Device device, typename value_t>
Status malloc(size_t len, value_t **values);

template <tomos::Device device, typename value_t>
Status free(value_t *values);

}  // namespace tomos

#endif  // TOMOS_MALLOC_HPP
