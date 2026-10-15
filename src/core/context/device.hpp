#ifndef TOMOS_DEVICE_HPP
#define TOMOS_DEVICE_HPP

#include <cstddef>

namespace tomos {

// enum class Device : std::size_t { CPU = 0, CUDA = 1, Count };
enum class Device : std::size_t { CPU = 0, OMP = 1, CUDA = 2, Count };

}  // namespace tomos

#endif
