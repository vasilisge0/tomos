#include "gaussian_sketch_kernels.hpp"

#include <random>

#include "precision.hpp"

namespace tomos {
namespace cpu {

template <typename value_t>
Status generate_gaussian_values(dim<2> size, value_t* values, size_t ld) {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t j = 0; j < size[1]; ++j)
        for (size_t i = 0; i < size[0]; ++i)
            values[i + j * ld] = static_cast<value_t>(dist(gen));
    return SUCCESS;
}

template Status generate_gaussian_values<double>(dim<2> size, double* values,
                                                  size_t ld);
template Status generate_gaussian_values<float>(dim<2> size, float* values,
                                                size_t ld);
template Status generate_gaussian_values<__half>(dim<2> size, __half* values,
                                                 size_t ld);
template Status generate_gaussian_values<__nv_bfloat16>(dim<2> size, __nv_bfloat16* values,
                                                 size_t ld);

}  // namespace cpu
}  // namespace tomos
