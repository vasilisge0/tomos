#include "countsketch.hpp"

#include "malloc.hpp"
#include "sketch_dispatch.hpp"
// #include "countsketch_kernels.cuh"
// #include "countsketch_kernels.hpp"
// #include "csr.hpp"
// #include "malloc.hpp"
// #include "precisions.hpp"
// #include "print_mtx_kernels.cuh"
// #include <type_traits>

namespace tomos {

void generate_count_sketch(Sparse& sketch) {
    dispatch::generate_count_sketch_idxs(sketch);
    dispatch::generate_count_sketch_values(sketch);
}

void apply_count_sketch(Sparse& sketch, Dense& source, Dense& target) {
    dispatch::apply_count_sketch(sketch, source, target);
}

void apply_count_sketch(Sparse& sketch, Sparse& source, Dense& target) {
    dispatch::apply_count_sketch(sketch, source, target);
}

std::unique_ptr<Countsketch> Countsketch::create(
    std::shared_ptr<Context> context, Device device, dim<2> size,
    size_t nnz_per_row, IntType t, NumericType p, bool prehash) {
    Countsketch* tmp =
        new Countsketch(context, device, size, nnz_per_row, t, p, prehash);
    return std::unique_ptr<Countsketch>(tmp);
}

Countsketch::Countsketch(std::shared_ptr<Context> context, Device device,
                         dim<2> size, size_t nnz_per_row, IntType t,
                         NumericType p, bool prehash) {
    size_              = size;
    num_elems_         = size[0] * nnz_per_row;
    compute_precision_ = p;
    index_precision_   = t;
    prehash_enabled_   = prehash;
}

// Status Countsketch::apply(Matrix& source, DenseView& target) {
//    NumericType p = this->get_precision();
//    std::visit(
//        [&](auto &obj) {
//            using T =
//                typename
//                std::remove_pointer<std::decay_t<decltype(obj)>>::type;
//            // using T = std::decay_t<decltype(obj)>;
//            //  auto t = static_cast<Csr<device>*>(this);
//            if constexpr (std::is_same<T, Dense<device>>::value) {
//                if (p == C_FP64) {
//                    // t->template apply<double>(Transpose, 1.0, obj, 0.0,
//                    // sketched_mtx);
//                } else if (p == C_FP32) {
//                    if (obj->get_precision() == C_FP32) {
//                        // t->template apply<float>(Transpose, 1.0, obj, 0.0,
//                        // sketched_mtx);
//                        tomos::cuda::countsketch_apply_impl(
//                            obj->get_size(), obj->template
//                            get_values<float>(), obj->get_size()[0],
//                            sketched_mtx->get_size(), sketched_mtx->template
//                            get_values<float>(), sketched_mtx->get_size()[0]);
//                    } else if (obj->get_precision() == C_FP32_FP16) {
//                        // t->template apply<float>(Transpose, 1.0, obj, 0.0,
//                        // sketched_mtx);
//                        tomos::cuda::countsketch_apply_impl(
//                            obj->get_size(),
//                            (uint16_t *)obj->value_ptrs_ascending_[0],
//                            (uint16_t *)obj->value_ptrs_ascending_[1],
//                            obj->get_size()[0], sketched_mtx->get_size(),
//                            sketched_mtx->template get_values<float>(),
//                            sketched_mtx->get_size()[0]);
//                    }
//                } else if (p == C_FP16) {
//                    if (obj->get_precision() == C_FP16) {
//                        // t->template apply<float>(Transpose, 1.0, obj, 0.0,
//                        // sketched_mtx);
//                        if (sketched_mtx->get_precision() == C_FP16) {
//
//                            tomos::cuda::countsketch_apply_impl(
//                                obj->get_size(),
//                                obj->template get_values<__half>(),
//                                obj->get_size()[0], sketched_mtx->get_size(),
//                                sketched_mtx->template get_values<__half>(),
//                                sketched_mtx->get_size()[0]);
//                        } else if (sketched_mtx->get_precision() == C_FP32) {
//
//                            if (prehash_enabled_) {
//                                tomos::cuda::apply_countsketch_prehashed(
//                                    size_[0], size_[1], obj->get_size()[1],
//                                    ihash_, shash_,
//                                    obj->template get_values<__half>(),
//                                    obj->get_size()[0],
//                                    sketched_mtx->template
//                                    get_values<float>(),
//                                    sketched_mtx->get_size()[0]);
//
//                            } else {
//
//                                tomos::cuda::countsketch_apply_impl(
//                                    obj->get_size(),
//                                    obj->template get_values<__half>(),
//                                    obj->get_size()[0],
//                                    sketched_mtx->get_size(),
//                                    sketched_mtx->template
//                                    get_values<float>(),
//                                    sketched_mtx->get_size()[0]);
//                            }
//                        }
//                    } else if (obj->get_precision() == C_FP32_FP16) {
//                        // t->template apply<float>(Transpose, 1.0, obj, 0.0,
//                        // sketched_mtx);
//                        tomos::cuda::countsketch_apply_impl(
//                            obj->get_size(),
//                            (uint16_t *)obj->value_ptrs_ascending_[0],
//                            (uint16_t *)obj->value_ptrs_ascending_[1],
//                            obj->get_size()[0], sketched_mtx->get_size(),
//                            sketched_mtx->template get_values<__half>(),
//                            sketched_mtx->get_size()[0]);
//                    }
//                }
//            }
//        },
//        mtx);
//     return SUCCESS;
// }

}  // namespace tomos
