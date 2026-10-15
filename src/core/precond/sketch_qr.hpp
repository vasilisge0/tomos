#ifndef RLS_SKETCH_QR_HPP
#define RLS_SKETCH_QR_HPP

#include "context.hpp"
#include "countsketch.hpp"
#include "device.hpp"

namespace tomos {

// Generates a QR-based sketch-and-precondition preconditioner in-place.
//
// On entry, sketched_mtx must already hold the sketch of the coefficient
// matrix (sketch * A), a k x n dense matrix with k >= n -- typically the
// result of apply_count_sketch(sketch, A, sketched_mtx). This function
// factors sketched_mtx via QR, inverts the resulting n x n upper-triangular
// R factor, and overwrites the leading n x n block of sketched_mtx with
// R^{-1}, ready to be used as a right-preconditioner (e.g. via
// precond_apply/mv_apply). `sketch` is consulted only for its device.
void generate_sketch_qr_precond(Sparse& sketch, Dense& sketched_mtx);

// template <Device D, typename value_t>
// struct dispatch_precond_apply
// {
//     transpose_t trans;
//     value_t alpha;
//     Dense<D>* v;
//     value_t beta;
//     Dense<D>* u;
//     template<typename M>
//     // Status operator()(M* mtx, value_t alpha)
//     Status operator()(M* mtx)
//     {
//         using T = std::decay_t<decltype(mtx)>;
//         if constexpr (std::is_same_v<T, dense<D>*>) {
//             //std::cout << "mtx_r_: " << '\n';
//             //print_mtx<D, value_t>({ 0, 13 }, { 0, 3 }, mtx);
//             mtx->template apply<value_t>(trans, alpha, v, beta, u);
//             return SUCCESS;
//         }
//         else if constexpr (std::is_same_v<T, Csr<D>*>) {
//             mtx->template apply<value_t>(trans, alpha, v, beta, u);
//             return SUCCESS;
//         }
//         return FAILURE;
//     }
// };  // end of dispatch_mtx_apply

}  // namespace tomos

#endif  // end of RLS_PRECOND_HPP
