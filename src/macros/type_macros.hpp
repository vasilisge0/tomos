#ifndef TOMOS_TYPE_MACROS_HPP
#define TOMOS_TYPE_MACROS_HPP

namespace tomos {

// Strip one layer of parentheses — used to pass comma-containing signatures as
// a single arg.
// clang-format off

#define REMOVE_PARENS(...) __VA_ARGS__

// ---------------------------------------------------------------------------
// Flat iterators — no nested macro calls, so no preprocessor blue-paint issue.
// Each calls F(T1, ..., TN, extra...) for all 3^N combinations.
//
// NOT USED anywhere in the codebase, down through INSTANTIATE_ALL below
// (its only public entry point) and all of INSTANTIATE_1/_2/_3 and the
// private _NARGS/_FOR_ALL_1..3/_INST_N_1..3/_INST_ALL_* dispatch helpers
// that build it. A parallel, superseded system to FOR_ALL_FP_SINGLES/
// _PAIRS/_TRIPLES further down (which real code does call directly) --
// this one iterates {double, float, __half, uint32_t} instead of
// {double, float, __half, __nv_bfloat16}, so it isn't even the same type
// set. Kept for now per user request (documented, not removed).
// ---------------------------------------------------------------------------
#define FOR_ALL_SINGLES(F, ...) \
    F(double, ##__VA_ARGS__)    \
    F(float, ##__VA_ARGS__)     \
    F(__half, ##__VA_ARGS__)    \
    F(uint32_t, ##__VA_ARGS__)

#define FOR_ALL_PAIRS(F, ...)          \
    F(double, double, ##__VA_ARGS__)   \
    F(double, float, ##__VA_ARGS__)    \
    F(double, __half, ##__VA_ARGS__)   \
    F(float, double, ##__VA_ARGS__)    \
    F(float, float, ##__VA_ARGS__)     \
    F(float, __half, ##__VA_ARGS__)    \
    F(__half, double, ##__VA_ARGS__)   \
    F(__half, float, ##__VA_ARGS__)    \
    F(__half, __half, ##__VA_ARGS__)   \
    F(uint32_t, double, ##__VA_ARGS__) \
    F(uint32_t, float, ##__VA_ARGS__)  \
    F(uint32_t, __half, ##__VA_ARGS__)

#define FOR_ALL_TRIPLES(F, ...)                \
    F(double, double, double, ##__VA_ARGS__)   \
    F(double, double, float, ##__VA_ARGS__)    \
    F(double, double, __half, ##__VA_ARGS__)   \
    F(double, float, double, ##__VA_ARGS__)    \
    F(double, float, float, ##__VA_ARGS__)     \
    F(double, float, __half, ##__VA_ARGS__)    \
    F(double, __half, double, ##__VA_ARGS__)   \
    F(double, __half, float, ##__VA_ARGS__)    \
    F(double, __half, __half, ##__VA_ARGS__)   \
    F(float, double, double, ##__VA_ARGS__)    \
    F(float, double, float, ##__VA_ARGS__)     \
    F(float, double, __half, ##__VA_ARGS__)    \
    F(float, float, double, ##__VA_ARGS__)     \
    F(float, float, float, ##__VA_ARGS__)      \
    F(float, float, __half, ##__VA_ARGS__)     \
    F(float, __half, double, ##__VA_ARGS__)    \
    F(float, __half, float, ##__VA_ARGS__)     \
    F(float, __half, __half, ##__VA_ARGS__)    \
    F(__half, double, double, ##__VA_ARGS__)   \
    F(__half, double, float, ##__VA_ARGS__)    \
    F(__half, double, __half, ##__VA_ARGS__)   \
    F(__half, float, double, ##__VA_ARGS__)    \
    F(__half, float, float, ##__VA_ARGS__)     \
    F(__half, float, __half, ##__VA_ARGS__)    \
    F(__half, __half, double, ##__VA_ARGS__)   \
    F(__half, __half, float, ##__VA_ARGS__)    \
    F(__half, __half, __half, ##__VA_ARGS__)   \
    F(uint32_t, double, double, ##__VA_ARGS__) \
    F(uint32_t, double, float, ##__VA_ARGS__)  \
    F(uint32_t, double, __half, ##__VA_ARGS__) \
    F(uint32_t, float, double, ##__VA_ARGS__)  \
    F(uint32_t, float, float, ##__VA_ARGS__)   \
    F(uint32_t, float, __half, ##__VA_ARGS__)  \
    F(uint32_t, __half, double, ##__VA_ARGS__) \
    F(uint32_t, __half, float, ##__VA_ARGS__)  \
    F(uint32_t, __half, __half, ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// Explicit instantiation helpers.
// Pass the parameter list wrapped in () to avoid comma ambiguity.
// ---------------------------------------------------------------------------
#define INSTANTIATE_1(T1, func, sig) template void func<T1>(REMOVE_PARENS sig);
#define INSTANTIATE_2(T1, T2, func, sig) \
    template void func<T1, T2>(REMOVE_PARENS sig);
#define INSTANTIATE_3(T1, T2, T3, func, sig) \
    template void func<T1, T2, T3>(REMOVE_PARENS sig);

// ---------------------------------------------------------------------------
// INSTANTIATE_ALL — counts placeholder args to determine template arity.
//
// Usage (placeholder names should match the function's template param names):
//   INSTANTIATE_ALL(func, (sig), value_t)          // 1 param  →  3
//   instantiations INSTANTIATE_ALL(func, (sig), T1, T2)           // 2 params →
//   9 instantiations INSTANTIATE_ALL(func, (sig), A_T, X_T, Y_T)   // 3 params
//   → 27 instantiations
// ---------------------------------------------------------------------------
#define _NARGS_IMPL(_1, _2, _3, N, ...) N
#define _NARGS(...) _NARGS_IMPL(__VA_ARGS__, 3, 2, 1, ~)

#define _FOR_ALL_1 FOR_ALL_SINGLES
#define _FOR_ALL_2 FOR_ALL_PAIRS
#define _FOR_ALL_3 FOR_ALL_TRIPLES
#define _INST_N_1 INSTANTIATE_1
#define _INST_N_2 INSTANTIATE_2
#define _INST_N_3 INSTANTIATE_3

#define _INST_ALL_EXPAND(N, func, sig) _FOR_ALL_##N(_INST_N_##N, func, sig)
#define _INST_ALL_DISPATCH(N, func, sig) _INST_ALL_EXPAND(N, func, sig)
#define INSTANTIATE_ALL(func, sig, ...) \
    _INST_ALL_DISPATCH(_NARGS(__VA_ARGS__), func, sig)

// ---------------------------------------------------------------------------
// FP-only iterators (double, float, __half) and INSTANTIATE_ALL_FP.
// ---------------------------------------------------------------------------
#define FOR_ALL_FP_SINGLES(F, ...) \
    F(double, ##__VA_ARGS__) \
    F(float, ##__VA_ARGS__) \
    F(__half, ##__VA_ARGS__) \
    F(__nv_bfloat16, ##__VA_ARGS__)

#define FOR_ALL_FP_PAIRS(F, ...) \
    F(double, double, ##__VA_ARGS__) \
    F(double, float, ##__VA_ARGS__) \
    F(double, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, ##__VA_ARGS__) \
    F(float, float, ##__VA_ARGS__) \
    F(float, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, ##__VA_ARGS__) \
    F(__half, float, ##__VA_ARGS__) \
    F(__half, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__)

#define FOR_ALL_FP_TRIPLES(F, ...) \
    F(double, double, double, ##__VA_ARGS__) \
    F(double, double, float, ##__VA_ARGS__) \
    F(double, double, __half, ##__VA_ARGS__) \
    F(double, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, float, double, ##__VA_ARGS__) \
    F(double, float, float, ##__VA_ARGS__) \
    F(double, float, __half, ##__VA_ARGS__) \
    F(double, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __half, double, ##__VA_ARGS__) \
    F(double, __half, float, ##__VA_ARGS__) \
    F(double, __half, __half, ##__VA_ARGS__) \
    F(double, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, double, ##__VA_ARGS__) \
    F(float, double, float, ##__VA_ARGS__) \
    F(float, double, __half, ##__VA_ARGS__) \
    F(float, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, float, double, ##__VA_ARGS__) \
    F(float, float, float, ##__VA_ARGS__) \
    F(float, float, __half, ##__VA_ARGS__) \
    F(float, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __half, double, ##__VA_ARGS__) \
    F(float, __half, float, ##__VA_ARGS__) \
    F(float, __half, __half, ##__VA_ARGS__) \
    F(float, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, double, ##__VA_ARGS__) \
    F(__half, double, float, ##__VA_ARGS__) \
    F(__half, double, __half, ##__VA_ARGS__) \
    F(__half, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, float, double, ##__VA_ARGS__) \
    F(__half, float, float, ##__VA_ARGS__) \
    F(__half, float, __half, ##__VA_ARGS__) \
    F(__half, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __half, double, ##__VA_ARGS__) \
    F(__half, __half, float, ##__VA_ARGS__) \
    F(__half, __half, __half, ##__VA_ARGS__) \
    F(__half, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__)

// FOR_ALL_FP_QUADRUPLES(F) — calls F(T1, T2, T3, T4) for all 3^4 = 81
// combinations. Used by LSQR's multiprecision registry/instantiations
// (refine_t, value_t, mv_t, precond_t independently selectable).
#define FOR_ALL_FP_QUADRUPLES(F, ...) \
    F(double, double, double, double, ##__VA_ARGS__) \
    F(double, double, double, float, ##__VA_ARGS__) \
    F(double, double, double, __half, ##__VA_ARGS__) \
    F(double, double, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, double, float, double, ##__VA_ARGS__) \
    F(double, double, float, float, ##__VA_ARGS__) \
    F(double, double, float, __half, ##__VA_ARGS__) \
    F(double, double, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, double, __half, double, ##__VA_ARGS__) \
    F(double, double, __half, float, ##__VA_ARGS__) \
    F(double, double, __half, __half, ##__VA_ARGS__) \
    F(double, double, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, double, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(double, double, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(double, double, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(double, double, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, float, double, double, ##__VA_ARGS__) \
    F(double, float, double, float, ##__VA_ARGS__) \
    F(double, float, double, __half, ##__VA_ARGS__) \
    F(double, float, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, float, float, double, ##__VA_ARGS__) \
    F(double, float, float, float, ##__VA_ARGS__) \
    F(double, float, float, __half, ##__VA_ARGS__) \
    F(double, float, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, float, __half, double, ##__VA_ARGS__) \
    F(double, float, __half, float, ##__VA_ARGS__) \
    F(double, float, __half, __half, ##__VA_ARGS__) \
    F(double, float, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, float, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(double, float, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(double, float, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(double, float, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __half, double, double, ##__VA_ARGS__) \
    F(double, __half, double, float, ##__VA_ARGS__) \
    F(double, __half, double, __half, ##__VA_ARGS__) \
    F(double, __half, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __half, float, double, ##__VA_ARGS__) \
    F(double, __half, float, float, ##__VA_ARGS__) \
    F(double, __half, float, __half, ##__VA_ARGS__) \
    F(double, __half, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __half, __half, double, ##__VA_ARGS__) \
    F(double, __half, __half, float, ##__VA_ARGS__) \
    F(double, __half, __half, __half, ##__VA_ARGS__) \
    F(double, __half, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __half, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(double, __half, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(double, __half, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(double, __half, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, double, double, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, double, float, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, double, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, float, double, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, float, float, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, float, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __half, double, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __half, float, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __half, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, double, double, ##__VA_ARGS__) \
    F(float, double, double, float, ##__VA_ARGS__) \
    F(float, double, double, __half, ##__VA_ARGS__) \
    F(float, double, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, float, double, ##__VA_ARGS__) \
    F(float, double, float, float, ##__VA_ARGS__) \
    F(float, double, float, __half, ##__VA_ARGS__) \
    F(float, double, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, __half, double, ##__VA_ARGS__) \
    F(float, double, __half, float, ##__VA_ARGS__) \
    F(float, double, __half, __half, ##__VA_ARGS__) \
    F(float, double, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, double, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(float, double, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(float, double, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(float, double, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, float, double, double, ##__VA_ARGS__) \
    F(float, float, double, float, ##__VA_ARGS__) \
    F(float, float, double, __half, ##__VA_ARGS__) \
    F(float, float, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, float, float, double, ##__VA_ARGS__) \
    F(float, float, float, float, ##__VA_ARGS__) \
    F(float, float, float, __half, ##__VA_ARGS__) \
    F(float, float, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, float, __half, double, ##__VA_ARGS__) \
    F(float, float, __half, float, ##__VA_ARGS__) \
    F(float, float, __half, __half, ##__VA_ARGS__) \
    F(float, float, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, float, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(float, float, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(float, float, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(float, float, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __half, double, double, ##__VA_ARGS__) \
    F(float, __half, double, float, ##__VA_ARGS__) \
    F(float, __half, double, __half, ##__VA_ARGS__) \
    F(float, __half, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __half, float, double, ##__VA_ARGS__) \
    F(float, __half, float, float, ##__VA_ARGS__) \
    F(float, __half, float, __half, ##__VA_ARGS__) \
    F(float, __half, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __half, __half, double, ##__VA_ARGS__) \
    F(float, __half, __half, float, ##__VA_ARGS__) \
    F(float, __half, __half, __half, ##__VA_ARGS__) \
    F(float, __half, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __half, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(float, __half, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(float, __half, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(float, __half, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, double, double, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, double, float, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, double, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, float, double, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, float, float, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, float, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __half, double, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __half, float, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __half, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, double, double, ##__VA_ARGS__) \
    F(__half, double, double, float, ##__VA_ARGS__) \
    F(__half, double, double, __half, ##__VA_ARGS__) \
    F(__half, double, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, float, double, ##__VA_ARGS__) \
    F(__half, double, float, float, ##__VA_ARGS__) \
    F(__half, double, float, __half, ##__VA_ARGS__) \
    F(__half, double, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, __half, double, ##__VA_ARGS__) \
    F(__half, double, __half, float, ##__VA_ARGS__) \
    F(__half, double, __half, __half, ##__VA_ARGS__) \
    F(__half, double, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, double, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__half, double, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__half, double, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__half, double, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, float, double, double, ##__VA_ARGS__) \
    F(__half, float, double, float, ##__VA_ARGS__) \
    F(__half, float, double, __half, ##__VA_ARGS__) \
    F(__half, float, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, float, float, double, ##__VA_ARGS__) \
    F(__half, float, float, float, ##__VA_ARGS__) \
    F(__half, float, float, __half, ##__VA_ARGS__) \
    F(__half, float, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, float, __half, double, ##__VA_ARGS__) \
    F(__half, float, __half, float, ##__VA_ARGS__) \
    F(__half, float, __half, __half, ##__VA_ARGS__) \
    F(__half, float, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, float, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__half, float, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__half, float, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__half, float, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __half, double, double, ##__VA_ARGS__) \
    F(__half, __half, double, float, ##__VA_ARGS__) \
    F(__half, __half, double, __half, ##__VA_ARGS__) \
    F(__half, __half, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __half, float, double, ##__VA_ARGS__) \
    F(__half, __half, float, float, ##__VA_ARGS__) \
    F(__half, __half, float, __half, ##__VA_ARGS__) \
    F(__half, __half, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __half, __half, double, ##__VA_ARGS__) \
    F(__half, __half, __half, float, ##__VA_ARGS__) \
    F(__half, __half, __half, __half, ##__VA_ARGS__) \
    F(__half, __half, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __half, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__half, __half, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__half, __half, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__half, __half, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, double, double, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, double, float, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, double, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, float, double, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, float, float, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, float, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __half, double, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __half, float, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __half, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, double, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, double, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, double, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, float, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, float, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, float, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __half, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __half, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __half, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, double, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, double, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, double, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, float, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, float, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, float, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __half, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __half, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __half, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, double, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, double, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, double, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, float, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, float, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, float, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __half, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __half, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __half, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, double, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, double, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, double, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, double, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, float, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, float, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, float, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __half, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __half, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __half, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __half, __nv_bfloat16, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, double, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __half, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ##__VA_ARGS__)

// FOR_ALL_FP_QUADRUPLES_MV_U32(F) -- like FOR_ALL_FP_QUADRUPLES but with the
// third slot (MvT, LSQR's convention for A's own storage/matvec type) fixed
// to uint32_t -- NumericType::FP32_16's raw packed storage element type
// (see fp32_16_pack.hpp) -- instead of ranging over FPTypes itself, since
// FP32_16 isn't a single arithmetic C++ type. MvT only drives buffer-sizing
// (as_strided_view<MvT>(A).size_) in dispatch::Lsqr's registry, never
// arithmetic (see lsqr_registries.cpp's own comment), so uint32_t is a safe
// stand-in here, same convention FOR_ALL_TRIPLES/FOR_ALL_SINGLES already use
// for FP32_16 elsewhere in this codebase. Reuses FOR_ALL_FP_TRIPLES's
// existing (RefineT, ValueT, PrecondT) enumeration (64 combinations) rather
// than duplicating it.
#define _QUAD_MV_U32_WRAP(RefineT, ValueT, PrecondT, F, ...) \
    F(RefineT, ValueT, uint32_t, PrecondT, ##__VA_ARGS__)
#define FOR_ALL_FP_QUADRUPLES_MV_U32(F, ...) \
    FOR_ALL_FP_TRIPLES(_QUAD_MV_U32_WRAP, F, ##__VA_ARGS__)

// NOT USED anywhere in the codebase. FOR_ALL_FP_SINGLES/_PAIRS/_TRIPLES
// themselves (aliased below as the building blocks for this wrapper) are
// genuinely used elsewhere -- but always called directly, never through
// INSTANTIATE_ALL_FP itself. Kept for now per user request (documented,
// not removed).
#define _FOR_ALL_FP_1 FOR_ALL_FP_SINGLES
#define _FOR_ALL_FP_2 FOR_ALL_FP_PAIRS
#define _FOR_ALL_FP_3 FOR_ALL_FP_TRIPLES

#define _INST_FP_ALL_EXPAND(N, func, sig) \
    _FOR_ALL_FP_##N(_INST_N_##N, func, sig)
#define _INST_FP_ALL_DISPATCH(N, func, sig) _INST_FP_ALL_EXPAND(N, func, sig)
#define INSTANTIATE_ALL_FP(func, sig, ...) \
    _INST_FP_ALL_DISPATCH(_NARGS(__VA_ARGS__), func, sig)

// NOT USED anywhere in the codebase -- no callers, external or internal to
// this file (contrast FOR_ALL_INDEX_PAIRS right below, which is used).
// Kept for now per user request (documented, not removed).
#define FOR_ALL_INDEX_TYPES(F, ...) \
    F(int32_t, ##__VA_ARGS__)       \
    F(int64_t, ##__VA_ARGS__)

#define FOR_ALL_INDEX_PAIRS(F, ...)    \
    F(int32_t, int32_t, ##__VA_ARGS__) \
    F(int32_t, int64_t, ##__VA_ARGS__) \
    F(int64_t, int32_t, ##__VA_ARGS__) \
    F(int64_t, int64_t, ##__VA_ARGS__)

#define FOR_ALL_FP_PAIRS_INDEX_PAIRS(F, ...) \
    F(double, double, int32_t, int32_t, ##__VA_ARGS__) \
    F(double, double, int32_t, int64_t, ##__VA_ARGS__) \
    F(double, double, int64_t, int32_t, ##__VA_ARGS__) \
    F(double, double, int64_t, int64_t, ##__VA_ARGS__) \
    F(double, float, int32_t, int32_t, ##__VA_ARGS__) \
    F(double, float, int32_t, int64_t, ##__VA_ARGS__) \
    F(double, float, int64_t, int32_t, ##__VA_ARGS__) \
    F(double, float, int64_t, int64_t, ##__VA_ARGS__) \
    F(double, __half, int32_t, int32_t, ##__VA_ARGS__) \
    F(double, __half, int32_t, int64_t, ##__VA_ARGS__) \
    F(double, __half, int64_t, int32_t, ##__VA_ARGS__) \
    F(double, __half, int64_t, int64_t, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, int32_t, int32_t, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, int32_t, int64_t, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, int64_t, int32_t, ##__VA_ARGS__) \
    F(double, __nv_bfloat16, int64_t, int64_t, ##__VA_ARGS__) \
    F(float, double, int32_t, int32_t, ##__VA_ARGS__) \
    F(float, double, int32_t, int64_t, ##__VA_ARGS__) \
    F(float, double, int64_t, int32_t, ##__VA_ARGS__) \
    F(float, double, int64_t, int64_t, ##__VA_ARGS__) \
    F(float, float, int32_t, int32_t, ##__VA_ARGS__) \
    F(float, float, int32_t, int64_t, ##__VA_ARGS__) \
    F(float, float, int64_t, int32_t, ##__VA_ARGS__) \
    F(float, float, int64_t, int64_t, ##__VA_ARGS__) \
    F(float, __half, int32_t, int32_t, ##__VA_ARGS__) \
    F(float, __half, int32_t, int64_t, ##__VA_ARGS__) \
    F(float, __half, int64_t, int32_t, ##__VA_ARGS__) \
    F(float, __half, int64_t, int64_t, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, int32_t, int32_t, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, int32_t, int64_t, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, int64_t, int32_t, ##__VA_ARGS__) \
    F(float, __nv_bfloat16, int64_t, int64_t, ##__VA_ARGS__) \
    F(__half, double, int32_t, int32_t, ##__VA_ARGS__) \
    F(__half, double, int32_t, int64_t, ##__VA_ARGS__) \
    F(__half, double, int64_t, int32_t, ##__VA_ARGS__) \
    F(__half, double, int64_t, int64_t, ##__VA_ARGS__) \
    F(__half, float, int32_t, int32_t, ##__VA_ARGS__) \
    F(__half, float, int32_t, int64_t, ##__VA_ARGS__) \
    F(__half, float, int64_t, int32_t, ##__VA_ARGS__) \
    F(__half, float, int64_t, int64_t, ##__VA_ARGS__) \
    F(__half, __half, int32_t, int32_t, ##__VA_ARGS__) \
    F(__half, __half, int32_t, int64_t, ##__VA_ARGS__) \
    F(__half, __half, int64_t, int32_t, ##__VA_ARGS__) \
    F(__half, __half, int64_t, int64_t, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, int32_t, int32_t, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, int32_t, int64_t, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, int64_t, int32_t, ##__VA_ARGS__) \
    F(__half, __nv_bfloat16, int64_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, int32_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, int32_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, int64_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, double, int64_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, int32_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, int32_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, int64_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, float, int64_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, int32_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, int32_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, int64_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __half, int64_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, int32_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, int32_t, int64_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, int64_t, int32_t, ##__VA_ARGS__) \
    F(__nv_bfloat16, __nv_bfloat16, int64_t, int64_t, ##__VA_ARGS__)

#define FOR_ALL_DEVICES(F, ...)   \
    F(Device::CPU, ##__VA_ARGS__) \
    F(Device::CUDA, ##__VA_ARGS__)

#define FOR_ALL_DEVICES_AND_TYPES(F, ...)   \
    F(Device::CPU,  double,   ##__VA_ARGS__) \
    F(Device::CPU,  float,    ##__VA_ARGS__) \
    F(Device::CPU,  __half,   ##__VA_ARGS__) \
    F(Device::CPU,  __nv_bfloat16, ##__VA_ARGS__) \
    F(Device::CPU,  uint32_t, ##__VA_ARGS__) \
    F(Device::CUDA, double,   ##__VA_ARGS__) \
    F(Device::CUDA, float,    ##__VA_ARGS__) \
    F(Device::CUDA, __half,   ##__VA_ARGS__) \
    F(Device::CUDA, __nv_bfloat16, ##__VA_ARGS__) \
    F(Device::CUDA, uint32_t, ##__VA_ARGS__)

#define _INST_DEVICE_TYPE(device, value_t, func, sig) \
    template void func<device, value_t>(REMOVE_PARENS sig);

#define INSTANTIATE_ALL_DEVICES_AND_TYPES(func, sig) \
    FOR_ALL_DEVICES(_INST_DEVICE_TYPE_OUTER, func, sig)

#define _INST_DEVICE_TYPE_OUTER(device, func, sig) \
    _INST_DEVICE_TYPE(device, double, func, sig)   \
    _INST_DEVICE_TYPE(device, float, func, sig)    \
    _INST_DEVICE_TYPE(device, __half, func, sig)   \
    _INST_DEVICE_TYPE(device, __nv_bfloat16, func, sig) \
    _INST_DEVICE_TYPE(device, uint32_t, func, sig)

#define _INST_DTI(device, value_t, index_t, func, sig) \
    template void func<device, value_t, index_t>(REMOVE_PARENS sig);

#define _INST_DTI_INDEX(index_t, device, value_t, func, sig) \
    _INST_DTI(device, value_t, index_t, func, sig)

#define _INST_DTI_TYPE(value_t, device, func, sig)       \
    _INST_DTI_INDEX(int32_t, device, value_t, func, sig) \
    _INST_DTI_INDEX(int64_t, device, value_t, func, sig)

#define _INST_DTI_DEVICE(device, func, sig)   \
    _INST_DTI_TYPE(double, device, func, sig) \
    _INST_DTI_TYPE(float, device, func, sig)  \
    _INST_DTI_TYPE(__half, device, func, sig) \
    _INST_DTI_TYPE(__nv_bfloat16, device, func, sig) \
    _INST_DTI_TYPE(uint32_t, device, func, sig)

#define INSTANTIATE_ALL_DEVICES_TYPES_INDEX(func, sig) \
    FOR_ALL_DEVICES(_INST_DTI_DEVICE, func, sig)

// ---------------------------------------------------------------------------
// Status-returning functions: template <Device device, typename value_t>.
// Use when the parameter list itself contains the value type (e.g. T**).
//
// sig_macro must be a single-argument macro: sig_macro(T) that expands to a
// parenthesized parameter list, e.g.:
//   #define MALLOC_SIG(T) (size_t len, T** values)
//   INSTANTIATE_STATUS_ALL_DEVICES_AND_TYPES(malloc, MALLOC_SIG)
//   #undef MALLOC_SIG
//
// The inner macro forces sig_macro(value_t) to be evaluated as an argument
// (and thus expanded) before REMOVE_PARENS sees it.
//
// NOT USED anywhere in the codebase, despite the malloc example right above
// -- malloc.cpp actually instantiates via FOR_ALL_DEVICES_AND_TYPES(MALLOC)
// directly instead, a different, simpler mechanism. This whole macro and
// its private _INST_STATUS_DV*/_INST_STATUS_DV_OUTER helpers are dead; the
// doc comment's worked example was never followed. Kept for now per user
// request (documented, not removed).
// ---------------------------------------------------------------------------
#define _INST_STATUS_DV_INNER(device, value_t, func, sig) \
    template Status func<device, value_t>(REMOVE_PARENS sig);

#define _INST_STATUS_DV(device, value_t, func, sig_macro) \
    _INST_STATUS_DV_INNER(device, value_t, func, sig_macro(value_t))

#define _INST_STATUS_DV_OUTER(device, func, sig_macro) \
    _INST_STATUS_DV(device, double, func, sig_macro)   \
    _INST_STATUS_DV(device, float, func, sig_macro)    \
    _INST_STATUS_DV(device, __half, func, sig_macro)   \
    _INST_STATUS_DV(device, uint32_t, func, sig_macro)

#define INSTANTIATE_STATUS_ALL_DEVICES_AND_TYPES(func, sig_macro) \
    FOR_ALL_DEVICES(_INST_STATUS_DV_OUTER, func, sig_macro)

// ---------------------------------------------------------------------------
// INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS
// For template<Device ds, Device dt, typename vi, typename vo>
// Generates 3 * 3 * 3 * 3 = 81 instantiations.
// ---------------------------------------------------------------------------
#define _INST_DPDP(ds, dt, vi, vo, func, sig) \
    template void func<ds, dt, vi, vo>(REMOVE_PARENS sig);

#define _INST_DPDP_VIN(vi, ds, dt, func, sig)      \
    _INST_DPDP(ds, dt, vi, double, func, sig)      \
    _INST_DPDP(ds, dt, vi, float, func, sig)       \
    _INST_DPDP(ds, dt, vi, __half, func, sig)      \
    _INST_DPDP(ds, dt, vi, __nv_bfloat16, func, sig)

#define _INST_DPDP_DTAR(dt, ds, func, sig)         \
    _INST_DPDP_VIN(double, ds, dt, func, sig)      \
    _INST_DPDP_VIN(float, ds, dt, func, sig)       \
    _INST_DPDP_VIN(__half, ds, dt, func, sig)      \
    _INST_DPDP_VIN(__nv_bfloat16, ds, dt, func, sig)

#define _INST_DPDP_DSRC(ds, func, sig)          \
    _INST_DPDP_DTAR(Device::CPU, ds, func, sig) \
    _INST_DPDP_DTAR(Device::OMP, ds, func, sig) \
    _INST_DPDP_DTAR(Device::CUDA, ds, func, sig)

#define INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS(func, sig) \
    FOR_ALL_DEVICES(_INST_DPDP_DSRC, func, sig)

// ---------------------------------------------------------------------------
// INSTANTIATE_ALL_FP_PAIRS_INDEX_PAIRS
// For template<typename vi, typename vo, typename ii, typename io>
// Generates 3 * 3 * 2 * 2 = 36 instantiations.
//
// NOT USED anywhere in the codebase, including its private _INST_FPIP*
// helpers (self-contained, only used by this macro). Kept for now per user
// request (documented, not removed).
// ---------------------------------------------------------------------------
#define _INST_FPIP(vi, vo, ii, io, func, sig) \
    template void func<vi, vo, ii, io>(REMOVE_PARENS sig);

#define _INST_FPIP_IOUT(io, vi, vo, ii, func, sig) \
    _INST_FPIP(vi, vo, ii, io, func, sig)

#define _INST_FPIP_IIN(ii, vi, vo, func, sig)       \
    _INST_FPIP_IOUT(int32_t, vi, vo, ii, func, sig) \
    _INST_FPIP_IOUT(int64_t, vi, vo, ii, func, sig)

#define _INST_FPIP_VOUT(vo, vi, func, sig)     \
    _INST_FPIP_IIN(int32_t, vi, vo, func, sig) \
    _INST_FPIP_IIN(int64_t, vi, vo, func, sig)

#define _INST_FPIP_VIN(vi, func, sig)      \
    _INST_FPIP_VOUT(double, vi, func, sig) \
    _INST_FPIP_VOUT(float, vi, func, sig)  \
    _INST_FPIP_VOUT(__half, vi, func, sig)

#define INSTANTIATE_ALL_FP_PAIRS_INDEX_PAIRS(func, sig) \
    _INST_FPIP_VIN(double, func, sig)                   \
    _INST_FPIP_VIN(float, func, sig)                    \
    _INST_FPIP_VIN(__half, func, sig)

}  // namespace tomos

// clang-format on

#endif  // TOMOS_TYPE_MACROS_HPP
