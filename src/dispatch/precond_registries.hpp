#ifndef TOMOS_PRECOND_REGISTRIES_HPP
#define TOMOS_PRECOND_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "storage_registry_base.hpp"
#include "transpose.hpp"

namespace tomos {
namespace dispatch {

// NOT CALLED FROM ANYWHERE CURRENTLY. The Precond registry and both
// apply_precond() overloads below are compiled (see CMakeLists.txt) and
// genuinely populated -- apply_trsv (precond_registries.cpp) is a real,
// working in-place upper-triangular solve -- but nothing in the codebase
// (library, benchmarks, or tests) actually calls Precond::get()/
// lookup_apply() or either apply_precond() overload. This looks like the
// dispatch-layer half of an earlier, abandoned "store R, solve R*y=x via
// TRSV at every apply" design for the preconditioner, superseded by the one
// actually in use: generate_sketch_qr_precond (src/core/precond/sketch_qr.
// hpp) inverts R once up front, so applying the precond later is just a
// plain matvec via precond_apply() (blas.hpp/blas.cpp) -- a different,
// similarly-named, easily-confused function. See also include/core/
// precond.hpp's create_sketch_qr_precond/apply_sketch_qr_precond, which look
// like the never-implemented top-level entry points for this same
// abandoned design. The templated apply_precond<device, value_t>(...,
// value_t* rhs) overload below is even less finished than "unused": its
// body only validates that precond/rhs match device and type, then returns
// without calling apply_trsv or touching rhs at all.

// In-place upper-triangular solve: rhs ← op(precond)^{-1} * rhs.
// Both precond and rhs must have the same numeric type; the caller is
// responsible for any type conversions before and after.
using apply_precond_fn = void (*)(TransposeType trans, tomos::Dense& precond,
                                  tomos::Dense& rhs);

struct Precond {
    // Singleton instance accessor.
    static const Precond& get();

    // Lookup; throws if no kernel is registered for the requested combination.
    apply_precond_fn lookup_apply(Device dev, NumericType numeric_type) const;

private:
    Precond();

    static constexpr std::size_t encode_apply(Device dev,
                                              NumericType numeric_type);

    static constexpr std::size_t N_apply =
        num_registry_devices * num_registry_numeric_types;

    std::array<apply_precond_fn, N_apply> apply_{};
};

void apply_precond(TransposeType trans, Dense& precond, Dense& rhs);

template <Device device, typename value_t>
void apply_precond(TransposeType trans, Dense& precond, value_t* rhs);

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_PRECOND_REGISTRIES_HPP