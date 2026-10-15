#ifndef RLS_COUNTSKETCH_CSR_HPP
#define RLS_COUNTSKETCH_CSR_HPP

#include <memory>

#include "context.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {

// NOT USED / NOT FULLY IMPLEMENTED. An earlier, class-based design for count
// sketching, superseded by the free functions below (generate_count_sketch /
// apply_count_sketch), which are what every caller in the codebase actually
// uses. create() and the constructor are implemented but never instantiated
// anywhere; generate_new()/generate_new_values()/generate_new_idxs() and
// apply() are declared here but have no definition at all (apply()'s old body
// survives only as a commented-out block in countsketch.cpp), so calling any
// of those four would fail to link.
struct Countsketch {
public:
    static std::unique_ptr<Countsketch> create(std::shared_ptr<Context> context,
                                               Device device, dim<2> size,
                                               size_t nnz_per_row, IntType t,
                                               NumericType p, bool prehash);

    Status generate_new();
    Status generate_new_values();
    Status generate_new_idxs();

    Status apply(Matrix& mtx, Dense& sketched_mtx);

    NumericType get_precision() { return compute_precision_; }

    dim<2> get_size() { return size_; }

    ~Countsketch() = default;

private:
    Countsketch(std::shared_ptr<Context> context, Device device, dim<2> size,
                size_t nnz_per_row, IntType t, NumericType p, bool prehash);

    NumericType compute_precision_;
    IntType     index_precision_;
    Device      device_;

    int*    ihash_           = nullptr;
    int8_t* shash_           = nullptr;
    bool    prehash_enabled_ = false;
    dim<2>  size_            = {0, 0};
    size_t  num_elems_       = 0;
    bool    prehash_         = false;
};  // Countsketch

void generate_count_sketch(Sparse& sketch);

void apply_count_sketch(Sparse& sketch, Dense& source, Dense& target);

void apply_count_sketch(Sparse& sketch, Sparse& source, Dense& target);

}  // namespace tomos

#endif
