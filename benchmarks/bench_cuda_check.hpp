#ifndef TOMOS_BENCH_CUDA_CHECK_HPP
#define TOMOS_BENCH_CUDA_CHECK_HPP

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <initializer_list>
#include <string>

// On a shared GPU, a large benchmark point's cudaMalloc can fail under
// memory pressure from other users' processes. None of these benchmarks
// used to check for that: a failed allocation left a null/garbage device
// pointer, and the subsequent kernel/cuBLAS call would either silently
// no-op (producing bogus near-zero timings that don't scale with problem
// size) or corrupt CUDA/cuBLAS state badly enough to crash a later,
// unrelated benchmark. The helpers here make that failure loud and
// contained instead: skip just the one benchmark point with a clear
// message via benchmark::State::SkipWithError, rather than silently
// running (or crashing) on bad pointers.

namespace tomos {
namespace bench {

// RAII device buffer: cudaMalloc's n elements of T and zero-fills them,
// tracking whether the allocation actually succeeded. Always cudaFrees on
// scope exit (including the failure path, where the pointer is null and
// cudaFree is a no-op), so a benchmark point that can't get enough VRAM
// doesn't leak into the next point's allocation attempt.
template <typename T>
class DevBuf {
public:
    explicit DevBuf(size_t n) {
        err_ = cudaMalloc(&ptr_, n * sizeof(T));
        if (err_ == cudaSuccess) cudaMemset(ptr_, 0, n * sizeof(T));
    }
    ~DevBuf() {
        if (ptr_) cudaFree(ptr_);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    cudaError_t status() const { return err_; }
    operator T*() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    cudaError_t err_ = cudaSuccess;
};

// Skips the benchmark (with a message identifying which allocation failed
// and why) if any of the given cudaError_t results indicate failure.
// Returns true iff all succeeded; call sites should `return` immediately
// when this returns false — SkipWithError does not itself unwind the
// caller.
inline bool check_alloc(benchmark::State& state,
                        std::initializer_list<cudaError_t> results) {
    for (cudaError_t e : results) {
        if (e != cudaSuccess) {
            // A cudaMalloc failure sets CUDA's internal "last error" state
            // even though we've already captured it via the return value
            // above; if left uncleared it would stick around and get
            // misattributed to the next benchmark's check_kernel_launch
            // (which calls cudaGetLastError() for the first time since
            // this failure). Clear it now that we're done with it.
            cudaGetLastError();
            state.SkipWithError(
                (std::string("cudaMalloc failed: ") + cudaGetErrorString(e))
                    .c_str());
            return false;
        }
    }
    return true;
}

// Checks for a kernel-launch error and waits for completion, checking that
// too. Meant to follow a raw kernel<<<...>>>() launch (cuda::mv_strided /
// cuda::mm_strided / cuda::mm_csr already report their own launch/sync
// errors to stderr internally, but a raw kernel launch has no such
// reporting at all otherwise). Returns true iff both checks passed.
inline bool check_kernel_launch(benchmark::State& state) {
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        state.SkipWithError(
            (std::string("kernel launch failed: ") +
             cudaGetErrorString(launch))
                .c_str());
        return false;
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        state.SkipWithError(
            (std::string("cudaDeviceSynchronize failed: ") +
             cudaGetErrorString(sync))
                .c_str());
        return false;
    }
    return true;
}

}  // namespace bench
}  // namespace tomos

#endif  // TOMOS_BENCH_CUDA_CHECK_HPP
