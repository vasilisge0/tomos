# Installing TOMOS

This document covers building TOMOS from source, targeting a different GPU
architecture, and the optional local-configuration mechanism for scripts
that need a CUTLASS checkout or LSQR test matrices. See the root
[`README.md`](../README.md) for running tests and reproducing the paper's
experiments once the build is in place.

---

## 1. Prerequisites

- **CMake ≥ 3.13** (the project pins `cmake_minimum_required(VERSION 3.13.11)`).
- **A CUDA toolkit**, with `CUDA_HOME` set to its install root (used for
  `$CUDA_HOME/include`/`$CUDA_HOME/lib64`).
- **`gfortran`** — needed only to build OpenBLAS/LAPACKE from source; CMake
  fetches and builds both automatically via `dependencies/CMakeLists.txt`'s
  `FetchContent`, so no separate system BLAS/LAPACK install is required.

## 2. Configure and build

```sh
scripts/configure/configure_sm_80.sh   # configures build/, targets sm_80 (A100)
scripts/configure/build.sh             # cd build; make -j4
```

`configure_sm_80.sh` runs CMake with the project's standard options:
`DEVICE_BACKEND=CUDA`, `BLAS=USE_OPENBLAS` (fetched and built from source),
`CMAKE_CUDA_ARCHITECTURES=80`, shared libraries, Release build type, and
LAPACKE enabled. This produces three shared-library targets —
`tomos_core` (device-independent core), `tomos_cpu` (CPU kernels), and
`tomos_cuda` (CUDA kernels) — plus the `test/` and `benchmarks/` targets.

`configure_sm_80.sh` is the correct, currently-maintained configure script —
it's what every build in this project's own history has actually used.
There is an older `scripts/configure/configure.sh` in the same directory;
**don't use it** — it passes `DEVICE_BACKEND="USE_CUDA"` instead of
`"CUDA"`, which the root `CMakeLists.txt` doesn't match against, so it
silently builds a CPU-only tree despite asking for CUDA.

## 3. Targeting a different GPU architecture

If you're not building for an A100 (`sm_80`), copy `configure_sm_80.sh` and
change `CMAKE_CUDA_ARCHITECTURES` to your target's compute capability (e.g.
`90` for an H100, `86` for an A6000/RTX 30-series).

## 4. Building a single target

Once configured, you don't need to rebuild everything on every change —
build just the target you need:

```sh
cmake --build build --target tomos_cuda -j4
cmake --build build --target test_blas -j4
```

See [`README.md`](../README.md)'s own `## 2. Tests` section for running the
resulting test binaries (there's no `ctest`/`make test` integration —
`cmake --build` only compiles, it never runs them).

## 5. Optional: local script configuration (`.configure.sh`)

A number of `scripts/*.sh` benchmark drivers need two things that aren't
part of the CMake build at all, since they're pure runtime/data-location
concerns, not build-time ones:

- **`CUTLASS_DIR`** — where to find (or shallow-clone, on first use) a
  CUTLASS checkout, used only by the standalone `*vs_cutlass_ref*.sh`
  comparison benchmarks. CUTLASS is never linked into `tomos_core`/
  `tomos_cpu`/`tomos_cuda` or compiled via CMake at all — see
  [`documentation/third-party-notices.md`](third-party-notices.md).
- **`MATRIX_DIR`** — where to find (or write, via
  `scripts/generate_lsqr_test_matrices.sh`) the tall-and-thin `.mtx` test
  problems LSQR's benchmark scripts read.

Both default to a `build/`-relative path (`build/cutlass-ref`,
`build/matrices`) if left unset, so this step is entirely optional. To
point either at a different, persistent location instead of passing
`CUTLASS_DIR=...`/`--matrix-dir` on every invocation, copy the tracked
template to a local, gitignored override file and edit it:

```sh
cp .configure.sh.example .configure.sh
# then edit .configure.sh and uncomment/set MATRIX_DIR and/or CUTLASS_DIR
```

`.configure.sh` is sourced automatically by every script that uses either
variable, right after that script resolves its own repo root, and is
never committed (it's in `.gitignore`) — so each person's own
machine-specific paths stay local to their own checkout.
