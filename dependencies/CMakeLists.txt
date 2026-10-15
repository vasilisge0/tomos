set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)
set(USE_THREAD ON CACHE BOOL "" FORCE)
set(NOFORTRAN OFF CACHE BOOL "" FORCE)
set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
set(BUILD_LAPACKE ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# 2. Help CMake find gfortran if it's struggling
# Replace "gfortran" with the full path if 'which gfortran' gives you one
set(CMAKE_Fortran_COMPILER "gfortran" CACHE STRING "" FORCE)

include(FetchContent)
FetchContent_Declare(
  openblas
  GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
  GIT_TAG v0.3.29
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(openblas)

# TODO: remove -- superseded/redundant. The root CMakeLists.txt's own
# if("${BLAS}" STREQUAL "USE_OPENBLAS") block already links tomos_core/
# tomos_cpu to OpenBLAS itself, via raw -L/-l flags against this same
# fetched build (plus its own add_dependencies(..., openblas_shared) for
# ordering) rather than through the proper `openblas` target below. These
# two lines are harmless (duplicate -lopenblas / duplicate include dirs,
# both resolving to the exact same fetched libopenblas.so) but dead weight
# that misleadingly suggests THIS file is what wires OpenBLAS into
# tomos_core -- it isn't, in practice. Left in place for now (not removed),
# since touching BLAS/OpenBLAS linking wasn't asked for; the FetchContent
# call above this comment is the genuinely load-bearing part of this file
# and should stay regardless.
#
# 3. Link correctly
# This target name must match the one defined by OpenBLAS (usually 'openblas')
target_link_libraries(tomos_core PRIVATE openblas)

# 4. Add the generated include path
# OpenBLAS puts generated headers in its own binary directory
target_include_directories(tomos_core PUBLIC
  ${openblas_BINARY_DIR}
  ${openblas_BINARY_DIR}/lapack-netlib/LAPACKE/include
)
