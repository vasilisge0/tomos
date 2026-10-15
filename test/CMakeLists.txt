include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(core/context)
add_subdirectory(core/matrix)
add_subdirectory(core/memory)
add_subdirectory(core/solver)
add_subdirectory(core/sketch)
add_subdirectory(core/precond)
add_subdirectory(core/workspace)
add_subdirectory(core/blas)
add_subdirectory(dispatch)

add_subdirectory(kernels/cpu)
add_subdirectory(kernels/cuda)