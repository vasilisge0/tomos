add_subdirectory(cpu)

if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    add_subdirectory(cuda)
endif()
