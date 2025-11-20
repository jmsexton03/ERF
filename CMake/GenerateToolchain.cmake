# Script to generate toolchain file from template

# Get current date
string(TIMESTAMP GENERATION_DATE "%Y-%m-%d %H:%M:%S")

# Prepare conditional sections
if(ERF_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CUDA_COMPILER_SECTION "
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER \"${CMAKE_CUDA_COMPILER}\" CACHE FILEPATH \"CUDA compiler\")
endif()")
    set(CUDA_FLAGS_SECTION "
# CUDA Flags
string(APPEND CMAKE_CUDA_FLAGS_INIT \"${CMAKE_CUDA_FLAGS}\")")
    set(COMPILER_STATUS_MESSAGES "${COMPILER_STATUS_MESSAGES}
message(STATUS \"    CUDA: ${CMAKE_CUDA_COMPILER}\")")
else()
    set(CUDA_COMPILER_SECTION "# CUDA not enabled in source build")
    set(CUDA_FLAGS_SECTION "")
endif()

if(ERF_ENABLE_HIP AND CMAKE_HIP_COMPILER)
    set(HIP_COMPILER_SECTION "
if(NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER \"${CMAKE_HIP_COMPILER}\" CACHE FILEPATH \"HIP compiler\")
endif()")
    set(HIP_FLAGS_SECTION "
# HIP Flags
string(APPEND CMAKE_HIP_FLAGS_INIT \"${CMAKE_HIP_FLAGS}\")")
    set(COMPILER_STATUS_MESSAGES "${COMPILER_STATUS_MESSAGES}
message(STATUS \"    HIP: ${CMAKE_HIP_COMPILER}\")")
else()
    set(HIP_COMPILER_SECTION "# HIP not enabled in source build")
    set(HIP_FLAGS_SECTION "")
endif()

if(ERF_ENABLE_MPI)
    set(MPI_SECTION "
# MPI Compilers
set(MPI_C_COMPILER \"${MPI_C_COMPILER}\" CACHE FILEPATH \"MPI C wrapper\")
set(MPI_CXX_COMPILER \"${MPI_CXX_COMPILER}\" CACHE FILEPATH \"MPI C++ wrapper\")
set(MPI_Fortran_COMPILER \"${MPI_Fortran_COMPILER}\" CACHE FILEPATH \"MPI Fortran wrapper\")")
else()
    set(MPI_SECTION "# MPI not enabled in source build")
endif()

if(AMReX_CUDA_ARCH)
    set(AMREX_ARCH_SECTION "set(AMReX_CUDA_ARCH \"${AMReX_CUDA_ARCH}\" CACHE STRING \"CUDA architecture\")")
elseif(AMReX_AMD_ARCH)
    set(AMREX_ARCH_SECTION "set(AMReX_AMD_ARCH \"${AMReX_AMD_ARCH}\" CACHE STRING \"AMD architecture\")")
else()
    set(AMREX_ARCH_SECTION "# No GPU architecture specified")
endif()

# Configure the template
configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE} @ONLY)

message(STATUS "Toolchain file generated: ${OUTPUT_FILE}")
message(STATUS "To use this toolchain file:")
message(STATUS "  Option 1: Pass directly to CMake:")
message(STATUS "    cmake -DCMAKE_TOOLCHAIN_FILE=${OUTPUT_FILE} ${CMAKE_SOURCE_DIR}")
message(STATUS "  Option 2: Set environment variable:")
message(STATUS "    export CMAKE_TOOLCHAIN_FILE=${OUTPUT_FILE}")
message(STATUS "    cmake ${CMAKE_SOURCE_DIR}")