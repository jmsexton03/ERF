# Add uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
        "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling files listed in install_manifest.txt"
    )
endif()

# Add distclean target
add_custom_target(distclean
    # Header
    COMMAND ${CMAKE_COMMAND} -E echo "=================================================================================="
    COMMAND ${CMAKE_COMMAND} -E echo "Distclean: ${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "=================================================================================="

    # CMake configuration files
    COMMAND ${CMAKE_COMMAND} -E remove -f
            ${CMAKE_BINARY_DIR}/CMakeCache.txt
            ${CMAKE_BINARY_DIR}/cmake_install.cmake
            ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
            ${CMAKE_BINARY_DIR}/Makefile
            ${CMAKE_BINARY_DIR}/install_manifest.txt
            ${CMAKE_BINARY_DIR}/cray_detected_config.cmake

    # CPack files
    COMMAND ${CMAKE_COMMAND} -E remove -f
            ${CMAKE_BINARY_DIR}/CPackConfig.cmake
            ${CMAKE_BINARY_DIR}/CPackSourceConfig.cmake

    # CTest files
    COMMAND ${CMAKE_COMMAND} -E remove -f
            ${CMAKE_BINARY_DIR}/CTestTestfile.cmake
            ${CMAKE_BINARY_DIR}/DartConfiguration.tcl

    # Project-specific generated files
    COMMAND ${CMAKE_COMMAND} -E remove -f
            ${CMAKE_BINARY_DIR}/ERFConfig.cmake
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_BINARY_DIR}/git-state.txt

    # CMake-generated directories
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/Testing
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/_deps

    # Build output directories
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/Exec
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/Submodules
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/Tests
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/erf_srclib
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/cmake_packages
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/externals

    # Removing generated files
    COMMAND ${CMAKE_COMMAND} -E echo "Removing generated files from ${CMAKE_BINARY_DIR}..."
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.pc ${CMAKE_BINARY_DIR}/lib*.a ${CMAKE_BINARY_DIR}/lib*.so ${CMAKE_BINARY_DIR}/build_*.log

    # Summary
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo " DONE: Distclean complete"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Next steps to reconfigure:"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  If you used a build script:"
    COMMAND ${CMAKE_COMMAND} -E echo "    cd ${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "    ERF_HOME=${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/Build/cmake.sh"
    COMMAND ${CMAKE_COMMAND} -E echo "    or whichever script you used"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  For manual cmake configuration:"
    COMMAND ${CMAKE_COMMAND} -E echo "    From build directory: cmake ${CMAKE_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Note: Install directories preserved"

    COMMENT "Removing all CMake configuration and build artifacts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Generate toolchain file from current configuration
add_custom_target(generate-toolchain
    COMMAND ${CMAKE_COMMAND} -E echo "Generating toolchain file..."
    COMMAND ${CMAKE_COMMAND}
        -DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        -DCMAKE_SYSTEM_NAME="${CMAKE_SYSTEM_NAME}"
        -DCMAKE_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
        -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
        -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
        -DCMAKE_Fortran_COMPILER="${CMAKE_Fortran_COMPILER}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
        -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
        -DCMAKE_Fortran_FLAGS="${CMAKE_Fortran_FLAGS}"
        -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
        -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}"
        -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS}"
        -DERF_ENABLE_MPI="${ERF_ENABLE_MPI}"
        -DERF_ENABLE_OPENMP="${ERF_ENABLE_OPENMP}"
        -DERF_ENABLE_CUDA="${ERF_ENABLE_CUDA}"
        -DERF_ENABLE_HIP="${ERF_ENABLE_HIP}"
        -DERF_ENABLE_SYCL="${ERF_ENABLE_SYCL}"
        -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}"
        -DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS}"
        -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}"
        -DCMAKE_HIP_FLAGS="${CMAKE_HIP_FLAGS}"
        -DAMREX_CUDA_ARCH="${AMReX_CUDA_ARCH}"
        -DAMREX_AMD_ARCH="${AMReX_AMD_ARCH}"
        -DMPI_C_COMPILER="${MPI_C_COMPILER}"
        -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
        -DMPI_Fortran_COMPILER="${MPI_Fortran_COMPILER}"
        -DTEMPLATE_FILE="${CMAKE_SOURCE_DIR}/CMake/ToolchainTemplate.cmake.in"
        -DOUTPUT_FILE="${CMAKE_BINARY_DIR}/erf_toolchain.cmake"
        -P "${CMAKE_SOURCE_DIR}/CMake/GenerateToolchain.cmake"
    COMMENT "Generating erf_toolchain.cmake from current configuration"
    VERBATIM
)

