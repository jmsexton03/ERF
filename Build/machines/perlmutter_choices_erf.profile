#!/bin/bash
# ==============================================================================
# System Profile - Can be used in module-only or interactive mode
# ==============================================================================

# SYSTEM CONFIGURATION - SET THIS FOR EACH MACHINE
SYSTEM_NAME="perlmutter"

# Set up paths
if [ -n "$ERF_HOME" ]; then
  : ${ERF_BUILD_DIR:="$ERF_HOME/build"}
  : ${ERF_SOURCE_DIR:="$ERF_HOME"}
  : ${ERF_INSTALL_DIR:="$ERF_HOME/install"}
else
  : ${ERF_BUILD_DIR:="build"}
  : ${ERF_SOURCE_DIR:="$(pwd)"}
  : ${ERF_INSTALL_DIR:="install"}
fi

# ============================================================================
# SYSTEM-SPECIFIC MODULES (always loaded)
# ============================================================================
module load gcc-native/13.2 cmake cudatoolkit cray-hdf5-parallel cray-netcdf-hdf5parallel cray-libsci

#module load gcc-native/13.2
#module load cray-mpich/8.1.30
#module load cray-hdf5-parallel/1.14.3.1
#module load cray-netcdf-hdf5parallel/4.9.0.13
#module load cmake/3.30.2
#module load cray-libsci/24.07.0
#module load cray-parallel-netcdf/1.12.3.13

# Automatically included with module load gpu
# export MPICH_GPU_SUPPORT_ENABLED=1

echo "Modules loaded for ${SYSTEM_NAME}"
echo "Paths: Source=$ERF_SOURCE_DIR | Build=$ERF_BUILD_DIR | Install=$ERF_INSTALL_DIR"

# Skip menu if explicitly requested
if [ -n "$ERF_SKIP_CONFIG_MENU" ]; then
    echo "Skipping config menu (ERF_SKIP_CONFIG_MENU set)"
    return 0 2>/dev/null || exit 0
fi

# ============================================================================
# Build Configuration Menu
# ============================================================================
echo ""
echo "ERF Build Configuration (ranked by automation level):"
echo ""
echo "  Automatic/Heuristic approaches:"
echo "  1) Cray auto-detection (CMake heuristics, module-independent)"
echo "  2) Cray toolchain file (auto-detection via toolchain)"
echo "  3) Bash setup scripts (explicit env vars, readable, module-dependent)"
echo ""
echo "  Manual/Explicit approaches:"
echo "  4) GPU-aware MPI manual (old pattern, explicit flags)"
echo "  5) Use current environment (assume configuration users responsibility)"
echo ""
echo "  Reproducible/Tracked configurations:"
echo "  6) Generated toolchain from previous build (cmake practice for compiler setting reproducibly)"
echo "  7) CMake config file -C (uses configuration files)"
echo ""
echo "  0) Skip menu (modules only, let build system decide)"
echo ""
read -p "Choice [0-7, default=0]: " choice
choice=${choice:-0}

case $choice in
    0)
        echo "Modules loaded, build system will auto-detect configuration"
        ;;
        
    # === AUTOMATIC/HEURISTIC ===
    1)
        echo "Cray auto-detection enabled (CMake will detect wrappers/libs)"
        export ERF_ENABLE_CRAY_AUTO_FIXES=ON
        echo "  Uses: CrayDetection.cmake heuristics"
        echo "  Usage: cmake .."
        ;;
        
    2)
        echo "Cray toolchain (auto-detection via toolchain file)"
        export CMAKE_TOOLCHAIN_FILE="$ERF_SOURCE_DIR/CMake/CrayToolchain.cmake"
        echo "  Uses: CrayToolchain.cmake (enables auto-detection)"
        echo "  Usage: cmake .."
        ;;
        
    3)
        echo "Bash setup scripts (explicit environment setup)"
        source $ERF_SOURCE_DIR/CMake/scripts/cray_setup_compilers.sh
        source $ERF_SOURCE_DIR/CMake/scripts/cray_setup_flags.sh
        export ERF_ENABLE_CRAY_AUTO_FIXES=OFF
        echo "  Sets: CC, CXX, CFLAGS, LDFLAGS, LIBS in shell"
        echo "  Usage: cmake .."
        echo "  Usage:  make .."
        ;;
        
    # === MANUAL/EXPLICIT ===
    4)
        echo "GPU-aware MPI with explicit flags (tested pattern)"
        export MPICH_GPU_SUPPORT_ENABLED=1
        export CRAY_ACCEL_TARGET=nvidia80  # CUSTOMIZE PER SYSTEM
        export AMREX_CUDA_ARCH=8.0
        export CXXFLAGS="${CXXFLAGS} -march=znver3"
        export CFLAGS="${CFLAGS} -march=znver3"
        export CUDAHOSTCXX=CC
        export ERF_ENABLE_CRAY_AUTO_FIXES=OFF
        echo "  Explicit: GPU target, MPI libs, optimization flags"
        echo "  Usage: cmake .."
        ;;
        
    5)
        echo "Using current environment (manual configuration)"
        export ERF_ENABLE_CRAY_AUTO_FIXES=OFF
        echo "  Assumes: CC/CXX/FC already set correctly"
        echo "  Usage: cmake .."
        ;;
        
    # === REPRODUCIBLE/TRACKED ===
    6)
        echo "Generated toolchain (from previous successful build)"
        export CMAKE_TOOLCHAIN_FILE="$ERF_BUILD_DIR/erf_toolchain.cmake"
        echo "  Captures: compilers, flags, paths from working build"
        echo "  Generate: cmake --build $ERF_BUILD_DIR --target generate-toolchain"
        echo "  Usage: cmake .."
        ;;
        
    7)
        echo "CMake config file (version-controlled cache)"
        export ERF_CMAKE_CONFIG="$ERF_SOURCE_DIR/Build/machines/${SYSTEM_NAME}_config.cmake"
        echo "  Pre-written: system-specific cache variables"
        echo "  Usage: cmake -C \$ERF_CMAKE_CONFIG .."
        ;;
        
    *)
        echo "Invalid choice"
        ;;
esac

echo ""#!/bin/bash

