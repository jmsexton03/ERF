#!/bin/bash
# ==============================================================================
# Cray System Environment Setup for CMake - Part 1: Compiler Configuration
# ==============================================================================
# This script sets up compiler environment variables for Cray systems
# Based on CrayCompilerDetection.cmake
#
# Sets: CC, CXX, FC, CUDACXX, CUDAHOSTCXX, HIPCXX, HIPHOSTCXX
#
# Usage:
#   source cray_setup_compilers.sh [options]
#
# Options:
#   --verbose         : Show verbose output
#   --debug           : Show debug output
#   --force           : Force setup even if compilers already set
#   --help            : Show this help message
# ==============================================================================

# Initialize variables
VERBOSE=0
DEBUG=0
FORCE=0
SCRIPT_NAME="cray_setup_compilers.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=1
            shift
            ;;
        --debug)
            DEBUG=1
            VERBOSE=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --help)
            echo "Usage: source $SCRIPT_NAME [options]"
            echo ""
            echo "Options:"
            echo "  --verbose         : Show verbose output"
            echo "  --debug           : Show debug output"
            echo "  --force           : Force setup even if compilers already set"
            echo "  --help            : Show this help message"
            echo ""
            echo "Sets environment variables:"
            echo "  CC                : C compiler"
            echo "  CXX               : C++ compiler"
            echo "  FC                : Fortran compiler"
            echo "  CUDACXX           : CUDA compiler"
            echo "  CUDAHOSTCXX       : CUDA host compiler"
            echo "  HIPCXX            : HIP compiler"
            echo "  HIPHOSTCXX        : HIP host compiler"
            echo ""
            echo "Also sets CMake cache variables via environment:"
            echo "  CMAKE_C_COMPILER"
            echo "  CMAKE_CXX_COMPILER"
            echo "  CMAKE_Fortran_COMPILER"
            echo "  CMAKE_CUDA_HOST_COMPILER"
            echo "  CMAKE_HIP_HOST_COMPILER"
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

# Logging functions
log_status() {
    echo "[STATUS] $*"
}

log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[VERBOSE] $*"
    fi
}

log_debug() {
    if [[ $DEBUG -eq 1 ]]; then
        echo "[DEBUG] $*"
    fi
}

log_warning() {
    echo "[WARNING] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

# ==============================================================================
# Helper function: Suggest machine profile
# ==============================================================================

suggest_machine_profile() {
    local hostname=$(hostname -s 2>/dev/null)
    local build_dir="${ERF_SOURCE_DIR:-$(pwd)/..}/Build/machines"
    
    if [[ ! -d "$build_dir" ]]; then
        build_dir="$(pwd)/Build/machines"
    fi
    
    log_status "Load modules from your machine profile:"
    echo ""
    
    if [[ -d "$build_dir" ]]; then
        for profile in "$build_dir"/*_erf.profile; do
            if [[ -f "$profile" ]]; then
                local name=$(basename "$profile" _erf.profile)
                if [[ "$hostname" =~ $name ]]; then
                    echo "    source $profile  <-- matches hostname '$hostname'"
                else
                    echo "    source $profile"
                fi
            fi
        done
    else
        log_warning "No profiles found in $build_dir"
    fi
    echo ""
}

# ==============================================================================
# Detect Cray Environment
# ==============================================================================

detect_cray_system() {
    local on_cray=0
    
    log_debug "Checking for Cray environment"
    
    # Check for Cray Programming Environment
    if [[ -n "${CRAYPE_VERSION}" ]]; then
        on_cray=1
        log_verbose "Detected Cray Programming Environment: ${CRAYPE_VERSION}"
    fi
    
    if [[ -n "${CRAY_MPICH_DIR}" ]]; then
        on_cray=1
        log_verbose "Detected Cray MPI: ${CRAY_MPICH_DIR}"
    fi
    
    if [[ -n "${PE_ENV}" ]]; then
        on_cray=1
        log_verbose "Detected Cray PE: ${PE_ENV}"
    fi
    
    return $((1-on_cray))
}

# ==============================================================================
# Check if compilers already set
# ==============================================================================

check_compilers_set() {
    local already_set=0
    
    # Check standard environment variables
    if [[ -n "${CC}" ]] || [[ -n "${CXX}" ]] || [[ -n "${FC}" ]]; then
        already_set=1
        log_verbose "Standard compiler variables already set"
        [[ -n "${CC}" ]] && log_debug "  CC=${CC}"
        [[ -n "${CXX}" ]] && log_debug "  CXX=${CXX}"
        [[ -n "${FC}" ]] && log_debug "  FC=${FC}"
    fi
    
    # Check CMake-specific variables
    if [[ -n "${CMAKE_C_COMPILER}" ]] || [[ -n "${CMAKE_CXX_COMPILER}" ]] || [[ -n "${CMAKE_Fortran_COMPILER}" ]]; then
        already_set=1
        log_verbose "CMake compiler variables already set"
        [[ -n "${CMAKE_C_COMPILER}" ]] && log_debug "  CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        [[ -n "${CMAKE_CXX_COMPILER}" ]] && log_debug "  CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        [[ -n "${CMAKE_Fortran_COMPILER}" ]] && log_debug "  CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
    fi
    
    return $already_set
}

# ==============================================================================
# Find Cray compiler wrappers
# ==============================================================================

find_cray_wrappers() {
    local cc_wrapper=$(command -v cc 2>/dev/null)
    local cxx_wrapper=$(command -v CC 2>/dev/null)
    local fc_wrapper=$(command -v ftn 2>/dev/null)
    
    if [[ -n "$cc_wrapper" ]]; then
        export CRAY_CC="$cc_wrapper"
        log_verbose "Found Cray C wrapper: $cc_wrapper"
    else
        log_warning "Cray C wrapper 'cc' not found in PATH"
    fi
    
    if [[ -n "$cxx_wrapper" ]]; then
        export CRAY_CXX="$cxx_wrapper"
        log_verbose "Found Cray C++ wrapper: $cxx_wrapper"
    else
        log_warning "Cray C++ wrapper 'CC' not found in PATH"
    fi
    
    if [[ -n "$fc_wrapper" ]]; then
        export CRAY_FC="$fc_wrapper"
        log_verbose "Found Cray Fortran wrapper: $fc_wrapper"
    else
        log_debug "Cray Fortran wrapper 'ftn' not found (may not be needed)"
    fi
}

# ==============================================================================
# Check GPU environment
# ==============================================================================

check_gpu_environment() {
    local gpu_type=""
    local need_accel_module=0
    
    # Check for CUDA
    if [[ -n "${CUDA_HOME}" ]] || [[ -n "${CUDATOOLKIT_HOME}" ]]; then
        gpu_type="CUDA"
        log_verbose "Detected CUDA environment"
        [[ -n "${CUDA_HOME}" ]] && log_debug "  CUDA_HOME=${CUDA_HOME}"
        [[ -n "${CUDATOOLKIT_HOME}" ]] && log_debug "  CUDATOOLKIT_HOME=${CUDATOOLKIT_HOME}"
        
        if [[ -n "${CRAYPE_VERSION}" ]] && [[ -z "${CRAY_ACCEL_TARGET}" ]]; then
            need_accel_module=1
            log_error "CUDA on Cray requires craype-accel-nvidia* module"
        fi
    fi
    
    # Check for ROCm/HIP
    if [[ -n "${ROCM_PATH}" ]] || [[ -n "${HIP_PATH}" ]]; then
        gpu_type="HIP"
        log_verbose "Detected ROCm/HIP environment"
        [[ -n "${ROCM_PATH}" ]] && log_debug "  ROCM_PATH=${ROCM_PATH}"
        [[ -n "${HIP_PATH}" ]] && log_debug "  HIP_PATH=${HIP_PATH}"
        
        if [[ -n "${CRAYPE_VERSION}" ]] && [[ -z "${CRAY_ACCEL_TARGET}" ]]; then
            need_accel_module=1
            log_error "HIP on Cray requires craype-accel-amd* module"
        fi
    fi
    
    # Check for Intel oneAPI/SYCL
    if [[ -n "${ONEAPI_ROOT}" ]] || [[ -n "${I_MPI_ROOT}" ]]; then
        gpu_type="SYCL"
        log_verbose "Detected Intel oneAPI/SYCL environment"
        [[ -n "${ONEAPI_ROOT}" ]] && log_debug "  ONEAPI_ROOT=${ONEAPI_ROOT}"
    fi
    
    if [[ $need_accel_module -eq 1 ]]; then
        echo ""
        echo "===================================================================="
        echo "GPU on Cray: Missing craype-accel Module"
        echo "===================================================================="
        echo ""
        echo "The Cray compiler wrappers need a craype-accel-* module loaded"
        echo "to configure GPU support (sets CRAY_ACCEL_TARGET)."
        echo ""
        suggest_machine_profile
        echo "Examples of craype-accel modules:"
        if [[ "$gpu_type" == "CUDA" ]]; then
            echo "  craype-accel-nvidia80   (A100)"
            echo "  craype-accel-nvidia90   (H100)"
        elif [[ "$gpu_type" == "HIP" ]]; then
            echo "  craype-accel-amd-gfx90a (MI250X)"
            echo "  craype-accel-amd-gfx942 (MI300)"
        fi
        echo ""
        echo "===================================================================="
        return 1
    fi
    
    if [[ -n "${CRAY_ACCEL_TARGET}" ]]; then
        log_verbose "Cray accelerator target: ${CRAY_ACCEL_TARGET}"
    fi
    
    echo "$gpu_type"
    return 0
}

# ==============================================================================
# Detect MPI and GTL libraries
# ==============================================================================

detect_mpi_gtl_libs() {
    local mpi_lib=""
    local gtl_lib=""
    
    # Try to get from CC --cray-print-opts=libs
    if [[ -n "${CRAY_CXX}" ]]; then
        local libs_output=$("${CRAY_CXX}" --cray-print-opts=libs 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            # Extract MPI library
            if [[ "$libs_output" =~ -lmpi_gnu_([0-9]+) ]]; then
                mpi_lib="mpi_gnu_${BASH_REMATCH[1]}"
            elif [[ "$libs_output" =~ -lmpi_cray ]]; then
                mpi_lib="mpi_cray"
            elif [[ "$libs_output" =~ -lmpi_intel ]]; then
                mpi_lib="mpi_intel"
            fi
            
            # Extract GTL library
            if [[ "$libs_output" =~ -lmpi_gtl_([a-z]+) ]]; then
                gtl_lib="mpi_gtl_${BASH_REMATCH[1]}"
            fi
            
            log_debug "Parsed from CC --cray-print-opts=libs:"
            log_debug "  MPI lib: ${mpi_lib:-none}"
            log_debug "  GTL lib: ${gtl_lib:-none}"
        fi
    fi
    
    # Fallback: derive from environment
    if [[ -z "$mpi_lib" ]] && [[ -n "${CRAY_MPICH_DIR}" ]]; then
        if [[ "${CRAY_MPICH_DIR}" =~ /gnu/([0-9]+)\.([0-9]+) ]]; then
            mpi_lib="mpi_gnu_${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            log_debug "Derived MPI lib from CRAY_MPICH_DIR: $mpi_lib"
        fi
    fi
    
    # Fallback: derive GTL from CRAY_ACCEL_TARGET
    if [[ -z "$gtl_lib" ]] && [[ -n "${CRAY_ACCEL_TARGET}" ]]; then
        # Check for specific environment variable
        local gtl_var="PE_MPICH_GTL_LIBS_${CRAY_ACCEL_TARGET}"
        if [[ -n "${!gtl_var}" ]]; then
            gtl_lib="${!gtl_var#-l}"  # Remove -l prefix if present
            log_debug "Got GTL lib from $gtl_var: $gtl_lib"
        elif [[ "${CRAY_ACCEL_TARGET}" =~ nvidia ]]; then
            gtl_lib="mpi_gtl_cuda"
            log_debug "Derived GTL lib for NVIDIA: $gtl_lib"
        elif [[ "${CRAY_ACCEL_TARGET}" =~ amd ]]; then
            gtl_lib="mpi_gtl_hsa"
            log_debug "Derived GTL lib for AMD: $gtl_lib"
        fi
    fi
    
    # Export for use in LIBS/LDFLAGS
    if [[ -n "$mpi_lib" ]]; then
        export CRAY_MPI_LIB="$mpi_lib"
    fi
    if [[ -n "$gtl_lib" ]]; then
        export CRAY_GTL_LIB="$gtl_lib"
    fi
    
    log_verbose "MPI/GTL libraries detected:"
    [[ -n "$mpi_lib" ]] && log_verbose "  MPI: $mpi_lib"
    [[ -n "$gtl_lib" ]] && log_verbose "  GTL: $gtl_lib"
}

# ==============================================================================
# Setup standard compiler variables
# ==============================================================================

setup_standard_compilers() {
    log_status "Setting up standard compiler environment variables"
    
    # C compiler
    if [[ -z "${CC}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_CC}" ]]; then
            export CC="${CRAY_CC}"
            log_status "Set CC=${CC}"
        fi
    else
        log_verbose "CC already set: ${CC}"
    fi
    
    # C++ compiler
    if [[ -z "${CXX}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_CXX}" ]]; then
            export CXX="${CRAY_CXX}"
            log_status "Set CXX=${CXX}"
        fi
    else
        log_verbose "CXX already set: ${CXX}"
    fi
    
    # Fortran compiler (optional)
    if [[ -z "${FC}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_FC}" ]]; then
            export FC="${CRAY_FC}"
            log_status "Set FC=${FC}"
        fi
    else
        log_verbose "FC already set: ${FC}"
    fi
}

# ==============================================================================
# Setup CMake compiler variables
# ==============================================================================

setup_cmake_compilers() {
    log_status "Setting up CMake compiler environment variables"
    
    # CMake C compiler
    if [[ -z "${CMAKE_C_COMPILER}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_CC}" ]]; then
            export CMAKE_C_COMPILER="${CRAY_CC}"
            log_status "Set CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        fi
    else
        log_verbose "CMAKE_C_COMPILER already set: ${CMAKE_C_COMPILER}"
    fi
    
    # CMake C++ compiler
    if [[ -z "${CMAKE_CXX_COMPILER}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_CXX}" ]]; then
            export CMAKE_CXX_COMPILER="${CRAY_CXX}"
            log_status "Set CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        fi
    else
        log_verbose "CMAKE_CXX_COMPILER already set: ${CMAKE_CXX_COMPILER}"
    fi
    
    # CMake Fortran compiler
    if [[ -z "${CMAKE_Fortran_COMPILER}" ]] || [[ $FORCE -eq 1 ]]; then
        if [[ -n "${CRAY_FC}" ]]; then
            export CMAKE_Fortran_COMPILER="${CRAY_FC}"
            log_status "Set CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
        fi
    else
        log_verbose "CMAKE_Fortran_COMPILER already set: ${CMAKE_Fortran_COMPILER}"
    fi
}

# ==============================================================================
# Setup GPU compiler variables
# ==============================================================================

setup_gpu_compilers() {
    local gpu_type=$(check_gpu_environment)
    
    if [[ -z "$gpu_type" ]]; then
        log_debug "No GPU environment detected"
        return 0
    fi
    
    log_status "Setting up $gpu_type compiler environment"
    
    case "$gpu_type" in
        CUDA)
            # CUDA host compiler (for nvcc or nvcc_wrapper)
            if [[ -z "${CUDAHOSTCXX}" ]] || [[ $FORCE -eq 1 ]]; then
                if [[ -n "${CRAY_CXX}" ]]; then
                    export CUDAHOSTCXX="${CRAY_CXX}"
                    log_status "Set CUDAHOSTCXX=${CUDAHOSTCXX}"
                fi
            else
                log_verbose "CUDAHOSTCXX already set: ${CUDAHOSTCXX}"
            fi
            
            # CMAKE_CUDA_HOST_COMPILER
            if [[ -z "${CMAKE_CUDA_HOST_COMPILER}" ]] || [[ $FORCE -eq 1 ]]; then
                if [[ -n "${CRAY_CXX}" ]]; then
                    export CMAKE_CUDA_HOST_COMPILER="${CRAY_CXX}"
                    log_status "Set CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
                fi
            else
                log_verbose "CMAKE_CUDA_HOST_COMPILER already set: ${CMAKE_CUDA_HOST_COMPILER}"
            fi
            
            # Note: We don't set CUDACXX to the Cray wrapper by default
            # because nvcc or nvcc_wrapper is typically preferred
            if [[ -n "${CUDACXX}" ]]; then
                log_verbose "CUDACXX already set: ${CUDACXX}"
            else
                log_verbose "CUDACXX not set (will use nvcc or nvcc_wrapper if available)"
            fi
            ;;
            
        HIP)
            # HIP host compiler
            if [[ -z "${HIPHOSTCXX}" ]] || [[ $FORCE -eq 1 ]]; then
                if [[ -n "${CRAY_CXX}" ]]; then
                    export HIPHOSTCXX="${CRAY_CXX}"
                    log_status "Set HIPHOSTCXX=${HIPHOSTCXX}"
                fi
            else
                log_verbose "HIPHOSTCXX already set: ${HIPHOSTCXX}"
            fi
            
            # CMAKE_HIP_HOST_COMPILER
            if [[ -z "${CMAKE_HIP_HOST_COMPILER}" ]] || [[ $FORCE -eq 1 ]]; then
                if [[ -n "${CRAY_CXX}" ]]; then
                    export CMAKE_HIP_HOST_COMPILER="${CRAY_CXX}"
                    log_status "Set CMAKE_HIP_HOST_COMPILER=${CMAKE_HIP_HOST_COMPILER}"
                fi
            else
                log_verbose "CMAKE_HIP_HOST_COMPILER already set: ${CMAKE_HIP_HOST_COMPILER}"
            fi
            
            # Note: Similar to CUDA, we typically don't override HIPCXX
            if [[ -n "${HIPCXX}" ]]; then
                log_verbose "HIPCXX already set: ${HIPCXX}"
            else
                log_verbose "HIPCXX not set (will use hipcc if available)"
            fi
            ;;
            
        SYCL)
            log_verbose "SYCL will use CXX=${CXX}"
            ;;
    esac
}

# ==============================================================================
# Setup GPU-aware MPI flags
# ==============================================================================

setup_gpu_mpi() {
    if [[ "${MPICH_GPU_SUPPORT_ENABLED}" != "1" ]]; then
        log_debug "GPU-aware MPI not enabled"
        return 0
    fi
    
    log_status "Configuring GPU-aware MPI"
    
    # Detect libraries
    detect_mpi_gtl_libs
    
    # Add to LIBS if needed
    local gpu_libs=""
    [[ -n "${CRAY_MPI_LIB}" ]] && gpu_libs="${gpu_libs} -l${CRAY_MPI_LIB}"
    [[ -n "${CRAY_GTL_LIB}" ]] && gpu_libs="${gpu_libs} -l${CRAY_GTL_LIB}"
    
    if [[ -n "$gpu_libs" ]]; then
        # Add CUDA runtime if needed
        if [[ "${CRAY_ACCEL_TARGET}" =~ nvidia ]]; then
            if [[ -n "${CUDA_HOME}" ]] || [[ -n "${CUDATOOLKIT_HOME}" ]]; then
                gpu_libs="${gpu_libs} -lcudart -lcuda"
                log_verbose "Added CUDA runtime libraries for GPU-aware MPI"
            else
                log_warning "GPU-aware MPI with NVIDIA GPU but CUDA toolkit not found"
                echo ""
                echo "===================================================================="
                echo "GPU-Aware MPI: CUDA Runtime Not Found"
                echo "===================================================================="
                echo ""
                echo "GPU-aware MPI is enabled but CUDA toolkit is not loaded."
                echo ""
                suggest_machine_profile
                echo "===================================================================="
                return 1
            fi
        fi
        
        export LIBS="${LIBS:+$LIBS }$gpu_libs"
        log_verbose "Added GPU-aware MPI libraries to LIBS"
        log_debug "GPU libs: $gpu_libs"
    fi
}

# ==============================================================================
# Main Setup
# ==============================================================================

main() {
    log_status "Cray System CMake Environment Setup - Compilers"
    
    # Check if compilers already set (unless forced)
    if [[ $FORCE -eq 0 ]]; then
        if check_compilers_set; then
            log_status "Compilers already set by user (use --force to override)"
            return 0
        fi
    fi
    
    # Detect Cray system
    if ! detect_cray_system; then
        log_status "Not on a Cray system, skipping Cray-specific compiler setup"
        return 0
    fi
    
    log_status "Detected Cray system"
    
    # Find Cray compiler wrappers
    find_cray_wrappers
    
    # Setup standard compiler variables
    setup_standard_compilers
    
    # Setup CMake-specific compiler variables
    setup_cmake_compilers
    
    # Setup GPU compilers
    setup_gpu_compilers
    
    # Setup GPU-aware MPI
    setup_gpu_mpi
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    
    log_status "Cray compiler configuration complete"
    
    if [[ $VERBOSE -eq 1 ]]; then
        echo ""
        echo "Compiler environment variables set:"
        echo "===================================="
        echo "Standard variables:"
        [[ -n "${CC}" ]] && echo "  CC=${CC}"
        [[ -n "${CXX}" ]] && echo "  CXX=${CXX}"
        [[ -n "${FC}" ]] && echo "  FC=${FC}"
        echo ""
        echo "CMake variables:"
        [[ -n "${CMAKE_C_COMPILER}" ]] && echo "  CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        [[ -n "${CMAKE_CXX_COMPILER}" ]] && echo "  CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        [[ -n "${CMAKE_Fortran_COMPILER}" ]] && echo "  CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
        echo ""
        echo "GPU variables:"
        [[ -n "${CUDACXX}" ]] && echo "  CUDACXX=${CUDACXX}"
        [[ -n "${CUDAHOSTCXX}" ]] && echo "  CUDAHOSTCXX=${CUDAHOSTCXX}"
        [[ -n "${CMAKE_CUDA_HOST_COMPILER}" ]] && echo "  CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
        [[ -n "${HIPCXX}" ]] && echo "  HIPCXX=${HIPCXX}"
        [[ -n "${HIPHOSTCXX}" ]] && echo "  HIPHOSTCXX=${HIPHOSTCXX}"
        [[ -n "${CMAKE_HIP_HOST_COMPILER}" ]] && echo "  CMAKE_HIP_HOST_COMPILER=${CMAKE_HIP_HOST_COMPILER}"
        echo ""
        echo "MPI/GTL libraries:"
        [[ -n "${CRAY_MPI_LIB}" ]] && echo "  CRAY_MPI_LIB=${CRAY_MPI_LIB}"
        [[ -n "${CRAY_GTL_LIB}" ]] && echo "  CRAY_GTL_LIB=${CRAY_GTL_LIB}"
        echo ""
    fi
    
    if [[ $DEBUG -eq 1 ]]; then
        echo ""
        echo "Debug: Export commands for manual setup"
        echo "========================================"
        [[ -n "${CC}" ]] && echo "export CC=\"${CC}\""
        [[ -n "${CXX}" ]] && echo "export CXX=\"${CXX}\""
        [[ -n "${FC}" ]] && echo "export FC=\"${FC}\""
        [[ -n "${CMAKE_C_COMPILER}" ]] && echo "export CMAKE_C_COMPILER=\"${CMAKE_C_COMPILER}\""
        [[ -n "${CMAKE_CXX_COMPILER}" ]] && echo "export CMAKE_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\""
        [[ -n "${CMAKE_Fortran_COMPILER}" ]] && echo "export CMAKE_Fortran_COMPILER=\"${CMAKE_Fortran_COMPILER}\""
        [[ -n "${CUDAHOSTCXX}" ]] && echo "export CUDAHOSTCXX=\"${CUDAHOSTCXX}\""
        [[ -n "${CMAKE_CUDA_HOST_COMPILER}" ]] && echo "export CMAKE_CUDA_HOST_COMPILER=\"${CMAKE_CUDA_HOST_COMPILER}\""
        [[ -n "${HIPHOSTCXX}" ]] && echo "export HIPHOSTCXX=\"${HIPHOSTCXX}\""
        [[ -n "${CMAKE_HIP_HOST_COMPILER}" ]] && echo "export CMAKE_HIP_HOST_COMPILER=\"${CMAKE_HIP_HOST_COMPILER}\""
        echo ""
    fi
}

# Run main function
main

# Mark setup as complete
export CRAY_SETUP_COMPLETE=1

# Return success
return 0 2>/dev/null || exit 0
