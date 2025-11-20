#!/bin/bash
# ==============================================================================
# Cray System Environment Setup for CMake - Part 2: Build Flags
# ==============================================================================
# This script sets up standard environment variables that CMake reads
# Based on CrayDetection.cmake automatic fixes
#
# Sets: LDFLAGS, LIBS, CFLAGS, CXXFLAGS, PKG_CONFIG_PATH, CMAKE_PREFIX_PATH
#
# Usage:
#   source cray_setup_flags.sh [options]
#
# Options:
#   --no-auto-fixes    : Disable automatic Cray system fixes
#   --check-modules    : Check for stale configuration
#   --verbose         : Show verbose output
#   --debug           : Show debug output
#   --help            : Show this help message
# ==============================================================================

# Initialize variables
VERBOSE=0
DEBUG=0
NO_AUTO_FIXES=0
CHECK_MODULES=0
SCRIPT_NAME="cray_setup_flags.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-auto-fixes)
            NO_AUTO_FIXES=1
            shift
            ;;
        --check-modules)
            CHECK_MODULES=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --debug)
            DEBUG=1
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Usage: source $SCRIPT_NAME [options]"
            echo ""
            echo "Options:"
            echo "  --no-auto-fixes    : Disable automatic Cray system fixes"
            echo "  --check-modules    : Check for stale configuration"
            echo "  --verbose         : Show verbose output"
            echo "  --debug           : Show debug output"
            echo "  --help            : Show this help message"
            echo ""
            echo "Sets environment variables:"
            echo "  LDFLAGS           : Linker flags"
            echo "  LIBS              : Libraries to link"
            echo "  CFLAGS            : C compiler flags"
            echo "  CXXFLAGS          : C++ compiler flags"
            echo "  CUDAFLAGS         : CUDA compiler flags"
            echo "  PKG_CONFIG_PATH   : Package config search path"
            echo "  CMAKE_PREFIX_PATH : CMake module search path"
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
# Detect Cray Environment
# ==============================================================================

# Quick fix - add these lines at the top of detect_cray_system() in cray_setup_flags.sh
detect_cray_system() {
    local on_cray=0
    
    log_debug "Checking for Cray environment"
    
    # Check for Cray environment variables
    if [[ -n "${CRAYPE_VERSION}" ]]; then
        on_cray=1
        log_verbose "Detected Cray Programming Environment: ${CRAYPE_VERSION}"
    fi
    
    if [[ -n "${CRAY_MPICH_DIR}" ]]; then
        on_cray=1
        log_verbose "Detected CRAY_MPICH_DIR: ${CRAY_MPICH_DIR}"
    fi

    # Check for cc/CC wrappers
    if command -v cc >/dev/null 2>&1 && command -v CC >/dev/null 2>&1; then
        if cc --version 2>&1 | grep -q "Cray" || CC --version 2>&1 | grep -q "Cray"; then
            on_cray=1
            log_verbose "Detected Cray compiler wrappers"
        fi
    fi
    
    # Fix: Actually return the status correctly
    if [[ $on_cray -eq 1 ]]; then
        return 0  # Success = on Cray
    else
        return 1  # Failure = not on Cray
    fi
}

# ==============================================================================
# Module Checking (optional)
# ==============================================================================

check_stale_config() {
    if [[ $CHECK_MODULES -eq 0 ]]; then
        return 0
    fi
    
    log_status "Checking for configuration changes"
    
    local config_file=".cray_module_state"
    local current_modules="${LOADEDMODULES:-none}"
    local current_pe_env="${PE_ENV:-none}"
    
    if [[ -f "$config_file" ]]; then
        source "$config_file"
        
        if [[ "${current_modules}" != "${CACHED_MODULES:-}" ]]; then
            log_warning "Module environment changed since last configuration"
            log_verbose "Previous: ${CACHED_MODULES:-none}"
            log_verbose "Current: ${current_modules}"
            return 1
        fi
        
        if [[ "${current_pe_env}" != "${CACHED_PE_ENV:-}" ]]; then
            log_warning "PE_ENV changed from ${CACHED_PE_ENV:-none} to ${current_pe_env}"
            return 1
        fi
    fi
    
    # Save current state
    cat > "$config_file" <<EOF
CACHED_MODULES="${current_modules}"
CACHED_PE_ENV="${current_pe_env}"
EOF
    
    return 0
}

# ==============================================================================
# Get Cray Compiler Flags
# ==============================================================================

get_cray_cflags() {
    local compiler="${1:-CC}"
    local flags=""
    
    if command -v "$compiler" >/dev/null 2>&1; then
        flags=$("$compiler" --cray-print-opts=cflags 2>/dev/null)
        if [[ $? -eq 0 && -n "$flags" ]]; then
            echo "$flags"
            return 0
        fi
    fi
    
    return 1
}

# ==============================================================================
# Get Cray Libraries
# ==============================================================================

get_cray_libs() {
    local compiler="${1:-CC}"
    local libs=""
    
    if command -v "$compiler" >/dev/null 2>&1; then
        libs=$("$compiler" --cray-print-opts=libs 2>/dev/null)
        if [[ $? -eq 0 && -n "$libs" ]]; then
            echo "$libs"
            return 0
        fi
    fi
    
    return 1
}

# ==============================================================================
# Clean Cray Libraries (remove problematic flags)
# ==============================================================================

clean_cray_libs() {
    local libs="$1"
    local cleaned=""
    
    # Remove --as-needed flags
    cleaned=$(echo "$libs" | sed 's/-Wl,--as-needed,//g; s/,--no-as-needed//g')
    
    # Convert comma-separated -l flags to space-separated
    cleaned=$(echo "$cleaned" | sed 's/,-l/ -l/g')
    
    # Remove GPU-related flags that might conflict
    cleaned=$(echo "$cleaned" | sed 's/--offload-arch=[^ ]*//g')
    cleaned=$(echo "$cleaned" | sed 's/--hip-link//g')
    cleaned=$(echo "$cleaned" | sed 's/-fgpu-rdc//g')
    cleaned=$(echo "$cleaned" | sed 's/-xhip//g')
    
    echo "$cleaned"
}

# ==============================================================================
# Extract linker flags vs libraries
# ==============================================================================

split_libs_and_flags() {
    local input="$1"
    local -n out_ldflags=$2
    local -n out_libs=$3
    
    out_ldflags=""
    out_libs=""
    
    for item in $input; do
        case "$item" in
            -L*|-Wl,*|--*)
                out_ldflags="${out_ldflags:+$out_ldflags }$item"
                ;;
            -l*)
                out_libs="${out_libs:+$out_libs }$item"
                ;;
            *)
                # Assume other flags go to LDFLAGS
                out_ldflags="${out_ldflags:+$out_ldflags }$item"
                ;;
        esac
    done
}

# ==============================================================================
# Get PKG_CONFIG_PATH from Cray
# ==============================================================================

get_cray_pkg_config_path() {
    local compiler="${1:-CC}"
    local pkg_path=""
    
    if command -v "$compiler" >/dev/null 2>&1; then
        pkg_path=$("$compiler" --cray-print-opts=pkg_config_path 2>/dev/null)
        if [[ $? -eq 0 && -n "$pkg_path" ]]; then
            echo "$pkg_path"
            return 0
        fi
        
        # Fallback to PKG_CONFIG_PATH
        pkg_path=$("$compiler" --cray-print-opts=PKG_CONFIG_PATH 2>/dev/null)
        if [[ $? -eq 0 && -n "$pkg_path" ]]; then
            echo "$pkg_path"
            return 0
        fi
    fi
    
    return 1
}

# ==============================================================================
# Detect MPI Library
# ==============================================================================

detect_mpi_library() {
    local mpi_lib=""
    
    # Try pkg-config first
    if command -v pkg-config >/dev/null 2>&1; then
        local pkg_path=$(get_cray_pkg_config_path)
        if [[ -n "$pkg_path" ]]; then
            export PKG_CONFIG_PATH="${pkg_path}:${PKG_CONFIG_PATH:-}"
        fi
        
        local mpi_libs=$(pkg-config --libs mpich 2>/dev/null | grep -o '\-lmpi_[^ ]*')
        for lib in $mpi_libs; do
            lib=${lib#-l}
            if [[ "$lib" =~ ^mpi_(gnu|cray|intel) ]]; then
                mpi_lib="$lib"
                log_debug "Found MPI library via pkg-config: $mpi_lib"
                break
            fi
        done
    fi
    
    # Fallback: search filesystem
    if [[ -z "$mpi_lib" ]]; then
        local search_paths="${MPICH_DIR}/lib ${CRAY_MPICH_DIR}/lib"
        for path in $search_paths; do
            if [[ -d "$path" ]]; then
                for libfile in "$path"/libmpi_*.so "$path"/libmpi_*.a; do
                    if [[ -f "$libfile" ]]; then
                        local libname=$(basename "$libfile")
                        libname=${libname#lib}
                        libname=${libname%.so*}
                        libname=${libname%.a}
                        if [[ "$libname" =~ ^mpi_(gnu|cray|intel) ]]; then
                            mpi_lib="$libname"
                            log_debug "Found MPI library in $path: $mpi_lib"
                            break 2
                        fi
                    fi
                done
            fi
        done
    fi
    
    # Last resort: heuristic based on compiler
    if [[ -z "$mpi_lib" ]]; then
        if CC --version 2>&1 | grep -q "GNU"; then
            # Try to get version
            local gcc_version=$(CC --version | grep -oP 'gcc.*\K[0-9]+\.[0-9]+' | head -1)
            if [[ -n "$gcc_version" ]]; then
                local major=$(echo "$gcc_version" | cut -d. -f1)
                local minor=$(echo "$gcc_version" | cut -d. -f2)
                mpi_lib="mpi_gnu_${major}${minor}"
            else
                mpi_lib="mpi_gnu_123"
            fi
            log_warning "Using heuristic MPI library: $mpi_lib"
        elif CC --version 2>&1 | grep -q "Cray"; then
            mpi_lib="mpi_cray"
            log_warning "Using heuristic MPI library: $mpi_lib"
        fi
    fi
    
    echo "$mpi_lib"
}

# ==============================================================================
# Main Setup
# ==============================================================================

main() {
    log_status "Cray System CMake Environment Setup - Build Flags"
    
    # Check if auto-fixes are disabled
    if [[ $NO_AUTO_FIXES -eq 1 ]]; then
        log_status "Auto-fixes disabled by user"
        export ERF_DISABLE_CRAY_AUTO_FIXES=ON
        return 0
    fi
    
    # Detect Cray system
    if ! detect_cray_system; then
        log_status "Not on a Cray system, skipping Cray-specific setup"
        return 0
    fi
    
    log_status "Detected Cray system"
    
    # Check for stale configuration
    if ! check_stale_config; then
        log_warning "Configuration may be stale - consider cleaning build directory"
    fi
    
    # ==============================================================================
    # Setup CFLAGS and CXXFLAGS
    # ==============================================================================
    
    log_status "Setting up compiler flags"
    
    # Get Cray C flags
    local cray_cflags=$(get_cray_cflags cc)
    if [[ -n "$cray_cflags" ]]; then
        export CFLAGS="${CFLAGS:+$CFLAGS }$cray_cflags"
        log_verbose "Added Cray flags to CFLAGS"
        log_debug "CFLAGS=$CFLAGS"
    fi
    
    # Get Cray C++ flags
    local cray_cxxflags=$(get_cray_cflags CC)
    if [[ -n "$cray_cxxflags" ]]; then
        export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }$cray_cxxflags"
        log_verbose "Added Cray flags to CXXFLAGS"
        log_debug "CXXFLAGS=$CXXFLAGS"
    fi
    
    # For CUDA if nvcc_wrapper is being used
    if [[ -n "${CUDA_HOME}" ]] || command -v nvcc >/dev/null 2>&1; then
        if [[ -n "$cray_cxxflags" ]]; then
            export CUDAFLAGS="${CUDAFLAGS:+$CUDAFLAGS }$cray_cxxflags"
            log_verbose "Added Cray flags to CUDAFLAGS for nvcc_wrapper"
            log_debug "CUDAFLAGS=$CUDAFLAGS"
        fi
    fi
    
    # ==============================================================================
    # Setup LDFLAGS and LIBS
    # ==============================================================================
    
    log_status "Setting up linker flags and libraries"
    
    # Get Cray libraries
    local cray_libs=$(get_cray_libs CC)
    if [[ -n "$cray_libs" ]]; then
        local cleaned_libs=$(clean_cray_libs "$cray_libs")
        log_verbose "Cleaned Cray libraries: $cleaned_libs"
        
        # Split into LDFLAGS and LIBS
        local extracted_ldflags=""
        local extracted_libs=""
        split_libs_and_flags "$cleaned_libs" extracted_ldflags extracted_libs
        
        # Add --no-as-needed to LDFLAGS
        extracted_ldflags="-Wl,--no-as-needed ${extracted_ldflags}"
        
        # Set LDFLAGS
        if [[ -n "$extracted_ldflags" ]]; then
            export LDFLAGS="${LDFLAGS:+$LDFLAGS }$extracted_ldflags"
            log_verbose "Added Cray linker flags to LDFLAGS"
            log_debug "LDFLAGS=$LDFLAGS"
        fi
        
        # Set LIBS
        if [[ -n "$extracted_libs" ]]; then
            export LIBS="${LIBS:+$LIBS }$extracted_libs"
            log_verbose "Added Cray libraries to LIBS"
            log_debug "LIBS=$LIBS"
        fi
    else
        log_warning "Could not retrieve Cray library paths"
    fi
    
    # ==============================================================================
    # Add GPU-aware MPI libraries
    # ==============================================================================
    
    if [[ "${MPICH_GPU_SUPPORT_ENABLED}" == "1" ]]; then
        log_status "Configuring GPU-aware MPI"
        
        local mpi_lib=$(detect_mpi_library)
        local gtl_lib=""
        
        # Determine GTL library based on accelerator
        if [[ -n "${CRAY_ACCEL_TARGET}" ]]; then
            case "${CRAY_ACCEL_TARGET}" in
                nvidia*)
                    gtl_lib="mpi_gtl_cuda"
                    ;;
                amd*)
                    gtl_lib="mpi_gtl_hsa"
                    ;;
            esac
        fi
        
        if [[ -n "$mpi_lib" && -n "$gtl_lib" ]]; then
            local mpi_libs="-l$mpi_lib -l$gtl_lib"
            export LIBS="${LIBS:+$LIBS }$mpi_libs"
            log_status "Added GPU-aware MPI libraries to LIBS"
            log_verbose "MPI libraries: $mpi_libs"
        fi
    fi
    
    # ==============================================================================
    # Setup PKG_CONFIG_PATH
    # ==============================================================================
    
    local pkg_path=$(get_cray_pkg_config_path CC)
    if [[ -n "$pkg_path" ]]; then
        export PKG_CONFIG_PATH="${pkg_path}:${PKG_CONFIG_PATH:-}"
        log_status "Set PKG_CONFIG_PATH for Cray modules"
        log_debug "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
    fi
    
    # ==============================================================================
    # Setup CMAKE_PREFIX_PATH for libraries
    # ==============================================================================
    
    local prefix_paths=""
    
    # CUDA math libraries
    if [[ -n "${CUDA_HOME}" ]]; then
        local cuda_math_path="${CUDA_HOME}/../../math_libs/lib64"
        if [[ -d "$cuda_math_path" ]]; then
            prefix_paths="${prefix_paths:+$prefix_paths:}$cuda_math_path"
            log_verbose "Added CUDA math libraries to CMAKE_PREFIX_PATH"
        fi
    fi
    
    # NetCDF
    if [[ -n "${NETCDF_DIR}" ]]; then
        prefix_paths="${prefix_paths:+$prefix_paths:}${NETCDF_DIR}"
        log_verbose "Added NetCDF to CMAKE_PREFIX_PATH"
    fi
    
    # HDF5
    if [[ -n "${HDF5_DIR}" ]]; then
        prefix_paths="${prefix_paths:+$prefix_paths:}${HDF5_DIR}"
        log_verbose "Added HDF5 to CMAKE_PREFIX_PATH"
    elif [[ -n "${HDF5_ROOT}" ]]; then
        prefix_paths="${prefix_paths:+$prefix_paths:}${HDF5_ROOT}"
        log_verbose "Added HDF5_ROOT to CMAKE_PREFIX_PATH"
    fi
    
    # FFTW
    if [[ -n "${FFTW_DIR}" ]]; then
        prefix_paths="${prefix_paths:+$prefix_paths:}${FFTW_DIR}"
        log_verbose "Added FFTW to CMAKE_PREFIX_PATH"
    elif [[ -n "${CRAY_FFTW_DIR}" ]]; then
        prefix_paths="${prefix_paths:+$prefix_paths:}${CRAY_FFTW_DIR}"
        log_verbose "Added CRAY_FFTW to CMAKE_PREFIX_PATH"
    fi
    
    if [[ -n "$prefix_paths" ]]; then
        export CMAKE_PREFIX_PATH="${prefix_paths}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        log_status "Set CMAKE_PREFIX_PATH for Cray libraries"
        log_debug "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
    fi
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    
    log_status "Cray build flags configuration complete"
    
    if [[ $VERBOSE -eq 1 ]]; then
        echo ""
        echo "Environment variables set:"
        echo "=========================="
        [[ -n "${CFLAGS}" ]] && echo "  CFLAGS"
        [[ -n "${CXXFLAGS}" ]] && echo "  CXXFLAGS"
        [[ -n "${CUDAFLAGS}" ]] && echo "  CUDAFLAGS"
        [[ -n "${LDFLAGS}" ]] && echo "  LDFLAGS"
        [[ -n "${LIBS}" ]] && echo "  LIBS"
        [[ -n "${PKG_CONFIG_PATH}" ]] && echo "  PKG_CONFIG_PATH"
        [[ -n "${CMAKE_PREFIX_PATH}" ]] && echo "  CMAKE_PREFIX_PATH"
        echo ""
    fi
    
    if [[ $DEBUG -eq 1 ]]; then
        echo ""
        echo "Debug: Full environment for CMake"
        echo "================================="
        [[ -n "${CFLAGS}" ]] && echo "export CFLAGS=\"${CFLAGS}\""
        [[ -n "${CXXFLAGS}" ]] && echo "export CXXFLAGS=\"${CXXFLAGS}\""
        [[ -n "${CUDAFLAGS}" ]] && echo "export CUDAFLAGS=\"${CUDAFLAGS}\""
        [[ -n "${LDFLAGS}" ]] && echo "export LDFLAGS=\"${LDFLAGS}\""
        [[ -n "${LIBS}" ]] && echo "export LIBS=\"${LIBS}\""
        [[ -n "${PKG_CONFIG_PATH}" ]] && echo "export PKG_CONFIG_PATH=\"${PKG_CONFIG_PATH}\""
        [[ -n "${CMAKE_PREFIX_PATH}" ]] && echo "export CMAKE_PREFIX_PATH=\"${CMAKE_PREFIX_PATH}\""
        echo ""
        echo "You can copy-paste these exports or source this script"
        echo ""
    fi
}

# Run main function
main

# Mark setup as complete
export CRAY_SETUP_COMPLETE=1

# Return success
return 0 2>/dev/null || exit 0
