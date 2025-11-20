# CMake/CrayToolchain.cmake
# Cray System Toolchain File
if(DEFINED __CRAY_TOOLCHAIN_LOADED)
    return()
endif()
set(__CRAY_TOOLCHAIN_LOADED TRUE)

# Set system
set(CMAKE_SYSTEM_NAME CrayLinuxEnvironment)

# Force enable Cray auto-fixes when using toolchain
set(ERF_ENABLE_CRAY_AUTO_FIXES ON CACHE BOOL "Enabled by toolchain" FORCE)

# Add module path so CMakeLists.txt can find the Cray modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")