# CMake/ListEAMxxSources.cmake

function(list_eamxx_sources)
    message(STATUS "Extracting EAMxx source lists from E3SM CMakeLists.txt...")
    
    # Set up E3SM expected variables
    set(SCREAM_LIB_ONLY TRUE)
    set(EAMXX_ENABLE_GPU ${ERF_ENABLE_CUDA})
    set(SCREAM_P3_SMALL_KERNELS OFF)
    set(SCREAM_SHOC_SMALL_KERNELS OFF)
    set(SCREAM_DOUBLE_PRECISION ON)
    set(SCREAM_DEBUG OFF)
    set(SCREAM_ONLY_GENERATE_BASELINES OFF)
    set(SCREAM_BASE_DIR "${CMAKE_SOURCE_DIR}/external/E3SM/components/eamxx")
    
    # Detect Kokkos GPU settings if available
    if(DEFINED Kokkos_ENABLE_CUDA)
        set(Kokkos_ENABLE_CUDA ${Kokkos_ENABLE_CUDA})
    endif()
    if(DEFINED Kokkos_ENABLE_HIP)
        set(Kokkos_ENABLE_HIP ${Kokkos_ENABLE_HIP})
    endif()
    
    # Create stub targets that E3SM CMakeLists expects
    if(NOT TARGET physics_share)
        add_library(physics_share INTERFACE)
    endif()
    if(NOT TARGET scream_share)
        add_library(scream_share INTERFACE)
    endif()
    if(NOT TARGET eamxx_physics)
        add_library(eamxx_physics INTERFACE)
    endif()
    
    # Stub the GetInputFile function
    function(GetInputFile filename)
        # No-op - we're just extracting source lists
    endfunction()
    
    # Create output directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/eamxx_source_lists)
    
    # Suppress E3SM's own messages
    set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
    
    # Process SHOC
    if(ERF_ENABLE_SHOC)
        add_subdirectory(
            ${CMAKE_SOURCE_DIR}/external/E3SM/components/eamxx/src/physics/shoc
            ${CMAKE_BINARY_DIR}/eamxx_source_lists/shoc_temp
            EXCLUDE_FROM_ALL
        )
        
        get_target_property(SHOC_SOURCES shoc SOURCES)
        get_target_property(SHOC_INCLUDE_DIRS shoc INTERFACE_INCLUDE_DIRECTORIES)
        
        # Write as CMake file
        set(SHOC_FILE ${CMAKE_BINARY_DIR}/eamxx_source_lists/EAMxxShocSources.cmake)
        file(WRITE ${SHOC_FILE} "# SHOC sources extracted from E3SM CMakeLists.txt\n")
        file(APPEND ${SHOC_FILE} "# This file can be included in CMake to use these lists\n")
        file(APPEND ${SHOC_FILE} "# Generated with ERF_ENABLE_CUDA=${ERF_ENABLE_CUDA}\n")
        file(APPEND ${SHOC_FILE} "# EAMXX_ENABLE_GPU=${EAMXX_ENABLE_GPU}\n\n")
        
        # Write sources as CMake list
        file(APPEND ${SHOC_FILE} "set(EAMXX_SHOC_SOURCES\n")
        foreach(src ${SHOC_SOURCES})
            # Use absolute paths so it works when included
            file(APPEND ${SHOC_FILE} "    ${src}\n")
        endforeach()
        file(APPEND ${SHOC_FILE} ")\n\n")
        
        # Write include directories as CMake list
        file(APPEND ${SHOC_FILE} "set(EAMXX_SHOC_INCLUDE_DIRS\n")
        if(SHOC_INCLUDE_DIRS)
            foreach(inc_dir ${SHOC_INCLUDE_DIRS})
                file(APPEND ${SHOC_FILE} "    ${inc_dir}\n")
            endforeach()
        endif()
        file(APPEND ${SHOC_FILE} ")\n\n")
        
        # Add a comment showing relative paths for easy reading
        file(APPEND ${SHOC_FILE} "# === For Reference: Relative Paths ===\n")
        file(APPEND ${SHOC_FILE} "# Sources (${SHOC_COUNT} files):\n")
        foreach(src ${SHOC_SOURCES})
            file(RELATIVE_PATH rel_src 
                 "${CMAKE_SOURCE_DIR}/external/E3SM/components/eamxx/src/physics/shoc"
                 "${src}")
            file(APPEND ${SHOC_FILE} "#   ${rel_src}\n")
        endforeach()
        
        file(APPEND ${SHOC_FILE} "#\n# Include directories:\n")
        if(SHOC_INCLUDE_DIRS)
            foreach(inc_dir ${SHOC_INCLUDE_DIRS})
                string(FIND "${inc_dir}" "${CMAKE_SOURCE_DIR}" is_in_source)
                if(NOT is_in_source EQUAL -1)
                    file(RELATIVE_PATH rel_inc "${CMAKE_SOURCE_DIR}" "${inc_dir}")
                    file(APPEND ${SHOC_FILE} "#   ${rel_inc}\n")
                else()
                    file(APPEND ${SHOC_FILE} "#   ${inc_dir}\n")
                endif()
            endforeach()
        endif()
        
        message(STATUS "  SHOC sources listed in: eamxx_source_lists/EAMxxShocSources.cmake")
        list(LENGTH SHOC_SOURCES SHOC_COUNT)
        message(STATUS "  SHOC files: ${SHOC_COUNT}")
    endif()
    
    # Process P3
    if(ERF_ENABLE_P3)
        add_subdirectory(
            ${CMAKE_SOURCE_DIR}/external/E3SM/components/eamxx/src/physics/p3
            ${CMAKE_BINARY_DIR}/eamxx_source_lists/p3_temp
            EXCLUDE_FROM_ALL
        )
        
        get_target_property(P3_SOURCES p3 SOURCES)
        get_target_property(P3_INCLUDE_DIRS p3 INTERFACE_INCLUDE_DIRECTORIES)
        
        # Write as CMake file
        set(P3_FILE ${CMAKE_BINARY_DIR}/eamxx_source_lists/EAMxxP3Sources.cmake)
        file(WRITE ${P3_FILE} "# P3 sources extracted from E3SM CMakeLists.txt\n")
        file(APPEND ${P3_FILE} "# This file can be included in CMake to use these lists\n")
        file(APPEND ${P3_FILE} "# Generated with ERF_ENABLE_CUDA=${ERF_ENABLE_CUDA}\n")
        file(APPEND ${P3_FILE} "# EAMXX_ENABLE_GPU=${EAMXX_ENABLE_GPU}\n\n")
        
        # Write sources as CMake list
        file(APPEND ${P3_FILE} "set(EAMXX_P3_SOURCES\n")
        foreach(src ${P3_SOURCES})
            file(APPEND ${P3_FILE} "    ${src}\n")
        endforeach()
        file(APPEND ${P3_FILE} ")\n\n")
        
        # Write include directories as CMake list
        file(APPEND ${P3_FILE} "set(EAMXX_P3_INCLUDE_DIRS\n")
        if(P3_INCLUDE_DIRS)
            foreach(inc_dir ${P3_INCLUDE_DIRS})
                file(APPEND ${P3_FILE} "    ${inc_dir}\n")
            endforeach()
        endif()
        file(APPEND ${P3_FILE} ")\n\n")
        
        # Add a comment showing relative paths for easy reading
        file(APPEND ${P3_FILE} "# === For Reference: Relative Paths ===\n")
        file(APPEND ${P3_FILE} "# Sources (${P3_COUNT} files):\n")
        foreach(src ${P3_SOURCES})
            file(RELATIVE_PATH rel_src 
                 "${CMAKE_SOURCE_DIR}/external/E3SM/components/eamxx/src/physics/p3"
                 "${src}")
            file(APPEND ${P3_FILE} "#   ${rel_src}\n")
        endforeach()
        
        file(APPEND ${P3_FILE} "#\n# Include directories:\n")
        if(P3_INCLUDE_DIRS)
            foreach(inc_dir ${P3_INCLUDE_DIRS})
                string(FIND "${inc_dir}" "${CMAKE_SOURCE_DIR}" is_in_source)
                if(NOT is_in_source EQUAL -1)
                    file(RELATIVE_PATH rel_inc "${CMAKE_SOURCE_DIR}" "${inc_dir}")
                    file(APPEND ${P3_FILE} "#   ${rel_inc}\n")
                else()
                    file(APPEND ${P3_FILE} "#   ${inc_dir}\n")
                endif()
            endforeach()
        endif()
        
        message(STATUS "  P3 sources listed in: eamxx_source_lists/EAMxxP3Sources.cmake")
        list(LENGTH P3_SOURCES P3_COUNT)
        message(STATUS "  P3 files: ${P3_COUNT}")
    endif()
    
    message(STATUS "Compare these against hardcoded sources in CMake/BuildERFExe.cmake")
    message(STATUS "Or include them directly: include(\${CMAKE_BINARY_DIR}/eamxx_source_lists/EAMxxShocSources.cmake)")
    
    # Restore message level
    set(CMAKE_MESSAGE_LOG_LEVEL STATUS PARENT_SCOPE)
endfunction()