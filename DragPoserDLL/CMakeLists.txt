# Set the minimum required version of CMake for this project
cmake_minimum_required(VERSION 3.15...3.22)

# Define the project name and the languages used, in this case C++
project(DragPoserDLL LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Include the FetchContent module to handle dependencies
include(FetchContent)

# Set a CMake policy to its new behavior
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW)

# Find and load settings for an installed Python3
find_package(Python3 COMPONENTS Interpreter Development)

# Output Python3 configuration details for debugging
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")

# Add a compile definition for Python3 executable path
add_compile_definitions(PYTHON3_EXECUTABLE="${Python3_EXECUTABLE}")

# Declare an external project (pybind11) to be fetched from a Git repository
FetchContent_Declare(pybind11_lib
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG e250155
)

# Check if the pybind11 content has been fetched, and if not, fetch and populate it
FetchContent_GetProperties(pybind11_lib)
if(NOT pybind11_lib_POPULATED)
    FetchContent_Populate(pybind11_lib)
    # Add the pybind11_lib subdirectory to the build, specifying source and binary directories
    add_subdirectory(${pybind11_lib_SOURCE_DIR} ${pybind11_lib_BINARY_DIR})
endif()

# Add an executable target to the project, specifying the source file
add_executable(DragPoserDebug main.cpp exportFunc.cpp exportFunc.h utils.cpp utils.h)

# Link the executable target with the pybind11 embed library
target_link_libraries(DragPoserDebug PRIVATE pybind11::embed)

# Add library for binding with Unity
add_library(DragPoserDLL SHARED exportFunc.cpp exportFunc.h utils.cpp utils.h)

# Link the executable target with the pybind11 embed library
target_link_libraries(DragPoserDLL PRIVATE pybind11::embed)
