cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(kdl_api)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)


message(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/../../../toolkit/Geometry/install/gmsh_install/include/)
aux_source_directory(${CMAKE_BINARY_DIR}/../src solver_DIR_SRCS)
add_executable(gmsh ${solver_DIR_SRCS})
target_link_libraries(gmsh ${CMAKE_BINARY_DIR}/../../../toolkit/Geometry/install/gmsh_install/lib/libgmsh.so ${OpenCASCADE_LIBRARIES})
set_target_properties(gmsh PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
