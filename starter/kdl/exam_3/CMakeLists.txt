cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(kdl_api)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

message(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/DAE/install/include/)
include_directories(/usr/include/eigen3)
aux_source_directory(${CMAKE_BINARY_DIR}/../src solver_DIR_SRCS)
add_executable(kdl ${solver_DIR_SRCS})
target_link_libraries(kdl ${CMAKE_BINARY_DIR}/../../../../toolkit/DAE/install/kdl_install/lib/liborocos-kdl.so)
set_target_properties(kdl PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
