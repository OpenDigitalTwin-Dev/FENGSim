cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(m++_api)

message(${CMAKE_BINARY_DIR})

find_package(OpenMP)
if(OpenMP_FOUND)
  message("OpenMPMP FOUND")
  include_directories(${OpenMP_INCLUDE_DIRS})
  link_directories(${OpenMP_LIBRARY_DIRS})
  add_definitions(${OpenMP_DEFINITIONS})
endif()

include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/Particles/install/openphase_install/include)

add_executable (AdditiveNiAl ../AdditiveNiAl.cpp)

target_link_libraries (AdditiveNiAl ${CMAKE_BINARY_DIR}/../../../../toolkit/Particles/install/openphase_install/lib/libOpenPhase.a ${OpenMP_CXX_LIBRARIES})
