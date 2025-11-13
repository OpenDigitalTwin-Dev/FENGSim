cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(m++_api)

#message(${CMAKE_BINARY_DIR})

include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/slepc_install/include/)
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/petsc_install/include/)

find_package(MPI)
if(MPI_FOUND)
  message("MPI FOUND")
  message(${MPI_LIBRARIES})
  include_directories(${MPI_INCLUDE_DIRS})
  link_directories(${MPI_LIBRARY_DIRS})
  add_definitions(${MPI_DEFINITIONS})
endif()
include_directories("/usr/include/x86_64-linux-gnu/openmpi/")

aux_source_directory(${CMAKE_BINARY_DIR}/../src ex7_srcs)

add_executable (multix ${ex7_srcs})

target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/slepc_install/lib/libslepc.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/petsc_install/lib/libpetsc.so)
target_link_libraries (multix ${MPI_LIBRARIES})
