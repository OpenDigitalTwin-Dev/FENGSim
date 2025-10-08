cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(m++_api)

message(${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/cmake/ibamr/)

find_package(HDF5)
if(HDF5_FOUND)
  message("HDF5 FOUND")
  include_directories(${HDF5_INCLUDE_DIRS})
  link_directories(${HDF5_LIBRARY_DIRS})
  add_definitions(${HDF5_DEFINITIONS})
endif()

find_package(MPI)
if(MPI_FOUND)
  message("MPI FOUND")
  include_directories(${MPI_INCLUDE_DIRS})
  link_directories(${MPI_LIBRARY_DIRS})
  add_definitions(${MPI_DEFINITIONS})
endif()

find_package(IBAMR PATHS ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/cmake/ibamr/)
if(IBAMR_FOUND)
  message("IBAMR FOUND")
  include_directories(${IBAMR_INCLUDE_DIRS})
  link_directories(${IBAMR_LIBRARY_DIRS})
  add_definitions(${IBAMR_DEFINITIONS})
endif()

# samrai libs:
SET(SAMRAI_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/samrai_install/include/")
SET(SAMRAI2d_LIBRARIES)
SET(SAMRAI3d_LIBRARIES)
ADD_LIBRARY(SAMRAI UNKNOWN IMPORTED)
FIND_LIBRARY(SAMRAI_path SAMRAI REQUIRED HINTS "${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/samrai_install/lib")
LIST(APPEND SAMRAI2d_LIBRARIES ${SAMRAI_path})
LIST(APPEND SAMRAI3d_LIBRARIES ${SAMRAI_path})
message(${SAMRAI2d_LIBRARIES})
message(${SAMRAI3d_LIBRARIES})

include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include/)
include_directories(/usr/include/eigen3/)
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/petsc_install/include/)
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/libmesh_install/include/)
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/samrai_install/include/)
include_directories(${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/include/)



add_definitions(-DNDIM=2)
add_executable (multix example.cpp)

target_link_libraries (multix ${IBAMR_LIBRARIES} ${HDF5_LIBRARIES} ${SAMRAI2d_LIBRARIES} ${PETSC_LIBRARIES})

target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/libBUNDLED_MUPARSER.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/libIBAMR2d.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/libIBAMR3d.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/libIBTK2d.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/ibamr_install/lib/libIBTK3d.so)

target_link_libraries (multix /usr/lib/x86_64-linux-gnu/libmetis.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/libmesh_install/lib/libmesh_devel.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/CFD/install/libmesh_install/lib/libtimpi_devel.so)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/petsc_install/lib/libpetsc.so)
target_link_libraries (multix /usr/lib/x86_64-linux-gnu/libmpi.so)



  
