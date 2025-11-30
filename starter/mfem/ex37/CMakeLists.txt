cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(mfem_api)

message(${CMAKE_BINARY_DIR})

find_package(MFEM PATHS ${CMAKE_BINARY_DIR}/../../../../toolkit/CEM/install/mfem_install/lib/cmake/mfem/)
if(MFEM_FOUND)
  message("MFEM FOUND")
  message(${MFEM_LIBRARIES})
  include_directories(${MFEM_INCLUDE_DIRS})
  link_directories(${MFEM_LIBRARY_DIRS})
  add_definitions(${MFEM_DEFINITIONS})
endif()

find_package(OpenMP REQUIRED)

aux_source_directory(${CMAKE_BINARY_DIR}/../src ex37)

add_executable (multix ${ex37})
target_link_libraries (multix ${MFEM_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries (multix ${CMAKE_BINARY_DIR}/../../../../toolkit/NLA/install/hypre_install/lib/libHYPRE.a)
