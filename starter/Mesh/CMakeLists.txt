cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(mesh_api)

find_package(Netgen PATHS /home/jiping/FENGSim/toolkit/install/netgen_install/lib/cmake/netgen)
if(Netgen_FOUND)
  message("NETGEN FOUND")
  message(${NETGEN_INCLUDE_DIRS})
  message(${CMAKE_BINARY_DIR})
  include_directories(${NETGEN_INCLUDE_DIRS})
  link_directories(${NETGEN_LIBRARY_DIRS})
  add_definitions(${NETGEN_DEFINITIONS})
endif()

aux_source_directory(src solver_DIR_SRCS)
message(${solver_DIR_SRCS})
add_executable (mesh ${solver_DIR_SRCS})
target_link_libraries (mesh ${CMAKE_BINARY_DIR}/../../../toolkit/install/netgen_install/lib/libnglib.so ${CMAKE_BINARY_DIR}/../../../toolkit/install/netgen_install/lib/libngcore.so)
set_target_properties(mesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#file(COPY "log" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "data/vtk" DESTINATION ${CMAKE_BINARY_DIR}/data)
#file(COPY "conf" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "solver/conf" DESTINATION ${CMAKE_BINARY_DIR}/solver)


