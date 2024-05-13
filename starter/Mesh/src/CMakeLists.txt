cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(mesh_api)

find_package(Netgen PATHS /home/jiping/FENGSim/toolkit/install/netgen_install/lib/cmake/netgen)
if(Netgen_FOUND)
  message("NETGEN FOUND")
  #message(${NETGEN_LIBRARIES})
  include_directories(${Netgen_INCLUDE_DIRS})
  link_directories(${Netgen_LIBRARY_DIRS})
  add_definitions(${Netgen_DEFINITIONS})
endif()

aux_source_directory(src solver_DIR_SRCS)
add_executable (mesh src/nglib_occ.cpp)
target_link_libraries (mesh ${Netgen_LIBRARIES})
#set_target_properties(mesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#file(COPY "log" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "data/vtk" DESTINATION ${CMAKE_BINARY_DIR}/data)
#file(COPY "conf" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "solver/conf" DESTINATION ${CMAKE_BINARY_DIR}/solver)

