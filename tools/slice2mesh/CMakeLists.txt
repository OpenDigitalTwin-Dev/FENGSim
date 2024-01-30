cmake_minimum_required(VERSION 3.1)
project(slice2mesh)

#find_package(Boost)
#if(Boost_FOUND)
#  message("Boost FOUND")
#  include_directories(${Boost_INCLUDE_DIRS})
#  link_directories(${Boost_LIBRARY_DIRS})
#  add_definitions(${Boost_DEFINITIONS})
#endif()

#find_package(Eigen3)
#if(Eigen3_FOUND)
#  message("Eigen3 FOUND")
#  include_directories(${Eigen3_INCLUDE_DIRS})
#  link_directories(${Eigen3_LIBRARY_DIRS})
#  add_definitions(${Eigen3_DEFINITIONS})
#endif()

message(${CMAKE_BINARY_DIR})

include_directories("include")
include_directories("src")
add_executable (slice2mesh_exec ../src/main.cpp ../src/plc2tet.cpp  ../src/slice2plc.cpp  ../src/trianglulate.cpp ../src/edge_processing.cpp)
target_link_libraries (slice2mesh_exec ${CMAKE_BINARY_DIR}/../../tetgen_install/lib/libtetgen.so ${CMAKE_BINARY_DIR}/../../triangle_install/lib/libtriangle.so)
#${Boost_LIBRARIES})

install(TARGETS
  slice2mesh_exec
  # for executables and dll on Win
  RUNTIME DESTINATION bin
  # shared libraries
  LIBRARY DESTINATION lib
  # for static libraries
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
  )




