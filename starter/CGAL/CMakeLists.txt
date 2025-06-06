cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cgal_api)

set(CGAL_DIR ../../toolkit/Geometry/install/cgal_install/lib/cmake/CGAL)
find_package(CGAL)
if(CGAL_FOUND)
  message("CGAL FOUND")
  include_directories(${CGAL_INCLUDE_DIRS})
  link_directories(${CGAL_LIBRARY_DIRS})
  add_definitions(${CGAL_DEFINITIONS})
endif()

aux_source_directory(src cgal_DIR_SRCS)
add_executable (cgal_test ${cgal_DIR_SRCS})
target_link_libraries (cgal_test /usr/lib/x86_64-linux-gnu/libgmp.so /usr/lib/x86_64-linux-gnu/libmpfr.so)
set_target_properties(cgal_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

