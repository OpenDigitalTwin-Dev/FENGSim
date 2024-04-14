cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(webgl_api)

find_package(OCE)
if(OCE_FOUND)
  message("OCE FOUND")
  message(${OCE_LIBRARIES})
  include_directories(${OCE_INCLUDE_DIRS})
  link_directories(${OCE_LIBRARY_DIRS})
  add_definitions(${OCE_DEFINITIONS})
endif()

include_directories("src")
aux_source_directory(src solver_DIR_SRCS)
add_executable (oceSolver ${solver_DIR_SRCS})
target_link_libraries (oceSolver ${VTK_LIBRARIES} ${OCE_LIBRARIES})
set_target_properties(oceSolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

