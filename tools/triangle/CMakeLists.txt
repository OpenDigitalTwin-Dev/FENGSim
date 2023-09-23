cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(triangle_api)

include_directories("src")
add_executable (triangle ../src/triangle.c)
target_link_libraries (triangle /usr/lib/x86_64-linux-gnu/libm.so)

find_package(X11)
if(X11_FOUND)
  message("X11 FOUND")
  message(${X11_LIBRARIES})
  include_directories(${X11_INCLUDE_DIRS})
  link_directories(${X11_LIBRARY_DIRS})
  add_definitions(${X11_DEFINITIONS})
endif()
add_executable (showme ../src/showme.c)
target_link_libraries (showme ${X11_LIBRARIES})

file(COPY "A.poly" DESTINATION ${CMAKE_BINARY_DIR}/)
