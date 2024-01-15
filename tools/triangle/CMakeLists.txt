cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(triangle_api)

set(var ON)
if (var)
  include_directories("src")
  add_executable (triangle_run ../src/triangle.c)
  target_link_libraries (triangle_run /usr/lib/x86_64-linux-gnu/libm.so)
else ()
  include_directories("src")
  add_executable (triangle_run ../src/tricall.c ../src/triangle.c)
  target_link_libraries (triangle_run /usr/lib/x86_64-linux-gnu/libm.so)
endif()

install(TARGETS
  triangle_run
  # for executables and dll on Win
  RUNTIME DESTINATION bin
  # shared libraries
  LIBRARY DESTINATION lib
  # for static libraries
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
  )
install(FILES
  src/triangle.h
  DESTINATION include
  )
install(FILES
  A.poly
  DESTINATION bin
  )  
  
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
install(TARGETS
  showme
  # for executables and dll on Win
  RUNTIME DESTINATION bin
  # shared libraries
  LIBRARY DESTINATION lib
  # for static libraries
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
  )

file(COPY "A.poly" DESTINATION ${CMAKE_BINARY_DIR}/)
