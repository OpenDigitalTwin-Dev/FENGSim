cmake_minimum_required(VERSION 3.1)
project(slice2mesh)

add_definitions(-DCINOLIB_USES_BOOST)
add_definitions(-DCINOLIB_USES_TRIANGLE)
add_definitions(-DCINOLIB_USES_TETGEN)
add_definitions(-DTETLIBRARY)
message(${CMAKE_BINARY_DIR})
message(${CMAKE_SOURCE_DIR})

# for cinolib
include_directories(${CMAKE_SOURCE_DIR}/../cinolib/include)
include_directories(/usr/include/eigen3)
include_directories(${CMAKE_SOURCE_DIR}/../triangle_install/include)
include_directories(${CMAKE_SOURCE_DIR}/../tetgen_install/include)
# for slice2mesh
include_directories("src")

add_executable (slice2mesh_exec ../src/main.cpp ../src/plc2tet.cpp  ../src/slice2plc.cpp  ../src/trianglulate.cpp ../src/edge_processing.cpp)
target_link_libraries (slice2mesh_exec ${CMAKE_SOURCE_DIR}/../tetgen_install/lib/libtetgen.so ${CMAKE_SOURCE_DIR}/../triangle_install/lib/libtriangle.so)

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

install(FILES
  msh2geo.py
  DESTINATION bin
  )

file(COPY "msh2geo.py" DESTINATION ${CMAKE_BINARY_DIR}/)

