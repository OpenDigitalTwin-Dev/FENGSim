cmake_minimum_required(VERSION 3.0)
project(NumHop)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_DEBUG_POSTFIX _d)


file(GLOB_RECURSE srcfiles src/*.cpp)

add_library(numhop STATIC ${srcfiles})
target_include_directories(numhop PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS numhop
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include )

enable_testing()
#add_subdirectory(test)
