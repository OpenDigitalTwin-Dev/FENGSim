cmake_minimum_required(VERSION 3.0)
project(IndexingCSVParser)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_DEBUG_POSTFIX _d)


file(GLOB_RECURSE srcfiles src/*.cpp include/*.h include/*.hpp)

add_library(indexingcsvparser STATIC ${srcfiles})
target_include_directories(indexingcsvparser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS indexingcsvparser
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include)

add_subdirectory(example)
enable_testing()
#add_subdirectory(test)
