cmake_minimum_required(VERSION 3.2)
project(indexingcsvparserexample)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_DEBUG_POSTFIX _d)

file(GLOB_RECURSE srcfiles *.cpp)
add_executable(indexingcsvparserexample ${srcfiles})
target_link_libraries(indexingcsvparserexample indexingcsvparser)

install(TARGETS indexingcsvparserexample
  RUNTIME DESTINATION bin
)

