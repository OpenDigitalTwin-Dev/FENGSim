cmake_minimum_required(VERSION 3.2)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
project(indexingcsvparsertest)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX _d)

# Download release when building, can not download .hpp directly
# since this must work on cmake 3.2.2 (travis-ci trusty)
ExternalProject_Add(Catch-get
  PREFIX ${CMAKE_BINARY_DIR}/Catch
  #URL https://github.com/philsquared/Catch/releases/download/v1.12.2/catch.hpp
  #URL_HASH MD5=6e3c2c7dd06d31ae9112b3816da24712
  URL https://github.com/catchorg/Catch2/archive/v2.6.1.tar.gz
  TIMEOUT 10
  #DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Catch
  #DOWNLOAD_NO_EXTRACT 1
  LOG_DOWNLOAD ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

# Set include path to Catch
add_library(Catch INTERFACE)
ExternalProject_Get_Property(Catch-get source_dir)
target_include_directories(Catch INTERFACE ${source_dir}/single_include)

file(GLOB_RECURSE srcfiles *.cpp)
add_executable(indexingcsvparsertest ${srcfiles})
target_link_libraries(indexingcsvparsertest indexingcsvparser Catch)
add_dependencies(indexingcsvparsertest Catch-get)

file(GLOB testdatafiles RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.csv)
foreach(tdf ${testdatafiles})
    configure_file(${tdf} ${tdf} COPYONLY)
    string(REPLACE "_lf" "_crlf" tdfcrlf ${tdf})
    configure_file(${tdf} ${tdfcrlf} NEWLINE_STYLE CRLF)
endforeach()


install(TARGETS indexingcsvparsertest
  RUNTIME DESTINATION bin
)

add_test(alltests indexingcsvparsertest)
