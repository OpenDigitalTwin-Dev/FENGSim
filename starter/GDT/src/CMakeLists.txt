cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(registration_api)

set(PCL_DIR ../../../toolkit/Geometry/install/pcl_install/share/pcl-1.11/)
find_package(PCL 1.3 REQUIRED)
if(PCL_FOUND)
  message("PCL FOUND")
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif()

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable (fengsim_meas main.cpp odt_pcl_tools.cpp)
target_link_libraries (fengsim_meas ${PCL_LIBRARIES})

