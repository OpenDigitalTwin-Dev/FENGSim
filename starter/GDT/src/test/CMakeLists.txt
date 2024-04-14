cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(registration_api)

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

#file(GLOB LIB_SOURCES test_*.cpp)
#message(${LIB_SOURCES})

# this is cmake&swig for c++&python. 
FIND_PACKAGE(SWIG REQUIRED)
if(SWIG_FOUND)
  message("SWIG FOUND")
  INCLUDE(${SWIG_USE_FILE})
  FIND_PACKAGE(PythonLibs 3.8)
  if (PythonLibs_FOUND)
    message("PythonLibs FOUND")
    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
    message(${PYTHON_INCLUDE_PATH})
    message(${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  SET(CMAKE_SWIG_FLAGS "")
  SET_SOURCE_FILES_PROPERTIES(odt_pcl.i PROPERTIES CPLUSPLUS ON)
  SWIG_ADD_LIBRARY(odt_pcl LANGUAGE python SOURCES odt_pcl.i test_uniform_sampling.cpp test_kdtree_search.cpp test_normal_estimation.cpp test_pfh_estimation.cpp test_sacia.cpp odt_pcl_tools.cpp)
  SWIG_LINK_LIBRARIES(odt_pcl ${PCL_LIBRARIES})
endif()


add_executable (test test_uniform_sampling.cpp test_kdtree_search.cpp test_normal_estimation.cpp test_pfh_estimation.cpp test_sacia.cpp odt_pcl_tools.cpp main.cpp)
target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_omp.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_normal_estimation_omp.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_fpfh_omp.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_sac_ia.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_registration.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (test test_icp.cpp)
#target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (fengsim_meas main.cpp)
#target_link_libraries (fengsim_meas ${PCL_LIBRARIES})
