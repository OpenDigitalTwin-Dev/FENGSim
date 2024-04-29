# - Try to find SuperLU
#  
#  OUTPUT:
#  SUPERLU_FOUND        - system has SuperLU
#  SUPERLU_INCLUDE_DIRS - include directories for SuperLU
#  SUPERLU_LIBRARIES    - libraries for SuperLU
#
#  Xiaozhe Hu
#  02/24/2014

message(STATUS "Checking for dependent packages of 'SuperLU'")

# Find packages that SuperLU depends on
find_package(BLAS)

message(STATUS "Checking for package 'SuperLU'")

# Check for header file
find_path(SUPERLU_INCLUDE_DIRS slu_ddefs.h
 HINTS ${SUPERLU_DIR}/include ${SUPERLU_DIR}/SRC ${SUPERLU_DIR}/SuperLU/SRC $ENV{SUPERLU_DIR}/SRC $ENV{SUPERLU_DIR}/SuperLU/SRC
 PATH_SUFFIXES  SuperLU
 DOC "Directory where the SuperLU header is located"
 )
mark_as_advanced(SUPERLU_INCLUDE_DIRS)

# Check for SuperLU library
find_library(SUPERLU_LIBRARIES superlu superlu_4.3 superlu_4.2 superlu_4.1 superlu_4.0 
  HINTS ${SUPERLU_DIR}/lib ${SUPERLU_DIR}/SuperLU/lib $ENV{SUPERLU_DIR}/lib $ENV{SUPERLU_DIR}/SuperLU/lib
  DOC "The SuperLU library"
  )
mark_as_advanced(SUPERLU_LIBRARIES)

# Collect libraries
if (BLAS_FOUND)
  set(SUPERLU_LIBRARIES ${SUPERLU_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUPERLU
  "SuperLU could not be found. Be sure to set SUPERLU_DIR."
  SUPERLU_LIBRARIES SUPERLU_INCLUDE_DIRS)
