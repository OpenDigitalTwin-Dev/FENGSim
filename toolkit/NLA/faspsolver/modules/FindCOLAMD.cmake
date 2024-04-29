# - Try to find COLAMD
# 
#  OUTPUT:
#  COLAMD_FOUND        - system has COLAMD
#  COLAMD_INCLUDE_DIRS - include directories for COLAMD
#  COLAMD_LIBRARIES    - libraries for COLAMD
#
#  Xiaozhe Hu
#  02/27/2013
#  Modified   2015-08-08   --ltz

message(STATUS "Checking for package 'COLAMD'")

# Check for header file
find_path(COLAMD_INCLUDE_DIRS colamd.h
 HINTS ${SUITESPARSE_DIR}/include ${SUITESPARSE_DIR}/COLAMD/include $ENV{SUITESPARSE_DIR}/include $ENV{SUITESPARSE_DIR}/COLAMD/include
 PATH_SUFFIXES suitesparse ufsparse
 DOC "Directory where the COLAMD header is located"
 )
mark_as_advanced(COLAMD_INCLUDE_DIRS)

# Check for COLAMD library
find_library(COLAMD_LIBRARIES colamd
  HINTS ${SUITESPARSE_DIR}/lib ${SUITESPARSE_DIR}/COLAMD/lib $ENV{SUITESPARSE_DIR}/lib $ENV{SUITESPARSE_DIR}/COLAMD/lib
  DOC "The COLAMD library"
  )
mark_as_advanced(COLAMD_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(COLAMD
  "COLAMD could not be found. Be sure to set SUITESPARSE_DIR correctly."
  COLAMD_LIBRARIES COLAMD_INCLUDE_DIRS)
