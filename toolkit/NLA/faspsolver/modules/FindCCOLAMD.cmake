# - Try to find CCOLAMD
# 
#  OUTPUT:
#  CCOLAMD_FOUND        - system has CCOLAMD
#  CCOLAMD_INCLUDE_DIRS - include directories for CCOLAMD
#  CCOLAMD_LIBRARIES    - libraries for CCOLAMD
#
#  Xiaozhe Hu
#  02/27/2013
#  Modified   2015-08-08   --ltz

message(STATUS "Checking for package 'CCOLAMD'")

# Check for header file
find_path(CCOLAMD_INCLUDE_DIRS ccolamd.h
 HINTS ${SUITESPARSE_DIR}/include ${SUITESPARSE_DIR}/CCOLAMD/include $ENV{SUITESPARSE_DIR}/include $ENV{SUITESPARSE_DIR}/CCOLAMD/include
 PATH_SUFFIXES suitesparse ufsparse
 DOC "Directory where the CCOLAMD header is located"
 )
mark_as_advanced(CCOLAMD_INCLUDE_DIRS)

# Check for CCOLAMD library
find_library(CCOLAMD_LIBRARIES ccolamd
  HINTS ${SUITESPARSE_DIR}/lib ${SUITESPARSE_DIR}/CCOLAMD/lib $ENV{SUITESPARSE_DIR}/lib $ENV{SUITESPARSE_DIR}/CCOLAMD/lib
  DOC "The CCOLAMD library"
  )
mark_as_advanced(CCOLAMD_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CCOLAMD
  "CCOLAMD could not be found. Be sure to set SUITESPARSE_DIR correctly."
  CCOLAMD_LIBRARIES CCOLAMD_INCLUDE_DIRS)
