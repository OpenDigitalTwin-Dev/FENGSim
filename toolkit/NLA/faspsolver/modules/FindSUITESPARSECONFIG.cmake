# - Try to find SUITEPARSE_CONFIG
#  
#  OUTPUT:
#  SUITESPARSECONFIG_FOUND        - system has SUITEPARSE_CONFIG
#  SUITESPARSECONFIG_INCLUDE_DIRS - include directories for SUITEPARSE_CONFIG
#  SUITESPARSECONFIG_LIBRARIES    - libraries for SUITEPARSE_CONFIG
#
#  Xiaozhe Hu
#  02/27/2013
#  Modified   2015-08-08   --ltz

message(STATUS "Checking for package 'SUITESPARSECONFIG'")

# Check for header file
find_path(SUITESPARSECONFIG_INCLUDE_DIRS SuiteSparse_config.h UFconfig.h
  HINTS ${SUITESPARSE_DIR} ${SUITESPARSE_DIR}/include ${SUITESPARSE_DIR}/SuiteSparse_config
        ${SUITESPARSE_DIR}/SuiteSparse_config/include $ENV{SUITESPARSE_DIR} $ENV{SUITESPARSE_DIR}/include
	$ENV{SUITESPARSE_DIR}/SuiteSparse_config $ENV{SUITESPARSE_DIR}/SuiteSparse_config/include 
  PATH_SUFFIXES suitesparse SuiteSparse ufsparse
  DOC "Directory where the SUITESPARSECONFIG header is located"
  )
mark_as_advanced(SUITESPARSECONFIG_INCLUDE_DIRS)

# Check for SUITESPARSE_CONFIG library
find_library(SUITESPARSECONFIG_LIBRARIES suitesparseconfig ufconfig 
  HINTS ${SUITESPARSE_DIR} ${SUITESPARSE_DIR}/lib ${SUITESPARSE_DIR}/SuiteSparse_config
        ${SUITESPARSE_DIR}/SuiteSparse_config/lib $ENV{SUITESPARSE_DIR} $ENV{SUITESPARSE_DIR}/lib
	$ENV{SUITESPARSE_DIR}/SuiteSparse_config $ENV{SUITESPARSE_DIR}/SuiteSparse_config/lib
  DOC "The SUITESPARSE_CONFIG library"
  )
mark_as_advanced(SUITESPARSECONFIG_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUITESPARSECONFIG
  "SUITESPARSECONFIG could not be found. Be sure to set SUITESPARSE_DIR correctly."
  SUITESPARSECONFIG_LIBRARIES SUITESPARSECONFIG_INCLUDE_DIRS)
