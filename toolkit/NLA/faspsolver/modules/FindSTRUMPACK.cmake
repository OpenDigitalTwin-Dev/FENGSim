# Defines the following variables:
#   - STRUMPACK_FOUND
#   - STRUMPACK_LIBRARIES
#   - STRUMPACK_INCLUDE_DIRS

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(METIS REQUIRED)
#find_package(ParMETIS REQUIRED)
#find_package(Scotch REQUIRED)
#find_package(ScaLAPACK REQUIRED)

# Find the inlcude directory
find_path(STRUMPACK_INCLUDE_DIRS StrumpackSparseSolver.h
  HINTS ${STRUMPACK_DIR} $ENV{STRUMPACK_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with STRUMPACK header.")
#find_path(STRUMPACK_INCLUDE_DIRS StrumpackSparseSolver.h)
mark_as_advanced(STRUMPACK_INCLUDE_DIRS)

find_library(STRUMPACK_LIBRARY strumpack
  HINTS ${STRUMPACK_DIR} $ENV{STRUMPACK_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The STRUMPACK library.")
#find_library(STRUMPACK_LIBRARY strumpack)
mark_as_advanced(STRUMPACK_LIBRARY)

if (NOT TARGET STRUMPACK::strumpack)
  add_library(STRUMPACK::strumpack STATIC IMPORTED)
endif (NOT TARGET STRUMPACK::strumpack)

# Set the location
set_property(TARGET STRUMPACK::strumpack
  PROPERTY IMPORTED_LOCATION ${STRUMPACK_LIBRARY})

# Set the includes
set_property(TARGET STRUMPACK::strumpack APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${STRUMPACK_INCLUDE_DIRS})

# Manage dependencies
set_property(TARGET STRUMPACK::strumpack APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES
  ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
  ${SCOTCH_LIBRARIES} ${ParMETIS_LIBRARIES} ${METIS_LIBRARIES}
  ${ScaLAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${STRUMPACK_EXTRA_LINKAGE})

# Set the include directories
set(STRUMPACK_INCLUDE_DIRS ${STRUMPACK_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for STRUMPACK.")
mark_as_advanced(FORCE STRUMPACK_INCLUDE_DIRS)

# Set the libraries
set(STRUMPACK_LIBRARIES STRUMPACK::strumpack)
mark_as_advanced(FORCE STRUMPACK_LIBRARY)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STRUMPACK
  "STRUMPACK could not be found. Be sure to set STRUMPACK_DIR."
  STRUMPACK_LIBRARIES STRUMPACK_INCLUDE_DIRS)
