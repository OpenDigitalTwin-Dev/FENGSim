# - Try to find MUMPS
#
# Once done this will define
#  MUMPS_FOUND - System has Mumps
#  MUMPS_INCLUDE_DIRS - The Mumps include directories
#  MUMPS_LIBRARY_DIRS - The library directories needed to use Mumps
#  MUMPS_LIBRARIES    - The libraries needed to use Mumps
#
#  Chensong Zhang
#  10/06/2014

message(STATUS "Checking for dependent packages of 'MUMPS'")

find_package(BLAS)
find_package(Threads REQUIRED)

if (MUMPS_INCLUDE_DIR)
  # in cache already
  SET(MUMPS_FIND_QUIETLY TRUE)
endif (MUMPS_INCLUDE_DIR)

# Check for header file
# find_path(MUMPS_INCLUDE_DIR dmumps_c.h mpi.h
# HINTS ${MUMPS_DIR}/include ${MUMPS_DIR}/libseq $ENV{MUMPS_DIR}/include $ENV{IPOPT_HOME}/MUMPS/include/
find_path(MUMPS_INCLUDE_DIR dmumps_c.h
 HINTS ${MUMPS_DIR}/include $ENV{MUMPS_DIR}/include $ENV{IPOPT_HOME}/MUMPS/include/
 DOC "Directory where the MUMPS header is located"
 )
mark_as_advanced(MUMPS_INCLUDE_DIR)

# Check for MUMPS libraries: dmumps, mumps_common pord mpiseq
find_library(MUMPS_LIB dmumps
  HINTS ${MUMPS_DIR}/lib $ENV{MUMPS_DIR}/lib $ENV{IPOPT_HOME}/MUMPS/lib/
  DOC "The MUMPS library"
  )

find_library(MUMPS_LIBC mumps_common
  HINTS ${MUMPS_DIR}/lib $ENV{MUMPS_DIR}/lib $ENV{IPOPT_HOME}/MUMPS/lib/
  DOC "The MUMPS Common library"
  )

find_library(MUMPS_LIB_PORD pord
  HINTS ${MUMPS_DIR}/lib $ENV{MUMPS_DIR}/lib ${MUMPS_DIR}/PORD/lib $ENV{MUMPS_DIR}/PORD/lib
  DOC "The MUMPS PORD library"
  )

find_library(MUMPS_LIB_MPISEQ mpiseq
  HINTS ${MUMPS_DIR}/lib $ENV{MUMPS_DIR}/lib ${MUMPS_DIR}/libseq $ENV{MUMPS_DIR}/libseq
  DOC "The MUMPS MPISEQ library"
  )

set(MUMPS_INCLUDE_DIRS "${MUMPS_INCLUDE_DIR}" )
set(MUMPS_LIBRARIES "${MUMPS_LIB}" "${MUMPS_LIBC}" "${MUMPS_LIB_PORD}" "${MUMPS_LIB_MPISEQ}")

if (Threads_FOUND)
  set(MUMPS_LIBRARIES ${MUMPS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

if (BLAS_FOUND)
  set(MUMPS_LIBRARIES ${MUMPS_LIBRARIES} ${BLAS_LIBRARIES})
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MUMPS  DEFAULT_MSG
                                  MUMPS_LIB MUMPS_INCLUDE_DIR)

mark_as_advanced(MUMPS_INCLUDE_DIR MUMPS_LIB MUMPS_LIBC MUMPS_LIB_PORD MUMPS_LIB_MPISEQ)
