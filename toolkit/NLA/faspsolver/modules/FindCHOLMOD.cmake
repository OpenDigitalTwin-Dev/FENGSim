# - Try to find CHOLMOD
# 
#  OUTPUT:
#  CHOLMOD_FOUND        - system has CHOLMOD
#  CHOLMOD_INCLUDE_DIRS - include directories for CHOLMOD
#  CHOLMOD_LIBRARIES    - libraries for CHOLMOD
#
#  Xiaozhe Hu
#  02/27/2013
#  Modified   2015-08-08   --ltz

# Find packages that CHOLMOD depends on 
find_package(BLAS)
find_package(LAPACK)
find_package(SUITESPARSECONFIG)
find_package(AMD)
find_package(COLAMD)
find_package(CAMD)
find_package(CCOLAMD)
if(NOT APPLE) 
 find_package(RT)
endif(NOT APPLE)
find_package(METIS)

message(STATUS "Checking for package 'CHOLMOD'")

# Check for header file
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
 HINTS ${SUITESPARSE_DIR}/include ${SUITESPARSE_DIR}/CHOLMOD/include $ENV{SUITESPARSE_DIR}/include $ENV{SUITESPARSE_DIR}/CHOLMOD/include
 PATH_SUFFIXES suitesparse ufsparse
 DOC "Directory where the CHOLMOD header is located"
 )
mark_as_advanced(CHOLMOD_INCLUDE_DIRS)

# Check for CHOLMOD library
find_library(CHOLMOD_LIBRARY cholmod
  HINTS ${SUITESPARSE_DIR}/lib ${SUITESPARSE_DIR}/CHOLMOD/lib $ENV{SUITESPARSE_DIR}/lib $ENV{SUITESPARSE_DIR}/CHOLMOD/lib 
  DOC "The CHOLMOD library"
  )
mark_as_advanced(CHOLMOD_LIBRARY)

# Collect libraries
if (AMD_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} ${AMD_LIBRARIES})
endif()
if (BLAS_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES})
endif()
if (LAPACK_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
if (SUITESPARSECONFIG_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${SUITESPARSECONFIG_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${SUITESPARSECONFIG_LIBRARIES})
endif()
if (COLAMD_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${COLAMD_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${COLAMD_LIBRARIES})
endif()
if (CAMD_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${CAMD_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${CAMD_LIBRARIES})
endif()
if (CCOLAMD_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${CCOLAMD_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${CCOLAMD_LIBRARIES})
endif()
if (RT_FOUND AND (NOT APPLE))
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${RT_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${RT_LIBRARIES})
endif()
if (METIS_FOUND)
  set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS} ${METIS_INCLUDE_DIRS})
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${METIS_LIBRARIES})
endif()

set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} "-lgfortran")

# Try to run a test program that uses CHOLMOD
if (CHOLMOD_INCLUDE_DIRS AND CHOLMOD_LIBRARIES)
  set(CMAKE_REQUIRED_INCLUDES  ${CHOLMOD_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CHOLMOD_LIBRARIES})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#include <stdio.h>
#include <cholmod.h>

int main()
{
  cholmod_dense *D;
  cholmod_sparse *S;
  cholmod_dense *x, *b, *r;
  cholmod_factor *L;
  double one[2] = {1,0}, m1[2] = {-1,0};
  double *dx;
  cholmod_common c;
  int n = 5;
  double K[5][5] = {{1.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 2.0,-1.0, 0.0, 0.0},
                    {0.0,-1.0, 2.0,-1.0, 0.0},
                    {0.0, 0.0,-1.0, 2.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 1.0}};
  cholmod_start (&c);
  D = cholmod_allocate_dense(n, n, n, CHOLMOD_REAL, &c);
  dx = (double*)D->x;
  for (int i=0; i < n; i++)
    for (int j=0; j < n; j++)
      dx[i+j*n] = K[i][j];
  S = cholmod_dense_to_sparse(D, 1, &c);
  S->stype = 1;
  cholmod_reallocate_sparse(cholmod_nnz(S, &c), S, &c);
  b = cholmod_ones(S->nrow, 1, S->xtype, &c);
  L = cholmod_analyze(S, &c);
  cholmod_factorize(S, L, &c);
  x = cholmod_solve(CHOLMOD_A, L, b, &c);
  r = cholmod_copy_dense(b, &c);
  cholmod_sdmult(S, 0, m1, one, x, r, &c);
  cholmod_free_factor(&L, &c);
  cholmod_free_dense(&D, &c);
  cholmod_free_sparse(&S, &c);
  cholmod_free_dense(&r, &c);
  cholmod_free_dense(&x, &c);
  cholmod_free_dense(&b, &c);
  cholmod_finish(&c);
  return 0;
}
" CHOLMOD_TEST_RUNS)

endif()


# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
  "CHOLMOD could not be found. Be sure to set SUITESPARSE_DIR correctly."
 CHOLMOD_LIBRARY CHOLMOD_INCLUDE_DIRS CHOLMOD_TEST_RUNS)
