/*! \file poisson_fdm.h
 *  \brief Main header file for the 2D/3d Finite Difference Method
 */

#ifndef _FSLS_HEADER_
#define _FSLS_HEADER_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define mytime(a,b) (double)(b - a)/(double)(CLOCKS_PER_SEC)

#define PI 3.1415926535897932
#define fsls_max(a,b)  (((a)<(b)) ? (b) : (a))
#define fsls_min(a,b)  (((a)<(b)) ? (a) : (b))
#define fsls_TFree(ptr) ( fsls_Free((char *)ptr), ptr = NULL )
#define fsls_CTAlloc(type, count) ( (type *)fsls_CAlloc((size_t)(count), (size_t)sizeof(type)) )

void  fsls_Free( char *ptr );
char *fsls_CAlloc( size_t count, size_t elt_size );
int   fsls_OutOfMemory( size_t size );

typedef struct
{
    double  *data;
    int     *i;
    int     *j;
    int      num_rows;
    int      num_cols;
    int      num_nonzeros;
    int     *rownnz;
    int      num_rownnz; 
    int      owns_data;
} fsls_CSRMatrix;

#define fsls_CSRMatrixData(matrix)         ((matrix) -> data)
#define fsls_CSRMatrixI(matrix)            ((matrix) -> i)
#define fsls_CSRMatrixJ(matrix)            ((matrix) -> j)
#define fsls_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define fsls_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define fsls_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define fsls_CSRMatrixRownnz(matrix)       ((matrix) -> rownnz)
#define fsls_CSRMatrixNumRownnz(matrix)    ((matrix) -> num_rownnz)
#define fsls_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)

typedef struct
{
    double  *data;
    int      size;
    int      owns_data;
    int      num_vectors;
    int      multivec_storage_method;
    int      vecstride, idxstride;

} fsls_Vector;

#define fsls_VectorData(vector)                  ((vector) -> data)
#define fsls_VectorSize(vector)                  ((vector) -> size)
#define fsls_VectorOwnsData(vector)              ((vector) -> owns_data)
#define fsls_VectorNumVectors(vector)            ((vector) -> num_vectors)
#define fsls_VectorMultiVecStorageMethod(vector) ((vector) -> multivec_storage_method)
#define fsls_VectorVectorStride(vector)          ((vector) -> vecstride )
#define fsls_VectorIndexStride(vector)           ((vector) -> idxstride )

typedef struct
{
    int      n;    /**< @brief order of the matrix */
    int      nx; /**< @brief number of nodes along x-direction(excluding boundary nodes) */
    int      ny; /**< @brief number of nodes along y-direction(excluding boundary nodes) */
    int      nz; /**< @brief number of nodes along z-direction(excluding boundary nodes) */
    int      nband; /**< @brief the number of offdiagonal bands */

    /**
     * @brief offsets of the offdiagonal bands (length is nband),
     *
     * offsets are ordered in the ascendling manner, the negative and positive values
     * corresband to lower left bands and upper right bands, respectively
     */
    int     *offsets;
    double  *diag; /**< @brief diagonal entries (length is n) */
    /**
     * @brief off-diagonal entries (dimension is nband X n),
     *
     * offdiag[i][j],i=0(1)nband-1,j=0(1)n-1: the j-th entry on the i-th offdiagonal band.
     */
    double **offdiag;
    double  *data_ext; /**< @brief data part, including diag_ext and offdiag_ext */

} fsls_BandMatrix;

#define fsls_BandMatrixN(matrix)         ((matrix) -> n)
#define fsls_BandMatrixNx(matrix)        ((matrix) -> nx)
#define fsls_BandMatrixNy(matrix)        ((matrix) -> ny)
#define fsls_BandMatrixNz(matrix)        ((matrix) -> nz)
#define fsls_BandMatrixNband(matrix)     ((matrix) -> nband)
#define fsls_BandMatrixOffsets(matrix)   ((matrix) -> offsets)
#define fsls_BandMatrixDiag(matrix)      ((matrix) -> diag)
#define fsls_BandMatrixOffdiag(matrix)   ((matrix) -> offdiag)
#define fsls_BandMatrixDataExt(matrix)   ((matrix) -> data_ext)

typedef struct
{
    int      size;     /**< @brief length of the vector  */
    double  *data;     /**< @brief data of the vector (length is size) */
    double  *data_ext; /**< @brief data part, including extended data */

} fsls_XVector;

#define fsls_XVectorSize(vector)     ((vector) -> size)
#define fsls_XVectorData(vector)     ((vector) -> data)
#define fsls_XVectorDataExt(vector)  ((vector) -> data_ext)

int fsls_BandMatrixPrint( fsls_BandMatrix *A, char *file_name );
fsls_BandMatrix *fsls_BandMatrixRead( char *file_name );
void fsls_BandMatrixDestroy( fsls_BandMatrix *matrix );
fsls_BandMatrix *fsls_BandMatrixCreate( int n, int nband );
void fsls_BandMatrixInitialize( fsls_BandMatrix *matrix );
void fsls_TriBand2FullMatrix( fsls_BandMatrix *A, double **full_ptr );
int fsls_Band2FullMatrix( fsls_BandMatrix *A, double **full_ptr );
int fsls_CheckDiagOdd( fsls_BandMatrix *matrix );
int fsls_XVectorPrint( fsls_XVector *vector, char *file_name );
fsls_XVector *fsls_XVectorRead( char *file_name );
fsls_XVector *fsls_XVectorCreate( int size );
int fsls_XVectorInitialize( fsls_XVector *vector );
int fsls_XVectorCopy( fsls_XVector *x, fsls_XVector *y );
int fsls_XVectorSetConstantValues( fsls_XVector *vector, double value );
int fsls_XVectorDestroy( fsls_XVector *vector );

void 
fsls_BuildLinearSystem_5pt2d( int               nt,
                              int               nx,
                              int               ny,
                              fsls_BandMatrix **A_ptr, 
                              fsls_XVector    **f_ptr,
                              fsls_XVector    **u_ptr );
void 
fsls_BuildLinearSystem_5pt2d_rb( int               nt,
                                 int               nx,
                                 int               ny,
                                 fsls_BandMatrix **A_ptr, 
                                 fsls_XVector    **f_ptr,
                                 fsls_XVector    **u_ptr );
void 
fsls_BuildLinearSystem_7pt3d( int               nt,
                              int               nx,
                              int               ny,
                              int               nz,
                              fsls_BandMatrix **A_ptr, 
                              fsls_XVector    **f_ptr,
                              fsls_XVector    **u_ptr );



int fsls_Band2CSRMatrix( fsls_BandMatrix *B, fsls_CSRMatrix **A_ptr );
int fsls_CSRMatrixPrint( fsls_CSRMatrix *matrix, char *file_name );

/**
 * @brief output the matrix with coo format
 *
 *  newly added 2012/01/08 by feiteng
 */
int fsls_COOMatrixPrint( fsls_CSRMatrix *matrix, char *file_name );

/**
 * @brief output the sparsity pattern of the matrix with gnuplot format
 *
 *  newly added 2012/01/08 by feiteng
 */
int fsls_MatrixSPGnuplot( fsls_CSRMatrix *matrix, char *file_name );

fsls_CSRMatrix *fsls_CSRMatrixCreate( int num_rows,int num_cols,int num_nonzeros );
int fsls_CSRMatrixInitialize( fsls_CSRMatrix *matrix );
int fsls_CSRMatrixDestroy( fsls_CSRMatrix *matrix );
fsls_CSRMatrix *fsls_CSRMatrixDeleteZeros( fsls_CSRMatrix *A, double tol );
int fsls_WriteSAMGData( fsls_CSRMatrix *A, fsls_XVector *b, fsls_XVector *u ); // newly added 2010/08/23

//newly added 2011/12/11 by feiteng
/**
 * @brief csr matrix to full matrix,
 *
 * lapack routine need full matrix, newly added 2011/12/11 by feiteng
 */
int fsls_CSR2FullMatrix( fsls_CSRMatrix *A, double **full_ptr);
/**
 * @brief matrix of (-lap)_h to (I - dt*lap)_h
 *
 * the matrix of time-dependent poisson equation would not change at each time step, newly added 2011/12/11 by feiteng
 */
int fsls_dtMatrix(double dt, int n_rows, int n_cols, double *A_full);

/**
 * @brief lapack routine, need liblapack.so,
 *
 * newly added 2011/12/11 by feiteng
 */
extern void dgetrf_(int*, int*, double*, int*, int*, int*);
extern void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
#endif
