/*! \file  fasp.h
 *
 *  \brief Main header file for the FASP project
 *
 *  \note  This header file contains general constants and data structures of
 *         FASP. It contains macros and data structure definitions; should not
 *         include function declarations here.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2008--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fasp_const.h"

#if WITH_MUMPS
#include "dmumps_c.h"
#endif

#if WITH_PARDISO
#include "mkl_pardiso.h"
#include "mkl_types.h"
#endif

#ifndef __FASP_HEADER__ /*-- allow multiple inclusions --*/
#define __FASP_HEADER__ /**< indicate fasp.h has been included before */

/*---------------------------*/
/*---  Macros definition  ---*/
/*---------------------------*/

/**
 * \brief FASP base version information
 */
#define FASP_VERSION 2.0 /**< faspsolver version */

#ifdef _OPENMP
#define MULTI_COLOR_ORDER                                                              \
    ON /**< Multicolor parallel GS smoothing method based on strongly connected        \
           matrix */
#else
#define MULTI_COLOR_ORDER                                                              \
    OFF /**< Multicolor parallel GS smoothing method based on strongly connected       \
           matrix */
#endif

/**
 * \brief For external software package support
 */
#define DLMALLOC  OFF /**< use dlmalloc instead of standard malloc */
#define NEDMALLOC OFF /**< use nedmalloc instead of standard malloc */

/**
 * \brief Flags for internal uses
 *
 * \warning Change the following marcos with caution!
 */
// When this flag is OFF, do not force C1 criterion for the classical AMG method
#define RS_C1 ON /**< CF splitting of RS: check C1 Criterion */
// When this flag is ON, the matrix rows will be reordered as diagonal entries first
#define DIAGONAL_PREF OFF /**< order each row such that diagonal appears first */

/**
 * \brief FASP integer and floating point numbers
 */
#define SHORT    short       /**< short integer type */
#define INT      int         /**< signed integer types: signed, long enough */
#define LONG     long        /**< long integer type */
#define LONGLONG long long   /**< long long integer type */
#define REAL     double      /**< float type */
#define LONGREAL long double /**< long double type */
#define STRLEN   256         /**< length of strings */

/**
 * \brief Definition of max, min, abs
 */
#define MAX(a, b) (((a) > (b)) ? (a) : (b))   /**< bigger one in a and b */
#define MIN(a, b) (((a) < (b)) ? (a) : (b))   /**< smaller one in a and b */
#define ABS(a)    (((a) >= 0.0) ? (a) : -(a)) /**< absolute value of a */

/**
 * \brief Definition of >, >=, <, <=, and isnan
 */
#define GT(a, b) (((a) > (b)) ? (TRUE) : (FALSE))  /**< is a > b? */
#define GE(a, b) (((a) >= (b)) ? (TRUE) : (FALSE)) /**< is a >= b? */
#define LS(a, b) (((a) < (b)) ? (TRUE) : (FALSE))  /**< is a < b? */
#define LE(a, b) (((a) <= (b)) ? (TRUE) : (FALSE)) /**< is a <= b? */
#define ISNAN(a) (((a) != (a)) ? (TRUE) : (FALSE)) /**< is a == NAN? */

/**
 * \brief Definition of print command in DEBUG mode
 */
#define PUT_INT(A)  printf("### DEBUG: %s = %d\n", #A, (A)) /**< print integer  */
#define PUT_REAL(A) printf("### DEBUG: %s = %e\n", #A, (A)) /**< print real num */

/*---------------------------*/
/*---  Matrix and vector  ---*/
/*---------------------------*/

/**
 * \struct ddenmat
 * \brief  Dense matrix of REAL type
 *
 * A dense REAL matrix
 */
typedef struct ddenmat {

    //! number of rows
    INT row;

    //! number of columns
    INT col;

    //! actual matrix entries
    REAL** val;

} ddenmat; /**< Dense matrix of REAL type */

/**
 * \struct idenmat
 * \brief  Dense matrix of INT type
 *
 * A dense INT matrix
 */
typedef struct idenmat {

    //! number of rows
    INT row;

    //! number of columns
    INT col;

    //! actual matrix entries
    INT** val;

} idenmat; /**< Dense matrix of INT type */

/**
 * \struct dCSRmat
 * \brief  Sparse matrix of REAL type in CSR format
 *
 * CSR Format (IA,JA,A) in REAL
 *
 * \note The starting index of A is 0.
 */
typedef struct dCSRmat {

    //! row number of matrix A, m
    INT row;

    //! column of matrix A, n
    INT col;

    //! number of nonzero entries
    INT nnz;

    //! integer array of row pointers, the size is m+1
    INT* IA;

    //! integer array of column indexes, the size is nnz
    INT* JA;

    //! nonzero entries of A
    REAL* val;

#if MULTI_COLOR_ORDER
    //! color numbers for the adjacency graph of A
    INT color;
    //! integer array of row pointers, the size is colors+1
    INT* IC;
    //!  inv map form multicolor order to original order of A.
    INT* ICMAP;
#endif

} dCSRmat; /**< Sparse matrix of REAL type in CSR format */

/**
 * \struct iCSRmat
 * \brief  Sparse matrix of INT type in CSR format
 *
 * CSR Format (IA,JA,A) in integer
 *
 * \note The starting index of A is 0.
 */
typedef struct iCSRmat {

    //! row number of matrix A, m
    INT row;

    //! column of matrix A, n
    INT col;

    //! number of nonzero entries
    INT nnz;

    //! integer array of row pointers, the size is m+1
    INT* IA;

    //! integer array of column indexes, the size is nnz
    INT* JA;

    //! nonzero entries of A
    INT* val;

} iCSRmat; /**< Sparse matrix of INT type in CSR format */

/**
 * \struct dCOOmat
 * \brief  Sparse matrix of REAL type in COO (IJ) format
 *
 * Coordinate Format (I,J,A)
 *
 * \note The starting index of A is 0.
 * \note Change I to rowind, J to colind. To avoid with complex.h confliction on I.
 */
typedef struct dCOOmat {

    //! row number of matrix A, m
    INT row;

    //! column of matrix A, n
    INT col;

    //! number of nonzero entries
    INT nnz;

    //! integer array of row indices, the size is nnz
    INT* rowind;

    //! integer array of column indices, the size is nnz
    INT* colind;

    //! nonzero entries of A
    REAL* val;

} dCOOmat; /**< Sparse matrix of REAL type in COO format */

/**
 * \struct iCOOmat
 * \brief  Sparse matrix of INT type in COO (IJ) format
 *
 * Coordinate Format (I,J,A)
 *
 * \note The starting index of A is 0.
 */
typedef struct iCOOmat {

    //! row number of matrix A, m
    INT row;

    //! column of matrix A, n
    INT col;

    //! number of nonzero entries
    INT nnz;

    //! integer array of row indices, the size is nnz
    INT* I;

    //! integer array of column indices, the size is nnz
    INT* J;

    //! nonzero entries of A
    INT* val;

} iCOOmat; /**< Sparse matrix of INT type in COO format */

/*!
 * \struct dCSRLmat
 * \brief  Sparse matrix of REAL type in CSRL format
 */
typedef struct dCSRLmat {

    //! number of rows
    INT row;

    //! number of cols
    INT col;

    //! number of nonzero entries
    INT nnz;

    //! number of different values in i-th row, i=0:nrows-1
    INT dif;

    //! nz_diff[i]: the i-th different value in 'nzrow'
    INT* nz_diff;

    //! row index of the matrix (length-grouped): rows with same nnz are together
    INT* index;

    //! j in {start[i],...,start[i+1]-1} means nz_diff[i] nnz in index[j]-row
    INT* start;

    //! column indices of all the nonzeros
    INT* ja;

    //! values of all the nonzero entries
    REAL* val;

} dCSRLmat; /**< Sparse matrix of REAL type in CSRL format */

/**
 * \struct dSTRmat
 * \brief  Structure matrix of REAL type
 *
 * \note Every nc^2 entries of the array diag and off-diag[i] store one block:
 *       For 2D matrix, the recommended offsets is [-1,1,-nx,nx];
 *       For 3D matrix, the recommended offsets is [-1,1,-nx,nx,-nxy,nxy].
 */
typedef struct dSTRmat {

    //! number of grids in x direction
    INT nx;

    //! number of grids in y direction
    INT ny;

    //! number of grids in z direction
    INT nz;

    //! number of grids on x-y plane
    INT nxy;

    //! size of each block (number of components)
    INT nc;

    //! number of grids
    INT ngrid;

    //! diagonal entries (length is ngrid*(nc^2))
    REAL* diag;

    //! number of off-diag bands
    INT nband;

    //! offsets of the off-diagonals (length is nband)
    INT* offsets;

    //! off-diagonal entries (dimension is nband * [(ngrid-|offsets|) * nc^2])
    REAL** offdiag;

} dSTRmat; /**< Structured matrix of REAL type */

/**
 * \struct dvector
 * \brief  Vector with n entries of REAL type
 */
typedef struct dvector {

    //! number of rows
    INT row;

    //! actual vector entries
    REAL* val;

} dvector; /**< Vector of REAL type */

/**
 * \struct ivector
 * \brief  Vector with n entries of INT type
 */
typedef struct ivector {

    //! number of rows
    INT row;

    //! actual vector entries
    INT* val;

} ivector; /**< Vector of INT type */

/*---------------------------*/
/*--- Parameter structures --*/
/*---------------------------*/

/**
 * \struct ITS_param
 * \brief  Parameters for iterative solvers
 */
typedef struct {

    SHORT print_level;   /**< print level: 0--10 */
    SHORT itsolver_type; /**< solver type: see fasp_const.h */
    SHORT decoup_type;   /**< decoupling type */
    SHORT precond_type;  /**< preconditioner type */
    SHORT stop_type;     /**< stopping type */
    INT   restart;       /**< number of steps for restarting: for GMRES etc */
    INT   maxit;         /**< max number of iterations */
    REAL  tol;           /**< convergence tolerance for relative residual */
    REAL  abstol;        /**< convergence tolerance for absolute residual */

} ITS_param;             /**< Parameters for iterative solvers */

/**
 * \struct ILU_param
 * \brief  Parameters for ILU
 */
typedef struct {

    //! print level
    SHORT print_level;

    //! ILU type for decomposition
    SHORT ILU_type;

    //! level of fill-in for ILUk
    INT ILU_lfil;

    //! drop tolerance for ILUt
    REAL ILU_droptol;

    //! add the sum of dropped elements to diagonal element in proportion relax
    REAL ILU_relax;

    //! permuted if permtol*|a(i,j)| > |a(i,i)|
    REAL ILU_permtol;

} ILU_param; /**< Parameters for ILU */

/**
 * \struct SWZ_param
 * \brief  Parameters for Schwarz method
 */
typedef struct {

    //! print leve
    SHORT print_level;

    //! type for Schwarz method
    SHORT SWZ_type;

    //! maximal level for constructing the blocks
    INT SWZ_maxlvl;

    //! maximal size of blocks
    INT SWZ_mmsize;

    //! type of Schwarz block solver
    INT SWZ_blksolver;

} SWZ_param; /**< Parameters for Schwarz method */

/**
 * \struct AMG_param
 * \brief  Parameters for AMG methods
 *
 * \note This is needed for the AMG solver/preconditioner.
 */
typedef struct {

    //! type of AMG method
    SHORT AMG_type;

    //! print level for AMG
    SHORT print_level;

    //! max number of iterations of AMG
    INT maxit;

    //! stopping tolerance for AMG solver
    REAL tol;

    //! max number of levels of AMG
    SHORT max_levels;

    //! max number of coarsest level DOF
    INT coarse_dof;

    //! type of AMG cycle
    SHORT cycle_type;

    //! quality threshold for pairwise aggregation
    REAL quality_bound;

    //! smoother type
    SHORT smoother;

    //! smoother order
    SHORT smooth_order; // 1: nature order 2: C/F order (both are symmetric)

    //! number of presmoothers
    SHORT presmooth_iter;

    //! number of postsmoothers
    SHORT postsmooth_iter;

    //! relaxation parameter for Jacobi and SOR smoother
    REAL relaxation;

    //! degree of the polynomial smoother
    SHORT polynomial_degree;

    //! coarse solver type
    SHORT coarse_solver;

    //! switch of scaling of the coarse grid correction
    SHORT coarse_scaling;

    //! degree of the polynomial used by AMLI cycle
    SHORT amli_degree;

    //! coefficients of the polynomial used by AMLI cycle
    REAL* amli_coef;

    //! type of Krylov method used by Nonlinear AMLI cycle
    SHORT nl_amli_krylov_type;

    //! coarsening type
    SHORT coarsening_type;

    //! aggregation type
    SHORT aggregation_type;

    //! aggregation norm type
    SHORT aggregation_norm_type;

    //! interpolation type
    SHORT interpolation_type;

    //! strong connection threshold for coarsening
    REAL strong_threshold;

    //! maximal row sum parameter
    REAL max_row_sum;

    //! truncation threshold
    REAL truncation_threshold;

    //! number of levels use aggressive coarsening
    INT aggressive_level;

    //! number of paths use to determine strongly coupled C points
    INT aggressive_path;

    //! number of pairwise matchings
    INT pair_number;

    //! strong coupled threshold for aggregate
    REAL strong_coupled;

    //! max size of each aggregate
    INT max_aggregation;

    //! relaxation parameter for smoothing the tentative prolongation
    REAL tentative_smooth;

    //! switch for filtered matrix used for smoothing the tentative prolongation
    SHORT smooth_filter;

    //! smooth the restriction for SA methods or not
    SHORT smooth_restriction;

    //! number of levels use ILU smoother
    SHORT ILU_levels;

    //! ILU type for smoothing
    SHORT ILU_type;

    //! level of fill-in for ILUs and ILUk
    INT ILU_lfil;

    //! drop tolerance for ILUt
    REAL ILU_droptol;

    //! relaxation for ILUs
    REAL ILU_relax;

    //! permuted if permtol*|a(i,j)| > |a(i,i)|
    REAL ILU_permtol;

    //! number of levels use Schwarz smoother
    INT SWZ_levels;

    //! maximal block size
    INT SWZ_mmsize;

    //! maximal levels
    INT SWZ_maxlvl;

    //! type of Schwarz method
    INT SWZ_type;

    //! type of Schwarz block solver
    INT SWZ_blksolver;

    //! theta for reduction-based amg
    REAL theta;

} AMG_param; /**< Parameters for AMG methods */

/*---------------------------*/
/*--- Work data structures --*/
/*---------------------------*/

/**
 * \struct Mumps_data
 * \brief  Data for MUMPS interface
 *
 * Added on 10/10/2014
 */
typedef struct {

#if WITH_MUMPS
    //! solver instance for MUMPS
    DMUMPS_STRUC_C id;
#endif

    //! work for MUMPS
    INT job;

} Mumps_data; /**< Data for MUMPS */

/**
 * \struct Pardiso_data
 * \brief  Data for Intel MKL PARDISO interface
 *
 * Added on 11/28/2015
 */
typedef struct {

    //! Internal solver memory pointer
    void* pt[64];

#if WITH_PARDISO
    //! Pardiso control parameters
    MKL_INT iparm[64];

    //! Type of the matrix
    MKL_INT mtype;

    //! Maximum number of numerical factorizations
    MKL_INT maxfct;

    //! Indicate the actual matrix for the solution phase, 1 <= mnum <= maxfct
    MKL_INT mnum;

#endif

} Pardiso_data; /**< Data for PARDISO */

/**
 * \struct ILU_data
 * \brief  Data for ILU setup
 */
typedef struct {

    //! pointer to the original coefficient matrix
    dCSRmat* A;

    //! type of ILUdata
    INT type;

    //! row number of matrix LU, m
    INT row;

    //! column of matrix LU, n
    INT col;

    //! number of nonzero entries
    INT nzlu;

    //! integer array of row pointers and column indexes, the size is nzlu
    INT* ijlu;

    //! nonzero entries of LU
    REAL* luval;

    //! block size for BSR type only
    INT nb;

    //! work space size
    INT nwork;

    //! work space
    REAL* work;

    //! permutation arrays for ILUtp
    INT* iperm;
    // iperm[0:n-1]   = old indices of unknowns
    // iperm[n:2*n-1] = reverse permutation = new indices.

    //! number of colors for multi-threading
    INT ncolors;

    //! indices for different colors
    INT* ic;

    //! mapping from vertex to color
    INT* icmap;

    //! temporary work space
    INT* uptr;

    //! number of colors for lower triangle
    INT nlevL;

    //! number of colors for upper triangle
    INT nlevU;

    //! number of vertices in each color for lower triangle
    INT* ilevL;

    //! number of vertices in each color for upper triangle
    INT* ilevU;

    //! mapping from row to color for lower triangle
    INT* jlevL;

    //! mapping from row to color for upper triangle
    INT* jlevU;

} ILU_data; /**< Data for ILU */

/**
 * \struct SWZ_data
 * \brief  Data for Schwarz methods
 *
 * This is needed for the Schwarz solver/preconditioner/smoother.
 */
typedef struct {

    /* matrix information */

    //! pointer to the original coefficient matrix
    dCSRmat A; // note: must start from 1!! Change later

    /* blocks information */

    //! number of blocks
    INT nblk;

    //! row index of blocks
    INT* iblock;

    //! column index of blocks
    INT* jblock;

    //! temp work space ???
    REAL* rhsloc;

    //! local right hand side
    dvector rhsloc1;

    //! local solution
    dvector xloc1;

    //! LU decomposition: the U block
    REAL* au;

    //! LU decomposition: the L block
    REAL* al;

    //! Schwarz method type
    INT SWZ_type;

    //! Schwarz block solver
    INT blk_solver;

    //! working space size
    INT memt;

    //! mask
    INT* mask;

    //! maximal block size
    INT maxbs;

    //! maxa
    INT* maxa;

    //! matrix for each partition
    dCSRmat* blk_data;

#if WITH_UMFPACK
    //! symbol factorize for UMFPACK
    void** numeric;
#endif

#if WITH_MUMPS
    //! instance for MUMPS
    DMUMPS_STRUC_C* id;
#endif

    //! param for MUMPS
    Mumps_data* mumps;

    //! param for Schwarz
    SWZ_param* swzparam;

} SWZ_data; /**< Data for Schwarz method */

/**
 * \struct AMG_data
 * \brief  Data for AMG methods
 *
 * \note This is needed for the AMG solver/preconditioner.
 */
typedef struct {

    /* Level information */

    //! max number of levels
    SHORT max_levels;

    //! number of levels in use <= max_levels
    SHORT num_levels;

    /* Problem information */

    //! pointer to the matrix at level level_num
    dCSRmat A;

    //! restriction operator at level level_num
    dCSRmat R;

    //! prolongation operator at level level_num
    dCSRmat P;

    //! pointer to the right-hand side at level level_num
    dvector b;

    //! pointer to the iterative solution at level level_num
    dvector x;

    /* Extra information */

    //! pointer to the numerical factorization from UMFPACK
    void* Numeric;

    //! data for Intel MKL PARDISO
    Pardiso_data pdata;

    //! pointer to the CF marker at level level_num
    ivector cfmark;

    //! number of levels use ILU smoother
    INT ILU_levels;

    //! ILU matrix for ILU smoother
    ILU_data LU;

    //! dimension of the near kernel for SAMG
    INT near_kernel_dim;

    //! basis of near kernel space for SAMG
    REAL** near_kernel_basis;

    // Smoother order information

    //! number of levels use Schwarz smoother
    INT SWZ_levels;

    //! data of Schwarz smoother
    SWZ_data Schwarz;

    //! temporary work space
    dvector w;

    //! data for MUMPS
    Mumps_data mumps;

    //! cycle type
    INT cycle_type;

    //! indices for different colors
    INT* ic;

    //! mapping from vertex to color
    INT* icmap;

    //! number of colors
    INT colors;

    //! weight for smoother
    REAL weight;

#if MULTI_COLOR_ORDER
    //! Gauss-Seidel Multicoloring factors. zhaoli,2021.08.25
    REAL GS_Theta;
#endif

} AMG_data; /**< Data for AMG methods */

/**
 * \struct precond_data
 * \brief  Data for preconditioners
 */
typedef struct {

    //! type of AMG method
    SHORT AMG_type;

    //! print level in AMG preconditioner
    SHORT print_level;

    //! max number of iterations of AMG preconditioner
    INT maxit;

    //! max number of AMG levels
    SHORT max_levels;

    //! tolerance for AMG preconditioner
    REAL tol;

    //! AMG cycle type
    SHORT cycle_type;

    //! AMG smoother type
    SHORT smoother;

    //! AMG smoother ordering
    SHORT smooth_order;

    //! number of presmoothing
    SHORT presmooth_iter;

    //! number of postsmoothing
    SHORT postsmooth_iter;

    //! relaxation parameter for SOR smoother
    REAL relaxation;

    //! degree of the polynomial smoother
    SHORT polynomial_degree;

    //! switch of scaling of the coarse grid correction
    SHORT coarsening_type;

    //! coarse solver type for AMG
    SHORT coarse_solver;

    //! switch of scaling of the coarse grid correction
    SHORT coarse_scaling;

    //! degree of the polynomial used by AMLI cycle
    SHORT amli_degree;

    //! type of Krylov method used by Nonlinear AMLI cycle
    SHORT nl_amli_krylov_type;

    //! smooth factor for smoothing the tentative prolongation
    REAL tentative_smooth;

    //! coefficients of the polynomial used by AMLI cycle
    REAL* amli_coef;

    //! AMG preconditioner data
    AMG_data* mgl_data;

    //! ILU preconditioner data (needed for CPR type preconditioner)
    ILU_data* LU;

    //! Matrix data
    dCSRmat* A;

    // extra near kernel space

    //! Matrix data for near kernel
    dCSRmat* A_nk;

    //! Prolongation for near kernel
    dCSRmat* P_nk;

    //! Restriction for near kernel
    dCSRmat* R_nk;

    // temporary work space

    //! temporary dvector used to store and restore the residual
    dvector r;

    //! temporary work space for other usage
    REAL* w;

} precond_data; /**< Data for preconditioners */

/**
 * \struct precond_data_str
 * \brief  Data for preconditioners in dSTRmat format
 */
typedef struct {

    //! type of AMG method
    SHORT AMG_type;

    //! print level in AMG preconditioner
    SHORT print_level;

    //! max number of iterations of AMG preconditioner
    INT maxit;

    //! max number of AMG levels
    SHORT max_levels;

    //! tolerance for AMG preconditioner
    REAL tol;

    //! AMG cycle type
    SHORT cycle_type;

    //! AMG smoother type
    SHORT smoother;

    //! number of presmoothing
    SHORT presmooth_iter;

    //! number of postsmoothing
    SHORT postsmooth_iter;

    //! coarsening type
    SHORT coarsening_type;

    //! relaxation parameter for SOR smoother
    REAL relaxation;

    //! switch of scaling of the coarse grid correction
    SHORT coarse_scaling;

    //! AMG preconditioner data
    AMG_data* mgl_data;

    //! ILU preconditioner data (needed for CPR type preconditioner)
    ILU_data* LU;

    //! whether the matrix are scaled or not
    SHORT scaled;

    //! the original CSR matrix
    dCSRmat* A;

    //! store the whole reservoir block in STR format
    dSTRmat* A_str;

    //! store Saturation block in STR format
    dSTRmat* SS_str;

    // data for GS/block GS smoothers (STR format)

    //! the inverse of the diagonals for GS/block GS smoother (whole reservoir matrix)
    dvector* diaginv;

    //! the pivot for the GS/block GS smoother (whole reservoir matrix)
    ivector* pivot;

    //! the inverse of the diagonals for GS/block GS smoother (saturation block)
    dvector* diaginvS;

    //! the pivot for the GS/block GS smoother (saturation block)
    ivector* pivotS;

    //! order for smoothing
    ivector* order;

    //! array to store neighbor information
    ivector* neigh;

    // temporary work space

    //! temporary dvector used to store and restore the residual
    dvector r;

    //! temporary work space for other usage
    REAL* w;

} precond_data_str; /**< Data for preconditioners of STR matrices */

/**
 * \struct precond_diag_str
 * \brief  Data for diagonal preconditioners in dSTRmat format
 *
 * \note This is needed for the diagonal preconditioner.
 */
typedef struct {

    //! number of components
    INT nc;

    //! diagonal elements
    dvector diag;

} precond_diag_str; /**< Data for diagonal preconditioners of STR matrices */

/**
 * \struct precond
 * \brief  Preconditioner data and action
 *
 * \note This is the preconditioner structure for preconditioned iterative methods.
 */
typedef struct {

    //! data for preconditioner, void pointer
    void* data;

    //! action for preconditioner, void function pointer
    void (*fct)(REAL*, REAL*, void*);

} precond; /**< Data for general preconditioner passed to iterative solvers */

/**
 * \struct mxv_matfree
 * \brief  Matrix-vector multiplication, replace the actual matrix
 */
typedef struct {

    //! data for MxV, can be a Matrix or something else
    void* data;

    //! action for MxV, void function pointer
    void (*fct)(const void*, const REAL*, REAL*);

} mxv_matfree; /**< Data for general matrix passed to iterative solvers */

/**
 * \struct input_param
 * \brief  Input parameters
 *
 * Input parameters, reading from disk file
 */
typedef struct {

    // output flags
    SHORT print_level; /**< print level */
    SHORT output_type; /**< type of output stream */

    // problem parameters
    char inifile[STRLEN]; /**< ini file name */
    char workdir[STRLEN]; /**< working directory for data files */
    INT  problem_num;     /**< problem number to solve */

    // parameters for iterative solvers
    SHORT solver_type;     /**< type of iterative solvers */
    SHORT decoup_type;     /**< type of decoupling method for PDE systems */
    SHORT precond_type;    /**< type of preconditioner for iterative solvers */
    SHORT stop_type;       /**< type of stopping criteria for iterative solvers */
    REAL  itsolver_tol;    /**< iterative tolerance for relative residual */
    REAL  itsolver_abstol; /**< iterative tolerance for absolute residaul */
    INT   itsolver_maxit;  /**< maximal number of iterations for iterative solvers */
    INT   restart;         /**< restart number used in GMRES */

    // parameters for ILU
    SHORT ILU_type;    /**< ILU type for decomposition*/
    INT   ILU_lfil;    /**< level of fill-in */
    REAL  ILU_droptol; /**< drop tolerance */
    REAL  ILU_relax;   /**< scaling factor: add the dropped entries to diagonal */
    REAL  ILU_permtol; /**< permutation tolerance */

    // parameter for Schwarz
    INT SWZ_mmsize;    /**< maximal block size */
    INT SWZ_maxlvl;    /**< maximal levels */
    INT SWZ_type;      /**< type of Schwarz method */
    INT SWZ_blksolver; /**< type of Schwarz block solver */

    // parameters for AMG
    SHORT AMG_type;                /**< Type of AMG */
    SHORT AMG_levels;              /**< maximal number of levels */
    SHORT AMG_cycle_type;          /**< type of cycle */
    SHORT AMG_smoother;            /**< type of smoother */
    SHORT AMG_smooth_order;        /**< order for smoothers */
    REAL  AMG_relaxation;          /**< over-relaxation parameter for SOR */
    SHORT AMG_polynomial_degree;   /**< degree of the polynomial smoother */
    SHORT AMG_presmooth_iter;      /**< number of presmoothing */
    SHORT AMG_postsmooth_iter;     /**< number of postsmoothing */
    REAL  AMG_tol;                 /**< tolerance for AMG as preconditioner */
    INT   AMG_coarse_dof;          /**< max number of coarsest level DOF */
    INT   AMG_maxit;               /**< number of AMG iterations as preconditioner */
    SHORT AMG_ILU_levels;          /**< how many levels use ILU smoother */
    SHORT AMG_coarse_solver;       /**< coarse solver type */
    SHORT AMG_coarse_scaling;      /**< switch of scaling of coarse grid correction */
    SHORT AMG_amli_degree;         /**< degree of the polynomial in AMLI cycle */
    SHORT AMG_nl_amli_krylov_type; /**< type of Krylov method in nonlinear AMLI cycle */
    INT   AMG_SWZ_levels;          /**< number of levels use Schwarz smoother */

    // parameters for classical AMG
    SHORT AMG_coarsening_type;      /**< coarsening type */
    SHORT AMG_aggregation_type;     /**< aggregation type */
    SHORT AMG_interpolation_type;   /**< interpolation type */
    REAL  AMG_strong_threshold;     /**< strong threshold for coarsening */
    REAL  AMG_truncation_threshold; /**< truncation factor for interpolation */
    REAL  AMG_max_row_sum;          /**< maximal row sum */
    INT   AMG_aggressive_level;     /**< number of levels use aggressive coarsening */
    INT   AMG_aggressive_path;      /**< number of paths for strongly coupled C-set */
    INT   AMG_pair_number;          /**< number of pairs in matching algorithm */
    REAL  AMG_quality_bound;        /**< threshold for pair wise aggregation */

    //  parameters for smoothed aggregation AMG
    REAL  AMG_strong_coupled;     /**< strong coupled threshold for aggregate */
    INT   AMG_max_aggregation;    /**< max size of each aggregate */
    REAL  AMG_tentative_smooth;   /**< relax factor for smoothing tentative prolong. */
    SHORT AMG_smooth_filter;      /**< filter for smoothing the tentative prolong. */
    SHORT AMG_smooth_restriction; /**< smoothing the restriction or not */
    SHORT AMG_aggregation_norm_type; /**< aggregation norm type for BSR */

} input_param;                       /**< Input parameters */

/*
 * OpenMP definitions and declarations
 */
#ifdef _OPENMP

#include "omp.h"

#define ILU_MC_OMP OFF /**< Use multicolor smoothing or not, default is not */

// extern INT omp_count;   /**< Counter for multiple calls: Remove later!!! --Chensong
// */

extern INT THDs_AMG_GS;        /**< AMG GS smoothing threads  */
extern INT THDs_CPR_lGS;       /**< Reservoir GS smoothing threads */
extern INT THDs_CPR_gGS;       /**< Global matrix GS smoothing threads  */
#ifdef DETAILTIME
extern REAL total_linear_time; /**< Total used time of linear solvers */
extern REAL total_start_time;  /**< Total used time */
extern REAL total_setup_time;  /**< Total setup time */
extern INT  total_iter;        /**< Total number of iterations */
extern INT  fasp_called_times; /**< Total FASP calls */
#endif

#endif /* end if for _OPENMP */

extern double PreSmoother_time_zl;
extern double PostSmoother_time_zl;
extern double Krylov_time_zl;
extern double Coarsen_time_zl;
extern double AMLI_cycle_time_zl;

#endif /* end if for __FASP_HEADER__ */

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
