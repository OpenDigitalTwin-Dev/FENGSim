/*! \file  fasp_block.h
 *
 *  \brief Header file for FASP block matrices
 *
 *  \note  This header file contains definitions of block matrices, including
 *         grid-major type and variable-major type. In this header, we only
 *         define macros and data structures, not function declarations.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"

#ifndef __FASPBLOCK_HEADER__ /*-- allow multiple inclusions --*/
#define __FASPBLOCK_HEADER__ /**< indicate fasp_block.h has been included before */

/*---------------------------*/
/*---   Data structures   ---*/
/*---------------------------*/

/**
 * \struct dBSRmat
 * \brief Block sparse row storage matrix of REAL type
 *
 * \note This data structure is adapted from the Intel MKL library. Refer to:
 * http://software.intel.com/sites/products/documentation/hpc/mkl/lin/index.htm
 *
 * \note Some of the following entries are capitalized to stress that they are
 *       for blocks!
 */
typedef struct dBSRmat {

    //! number of rows of sub-blocks in matrix A, M
    INT ROW;

    //! number of cols of sub-blocks in matrix A, N
    INT COL;

    //! number of nonzero sub-blocks in matrix A, NNZ
    INT NNZ;

    //! dimension of each sub-block
    INT nb; // NOTE: for the moment, allow nb*nb full block

    //! storage manner for each sub-block
    INT storage_manner; // 0: row-major order, 1: column-major order

    //! A real array that contains the elements of the non-zero blocks of
    //! a sparse matrix. The elements are stored block-by-block in row major
    //! order. A non-zero block is the block that contains at least one non-zero
    //! element. All elements of non-zero blocks are stored, even if some of
    //! them is equal to zero. Within each nonzero block elements are stored
    //! in row-major order and the size is (NNZ*nb*nb).
    REAL* val;

    //! integer array of row pointers, the size is ROW+1
    INT* IA;

    //! Element i of the integer array columns is the number of the column in the
    //! block matrix that contains the i-th non-zero block. The size is NNZ.
    INT* JA;

} dBSRmat; /**< Matrix of REAL type in BSR format */

/**
 * \struct dBLCmat
 * \brief Block REAL CSR matrix format
 *
 * \note The starting index of A is 0.
 */
typedef struct dBLCmat {

    //! row number of blocks in A, m
    INT brow;

    //! column number of blocks A, n
    INT bcol;

    //! blocks of dCSRmat, point to blocks[brow][bcol]
    dCSRmat** blocks;

} dBLCmat; /**< Matrix of REAL type in Block CSR format */

/**
 * \struct iBLCmat
 * \brief Block INT CSR matrix format
 *
 * \note The starting index of A is 0.
 */
typedef struct iBLCmat {

    //! row number of blocks in A, m
    INT brow;

    //! column number of blocks A, n
    INT bcol;

    //! blocks of iCSRmat, point to blocks[brow][bcol]
    iCSRmat** blocks;

} iBLCmat; /**< Matrix of INT type in Block CSR format */

/**
 * \struct block_dvector
 * \brief Block REAL vector structure
 */
typedef struct block_dvector {

    //! row number of blocks in A, m
    INT brow;

    //! blocks of dvector, point to blocks[brow]
    dvector** blocks;

} block_dvector; /**< Vector of REAL type in Block format */

/**
 * \struct block_ivector
 * \brief Block INT vector structure
 *
 * \note The starting index of A is 0.
 */
typedef struct block_ivector {

    //! row number of blocks in A, m
    INT brow;

    //! blocks of dvector, point to blocks[brow]
    ivector** blocks;

} block_ivector; /**< Vector of INT type in Block format */

/*---------------------------*/
/*--- Parameter structures --*/
/*---------------------------*/

/**
 * \struct AMG_data_bsr
 * \brief Data for multigrid levels in dBSRmat format
 *
 * \note This structure is needed for the AMG solver/preconditioner in BSR format
 */
typedef struct {

    //! max number of levels
    INT max_levels;

    //! number of levels in use <= max_levels
    INT num_levels;

    //! pointer to the matrix at level level_num
    dBSRmat A;

    //! restriction operator at level level_num
    dBSRmat R;

    //! prolongation operator at level level_num
    dBSRmat P;

    //! pointer to the right-hand side at level level_num
    dvector b;

    //! pointer to the iterative solution at level level_num
    dvector x;

    //! pointer to the diagonal inverse at level level_num
    dvector diaginv;

    //! pointer to the matrix at level level_num (csr format)
    dCSRmat Ac;

    //! pointer to the numerical dactorization from UMFPACK
    void* Numeric;

    //! data for Intel MKL PARDISO
    Pardiso_data pdata;

    //! pointer to the pressure block (only for reservoir simulation)
    dCSRmat PP;

    //! AMG data for PP, Li Zhao, 05/19/2023
    AMG_data* mglP;

    //! pointer to the temperature block (only for thermal reservoir simulation), Li
    //! Zhao, 05/23/2023
    dCSRmat TT;

    //! AMG data for TT, Li Zhao, 05/23/2023
    AMG_data* mglT;

    //! pointer to the pressure-temperature block (only for thermal reservoir
    //! simulation), Li Zhao, 05/19/2023
    dBSRmat PT;

    //! pointer to the auxiliary vectors for pressure block
    REAL* pw;

    //! pointer to the saturation block (only for reservoir simulation)
    dBSRmat SS;

    //! pointer to the auxiliary vectors for saturation block
    REAL* sw;

    //! pointer to the diagonal inverse of the saturation block at level level_num
    dvector diaginv_SS;

    //! ILU data for pressure block
    ILU_data PP_LU;

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

    //-----------------------------------------
    // extra near kernal space for extra solve

    //! Matrix data for near kernal
    dCSRmat* A_nk;

    //! Prolongation for near kernal
    dCSRmat* P_nk;

    //! Resriction for near kernal
    dCSRmat* R_nk;
    //-----------------------------------------

    //! temporary work space
    dvector w;

    //! data for MUMPS
    Mumps_data mumps;

} AMG_data_bsr; /**< AMG data for BSR matrices */

/**
 * \struct precond_diag_bsr
 * \brief Data for diagnal preconditioners in dBSRmat format
 *
 * \note This is needed for the diagnal preconditioner.
 */
typedef struct {

    //! dimension of each sub-block
    INT nb;

    //! diagnal elements
    dvector diag;

} precond_diag_bsr; /**< Data for diagnal preconditioners in dBSRmat format */

/**
 * \struct precond_data_bsr
 * \brief Data for preconditioners in dBSRmat format
 *
 * \note This structure is needed for the AMG solver/preconditioner in BSR format
 */
typedef struct {

    //! type of AMG method
    SHORT AMG_type;

    //! print level in AMG preconditioner
    SHORT print_level;

    //! max number of iterations of AMG preconditioner
    INT maxit;

    //! max number of AMG levels
    INT max_levels;

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

    //! coarsening type
    SHORT coarsening_type;

    //! relaxation parameter for SOR smoother
    REAL relaxation;

    //! coarse solver type for AMG
    SHORT coarse_solver;

    //! switch of scaling of the coarse grid correction
    SHORT coarse_scaling;

    //! degree of the polynomial used by AMLI cycle
    SHORT amli_degree;

    //! coefficients of the polynomial used by AMLI cycle
    REAL* amli_coef;

    //! smooth factor for smoothing the tentative prolongation
    REAL tentative_smooth;

    //! type of krylov method used by Nonlinear AMLI cycle
    SHORT nl_amli_krylov_type;

    //! AMG preconditioner data
    AMG_data_bsr* mgl_data;

    //! AMG preconditioner data for pressure block
    AMG_data* pres_mgl_data;

    //! ILU preconditioner data (needed for CPR type preconditioner)
    ILU_data* LU;

    //! Matrix data
    dBSRmat* A;

    // extra near kernal space

    //! Matrix data for near kernal
    dCSRmat* A_nk;

    //! Prolongation for near kernal
    dCSRmat* P_nk;

    //! Resriction for near kernal
    dCSRmat* R_nk;

    //! temporary dvector used to store and restore the residual
    dvector r;

    //! temporary work space for other usage
    REAL* w;

} precond_data_bsr; /**< Data for preconditioners in dBSRmat format */

/**
 * \brief Data for block preconditioners in dBLCmat format
 *
 * This is needed for the block preconditioner.
 */
typedef struct {

    /*-------------------------------------*/
    /* Basic data for block preconditioner */
    /*-------------------------------------*/
    dBLCmat* Ablc;   /**< problem data, the blocks */

    dCSRmat* A_diag; /**< data for each diagonal block*/

    dvector r;       /**< temp work space */

    /*------------------------------*/
    /* Data for the diagonal blocks */
    /*------------------------------*/

    /*--- solve by direct solver ---*/
    void** LU_diag; /**< LU decomposition for the diagonal blocks (for UMFpack) */

    /*--- solve by AMG ---*/
    AMG_data** mgl;      /**< AMG data for the diagonal blocks */

    AMG_param* amgparam; /**< parameters for AMG */

} precond_data_blc;      /**< Precond data for block matrices */

/**
 * \struct precond_data_sweeping
 *
 * \brief  Data for sweeping preconditioner
 *
 * \author Xiaozhe Hu
 * \date   05/01/2014
 *
 * \note This is needed for the sweeping preconditioner.
 */
typedef struct {

    INT NumLayers;        /**< number of layers */

    dBLCmat* A;           /**< problem data, the sparse matrix */
    dBLCmat* Ai;          /**< preconditioner data, the sparse matrix */

    dCSRmat* local_A;     /**< local stiffness matrix for each layer */
    void**   local_LU;    /**< lcoal LU decomposition (for UMFpack) */

    ivector* local_index; /**< local index for each layer */

    // temprary work spaces
    dvector r; /**< temporary dvector used to store and restore the residual */
    REAL*   w; /**< temporary work space for other usage */

} precond_data_sweeping; /**< Data for sweeping preconditioner */

#endif                   /* end if for __FASPBLOCK_HEADER__ */

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
