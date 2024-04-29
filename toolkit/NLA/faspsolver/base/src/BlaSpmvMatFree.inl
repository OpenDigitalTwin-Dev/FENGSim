/*! \file  BlaSpmvMatFree.inl
 *
 *  \brief BLAS2 operations when matrix is implicit or its format is not specified
 *
 *  \note  This file contains Level-1 (Bla) functions, which are used in:
 *         SolMatFree.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static inline void fasp_blas_mxv_csr (const void *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A               Pointer to CSR matrix A
 * \param x               Pointer to array x
 * \param y               Pointer to array y
 *
 * \author Feiteng Huang, Chensong Zhang
 * \date   09/19/2012
 */
static inline void fasp_blas_mxv_csr (const void *A,
                                      const REAL *x,
                                      REAL       *y)
{
    fasp_blas_dcsr_mxv((const dCSRmat *)A, x, y);
}

/**
 * \fn static inline void fasp_blas_mxv_bsr (const void *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A               Pointer to BSR matrix A
 * \param x               Pointer to array x
 * \param y               Pointer to array y
 *
 * \author Feiteng Huang, Chensong Zhang
 * \date   09/19/2012
 */
static inline void fasp_blas_mxv_bsr (const void *A,
                                      const REAL *x,
                                      REAL       *y)
{
    fasp_blas_dbsr_mxv((const dBSRmat *)A, x, y);
}

/**
 * \fn static inline void fasp_blas_mxv_str (const void *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A               Pointer to STR matrix A
 * \param x               Pointer to array x
 * \param y               Pointer to array y
 *
 * \author Feiteng Huang, Chensong Zhang
 * \date   09/19/2012
 */
static inline void fasp_blas_mxv_str (const void *A,
                                      const REAL *x,
                                      REAL       *y)
{
    fasp_blas_dstr_mxv((const dSTRmat *)A, x, y);
}

/**
 * \fn static inline void fasp_blas_mxv_blc (const void *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A               Pointer to BLC matrix A
 * \param x               Pointer to array x
 * \param y               Pointer to array y
 *
 * \author Feiteng Huang, Chensong Zhang
 * \date   09/19/2012
 */
static inline void fasp_blas_mxv_blc (const void *A,
                                      const REAL *x,
                                      REAL       *y)
{
    fasp_blas_dblc_mxv((const dBLCmat *)A, x, y);
}

/**
 * \fn static inline void fasp_blas_mxv_csrl (const void *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A               Pointer to CSRL matrix A
 * \param x               Pointer to array x
 * \param y               Pointer to array y
 *
 * \author Feiteng Huang, Chensong Zhang
 * \date   09/19/2012
 */
static inline void fasp_blas_mxv_csrl (const void *A,
                                       const REAL *x,
                                       REAL       *y)
{
    fasp_blas_dcsrl_mxv((const dCSRLmat *)A, x, y);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
