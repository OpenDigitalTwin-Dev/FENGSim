/*! \file  BlaSparseCOO.c
 *
 *  \brief Sparse matrix operations for dCOOmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c and AuxThreads.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn dCOOmat fasp_dcoo_create (const INT m, const INT n, const INT nnz)
 *
 * \brief Create IJ sparse matrix data memory space
 *
 * \param m    Number of rows
 * \param n    Number of columns
 * \param nnz  Number of nonzeros
 *
 * \return A   The new dCOOmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 */
dCOOmat fasp_dcoo_create (const INT  m,
                          const INT  n,
                          const INT  nnz)
{
    dCOOmat A;
    
    A.rowind = (INT *)fasp_mem_calloc(nnz, sizeof(INT));
    A.colind = (INT *)fasp_mem_calloc(nnz, sizeof(INT));
    A.val    = (REAL *)fasp_mem_calloc(nnz, sizeof(REAL));
    
    A.row = m; A.col = n; A.nnz = nnz;
    
    return A;
}

/**
 * \fn void fasp_dcoo_alloc (const INT m, const INT n, const INT nnz, dCOOmat *A)
 *
 * \brief Allocate COO sparse matrix memory space
 *
 * \param m      Number of rows
 * \param n      Number of columns
 * \param nnz    Number of nonzeros
 * \param A      Pointer to the dCSRmat matrix
 *
 * \author Xiaozhe Hu
 * \date   03/25/2013
 */
void fasp_dcoo_alloc (const INT  m,
                      const INT  n,
                      const INT  nnz,
                      dCOOmat   *A)
{
    
    if ( nnz > 0 ) {
        A->rowind = (INT *)fasp_mem_calloc(nnz, sizeof(INT));
        A->colind = (INT *)fasp_mem_calloc(nnz, sizeof(INT));
        A->val    = (REAL*)fasp_mem_calloc(nnz,sizeof(REAL));
    }
    else {
        A->rowind = NULL;
        A->colind = NULL;
        A->val    = NULL;
    }
    
    A->row = m; A->col = n; A->nnz = nnz;
    
    return;
}

/**
 * \fn void fasp_dcoo_free (dCOOmat *A)
 *
 * \brief Free IJ sparse matrix data memory space
 *
 * \param A   Pointer to the dCOOmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/03
 */
void fasp_dcoo_free (dCOOmat *A)
{
    if (A==NULL) return;
    
    fasp_mem_free(A->rowind); A->rowind = NULL;
    fasp_mem_free(A->colind); A->colind = NULL;
    fasp_mem_free(A->val);    A->val    = NULL;
}

/**
 * \fn void fasp_dcoo_shift (dCOOmat *A, const INT offset)
 *
 * \brief Re-index a REAL matrix in IJ format to make the index starting from 0 or 1.
 *
 * \param A       Pointer to IJ matrix
 * \param offset  Size of offset (1 or -1)
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/25/2012
 */
void fasp_dcoo_shift (dCOOmat   *A,
                      const INT  offset)
{
    const INT nnz = A->nnz;
    INT       i, *ai = A->rowind, *aj = A->colind;
    
    // Variables for OpenMP
    SHORT nthreads = 1, use_openmp = FALSE;
    INT myid, mybegin, myend;
    
#ifdef _OPENMP
    if (nnz > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads = fasp_get_num_threads();
    }
#endif
        
    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for private(myid, i, mybegin, myend)
#endif
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, nnz, &mybegin, &myend);
            for (i=mybegin; i<myend; ++i) {
                ai[i]+=offset; aj[i]+=offset;
            }
        }
    }
    else {
        for (i=0;i<nnz;++i) {
            ai[i]+=offset; aj[i]+=offset;
        }
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
