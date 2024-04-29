/*! \file  BlaSpmvBSR.c
 *
 *  \brief Linear algebraic operations for dBSRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxThreads.c, BlaSmallMat.c, and BlaArray.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_blas_dbsr_add (const dBSRmat *A, const REAL alpha,
 *                               const dBSRmat *B, const REAL beta, dBSRmat *C)
 *
 * \brief compute C = alpha*A + beta*B in BSR format
 *
 * \param A      Pointer to dBSRmat matrix
 * \param alpha  REAL factor alpha
 * \param B      Pointer to dBSRmat matrix
 * \param beta   REAL factor beta
 * \param C      Pointer to dBSRmat matrix
 *
 * \return       FASP_SUCCESS if succeed, ERROR if not
 *
 * \author Li Zhao
 * \date   07/03/2023
 *
 */
SHORT fasp_blas_dbsr_add(
    const dBSRmat* A, const REAL alpha, const dBSRmat* B, const REAL beta, dBSRmat* C)
{
    INT i, j, k, l;
    INT count = 0, added, countrow;
    INT nb = A->nb, nb2 = nb * nb;

    SHORT status = FASP_SUCCESS, use_openmp = FALSE;

#ifdef _OPENMP
    INT mybegin, myend, myid, nthreads;
    if (A->nnz > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (A->ROW != B->ROW || A->COL != B->COL || A->nb != B->nb) {
        printf("### ERROR: Matrix sizes do not match!\n");
        status = ERROR_MAT_SIZE;
        goto FINISHED;
    }

    if (A == NULL && B == NULL) {
        C->ROW            = 0;
        C->COL            = 0;
        C->NNZ            = 0;
        C->nb             = nb;
        C->storage_manner = A->storage_manner;
        status            = FASP_SUCCESS;
        goto FINISHED;
    }

    if (A->NNZ == 0 && B->NNZ == 0) {
        C->ROW            = A->ROW;
        C->COL            = A->COL;
        C->NNZ            = A->NNZ;
        C->nb             = nb;
        C->storage_manner = A->storage_manner;
        status            = FASP_SUCCESS;
        goto FINISHED;
    }

    // empty matrix A
    if (A->NNZ == 0 || A == NULL) {
        fasp_dbsr_alloc(B->ROW, B->COL, B->NNZ, B->nb, B->storage_manner, C);
        memcpy(C->IA, B->IA, (B->ROW + 1) * sizeof(INT));
        memcpy(C->JA, B->JA, (B->NNZ) * sizeof(INT));

        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, B->NNZ, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i)
                    fasp_blas_smat_axm1(&B->val[i * nb2], nb, beta, &C->val[i * nb2]);
            }
#endif
        } else {
            for (i = 0; i < B->NNZ; ++i) {
                // C->val[i] = B->val[i] * beta;
                fasp_blas_smat_axm1(&B->val[i * nb2], nb, beta, &C->val[i * nb2]);
            }
        }

        status = FASP_SUCCESS;
        goto FINISHED;
    }

    // empty matrix B
    if (B->NNZ == 0 || B == NULL) {
        fasp_dbsr_alloc(A->ROW, A->COL, A->NNZ, A->nb, A->storage_manner, C);
        memcpy(C->IA, A->IA, (A->ROW + 1) * sizeof(INT));
        memcpy(C->JA, A->JA, (A->NNZ) * sizeof(INT));

        if (use_openmp) {
#ifdef _OPENMP
            INT mybegin, myend, myid;
#pragma omp parallel private(myid, mybegin, myend, i)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, A->NNZ, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i)
                    fasp_blas_smat_axm1(&A->val[i * nb2], nb, alpha, &C->val[i * nb2]);
            }
#endif
        } else {
            for (i = 0; i < A->NNZ; ++i)
                // C->val[i] = A->val[i] * alpha;
                fasp_blas_smat_axm1(&A->val[i * nb2], nb, alpha, &C->val[i * nb2]);
        }

        status = FASP_SUCCESS;
        goto FINISHED;
    }

    C->ROW = A->ROW;
    C->COL = A->COL;

    C->nb             = A->nb;
    C->storage_manner = A->storage_manner;

    C->IA = (INT*)fasp_mem_calloc(C->ROW + 1, sizeof(INT));

    // allocate work space for C->JA and C->val
    C->JA = (INT*)fasp_mem_calloc(A->NNZ + B->NNZ, sizeof(INT));

    C->val = (REAL*)fasp_mem_calloc((A->NNZ + B->NNZ) * nb2, sizeof(REAL));

    // initial C->IA
    memset(C->IA, 0, sizeof(INT) * (C->ROW + 1));
    memset(C->JA, -1, sizeof(INT) * (A->NNZ + B->NNZ));

    for (i = 0; i < A->ROW; ++i) {
        countrow = 0;
        for (j = A->IA[i]; j < A->IA[i + 1]; ++j) {
            // C->val[count] = alpha * A->val[j];
            fasp_blas_smat_axm1(&A->val[j * nb2], nb, alpha, &C->val[count * nb2]);
            C->JA[count] = A->JA[j];
            C->IA[i + 1]++;
            count++;
            countrow++;
        } // end for js

        for (k = B->IA[i]; k < B->IA[i + 1]; ++k) {
            added = 0;

            for (l = C->IA[i]; l < C->IA[i] + countrow + 1; l++) {
                if (B->JA[k] == C->JA[l]) {
                    // C->val[l] = C->val[l] + beta * B->val[k];
                    fasp_blas_smat_add(&C->val[l * nb2], &B->val[k * nb2], nb, 1.0,
                                       beta, &C->val[l * nb2]);
                    added = 1;
                    break;
                }
            } // end for l

            if (added == 0) {
                // C->val[count] = beta * B->val[k];
                fasp_blas_smat_axm1(&B->val[k * nb2], nb, beta, &C->val[count * nb2]);
                C->JA[count] = B->JA[k];
                C->IA[i + 1]++;
                count++;
            }

        } // end for k

        C->IA[i + 1] += C->IA[i];
    }

    C->NNZ = count;
    C->JA  = (INT*)fasp_mem_realloc(C->JA, (count) * sizeof(INT));
    C->val = (REAL*)fasp_mem_realloc(C->val, (nb2 * count) * sizeof(REAL));

FINISHED:
    return status;
}

/**
 * \fn void fasp_blas_dbsr_axm (dBSRmat *A, const REAL alpha)
 *
 * \brief Multiply a sparse matrix A in BSR format by a scalar alpha.
 *
 * \param A      Pointer to dBSRmat matrix A
 * \param alpha  REAL factor alpha
 *
 * \author Xiaozhe Hu
 * \date   05/26/2014
 */
void fasp_blas_dbsr_axm(dBSRmat* A, const REAL alpha)
{
    const INT nnz = A->NNZ;
    const INT nb  = A->nb;

    // A direct calculation can be written as:
    fasp_blas_darray_ax(nnz * nb * nb, alpha, A->val);
}

/*!
 * \fn void fasp_blas_dbsr_aAxpby (const REAL alpha, dBSRmat *A,
 *                                 REAL *x, const REAL beta, REAL *y)
 *
 * \brief Compute y := alpha*A*x + beta*y
 *
 * \param alpha  REAL factor alpha
 * \param A      Pointer to the dBSRmat matrix
 * \param x      Pointer to the array x
 * \param beta   REAL factor beta
 * \param y      Pointer to the array y
 *
 * \author Zhiyang Zhou
 * \date   10/25/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 06/29/2012
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_blas_dbsr_aAxpby(
    const REAL alpha, dBSRmat* A, REAL* x, const REAL beta, REAL* y)
{
    /* members of A */
    INT   ROW = A->ROW;
    INT   nb  = A->nb;
    INT*  IA  = A->IA;
    INT*  JA  = A->JA;
    REAL* val = A->val;

    /* local variables */
    INT   size = ROW * nb;
    INT   jump = nb * nb;
    INT   i, j, k, iend;
    REAL  temp;
    REAL* pA  = NULL;
    REAL* px0 = NULL;
    REAL* py0 = NULL;
    REAL* py  = NULL;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    //----------------------------------------------
    //   Treat (alpha == 0.0) computation
    //----------------------------------------------

    if (alpha == 0.0) {
        fasp_blas_darray_ax(size, beta, y);
        return;
    }

    //-------------------------------------------------
    //   y = (beta/alpha)*y
    //-------------------------------------------------

    temp = beta / alpha;
    if (temp != 1.0) {
        if (temp == 0.0) {
            memset(y, 0X0, size * sizeof(REAL));
        } else {
            // for (i = size; i--; ) y[i] *= temp; // modified by Xiaozhe, 03/11/2011
            fasp_blas_darray_ax(size, temp, y);
        }
    }

    //-----------------------------------------------------------------
    //   y += A*x (Core Computation)
    //   each non-zero block elements are stored in row-major order
    //-----------------------------------------------------------------

    switch (nb) {
        case 2:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 2];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 4; // &val[k*jump];
                                px0 = x + j * 2;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc2(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 2];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 4; // &val[k*jump];
                            px0 = x + j * 2;   // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc2(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 3:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 3];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 3];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 9; // &val[k*jump];
                            px0 = x + j * 3;   // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc3(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 5:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 5];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 5];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 25; // &val[k*jump];
                            px0 = x + j * 5;    // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc5(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 7:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 7];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb]��
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 7];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 49; // &val[k*jump];
                            px0 = x + j * 7;    // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc7(pA, px0, py);
                        }
                    }
                }
            }
            break;

        default:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * nb];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * nb];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * jump; // &val[k*jump];
                            px0 = x + j * nb;     // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx(pA, px0, py, nb);
                        }
                    }
                }
            }
            break;
    }

    //------------------------------------------
    //   y = alpha*y
    //------------------------------------------

    if (alpha != 1.0) {
        fasp_blas_darray_ax(size, alpha, y);
    }
}

/*!
 * \fn void fasp_blas_dbsr_aAxpy (const REAL alpha, const dBSRmat *A,
 *                                const REAL *x, REAL *y)
 *
 * \brief Compute y := alpha*A*x + y
 *
 * \param alpha  REAL factor alpha
 * \param A      Pointer to the dBSRmat matrix
 * \param x      Pointer to the array x
 * \param y      Pointer to the array y
 *
 * \author Zhiyang Zhou
 * \date   10/25/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_blas_dbsr_aAxpy(const REAL alpha, const dBSRmat* A, const REAL* x, REAL* y)
{
    /* members of A */
    const INT   ROW = A->ROW;
    const INT   nb  = A->nb;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;
    const REAL* val = A->val;

    /* local variables */
    const REAL* pA  = NULL;
    const REAL* px0 = NULL;
    REAL*       py0 = NULL;
    REAL*       py  = NULL;

    REAL temp = 0.0;
    INT  size = ROW * nb;
    INT  jump = nb * nb;
    INT  i, j, k, iend;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    //----------------------------------------------
    //   Treat (alpha == 0.0) computation
    //----------------------------------------------

    if (alpha == 0.0) {
        return; // Nothing to compute
    }

    //-------------------------------------------------
    //   y = (1.0/alpha)*y
    //-------------------------------------------------

    if (alpha != 1.0) {
        temp = 1.0 / alpha;
        fasp_blas_darray_ax(size, temp, y);
    }

    //-----------------------------------------------------------------
    //   y += A*x (Core Computation)
    //   each non-zero block elements are stored in row-major order
    //-----------------------------------------------------------------

    switch (nb) {
        case 2:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 2];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 4; // &val[k*jump];
                                px0 = x + j * 2;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc2(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 2];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 4; // &val[k*jump];
                            px0 = x + j * 2;   // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc2(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 3:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 3];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 3];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 9; // &val[k*jump];
                            px0 = x + j * 3;   // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc3(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 5:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 5];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 5];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 25; // &val[k*jump];
                            px0 = x + j * 5;    // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc5(pA, px0, py);
                        }
                    }
                }
            }
            break;

        case 7:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 7];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 7];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * 49; // &val[k*jump];
                            px0 = x + j * 7;    // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx_nc7(pA, px0, py);
                        }
                    }
                }
            }
            break;

        default:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, pA, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * nb];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * nb];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            pA  = val + k * jump; // &val[k*jump];
                            px0 = x + j * nb;     // &x[j*nb];
                            py  = py0;
                            fasp_blas_smat_ypAx(pA, px0, py, nb);
                        }
                    }
                }
            }
            break;
    }

    //------------------------------------------
    //   y = alpha*y
    //------------------------------------------

    if (alpha != 1.0) {
        fasp_blas_darray_ax(size, alpha, y);
    }
    return;
}

/*!
 * \fn void fasp_blas_dbsr_aAxpy_agg (const REAL alpha, const dBSRmat *A,
 *                                    const REAL *x, REAL *y)
 *
 * \brief Compute y := alpha*A*x + y where each small block matrix is an identity matrix
 *
 * \param alpha  REAL factor alpha
 * \param A      Pointer to the dBSRmat matrix
 * \param x      Pointer to the array x
 * \param y      Pointer to the array y
 *
 * \author Xiaozhe Hu
 * \date   01/02/2014
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_blas_dbsr_aAxpy_agg(const REAL     alpha,
                              const dBSRmat* A,
                              const REAL*    x,
                              REAL*          y)
{
    /* members of A */
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;

    /* local variables */
    const REAL* px0 = NULL;
    REAL *      py0 = NULL, *py = NULL;
    SHORT       nthreads = 1, use_openmp = FALSE;

    INT  size = ROW * nb;
    INT  i, j, k, iend;
    REAL temp = 0.0;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    //----------------------------------------------
    //   Treat (alpha == 0.0) computation
    //----------------------------------------------

    if (alpha == 0.0) {
        return; // Nothing to compute
    }

    //-------------------------------------------------
    //   y = (1.0/alpha)*y
    //-------------------------------------------------

    if (alpha != 1.0) {
        temp = 1.0 / alpha;
        fasp_blas_darray_ax(size, temp, y);
    }

    //-----------------------------------------------------------------
    //   y += A*x (Core Computation)
    //   each non-zero block elements are stored in row-major order
    //-----------------------------------------------------------------

    switch (nb) {
        case 2:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 2];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                px0 = x + j * 2; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 2];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            px0 = x + j * 2; // &x[j*nb];
                            py  = py0;
                            py[0] += px0[0];
                            py[1] += px0[1];
                        }
                    }
                }
            }
            break;

        case 3:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 3];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 3];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            px0 = x + j * 3; // &x[j*nb];
                            py  = py0;
                            py[0] += px0[0];
                            py[1] += px0[1];
                            py[2] += px0[2];
                        }
                    }
                }
            }
            break;

        case 5:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 5];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 5];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            px0 = x + j * 5; // &x[j*nb];
                            py  = py0;
                            py[0] += px0[0];
                            py[1] += px0[1];
                            py[2] += px0[2];
                            py[3] += px0[3];
                            py[4] += px0[4];
                        }
                    }
                }
            }
            break;

        case 7:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * 7];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * 7];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            px0 = x + j * 7; // &x[j*nb];
                            py  = py0;
                            py[0] += px0[0];
                            py[1] += px0[1];
                            py[2] += px0[2];
                            py[3] += px0[3];
                            py[4] += px0[4];
                            py[5] += px0[5];
                            py[6] += px0[6];
                        }
                    }
                }
            }
            break;

        default:
            {
                if (use_openmp) {
                    INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, py0, k, j, px0, py, iend)
#endif
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0  = &y[i * nb];
                            iend = IA[i + 1];
                            for (k = IA[i]; k < iend; ++k) {
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0  = &y[i * nb];
                        iend = IA[i + 1];
                        for (k = IA[i]; k < iend; ++k) {
                            j   = JA[k];
                            px0 = x + j * nb; // &x[j*nb];
                            py  = py0;
                            fasp_blas_darray_axpy(nb, 1.0, px0, py);
                        }
                    }
                }
            }
            break;
    }

    //------------------------------------------
    //   y = alpha*y
    //------------------------------------------

    if (alpha != 1.0) fasp_blas_darray_ax(size, alpha, y);

    return;
}

/*!
 * \fn void fasp_blas_dbsr_mxv (const dBSRmat *A, const REAL *x, REAL *y)
 *
 * \brief Compute y := A*x
 *
 * \param A      Pointer to the dBSRmat matrix
 * \param x      Pointer to the array x
 * \param y      Pointer to the array y
 *
 * \author Zhiyang Zhou
 * \date   10/25/2010
 *
 * \note Works for general nb (Xiaozhe)
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_dbsr_mxv(const dBSRmat* A, const REAL* x, REAL* y)
{
    /* members of A */
    const INT   ROW = A->ROW;
    const INT   nb  = A->nb;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;
    const REAL* val = A->val;

    /* local variables */
    INT size = ROW * nb;
    INT jump = nb * nb;
    INT i, j, k, num_nnz_row;

    const REAL* pA  = NULL;
    const REAL* px0 = NULL;
    REAL*       py0 = NULL;
    REAL*       py  = NULL;

    SHORT use_openmp = FALSE;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    //-----------------------------------------------------------------
    //  zero out 'y'
    //-----------------------------------------------------------------
    fasp_darray_set(size, y, 0.0);

    //-----------------------------------------------------------------
    //   y = A*x (Core Computation)
    //   each non-zero block elements are stored in row-major order
    //-----------------------------------------------------------------

    switch (nb) {
        case 3:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 3];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {
                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 9;
                                        px0 = x + j * 3;
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc3(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 3];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {
                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 9; // &val[k*jump];
                                px0 = x + j * 3;   // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    pA  = val + k * 9; // &val[k*jump];
                                    px0 = x + j * 3;   // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);
                                }
                                break;
                        }
                    }
                }
            }
            break;

        case 5:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 5];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {
                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 25; // &val[k*jump];
                                        px0 = x + j * 5;    // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc5(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 5];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {
                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 25; // &val[k*jump];
                                px0 = x + j * 5;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);
                                }
                                break;
                        }
                    }
                }
            }
            break;

        case 7:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 7];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {
                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 49; // &val[k*jump];
                                        px0 = x + j * 7;    // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc7(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 7];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {
                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                k++;
                                j   = JA[k];
                                pA  = val + k * 49; // &val[k*jump];
                                px0 = x + j * 7;    // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);
                                }
                                break;
                        }
                    }
                }
            }
            break;

        default:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * nb];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {
                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * jump; // &val[k*jump];
                                        px0 = x + j * nb;     // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_smat_ypAx(pA, px0, py, nb);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * nb];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {
                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                k++;
                                j   = JA[k];
                                pA  = val + k * jump; // &val[k*jump];
                                px0 = x + j * nb;     // &x[j*nb];
                                py  = py0;
                                fasp_blas_smat_ypAx(pA, px0, py, nb);

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    pA  = val + k * jump; // &val[k*jump];
                                    px0 = x + j * nb;     // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx(pA, px0, py, nb);
                                }
                                break;
                        }
                    }
                }
            }
            break;
    }
}

/*!
 * \fn void fasp_blas_dbsr_mxv_agg (const dBSRmat *A, const REAL *x, REAL *y)
 *
 * \brief Compute y := A*x, where each small block matrices of A is an identity
 *
 * \param A      Pointer to the dBSRmat matrix
 * \param x      Pointer to the array x
 * \param y      Pointer to the array y
 *
 * \author Xiaozhe Hu
 * \date   01/02/2014
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_blas_dbsr_mxv_agg(const dBSRmat* A, const REAL* x, REAL* y)
{
    /* members of A */
    const INT  ROW  = A->ROW;
    const INT  nb   = A->nb;
    const INT  size = ROW * nb;
    const INT* IA   = A->IA;
    const INT* JA   = A->JA;

    /* local variables */
    const REAL* px0 = NULL;
    REAL *      py0 = NULL, *py = NULL;
    INT         i, j, k, num_nnz_row;
    SHORT       use_openmp = FALSE;

#ifdef _OPENMP
    const REAL* val = A->val;
    const REAL* pA;
    INT         myid, mybegin, myend, nthreads;
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    //-----------------------------------------------------------------
    //  zero out 'y'
    //-----------------------------------------------------------------
    fasp_darray_set(size, y, 0.0);

    //-----------------------------------------------------------------
    //   y = A*x (Core Computation)
    //   each non-zero block elements are stored in row-major order
    //-----------------------------------------------------------------

    switch (nb) {
        case 3:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 3];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {
                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 9;
                                    px0 = x + j * 3;
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc3(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 9;
                                        px0 = x + j * 3;
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc3(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 3];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {
                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 3; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    px0 = x + j * 3; // &x[j*nb];
                                    py  = py0;
                                    py[0] += px0[0];
                                    py[1] += px0[1];
                                    py[2] += px0[2];
                                }
                                break;
                        }
                    }
                }
            }
            break;

        case 5:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 5];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {

                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 25; // &val[k*jump];
                                    px0 = x + j * 5;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc5(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 25; // &val[k*jump];
                                        px0 = x + j * 5;    // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc5(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 5];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {

                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 5; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    px0 = x + j * 5; // &x[j*nb];
                                    py  = py0;
                                    py[0] += px0[0];
                                    py[1] += px0[1];
                                    py[2] += px0[2];
                                    py[3] += px0[3];
                                    py[4] += px0[4];
                                }
                                break;
                        }
                    }
                }
            }
            break;

        case 7:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * 7];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {

                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    k++;
                                    j   = JA[k];
                                    pA  = val + k * 49; // &val[k*jump];
                                    px0 = x + j * 7;    // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_smat_ypAx_nc7(pA, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        pA  = val + k * 49; // &val[k*jump];
                                        px0 = x + j * 7;    // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_smat_ypAx_nc7(pA, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * 7];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {

                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                k++;
                                j   = JA[k];
                                px0 = x + j * 7; // &x[j*nb];
                                py  = py0;
                                py[0] += px0[0];
                                py[1] += px0[1];
                                py[2] += px0[2];
                                py[3] += px0[3];
                                py[4] += px0[4];
                                py[5] += px0[5];
                                py[6] += px0[6];

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    px0 = x + j * 7; // &x[j*nb];
                                    py  = py0;
                                    py[0] += px0[0];
                                    py[1] += px0[1];
                                    py[2] += px0[2];
                                    py[3] += px0[3];
                                    py[4] += px0[4];
                                    py[5] += px0[5];
                                    py[6] += px0[6];
                                }
                                break;
                        }
                    }
                }
            }
            break;

        default:
            {
                if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i, py0, num_nnz_row, k, j, pA, px0, \
                                 py)
                    {
                        myid = omp_get_thread_num();
                        fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            py0         = &y[i * nb];
                            num_nnz_row = IA[i + 1] - IA[i];
                            switch (num_nnz_row) {

                                case 3:
                                    k   = IA[i];
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    break;

                                case 4:
                                    k   = IA[i];
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    break;

                                case 5:
                                    k   = IA[i];
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    break;

                                case 6:
                                    k   = IA[i];
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    break;

                                case 7:
                                    k   = IA[i];
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    k++;
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                    break;

                                default:
                                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                                        j   = JA[k];
                                        px0 = x + j * nb; // &x[j*nb];
                                        py  = py0;
                                        fasp_blas_darray_axpy(nb, 1.0, px0, py);
                                    }
                                    break;
                            }
                        }
                    }
#endif
                } else {
                    for (i = 0; i < ROW; ++i) {
                        py0         = &y[i * nb];
                        num_nnz_row = IA[i + 1] - IA[i];
                        switch (num_nnz_row) {

                            case 3:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                break;

                            case 4:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                break;

                            case 5:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                break;

                            case 6:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                break;

                            case 7:
                                k   = IA[i];
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                k++;
                                j   = JA[k];
                                px0 = x + j * nb; // &x[j*nb];
                                py  = py0;
                                fasp_blas_darray_axpy(nb, 1.0, px0, py);

                                break;

                            default:
                                for (k = IA[i]; k < IA[i + 1]; ++k) {
                                    j   = JA[k];
                                    px0 = x + j * nb; // &x[j*nb];
                                    py  = py0;
                                    fasp_blas_darray_axpy(nb, 1.0, px0, py);
                                }
                                break;
                        }
                    }
                }
            }
            break;
    }
}

/**
 * \fn void fasp_blas_dbsr_mxm2 (const dBSRmat *A, const dBSRmat *B, dBSRmat *C)
 *
 * \brief Sparse matrix multiplication C=A*B
 *
 * \param A   Pointer to the dBSRmat matrix A
 * \param B   Pointer to the dBSRmat matrix B
 * \param C   Pointer to dBSRmat matrix equal to A*B
 *
 * \author Xiaozhe Hu
 * \date   05/26/2014
 *
 * \note This fct will be replaced! -- Xiaozhe; Rename fasp_blas_dbsr_mxm to
 * fasp_blas_dbsr_mxm2 -- Li Zhao (07/02/2023)
 */
void fasp_blas_dbsr_mxm2(const dBSRmat* A, const dBSRmat* B, dBSRmat* C)
{

    INT  i, j, k, l, count;
    INT* JD = (INT*)fasp_mem_calloc(B->COL, sizeof(INT));

    const INT nb  = A->nb;
    const INT nb2 = nb * nb;

    // check A and B see if there are compatible for multiplication
    if ((A->COL != B->ROW) && (A->nb != B->nb)) {
        printf("### ERROR: Matrix sizes do not match!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    C->ROW            = A->ROW;
    C->COL            = B->COL;
    C->nb             = A->nb;
    C->storage_manner = A->storage_manner;

    C->val = NULL;
    C->JA  = NULL;
    C->IA  = (INT*)fasp_mem_calloc(C->ROW + 1, sizeof(INT));

    REAL* temp = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));

    for (i = 0; i < B->COL; ++i) JD[i] = -1;

    // step 1: Find first the structure IA of C
    for (i = 0; i < C->ROW; ++i) {
        count = 0;

        for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
            for (j = B->IA[A->JA[k]]; j < B->IA[A->JA[k] + 1]; ++j) {
                for (l = 0; l < count; l++) {
                    if (JD[l] == B->JA[j]) break;
                }

                if (l == count) {
                    JD[count] = B->JA[j];
                    count++;
                }
            }
        }
        C->IA[i + 1] = count;
        for (j = 0; j < count; ++j) {
            JD[j] = -1;
        }
    }

    for (i = 0; i < C->ROW; ++i) C->IA[i + 1] += C->IA[i];

    // step 2: Find the structure JA of C
    INT countJD;

    C->JA = (INT*)fasp_mem_calloc(C->IA[C->ROW], sizeof(INT));

    for (i = 0; i < C->ROW; ++i) {
        countJD = 0;
        count   = C->IA[i];
        for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
            for (j = B->IA[A->JA[k]]; j < B->IA[A->JA[k] + 1]; ++j) {
                for (l = 0; l < countJD; l++) {
                    if (JD[l] == B->JA[j]) break;
                }

                if (l == countJD) {
                    C->JA[count] = B->JA[j];
                    JD[countJD]  = B->JA[j];
                    count++;
                    countJD++;
                }
            }
        }

        // for (j=0;j<countJD;++j) JD[j]=-1;
        fasp_iarray_set(countJD, JD, -1);
    }

    fasp_mem_free(JD);
    JD = NULL;

    // step 3: Find the structure A of C
    C->val = (REAL*)fasp_mem_calloc((C->IA[C->ROW]) * nb2, sizeof(REAL));

    for (i = 0; i < C->ROW; ++i) {
        for (j = C->IA[i]; j < C->IA[i + 1]; ++j) {

            fasp_darray_set(nb2, C->val + (j * nb2), 0x0);

            for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
                for (l = B->IA[A->JA[k]]; l < B->IA[A->JA[k] + 1]; l++) {
                    if (B->JA[l] == C->JA[j]) {
                        fasp_blas_smat_mul(A->val + (k * nb2), B->val + (l * nb2), temp,
                                           nb);
                        fasp_blas_darray_axpy(nb2, 1.0, temp, C->val + (j * nb2));
                    } // end if
                }     // end for l
            }         // end for k
        }             // end for j
    }                 // end for i

    C->NNZ = C->IA[C->ROW] - C->IA[0];

    fasp_mem_free(temp);
    temp = NULL;
}

/**
 * \fn void fasp_blas_dbsr_mxm (const dBSRmat *A, const dBSRmat *B, dBSRmat *C)
 *
 * \brief Sparse matrix multiplication C=A*B
 *
 * \param A   Pointer to the dBSRmat matrix A
 * \param B   Pointer to the dBSRmat matrix B
 * \param C   Pointer to dBSRmat matrix equal to A*B
 *
 * \author Li Zhao
 * \date   07/02/2023
 *
 * \note This is a new function that is 50% faster than old version (i.e.,
 * fasp_blas_dbsr_mxm2) -- Li Zhao (07/02/2023)
 */
void fasp_blas_dbsr_mxm(const dBSRmat* A, const dBSRmat* B, dBSRmat* C)
{
    REAL* A_data  = A->val;
    INT*  A_i     = A->IA;
    INT*  A_j     = A->JA;
    INT   nrows_A = A->ROW;
    INT   ncols_A = A->COL;

    REAL* B_data  = B->val;
    INT*  B_i     = B->IA;
    INT*  B_j     = B->JA;
    INT   nrows_B = B->ROW;
    INT   ncols_B = B->COL;

    INT   ia, ib, ic, ja, jb;
    INT   num_nonzeros = 0;
    INT   row_start, counter;
    REAL *a_entry, *b_entry;
    INT*  B_marker = NULL;

    const INT nb  = A->nb;
    const INT nb2 = nb * nb;

    B_marker = (INT*)fasp_mem_calloc(ncols_B, sizeof(INT));

    // check A and B see if there are compatible for multiplication
    if ((A->COL != B->ROW) && (A->nb != B->nb)) {
        printf("### ERROR: Matrix sizes do not match!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    C->ROW            = A->ROW;
    C->COL            = B->COL;
    C->nb             = A->nb;
    C->storage_manner = A->storage_manner;

    C->val = NULL;
    C->JA  = NULL;
    C->IA  = (INT*)fasp_mem_calloc(C->ROW + 1, sizeof(INT));

    /* initialize the marker array */
    fasp_iarray_set(ncols_B, B_marker, -1);

    /* step 1: obtain the nonzero-structure of C */
    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb = B_j[ib];
                if (B_marker[jb] != ic) {
                    B_marker[jb] = ic;
                    num_nonzeros++;
                }
            }
        }
        C->IA[ic + 1] = num_nonzeros;
    }

    C->NNZ = num_nonzeros;
    C->JA  = (INT*)fasp_mem_calloc(num_nonzeros, sizeof(INT));
    C->val = (REAL*)fasp_mem_calloc(num_nonzeros * nb2, sizeof(REAL));

    /* initialize the marker array again */
    fasp_iarray_set(ncols_B, B_marker, -1);

    REAL* temp = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));

    /* step 2: fill in the nonzero entries of C */
    counter = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        row_start = C->IA[ic];
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja      = A_j[ia];
            a_entry = A_data + ia * nb2;
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb      = B_j[ib];
                b_entry = B_data + ib * nb2;
                // temp = a_entry * b_entry
                fasp_blas_smat_mul(a_entry, b_entry, temp, nb);
                if (B_marker[jb] < row_start) {
                    B_marker[jb]        = counter;
                    C->JA[B_marker[jb]] = jb;
                    // C->val[B_marker[jb]*nb2] = a_entry * b_entry;
                    memcpy(C->val + B_marker[jb] * nb2, temp, nb2 * sizeof(REAL));
                    counter++;
                } else {
                    // C->val[B_marker[jb]*nb2] += a_entry * b_entry;
                    fasp_blas_darray_axpy(nb2, 1.0, temp, C->val + B_marker[jb] * nb2);
                }
            }
        }
    }

    fasp_mem_free(temp);
    temp = NULL;
    fasp_mem_free(B_marker);
    B_marker = NULL;
}

/**
 * \fn void fasp_blas_dbsr_mxm_adb(const dBSRmat* A, dvector *D, const dBSRmat* B,
 * dBSRmat* C)
 *
 * \brief Sparse matrix multiplication C=A*D*B, where D is diagnal matrix
 *
 * \param A   Pointer to the dBSRmat matrix A
 * \param D   Pointer to the block diagnal matrix D
 * \param B   Pointer to the dBSRmat matrix B
 * \param C   Pointer to dBSRmat matrix equal to A*D*B
 *
 * \author Li Zhao
 * \date   07/02/2023
 *
 */
void fasp_blas_dbsr_mxm_adb(const dBSRmat* A, dvector* D, const dBSRmat* B, dBSRmat* C)
{
    REAL* A_data  = A->val;
    INT*  A_i     = A->IA;
    INT*  A_j     = A->JA;
    INT   nrows_A = A->ROW;
    INT   ncols_A = A->COL;

    REAL* B_data  = B->val;
    INT*  B_i     = B->IA;
    INT*  B_j     = B->JA;
    INT   nrows_B = B->ROW;
    INT   ncols_B = B->COL;

    INT   ia, ib, ic, ja, jb;
    INT   num_nonzeros = 0;
    INT   row_start, counter;
    REAL *a_entry, *b_entry, *d_entry;
    INT*  B_marker = NULL;

    const INT nb  = A->nb;
    const INT nb2 = nb * nb;

    B_marker = (INT*)fasp_mem_calloc(ncols_B, sizeof(INT));

    // check A and B see if there are compatible for multiplication
    if ((A->COL != B->ROW) && (A->nb != B->nb) && (D->row / nb2 != B->ROW)) {
        printf("### ERROR: Matrix sizes do not match!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    C->ROW            = A->ROW;
    C->COL            = B->COL;
    C->nb             = A->nb;
    C->storage_manner = A->storage_manner;

    C->val = NULL;
    C->JA  = NULL;
    C->IA  = (INT*)fasp_mem_calloc(C->ROW + 1, sizeof(INT));

    /* initialize the marker array */
    fasp_iarray_set(ncols_B, B_marker, -1);

    /* step 1: obtain the nonzero-structure of C */
    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb = B_j[ib];
                if (B_marker[jb] != ic) {
                    B_marker[jb] = ic;
                    num_nonzeros++;
                }
            }
        }
        C->IA[ic + 1] = num_nonzeros;
    }

    C->NNZ = num_nonzeros;
    C->JA  = (INT*)fasp_mem_calloc(num_nonzeros, sizeof(INT));
    C->val = (REAL*)fasp_mem_calloc(num_nonzeros * nb2, sizeof(REAL));

    /* initialize the marker array again */
    fasp_iarray_set(ncols_B, B_marker, -1);

    REAL* temp  = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));
    REAL* temp1 = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));

    /* step 2: fill in the nonzero entries of C */
    counter = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        row_start = C->IA[ic];
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja      = A_j[ia];
            a_entry = A_data + ia * nb2;
            d_entry = D->val + ja * nb2; // d
            // temp1 = a_entry * d_entry
            fasp_blas_smat_mul(a_entry, d_entry, temp1, nb);
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb      = B_j[ib];
                b_entry = B_data + ib * nb2;
                // temp = (a_entry * d_entry) * b_entry = temp1 * b_entry
                fasp_blas_smat_mul(temp1, b_entry, temp, nb);
                if (B_marker[jb] < row_start) {
                    B_marker[jb]        = counter;
                    C->JA[B_marker[jb]] = jb;
                    // C->val[B_marker[jb]*nb2] = a_entry * b_entry;
                    memcpy(C->val + B_marker[jb] * nb2, temp, nb2 * sizeof(REAL));
                    counter++;
                } else {
                    // C->val[B_marker[jb]*nb2] += a_entry * b_entry;
                    fasp_blas_darray_axpy(nb2, 1.0, temp, C->val + B_marker[jb] * nb2);
                }
            }
        }
    }

    fasp_mem_free(temp);
    temp = NULL;
    fasp_mem_free(temp1);
    temp1 = NULL;
    fasp_mem_free(B_marker);
    B_marker = NULL;
}

/**
 * \fn void fasp_blas_dbsr_schur(dvector *D2, const dBSRmat* A, dvector *D1, const
 * dBSRmat* B, dBSRmat* C)
 *
 * \brief Sparse matrix multiplication C=D2-A*D1*B, where D1 and D2 are diagnal matrices
 *
 * \param A   Pointer to the dBSRmat matrix A
 * \param D   Pointer to the block diagnal matrix D
 * \param B   Pointer to the dBSRmat matrix B
 * \param C   Pointer to dBSRmat matrix equal to D2-A*D1*B
 *
 * \author Li Zhao
 * \date   07/02/2023
 *
 */
void fasp_blas_dbsr_schur(
    dvector* D2, const dBSRmat* A, dvector* D1, const dBSRmat* B, dBSRmat* C)
{
    REAL* A_data  = A->val;
    INT*  A_i     = A->IA;
    INT*  A_j     = A->JA;
    INT   nrows_A = A->ROW;
    INT   ncols_A = A->COL;

    REAL* B_data  = B->val;
    INT*  B_i     = B->IA;
    INT*  B_j     = B->JA;
    INT   nrows_B = B->ROW;
    INT   ncols_B = B->COL;

    INT   ia, ib, ic, ja, jb, i;
    INT   num_nonzeros = 0;
    INT   row_start, counter;
    REAL *a_entry, *b_entry, *d1_entry, *d2_entry;
    INT*  B_marker = NULL;

    const INT nb  = A->nb;
    const INT nb2 = nb * nb;

    B_marker = (INT*)fasp_mem_calloc(ncols_B, sizeof(INT));

    // check A and B see if there are compatible for multiplication
    if ((A->COL != B->ROW) && (A->nb != B->nb) && (D1->row / nb2 != B->ROW)) {
        printf("### ERROR: Matrix sizes do not match!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    C->ROW            = A->ROW;
    C->COL            = B->COL;
    C->nb             = A->nb;
    C->storage_manner = A->storage_manner;

    C->val = NULL;
    C->JA  = NULL;
    C->IA  = (INT*)fasp_mem_calloc(C->ROW + 1, sizeof(INT));

    /* initialize the marker array */
    fasp_iarray_set(ncols_B, B_marker, -1);

    /* step 1: obtain the nonzero-structure of C */
    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb = B_j[ib];
                if (B_marker[jb] != ic) {
                    B_marker[jb] = ic;
                    num_nonzeros++;
                }
            }
        }
        C->IA[ic + 1] = num_nonzeros;
    }

    C->NNZ = num_nonzeros;
    C->JA  = (INT*)fasp_mem_calloc(num_nonzeros, sizeof(INT));
    C->val = (REAL*)fasp_mem_calloc(num_nonzeros * nb2, sizeof(REAL));

    /* initialize the marker array again */
    fasp_iarray_set(ncols_B, B_marker, -1);

    REAL* temp  = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));
    REAL* temp1 = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));
    // REAL* d1_entry_tmp = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));

    dvector D1_neg = fasp_dvec_create(D1->row);
    for (i = 0; i < D1->row; i++) {
        D1_neg.val[i] = -D1->val[i];
    }

    /* step 2: fill in the nonzero entries of C */
    counter = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        row_start = C->IA[ic];
        d2_entry  = D2->val + ic * nb2; // d
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja       = A_j[ia];
            a_entry  = A_data + ia * nb2;
            d1_entry = D1_neg.val + ja * nb2; // - d1_entry
            // temp1 = - a_entry * d1_entry
            fasp_blas_smat_mul(a_entry, d1_entry, temp1, nb);
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb      = B_j[ib];
                b_entry = B_data + ib * nb2;
                // temp = - (a_entry * d1_entry) * b_entry = temp1 * b_entry
                fasp_blas_smat_mul(temp1, b_entry, temp, nb);
                if (B_marker[jb] < row_start) {
                    B_marker[jb]        = counter;
                    C->JA[B_marker[jb]] = jb;
                    if (ic == jb) { // diag entry
                        // C->val[B_marker[jb] * nb2] = d2_entry - (a_entry * d1_entry)
                        // * b_entry;
                        fasp_blas_smat_add(d2_entry, temp, nb, 1.0, 1.0,
                                           C->val + B_marker[jb] * nb2);
                    } else { // nondiag entries
                        // C->val[B_marker[jb] * nb2] = -(a_entry * d1_entry) * b_entry;
                        memcpy(C->val + B_marker[jb] * nb2, temp, nb2 * sizeof(REAL));
                    }
                    counter++;
                } else {
                    // C->val[B_marker[jb]*nb2] += - (a_entry * d1_entry) * b_entry;
                    fasp_blas_darray_axpy(nb2, 1.0, temp, C->val + B_marker[jb] * nb2);
                }
            }
        }
    }

    fasp_mem_free(temp);
    temp = NULL;
    fasp_mem_free(temp1);
    temp1 = NULL;
    fasp_mem_free(B_marker);
    B_marker = NULL;
    fasp_dvec_free(&D1_neg);
}

/**
 * \fn void fasp_blas_dbsr_rap1 (const dBSRmat *R, const dBSRmat *A,
 *                               const dBSRmat *P, dBSRmat *B)
 *
 * \brief dBSRmat sparse matrix multiplication B=R*A*P
 *
 * \param R   Pointer to the dBSRmat matrix
 * \param A   Pointer to the dBSRmat matrix
 * \param P   Pointer to the dBSRmat matrix
 * \param B   Pointer to dBSRmat matrix equal to R*A*P (output)
 *
 * \author Chunsheng Feng, Xiaoqiang Yue and Xiaozhe Hu
 * \date   08/08/2011
 *
 * \note Ref. R.E. Bank and C.C. Douglas. SMMP: Sparse Matrix Multiplication
 * Package. Advances in Computational Mathematics, 1 (1993), pp. 127-137.
 */
void fasp_blas_dbsr_rap1(const dBSRmat* R,
                         const dBSRmat* A,
                         const dBSRmat* P,
                         dBSRmat*       B)
{
    const INT row = R->ROW, col = P->COL, nb = A->nb, nb2 = A->nb * A->nb;

    const REAL *rj = R->val, *aj = A->val, *pj = P->val;
    const INT * ir = R->IA, *ia = A->IA, *ip = P->IA;
    const INT * jr = R->JA, *ja = A->JA, *jp = P->JA;

    REAL* acj;
    INT * iac, *jac;

    INT nB = A->NNZ;
    INT i, i1, j, jj, k, length;
    INT begin_row, end_row, begin_rowA, end_rowA, begin_rowR, end_rowR;
    INT istart, iistart, count;

    INT* index = (INT*)fasp_mem_calloc(A->COL, sizeof(INT));

    REAL* smat_tmp = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));

    INT* iindex = (INT*)fasp_mem_calloc(col, sizeof(INT));

    for (i = 0; i < A->COL; ++i) index[i] = -2;

    memcpy(iindex, index, col * sizeof(INT));

    jac = (INT*)fasp_mem_calloc(nB, sizeof(INT));

    iac = (INT*)fasp_mem_calloc(row + 1, sizeof(INT));

    REAL* temp = (REAL*)fasp_mem_calloc(A->COL * nb2, sizeof(REAL));

    iac[0] = 0;

    // First loop: form sparsity pattern of R*A*P
    for (i = 0; i < row; ++i) {
        // reset istart and length at the beginning of each loop
        istart = -1;
        length = 0;
        i1     = i + 1;

        // go across the rows in R
        begin_rowR = ir[i];
        end_rowR   = ir[i1];
        for (jj = begin_rowR; jj < end_rowR; ++jj) {
            j = jr[jj];
            // for each column in A
            begin_rowA = ia[j];
            end_rowA   = ia[j + 1];
            for (k = begin_rowA; k < end_rowA; ++k) {
                if (index[ja[k]] == -2) {
                    index[ja[k]] = istart;
                    istart       = ja[k];
                    ++length;
                }
            }
        }

        // book-keeping [resetting length and setting iistart]
        count   = length;
        iistart = -1;
        length  = 0;

        // use each column that would have resulted from R*A
        for (j = 0; j < count; ++j) {
            jj        = istart;
            istart    = index[istart];
            index[jj] = -2;

            // go across the row of P
            begin_row = ip[jj];
            end_row   = ip[jj + 1];
            for (k = begin_row; k < end_row; ++k) {
                // pull out the appropriate columns of P
                if (iindex[jp[k]] == -2) {
                    iindex[jp[k]] = iistart;
                    iistart       = jp[k];
                    ++length;
                }
            } // end for k
        }     // end for j

        // set B->IA
        iac[i1] = iac[i] + length;

        if (iac[i1] > nB) {
            nB  = nB * 2;
            jac = (INT*)fasp_mem_realloc(jac, nB * sizeof(INT));
        }

        // put the correct columns of p into the column list of the products
        begin_row = iac[i];
        end_row   = iac[i1];
        for (j = begin_row; j < end_row; ++j) {
            // put the value in B->JA
            jac[j] = iistart;
            // set istart to the next value
            iistart = iindex[iistart];
            // set the iindex spot to 0
            iindex[jac[j]] = -2;
        } // end j
    }     // end i: First loop

    jac = (INT*)fasp_mem_realloc(jac, (iac[row]) * sizeof(INT));

    acj = (REAL*)fasp_mem_calloc(iac[row] * nb2, sizeof(REAL));

    INT* BTindex = (INT*)fasp_mem_calloc(col, sizeof(INT));

    // Second loop: compute entries of R*A*P
    for (i = 0; i < row; ++i) {
        i1 = i + 1;
        // each col of B
        begin_row = iac[i];
        end_row   = iac[i1];
        for (j = begin_row; j < end_row; ++j) {
            BTindex[jac[j]] = j;
        }
        // reset istart and length at the beginning of each loop
        istart = -1;
        length = 0;

        // go across the rows in R
        begin_rowR = ir[i];
        end_rowR   = ir[i1];
        for (jj = begin_rowR; jj < end_rowR; ++jj) {
            j = jr[jj];
            // for each column in A
            begin_rowA = ia[j];
            end_rowA   = ia[j + 1];
            for (k = begin_rowA; k < end_rowA; ++k) {
                if (index[ja[k]] == -2) {
                    index[ja[k]] = istart;
                    istart       = ja[k];
                    ++length;
                }
                fasp_blas_smat_mul(&rj[jj * nb2], &aj[k * nb2], smat_tmp, nb);
                // fasp_darray_xpy(nb2,&temp[ja[k]*nb2], smat_tmp );
                fasp_blas_darray_axpy(nb2, 1.0, smat_tmp, &temp[ja[k] * nb2]);

                // temp[ja[k]]+=rj[jj]*aj[k];
                //  change to   X = X+Y*Z
            }
        }

        // book-keeping [resetting length and setting iistart]
        // use each column that would have resulted from R*A
        for (j = 0; j < length; ++j) {
            jj        = istart;
            istart    = index[istart];
            index[jj] = -2;

            // go across the row of P
            begin_row = ip[jj];
            end_row   = ip[jj + 1];
            for (k = begin_row; k < end_row; ++k) {
                // pull out the appropriate columns of P
                // acj[BTindex[jp[k]]]+=temp[jj]*pj[k];
                fasp_blas_smat_mul(&temp[jj * nb2], &pj[k * nb2], smat_tmp, nb);
                // fasp_darray_xpy(nb2,&acj[BTindex[jp[k]]*nb2], smat_tmp );
                fasp_blas_darray_axpy(nb2, 1.0, smat_tmp, &acj[BTindex[jp[k]] * nb2]);

                // change to   X = X+Y*Z
            }
            // temp[jj]=0.0; // change to   X[nb,nb] = 0;
            fasp_darray_set(nb2, &temp[jj * nb2], 0.0);
        }
    } // end for i: Second loop
    // setup coarse matrix B
    B->ROW = row;
    B->COL = col;
    B->IA  = iac;
    B->JA  = jac;
    B->val = acj;
    B->NNZ = B->IA[B->ROW] - B->IA[0];

    B->nb             = A->nb;
    B->storage_manner = A->storage_manner;

    fasp_mem_free(temp);
    temp = NULL;
    fasp_mem_free(index);
    index = NULL;
    fasp_mem_free(iindex);
    iindex = NULL;
    fasp_mem_free(BTindex);
    BTindex = NULL;
    fasp_mem_free(smat_tmp);
    smat_tmp = NULL;
}

/**
 * \fn void fasp_blas_dbsr_rap (const dBSRmat *R, const dBSRmat *A,
 *                              const dBSRmat *P, dBSRmat *B)
 *
 * \brief dBSRmat sparse matrix multiplication B=R*A*P
 *
 * \param R   Pointer to the dBSRmat matrix
 * \param A   Pointer to the dBSRmat matrix
 * \param P   Pointer to the dBSRmat matrix
 * \param B   Pointer to dBSRmat matrix equal to R*A*P (output)
 *
 * \author Xiaozhe Hu, Chunsheng Feng, Zheng Li
 * \date   10/24/2012
 *
 * \note Ref. R.E. Bank and C.C. Douglas. SMMP: Sparse Matrix Multiplication
 * Package. Advances in Computational Mathematics, 1 (1993), pp. 127-137.
 */
void fasp_blas_dbsr_rap(const dBSRmat* R,
                        const dBSRmat* A,
                        const dBSRmat* P,
                        dBSRmat*       B)
{
    const INT row = R->ROW, col = P->COL, nb = A->nb, nb2 = A->nb * A->nb;

    const REAL *rj = R->val, *aj = A->val, *pj = P->val;
    const INT * ir = R->IA, *ia = A->IA, *ip = P->IA;
    const INT * jr = R->JA, *ja = A->JA, *jp = P->JA;

    REAL* acj;
    INT * iac, *jac;

    INT* Ps_marker = NULL;
    INT* As_marker = NULL;

#ifdef _OPENMP
    INT*  P_marker = NULL;
    INT*  A_marker = NULL;
    REAL* smat_tmp = NULL;
#endif

    INT i, i1, i2, i3, jj1, jj2, jj3;
    INT counter, jj_row_begining;

    INT nthreads = 1;

#ifdef _OPENMP
    INT myid, mybegin, myend, Ctemp;
    nthreads = fasp_get_num_threads();
#endif

    INT n_coarse            = row;
    INT n_fine              = A->ROW;
    INT coarse_mul_nthreads = n_coarse * nthreads;
    INT fine_mul_nthreads   = n_fine * nthreads;
    INT coarse_add_nthreads = n_coarse + nthreads;
    INT minus_one_length    = coarse_mul_nthreads + fine_mul_nthreads;
    INT total_calloc        = minus_one_length + coarse_add_nthreads + nthreads;

    Ps_marker = (INT*)fasp_mem_calloc(total_calloc, sizeof(INT));
    As_marker = Ps_marker + coarse_mul_nthreads;

    /*------------------------------------------------------*
     *  First Pass: Determine size of B and set up B_i  *
     *------------------------------------------------------*/
    iac = (INT*)fasp_mem_calloc(n_coarse + 1, sizeof(INT));

    fasp_iarray_set(minus_one_length, Ps_marker, -1);

    REAL* tmp = (REAL*)fasp_mem_calloc(2 * nthreads * nb2, sizeof(REAL));

#ifdef _OPENMP
    INT* RAP_temp = As_marker + fine_mul_nthreads;
    INT* part_end = RAP_temp + coarse_add_nthreads;

    if (n_coarse > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, Ctemp, P_marker, A_marker,      \
                                     counter, i, jj_row_begining, jj1, i1, jj2, i2,    \
                                     jj3, i3)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n_coarse, &mybegin, &myend);
            P_marker = Ps_marker + myid * n_coarse;
            A_marker = As_marker + myid * n_fine;
            counter  = 0;
            for (i = mybegin; i < myend; ++i) {
                P_marker[i]     = counter;
                jj_row_begining = counter;
                counter++;
                for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                    i1 = jr[jj1];
                    for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                        i2 = ja[jj2];
                        if (A_marker[i2] != i) {
                            A_marker[i2] = i;
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                                i3 = jp[jj3];
                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = counter;
                                    counter++;
                                }
                            }
                        }
                    }
                }
                RAP_temp[i + myid] = jj_row_begining;
            }
            RAP_temp[myend + myid] = counter;
            part_end[myid]         = myend + myid + 1;
        }
        fasp_iarray_cp(part_end[0], RAP_temp, iac);
        counter = part_end[0];
        Ctemp   = 0;
        for (i1 = 1; i1 < nthreads; i1++) {
            Ctemp += RAP_temp[part_end[i1 - 1] - 1];
            for (jj1 = part_end[i1 - 1] + 1; jj1 < part_end[i1]; jj1++) {
                iac[counter] = RAP_temp[jj1] + Ctemp;
                counter++;
            }
        }
    } else {
#endif
        counter = 0;
        for (i = 0; i < row; ++i) {
            Ps_marker[i]    = counter;
            jj_row_begining = counter;
            counter++;

            for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                i1 = jr[jj1];
                for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                    i2 = ja[jj2];
                    if (As_marker[i2] != i) {
                        As_marker[i2] = i;
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                            i3 = jp[jj3];
                            if (Ps_marker[i3] < jj_row_begining) {
                                Ps_marker[i3] = counter;
                                counter++;
                            }
                        }
                    }
                }
            }
            iac[i] = jj_row_begining;
        }
#ifdef _OPENMP
    }
#endif

    iac[row] = counter;

    jac = (INT*)fasp_mem_calloc(iac[row], sizeof(INT));

    acj = (REAL*)fasp_mem_calloc(iac[row] * nb2, sizeof(REAL));

    fasp_iarray_set(minus_one_length, Ps_marker, -1);

    /*------------------------------------------------------*
     *  Second Pass: compute entries of B=R*A*P             *
     *------------------------------------------------------*/
#ifdef _OPENMP
    if (n_coarse > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, Ctemp, P_marker, A_marker,      \
                                     counter, i, jj_row_begining, jj1, i1, jj2, i2,    \
                                     jj3, i3, smat_tmp)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n_coarse, &mybegin, &myend);
            P_marker = Ps_marker + myid * n_coarse;
            A_marker = As_marker + myid * n_fine;
            smat_tmp = tmp + myid * 2 * nb2;
            counter  = iac[mybegin];
            for (i = mybegin; i < myend; ++i) {
                P_marker[i]     = counter;
                jj_row_begining = counter;
                jac[counter]    = i;
                fasp_darray_set(nb2, &acj[counter * nb2], 0x0);
                counter++;

                for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                    i1 = jr[jj1];
                    for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                        fasp_blas_smat_mul(&rj[jj1 * nb2], &aj[jj2 * nb2], smat_tmp,
                                           nb);
                        i2 = ja[jj2];
                        if (A_marker[i2] != i) {
                            A_marker[i2] = i;
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                                i3 = jp[jj3];
                                fasp_blas_smat_mul(smat_tmp, &pj[jj3 * nb2],
                                                   smat_tmp + nb2, nb);
                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = counter;
                                    fasp_darray_cp(nb2, smat_tmp + nb2,
                                                   &acj[counter * nb2]);
                                    jac[counter] = i3;
                                    counter++;
                                } else {
                                    fasp_blas_darray_axpy(nb2, 1.0, smat_tmp + nb2,
                                                          &acj[P_marker[i3] * nb2]);
                                }
                            }
                        } else {
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; jj3++) {
                                i3 = jp[jj3];
                                fasp_blas_smat_mul(smat_tmp, &pj[jj3 * nb2],
                                                   smat_tmp + nb2, nb);
                                fasp_blas_darray_axpy(nb2, 1.0, smat_tmp + nb2,
                                                      &acj[P_marker[i3] * nb2]);
                            }
                        }
                    }
                }
            }
        }
    } else {
#endif
        counter = 0;
        for (i = 0; i < row; ++i) {
            Ps_marker[i]    = counter;
            jj_row_begining = counter;
            jac[counter]    = i;
            fasp_darray_set(nb2, &acj[counter * nb2], 0x0);
            counter++;

            for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                i1 = jr[jj1];
                for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                    fasp_blas_smat_mul(&rj[jj1 * nb2], &aj[jj2 * nb2], tmp, nb);
                    i2 = ja[jj2];
                    if (As_marker[i2] != i) {
                        As_marker[i2] = i;
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                            i3 = jp[jj3];
                            fasp_blas_smat_mul(tmp, &pj[jj3 * nb2], tmp + nb2, nb);
                            if (Ps_marker[i3] < jj_row_begining) {
                                Ps_marker[i3] = counter;
                                fasp_darray_cp(nb2, tmp + nb2, &acj[counter * nb2]);
                                jac[counter] = i3;
                                counter++;
                            } else {
                                fasp_blas_darray_axpy(nb2, 1.0, tmp + nb2,
                                                      &acj[Ps_marker[i3] * nb2]);
                            }
                        }
                    } else {
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; jj3++) {
                            i3 = jp[jj3];
                            fasp_blas_smat_mul(tmp, &pj[jj3 * nb2], tmp + nb2, nb);
                            fasp_blas_darray_axpy(nb2, 1.0, tmp + nb2,
                                                  &acj[Ps_marker[i3] * nb2]);
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
    }
#endif
    // setup coarse matrix B
    B->ROW            = row;
    B->COL            = col;
    B->IA             = iac;
    B->JA             = jac;
    B->val            = acj;
    B->NNZ            = B->IA[B->ROW] - B->IA[0];
    B->nb             = A->nb;
    B->storage_manner = A->storage_manner;

    fasp_mem_free(Ps_marker);
    Ps_marker = NULL;
    fasp_mem_free(tmp);
    tmp = NULL;
}

/**
 * \fn void fasp_blas_dbsr_rap_agg (const dBSRmat *R, const dBSRmat *A,
 *                                  const dBSRmat *P, dBSRmat *B)
 *
 * \brief dBSRmat sparse matrix multiplication B=R*A*P, where small block matrices
 * in P and R are identity matrices!
 *
 * \param R   Pointer to the dBSRmat matrix
 * \param A   Pointer to the dBSRmat matrix
 * \param P   Pointer to the dBSRmat matrix
 * \param B   Pointer to dBSRmat matrix equal to R*A*P (output)
 *
 * \author Xiaozhe Hu
 * \date   10/24/2012
 *
 * \note Bugs for OpenMP modified by Li Zhao, 2023.06.17
 */
void fasp_blas_dbsr_rap_agg(const dBSRmat* R,
                            const dBSRmat* A,
                            const dBSRmat* P,
                            dBSRmat*       B)
{
    const INT row = R->ROW, col = P->COL, nb2 = A->nb * A->nb;

    const REAL* aj = A->val;
    const INT * ir = R->IA, *ia = A->IA, *ip = P->IA;
    const INT * jr = R->JA, *ja = A->JA, *jp = P->JA;

    INT * iac, *jac;
    REAL* acj;
    INT*  Ps_marker = NULL;
    INT*  As_marker = NULL;

#ifdef _OPENMP
    INT* P_marker = NULL;
    INT* A_marker = NULL;
#endif

    INT i, i1, i2, i3, jj1, jj2, jj3;
    INT counter, jj_row_begining;
    INT RAP_nnz; // for OpenMP, Li Zhao, 2023.06.17

    INT nthreads = 1;

#ifdef _OPENMP
    INT myid, mybegin, myend, Ctemp;
    nthreads = fasp_get_num_threads();
#endif

    INT n_coarse            = row;
    INT n_fine              = A->ROW;
    INT coarse_mul_nthreads = n_coarse * nthreads;
    INT fine_mul_nthreads   = n_fine * nthreads;
    INT coarse_add_nthreads = n_coarse + nthreads;
    INT minus_one_length    = coarse_mul_nthreads + fine_mul_nthreads;
    INT total_calloc        = minus_one_length + coarse_add_nthreads + nthreads;

    Ps_marker = (INT*)fasp_mem_calloc(total_calloc, sizeof(INT));
    As_marker = Ps_marker + coarse_mul_nthreads;

    /*------------------------------------------------------*
     *  First Pass: Determine size of B and set up B_i  *
     *------------------------------------------------------*/
    iac = (INT*)fasp_mem_calloc(n_coarse + 1, sizeof(INT));

    fasp_iarray_set(minus_one_length, Ps_marker, -1);

#ifdef _OPENMP
    INT* RAP_temp = As_marker + fine_mul_nthreads;
    INT* part_end = RAP_temp + coarse_add_nthreads;

    if (n_coarse > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, Ctemp, P_marker, A_marker,      \
                                     counter, i, jj_row_begining, jj1, i1, jj2, i2,    \
                                     jj3, i3)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n_coarse, &mybegin, &myend);
            P_marker = Ps_marker + myid * n_coarse;
            A_marker = As_marker + myid * n_fine;
            counter  = 0;
            for (i = mybegin; i < myend; ++i) {
                P_marker[i]     = counter;
                jj_row_begining = counter;
                counter++;
                for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                    i1 = jr[jj1];
                    for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                        i2 = ja[jj2];
                        if (A_marker[i2] != i) {
                            A_marker[i2] = i;
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                                i3 = jp[jj3];
                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = counter;
                                    counter++;
                                }
                            }
                        }
                    }
                }
                RAP_temp[i + myid] = jj_row_begining;
            }
            RAP_temp[myend + myid] = counter;
            part_end[myid]         = myend + myid + 1;
        }
        fasp_iarray_cp(part_end[0], RAP_temp, iac);
        counter = part_end[0];
        Ctemp   = 0;
        for (i1 = 1; i1 < nthreads; i1++) {
            Ctemp += RAP_temp[part_end[i1 - 1] - 1];
            for (jj1 = part_end[i1 - 1] + 1; jj1 < part_end[i1]; jj1++) {
                iac[counter] = RAP_temp[jj1] + Ctemp;
                counter++;
            }
        }
        RAP_nnz = iac[row]; // for OpenMP, Li Zhao, 2023.06.17
    } else {
#endif
        counter = 0;
        for (i = 0; i < row; ++i) {
            Ps_marker[i]    = counter;
            jj_row_begining = counter;
            counter++;

            for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                i1 = jr[jj1];
                for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {
                    i2 = ja[jj2];
                    if (As_marker[i2] != i) {
                        As_marker[i2] = i;
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                            i3 = jp[jj3];
                            if (Ps_marker[i3] < jj_row_begining) {
                                Ps_marker[i3] = counter;
                                counter++;
                            }
                        }
                    }
                }
            }
            iac[i] = jj_row_begining;
        }

        iac[row] = counter;
        RAP_nnz  = counter; // for OpenMP, Li Zhao, 2023.06.17
#ifdef _OPENMP
    }
#endif

    // iac[row] = counter; // this case only for sequetial

    jac = (INT*)fasp_mem_calloc(RAP_nnz, sizeof(INT));

    acj = (REAL*)fasp_mem_calloc(RAP_nnz * nb2, sizeof(REAL));

    fasp_iarray_set(minus_one_length, Ps_marker, -1);

    /*------------------------------------------------------*
     *  Second Pass: compute entries of B=R*A*P             *
     *------------------------------------------------------*/
#ifdef _OPENMP
    if (n_coarse > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, P_marker, A_marker, counter, i, \
                                     jj_row_begining, jj1, i1, jj2, i2, jj3, i3)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n_coarse, &mybegin, &myend);
            P_marker = Ps_marker + myid * n_coarse;
            A_marker = As_marker + myid * n_fine;
            counter  = iac[mybegin];
            for (i = mybegin; i < myend; ++i) {
                P_marker[i]     = counter;
                jj_row_begining = counter;
                jac[counter]    = i;
                fasp_darray_set(nb2, &acj[counter * nb2], 0x0);
                // printf("i=%d nb2=%d counter=%d s\n", i, nb2, counter); // zhaoli
                counter++;

                for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                    i1 = jr[jj1];
                    for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {

                        i2 = ja[jj2];
                        if (A_marker[i2] != i) {
                            A_marker[i2] = i;
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                                i3 = jp[jj3];

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = counter;
                                    fasp_darray_cp(nb2, &aj[jj2 * nb2],
                                                   &acj[counter * nb2]);
                                    jac[counter] = i3;
                                    counter++;
                                } else {
                                    fasp_blas_darray_axpy(nb2, 1.0, &aj[jj2 * nb2],
                                                          &acj[P_marker[i3] * nb2]);
                                }
                            }
                        } else {
                            for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; jj3++) {
                                i3 = jp[jj3];
                                fasp_blas_darray_axpy(nb2, 1.0, &aj[jj2 * nb2],
                                                      &acj[P_marker[i3] * nb2]);
                            }
                        }
                    }
                }
            }
        }
    } else {
#endif
        counter = 0;
        for (i = 0; i < row; ++i) {
            Ps_marker[i]    = counter;
            jj_row_begining = counter;
            jac[counter]    = i;
            fasp_darray_set(nb2, &acj[counter * nb2], 0x0);
            counter++;

            for (jj1 = ir[i]; jj1 < ir[i + 1]; ++jj1) {
                i1 = jr[jj1];
                for (jj2 = ia[i1]; jj2 < ia[i1 + 1]; ++jj2) {

                    i2 = ja[jj2];
                    if (As_marker[i2] != i) {
                        As_marker[i2] = i;
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; ++jj3) {
                            i3 = jp[jj3];
                            if (Ps_marker[i3] < jj_row_begining) {
                                Ps_marker[i3] = counter;
                                fasp_darray_cp(nb2, &aj[jj2 * nb2],
                                               &acj[counter * nb2]);
                                jac[counter] = i3;
                                counter++;
                            } else {
                                fasp_blas_darray_axpy(nb2, 1.0, &aj[jj2 * nb2],
                                                      &acj[Ps_marker[i3] * nb2]);
                            }
                        }
                    } else {
                        for (jj3 = ip[i2]; jj3 < ip[i2 + 1]; jj3++) {
                            i3 = jp[jj3];
                            fasp_blas_darray_axpy(nb2, 1.0, &aj[jj2 * nb2],
                                                  &acj[Ps_marker[i3] * nb2]);
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    // setup coarse matrix B
    B->ROW            = row;
    B->COL            = col;
    B->IA             = iac;
    B->JA             = jac;
    B->val            = acj;
    B->NNZ            = B->IA[B->ROW] - B->IA[0];
    B->nb             = A->nb;
    B->storage_manner = A->storage_manner;

    fasp_mem_free(Ps_marker);
    Ps_marker = NULL;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
