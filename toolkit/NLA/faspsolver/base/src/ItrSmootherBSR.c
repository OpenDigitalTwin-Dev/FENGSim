/*! \file  ItrSmootherBSR.c
 *
 *  \brief Smoothers for dBSRmat matrices
 *
 *  \note  This file contains Level-2 (Itr) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxThreads.c, AuxTiming.c,
 *         BlaSmallMatInv.c, BlaSmallMat.c, BlaArray.c, BlaSpmvBSR.c, and PreBSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  // TODO: Need to optimize routines here! --Chensong
 */

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#ifdef _OPENMP

#if ILU_MC_OMP
static inline void perm(const INT, const INT, const REAL*, const INT*, REAL*);
static inline void invperm(const INT, const INT, const REAL*, const INT*, REAL*);
#endif

#endif

REAL ilu_solve_time = 0.0; /**< ILU time for the SOLVE phase */

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_smoother_dbsr_jacobi (dBSRmat *A, dvector *b, dvector *u)
 *
 * \brief Jacobi relaxation
 *
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/02/2012
 */
void fasp_smoother_dbsr_jacobi(dBSRmat* A, dvector* b, dvector* u)
{
    // members of A
    const INT   ROW  = A->ROW;
    const INT   nb   = A->nb;
    const INT   nb2  = nb * nb;
    const INT   size = ROW * nb2;
    const INT*  IA   = A->IA;
    const INT*  JA   = A->JA;
    const REAL* val  = A->val;

    // local variables
    INT   i, k;
    SHORT nthreads = 1, use_openmp = FALSE;
    REAL* diaginv = (REAL*)fasp_mem_calloc(size, sizeof(REAL));

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // get all the diagonal sub-blocks
    if (use_openmp) {
        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                for (k = IA[i]; k < IA[i + 1]; ++k)
                    if (JA[k] == i)
                        memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    } else {
        for (i = 0; i < ROW; ++i) {
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                if (JA[k] == i)
                    memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    }

    // compute the inverses of all the diagonal sub-blocks
    if (nb > 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    fasp_smat_inv(diaginv + i * nb2, nb);
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                fasp_smat_inv(diaginv + i * nb2, nb);
            }
        }
    } else {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    diaginv[i] = 1.0 / diaginv[i];
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                // zero-diagonal should be tested previously
                diaginv[i] = 1.0 / diaginv[i];
            }
        }
    }

    fasp_smoother_dbsr_jacobi1(A, b, u, diaginv);

    fasp_mem_free(diaginv);
    diaginv = NULL;
}

/**
 * \fn void fasp_smoother_dbsr_jacobi_setup (dBSRmat *A, REAL *diaginv)
 *
 * \brief Setup for jacobi relaxation, fetch the diagonal sub-block matrixes and
 *        make them inverse first
 *
 * \param A       Pointer to dBSRmat: the coefficient matrix
 * \param diaginv Inverse of the diagonal entries
 *
 * \author Zhiyang Zhou
 * \date   10/25/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/02/2012
 */
void fasp_smoother_dbsr_jacobi_setup(dBSRmat* A, REAL* diaginv)
{
    // members of A
    const INT   ROW = A->ROW;
    const INT   nb  = A->nb;
    const INT   nb2 = nb * nb;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;
    const REAL* val = A->val;

    // local variables
    INT i, k;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // get all the diagonal sub-blocks
    if (use_openmp) {
        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                for (k = IA[i]; k < IA[i + 1]; ++k)
                    if (JA[k] == i)
                        memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    } else {
        for (i = 0; i < ROW; ++i) {
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                if (JA[k] == i)
                    memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    }

    // compute the inverses of all the diagonal sub-blocks
    if (nb > 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    fasp_smat_inv(diaginv + i * nb2, nb);
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                fasp_smat_inv(diaginv + i * nb2, nb);
            }
        }
    } else {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    diaginv[i] = 1.0 / diaginv[i];
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                // zero-diagonal should be tested previously
                diaginv[i] = 1.0 / diaginv[i];
            }
        }
    }
}

/**
 * \fn void fasp_smoother_dbsr_jacobi1 (dBSRmat *A, dvector *b, dvector *u,
 *                                      REAL *diaginv)
 *
 * \brief Jacobi relaxation
 *
 * \param A        Pointer to dBSRmat: the coefficient matrix
 * \param b        Pointer to dvector: the right hand side
 * \param u        Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/03/2012
 */
void fasp_smoother_dbsr_jacobi1(dBSRmat* A, dvector* b, dvector* u, REAL* diaginv)
{
    // members of A
    const INT  ROW  = A->ROW;
    const INT  nb   = A->nb;
    const INT  nb2  = nb * nb;
    const INT  size = ROW * nb;
    const INT* IA   = A->IA;
    const INT* JA   = A->JA;
    REAL*      val  = A->val;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // auxiliary array
    REAL* b_tmp = NULL;

    // local variables
    INT i, j, k;
    INT pb;

    // b_tmp = b_val
    b_tmp = (REAL*)fasp_mem_calloc(size, sizeof(REAL));
    memcpy(b_tmp, b_val, size * sizeof(REAL));

    // No need to assign the smoothing order since the result doesn't depend on it
    if (nb == 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, j, k)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i) b_tmp[i] -= val[k] * u_val[j];
                    }
                }
            }
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    u_val[i] = b_tmp[i] * diaginv[i];
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i) b_tmp[i] -= val[k] * u_val[j];
                }
            }
            for (i = 0; i < ROW; ++i) {
                u_val[i] = b_tmp[i] * diaginv[i];
            }
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else if (nb > 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, pb, k, j)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    pb = i * nb;
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i)
                            fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb,
                                                b_tmp + pb, nb);
                    }
                }
            }
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, pb)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    pb = i * nb;
                    fasp_blas_smat_mxv(diaginv + nb2 * i, b_tmp + pb, u_val + pb, nb);
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                pb = i * nb;
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i)
                        fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp + pb,
                                            nb);
                }
            }

            for (i = 0; i < ROW; ++i) {
                pb = i * nb;
                fasp_blas_smat_mxv(diaginv + nb2 * i, b_tmp + pb, u_val + pb, nb);
            }
        }
        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        printf("### ERROR: nb is illegal! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs (dBSRmat *A, dvector *b, dvector *u, INT order,
 *                                 INT *mark)
 *
 * \brief Gauss-Seidel relaxation
 *
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param order  Flag to indicate the order for smoothing
 *               If mark = NULL
 *                    ASCEND       12: in ascending order
 *                    DESCEND      21: in descending order
 *               If mark != NULL:  in the user-defined order
 * \param mark   Pointer to NULL or to the user-defined ordering
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/03/2012
 */
void fasp_smoother_dbsr_gs(dBSRmat* A, dvector* b, dvector* u, INT order, INT* mark)
{
    // members of A
    const INT   ROW  = A->ROW;
    const INT   nb   = A->nb;
    const INT   nb2  = nb * nb;
    const INT   size = ROW * nb2;
    const INT*  IA   = A->IA;
    const INT*  JA   = A->JA;
    const REAL* val  = A->val;

    // local variables
    INT   i, k;
    SHORT nthreads = 1, use_openmp = FALSE;
    REAL* diaginv = (REAL*)fasp_mem_calloc(size, sizeof(REAL));

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // get all the diagonal sub-blocks
    if (use_openmp) {
        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                for (k = IA[i]; k < IA[i + 1]; ++k)
                    if (JA[k] == i)
                        memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    } else {
        for (i = 0; i < ROW; ++i) {
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                if (JA[k] == i)
                    memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    }

    // compute the inverses of all the diagonal sub-blocks
    if (nb > 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    fasp_smat_inv(diaginv + i * nb2, nb);
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                fasp_smat_inv(diaginv + i * nb2, nb);
            }
        }
    } else {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    diaginv[i] = 1.0 / diaginv[i];
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                // zero-diagonal should be tested previously
                diaginv[i] = 1.0 / diaginv[i];
            }
        }
    }

    fasp_smoother_dbsr_gs1(A, b, u, order, mark, diaginv);

    fasp_mem_free(diaginv);
    diaginv = NULL;
}

/**
 * \fn void fasp_smoother_dbsr_gs1 (dBSRmat *A, dvector *b, dvector *u, INT order,
 *                                  INT *mark, REAL *diaginv)
 *
 * \brief Gauss-Seidel relaxation
 *
 * \param A        Pointer to dBSRmat: the coefficient matrix
 * \param b        Pointer to dvector: the right hand side
 * \param u        Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param order    Flag to indicate the order for smoothing
 *                 If mark = NULL
 *                    ASCEND       12: in ascending order
 *                    DESCEND      21: in descending order
 *                 If mark != NULL:  in the user-defined order
 * \param mark     Pointer to NULL or to the user-defined ordering
 * \param diaginv  Inverses for all the diagonal blocks of A
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 */
void fasp_smoother_dbsr_gs1(
    dBSRmat* A, dvector* b, dvector* u, INT order, INT* mark, REAL* diaginv)
{
    if (!mark) {
        if (order == ASCEND) // smooth ascendingly
        {
            fasp_smoother_dbsr_gs_ascend(A, b, u, diaginv);
        } else if (order == DESCEND) // smooth descendingly
        {
            fasp_smoother_dbsr_gs_descend(A, b, u, diaginv);
        }
    }
    // smooth according to the order 'mark' defined by user
    else {
        fasp_smoother_dbsr_gs_order1(A, b, u, diaginv, mark);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_ascend (dBSRmat *A, dvector *b, dvector *u,
 *                                        REAL *diaginv)
 *
 * \brief Gauss-Seidel relaxation in the ascending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 */
void fasp_smoother_dbsr_gs_ascend(dBSRmat* A, dvector* b, dvector* u, REAL* diaginv)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (i = 0; i < ROW; ++i) {
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs * diaginv[i];
        }
    } else if (nb > 1) {
        REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));

        for (i = 0; i < ROW; ++i) {
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            fasp_blas_smat_mxv(diaginv + nb2 * i, b_tmp, u_val + pb, nb);
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        printf("### ERROR: nb is illegal! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_ascend1 (dBSRmat *A, dvector *b, dvector *u)
 *
 * \brief Gauss-Seidel relaxation in the ascending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 *
 * \author Xiaozhe Hu
 * \date   01/01/2014
 *
 * \note The only difference between the functions 'fasp_smoother_dbsr_gs_ascend1'
 *       and 'fasp_smoother_dbsr_gs_ascend' is that we don't have to multiply
 *       by the inverses of the diagonal blocks in each ROW since matrix A has
 *       been such scaled that all the diagonal blocks become identity matrices.
 */
void fasp_smoother_dbsr_gs_ascend1(dBSRmat* A, dvector* b, dvector* u)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (i = 0; i < ROW; ++i) {
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs;
        }
    } else if (nb > 1) {
        REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));

        for (i = 0; i < ROW; ++i) {
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            memcpy(u_val + pb, b_tmp, nb * sizeof(REAL));
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        printf("### ERROR: nb is illegal! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_descend (dBSRmat *A, dvector *b, dvector *u,
 *                                         REAL *diaginv)
 *
 * \brief Gauss-Seidel relaxation in the descending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 */
void fasp_smoother_dbsr_gs_descend(dBSRmat* A, dvector* b, dvector* u, REAL* diaginv)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (i = ROW - 1; i >= 0; i--) {
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs * diaginv[i];
        }
    } else if (nb > 1) {
        REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));

        for (i = ROW - 1; i >= 0; i--) {
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            fasp_blas_smat_mxv(diaginv + nb2 * i, b_tmp, u_val + pb, nb);
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        printf("### ERROR: nb is illegal! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_descend1 (dBSRmat *A, dvector *b, dvector *u)
 *
 * \brief Gauss-Seidel relaxation in the descending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 *
 * \author Xiaozhe Hu
 * \date   01/01/2014
 *
 * \note The only difference between the functions 'fasp_smoother_dbsr_gs_ascend1'
 *       and 'fasp_smoother_dbsr_gs_ascend' is that we don't have to multiply
 *       by the inverses of the diagonal blocks in each ROW since matrix A has
 *       been such scaled that all the diagonal blocks become identity matrices.
 *
 */
void fasp_smoother_dbsr_gs_descend1(dBSRmat* A, dvector* b, dvector* u)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (i = ROW - 1; i >= 0; i--) {
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs;
        }
    } else if (nb > 1) {
        REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));

        for (i = ROW - 1; i >= 0; i--) {
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            memcpy(u_val + pb, b_tmp, nb * sizeof(REAL));
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        printf("### ERROR: nb is illegal! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_order1 (dBSRmat *A, dvector *b, dvector *u,
 *                                        REAL *diaginv, INT *mark)
 *
 * \brief Gauss-Seidel relaxation in the user-defined order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 * \param mark     Pointer to the user-defined ordering
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 */
void fasp_smoother_dbsr_gs_order1(
    dBSRmat* A, dvector* b, dvector* u, REAL* diaginv, INT* mark)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  I, pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (I = 0; I < ROW; ++I) {
            i   = mark[I];
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs * diaginv[i];
        }
    } else if (nb > 1) {
        REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));

        for (I = 0; I < ROW; ++I) {
            i  = mark[I];
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            fasp_blas_smat_mxv(diaginv + nb2 * i, b_tmp, u_val + pb, nb);
        }

        fasp_mem_free(b_tmp);
        b_tmp = NULL;
    } else {
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_gs_order2 (dBSRmat *A, dvector *b, dvector *u,
 *                                        INT *mark, REAL *work)
 *
 * \brief Gauss-Seidel relaxation in the user-defined order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param mark   Pointer to the user-defined ordering
 * \param work   Work temp array
 *
 * \author Zhiyang Zhou
 * \date   2010/11/08
 *
 * \note The only difference between the functions 'fasp_smoother_dbsr_gs_order2'
 *       and 'fasp_smoother_dbsr_gs_order1' lies in that we don't have to multiply
 *       by the inverses of the diagonal blocks in each ROW since matrix A has
 *       been such scaled that all the diagonal blocks become identity matrices.
 */
void fasp_smoother_dbsr_gs_order2(
    dBSRmat* A, dvector* b, dvector* u, INT* mark, REAL* work)
{
    // members of A
    const INT  ROW = A->ROW;
    const INT  nb  = A->nb;
    const INT  nb2 = nb * nb;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // auxiliary array
    REAL* b_tmp = work;

    // local variables
    INT  i, j, k, I, pb;
    REAL rhs = 0.0;

    if (nb == 1) {
        for (I = 0; I < ROW; ++I) {
            i   = mark[I];
            rhs = b_val[i];
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i) rhs -= val[k] * u_val[j];
            }
            u_val[i] = rhs;
        }
    } else if (nb > 1) {
        for (I = 0; I < ROW; ++I) {
            i  = mark[I];
            pb = i * nb;
            memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                j = JA[k];
                if (j != i)
                    fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
            }
            memcpy(u_val + pb, b_tmp, nb * sizeof(REAL));
        }
    } else {
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_sor (dBSRmat *A, dvector *b, dvector *u, INT order,
 *                                  INT *mark, REAL weight)
 *
 * \brief SOR relaxation
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param order  Flag to indicate the order for smoothing
 *               If mark = NULL
 *                    ASCEND       12: in ascending order
 *                    DESCEND      21: in descending order
 *               If mark != NULL:  in the user-defined order
 * \param mark   Pointer to NULL or to the user-defined ordering
 * \param weight Over-relaxation weight
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/03/2012
 */
void fasp_smoother_dbsr_sor(
    dBSRmat* A, dvector* b, dvector* u, INT order, INT* mark, REAL weight)
{
    // members of A
    const INT   ROW  = A->ROW;
    const INT   nb   = A->nb;
    const INT   nb2  = nb * nb;
    const INT   size = ROW * nb2;
    const INT*  IA   = A->IA;
    const INT*  JA   = A->JA;
    const REAL* val  = A->val;

    // local variables
    INT   i, k;
    REAL* diaginv = NULL;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // allocate memory
    diaginv = (REAL*)fasp_mem_calloc(size, sizeof(REAL));

    // get all the diagonal sub-blocks
    if (use_openmp) {
        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                for (k = IA[i]; k < IA[i + 1]; ++k)
                    if (JA[k] == i)
                        memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    } else {
        for (i = 0; i < ROW; ++i) {
            for (k = IA[i]; k < IA[i + 1]; ++k) {
                if (JA[k] == i)
                    memcpy(diaginv + i * nb2, val + k * nb2, nb2 * sizeof(REAL));
            }
        }
    }

    // compute the inverses of all the diagonal sub-blocks
    if (nb > 1) {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    fasp_smat_inv(diaginv + i * nb2, nb);
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                fasp_smat_inv(diaginv + i * nb2, nb);
            }
        }
    } else {
        if (use_openmp) {
            INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    diaginv[i] = 1.0 / diaginv[i];
                }
            }
        } else {
            for (i = 0; i < ROW; ++i) {
                // zero-diagonal should be tested previously
                diaginv[i] = 1.0 / diaginv[i];
            }
        }
    }

    fasp_smoother_dbsr_sor1(A, b, u, order, mark, diaginv, weight);

    fasp_mem_free(diaginv);
    diaginv = NULL;
}

/**
 * \fn void fasp_smoother_dbsr_sor1 (dBSRmat *A, dvector *b, dvector *u, INT order,
 *                                   INT *mark, REAL *diaginv, REAL weight)
 *
 * \brief SOR relaxation
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param order   Flag to indicate the order for smoothing
 *                If mark = NULL
 *                    ASCEND       12: in ascending order
 *                    DESCEND      21: in descending order
 *                If mark != NULL:  in the user-defined order
 * \param mark    Pointer to NULL or to the user-defined ordering
 * \param diaginv Inverses for all the diagonal blocks of A
 * \param weight  Over-relaxation weight
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 */
void fasp_smoother_dbsr_sor1(dBSRmat* A,
                             dvector* b,
                             dvector* u,
                             INT      order,
                             INT*     mark,
                             REAL*    diaginv,
                             REAL     weight)
{
    if (!mark) {
        if (order == ASCEND) // smooth ascendingly
        {
            fasp_smoother_dbsr_sor_ascend(A, b, u, diaginv, weight);
        } else if (order == DESCEND) // smooth descendingly
        {
            fasp_smoother_dbsr_sor_descend(A, b, u, diaginv, weight);
        }
    }
    // smooth according to the order 'mark' defined by user
    else {
        fasp_smoother_dbsr_sor_order(A, b, u, diaginv, mark, weight);
    }
}

/**
 * \fn void fasp_smoother_dbsr_sor_ascend (dBSRmat *A, dvector *b, dvector *u,
 *                                         REAL *diaginv, REAL weight)
 *
 * \brief SOR relaxation in the ascending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 * \param weight   Over-relaxation weight
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/09/04
 */
void fasp_smoother_dbsr_sor_ascend(
    dBSRmat* A, dvector* b, dvector* u, REAL* diaginv, REAL weight)
{
    // members of A
    const INT   ROW = A->ROW;
    const INT   nb  = A->nb;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;
    const REAL* val = A->val;

    // values of dvector b and u
    const REAL* b_val = b->val;
    REAL*       u_val = u->val;

    // local variables
    const INT nb2 = nb * nb;
    INT       i, j, k;
    INT       pb;
    REAL      rhs              = 0.0;
    REAL      one_minus_weight = 1.0 - weight;

#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    if (nb == 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, rhs, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    rhs = b_val[i];
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i) rhs -= val[k] * u_val[j];
                    }
                    u_val[i] =
                        one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
                }
            }
        } else {
#endif
            for (i = 0; i < ROW; ++i) {
                rhs = b_val[i];
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i) rhs -= val[k] * u_val[j];
                }
                u_val[i] = one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
            }
#ifdef _OPENMP
        }
#endif
    } else if (nb > 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb * nthreads, sizeof(REAL));
#pragma omp parallel for private(myid, mybegin, myend, i, pb, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    pb = i * nb;
                    memcpy(b_tmp + myid * nb, b_val + pb, nb * sizeof(REAL));
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i)
                            fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp,
                                                nb);
                    }
                    fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp + myid * nb,
                                          one_minus_weight, u_val + pb, nb);
                }
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
        } else {
#endif
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
            for (i = 0; i < ROW; ++i) {
                pb = i * nb;
                memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i)
                        fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
                }
                fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp,
                                      one_minus_weight, u_val + pb, nb);
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
#ifdef _OPENMP
        }
#endif
    } else {
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_sor_descend (dBSRmat *A, dvector *b, dvector *u,
 *                                          REAL *diaginv, REAL weight)
 *
 * \brief SOR relaxation in the descending order
 *
 * \param A  Pointer to dBSRmat: the coefficient matrix
 * \param b  Pointer to dvector: the right hand side
 * \param u  Pointer to dvector: the unknowns (IN: initial guess, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 * \param weight   Over-relaxation weight
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/09/04
 */
void fasp_smoother_dbsr_sor_descend(
    dBSRmat* A, dvector* b, dvector* u, REAL* diaginv, REAL weight)
{
    // members of A
    const INT  ROW              = A->ROW;
    const INT  nb               = A->nb;
    const INT  nb2              = nb * nb;
    const INT* IA               = A->IA;
    const INT* JA               = A->JA;
    REAL*      val              = A->val;
    const REAL one_minus_weight = 1.0 - weight;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  pb;
    REAL rhs = 0.0;

#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    if (nb == 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, rhs, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                mybegin = ROW - 1 - mybegin;
                myend   = ROW - 1 - myend;
                for (i = mybegin; i > myend; i--) {
                    rhs = b_val[i];
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i) rhs -= val[k] * u_val[j];
                    }
                    u_val[i] =
                        one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
                }
            }
        } else {
#endif
            for (i = ROW - 1; i >= 0; i--) {
                rhs = b_val[i];
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i) rhs -= val[k] * u_val[j];
                }
                u_val[i] = one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
            }
#ifdef _OPENMP
        }
#endif
    } else if (nb > 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb * nthreads, sizeof(REAL));
#pragma omp parallel for private(myid, mybegin, myend, i, pb, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                mybegin = ROW - 1 - mybegin;
                myend   = ROW - 1 - myend;
                for (i = mybegin; i > myend; i--) {
                    pb = i * nb;
                    memcpy(b_tmp + myid * nb, b_val + pb, nb * sizeof(REAL));
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i)
                            fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb,
                                                b_tmp + myid * nb, nb);
                    }
                    fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp + myid * nb,
                                          one_minus_weight, u_val + pb, nb);
                }
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
        } else {
#endif
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
            for (i = ROW - 1; i >= 0; i--) {
                pb = i * nb;
                memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i)
                        fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
                }
                fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp,
                                      one_minus_weight, u_val + pb, nb);
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
#ifdef _OPENMP
        }
#endif
    } else {
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_sor_order (dBSRmat *A, dvector *b, dvector *u,
 *                                        REAL *diaginv, INT *mark, REAL weight)
 *
 * \brief SOR relaxation in the user-defined order
 *
 * \param A        Pointer to dBSRmat: the coefficient matrix
 * \param b        Pointer to dvector: the right hand side
 * \param u        Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param diaginv  Inverses for all the diagonal blocks of A
 * \param mark     Pointer to the user-defined ordering
 * \param weight   Over-relaxation weight
 *
 * \author Zhiyang Zhou
 * \date   2010/10/25
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/09/04
 */
void fasp_smoother_dbsr_sor_order(
    dBSRmat* A, dvector* b, dvector* u, REAL* diaginv, INT* mark, REAL weight)
{
    // members of A
    const INT  ROW              = A->ROW;
    const INT  nb               = A->nb;
    const INT  nb2              = nb * nb;
    const INT* IA               = A->IA;
    const INT* JA               = A->JA;
    REAL*      val              = A->val;
    const REAL one_minus_weight = 1.0 - weight;

    // values of dvector b and u
    REAL* b_val = b->val;
    REAL* u_val = u->val;

    // local variables
    INT  i, j, k;
    INT  I, pb;
    REAL rhs = 0.0;

#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    if (nb == 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, I, i, rhs, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (I = mybegin; I < myend; ++I) {
                    i   = mark[I];
                    rhs = b_val[i];
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i) rhs -= val[k] * u_val[j];
                    }
                    u_val[i] =
                        one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
                }
            }
        } else {
#endif
            for (I = 0; I < ROW; ++I) {
                i   = mark[I];
                rhs = b_val[i];
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i) rhs -= val[k] * u_val[j];
                }
                u_val[i] = one_minus_weight * u_val[i] + weight * (rhs * diaginv[i]);
            }
#ifdef _OPENMP
        }
#endif
    } else if (nb > 1) {
#ifdef _OPENMP
        if (ROW > OPENMP_HOLDS) {
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb * nthreads, sizeof(REAL));
#pragma omp parallel for private(myid, mybegin, myend, I, i, pb, k, j)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
                for (I = mybegin; I < myend; ++I) {
                    i  = mark[I];
                    pb = i * nb;
                    memcpy(b_tmp + myid * nb, b_val + pb, nb * sizeof(REAL));
                    for (k = IA[i]; k < IA[i + 1]; ++k) {
                        j = JA[k];
                        if (j != i)
                            fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb,
                                                b_tmp + myid * nb, nb);
                    }
                    fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp + myid * nb,
                                          one_minus_weight, u_val + pb, nb);
                }
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
        } else {
#endif
            REAL* b_tmp = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
            for (I = 0; I < ROW; ++I) {
                i  = mark[I];
                pb = i * nb;
                memcpy(b_tmp, b_val + pb, nb * sizeof(REAL));
                for (k = IA[i]; k < IA[i + 1]; ++k) {
                    j = JA[k];
                    if (j != i)
                        fasp_blas_smat_ymAx(val + k * nb2, u_val + j * nb, b_tmp, nb);
                }
                fasp_blas_smat_aAxpby(weight, diaginv + nb2 * i, b_tmp,
                                      one_minus_weight, u_val + pb, nb);
            }
            fasp_mem_free(b_tmp);
            b_tmp = NULL;
#ifdef _OPENMP
        }
#endif
    } else {
        fasp_chkerr(ERROR_NUM_BLOCKS, __FUNCTION__);
    }
}

/**
 * \fn void fasp_smoother_dbsr_ilu (dBSRmat *A, dvector *b, dvector *x, void *data)
 *
 * \brief ILU method as the smoother in solving Au=b with multigrid method
 *
 * \param A     Pointer to dBSRmat: the coefficient matrix
 * \param b     Pointer to dvector: the right hand side
 * \param x     Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param data  Pointer to user defined data
 *
 * \author Zhiyang Zhou, Zheng Li
 * \date   2010/10/25
 *
 * NOTE: Add multi-threads parallel ILU block by Zheng Li 12/04/2016.
 */
void fasp_smoother_dbsr_ilu(dBSRmat* A, dvector* b, dvector* x, void* data)
{
    ILU_data* iludata = (ILU_data*)data;
    const INT nb = iludata->nb, m = A->ROW * nb, memneed = 5 * m;

    REAL *xval = x->val, *bval = b->val;
    REAL* zr = iludata->work + 3 * m;
    REAL* z  = zr + m;

    double start, end;

    if (iludata->nwork < memneed) goto MEMERR;

    /** form residual zr = b - A x */
    fasp_darray_cp(m, bval, zr);
    fasp_blas_dbsr_aAxpy(-1.0, A, xval, zr);

    /** solve LU z=zr */
#ifdef _OPENMP

#if ILU_MC_OMP
    REAL* tz  = (REAL*)fasp_mem_calloc(A->ROW * A->nb, sizeof(REAL));
    REAL* tzr = (REAL*)fasp_mem_calloc(A->ROW * A->nb, sizeof(REAL));
    perm(A->ROW, A->nb, zr, iludata->jlevL, tzr);

    fasp_gettime(&start);
    fasp_precond_dbsr_ilu_mc_omp(tzr, tz, iludata);
    fasp_gettime(&end);

    invperm(A->ROW, A->nb, tz, iludata->jlevL, z);
    fasp_mem_free(tzr);
    tzr = NULL;
    fasp_mem_free(tz);
    tz = NULL;
#else
    fasp_gettime(&start);
    fasp_precond_dbsr_ilu_ls_omp(zr, z, iludata);
    fasp_gettime(&end);
#endif

    ilu_solve_time += end - start;

#else

    fasp_gettime(&start);
    fasp_precond_dbsr_ilu(zr, z, iludata);
    fasp_gettime(&end);
    ilu_solve_time += end - start;

#endif

    /** x=x+z */
    fasp_blas_darray_axpy(m, 1, z, xval);

    return;

MEMERR:
    printf("### ERROR: ILU needs %d memory, only %d available! [%s:%d]\n", memneed,
           iludata->nwork, __FILE__, __LINE__);
    fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

#ifdef _OPENMP

#if ILU_MC_OMP

/**
 * \fn static inline void perm (const INT n, const INT nb, const REAL*x,
 *                              const INT *p, REAL*y)
 *
 * \brief Array permutation
 *
 * \param n    Size of array
 * \param nb   Step size
 * \param x    Pointer to the original vector
 * \param p    Pointer to index mapping
 * \param y    Pointer to the destination vector
 *
 * \author Zheng Li
 * \date   12/04/2016
 */
static inline void perm(const INT n, const INT nb, const REAL* x, const INT* p, REAL* y)
{
    INT i, j, indx, indy;

#ifdef _OPENMP
#pragma omp parallel for private(i, j, indx, indy)
#endif
    for (i = 0; i < n; ++i) {
        indx = p[i] * nb;
        indy = i * nb;
        for (j = 0; j < nb; ++j) {
            y[indy + j] = x[indx + j];
        }
    }
}

/**
 * \fn static inline void invperm (const INT n, const INT nb,
 *                                 const REAL*x, const INT *p, REAL*y)
 *
 * \brief Array inverse permutation
 *
 * \param n    Size of array
 * \param nb   Step size
 * \param x    Pointer to the original vector
 * \param p    Pointer to index mapping
 * \param y    Pointer to the destination vector
 *
 * \author Zheng Li
 * \date   12/04/2016
 */
static inline void
invperm(const INT n, const INT nb, const REAL* x, const INT* p, REAL* y)
{
    INT i, j, indx, indy;

#ifdef _OPENMP
#pragma omp parallel for private(i, j, indx, indy)
#endif
    for (i = 0; i < n; ++i) {
        indx = i * nb;
        indy = p[i] * nb;
        for (j = 0; j < nb; ++j) {
            y[indy + j] = x[indx + j];
        }
    }
}

#endif // end of ILU_MC_OMP

#endif // end of _OPENMP

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
