/*! \file  ItrSmootherCSR.c
 *
 *  \brief Smoothers for dCSRmat matrices
 *
 *  \note  This file contains Level-2 (Itr) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxThreads.c, BlaArray.c,
 *         and BlaSpmvCSR.c
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

// fasp_smoother_dcsr_jacobi_ff(x, A, b, nsweeps, ordering, relax);
/**
 * @brief Weighted Jacobi method as a smoother only for the fine points
 * @author Xie Yan
 * @date 2022/12/12
 */
void fasp_smoother_dcsr_jacobi_ff(dvector*   x,
                                  dCSRmat*   A,
                                  dvector*   b,
                                  const INT  nsweeps,
                                  INT*       ordering,
                                  const REAL relax)
{
    REAL* r    = (REAL*)fasp_mem_calloc(A->col, sizeof(REAL));
    REAL* xval = (REAL*)fasp_mem_calloc(A->col, sizeof(REAL));
    INT   i, j, k, diagptr;
    for (i = 0; i < A->col; ++i) {
        xval[i] = x->val[i];
    }
    INT sweep_now;
    for (sweep_now = 0; sweep_now < nsweeps; ++sweep_now) {
        for (i = 0; i < A->row; ++i) {
            if (ordering[i] == FGPT) {
                diagptr       = -1;
                r[i]          = b->val[i];
                INT row_start = A->IA[i];
                INT row_end   = A->IA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    k = A->JA[j];
                    // find diagonal entry and compute residual
                    if (k == i) diagptr = j;
                    r[i] -= A->val[j] * x->val[k]; // use old x entry
                }
                // update solution
                xval[i] += relax * r[i] / A->val[diagptr]; // compute new x entry
            }
        }
        for (i = 0; i < A->col; ++i) {
            if (ordering[i] == FGPT) {
                x->val[i] = xval[i];
            }
        }
    }
    fasp_mem_free(r);
    fasp_mem_free(xval);
    // printf("###DEBUG: fasp_smoother_dcsr_jacobi_ff() -- Done!\n");
}

/**
 * \fn void fasp_smoother_dcsr_jacobi (dvector *u, const INT i_1, const INT i_n,
 *                                     const INT s, dCSRmat *A, dvector *b, INT L,
 *                                     const REAL w)
 *
 * \brief Weighted Jacobi method as a smoother
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param i_1    Starting index
 * \param i_n    Ending index
 * \param s      Increasing step
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param w      Relaxation weight
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/26/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/29/2012
 * Modified by Chensong Zhang on 08/24/2017: Pass weight w as a parameter
 */
void fasp_smoother_dcsr_jacobi(dvector*   u,
                               const INT  i_1,
                               const INT  i_n,
                               const INT  s,
                               dCSRmat*   A,
                               dvector*   b,
                               INT        L,
                               const REAL w)
{
    const INT   N  = ABS(i_n - i_1) + 1;
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT i, j, k, begin_row, end_row;

    // OpenMP variables
#ifdef _OPENMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    REAL* t = (REAL*)fasp_mem_calloc(N, sizeof(REAL));
    REAL* d = (REAL*)fasp_mem_calloc(N, sizeof(REAL));

    while (L--) {

        if (s > 0) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, begin_row, end_row, i, k, j)
                for (myid = 0; myid < nthreads; ++myid) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin += i_1;
                    myend += i_1;
                    for (i = mybegin; i < myend; i += s) {
                        t[i]      = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            if (i != j)
                                t[i] -= aval[k] * uval[j];
                            else
                                d[i] = aval[k];
                        }
                    }
                }
            } else {
#endif
                for (i = i_1; i <= i_n; i += s) {
                    t[i]      = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        if (i != j)
                            t[i] -= aval[k] * uval[j]; // diagonal entry is not included
                        else
                            d[i] = aval[k];
                    }
                }
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
            for (i = i_1; i <= i_n; i += s) {
                if (ABS(d[i]) > SMALLREAL)
                    uval[i] = (1 - w) * uval[i] + w * t[i] / d[i];
            }

        }

        else {

#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, begin_row, end_row, k, j)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin = i_1 - mybegin;
                    myend   = i_1 - myend;
                    for (i = mybegin; i > myend; i += s) {
                        t[i]      = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            if (i != j)
                                t[i] -= aval[k] * uval[j];
                            else
                                d[i] = aval[k];
                        }
                    }
                }
            } else {
#endif
                for (i = i_1; i >= i_n; i += s) {
                    t[i]      = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        if (i != j)
                            t[i] -= aval[k] * uval[j];
                        else
                            d[i] = aval[k];
                    }
                }
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
            for (i = i_1; i >= i_n; i += s) {
                if (ABS(d[i]) > SMALLREAL)
                    uval[i] = (1 - w) * uval[i] + w * t[i] / d[i];
            }
        }

    } // end while

    fasp_mem_free(t);
    t = NULL;
    fasp_mem_free(d);
    d = NULL;

    return;
}

/**
 * \fn void fasp_smoother_dcsr_gs (dvector *u, const INT i_1, const INT i_n,
 *                                 const INT s, dCSRmat *A, dvector *b, INT L)
 *
 * \brief Gauss-Seidel method as a smoother
 *
 * \param u    Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param i_1  Starting index
 * \param i_n  Ending index
 * \param s    Increasing step
 * \param A    Pointer to dBSRmat: the coefficient matrix
 * \param b    Pointer to dvector: the right hand side
 * \param L    Number of iterations
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/26/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 09/01/2012
 */
void fasp_smoother_dcsr_gs(dvector*  u,
                           const INT i_1,
                           const INT i_n,
                           const INT s,
                           dCSRmat*  A,
                           dvector*  b,
                           INT       L)
{
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0.0;

#ifdef _OPENMP
    const INT N = ABS(i_n - i_1) + 1;
    INT       myid, mybegin, myend;
    INT       nthreads = fasp_get_num_threads();
#endif

    if (s > 0) {

        while (L--) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, d, k, \
                                 j)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin += i_1, myend += i_1;
                    for (i = mybegin; i < myend; i += s) {
                        t         = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // diagonal first
                        d = aval[begin_row];
                        if (ABS(d) > SMALLREAL) {
                            for (k = begin_row + 1; k < end_row; ++k) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            }
                            uval[i] = t / d;
                        }
#else                 // general order
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else if (ABS(aval[k]) > SMALLREAL)
                                d = 1.e+0 / aval[k];
                        }
                        uval[i] = t * d;
#endif                // end DIAGONAL_PREF
                    } // end for i
                }

            }

            else {
#endif
                for (i = i_1; i <= i_n; i += s) {
                    t         = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];

#if DIAGONAL_PREF // diagonal first
                    d = aval[begin_row];
                    if (ABS(d) > SMALLREAL) {
                        for (k = begin_row + 1; k < end_row; ++k) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        }
                        uval[i] = t / d;
                    }
#else // general order
                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= aval[k] * uval[j];
                    else if (ABS(aval[k]) > SMALLREAL)
                        d = 1.e+0 / aval[k];
                }
                uval[i] = t * d;
#endif
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    } // if s
    else {

        while (L--) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, begin_row, end_row, d, k, j, \
                                 t)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin = i_1 - mybegin;
                    myend   = i_1 - myend;
                    for (i = mybegin; i > myend; i += s) {
                        t         = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // diagonal first
                        d = aval[begin_row];
                        if (ABS(d) > SMALLREAL) {
                            for (k = begin_row + 1; k < end_row; ++k) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            }
                            uval[i] = t / d;
                        }
#else // general order
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else if (ABS(aval[k]) > SMALLREAL)
                                d = 1.0 / aval[k];
                        }
                        uval[i] = t * d;
#endif
                    } // end for i
                }
            } else {
#endif
                for (i = i_1; i >= i_n; i += s) {
                    t         = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
#if DIAGONAL_PREF // diagonal first
                    d = aval[begin_row];
                    if (ABS(d) > SMALLREAL) {
                        for (k = begin_row + 1; k < end_row; ++k) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        }
                        uval[i] = t / d;
                    }
#else // general order
                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= aval[k] * uval[j];
                    else if (ABS(aval[k]) > SMALLREAL)
                        d = 1.0 / aval[k];
                }
                uval[i] = t * d;
#endif
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    } // end if

    return;
}

/**
 * \fn void fasp_smoother_dcsr_gs_cf (dvector *u, dCSRmat *A, dvector *b, INT L,
 *                                    INT *mark, const INT order)
 *
 * \brief Gauss-Seidel smoother with C/F ordering for Au=b
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param mark   C/F marker array
 * \param order  C/F ordering: -1: F-first; 1: C-first
 *
 * \author Zhiyang Zhou
 * \date   11/12/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012
 */
void fasp_smoother_dcsr_gs_cf(
    dvector* u, dCSRmat* A, dvector* b, INT L, INT* mark, const INT order)
{
    const INT   nrow = b->row; // number of rows
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0.0;

#ifdef _OPENMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    // F-point first, C-point second
    if (order == FPFIRST) {

        while (L--) {

#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] != 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                            d = aval[begin_row];
                            for (k = begin_row + 1; k < end_row; k++) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            } // end for k
#else
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
#endif // end if DIAG_PREF
                            if (ABS(d) > SMALLREAL) uval[i] = t / d;
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] != 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                        d = aval[begin_row];
                        for (k = begin_row + 1; k < end_row; k++) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        } // end for k
#else
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
#endif // end if DIAG_PREF
                        if (ABS(d) > SMALLREAL) uval[i] = t / d;
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] == 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                            d = aval[begin_row];
                            for (k = begin_row + 1; k < end_row; k++) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            } // end for k
#else
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
#endif // end if DIAG_PREF
                            if (ABS(d) > SMALLREAL) uval[i] = t / d;
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] == 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                        d = aval[begin_row];
                        for (k = begin_row + 1; k < end_row; k++) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        } // end for k
#else
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
#endif // end if DIAG_PREF
                        if (ABS(d) > SMALLREAL) uval[i] = t / d;
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    }

    // C-point first, F-point second
    else {

        while (L--) {
#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, t, i, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] == 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                            d = aval[begin_row];
                            for (k = begin_row + 1; k < end_row; k++) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            } // end for k
#else
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
#endif // end if DIAG_PREF
                            if (ABS(d) > SMALLREAL) uval[i] = t / d;
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] == 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 09/22/2012
                        d = aval[begin_row];
                        for (k = begin_row + 1; k < end_row; k++) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        } // end for k
#else
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
#endif // end if DIAG_PREF
                        if (ABS(d) > SMALLREAL) uval[i] = t / d;
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] != 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                            d = aval[begin_row];
                            for (k = begin_row + 1; k < end_row; k++) {
                                j = ja[k];
                                t -= aval[k] * uval[j];
                            } // end for k
#else
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
#endif // end if DIAG_PREF
                            if (ABS(d) > SMALLREAL) uval[i] = t / d;
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] != 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 09/22/2012
                        d = aval[begin_row];
                        for (k = begin_row + 1; k < end_row; k++) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        } // end for k
#else
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
#endif // end if DIAG_PREF
                        if (ABS(d) > SMALLREAL) uval[i] = t / d;
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    } // end if order

    return;
}

/**
 * \fn void fasp_smoother_dcsr_gs_ff (dvector *u, dCSRmat *A, dvector *b, INT L,
 *                                    INT *mark)
 *
 * \brief Gauss-Seidel smoother with on F-points only for Au=b
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param mark   C/F marker array
 *
 * \author Chensong Zhang
 * \date   10/11/2022
 */
void fasp_smoother_dcsr_gs_ff(dvector* u, dCSRmat* A, dvector* b, INT L, INT* mark)
{
    const INT   nrow = b->row; // number of rows
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0.0;

#ifdef _OPENMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    while (L--) {

#ifdef _OPENMP
        if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    if (mark[i] != 1) {
                        t         = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                        d = aval[begin_row];
                        for (k = begin_row + 1; k < end_row; k++) {
                            j = ja[k];
                            t -= aval[k] * uval[j];
                        } // end for k
#else
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        } // end for k
#endif // end if DIAG_PREF
                        if (ABS(d) > SMALLREAL) uval[i] = t / d;
                    }
                } // end for i
            }
        } else {
#endif
            for (i = 0; i < nrow; i++) {
                if (mark[i] != 1) {
                    t         = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
#if DIAGONAL_PREF // Added by Chensong on 01/17/2013
                    d = aval[begin_row];
                    for (k = begin_row + 1; k < end_row; k++) {
                        j = ja[k];
                        t -= aval[k] * uval[j];
                    } // end for k
#else
                for (k = begin_row; k < end_row; k++) {
                    j = ja[k];
                    if (i != j)
                        t -= aval[k] * uval[j];
                    else
                        d = aval[k];
                } // end for k
#endif // end if DIAG_PREF
                    if (ABS(d) > SMALLREAL) uval[i] = t / d;
                }
            } // end for i
#ifdef _OPENMP
        }
#endif

    } // end while

    return;
}

/**
 * \fn void fasp_smoother_dcsr_sgs (dvector *u, dCSRmat *A, dvector *b, INT L)
 *
 * \brief Symmetric Gauss-Seidel method as a smoother
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 *
 * \author Xiaozhe Hu
 * \date   10/26/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 09/01/2012
 */
void fasp_smoother_dcsr_sgs(dvector* u, dCSRmat* A, dvector* b, INT L)
{
    const INT   nm1 = b->row - 1;
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0;

#ifdef _OPENMP
    INT myid, mybegin, myend, up;
    INT nthreads = fasp_get_num_threads();
#endif

    while (L--) {
        // forward sweep
#ifdef _OPENMP
        up = nm1 + 1;
        if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, j, k, \
                                 d)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
                for (i = mybegin; i < myend; i++) {
                    t         = bval[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
                    if (ABS(d) > SMALLREAL) uval[i] = t / d;
                } // end for i
            }
        } else {
#endif
            for (i = 0; i <= nm1; ++i) {
                t         = bval[i];
                begin_row = ia[i];
                end_row   = ia[i + 1];
                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= aval[k] * uval[j];
                    else
                        d = aval[k];
                } // end for k
                if (ABS(d) > SMALLREAL) uval[i] = t / d;
            } // end for i
#ifdef _OPENMP
        }
#endif

        // backward sweep
#ifdef _OPENMP
        up = nm1;
        if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
            for (myid = 0; myid < nthreads; myid++) {
                fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
                mybegin = nm1 - 1 - mybegin;
                myend   = nm1 - 1 - myend;
                for (i = mybegin; i > myend; i--) {
                    t         = bval[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    } // end for k
                    if (ABS(d) > SMALLREAL) uval[i] = t / d;
                } // end for i
            }
        } else {
#endif
            for (i = nm1 - 1; i >= 0; --i) {
                t         = bval[i];
                begin_row = ia[i];
                end_row   = ia[i + 1];
                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= aval[k] * uval[j];
                    else
                        d = aval[k];
                } // end for k
                if (ABS(d) > SMALLREAL) uval[i] = t / d;
            } // end for i
#ifdef _OPENMP
        }
#endif
    } // end while

    return;
}

/**
 * \fn void fasp_smoother_dcsr_sor (dvector *u, const INT i_1, const INT i_n,
 *                                  const INT s, dCSRmat *A, dvector *b, INT L,
 *                                  const REAL w)
 *
 * \brief SOR method as a smoother
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param i_1    Starting index
 * \param i_n    Ending index
 * \param s      Increasing step
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param w      Over-relaxation weight
 *
 * \author Xiaozhe Hu
 * \date   10/26/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 09/01/2012
 */
void fasp_smoother_dcsr_sor(dvector*   u,
                            const INT  i_1,
                            const INT  i_n,
                            const INT  s,
                            dCSRmat*   A,
                            dvector*   b,
                            INT        L,
                            const REAL w)
{
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0;

#ifdef _OPENMP
    const INT N = ABS(i_n - i_1) + 1;
    INT       myid, mybegin, myend;
    INT       nthreads = fasp_get_num_threads();
#endif

    while (L--) {
        if (s > 0) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin += i_1, myend += i_1;
                    for (i = mybegin; i < myend; i += s) {
                        t         = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        }
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                }

            } else {
#endif
                for (i = i_1; i <= i_n; i += s) {
                    t         = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    }
                    if (ABS(d) > SMALLREAL) uval[i] = w * (t / d) + (1 - w) * uval[i];
                }
#ifdef _OPENMP
            }
#endif
        } else {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin = i_1 - mybegin, myend = i_1 - myend;
                    for (i = mybegin; i > myend; i += s) {
                        t         = bval[i];
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        }
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                }
            } else {
#endif
                for (i = i_1; i >= i_n; i += s) {
                    t         = bval[i];
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        if (i != j)
                            t -= aval[k] * uval[j];
                        else
                            d = aval[k];
                    }
                    if (ABS(d) > SMALLREAL) uval[i] = w * (t / d) + (1 - w) * uval[i];
                }
#ifdef _OPENMP
            }
#endif
        }
    } // end while

    return;
}

/**
 * \fn void fasp_smoother_dcsr_sor_cf (dvector *u, dCSRmat *A, dvector *b, INT L,
 *                                     const REAL w, INT *mark, const INT order)
 *
 * \brief SOR smoother with C/F ordering for Au=b
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param w      Over-relaxation weight
 * \param mark   C/F marker array
 * \param order  C/F ordering: -1: F-first; 1: C-first
 *
 * \author Zhiyang Zhou
 * \date   2010/11/12
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/29/2012
 */
void fasp_smoother_dcsr_sor_cf(
    dvector* u, dCSRmat* A, dvector* b, INT L, const REAL w, INT* mark, const INT order)
{
    const INT   nrow = b->row; // number of rows
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0.0;

#ifdef _OPENMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    // F-point first
    if (order == -1) {
        while (L--) {
#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] == 0 || mark[i] == 2) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
                            if (ABS(d) > SMALLREAL)
                                uval[i] = w * (t / d) + (1 - w) * uval[i];
                        }
                    }
                }
            } // end for i
            else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] == 0 || mark[i] == 2) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        } // end for k
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, i, mybegin, myend, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] == 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
                            if (ABS(d) > SMALLREAL)
                                uval[i] = w * (t / d) + (1 - w) * uval[i];
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] == 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        } // end for k
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while
    } else {
        while (L--) {
#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, k, j, d, begin_row,       \
                                 end_row)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] == 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
                            if (ABS(d) > SMALLREAL)
                                uval[i] = w * (t / d) + (1 - w) * uval[i];
                        }
                    } // end for i
                }
            } else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] == 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        } // end for k
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif

#ifdef _OPENMP
            if (nrow > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, t, begin_row, end_row, k, j, \
                                 d)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
                    for (i = mybegin; i < myend; i++) {
                        if (mark[i] != 1) {
                            t         = bval[i];
                            begin_row = ia[i], end_row = ia[i + 1];
                            for (k = begin_row; k < end_row; k++) {
                                j = ja[k];
                                if (i != j)
                                    t -= aval[k] * uval[j];
                                else
                                    d = aval[k];
                            } // end for k
                            if (ABS(d) > SMALLREAL)
                                uval[i] = w * (t / d) + (1 - w) * uval[i];
                        }
                    }
                }
            } // end for i
            else {
#endif
                for (i = 0; i < nrow; i++) {
                    if (mark[i] != 1) {
                        t         = bval[i];
                        begin_row = ia[i];
                        end_row   = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            if (i != j)
                                t -= aval[k] * uval[j];
                            else
                                d = aval[k];
                        } // end for k
                        if (ABS(d) > SMALLREAL)
                            uval[i] = w * (t / d) + (1 - w) * uval[i];
                    }
                } // end for i
#ifdef _OPENMP
            }
#endif
        } // end while
    }

    return;
}

/**
 * \fn void fasp_smoother_dcsr_ilu (dCSRmat *A, dvector *b, dvector *x, void *data)
 *
 * \brief ILU method as a smoother
 *
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param x      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param data   Pointer to user defined data
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/11/12
 */
void fasp_smoother_dcsr_ilu(dCSRmat* A, dvector* b, dvector* x, void* data)
{
    const INT       m = A->row, m2 = 2 * m, memneed = 3 * m;
    const ILU_data* iludata = (ILU_data*)data;

    REAL* zz = iludata->work;
    REAL* zr = iludata->work + m;
    REAL* z  = iludata->work + m2;

    if (iludata->nwork < memneed) goto MEMERR;

    {
        INT   i, j, jj, begin_row, end_row;
        REAL* lu   = iludata->luval;
        INT*  ijlu = iludata->ijlu;
        REAL *xval = x->val, *bval = b->val;

        /** form residual zr = b - A x */
        fasp_darray_cp(m, bval, zr);
        fasp_blas_dcsr_aAxpy(-1.0, A, xval, zr);

        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0] = zr[0];
        for (i = 1; i < m; ++i) {
            begin_row = ijlu[i];
            end_row   = ijlu[i + 1];
            for (j = begin_row; j < end_row; ++j) {
                jj = ijlu[j];
                if (jj < i)
                    zr[i] -= lu[j] * zz[jj];
                else
                    break;
            }
            zz[i] = zr[i];
        }

        // backward sweep: solve upper matrix equation U*z=zz
        z[m - 1] = zz[m - 1] * lu[m - 1];
        for (i = m - 2; i >= 0; --i) {
            begin_row = ijlu[i];
            end_row   = ijlu[i + 1] - 1;
            for (j = end_row; j >= begin_row; --j) {
                jj = ijlu[j];
                if (jj > i)
                    zz[i] -= lu[j] * z[jj];
                else
                    break;
            }
            z[i] = zz[i] * lu[i];
        }

        fasp_blas_darray_axpy(m, 1, z, xval);
    }

    return;

MEMERR:
    printf("### ERROR: ILU needs %d memory, only %d available! [%s:%d]\n", memneed,
           iludata->nwork, __FILE__, __LINE__);
    fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
}

/**
 * \fn void fasp_smoother_dcsr_kaczmarz (dvector *u, const INT i_1, const INT i_n,
 *                                       const INT s, dCSRmat *A, dvector *b,
 *                                       INT L, const REAL w)
 *
 * \brief Kaczmarz method as a smoother
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param i_1    Starting index
 * \param i_n    Ending index
 * \param s      Increasing step
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 * \param w      Over-relaxation weight
 *
 * \author Xiaozhe Hu
 * \date   2010/11/12
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/09/01
 */
void fasp_smoother_dcsr_kaczmarz(dvector*   u,
                                 const INT  i_1,
                                 const INT  i_n,
                                 const INT  s,
                                 dCSRmat*   A,
                                 dvector*   b,
                                 INT        L,
                                 const REAL w)
{
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT  i, j, k, begin_row, end_row;
    REAL temp1, temp2, alpha;

#ifdef _OPENMP
    const INT N = ABS(i_n - i_1) + 1;
    INT       myid, mybegin, myend;
    INT       nthreads = fasp_get_num_threads();
#endif

    if (s > 0) {

        while (L--) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, temp1, temp2, begin_row,     \
                                 end_row, k, alpha, j)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin += i_1, myend += i_1;
                    for (i = mybegin; i < myend; i += s) {
                        temp1     = 0;
                        temp2     = 0;
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            temp1 += aval[k] * aval[k];
                            temp2 += aval[k] * uval[j];
                        } // end for k
                    }
                    alpha = (bval[i] - temp2) / temp1;
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        uval[j] += w * alpha * aval[k];
                    } // end for k
                }     // end for i
            } else {
#endif
                for (i = i_1; i <= i_n; i += s) {
                    temp1     = 0;
                    temp2     = 0;
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        temp1 += aval[k] * aval[k];
                        temp2 += aval[k] * uval[j];
                    } // end for k
                    alpha = (bval[i] - temp2) / temp1;
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        uval[j] += w * alpha * aval[k];
                    } // end for k
                }     // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    } // if s

    else {
        while (L--) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, temp1, temp2, begin_row,     \
                                 end_row, k, alpha, j)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin = i_1 - mybegin, myend = i_1 - myend;
                    for (i = mybegin; i > myend; i += s) {
                        temp1     = 0;
                        temp2     = 0;
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            temp1 += aval[k] * aval[k];
                            temp2 += aval[k] * uval[j];
                        } // end for k
                        alpha = (bval[i] - temp2) / temp1;
                        for (k = begin_row; k < end_row; ++k) {
                            j = ja[k];
                            uval[j] += w * alpha * aval[k];
                        } // end for k
                    }     // end for i
                }
            } else {
#endif
                for (i = i_1; i >= i_n; i += s) {
                    temp1     = 0;
                    temp2     = 0;
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        temp1 += aval[k] * aval[k];
                        temp2 += aval[k] * uval[j];
                    } // end for k
                    alpha = (bval[i] - temp2) / temp1;
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        uval[j] += w * alpha * aval[k];
                    } // end for k
                }     // end for i
#ifdef _OPENMP
            }
#endif
        } // end while

    } // end if

    return;
}

/**
 * \fn void fasp_smoother_dcsr_L1diag (dvector *u, const INT i_1, const INT i_n,
 *                                     const INT s, dCSRmat *A, dvector *b, INT L)
 *
 * \brief Diagonal scaling (using L1 norm) as a smoother
 *
 * \param u      Pointer to dvector: the unknowns (IN: initial, OUT: approximation)
 * \param i_1    Starting index
 * \param i_n    Ending index
 * \param s      Increasing step
 * \param A      Pointer to dBSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param L      Number of iterations
 *
 * \author Xiaozhe Hu, James Brannick
 * \date   01/26/2011
 *
 * Modified by Chunsheng Feng, Zheng Li on 09/01/2012
 */
void fasp_smoother_dcsr_L1diag(dvector*  u,
                               const INT i_1,
                               const INT i_n,
                               const INT s,
                               dCSRmat*  A,
                               dvector*  b,
                               INT       L)
{
    const INT   N  = ABS(i_n - i_1) + 1;
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aval = A->val, *bval = b->val;
    REAL*       uval = u->val;

    // local variables
    INT i, j, k, begin_row, end_row;

#ifdef _OPENMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif

    // Checks should be outside of for; t,d can be allocated before calling!!!
    // --Chensong
    REAL* t = (REAL*)fasp_mem_calloc(N, sizeof(REAL));
    REAL* d = (REAL*)fasp_mem_calloc(N, sizeof(REAL));

    while (L--) {
        if (s > 0) {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, begin_row, end_row, k, j)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin += i_1, myend += i_1;
                    for (i = mybegin; i < myend; i += s) {
                        t[i]      = bval[i];
                        d[i]      = 0.0;
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            t[i] -= aval[k] * uval[j];
                            d[i] += ABS(aval[k]);
                        }
                    }
                }
#pragma omp parallel for private(i)
                for (i = i_1; i <= i_n; i += s) {
                    if (ABS(d[i]) > SMALLREAL) u->val[i] += t[i] / d[i];
                }
            } else {
#endif
                for (i = i_1; i <= i_n; i += s) {
                    t[i]      = bval[i];
                    d[i]      = 0.0;
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        t[i] -= aval[k] * uval[j];
                        d[i] += ABS(aval[k]);
                    }
                }

                for (i = i_1; i <= i_n; i += s) {
                    if (ABS(d[i]) > SMALLREAL) u->val[i] += t[i] / d[i];
                }
#ifdef _OPENMP
            }
#endif
        } else {
#ifdef _OPENMP
            if (N > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, k, j, begin_row, end_row)
                for (myid = 0; myid < nthreads; myid++) {
                    fasp_get_start_end(myid, nthreads, N, &mybegin, &myend);
                    mybegin = i_1 - mybegin, myend = i_1 - myend;
                    for (i = mybegin; i > myend; i += s) {
                        t[i]      = bval[i];
                        d[i]      = 0.0;
                        begin_row = ia[i], end_row = ia[i + 1];
                        for (k = begin_row; k < end_row; k++) {
                            j = ja[k];
                            t[i] -= aval[k] * uval[j];
                            d[i] += ABS(aval[k]);
                        }
                    }
                }
#pragma omp parallel for private(i)
                for (i = i_1; i >= i_n; i += s) {
                    if (ABS(d[i]) > SMALLREAL) u->val[i] += t[i] / d[i];
                }
            } else {
#endif
                for (i = i_1; i >= i_n; i += s) {
                    t[i]      = bval[i];
                    d[i]      = 0.0;
                    begin_row = ia[i];
                    end_row   = ia[i + 1];
                    for (k = begin_row; k < end_row; ++k) {
                        j = ja[k];
                        t[i] -= aval[k] * uval[j];
                        d[i] += ABS(aval[k]);
                    }
                }

                for (i = i_1; i >= i_n; i += s) {
                    if (ABS(d[i]) > SMALLREAL) u->val[i] += t[i] / d[i];
                }
#ifdef _OPENMP
            }
#endif
        }

    } // end while

    fasp_mem_free(t);
    t = NULL;
    fasp_mem_free(d);
    d = NULL;

    return;
}

#if 0
/**
 * \fn static dCSRmat form_contractor (dCSRmat *A, const INT smoother, const INT steps,
 *                                     const INT ndeg, const REAL relax, const REAL dtol)
 *
 * \brief Form contractor I-BA
 *
 * \param A          Pointer to the dCSRmat
 * \param smoother   Smoother type
 * \param steps      Smoothing steps
 * \param ndeg       Degree of the polynomial smoother
 * \param relax      Relaxation parameter for SOR smoother
 * \param dtol       Drop tplerance for droping small entries in matrix
 *
 * \return The contractor in dCSRmat format
 *
 * \author Xiaozhe Hu, James Brannick
 * \date   01/26/2011
 *
 * \note This is NOT an O(N) algorithm, need to be modified!!!!
 */
static dCSRmat form_contractor (dCSRmat    *A,
                                const INT   smoother,
                                const INT   steps,
                                const INT   ndeg,
                                const REAL  relax,
                                const REAL  dtol)
{
    const INT   n=A->row;
    INT         i;
    
    REAL *work = (REAL *)fasp_mem_calloc(2*n,sizeof(REAL));
    
    dvector b, x;
    b.row=x.row=n;
    b.val=work; x.val=work+n;
    
    INT *index = (INT *)fasp_mem_calloc(n,sizeof(INT));
    
    for (i=0; i<n; ++i) index[i]=i;
    
    dCSRmat B = fasp_dcsr_create(n, n, n*n); // too much memory required, need to change!!
    
    dCSRmat C, D;
    
    for (i=0; i<n; ++i){
        
        // get i-th column
        fasp_dcsr_getcol(i, A, b.val);
        
        // set x =0.0
        fasp_dvec_set(n, &x, 0.0);
        
        // smooth
        switch (smoother) {
            case GS:
                fasp_smoother_dcsr_gs(&x, 0, n-1, 1, A, &b, steps);
                break;
            case POLY:
                fasp_smoother_dcsr_poly(A, &b, &x, n, ndeg, steps);
                break;
            case JACOBI:
                fasp_smoother_dcsr_jacobi(&x, 0, n-1, 1, A, &b, steps);
                break;
            case SGS:
                fasp_smoother_dcsr_sgs(&x, A, &b, steps);
                break;
            case SOR:
                fasp_smoother_dcsr_sor(&x, 0, n-1, 1, A, &b, steps, relax);
                break;
            case SSOR:
                fasp_smoother_dcsr_sor(&x, 0, n-1, 1, A, &b, steps, relax);
                fasp_smoother_dcsr_sor(&x, n-1, 0,-1, A, &b, steps, relax);
                break;
            case GSOR:
                fasp_smoother_dcsr_gs(&x, 0, n-1, 1, A, &b, steps);
                fasp_smoother_dcsr_sor(&x, n-1, 0, -1, A, &b, steps, relax);
                break;
            case SGSOR:
                fasp_smoother_dcsr_gs(&x, 0, n-1, 1, A, &b, steps);
                fasp_smoother_dcsr_gs(&x, n-1, 0,-1, A, &b, steps);
                fasp_smoother_dcsr_sor(&x, 0, n-1, 1, A, &b, steps, relax);
                fasp_smoother_dcsr_sor(&x, n-1, 0,-1, A, &b, steps, relax);
                break;
            default:
                printf("### ERROR: Unknown smoother type! [%s:%d]\n",
                       __FILE__, __LINE__);
                fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
        }
        
        // store to B
        B.IA[i] = i*n;
        memcpy(&(B.JA[i*n]), index, n*sizeof(INT));
        memcpy(&(B.val[i*n]), x.val, x.row*sizeof(REAL));
        
    }
    
    B.IA[n] = n*n;
    
    // drop small entries
    compress_dCSRmat(&B, &D, dtol);
    
    // get contractor
    fasp_dcsr_trans(&D, &C);
    
    // clean up
    fasp_mem_free(work); work = NULL;
    fasp_dcsr_free(&B);
    fasp_dcsr_free(&D);
    
    return C;
}
#endif

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
