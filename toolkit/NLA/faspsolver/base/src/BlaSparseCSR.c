/*! \file  BlaSparseCSR.c
 *
 *  \brief Sparse matrix operations for dCSRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxSort.c, AuxThreads.c,
 *         AuxVector.c, and BlaSpmvCSR.c
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

#if MULTI_COLOR_ORDER
static void generate_S_theta(dCSRmat*, iCSRmat*, REAL);
#endif

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn dCSRmat fasp_dcsr_create (const INT m, const INT n, const INT nnz)
 *
 * \brief Create CSR sparse matrix data memory space
 *
 * \param m    Number of rows
 * \param n    Number of columns
 * \param nnz  Number of nonzeros
 *
 * \return A   the new dCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 */
dCSRmat fasp_dcsr_create(const INT m, const INT n, const INT nnz)
{
    dCSRmat A;

    if (m > 0) {
        A.IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    } else {
        A.IA = NULL;
    }

    if (n > 0) {
        A.JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    } else {
        A.JA = NULL;
    }

    if (nnz > 0) {
        A.val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    } else {
        A.val = NULL;
    }

    A.row = m;
    A.col = n;
    A.nnz = nnz;

#if MULTI_COLOR_ORDER
    A.color = 0;
    A.IC    = NULL;
    A.ICMAP = NULL;
#endif

    return A;
}

/**
 * \fn iCSRmat fasp_icsr_create (const INT m, const INT n, const INT nnz)
 *
 * \brief Create CSR sparse matrix data memory space
 *
 * \param m    Number of rows
 * \param n    Number of columns
 * \param nnz  Number of nonzeros
 *
 * \return A   the new iCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 */
iCSRmat fasp_icsr_create(const INT m, const INT n, const INT nnz)
{
    iCSRmat A;

    if (m > 0) {
        A.IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    } else {
        A.IA = NULL;
    }

    if (n > 0) {
        A.JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    } else {
        A.JA = NULL;
    }

    if (nnz > 0) {
        A.val = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    } else {
        A.val = NULL;
    }

    A.row = m;
    A.col = n;
    A.nnz = nnz;

    return A;
}

/**
 * \fn void fasp_dcsr_alloc (const INT m, const INT n, const INT nnz, dCSRmat *A)
 *
 * \brief Allocate CSR sparse matrix memory space
 *
 * \param m      Number of rows
 * \param n      Number of columns
 * \param nnz    Number of nonzeros
 * \param A      Pointer to the dCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 */
void fasp_dcsr_alloc(const INT m, const INT n, const INT nnz, dCSRmat* A)
{
    if (m <= 0 || n <= 0) {
        printf("### ERROR: Matrix dim %d, %d must be positive! [%s]\n", m, n,
               __FUNCTION__);
        return;
    }

    if (m > 0) {
        A->IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    } else {
        A->IA = NULL;
    }

    if (nnz > 0) {
        A->JA  = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
        A->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    } else {
        A->JA  = NULL;
        A->val = NULL;
    }

    A->row = m;
    A->col = n;
    A->nnz = nnz;

#if MULTI_COLOR_ORDER
    A->color = 0;
    A->IC    = NULL;
    A->ICMAP = NULL;
#endif

    return;
}

/**
 * \fn void fasp_dcsr_free (dCSRmat *A)
 *
 * \brief Free CSR sparse matrix data memory space
 *
 * \param A   Pointer to the dCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *  Modified by Chunsheng Feng on 08/11/2017: init A to NULL
 */
void fasp_dcsr_free(dCSRmat* A)
{
    if (A == NULL) return;

    fasp_mem_free(A->IA);
    A->IA = NULL;
    fasp_mem_free(A->JA);
    A->JA = NULL;
    fasp_mem_free(A->val);
    A->val = NULL;

#if MULTI_COLOR_ORDER
    fasp_mem_free(A->IC);
    A->IC = NULL;
    fasp_mem_free(A->ICMAP);
    A->ICMAP = NULL;
#endif

    A->col = 0;
    A->row = 0;
    A->nnz = 0;
    A      = NULL;
}

/**
 * \fn void fasp_icsr_free (iCSRmat *A)
 *
 * \brief Free CSR sparse matrix data memory space
 *
 * \param A   Pointer to the iCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *  Modified by Chunsheng Feng on 08/11/2017: init A to NULL
 */
void fasp_icsr_free(iCSRmat* A)
{
    if (A == NULL) return;

    fasp_mem_free(A->IA);
    A->IA = NULL;
    fasp_mem_free(A->JA);
    A->JA = NULL;
    fasp_mem_free(A->val);
    A->val = NULL;
    A->col = 0;
    A->row = 0;
    A->nnz = 0;
    A      = NULL;
}

/**
 * \fn INT fasp_dcsr_bandwidth (const dCSRmat *A)
 *
 * \brief Get bandwith of matrix
 *
 * \param A       pointer to the dCSRmat matrix
 *
 * \author Zheng Li
 * \date   03/22/2015
 */
INT fasp_dcsr_bandwidth(const dCSRmat* A)
{
    const INT  row = A->row;
    const INT* ia  = A->IA;
    INT        i, max;

    for (max = i = 0; i < row; ++i) max = MAX(max, ia[i + 1] - ia[i]);

    return (max);
}

/**
 * \fn dCSRmat fasp_dcsr_perm (dCSRmat *A, INT *P)
 *
 * \brief Apply permutation of A, i.e. Aperm=PAP' by the orders given in P
 *
 * \param A  Pointer to the original dCSRmat matrix
 * \param P  Pointer to orders
 *
 * \return   The new ordered dCSRmat matrix if succeed, NULL if fail
 *
 * \author Shiquan Zhang
 * \date   03/10/2010
 *
 * \note   P[i] = k means k-th row and column become i-th row and column!
 *
 * \note   Deprecated! Will be replaced by fasp_dcsr_permz later. --Chensong
 *
 * Modified by Chunsheng Feng, Zheng Li on 07/12/2012
 */
dCSRmat fasp_dcsr_perm(dCSRmat* A, INT* P)
{
    const INT   n = A->row, nnz = A->nnz;
    const INT * ia = A->IA, *ja = A->JA;
    const REAL* Aval = A->val;
    INT         i, j, k, jaj, i1, i2, start;
    SHORT       nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (MIN(n, nnz) > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    dCSRmat Aperm = fasp_dcsr_create(n, n, nnz);

    // form the transpose of P
    INT* Pt = (INT*)fasp_mem_calloc(n, sizeof(INT));

    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; ++myid) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) Pt[P[i]] = i;
        }
    } else {
        for (i = 0; i < n; ++i) Pt[P[i]] = i;
    }

    // compute IA of P*A (row permutation)
    Aperm.IA[0] = 0;
    for (i = 0; i < n; ++i) {
        k               = P[i];
        Aperm.IA[i + 1] = Aperm.IA[i] + (ia[k + 1] - ia[k]);
    }

    // perform actual P*A
    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i1, i2, k, start, j, jaj)
#endif
        for (myid = 0; myid < nthreads; ++myid) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                i1    = Aperm.IA[i];
                i2    = Aperm.IA[i + 1] - 1;
                k     = P[i];
                start = ia[k];
                for (j = i1; j <= i2; ++j) {
                    jaj          = start + j - i1;
                    Aperm.JA[j]  = ja[jaj];
                    Aperm.val[j] = Aval[jaj];
                }
            }
        }
    } else {
        for (i = 0; i < n; ++i) {
            i1    = Aperm.IA[i];
            i2    = Aperm.IA[i + 1] - 1;
            k     = P[i];
            start = ia[k];
            for (j = i1; j <= i2; ++j) {
                jaj          = start + j - i1;
                Aperm.JA[j]  = ja[jaj];
                Aperm.val[j] = Aval[jaj];
            }
        }
    }

    // perform P*A*P' (column permutation)
    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, k, j)
#endif
        for (myid = 0; myid < nthreads; ++myid) {
            fasp_get_start_end(myid, nthreads, nnz, &mybegin, &myend);
            for (k = mybegin; k < myend; ++k) {
                j           = Aperm.JA[k];
                Aperm.JA[k] = Pt[j];
            }
        }
    } else {
        for (k = 0; k < nnz; ++k) {
            j           = Aperm.JA[k];
            Aperm.JA[k] = Pt[j];
        }
    }

    fasp_mem_free(Pt);
    Pt = NULL;

    return (Aperm);
}

/**
 * \fn void fasp_dcsr_sort (dCSRmat *A)
 *
 * \brief Sort each row of A in ascending order w.r.t. column indices.
 *
 * \param A   Pointer to the dCSRmat matrix
 *
 * \author Shiquan Zhang
 * \date   06/10/2010
 */
void fasp_dcsr_sort(dCSRmat* A)
{
    const INT n = A->col;
    INT       i, j, start, row_length;

    // temp memory for sorting rows of A
    INT * index, *ja;
    REAL* a;

    index = (INT*)fasp_mem_calloc(n, sizeof(INT));
    ja    = (INT*)fasp_mem_calloc(n, sizeof(INT));
    a     = (REAL*)fasp_mem_calloc(n, sizeof(REAL));

    for (i = 0; i < n; ++i) {
        start      = A->IA[i];
        row_length = A->IA[i + 1] - start;

        for (j = 0; j < row_length; ++j) index[j] = j;

        fasp_aux_iQuickSortIndex(&(A->JA[start]), 0, row_length - 1, index);

        for (j = 0; j < row_length; ++j) {
            ja[j] = A->JA[start + index[j]];
            a[j]  = A->val[start + index[j]];
        }

        for (j = 0; j < row_length; ++j) {
            A->JA[start + j]  = ja[j];
            A->val[start + j] = a[j];
        }
    }

    // clean up memory
    fasp_mem_free(index);
    index = NULL;
    fasp_mem_free(ja);
    ja = NULL;
    fasp_mem_free(a);
    a = NULL;
}

/**
 * \fn SHORT fasp_dcsr_getblk (const dCSRmat *A, const INT *Is, const INT *Js,
 *                             const INT m, const INT n, dCSRmat *B)
 *
 * \brief Get a sub CSR matrix of A with specified rows and columns
 *
 * \param A     Pointer to dCSRmat matrix
 * \param B     Pointer to dCSRmat matrix
 * \param Is    Pointer to selected rows
 * \param Js    Pointer to selected columns
 * \param m     Number of selected rows
 * \param n     Number of selected columns
 *
 * \return      FASP_SUCCESS if succeeded, otherwise return error information.
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   12/25/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
SHORT fasp_dcsr_getblk(const dCSRmat* A, const INT* Is, const INT* Js, const INT m,
                       const INT n, dCSRmat* B)
{
    SHORT use_openmp = FALSE;
    SHORT status     = FASP_SUCCESS;
    INT   i, j, k, nnz = 0;
    INT*  col_flag;

#ifdef _OPENMP
    INT stride_i, mybegin, myend, myid, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // create column flags
    col_flag = (INT*)fasp_mem_calloc(A->col, sizeof(INT));

    B->row = m;
    B->col = n;

    B->IA  = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    B->JA  = (INT*)fasp_mem_calloc(A->nnz, sizeof(INT));
    B->val = (REAL*)fasp_mem_calloc(A->nnz, sizeof(REAL));

#if MULTI_COLOR_ORDER
    B->color = 0;
    B->IC    = NULL;
    B->ICMAP = NULL;
#endif

    if (use_openmp) {
#ifdef _OPENMP
        stride_i = n / nthreads;
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
        {
            myid    = omp_get_thread_num();
            mybegin = myid * stride_i;
            if (myid < nthreads - 1)
                myend = mybegin + stride_i;
            else
                myend = n;
            for (i = mybegin; i < myend; ++i) {
                col_flag[Js[i]] = i + 1;
            }
        }
#endif
    } else {
        for (i = 0; i < n; ++i) col_flag[Js[i]] = i + 1;
    }

    // Count nonzeros for sub matrix and fill in
    B->IA[0] = 0;
    for (i = 0; i < m; ++i) {
        for (k = A->IA[Is[i]]; k < A->IA[Is[i] + 1]; ++k) {
            j = A->JA[k];
            if (col_flag[j] > 0) {
                B->JA[nnz]  = col_flag[j] - 1;
                B->val[nnz] = A->val[k];
                nnz++;
            }
        } /* end for k */
        B->IA[i + 1] = nnz;
    } /* end for i */
    B->nnz = nnz;

    // re-allocate memory space
    B->JA  = (INT*)fasp_mem_realloc(B->JA, sizeof(INT) * nnz);
    B->val = (REAL*)fasp_mem_realloc(B->val, sizeof(REAL) * nnz);

    fasp_mem_free(col_flag);
    col_flag = NULL;

    return (status);
}

/**
 * \fn void fasp_dcsr_getdiag (INT n, const dCSRmat *A, dvector *diag)
 *
 * \brief Get first n diagonal entries of a CSR matrix A
 *
 * \param n     Number of diagonal entries to get (if n=0, then get all diagonal
 * entries) \param A     Pointer to dCSRmat CSR matrix \param diag  Pointer to the
 * diagonal as a dvector
 *
 * \author Chensong Zhang
 * \date   05/20/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_dcsr_getdiag(INT n, const dCSRmat* A, dvector* diag)
{
    INT i, k, j, ibegin, iend;

    SHORT nthreads = 1, use_openmp = FALSE;

    if (n == 0 || n > A->row || n > A->col) n = MIN(A->row, A->col);

#ifdef _OPENMP
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    fasp_dvec_alloc(n, diag);

    if (use_openmp) {
        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, ibegin, iend, k, j)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                ibegin = A->IA[i];
                iend   = A->IA[i + 1];
                for (k = ibegin; k < iend; ++k) {
                    j = A->JA[k];
                    if ((j - i) == 0) {
                        diag->val[i] = A->val[k];
                        break;
                    } // end if
                }     // end for k
            }         // end for i
        }
    } else {
        for (i = 0; i < n; ++i) {
            ibegin = A->IA[i];
            iend   = A->IA[i + 1];
            for (k = ibegin; k < iend; ++k) {
                j = A->JA[k];
                if ((j - i) == 0) {
                    diag->val[i] = A->val[k];
                    break;
                } // end if
            }     // end for k
        }         // end for i
    }
}

/**
 * \fn void fasp_dcsr_getcol (const INT n, const dCSRmat *A, REAL *col)
 *
 * \brief Get the n-th column of a CSR matrix A
 *
 * \param n    Index of a column of A (0 <= n <= A.col-1)
 * \param A    Pointer to dCSRmat CSR matrix
 * \param col  Pointer to the column
 *
 * \author Xiaozhe Hu
 * \date   11/07/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 07/08/2012
 */
void fasp_dcsr_getcol(const INT n, const dCSRmat* A, REAL* col)
{
    INT i, j, row_begin, row_end;
    INT nrow = A->row, ncol = A->col;
    INT status = FASP_SUCCESS;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (nrow > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // check the column index n
    if (n < 0 || n >= ncol) {
        printf("### ERROR: Illegal column index %d! [%s]\n", n, __FUNCTION__);
        status = ERROR_DUMMY_VAR;
        goto FINISHED;
    }

    // get the column
    if (use_openmp) {
        INT mybegin, myend, myid;

#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, j, row_begin, row_end)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, nrow, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                col[i]    = 0.0;
                row_begin = A->IA[i];
                row_end   = A->IA[i + 1];
                for (j = row_begin; j < row_end; ++j) {
                    if (A->JA[j] == n) {
                        col[i] = A->val[j];
                    }
                } // end for j
            }     // end for i
        }
    } else {
        for (i = 0; i < nrow; ++i) {
            // set the entry to zero
            col[i]    = 0.0;
            row_begin = A->IA[i];
            row_end   = A->IA[i + 1];
            for (j = row_begin; j < row_end; ++j) {
                if (A->JA[j] == n) {
                    col[i] = A->val[j];
                }
            } // end for j
        }     // end for i
    }

FINISHED:
    fasp_chkerr(status, __FUNCTION__);
}

/*!
 * \fn void fasp_dcsr_diagpref (dCSRmat *A)
 *
 * \brief Re-order the column and data arrays of a CSR matrix,
 *        so that the first entry in each row is the diagonal
 *
 * \param A   Pointer to the matrix to be re-ordered
 *
 * \author Zhiyang Zhou
 * \date   09/09/2010
 *
 * \author Chunsheng Feng, Zheng Li
 * \date   09/02/2012
 *
 * \note Reordering is done in place.
 *
 * Modified by Chensong Zhang on Dec/21/2012
 */
void fasp_dcsr_diagpref(dCSRmat* A)
{
    const INT num_rowsA = A->row;
    REAL*     A_data    = A->val;
    INT*      A_i       = A->IA;
    INT*      A_j       = A->JA;

    // Local variable
    INT  i, j;
    INT  tempi, row_size;
    REAL tempd;

#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend, ibegin, iend;
    INT nthreads = fasp_get_num_threads();
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

#ifdef _OPENMP
    if (num_rowsA > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, i, j, ibegin, iend, tempi, tempd, mybegin, myend)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, num_rowsA, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                ibegin = A_i[i];
                iend   = A_i[i + 1];
                // check whether the first entry is already diagonal
                if (A_j[ibegin] != i) {
                    for (j = ibegin + 1; j < iend; j++) {
                        if (A_j[j] == i) {
#if DEBUG_MODE > 2
                            printf("### DEBUG: Switch entry_%d with entry_0\n", j);
#endif
                            tempi       = A_j[ibegin];
                            A_j[ibegin] = A_j[j];
                            A_j[j]      = tempi;

                            tempd          = A_data[ibegin];
                            A_data[ibegin] = A_data[j];
                            A_data[j]      = tempd;
                            break;
                        }
                    }
                    if (j == iend) {
                        printf("### ERROR: Diagonal entry %d is zero!\n", i);
                        fasp_chkerr(ERROR_MISC, __FUNCTION__);
                    }
                }
            }
        }
    } else {
#endif
        for (i = 0; i < num_rowsA; i++) {
            row_size = A_i[i + 1] - A_i[i];
            // check whether the first entry is already diagonal
            if (A_j[0] != i) {
                for (j = 1; j < row_size; j++) {
                    if (A_j[j] == i) {
#if DEBUG_MODE > 2
                        printf("### DEBUG: Switch entry_%d with entry_0\n", j);
#endif
                        tempi  = A_j[0];
                        A_j[0] = A_j[j];
                        A_j[j] = tempi;

                        tempd     = A_data[0];
                        A_data[0] = A_data[j];
                        A_data[j] = tempd;

                        break;
                    }
                }
                if (j == row_size) {
                    printf("### ERROR: Diagonal entry %d is zero!\n", i);
                    fasp_chkerr(ERROR_MISC, __FUNCTION__);
                }
            }
            A_j += row_size;
            A_data += row_size;
        }
#ifdef _OPENMP
    }
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * \fn SHORT fasp_dcsr_regdiag (dCSRmat *A, const REAL value)
 *
 * \brief Regularize diagonal entries of a CSR sparse matrix
 *
 * \param A      Pointer to the dCSRmat matrix
 * \param value  Set a value on diag(A) which is too close to zero to "value"
 *
 * \return       FASP_SUCCESS if no diagonal entry is close to zero, else ERROR
 *
 * \author Shiquan Zhang
 * \date   11/07/2009
 */
SHORT fasp_dcsr_regdiag(dCSRmat* A, const REAL value)
{
    const INT  m  = A->row;
    const INT *ia = A->IA, *ja = A->JA;
    REAL*      aj = A->val;

    // Local variables
    INT   i, j, k, begin_row, end_row;
    SHORT status = ERROR_UNKNOWN;

    for (i = 0; i < m; ++i) {
        begin_row = ia[i];
        end_row   = ia[i + 1];
        for (k = begin_row; k < end_row; ++k) {
            j = ja[k];
            if (i == j) {
                if (aj[k] < 0.0)
                    goto FINISHED;
                else if (aj[k] < SMALLREAL)
                    aj[k] = value;
            }
        } // end for k
    }     // end for i

    status = FASP_SUCCESS;

FINISHED:
    return status;
}

/**
 * \fn void fasp_icsr_cp (const iCSRmat *A, iCSRmat *B)
 *
 * \brief Copy a iCSRmat to a new one B=A
 *
 * \param A   Pointer to the iCSRmat matrix
 * \param B   Pointer to the iCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   05/16/2013
 */
void fasp_icsr_cp(const iCSRmat* A, iCSRmat* B)
{
    B->row = A->row;
    B->col = A->col;
    B->nnz = A->nnz;

    fasp_iarray_cp(A->row + 1, A->IA, B->IA);
    fasp_iarray_cp(A->nnz, A->JA, B->JA);
    fasp_iarray_cp(A->nnz, A->val, B->val);
}

/**
 * \fn void fasp_dcsr_cp (const dCSRmat *A, dCSRmat *B)
 *
 * \brief copy a dCSRmat to a new one B=A
 *
 * \param A   Pointer to the dCSRmat matrix
 * \param B   Pointer to the dCSRmat matrix
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_dcsr_cp(const dCSRmat* A, dCSRmat* B)
{
    B->row = A->row;
    B->col = A->col;
    B->nnz = A->nnz;

    fasp_iarray_cp(A->row + 1, A->IA, B->IA);
    fasp_iarray_cp(A->nnz, A->JA, B->JA);
    fasp_darray_cp(A->nnz, A->val, B->val);
}

/**
 * \fn void fasp_icsr_trans (const iCSRmat *A, iCSRmat *AT)
 *
 * \brief Find transpose of iCSRmat matrix A
 *
 * \param A   Pointer to the iCSRmat matrix A
 * \param AT  Pointer to the iCSRmat matrix A'
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 06/20/2012
 */
void fasp_icsr_trans(const iCSRmat* A, iCSRmat* AT)
{
    const INT n = A->row, m = A->col, nnz = A->nnz, m1 = m - 1;

    // Local variables
    INT i, j, k, p;
    INT ibegin, iend;

#if DEBUG_MODE > 1
    printf("### DEBUG: m=%d, n=%d, nnz=%d\n", m, n, nnz);
#endif

    AT->row = m;
    AT->col = n;
    AT->nnz = nnz;

    AT->IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));

    AT->JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));

    if (A->val) {
        AT->val = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    } else {
        AT->val = NULL;
    }

    // first pass: find the Number of nonzeros in the first m-1 columns of A
    // Note: these Numbers are stored in the array AT.IA from 1 to m-1
    fasp_iarray_set(m + 1, AT->IA, 0);

    for (j = 0; j < nnz; ++j) {
        i = A->JA[j]; // column Number of A = row Number of A'
        if (i < m1) AT->IA[i + 2]++;
    }

    for (i = 2; i <= m; ++i) AT->IA[i] += AT->IA[i - 1];

    // second pass: form A'
    if (A->val != NULL) {
        for (i = 0; i < n; ++i) {
            ibegin = A->IA[i];
            iend   = A->IA[i + 1];
            for (p = ibegin; p < iend; p++) {
                j          = A->JA[p] + 1;
                k          = AT->IA[j];
                AT->JA[k]  = i;
                AT->val[k] = A->val[p];
                AT->IA[j]  = k + 1;
            } // end for p
        }     // end for i
    } else {
        for (i = 0; i < n; ++i) {
            ibegin = A->IA[i];
            iend   = A->IA[i + 1];
            for (p = ibegin; p < iend; p++) {
                j         = A->JA[p] + 1;
                k         = AT->IA[j];
                AT->JA[k] = i;
                AT->IA[j] = k + 1;
            } // end for p
        }     // end for i
    }         // end if
}

/**
 * \fn void fasp_dcsr_trans (const dCSRmat *A, dCSRmat *AT)
 *
 * \brief Find transpose of dCSRmat matrix A
 *
 * \param A   Pointer to the dCSRmat matrix
 * \param AT  Pointer to the transpose of dCSRmat matrix A (output)
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 *
 *  Modified by Chunsheng Feng, Zheng Li on 06/20/2012
 */
INT fasp_dcsr_trans(const dCSRmat* A, dCSRmat* AT)
{
    const INT n = A->row, m = A->col, nnz = A->nnz;

    // Local variables
    INT i, j, k, p;

    AT->row = m;
    AT->col = n;
    AT->nnz = nnz;

    AT->IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));

    AT->JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));

    if (A->val) {
        AT->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));

    } else {
        AT->val = NULL;
    }

#if MULTI_COLOR_ORDER
    AT->color = 0;
    AT->IC    = NULL;
    AT->ICMAP = NULL;
#endif

    // first pass: find the Number of nonzeros in the first m-1 columns of A
    // Note: these Numbers are stored in the array AT.IA from 1 to m-1

    // fasp_iarray_set(m+1, AT->IA, 0);
    memset(AT->IA, 0, sizeof(INT) * (m + 1));

    for (j = 0; j < nnz; ++j) {
        i = A->JA[j]; // column Number of A = row Number of A'
        if (i < m - 1) AT->IA[i + 2]++;
    }

    for (i = 2; i <= m; ++i) AT->IA[i] += AT->IA[i - 1];

    // second pass: form A'
    if (A->val) {
        for (i = 0; i < n; ++i) {
            INT ibegin = A->IA[i], iend = A->IA[i + 1];
            for (p = ibegin; p < iend; p++) {
                j          = A->JA[p] + 1;
                k          = AT->IA[j];
                AT->JA[k]  = i;
                AT->val[k] = A->val[p];
                AT->IA[j]  = k + 1;
            } // end for p
        }     // end for i
    } else {
        for (i = 0; i < n; ++i) {
            INT ibegin = A->IA[i], iend1 = A->IA[i + 1];
            for (p = ibegin; p < iend1; p++) {
                j         = A->JA[p] + 1;
                k         = AT->IA[j];
                AT->JA[k] = i;
                AT->IA[j] = k + 1;
            } // end for p
        }     // end of i
    }         // end if

    return FASP_SUCCESS;
}

/**
 * \fn void fasp_dcsr_transpose (INT *row[2], INT *col[2], REAL *val[2],
 *                               INT *nn, INT *tniz)
 *
 * \brief Transpose of a dCSRmat matrix
 *
 * \note  This subroutine transpose in CSR format IN ORDER
 *
 * \param row    Pointers of the rows of the matrix and its transpose
 * \param col    Pointers of the columns of the matrix and its transpose
 * \param val    Pointers to the values of the matrix and its transpose
 * \param nn     Pointer to the number of rows/columns of A and A'
 * \param tniz   Pointer to the number of nonzeros A and A'
 *
 * \author Shuo Zhang
 * \date   07/06/2009
 */
void fasp_dcsr_transpose(INT* row[2], INT* col[2], REAL* val[2], INT* nn, INT* tniz)
{
    const INT nca = nn[1]; // Number of columns

    INT* izc    = (INT*)fasp_mem_calloc(nn[1], sizeof(INT));
    INT* izcaux = (INT*)fasp_mem_calloc(nn[1], sizeof(INT));

    // Local variables
    INT i, m, itmp;

    // first pass: to set order right
    for (i = 0; i < tniz[0]; ++i) izc[col[0][i]]++;

    izcaux[0] = 0;
    for (i = 1; i < nca; ++i) izcaux[i] = izcaux[i - 1] + izc[i - 1];

    // second pass: form transpose
    memset(izc, 0, nca * sizeof(INT));

    for (i = 0; i < tniz[0]; ++i) {
        m            = col[0][i];
        itmp         = izcaux[m] + izc[m];
        row[1][itmp] = m;
        col[1][itmp] = row[0][i];
        val[1][itmp] = val[0][i];
        izc[m]++;
    }

    fasp_mem_free(izc);
    izc = NULL;
    fasp_mem_free(izcaux);
    izcaux = NULL;
}

/**
 * \fn void fasp_dcsr_compress (const dCSRmat *A, dCSRmat *B, const REAL dtol)
 *
 * \brief Compress a CSR matrix A and store in CSR matrix B by
 *        dropping small entries abs(aij)<=dtol
 *
 * \param A     Pointer to dCSRmat CSR matrix
 * \param B     Pointer to dCSRmat CSR matrix
 * \param dtol  Drop tolerance
 *
 * \author Shiquan Zhang
 * \date   03/10/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/25/2012
 */
void fasp_dcsr_compress(const dCSRmat* A, dCSRmat* B, const REAL dtol)
{
    INT i, j, k;
    INT ibegin, iend1;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (B->nnz > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    INT* index = (INT*)fasp_mem_calloc(A->nnz, sizeof(INT));

    B->row = A->row;
    B->col = A->col;

    B->IA = (INT*)fasp_mem_calloc(A->row + 1, sizeof(INT));

    B->IA[0] = A->IA[0];

    // first pass: determine the size of B
    k = 0;
    for (i = 0; i < A->row; ++i) {
        ibegin = A->IA[i];
        iend1  = A->IA[i + 1];
        for (j = ibegin; j < iend1; ++j)
            if (ABS(A->val[j]) > dtol) {
                index[k] = j;
                ++k;
            } /* end of j */
        B->IA[i + 1] = k;
    } /* end of i */
    B->nnz = k;
    B->JA  = (INT*)fasp_mem_calloc(B->nnz, sizeof(INT));
    B->val = (REAL*)fasp_mem_calloc(B->nnz, sizeof(REAL));

    // second pass: generate the index and element to B
    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, i, mybegin, myend)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, B->nnz, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                B->JA[i]  = A->JA[index[i]];
                B->val[i] = A->val[index[i]];
            }
        }
    } else {
        for (i = 0; i < B->nnz; ++i) {
            B->JA[i]  = A->JA[index[i]];
            B->val[i] = A->val[index[i]];
        }
    }

    fasp_mem_free(index);
    index = NULL;
}

/**
 * \fn SHORT fasp_dcsr_compress_inplace (dCSRmat *A, const REAL dtol)
 *
 * \brief Compress a CSR matrix A IN PLACE by
 *        dropping small entries abs(aij)<=dtol
 *
 * \param A     Pointer to dCSRmat CSR matrix
 * \param dtol  Drop tolerance
 *
 * \author Xiaozhe Hu
 * \date   12/25/2010
 *
 * Modified by Chensong Zhang on 02/21/2013
 * Modified by Chunsheng Feng on 10/16/2020: Avoid filtering diagonal entries.
 *
 * \note This routine can be modified for filtering.
 */
SHORT fasp_dcsr_compress_inplace(dCSRmat* A, const REAL dtol)
{
    const INT row = A->row;
    const INT nnz = A->nnz;

    INT   i, j, k;
    INT   ibegin, iend = A->IA[0];
    SHORT status = FASP_SUCCESS;
    k            = 0;
    for (i = 0; i < row; ++i) {
        ibegin = iend;
        iend   = A->IA[i + 1];
        for (j = ibegin; j < iend; ++j)
            if (ABS(A->val[j]) > dtol || i == A->JA[j]) {
                A->JA[k]  = A->JA[j];
                A->val[k] = A->val[j];
                ++k;
            } /* end of j */
        A->IA[i + 1] = k;
    } /* end of i */

    if (k <= nnz) {
        A->nnz = k;
        A->JA  = (INT*)fasp_mem_realloc(A->JA, k * sizeof(INT));
        A->val = (REAL*)fasp_mem_realloc(A->val, k * sizeof(REAL));
    } else {
        printf("### WARNING: Size of compressed matrix is bigger than original!\n");
        status = ERROR_UNKNOWN;
    }

    return (status);
}

/**
 * \fn void fasp_dcsr_shift (dCSRmat *A, const INT offset)
 *
 * \brief Re-index a REAL matrix in CSR format to make the index starting from 0 or 1
 *
 * \param A         Pointer to CSR matrix
 * \param  offset   Size of offset (1 or -1)
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 07/11/2012
 */
void fasp_dcsr_shift(dCSRmat* A, const INT offset)
{
    const INT nnz = A->nnz;
    const INT n   = A->row + 1;
    INT       i, *ai = A->IA, *aj = A->JA;
    SHORT     nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (MIN(n, nnz) > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                ai[i] += offset;
            }
        }
    } else {
        for (i = 0; i < n; ++i) ai[i] += offset;
    }

    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, nnz, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                aj[i] += offset;
            }
        }
    } else {
        for (i = 0; i < nnz; ++i) aj[i] += offset;
    }
}

/**
 * \fn void fasp_dcsr_symdiagscale (dCSRmat *A, const dvector *diag)
 *
 * \brief Symmetric diagonal scaling D^{-1/2}AD^{-1/2}
 *
 * \param A     Pointer to the dCSRmat matrix
 * \param diag  Pointer to the diagonal entries
 *
 * \author Xiaozhe Hu
 * \date   01/31/2011
 *
 * Modified by Chunsheng Feng, Zheng Li on 07/11/2012
 */
void fasp_dcsr_symdiagscale(dCSRmat* A, const dvector* diag)
{
    // information about matrix A
    const INT  n   = A->row;
    const INT* IA  = A->IA;
    const INT* JA  = A->JA;
    REAL*      val = A->val;
    REAL*      work;

    SHORT nthreads = 1, use_openmp = FALSE;

    // local variables
    INT i, j, k, row_start, row_end;

#ifdef _OPENMP
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (diag->row != n) {
        printf("### ERROR: Size of diag = %d != size of matrix = %d!", diag->row, n);
        fasp_chkerr(ERROR_MISC, __FUNCTION__);
    }

    // work space
    work = (REAL*)fasp_mem_calloc(n, sizeof(REAL));

    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) work[i] = sqrt(diag->val[i]);
        }
    } else {
        // square root of diagonal entries
        for (i = 0; i < n; i++) work[i] = sqrt(diag->val[i]);
    }

    if (use_openmp) {
        INT myid, mybegin, myend;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, row_start, row_end, i, j, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                row_start = IA[i];
                row_end   = IA[i + 1];
                for (j = row_start; j < row_end; j++) {
                    k      = JA[j];
                    val[j] = val[j] / (work[i] * work[k]);
                }
            }
        }
    } else {
        // main loop
        for (i = 0; i < n; i++) {
            row_start = IA[i];
            row_end   = IA[i + 1];
            for (j = row_start; j < row_end; j++) {
                k      = JA[j];
                val[j] = val[j] / (work[i] * work[k]);
            }
        }
    }

    // free work space
    fasp_mem_free(work);
    work = NULL;
}

/**
 * \fn dCSRmat fasp_dcsr_sympart (dCSRmat *A)
 * \brief Get symmetric part of a dCSRmat matrix
 *
 * \param A   Pointer to the dCSRmat matrix
 *
 * \return    Symmetrized the dCSRmat matrix
 *
 * \author Xiaozhe Hu
 * \date 03/21/2011
 */
dCSRmat fasp_dcsr_sympart(dCSRmat* A)
{
    // local variable
    dCSRmat AT;

    // return variable
    dCSRmat SA;

#if MULTI_COLOR_ORDER
    AT.IC    = NULL;
    SA.IC    = NULL;
    AT.ICMAP = NULL;
    SA.ICMAP = NULL;
#endif

    // get the transpose of A
    fasp_dcsr_trans(A, &AT);

    // get symmetrized A
    fasp_blas_dcsr_add(A, 1.0, &AT, 0.0, &SA);

    // clean
    fasp_dcsr_free(&AT);

    // return
    return SA;
}

/**
 * \fn void fasp_dcsr_transz (dCSRmat *A,  INT *p, dCSRmat *AT)
 *
 * \brief Generalized transpose of A: (n x m) matrix given in dCSRmat format
 *
 * \param A     Pointer to matrix in dCSRmat for transpose, INPUT
 * \param p     Permutation, INPUT
 * \param AT    Pointer to matrix AT = transpose(A)  if p = NULL, OR
 *                                AT = transpose(A)p if p is not NULL
 *
 * \note The storage for all pointers in AT should already be allocated,
 *       i.e. AT->IA, AT->JA and AT->val should be allocated before calling
 *       this function. If A.val=NULL, then AT->val[] is not changed.
 *
 * \note performs AT=transpose(A)p, where p is a permutation. If
 *       p=NULL then p=I is assumed. Applying twice this procedure one
 *       gets At=transpose(transpose(A)p)p = transpose(p)Ap, which is the
 *       same A with rows and columns permutted according to p.
 *
 * \note If A=NULL, then only transposes/permutes the structure of A.
 *
 * \note For p=NULL, applying this two times A-->AT-->A orders all the
 *       row indices in A in increasing order.
 *
 * Reference: Fred G. Gustavson. Two fast algorithms for sparse
 *            matrices: multiplication and permuted transposition.
 *            ACM Trans. Math. Software, 4(3):250�C269, 1978.
 *
 * \author Ludmil Zikatanov
 * \date   19951219 (Fortran), 20150912 (C)
 */
void fasp_dcsr_transz(dCSRmat* A, INT* p, dCSRmat* AT)
{
    /* tested for permutation and transposition */
    /* transpose or permute; if A.val is null ===> transpose the
       structure only */
    const INT   n = A->row, m = A->col, nnz = A->nnz;
    const INT * ia = NULL, *ja = NULL;
    const REAL* a  = NULL;
    INT         m1 = m + 1;
    ia             = A->IA;
    ja             = A->JA;
    a              = A->val;
    /* introducing few extra pointers hould not hurt too much the speed */
    INT * iat = NULL, *jat = NULL;
    REAL* at = NULL;

    /* loop variables */
    INT i, j, jp, pi, iabeg, iaend, k;

    /* initialize */
    AT->row = m;
    AT->col = n;
    AT->nnz = nnz;

    /* all these should be allocated or change this to allocate them here */
    iat = AT->IA;
    jat = AT->JA;
    at  = AT->val;
    for (i = 0; i < m1; ++i) iat[i] = 0;
    iaend = ia[n];
    for (i = 0; i < iaend; ++i) {
        j = ja[i] + 2;
        if (j < m1) iat[j]++;
    }
    iat[0] = 0;
    iat[1] = 0;
    if (m != 1) {
        for (i = 2; i < m1; ++i) {
            iat[i] += iat[i - 1];
        }
    }

    if (p && a) {
        /* so we permute and also use matrix entries */
        for (i = 0; i < n; ++i) {
            pi    = p[i];
            iabeg = ia[pi];
            iaend = ia[pi + 1];
            if (iaend > iabeg) {
                for (jp = iabeg; jp < iaend; ++jp) {
                    j      = ja[jp] + 1;
                    k      = iat[j];
                    jat[k] = i;
                    at[k]  = a[jp];
                    iat[j] = k + 1;
                }
            }
        }
    } else if (a && !p) {
        /* transpose values, no permutation */
        for (i = 0; i < n; ++i) {
            iabeg = ia[i];
            iaend = ia[i + 1];
            if (iaend > iabeg) {
                for (jp = iabeg; jp < iaend; ++jp) {
                    j      = ja[jp] + 1;
                    k      = iat[j];
                    jat[k] = i;
                    at[k]  = a[jp];
                    iat[j] = k + 1;
                }
            }
        }
    } else if (!a && p) {
        /* Only integers and permutation (only a is null) */
        for (i = 0; i < n; ++i) {
            pi    = p[i];
            iabeg = ia[pi];
            iaend = ia[pi + 1];
            if (iaend > iabeg) {
                for (jp = iabeg; jp < iaend; ++jp) {
                    j      = ja[jp] + 1;
                    k      = iat[j];
                    jat[k] = i;
                    iat[j] = k + 1;
                }
            }
        }
    } else {
        /* Only integers and no permutation (both a and p are null */
        for (i = 0; i < n; ++i) {
            iabeg = ia[i];
            iaend = ia[i + 1];
            if (iaend > iabeg) {
                for (jp = iabeg; jp < iaend; ++jp) {
                    j      = ja[jp] + 1;
                    k      = iat[j];
                    jat[k] = i;
                    iat[j] = k + 1;
                }
            }
        }
    }

    return;
}

/**
 * \fn dCSRmat fasp_dcsr_permz (dCSRmat *A, INT *p)
 *
 * \brief Permute rows and cols of A, i.e. A=PAP' by the ordering in p
 *
 * \param A  Pointer to the original dCSRmat matrix
 * \param p  Pointer to ordering
 *
 * \note This is just applying twice fasp_dcsr_transz(&A,p,At).
 *
 * \note In matlab notation: Aperm=A(p,p);
 *
 * \return The new ordered dCSRmat matrix if succeed, NULL if fail
 *
 * \author Ludmil Zikatanov
 * \date   19951219 (Fortran), 20150912 (C)
 */
dCSRmat fasp_dcsr_permz(dCSRmat* A, INT* p)
{
    const INT n = A->row, nnz = A->nnz;
    dCSRmat   Aperm1, Aperm;

    Aperm1 = fasp_dcsr_create(n, n, nnz);
    Aperm  = fasp_dcsr_create(n, n, nnz);

    fasp_dcsr_transz(A, p, &Aperm1);
    fasp_dcsr_transz(&Aperm1, p, &Aperm);

    // clean up
    fasp_dcsr_free(&Aperm1);

    return (Aperm);
}

/**
 * \fn void fasp_dcsr_sortz (dCSRmat *A, const SHORT isym)
 *
 * \brief Sort each row of A in ascending order w.r.t. column indices.
 *
 * \param A     Pointer to the dCSRmat matrix
 * \param isym  Flag for symmetry, =[0/nonzero]=[general/symmetric] matrix
 *
 * \note Applying twice fasp_dcsr_transz(), if A is symmetric, then
 *       the transpose is applied only once and then AT copied on A.
 *
 * \author Ludmil Zikatanov
 * \date   19951219 (Fortran), 20150912 (C)
 */
void fasp_dcsr_sortz(dCSRmat* A, const SHORT isym)
{
    const INT n = A->row, m = A->col, nnz = A->nnz;
    dCSRmat   AT = fasp_dcsr_create(m, n, nnz);

    /* watch carefully who is a pointer and who is not in fasp_dcsr_transz() */
    fasp_dcsr_transz(A, NULL, &AT);

    /* if the matrix is symmetric, then only one transpose is needed
       and now we just copy */
    if ((m == n) && (isym))
        fasp_dcsr_cp(&AT, A);
    else
        fasp_dcsr_transz(&AT, NULL, A);

    // clean up
    fasp_dcsr_free(&AT);
}

/**
 * \fn void fasp_dcsr_multicoloring (dCSRmat *A, INT *flags, INT *groups)
 *
 * \brief Use the greedy multi-coloring to get color groups of the adjacency graph of A
 *
 * \param A       Input dCSRmat
 * \param flags   flags for the independent group
 * \param groups  Return group numbers
 *
 * \author Chunsheng Feng
 * \date   09/15/2012
 */
void fasp_dcsr_multicoloring(dCSRmat* A, INT* flags, INT* groups)
{
#if MULTI_COLOR_ORDER
    INT  k, i, j, pre, group;
    INT  iend;
    INT  icount;
    INT  front, rear;
    INT  n    = A->row;
    INT* IA   = A->IA;
    INT* JA   = A->JA;
    INT* cq   = (INT*)malloc(sizeof(INT) * (n + 1));
    INT* newr = (INT*)malloc(sizeof(INT) * (n + 1));

#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
    for (k = 0; k < n; k++) cq[k] = k;

    group = 0;
    for (k = 0; k < n; k++) {
        if ((IA[k + 1] - IA[k]) > group) group = IA[k + 1] - IA[k];
    }

    A->IC    = (INT*)malloc(sizeof(INT) * (group + 2));
    A->ICMAP = (INT*)malloc(sizeof(INT) * (n));

    front = n - 1;
    rear  = n - 1;

    memset(newr, -1, sizeof(INT) * (n + 1));
    memset(A->ICMAP, 0, sizeof(INT) * n);

    group    = 0;
    icount   = 0;
    A->IC[0] = 0;
    pre      = 0;

    do {
        front++;
        if (front == n) front = 0;
        i = cq[front];
        if (i <= pre) {
            A->IC[group]     = icount;
            A->ICMAP[icount] = i;
            group++;
            icount++;
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        } else if (newr[i] == group) {
            rear++;
            if (rear == n) rear = 0;
            cq[rear] = i;
        } else {
            A->ICMAP[icount] = i;
            icount++;
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        }
        pre = i;

    } while (rear != front);

    A->IC[group] = icount;
    A->color     = group;
    free(cq);
    free(newr);
    *groups = group;
#else
    printf("### ERROR: %s has not been defined!\n", __FUNCTION__);
#endif
}

/**
 * \fn void dCSRmat_Multicoloring(dCSRmat* A, INT *rowmax , INT *groups)
 *
 * \brief Use the greedy multicoloring algorithm to get color groups for for the
 * adjacency graph of A
 *
 * \param A       Input dCSRmat
 * \param rowmax  Max row nonzeros of A
 * \param groups  Return group numbers
 *
 * \author Chunsheng Feng
 * \date   09/15/2012
 */
void dCSRmat_Multicoloring(dCSRmat* A, INT* rowmax, INT* groups)
{
#if MULTI_COLOR_ORDER
    INT  k, i, j, pre, group;
    INT  igold, iend, iavg;
    INT  icount;
    INT  front, rear;
    INT  n  = A->row;
    INT* IA = A->IA;
    INT* JA = A->JA;

    INT* cq   = (INT*)malloc(sizeof(INT) * (n + 1));
    INT* newr = (INT*)malloc(sizeof(INT) * (n + 1));

    for (k = 0; k < n; k++) cq[k] = k;

    group = 0;

    for (k = 0; k < n; k++) {
        if ((IA[k + 1] - IA[k]) > group) group = IA[k + 1] - IA[k];
    }
    *rowmax = group;
#if 0
    iavg = IA[n]/n ;	
    igold = (INT)MAX(iavg,group*0.618) +1;
    igold =group ;
#endif

    A->IC    = (INT*)malloc(sizeof(INT) * (group + 2));
    A->ICMAP = (INT*)malloc(sizeof(INT) * (n));

    front = n - 1;
    rear  = n - 1;

    memset(newr, -1, sizeof(INT) * (n + 1));
    memset(A->ICMAP, 0, sizeof(INT) * n);

    group    = 0;
    icount   = 0;
    A->IC[0] = 0;
    pre      = 0;

    do {
        // front = (front+1)%n;
        front++;
        if (front == n) front = 0; // front = front < n ? front : 0 ;
        i = cq[front];

        if (i <= pre) {
            A->IC[group]     = icount;
            A->ICMAP[icount] = i;
            group++;
            icount++;
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        } else if (newr[i] == group) {
            // rear = (rear +1)%n;
            rear++;
            if (rear == n) rear = 0;
            cq[rear] = i;
        } else {
            A->ICMAP[icount] = i;
            icount++;
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        }
        pre = i;
    } while (rear != front);

    A->IC[group] = icount;
    A->color     = group;

#if 0
    for(i=0; i < A->color; i++ ){
        for(j=A -> IC[i]; j < A-> IC[i+1];j++) 			
            printf("color %d  ICMAP[%d] = %d \n", i,j,A-> ICMAP[j]);
            printf( "A.color = %d A.row= %d %d\n",A -> color,A -> row,A-> IC[i+1] - A-> IC[i] );
	    getchar();
    }
#endif

    // printf(" Max Row Numbers %d avg %d igold %d max %d %d\n", group, iavg, igold,
    // (INT)MAX(iavg,group*0.618),A->IA[n]/n );
    free(cq);
    free(newr);
    *groups = group;
#endif
}

#if MULTI_COLOR_ORDER
static void generate_S_theta(dCSRmat* A, iCSRmat* S, REAL theta)
{
    const INT row = A->row, col = A->col;
    const INT row_plus_one = row + 1;
    const INT nnz          = A->IA[row] - A->IA[0];

    INT   index, i, j, begin_row, end_row;
    INT * ia = A->IA, *ja = A->JA;
    REAL* aj = A->val;

    // get the diagnal entry of A
    // dvector diag; fasp_dcsr_getdiag(0, A, &diag);

    /* generate S */
    REAL row_abs_sum;

    // copy the structure of A to S
    S->row = row;
    S->col = col;
    S->nnz = nnz;
    S->val = NULL;

    S->IA = (INT*)fasp_mem_calloc(row_plus_one, sizeof(INT));

    S->JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));

    fasp_iarray_cp(row_plus_one, ia, S->IA);
    fasp_iarray_cp(nnz, ja, S->JA);

#ifdef _OPENMP
#pragma omp parallel for private(i, j, begin_row, end_row, row_abs_sum)
#endif
    for (i = 0; i < row; ++i) {
        /* compute scaling factor and row sum */
        row_abs_sum = 0;
        begin_row   = ia[i];
        end_row     = ia[i + 1];
        for (j = begin_row; j < end_row; j++) {
            row_abs_sum += ABS(aj[j]);
        }
        row_abs_sum = row_abs_sum * theta;

        /* deal with  the element of S */
        for (j = begin_row; j < end_row; j++) {
            if ((row_abs_sum >= ABS(aj[j])) && (ja[j] != i)) {
                S->JA[j] = -1;
            }
        }
    } // end for i

    /* Compress the strength matrix */
    index = 0;
    for (i = 0; i < row; ++i) {
        S->IA[i]  = index;
        begin_row = ia[i];
        end_row   = ia[i + 1] - 1;
        for (j = begin_row; j <= end_row; j++) {
            if (S->JA[j] > -1) {
                S->JA[index] = S->JA[j];
                index++;
            }
        }
    }

    if (index > 0) {
        S->IA[row] = index;
        S->nnz     = index;
        S->JA      = (INT*)fasp_mem_realloc(S->JA, index * sizeof(INT));
    } else {
        S->nnz = 0;
        S->JA  = NULL;
    }
}
#endif

/**
 * \fn void dCSRmat_Multicoloring_Strong_Coupled(dCSRmat *A, iCSRmat *S, INT *flags, 
 *                                               INT *groups)
 *
 * \brief Use the greedy multicoloring algorithm to get color groups for the
 *        adjacency graph of A
 *
 * \param A       Input dCSRmat
 * \param S       Input iCSRmat Strong Coupled Matrix of A.
 * \param flags   Flags for the independent group
 * \param groups  Return group numbers
 *
 * \author Chunsheng Feng
 * \date   09/15/2012
 */
void dCSRmat_Multicoloring_Strong_Coupled(dCSRmat* A, iCSRmat* S, INT* flags,
                                          INT* groups)
{
#if MULTI_COLOR_ORDER
    INT  k, i, j, pre, group;
    INT  igold, iend, iavg;
    INT  icount;
    INT  front, rear;
    INT  n  = A->row;
    INT* IA = S->IA;
    INT* JA = S->JA;

    INT* cq   = (INT*)malloc(sizeof(INT) * (n + 1));
    INT* newr = (INT*)malloc(sizeof(INT) * (n + 1));

#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
    for (k = 0; k < n; k++) {
        cq[k] = k;
    }
    group = 0;
    for (k = 0; k < n; k++) {
        if ((IA[k + 1] - IA[k]) > group) group = IA[k + 1] - IA[k];
    }
    *flags = group;
#if 1
    iavg  = IA[n] / n;
    igold = (INT)MAX(iavg, group * 0.618) + 1;
    igold = group;
#endif

    A->IC    = (INT*)malloc(sizeof(INT) * (group + 2));
    A->ICMAP = (INT*)malloc(sizeof(INT) * (n + 1));

    front = n - 1;
    rear  = n - 1;

    memset(newr, -1, sizeof(INT) * (n + 1));
    memset(A->ICMAP, 0, sizeof(INT) * n);

    group    = 0;
    icount   = 0;
    A->IC[0] = 0;
    pre      = 0;

    do {
        // front = (front+1)%n;
        front++;
        if (front == n) front = 0; // front = front < n ? front : 0 ;
        i = cq[front];

        if (i <= pre) {
            A->IC[group]     = icount;
            A->ICMAP[icount] = i;
            group++;
            icount++;
#if 0
            if ((IA[i+1]-IA[i]) > igold) 
                iend = MIN(IA[i+1], (IA[i] + igold));
	    else
#endif
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        } else if (newr[i] == group) {
            // rear = (rear +1)%n;
            rear++;
            if (rear == n) rear = 0;
            cq[rear] = i;
        } else {
            A->ICMAP[icount] = i;
            icount++;
#if 0
            if ((IA[i+1] - IA[i]) > igold)  iend =MIN(IA[i+1], (IA[i] + igold));
            else
#endif
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        }
        pre = i;

    } while (rear != front);

    A->IC[group] = icount;
    A->color     = group;

#if 0
    for(i=0; i < A->color; i++ ){
        for(j=A -> IC[i]; j < A-> IC[i+1];j++) 			
            printf("color %d  ICMAP[%d] = %d \n", i,j,A-> ICMAP[j]);
        printf( "A.color = %d A.row= %d %d\n",A -> color,A -> row,A-> IC[i+1] - A-> IC[i] );
        getchar();
    }
#endif
    printf(" Max Row Numbers %d avg %d igold %d max %d %d\n", group, iavg, igold,
           (INT)MAX(iavg, group * 0.618), A->IA[n] / n);
    free(cq);
    free(newr);
    *groups = group;
#endif
}

/**
 * \fn void dCSRmat_Multicoloring_Theta(dCSRmat* A, REAL theta, INT *rowmax, 
 *                                      INT *groups)
 *
 * \brief Use the greedy multicoloring algorithm to get color groups for for the
 * adjacency graph of A
 *
 * \param A       Input dCSRmat
 * \param theta   Strength threshold parameter
 * \param rowmax  Max row nonzeros of A
 * \param groups  Return group numbers
 *
 * \author Li Zhao
 * \date   04/15/2022
 */
void dCSRmat_Multicoloring_Theta(dCSRmat* A, REAL theta, INT* rowmax, INT* groups)
{
#if MULTI_COLOR_ORDER
    INT k, i, j, pre, group;
    INT igold, iend, iavg;
    INT icount;
    INT front, rear;
    INT n = A->row;
    //---------------------------------------------------------------------------
    iCSRmat S;
    INT *   IA, *JA;
    if (theta > 0 && theta < 1.0) {
        generate_S_theta(A, &S, theta);
        IA = S.IA;
        JA = S.JA;
    } else if (theta == 1.0) {

        A->IC    = (INT*)malloc(sizeof(INT) * 2);
        A->ICMAP = (INT*)malloc(sizeof(INT) * (n + 1));
        A->IC[0] = 0;
        A->IC[1] = n;
#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
        for (k = 0; k < n; k++) A->ICMAP[k] = k;

        A->color = 1;
        *groups  = 1;
        *rowmax  = 1;
        printf("Theta = %lf \n", theta);

        return;

    } else {
        IA = A->IA;
        JA = A->JA;
    }
    //---------------------------------------------------------------------------
    INT* cq   = (INT*)malloc(sizeof(INT) * (n + 1));
    INT* newr = (INT*)malloc(sizeof(INT) * (n + 1));

#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
    for (k = 0; k < n; k++) {
        cq[k] = k;
    }
    group = 0;
    for (k = 0; k < n; k++) {
        if ((A->IA[k + 1] - A->IA[k]) > group) group = A->IA[k + 1] - A->IA[k];
    }
    *rowmax = group;

#if 0
    iavg = IA[n]/n ;	
    igold = (INT)MAX(iavg,group*0.618) +1;
    igold = group ;
#endif

    A->IC    = (INT*)malloc(sizeof(INT) * (group + 2));
    A->ICMAP = (INT*)malloc(sizeof(INT) * (n + 1));

    front = n - 1;
    rear  = n - 1;

    memset(newr, -1, sizeof(INT) * (n + 1));
    memset(A->ICMAP, 0, sizeof(INT) * n);

    group    = 0;
    icount   = 0;
    A->IC[0] = 0;
    pre      = 0;

    do {
        // front = (front+1)%n;
        front++;
        if (front == n) front = 0; // front = front < n ? front : 0 ;
        i = cq[front];

        if (i <= pre) {
            A->IC[group]     = icount;
            A->ICMAP[icount] = i;
            group++;
            icount++;
#if 0
            if ((IA[i+1]-IA[i]) > igold) 
                iend = MIN(IA[i+1], (IA[i] + igold));
	    else
#endif
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        } else if (newr[i] == group) {
            // rear = (rear +1)%n;
            rear++;
            if (rear == n) rear = 0;
            cq[rear] = i;
        } else {
            A->ICMAP[icount] = i;
            icount++;
#if 0
            if ((IA[i+1] - IA[i]) > igold)  iend =MIN(IA[i+1], (IA[i] + igold));
            else
#endif
            iend = IA[i + 1];
            for (j = IA[i]; j < iend; j++) newr[JA[j]] = group;
        }
        pre = i;

        //    printf("pre = %d\n",pre);
    } while (rear != front);

    //    printf("group\n");
    A->IC[group] = icount;
    A->color     = group;

#if 0
    for(i=0; i < A->color; i++ ){
        for(j=A -> IC[i]; j < A-> IC[i+1];j++) 			
            printf("color %d  ICMAP[%d] = %d \n", i,j,A-> ICMAP[j]);
        printf( "A.color = %d A.row= %d %d\n",A -> color,A -> row,A-> IC[i+1] - A-> IC[i] );
        getchar();
    }
    printf(" Max Row Numbers %d avg %d igold %d max %d %d\n", group, iavg, igold, (INT)MAX(iavg,group*0.618),A->IA[n]/n );
#endif
    free(cq);
    free(newr);
    if (theta > 0) {
        fasp_mem_free(S.IA);
        fasp_mem_free(S.JA);
    }
    *groups = group;
#endif
    return;
}

/* 
 * TODO: Why it is not in ItrSmootherCSR.c? Move? 
 * TODO: Add Doxygen!
 */
void fasp_smoother_dcsr_gs_multicolor(dvector* u, dCSRmat* A, dvector* b, INT L,
                                      const INT order)
{
#if MULTI_COLOR_ORDER
    const INT   nrow = A->row; // number of rows
    const INT * ia = A->IA, *ja = A->JA;
    const REAL *aj = A->val, *bval = b->val;
    REAL*       uval = u->val;

    INT  i, j, k, begin_row, end_row;
    REAL t, d = 0.0;

    INT  myid, mybegin, myend;
    INT  color = A->color;
    INT* IC    = A->IC;
    INT* ICMAP = A->ICMAP;
    INT  I;

    // From color to 0 order
    if (order == -1) {
        while (L--) {
            for (myid = color - 1; myid > -1; myid--) {
                mybegin = IC[myid];
                myend   = IC[myid + 1];
#ifdef _OPENMP
#pragma omp parallel for private(I, i, t, begin_row, end_row, k, j, d)
#endif
                for (I = mybegin; I < myend; I++) {
                    i         = ICMAP[I];
                    t         = bval[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aj[k] * uval[j];
                        else
                            d = aj[k];
                    } // end for k
                    if (ABS(d) > SMALLREAL) uval[i] = t / d;
                } // end for I
            }     // end for myid
        }         // end while
    }
    // From 0 to color order
    else {
        while (L--) {
            for (myid = 0; myid < color; myid++) {
                mybegin = IC[myid];
                myend   = IC[myid + 1];
#ifdef _OPENMP
#pragma omp parallel for private(I, i, t, begin_row, end_row, k, j, d)
#endif
                for (I = mybegin; I < myend; I++) {
                    i         = ICMAP[I];
                    t         = bval[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= aj[k] * uval[j];
                        else
                            d = aj[k];
                    } // end for k
                    if (ABS(d) > SMALLREAL) uval[i] = t / d;
                } // end for I
            }     // end for myid
        }         // end while
    }             // end if order
#else
    printf("### ERROR: MULTI_COLOR_ORDER  has not been turn on!!! \n");
#endif
    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
