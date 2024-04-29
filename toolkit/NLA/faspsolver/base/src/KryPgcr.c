/*! \file  KryPgcr.c
 *
 *  \brief Krylov subspace methods -- Preconditioned GCR
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, BlaArray.c, BlaSpmvCSR.c,
 *         and BlaVector.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2014--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  TODO: abstol not used yet! --Chensong
 */

#include "fasp.h"
#include "fasp_functs.h"
#include <math.h>

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "KryUtil.inl"

static void dense_aAtxpby(INT, INT, REAL*, REAL, REAL*, REAL, REAL*);

/**
 * \fn INT fasp_solver_dcsr_pgcr (dCSRmat *A,  dvector *b, dvector *x, precond *pc,
 *                                const REAL tol, const REAL abstol, const INT MaxIt,
 *                                const SHORT restart, const SHORT StopType,
 *                                const SHORT PrtLvl)
 *
 * \brief A preconditioned GCR method for solving Au=b
 *
 * \param A         Pointer to coefficient matrix
 * \param b         Pointer to dvector of right hand side
 * \param x         Pointer to dvector of dofs
 * \param pc        Pointer to structure of precondition (precond)
 * \param tol       Tolerance for relative residual
 * \param abstol    Tolerance for absolute residual
 * \param MaxIt     Maximal number of iterations
 * \param restart   Restart number for GCR
 * \param StopType  Stopping type
 * \param PrtLvl    How much information to print out
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * Reference: YVAN NOTAY "AN AGGREGATION-BASED ALGEBRAIC MULTIGRID METHOD"
 *
 * \author Zheng Li
 * \date   12/23/2014
 */
INT fasp_solver_dcsr_pgcr(dCSRmat* A, dvector* b, dvector* x, precond* pc,
                          const REAL tol, const REAL abstol, const INT MaxIt,
                          const SHORT restart, const SHORT StopType, const SHORT PrtLvl)
{
    const INT n = b->row;

    // local variables
    INT iter = 0;
    int i, j, k, rst = -1; // must be signed! -zcs

    REAL gamma, alpha, beta, checktol;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL;

    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL * c = NULL, *z = NULL, *alp = NULL, *tmpx = NULL;
    REAL * norms = NULL, *r = NULL, *work = NULL;
    REAL** h = NULL;

    INT  Restart  = MIN(restart, MaxIt);
    LONG worksize = n + 2 * Restart * n + Restart + Restart;

    // Output some info for debugging
    if (PrtLvl > PRINT_NONE) printf("\nCalling GCR solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    work = (REAL*)fasp_mem_calloc(worksize, sizeof(REAL));

    /* check whether memory is enough for GCR */
    while ((work == NULL) && (Restart > 5)) {
        Restart  = Restart - 5;
        worksize = n + 2 * Restart * n + Restart + Restart;
        work     = (REAL*)fasp_mem_calloc(worksize, sizeof(REAL));
    }

    if (work == NULL) {
        printf("### ERROR: No enough memory for GCR! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    if (PrtLvl > PRINT_MIN && Restart < restart) {
        printf("### WARNING: GCR restart number set to %d!\n", Restart);
    }

    r    = work;
    z    = r + n;
    c    = z + Restart * n;
    alp  = c + Restart * n;
    tmpx = alp + Restart;

    h = (REAL**)fasp_mem_calloc(Restart, sizeof(REAL*));
    for (i = 0; i < Restart; i++) h[i] = (REAL*)fasp_mem_calloc(Restart, sizeof(REAL));

    norms = (REAL*)fasp_mem_calloc(MaxIt + 1, sizeof(REAL));

    // r = b-A*x
    fasp_darray_cp(n, b->val, r);
    fasp_blas_dcsr_aAxpy(-1.0, A, x->val, r);

    absres = fasp_blas_darray_dotprod(n, r, r);

    absres0 = MAX(SMALLREAL, absres);

    relres = absres / absres0;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, 0, relres, sqrt(absres0), 0.0);

    // store initial residual
    norms[0] = relres;

    checktol = MAX(tol * tol * absres0, absres * 1.0e-4);

    while (iter < MaxIt && sqrt(relres) > tol) {

        i = -1;
        rst++;

        while (i < Restart - 1 && iter < MaxIt) {

            i++;
            iter++;

            // z = B^-1r
            if (pc == NULL)
                fasp_darray_cp(n, r, &z[i * n]);
            else
                pc->fct(r, &z[i * n], pc->data);

            // c = Az
            fasp_blas_dcsr_mxv(A, &z[i * n], &c[i * n]);

            /* Modified Gram_Schmidt orthogonalization */
            for (j = 0; j < i; j++) {
                gamma   = fasp_blas_darray_dotprod(n, &c[j * n], &c[i * n]);
                h[i][j] = gamma / h[j][j];
                fasp_blas_darray_axpy(n, -h[i][j], &c[j * n], &c[i * n]);
            }
            // gamma = (c,c)
            gamma = fasp_blas_darray_dotprod(n, &c[i * n], &c[i * n]);

            h[i][i] = gamma;

            // alpha = (c, r)
            alpha = fasp_blas_darray_dotprod(n, &c[i * n], r);

            beta = alpha / gamma;

            alp[i] = beta;

            // r = r - beta*c
            fasp_blas_darray_axpy(n, -beta, &c[i * n], r);

            // equivalent to ||r||_2
            absres = absres - alpha * alpha / gamma;

            if (absres < checktol) {
                absres   = fasp_blas_darray_dotprod(n, r, r);
                checktol = MAX(tol * tol * absres0, absres * 1.0e-4);
            }

            relres = absres / absres0;

            norms[iter] = relres;

            fasp_itinfo(PrtLvl, StopType, iter, sqrt(relres), sqrt(absres),
                        sqrt(norms[iter] / norms[iter - 1]));

            if (sqrt(relres) < tol) break;
        }

        for (k = i; k >= 0; k--) {
            tmpx[k] = alp[k];
            for (j = 0; j < k; ++j) {
                alp[j] -= h[k][j] * tmpx[k];
            }
        }

        if (rst == 0)
            dense_aAtxpby(n, i + 1, z, 1.0, tmpx, 0.0, x->val);
        else
            dense_aAtxpby(n, i + 1, z, 1.0, tmpx, 1.0, x->val);
    }

    if (PrtLvl > PRINT_NONE) ITS_FINAL(iter, MaxIt, sqrt(relres));

    // clean up memory
    for (i = 0; i < Restart; i++) {
        fasp_mem_free(h[i]);
        h[i] = NULL;
    }
    fasp_mem_free(h);
    h = NULL;

    fasp_mem_free(work);
    work = NULL;
    fasp_mem_free(norms);
    norms = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (iter >= MaxIt)
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_solver_dblc_pgcr (dBLCmat *A,  dvector *b, dvector *x, precond *pc,
 *                                const REAL tol, const REAL abstol, const INT MaxIt,
 *                                const SHORT restart, const SHORT StopType,
 *                                const SHORT PrtLvl)
 *
 * \brief A preconditioned GCR method for solving Au=b
 *
 * \param A         Pointer to coefficient matrix
 * \param b         Pointer to dvector of right hand side
 * \param x         Pointer to dvector of dofs
 * \param pc        Pointer to structure of precondition (precond)
 * \param tol       Tolerance for relative residual
 * \param abstol    Tolerance for absolute residual
 * \param MaxIt     Maximal number of iterations
 * \param restart   Restart number for GCR
 * \param StopType  Stopping type
 * \param PrtLvl    How much information to print out
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * Reference: YVAN NOTAY "AN AGGREGATION-BASED ALGEBRAIC MULTIGRID METHOD"
 *
 * \author Zheng Li
 * \date   12/23/2014
 */
INT fasp_solver_dblc_pgcr(dBLCmat* A, dvector* b, dvector* x, precond* pc,
                          const REAL tol, const REAL abstol, const INT MaxIt,
                          const SHORT restart, const SHORT StopType, const SHORT PrtLvl)
{
    const INT n = b->row;

    // local variables
    INT iter = 0;
    int i, j, k, rst = -1; // must be signed! -zcs

    REAL gamma, alpha, beta, checktol;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL;

    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL * c = NULL, *z = NULL, *alp = NULL, *tmpx = NULL;
    REAL * norms = NULL, *r = NULL, *work = NULL;
    REAL** h = NULL;

    INT  Restart  = MIN(restart, MaxIt);
    LONG worksize = n + 2 * Restart * n + Restart + Restart;

    // Output some info for debugging
    if (PrtLvl > PRINT_NONE) printf("\nCalling GCR solver (BLC) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    work = (REAL*)fasp_mem_calloc(worksize, sizeof(REAL));

    /* check whether memory is enough for GCR */
    while ((work == NULL) && (Restart > 5)) {
        Restart  = Restart - 5;
        worksize = n + 2 * Restart * n + Restart + Restart;
        work     = (REAL*)fasp_mem_calloc(worksize, sizeof(REAL));
    }

    if (work == NULL) {
        printf("### ERROR: No enough memory for GCR! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    if (PrtLvl > PRINT_MIN && Restart < restart) {
        printf("### WARNING: GCR restart number set to %d!\n", Restart);
    }

    r    = work;
    z    = r + n;
    c    = z + Restart * n;
    alp  = c + Restart * n;
    tmpx = alp + Restart;

    h = (REAL**)fasp_mem_calloc(Restart, sizeof(REAL*));
    for (i = 0; i < Restart; i++) h[i] = (REAL*)fasp_mem_calloc(Restart, sizeof(REAL));

    norms = (REAL*)fasp_mem_calloc(MaxIt + 1, sizeof(REAL));

    // r = b-A*x
    fasp_darray_cp(n, b->val, r);
    fasp_blas_dblc_aAxpy(-1.0, A, x->val, r);

    absres = fasp_blas_darray_dotprod(n, r, r);

    absres0 = MAX(SMALLREAL, absres);

    relres = absres / absres0;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, 0, relres, sqrt(absres0), 0.0);

    // store initial residual
    norms[0] = relres;

    checktol = MAX(tol * tol * absres0, absres * 1.0e-4);

    while (iter < MaxIt && sqrt(relres) > tol) {

        i = 0;
        rst++;
        while (i < Restart && iter < MaxIt) {

            iter++;

            // z = B^-1r
            if (pc == NULL)
                fasp_darray_cp(n, r, &z[i * n]);
            else
                pc->fct(r, &z[i * n], pc->data);

            // c = Az
            fasp_blas_dblc_mxv(A, &z[i * n], &c[i * n]);

            /* Modified Gram_Schmidt orthogonalization */
            for (j = 0; j < i; j++) {
                gamma   = fasp_blas_darray_dotprod(n, &c[j * n], &c[i * n]);
                h[i][j] = gamma / h[j][j];
                fasp_blas_darray_axpy(n, -h[i][j], &c[j * n], &c[i * n]);
            }
            // gamma = (c,c)
            gamma = fasp_blas_darray_dotprod(n, &c[i * n], &c[i * n]);

            h[i][i] = gamma;

            // alpha = (c, r)
            alpha = fasp_blas_darray_dotprod(n, &c[i * n], r);

            beta = alpha / gamma;

            alp[i] = beta;

            // r = r - beta*c
            fasp_blas_darray_axpy(n, -beta, &c[i * n], r);

            // equivalent to ||r||_2
            absres = absres - alpha * alpha / gamma;

            if (absres < checktol) {
                absres   = fasp_blas_darray_dotprod(n, r, r);
                checktol = MAX(tol * tol * absres0, absres * 1.0e-4);
            }

            relres = absres / absres0;

            norms[iter] = relres;

            fasp_itinfo(PrtLvl, StopType, iter, sqrt(relres), sqrt(absres),
                        sqrt(norms[iter] / norms[iter - 1]));

            if (sqrt(relres) < tol) break;

            i++;
        }

        for (k = i; k >= 0; k--) {
            tmpx[k] = alp[k];
            for (j = 0; j < k; ++j) {
                alp[j] -= h[k][j] * tmpx[k];
            }
        }

        if (rst == 0)
            dense_aAtxpby(n, i + 1, z, 1.0, tmpx, 0.0, x->val);
        else
            dense_aAtxpby(n, i + 1, z, 1.0, tmpx, 1.0, x->val);
    }

    if (PrtLvl > PRINT_NONE) ITS_FINAL(iter, MaxIt, sqrt(relres));

    // clean up memory
    for (i = 0; i < Restart; i++) {
        fasp_mem_free(h[i]);
        h[i] = NULL;
    }
    fasp_mem_free(h);
    h = NULL;

    fasp_mem_free(work);
    work = NULL;
    fasp_mem_free(norms);
    norms = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (iter >= MaxIt)
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*---------------------------------*/
/*--    Private Functions        --*/
/*---------------------------------*/

/**
 * \fn static void dense_aAtxpby (INT n, INT m, REAL *A, REAL alpha,
 *                                REAL *x, REAL beta, REAL *y)
 *
 * \brief  y = alpha*A^T*x + beta*y
 *
 * \param n     Pointer to row
 * \param m     Pointer to col
 * \param A     Pointer to CSR matrix
 * \param alpha Real factor alpha
 * \param x     Pointer to dvector of right hand side
 * \param beta  Real factor beta
 * \param y     Maximal number of iterations
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 *
 * \warning This is a special function. Move it to blas_smat.c if needed else where.
 */
static void dense_aAtxpby(INT n, INT m, REAL* A, REAL alpha, REAL* x, REAL beta,
                          REAL* y)
{
    INT i, j;

    for (i = 0; i < m; i++) fasp_blas_darray_ax(n, x[i], &A[i * n]);

    for (j = 1; j < m; j++) {
        for (i = 0; i < n; i++) {
            A[i] += A[i + j * n];
        }
    }

    fasp_blas_darray_axpby(n, alpha, A, beta, y);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
