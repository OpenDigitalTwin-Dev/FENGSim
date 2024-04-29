/*! \file  KryPgcg.c
 *
 *  \brief Krylov subspace methods -- Preconditioned generalized CG
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, BlaArray.c, and BlaSpmvCSR.c
 *
 *  Reference:
 *         Concus, P. and Golub, G.H. and O'Leary, D.P.
 *         A Generalized Conjugate Gradient Method for the Numerical:
 *         Solution of Elliptic Partial Differential Equations,
 *         Computer Science Department, Stanford University, 1976
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  TODO: Not completely implemented yet! --Chensong
 */

#include <math.h>
#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "KryUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_dcsr_pgcg (dCSRmat *A, dvector *b, dvector *u, precond *pc,
 *                                const REAL tol, const REAL abstol, const INT MaxIt,
 *                                const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned generilzed conjugate gradient (GCG) method for solving Au=b
 *
 * \param A            Pointer to dCSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param u            Pointer to dvector: unknowns
 * \param pc           Pointer to precond: structure of precondition
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param MaxIt        Maximal number of iterations
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/01/2012
 *
 * Modified by Chensong Zhang on 05/01/2012
 */
INT fasp_solver_dcsr_pgcg(dCSRmat* A, dvector* b, dvector* u, precond* pc,
                          const REAL tol, const REAL abstol, const INT MaxIt,
                          const SHORT StopType, const SHORT PrtLvl)
{
    INT  iter = 0, m = A->row, i;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normb = BIGREAL;
    REAL alpha, factor;

    // allocate temp memory
    REAL* work = (REAL*)fasp_mem_calloc(2 * m + MaxIt + MaxIt * m, sizeof(REAL));

    REAL *r, *Br, *beta, *p;
    r    = work;
    Br   = r + m;
    beta = Br + m;
    p    = beta + MaxIt;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling GCG solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    normb = fasp_blas_darray_norm2(m, b->val);

    // -------------------------------------
    // 1st iteration (Steepest descent)
    // -------------------------------------
    // r = b-A*u
    fasp_darray_cp(m, b->val, r);
    fasp_blas_dcsr_aAxpy(-1.0, A, u->val, r);

    // Br
    if (pc != NULL)
        pc->fct(r, p, pc->data); /* Preconditioning */
    else
        fasp_darray_cp(m, r, p); /* No preconditioner, B=I */

    // alpha = (p'r)/(p'Ap)
    alpha = fasp_blas_darray_dotprod(m, r, p) / fasp_blas_dcsr_vmv(A, p, p);

    // u = u + alpha *p
    fasp_blas_darray_axpy(m, alpha, p, u->val);

    // r = r - alpha *Ap
    fasp_blas_dcsr_aAxpy((-1.0 * alpha), A, p, r);

    // norm(r), factor
    absres = fasp_blas_darray_norm2(m, r);
    factor = absres / absres0;

    // compute relative residual
    relres = absres / normb;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

    // update relative residual here
    absres0 = absres;

    for (iter = 1; iter < MaxIt; iter++) {

        // Br
        if (pc != NULL)
            pc->fct(r, Br, pc->data); // Preconditioning
        else
            fasp_darray_cp(m, r, Br); // No preconditioner, B=I

        // form p
        fasp_darray_cp(m, Br, p + iter * m);

        for (i = 0; i < iter; i++) {
            beta[i] = (-1.0) * (fasp_blas_dcsr_vmv(A, Br, p + i * m) /
                                fasp_blas_dcsr_vmv(A, p + i * m, p + i * m));

            fasp_blas_darray_axpy(m, beta[i], p + i * m, p + iter * m);
        }

        // -------------------------------------
        // next iteration
        // -------------------------------------

        // alpha = (p'r)/(p'Ap)
        alpha = fasp_blas_darray_dotprod(m, r, p + iter * m) /
                fasp_blas_dcsr_vmv(A, p + iter * m, p + iter * m);

        // u = u + alpha *p
        fasp_blas_darray_axpy(m, alpha, p + iter * m, u->val);

        // r = r - alpha *Ap
        fasp_blas_dcsr_aAxpy((-1.0 * alpha), A, p + iter * m, r);

        // norm(r), factor
        absres = fasp_blas_darray_norm2(m, r);
        factor = absres / absres0;

        // compute relative residual
        relres = absres / normb;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        if (relres < tol || absres < abstol) break;

        // update relative residual here
        absres0 = absres;

    } // end of main GCG loop.

    // finish iterative method
    if (PrtLvl > PRINT_NONE) ITS_FINAL(iter, MaxIt, relres);

    // clean up temp memory
    fasp_mem_free(work);
    work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (iter > MaxIt)
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_solver_pgcg (mxv_matfree *mf, dvector *b, dvector *u, precond *pc,
 *                           const REAL tol, const REAL abstol, const INT MaxIt,
 *                           const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned generilzed conjugate gradient (GCG) method for solving Au=b
 *
 * \param mf           Pointer to mxv_matfree: spmv operation
 * \param b            Pointer to dvector: right hand side
 * \param u            Pointer to dvector: unknowns
 * \param pc           Pointer to precond: structure of precondition
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param MaxIt        Maximal number of iterations
 * \param StopType     Stopping criteria type -- DOES not support this parameter
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/01/2012
 *
 * Modified by Feiteng Huang on 09/26/2012: matrix free
 */
INT fasp_solver_pgcg(mxv_matfree* mf, dvector* b, dvector* u, precond* pc,
                     const REAL tol, const REAL abstol, const INT MaxIt,
                     const SHORT StopType, const SHORT PrtLvl)
{
    INT  iter = 0, m = b->row, i;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normb = BIGREAL;
    REAL alpha, factor, gama_1, gama_2;

    // allocate temp memory
    REAL* work = (REAL*)fasp_mem_calloc(3 * m + MaxIt + MaxIt * m, sizeof(REAL));

    REAL *r, *Br, *beta, *p, *q;
    q    = work;
    r    = q + m;
    Br   = r + m;
    beta = Br + m;
    p    = beta + MaxIt;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling GCG solver (MatFree) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    normb = fasp_blas_darray_norm2(m, b->val);

    // -------------------------------------
    // 1st iteration (Steepest descent)
    // -------------------------------------
    // r = b-A*u
    mf->fct(mf->data, u->val, r);
    fasp_blas_darray_axpby(m, 1.0, b->val, -1.0, r);

    // Br
    if (pc != NULL)
        pc->fct(r, p, pc->data); /* Preconditioning */
    else
        fasp_darray_cp(m, r, p); /* No preconditioner, B=I */

    // alpha = (p'r)/(p'Ap)
    mf->fct(mf->data, p, q);
    alpha = fasp_blas_darray_dotprod(m, r, p) / fasp_blas_darray_dotprod(m, p, q);

    // u = u + alpha *p
    fasp_blas_darray_axpy(m, alpha, p, u->val);

    // r = r - alpha *Ap
    mf->fct(mf->data, p, q);
    fasp_blas_darray_axpby(m, (-1.0 * alpha), q, 1.0, r);

    // norm(r), factor
    absres = fasp_blas_darray_norm2(m, r);
    factor = absres / absres0;

    // compute relative residual
    relres = absres / normb;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter + 1, relres, absres, factor);

    // update relative residual here
    absres0 = absres;

    for (iter = 1; iter < MaxIt; iter++) {

        // Br
        if (pc != NULL)
            pc->fct(r, Br, pc->data); // Preconditioning
        else
            fasp_darray_cp(m, r, Br); // No preconditioner, B=I

        // form p
        fasp_darray_cp(m, Br, p + iter * m);

        for (i = 0; i < iter; i++) {
            mf->fct(mf->data, Br, q);
            gama_1 = fasp_blas_darray_dotprod(m, p + i * m, q);
            mf->fct(mf->data, p + i * m, q);
            gama_2  = fasp_blas_darray_dotprod(m, p + i * m, q);
            beta[i] = (-1.0) * (gama_1 / gama_2);

            fasp_blas_darray_axpy(m, beta[i], p + i * m, p + iter * m);
        }

        // -------------------------------------
        // next iteration
        // -------------------------------------

        // alpha = (p'r)/(p'Ap)
        mf->fct(mf->data, p + iter * m, q);
        alpha = fasp_blas_darray_dotprod(m, r, p + iter * m) /
                fasp_blas_darray_dotprod(m, q, p + iter * m);

        // u = u + alpha *p
        fasp_blas_darray_axpy(m, alpha, p + iter * m, u->val);

        // r = r - alpha *Ap
        mf->fct(mf->data, p + iter * m, q);
        fasp_blas_darray_axpby(m, (-1.0 * alpha), q, 1.0, r);

        // norm(r), factor
        absres = fasp_blas_darray_norm2(m, r);
        factor = absres / absres0;

        // compute relative residual
        relres = absres / normb;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter + 1, relres, absres, factor);

        if (relres < tol || absres < abstol) break;

        // update relative residual here
        absres0 = absres;

    } // end of main GCG loop.

    // finish iterative method
    if (PrtLvl > PRINT_NONE) ITS_FINAL(iter, MaxIt, relres);

    // clean up temp memory
    fasp_mem_free(work);
    work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (iter > MaxIt)
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
