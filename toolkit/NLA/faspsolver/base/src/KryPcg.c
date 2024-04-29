/*! \file  KryPcg.c
 *
 *  \brief Krylov subspace methods -- Preconditioned CG
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, BlaArray.c, BlaSpmvBLC.c,
 *         BlaSpmvBSR.c, BlaSpmvCSR.c, and BlaSpmvSTR.c
 *
 *  \note  See KrySPcg.c for a safer version
 *
 *  Reference:
 *         Y. Saad 2003
 *         Iterative methods for sparse linear systems (2nd Edition), SIAM
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  Abstract algorithm
 *
 *  PCG method to solve A*x=b is to generate {x_k} to approximate x
 *
 *  Step 0. Given A, b, x_0, M
 *
 *  Step 1. Compute residual r_0 = b-A*x_0 and convergence check;
 *
 *  Step 2. Initialization z_0 = M^{-1}*r_0, p_0=z_0;
 *
 *  Step 3. Main loop ...
 *
 *    FOR k = 0:MaxIt
 *      - get step size alpha = f(r_k,z_k,p_k);
 *      - update solution: x_{k+1} = x_k + alpha*p_k;
 *      - perform stagnation check;
 *      - update residual: r_{k+1} = r_k - alpha*(A*p_k);
 *      - perform residual check;
 *      - obtain p_{k+1} using {p_0, p_1, ... , p_k};
 *      - prepare for next iteration;
 *      - print the result of k-th iteration;
 *    END FOR
 *
 *  Convergence check: norm(r)/norm(b) < tol
 *
 *  Stagnation check:
 *      - IF norm(alpha*p_k)/norm(x_{k+1}) < tol_stag
 *          -# compute r=b-A*x_{k+1};
 *          -# convergence check;
 *          -# IF ( not converged & restart_number < Max_Stag_Check ) restart;
 *      - END IF
 *
 *  Residual check:
 *      - IF norm(r_{k+1})/norm(b) < tol
 *          -# compute the real residual r = b-A*x_{k+1};
 *          -# convergence check;
 *          -# IF ( not converged & restart_number < Max_Res_Check ) restart;
 *      - END IF
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
 * \fn INT fasp_solver_dcsr_pcg (dCSRmat *A, dvector *b, dvector *u, precond *pc,
 *                               const REAL tol, const REAL abstol, const INT MaxIt,
 *                               const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned conjugate gradient method for solving Au=b
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
 * \author Chensong Zhang, Xiaozhe Hu, Shiquan Zhang
 * \date   05/06/2010
 */
INT fasp_solver_dcsr_pcg(dCSRmat* A, dvector* b, dvector* u, precond* pc,
                         const REAL tol, const REAL abstol, const INT MaxIt,
                         const SHORT StopType, const SHORT PrtLvl)
{
    const SHORT MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT   m           = b->row;
    const REAL  maxdiff     = tol * STAG_RATIO; // stagnation tolerance
    const REAL  sol_inf_tol = SMALLREAL;        // infinity norm tolerance

    // local variables
    INT  iter = 0, stag = 1, more_step = 1;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normu = BIGREAL, normr0 = BIGREAL;
    REAL reldiff, factor, normuinf;
    REAL alpha, beta, temp1, temp2;

    // allocate temp memory (need 4*m REAL numbers)
    REAL* work = (REAL*)fasp_mem_calloc(4 * m, sizeof(REAL));
    REAL *p = work, *z = work + m, *r = z + m, *t = r + m;

    // Output some info for debugging
    if (PrtLvl > PRINT_NONE) printf("\nCalling CG solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    // r = b-A*u
    fasp_darray_cp(m, b->val, r);
    fasp_blas_dcsr_aAxpy(-1.0, A, u->val, r);

    if (pc != NULL)
        pc->fct(r, z, pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m, r, z); /* No preconditioner */

    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m, r, z));
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normu   = MAX(SMALLREAL, fasp_blas_darray_norm2(m, u->val));
            relres  = absres0 / normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol || absres0 < abstol) goto FINISHED;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter, relres, absres0, 0.0);

    fasp_darray_cp(m, z, p);
    temp1 = fasp_blas_darray_dotprod(m, z, r);

    // main PCG loop
    while (iter++ < MaxIt) {

        // t = A*p
        fasp_blas_dcsr_mxv(A, p, t);

        // alpha_k = (z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, t, p);
        if (ABS(temp2) > SMALLREAL2) {
            alpha = temp1 / temp2;
        } else { // Possible breakdown
            ITS_DIVZERO;
            goto FINISHED;
        }

        // u_k = u_{k-1} + alpha_k*p_{k-1}
        fasp_blas_darray_axpy(m, alpha, p, u->val);

        // r_k = r_{k-1} - alpha_k*A*p_{k-1}
        fasp_blas_darray_axpy(m, -alpha, t, r);

        // compute norm of residual
        switch (StopType) {
            case STOP_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normr0;
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if (pc != NULL)
                    pc->fct(r, z, pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m, r, z); /* No preconditioner */
                absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                relres = absres / normr0;
                break;
            case STOP_MOD_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normu;
                break;
        }

        // compute reduction factor of residual ||r||
        factor = absres / absres0;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        if (factor > 0.9) { // Only check when converge slowly

            // Check I: if solution is close to zero, return ERROR_SOLVER_SOLSTAG
            normuinf = fasp_blas_darray_norminf(m, u->val);
            if (normuinf <= sol_inf_tol) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROSOL;
                iter = ERROR_SOLVER_SOLSTAG;
                break;
            }

            // Check II: if stagnated, try to restart
            normu = fasp_blas_darray_norm2(m, u->val);

            // compute relative difference
            reldiff = ABS(alpha) * fasp_blas_darray_norm2(m, p) / normu;
            if ((stag <= MaxStag) & (reldiff < maxdiff)) {

                if (PrtLvl >= PRINT_MORE) {
                    ITS_DIFFRES(reldiff, relres);
                    ITS_RESTART;
                }

                fasp_darray_cp(m, b->val, r);
                fasp_blas_dcsr_aAxpy(-1.0, A, u->val, r);

                // compute residual norms
                switch (StopType) {
                    case STOP_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normr0;
                        break;
                    case STOP_REL_PRECRES:
                        // z = B(r)
                        if (pc != NULL)
                            pc->fct(r, z, pc->data); /* Apply preconditioner */
                        else
                            fasp_darray_cp(m, r, z); /* No preconditioner */
                        absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                        relres = absres / normr0;
                        break;
                    case STOP_MOD_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normu;
                        break;
                }

                if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

                if (relres < tol)
                    break;
                else {
                    if (stag >= MaxStag) {
                        if (PrtLvl > PRINT_MIN) ITS_STAGGED;
                        iter = ERROR_SOLVER_STAG;
                        break;
                    }
                    fasp_darray_set(m, p, 0.0);
                    ++stag;
                }

            } // end of stagnation check!

        } // end of check I and II

        // Check III: prevent false convergence
        if (relres < tol) {

            REAL updated_relres = relres;

            // compute true residual r = b - Ax and update residual
            fasp_darray_cp(m, b->val, r);
            fasp_blas_dcsr_aAxpy(-1.0, A, u->val, r);

            // compute residual norms
            switch (StopType) {
                case STOP_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normr0;
                    break;
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                    relres = absres / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            // check convergence
            if (relres < tol) break;

            if (PrtLvl >= PRINT_MORE) {
                ITS_COMPRES(updated_relres);
                ITS_REALRES(relres);
            }

            if (more_step >= MaxRestartStep) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }

            // prepare for restarting method
            fasp_darray_set(m, p, 0.0);
            ++more_step;

        } // end of safe-guard check!

        // save residual for next iteration
        absres0 = absres;

        // compute z_k = B(r_k)
        if (StopType != STOP_REL_PRECRES) {
            if (pc != NULL)
                pc->fct(r, z, pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m, r, z); /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, z, r);
        beta  = temp2 / temp1;
        temp1 = temp2;

        // compute p_k = z_k + beta_k*p_{k-1}
        fasp_blas_darray_axpby(m, 1.0, z, beta, p);

    } // end of main PCG loop.

FINISHED: // finish iterative method
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
 * \fn INT fasp_solver_dbsr_pcg (dBSRmat *A, dvector *b, dvector *u, precond *pc,
 *                               const REAL tol, const REAL abstol, const INT MaxIt,
 *                               const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned conjugate gradient method for solving Au=b
 *
 * \param A            Pointer to dBSRmat: coefficient matrix
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
 * \date   05/26/2014
 */
INT fasp_solver_dbsr_pcg(dBSRmat* A, dvector* b, dvector* u, precond* pc,
                         const REAL tol, const REAL abstol, const INT MaxIt,
                         const SHORT StopType, const SHORT PrtLvl)
{
    const SHORT MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT   m           = b->row;
    const REAL  maxdiff     = tol * STAG_RATIO; // stagnation tolerance
    const REAL  sol_inf_tol = SMALLREAL;        // infinity norm tolerance

    // local variables
    INT  iter = 0, stag = 1, more_step = 1;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normu = BIGREAL, normr0 = BIGREAL;
    REAL reldiff, factor, normuinf;
    REAL alpha, beta, temp1, temp2;

    // allocate temp memory (need 4*m REAL numbers)
    REAL* work = (REAL*)fasp_mem_calloc(4 * m, sizeof(REAL));
    REAL *p = work, *z = work + m, *r = z + m, *t = r + m;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling CG solver (BSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    // r = b-A*u
    fasp_darray_cp(m, b->val, r);
    fasp_blas_dbsr_aAxpy(-1.0, A, u->val, r);

    if (pc != NULL)
        pc->fct(r, z, pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m, r, z); /* No preconditioner */

    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m, r, z));
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normu   = MAX(SMALLREAL, fasp_blas_darray_norm2(m, u->val));
            relres  = absres0 / normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol || absres0 < abstol) goto FINISHED;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter, relres, absres0, 0.0);

    fasp_darray_cp(m, z, p);
    temp1 = fasp_blas_darray_dotprod(m, z, r);

    // main PCG loop
    while (iter++ < MaxIt) {

        // t = A*p
        fasp_blas_dbsr_mxv(A, p, t);

        // alpha_k = (z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, t, p);
        if (ABS(temp2) > SMALLREAL2) {
            alpha = temp1 / temp2;
        } else { // Possible breakdown
            ITS_DIVZERO;
            goto FINISHED;
        }

        // u_k = u_{k-1} + alpha_k*p_{k-1}
        fasp_blas_darray_axpy(m, alpha, p, u->val);

        // r_k = r_{k-1} - alpha_k*A*p_{k-1}
        fasp_blas_darray_axpy(m, -alpha, t, r);

        // compute norm of residual
        switch (StopType) {
            case STOP_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normr0;
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if (pc != NULL)
                    pc->fct(r, z, pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m, r, z); /* No preconditioner */
                absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                relres = absres / normr0;
                break;
            case STOP_MOD_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normu;
                break;
        }

        // compute reduction factor of residual ||r||
        factor = absres / absres0;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        if (factor > 0.9) { // Only check when converge slowly

            // Check I: if solution is close to zero, return ERROR_SOLVER_SOLSTAG
            normuinf = fasp_blas_darray_norminf(m, u->val);
            if (normuinf <= sol_inf_tol) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROSOL;
                iter = ERROR_SOLVER_SOLSTAG;
                break;
            }

            // Check II: if stagnated, try to restart
            normu = fasp_blas_darray_norm2(m, u->val);

            // compute relative difference
            reldiff = ABS(alpha) * fasp_blas_darray_norm2(m, p) / normu;
            if ((stag <= MaxStag) & (reldiff < maxdiff)) {

                if (PrtLvl >= PRINT_MORE) {
                    ITS_DIFFRES(reldiff, relres);
                    ITS_RESTART;
                }

                fasp_darray_cp(m, b->val, r);
                fasp_blas_dbsr_aAxpy(-1.0, A, u->val, r);

                // compute residual norms
                switch (StopType) {
                    case STOP_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normr0;
                        break;
                    case STOP_REL_PRECRES:
                        // z = B(r)
                        if (pc != NULL)
                            pc->fct(r, z, pc->data); /* Apply preconditioner */
                        else
                            fasp_darray_cp(m, r, z); /* No preconditioner */
                        absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                        relres = absres / normr0;
                        break;
                    case STOP_MOD_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normu;
                        break;
                }

                if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

                if (relres < tol)
                    break;
                else {
                    if (stag >= MaxStag) {
                        if (PrtLvl > PRINT_MIN) ITS_STAGGED;
                        iter = ERROR_SOLVER_STAG;
                        break;
                    }
                    fasp_darray_set(m, p, 0.0);
                    ++stag;
                }

            } // end of stagnation check!

        } // end of check I and II

        // Check III: prevent false convergence
        if (relres < tol) {

            REAL updated_relres = relres;

            // compute true residual r = b - Ax and update residual
            fasp_darray_cp(m, b->val, r);
            fasp_blas_dbsr_aAxpy(-1.0, A, u->val, r);

            // compute residual norms
            switch (StopType) {
                case STOP_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normr0;
                    break;
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                    relres = absres / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            // check convergence
            if (relres < tol) break;

            if (PrtLvl >= PRINT_MORE) {
                ITS_COMPRES(updated_relres);
                ITS_REALRES(relres);
            }

            if (more_step >= MaxRestartStep) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }

            // prepare for restarting method
            fasp_darray_set(m, p, 0.0);
            ++more_step;

        } // end of safe-guard check!

        // save residual for next iteration
        absres0 = absres;

        // compute z_k = B(r_k)
        if (StopType != STOP_REL_PRECRES) {
            if (pc != NULL)
                pc->fct(r, z, pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m, r, z); /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, z, r);
        beta  = temp2 / temp1;
        temp1 = temp2;

        // compute p_k = z_k + beta_k*p_{k-1}
        fasp_blas_darray_axpby(m, 1.0, z, beta, p);

    } // end of main PCG loop.

FINISHED: // finish iterative method
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
 * \fn INT fasp_solver_dblc_pcg (dBLCmat *A, dvector *b, dvector *u, precond *pc,
 *                               const REAL tol, const REAL abstol, const INT MaxIt,
 *                               const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned conjugate gradient method for solving Au=b
 *
 * \param A            Pointer to dBLCmat: coefficient matrix
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
 * \date   05/24/2010
 *
 * Modified by Chensong Zhang on 03/28/2013
 */
INT fasp_solver_dblc_pcg(dBLCmat* A, dvector* b, dvector* u, precond* pc,
                         const REAL tol, const REAL abstol, const INT MaxIt,
                         const SHORT StopType, const SHORT PrtLvl)
{
    const SHORT MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT   m           = b->row;
    const REAL  maxdiff     = tol * STAG_RATIO; // stagnation tolerance
    const REAL  sol_inf_tol = SMALLREAL;        // infinity norm tolerance

    // local variables
    INT  iter = 0, stag = 1, more_step = 1;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normu = BIGREAL, normr0 = BIGREAL;
    REAL reldiff, factor, normuinf;
    REAL alpha, beta, temp1, temp2;

    // allocate temp memory (need 4*m REAL numbers)
    REAL* work = (REAL*)fasp_mem_calloc(4 * m, sizeof(REAL));
    REAL *p = work, *z = work + m, *r = z + m, *t = r + m;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling CG solver (BLC) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    // r = b-A*u
    fasp_darray_cp(m, b->val, r);
    fasp_blas_dblc_aAxpy(-1.0, A, u->val, r);

    if (pc != NULL)
        pc->fct(r, z, pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m, r, z); /* No preconditioner */

    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m, r, z));
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normu   = MAX(SMALLREAL, fasp_blas_darray_norm2(m, u->val));
            relres  = absres0 / normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol || absres0 < abstol) goto FINISHED;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter, relres, absres0, 0.0);

    fasp_darray_cp(m, z, p);
    temp1 = fasp_blas_darray_dotprod(m, z, r);

    // main PCG loop
    while (iter++ < MaxIt) {

        // t = A*p
        fasp_blas_dblc_mxv(A, p, t);

        // alpha_k = (z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, t, p);
        if (ABS(temp2) > SMALLREAL2) {
            alpha = temp1 / temp2;
        } else { // Possible breakdown
            ITS_DIVZERO;
            goto FINISHED;
        }

        // u_k = u_{k-1} + alpha_k*p_{k-1}
        fasp_blas_darray_axpy(m, alpha, p, u->val);

        // r_k = r_{k-1} - alpha_k*A*p_{k-1}
        fasp_blas_darray_axpy(m, -alpha, t, r);

        // compute norm of residual
        switch (StopType) {
            case STOP_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normr0;
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if (pc != NULL)
                    pc->fct(r, z, pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m, r, z); /* No preconditioner */
                absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                relres = absres / normr0;
                break;
            case STOP_MOD_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normu;
                break;
        }

        // compute reduction factor of residual ||r||
        factor = absres / absres0;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        if (factor > 0.9) { // Only check when converge slowly

            // Check I: if solution is close to zero, return ERROR_SOLVER_SOLSTAG
            normuinf = fasp_blas_darray_norminf(m, u->val);
            if (normuinf <= sol_inf_tol) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROSOL;
                iter = ERROR_SOLVER_SOLSTAG;
                break;
            }

            // Check II: if stagnated, try to restart
            normu = fasp_blas_darray_norm2(m, u->val);

            // compute relative difference
            reldiff = ABS(alpha) * fasp_blas_darray_norm2(m, p) / normu;
            if ((stag <= MaxStag) & (reldiff < maxdiff)) {

                if (PrtLvl >= PRINT_MORE) {
                    ITS_DIFFRES(reldiff, relres);
                    ITS_RESTART;
                }

                fasp_darray_cp(m, b->val, r);
                fasp_blas_dblc_aAxpy(-1.0, A, u->val, r);

                // compute residual norms
                switch (StopType) {
                    case STOP_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normr0;
                        break;
                    case STOP_REL_PRECRES:
                        // z = B(r)
                        if (pc != NULL)
                            pc->fct(r, z, pc->data); /* Apply preconditioner */
                        else
                            fasp_darray_cp(m, r, z); /* No preconditioner */
                        absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                        relres = absres / normr0;
                        break;
                    case STOP_MOD_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normu;
                        break;
                }

                if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

                if (relres < tol)
                    break;
                else {
                    if (stag >= MaxStag) {
                        if (PrtLvl > PRINT_MIN) ITS_STAGGED;
                        iter = ERROR_SOLVER_STAG;
                        break;
                    }
                    fasp_darray_set(m, p, 0.0);
                    ++stag;
                }

            } // end of stagnation check!

        } // end of check I and II

        // Check III: prevent false convergence
        if (relres < tol) {

            REAL updated_relres = relres;

            // compute true residual r = b - Ax and update residual
            fasp_darray_cp(m, b->val, r);
            fasp_blas_dblc_aAxpy(-1.0, A, u->val, r);

            // compute residual norms
            switch (StopType) {
                case STOP_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normr0;
                    break;
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                    relres = absres / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            // check convergence
            if (relres < tol) break;

            if (PrtLvl >= PRINT_MORE) {
                ITS_COMPRES(updated_relres);
                ITS_REALRES(relres);
            }

            if (more_step >= MaxRestartStep) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }

            // prepare for restarting method
            fasp_darray_set(m, p, 0.0);
            ++more_step;

        } // end of safe-guard check!

        // save residual for next iteration
        absres0 = absres;

        // compute z_k = B(r_k)
        if (StopType != STOP_REL_PRECRES) {
            if (pc != NULL)
                pc->fct(r, z, pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m, r, z); /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, z, r);
        beta  = temp2 / temp1;
        temp1 = temp2;

        // compute p_k = z_k + beta_k*p_{k-1}
        fasp_blas_darray_axpby(m, 1.0, z, beta, p);

    } // end of main PCG loop.

FINISHED: // finish iterative method
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
 * \fn INT fasp_solver_dstr_pcg (dSTRmat *A, dvector *b, dvector *u, precond *pc,
 *                               const REAL tol, const REAL abstol, const INT MaxIt,
 *                               const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned conjugate gradient method for solving Au=b
 *
 * \param A            Pointer to dSTRmat: coefficient matrix
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
 * \author Zhiyang Zhou
 * \date   04/25/2010
 *
 * Modified by Chensong Zhang on 03/28/2013
 */
INT fasp_solver_dstr_pcg(dSTRmat* A, dvector* b, dvector* u, precond* pc,
                         const REAL tol, const REAL abstol, const INT MaxIt,
                         const SHORT StopType, const SHORT PrtLvl)
{
    const SHORT MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT   m           = b->row;
    const REAL  maxdiff     = tol * STAG_RATIO; // stagnation tolerance
    const REAL  sol_inf_tol = SMALLREAL;        // infinity norm tolerance

    // local variables
    INT  iter = 0, stag = 1, more_step = 1;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normu = BIGREAL, normr0 = BIGREAL;
    REAL reldiff, factor, normuinf;
    REAL alpha, beta, temp1, temp2;

    // allocate temp memory (need 4*m REAL numbers)
    REAL* work = (REAL*)fasp_mem_calloc(4 * m, sizeof(REAL));
    REAL *p = work, *z = work + m, *r = z + m, *t = r + m;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling CG solver (STR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    // r = b-A*u
    fasp_darray_cp(m, b->val, r);
    fasp_blas_dstr_aAxpy(-1.0, A, u->val, r);

    if (pc != NULL)
        pc->fct(r, z, pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m, r, z); /* No preconditioner */

    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m, r, z));
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normu   = MAX(SMALLREAL, fasp_blas_darray_norm2(m, u->val));
            relres  = absres0 / normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol || absres0 < abstol) goto FINISHED;

    // output iteration information if needed
    fasp_itinfo(PrtLvl, StopType, iter, relres, absres0, 0.0);

    fasp_darray_cp(m, z, p);
    temp1 = fasp_blas_darray_dotprod(m, z, r);

    // main PCG loop
    while (iter++ < MaxIt) {

        // t = A*p
        fasp_blas_dstr_mxv(A, p, t);

        // alpha_k = (z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, t, p);
        if (ABS(temp2) > SMALLREAL2) {
            alpha = temp1 / temp2;
        } else { // Possible breakdown
            ITS_DIVZERO;
            goto FINISHED;
        }

        // u_k = u_{k-1} + alpha_k*p_{k-1}
        fasp_blas_darray_axpy(m, alpha, p, u->val);

        // r_k = r_{k-1} - alpha_k*A*p_{k-1}
        fasp_blas_darray_axpy(m, -alpha, t, r);

        // compute norm of residual
        switch (StopType) {
            case STOP_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normr0;
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if (pc != NULL)
                    pc->fct(r, z, pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m, r, z); /* No preconditioner */
                absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                relres = absres / normr0;
                break;
            case STOP_MOD_REL_RES:
                absres = fasp_blas_darray_norm2(m, r);
                relres = absres / normu;
                break;
        }

        // compute reduction factor of residual ||r||
        factor = absres / absres0;

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        if (factor > 0.9) { // Only check when converge slowly

            // Check I: if solution is close to zero, return ERROR_SOLVER_SOLSTAG
            normuinf = fasp_blas_darray_norminf(m, u->val);
            if (normuinf <= sol_inf_tol) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROSOL;
                iter = ERROR_SOLVER_SOLSTAG;
                break;
            }

            // Check II: if stagnated, try to restart
            normu = fasp_blas_darray_norm2(m, u->val);

            // compute relative difference
            reldiff = ABS(alpha) * fasp_blas_darray_norm2(m, p) / normu;
            if ((stag <= MaxStag) & (reldiff < maxdiff)) {

                if (PrtLvl >= PRINT_MORE) {
                    ITS_DIFFRES(reldiff, relres);
                    ITS_RESTART;
                }

                fasp_darray_cp(m, b->val, r);
                fasp_blas_dstr_aAxpy(-1.0, A, u->val, r);

                // compute residual norms
                switch (StopType) {
                    case STOP_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normr0;
                        break;
                    case STOP_REL_PRECRES:
                        // z = B(r)
                        if (pc != NULL)
                            pc->fct(r, z, pc->data); /* Apply preconditioner */
                        else
                            fasp_darray_cp(m, r, z); /* No preconditioner */
                        absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                        relres = absres / normr0;
                        break;
                    case STOP_MOD_REL_RES:
                        absres = fasp_blas_darray_norm2(m, r);
                        relres = absres / normu;
                        break;
                }

                if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

                if (relres < tol)
                    break;
                else {
                    if (stag >= MaxStag) {
                        if (PrtLvl > PRINT_MIN) ITS_STAGGED;
                        iter = ERROR_SOLVER_STAG;
                        break;
                    }
                    fasp_darray_set(m, p, 0.0);
                    ++stag;
                }

            } // end of stagnation check!

        } // end of check I and II

        // Check III: prevent false convergence
        if (relres < tol) {

            REAL updated_relres = relres;

            // compute true residual r = b - Ax and update residual
            fasp_darray_cp(m, b->val, r);
            fasp_blas_dstr_aAxpy(-1.0, A, u->val, r);

            // compute residual norms
            switch (StopType) {
                case STOP_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normr0;
                    break;
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    absres = sqrt(ABS(fasp_blas_darray_dotprod(m, z, r)));
                    relres = absres / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            // check convergence
            if (relres < tol) break;

            if (PrtLvl >= PRINT_MORE) {
                ITS_COMPRES(updated_relres);
                ITS_REALRES(relres);
            }

            if (more_step >= MaxRestartStep) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }

            // prepare for restarting method
            fasp_darray_set(m, p, 0.0);
            ++more_step;

        } // end of safe-guard check!

        // save residual for next iteration
        absres0 = absres;

        // compute z_k = B(r_k)
        if (StopType != STOP_REL_PRECRES) {
            if (pc != NULL)
                pc->fct(r, z, pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m, r, z); /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, z, r);
        beta  = temp2 / temp1;
        temp1 = temp2;

        // compute p_k = z_k + beta_k*p_{k-1}
        fasp_blas_darray_axpby(m, 1.0, z, beta, p);

    } // end of main PCG loop.

FINISHED: // finish iterative method
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
 * \fn INT fasp_solver_pcg (mxv_matfree *mf, dvector *b, dvector *u, precond *pc,
 *                          const REAL tol, const INT MaxIt, const REAL abstol,
 *                          const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Preconditioned conjugate gradient (CG) method for solving Au=b
 *
 * \param mf           Pointer to mxv_matfree: spmv operation
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
 * \author Chensong Zhang, Xiaozhe Hu, Shiquan Zhang
 * \date   05/06/2010
 */
INT fasp_solver_pcg(mxv_matfree* mf, dvector* b, dvector* u, precond* pc,
                    const REAL tol, const REAL abstol, const INT MaxIt,
                    const SHORT StopType, const SHORT PrtLvl)
{
    const SHORT MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT   m           = b->row;
    const REAL  maxdiff     = tol * STAG_RATIO; // staganation tolerance
    const REAL  sol_inf_tol = SMALLREAL;        // infinity norm tolerance

    // local variables
    INT  iter    = 0, stag, more_step, restart_step;
    REAL absres0 = BIGREAL, absres = BIGREAL;
    REAL relres = BIGREAL, normu = BIGREAL, normr0 = BIGREAL;
    REAL reldiff, factor, infnormu;
    REAL alpha, beta, temp1, temp2;

    // allocate temp memory (need 4*m REAL numbers)
    REAL* work = (REAL*)fasp_mem_calloc(4 * m, sizeof(REAL));
    REAL *p = work, *z = work + m, *r = z + m, *t = r + m;

    // Output some info for debuging
    if (PrtLvl > PRINT_NONE) printf("\nCalling CG solver (MatFree) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    // initialize counters
    stag         = 1;
    more_step    = 1;
    restart_step = 1;

    // r = b-A*u
    mf->fct(mf->data, u->val, r);
    fasp_blas_darray_axpby(m, 1.0, b->val, -1.0, r);

    if (pc != NULL)
        pc->fct(r, z, pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m, r, z); /* No preconditioner */

    // compute initial relative residual
    switch (StopType) {
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m, r, z));
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m, r);
            normu   = MAX(SMALLREAL, fasp_blas_darray_norm2(m, u->val));
            relres  = absres0 / normu;
            break;
        default:
            absres0 = fasp_blas_darray_norm2(m, r);
            normr0  = MAX(SMALLREAL, absres0);
            relres  = absres0 / normr0;
            break;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol || absres0 < abstol) goto FINISHED;

    fasp_darray_cp(m, z, p);
    temp1 = fasp_blas_darray_dotprod(m, z, r);

    while (iter++ < MaxIt) {

        // t=A*p
        mf->fct(mf->data, p, t);

        // alpha_k=(z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, t, p);
        alpha = temp1 / temp2;

        // u_k=u_{k-1} + alpha_k*p_{k-1}
        fasp_blas_darray_axpy(m, alpha, p, u->val);

        // r_k=r_{k-1} - alpha_k*A*p_{k-1}
        fasp_blas_darray_axpy(m, -alpha, t, r);
        absres = fasp_blas_darray_norm2(m, r);

        // compute reducation factor of residual ||r||
        factor = absres / absres0;

        // compute relative residual
        switch (StopType) {
            case STOP_REL_PRECRES:
                // z = B(r)
                if (pc != NULL)
                    pc->fct(r, z, pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m, r, z); /* No preconditioner */
                temp2  = fasp_blas_darray_dotprod(m, z, r);
                relres = sqrt(ABS(temp2)) / normr0;
                break;
            case STOP_MOD_REL_RES:
                relres = absres / normu;
                break;
            default:
                relres = absres / normr0;
                break;
        }

        // output iteration information if needed
        fasp_itinfo(PrtLvl, StopType, iter, relres, absres, factor);

        // solution check, if soultion is too small, return ERROR_SOLVER_SOLSTAG.
        infnormu = fasp_blas_darray_norminf(m, u->val);
        if (infnormu <= sol_inf_tol) {
            if (PrtLvl > PRINT_MIN) ITS_ZEROSOL;
            iter = ERROR_SOLVER_SOLSTAG;
            break;
        }

        // compute relative difference
        normu   = fasp_blas_darray_norm2(m, u->val);
        reldiff = ABS(alpha) * fasp_blas_darray_norm2(m, p) / normu;

        // stagnation check
        if ((stag <= MaxStag) & (reldiff < maxdiff)) {

            if (PrtLvl >= PRINT_MORE) {
                ITS_DIFFRES(reldiff, relres);
                ITS_RESTART;
            }

            mf->fct(mf->data, u->val, r);
            fasp_blas_darray_axpby(m, 1.0, b->val, -1.0, r);
            absres = fasp_blas_darray_norm2(m, r);

            // relative residual
            switch (StopType) {
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    temp2  = fasp_blas_darray_dotprod(m, z, r);
                    relres = sqrt(ABS(temp2)) / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    relres = absres / normu;
                    break;
                default:
                    relres = absres / normr0;
                    break;
            }

            if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

            if (relres < tol)
                break;
            else {
                if (stag >= MaxStag) {
                    if (PrtLvl > PRINT_MIN) ITS_STAGGED;
                    iter = ERROR_SOLVER_STAG;
                    break;
                }
                fasp_darray_set(m, p, 0.0);
                ++stag;
                ++restart_step;
            }
        } // end of stagnation check!

        // safe-guard check
        if (relres < tol) {
            if (PrtLvl >= PRINT_MORE) ITS_COMPRES(relres);

            mf->fct(mf->data, u->val, r);
            fasp_blas_darray_axpby(m, 1.0, b->val, -1.0, r);

            // relative residual
            switch (StopType) {
                case STOP_REL_PRECRES:
                    // z = B(r)
                    if (pc != NULL)
                        pc->fct(r, z, pc->data); /* Apply preconditioner */
                    else
                        fasp_darray_cp(m, r, z); /* No preconditioner */
                    temp2  = fasp_blas_darray_dotprod(m, z, r);
                    relres = sqrt(ABS(temp2)) / normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normu;
                    break;
                default:
                    absres = fasp_blas_darray_norm2(m, r);
                    relres = absres / normr0;
                    break;
            }

            if (PrtLvl >= PRINT_MORE) ITS_REALRES(relres);

            // check convergence
            if (relres < tol) break;

            if (more_step >= MaxRestartStep) {
                if (PrtLvl > PRINT_MIN) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }

            // prepare for restarting method
            fasp_darray_set(m, p, 0.0);
            ++more_step;
            ++restart_step;

        } // end of safe-guard check!

        // update relative residual here
        absres0 = absres;

        // compute z_k = B(r_k)
        if (StopType != STOP_REL_PRECRES) {
            if (pc != NULL)
                pc->fct(r, z, pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m, r, z); /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = fasp_blas_darray_dotprod(m, z, r);
        beta  = temp2 / temp1;
        temp1 = temp2;

        // compute p_k = z_k + beta_k*p_{k-1}
        fasp_blas_darray_axpby(m, 1.0, z, beta, p);

    } // end of main PCG loop.

FINISHED: // finish iterative method
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
