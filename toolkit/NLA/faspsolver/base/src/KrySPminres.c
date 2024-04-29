/*! \file  KrySPminres.c
 *
 *  \brief Krylov subspace methods -- Preconditioned MINRES with safety net
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxVector.c, BlaArray.c,
 *         BlaSpmvBLC.c, BlaSpmvCSR.c, and BlaSpmvSTR.c
 *
 *  \note  The `best' iterative solution will be saved and used upon exit;
 *         See KryPminres.c for a version without safety net
 *
 *  Reference:
 *         Y. Saad 2003
 *         Iterative methods for sparse linear systems (2nd Edition), SIAM
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2013--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  TODO: Use one single function for all! --Chensong
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
 * \fn INT fasp_solver_dcsr_spminres (const dCSRmat *A, const dvector *b, dvector *u,
 *                                    precond *pc, const REAL tol, const INT MaxIt,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief A preconditioned minimal residual (Minres) method for solving Au=b with safety net
 *
 * \param A            Pointer to dCSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param u            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/09/2013
 */
INT fasp_solver_dcsr_spminres (const dCSRmat  *A,
                               const dvector  *b,
                               dvector        *u,
                               precond        *pc,
                               const REAL      tol,
                               const INT       MaxIt,
                               const SHORT     StopType,
                               const SHORT     PrtLvl)
{
    const SHORT  MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT    m = b->row;
    const REAL   maxdiff = tol*STAG_RATIO; // staganation tolerance
    const REAL   sol_inf_tol = SMALLREAL; // infinity norm tolerance
    
    // local variables
    INT          iter = 0, stag = 1, more_step = 1, restart_step = 1;
    REAL         absres0 = BIGREAL, absres = BIGREAL;
    REAL         normr0  = BIGREAL, relres  = BIGREAL;
    REAL         normu2, normuu, normp, normuinf, factor;
    REAL         alpha, alpha0, alpha1, temp2;
    INT          iter_best = 0; // initial best known iteration
    REAL         absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need 12*m REAL)
    REAL *work=(REAL *)fasp_mem_calloc(12*m,sizeof(REAL));
    REAL *p0=work, *p1=work+m, *p2=p1+m, *z0=p2+m, *z1=z0+m, *t0=z1+m;
    REAL *t1=t0+m, *t=t1+m, *tp=t+m, *tz=tp+m, *r=tz+m, *u_best = r+m;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe MinRes solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    // p0 = 0
    fasp_darray_set(m,p0,0.0);
    
    // r = b-A*u
    fasp_darray_cp(m,b->val,r);
    fasp_blas_dcsr_aAxpy(-1.0,A,u->val,r);
    
    // p1 = B(r)
    if ( pc != NULL )
        pc->fct(r,p1,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,r,p1); /* No preconditioner */
    
    // compute initial residuals
    switch ( StopType ) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m,r,p1));
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normu2  = MAX(SMALLREAL,fasp_blas_darray_norm2(m,u->val));
            relres  = absres0/normu2;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,iter,relres,absres0,0.0);
    
    // tp = A*p1
    fasp_blas_dcsr_mxv(A,p1,tp);
    
    // tz = B(tp)
    if ( pc != NULL )
        pc->fct(tp,tz,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,tp,tz); /* No preconditioner */
    
    // p1 = p1/normp
    normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
    normp = sqrt(normp);
    fasp_darray_cp(m,p1,t);
    fasp_darray_set(m,p1,0.0);
    fasp_blas_darray_axpy(m,1/normp,t,p1);
    
    // t0 = A*p0 = 0
    fasp_darray_set(m,t0,0.0);
    fasp_darray_cp(m,t0,z0);
    fasp_darray_cp(m,t0,t1);
    fasp_darray_cp(m,t0,z1);
    
    // t1 = tp/normp, z1 = tz/normp
    fasp_blas_darray_axpy(m,1.0/normp,tp,t1);
    fasp_blas_darray_axpy(m,1.0/normp,tz,z1);
    
    // main MinRes loop
    while ( iter++ < MaxIt ) {
        
        // alpha = <r,z1>
        alpha=fasp_blas_darray_dotprod(m,r,z1);
        
        // u = u+alpha*p1
        fasp_blas_darray_axpy(m,alpha,p1,u->val);
        
        // r = r-alpha*Ap1
        fasp_blas_darray_axpy(m,-alpha,t1,r);
        
        // compute t = A*z1 alpha1 = <z1,t>
        fasp_blas_dcsr_mxv(A,z1,t);
        alpha1=fasp_blas_darray_dotprod(m,z1,t);
        
        // compute t = A*z0 alpha0 = <z1,t>
        fasp_blas_dcsr_mxv(A,z0,t);
        alpha0=fasp_blas_darray_dotprod(m,z1,t);
        
        // p2 = z1-alpha1*p1-alpha0*p0
        fasp_darray_cp(m,z1,p2);
        fasp_blas_darray_axpy(m,-alpha1,p1,p2);
        fasp_blas_darray_axpy(m,-alpha0,p0,p2);
        
        // tp = A*p2
        fasp_blas_dcsr_mxv(A,p2,tp);
        
        // tz = B(tp)
        if ( pc != NULL )
            pc->fct(tp,tz,pc->data); /* Apply preconditioner */
        else
            fasp_darray_cp(m,tp,tz); /* No preconditioner */
        
        // p2 = p2/normp
        normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
        normp = sqrt(normp);
        fasp_darray_cp(m,p2,t);
        fasp_darray_set(m,p2,0.0);
        fasp_blas_darray_axpy(m,1/normp,t,p2);
        
        // prepare for next iteration
        fasp_darray_cp(m,p1,p0);
        fasp_darray_cp(m,p2,p1);
        fasp_darray_cp(m,t1,t0);
        fasp_darray_cp(m,z1,z0);
        
        // t1=tp/normp,z1=tz/normp
        fasp_darray_set(m,t1,0.0);
        fasp_darray_cp(m,t1,z1);
        fasp_blas_darray_axpy(m,1/normp,tp,t1);
        fasp_blas_darray_axpy(m,1/normp,tz,z1);
        
        normu2 = fasp_blas_darray_norm2(m,u->val);
        
        // compute residuals
        switch ( StopType ) {
            case STOP_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_REL_PRECRES:
                if (pc == NULL)
                    fasp_darray_cp(m,r,t);
                else
                    pc->fct(r,t,pc->data);
                temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_MOD_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normu2;
                break;
        }
        
        // compute reducation factor of residual ||r||
        factor = absres/absres0;
        
        // output iteration information if needed
        fasp_itinfo(PrtLvl,StopType,iter,relres,absres,factor);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(u) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(m,u->val,u_best);
        }
        
        // Check I: if soultion is close to zero, return ERROR_SOLVER_SOLSTAG
        normuinf = fasp_blas_darray_norminf(m, u->val);
        if (normuinf <= sol_inf_tol) {
            if ( PrtLvl > PRINT_MIN ) ITS_ZEROSOL;
            iter = ERROR_SOLVER_SOLSTAG;
            break;
        }
        
        // Check II: if staggenated, try to restart
        normuu = fasp_blas_darray_norm2(m,p1);
        normuu = ABS(alpha)*(normuu/normu2);
        
        if ( normuu < maxdiff ) {
            
            if ( stag < MaxStag ) {
                if ( PrtLvl >= PRINT_MORE ) {
                    ITS_DIFFRES(normuu,relres);
                    ITS_RESTART;
                }
            }
            
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dcsr_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            if ( relres < tol )
                break;
            else {
                if ( stag >= MaxStag ) {
                    if ( PrtLvl > PRINT_MIN ) ITS_STAGGED;
                    iter = ERROR_SOLVER_STAG;
                    break;
                }
                fasp_darray_set(m,p0,0.0);
                ++stag;
                ++restart_step;
                
                // p1 = B(r)
                if ( pc != NULL )
                    pc->fct(r,p1,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,p1); /* No preconditioner */
                
                // tp = A*p1
                fasp_blas_dcsr_mxv(A,p1,tp);
                
                // tz = B(tp)
                if ( pc != NULL )
                    pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
                else
                    fasp_darray_cp(m,tp,tz); /* No preconditioner */
                
                // p1 = p1/normp
                normp = fasp_blas_darray_dotprod(m,tz,tp);
                normp = sqrt(normp);
                fasp_darray_cp(m,p1,t);
                
                // t0 = A*p0=0
                fasp_darray_set(m,t0,0.0);
                fasp_darray_cp(m,t0,z0);
                fasp_darray_cp(m,t0,t1);
                fasp_darray_cp(m,t0,z1);
                fasp_darray_cp(m,t0,p1);
                
                fasp_blas_darray_axpy(m,1/normp,t,p1);
                
                // t1 = tp/normp, z1 = tz/normp
                fasp_blas_darray_axpy(m,1/normp,tp,t1);
                fasp_blas_darray_axpy(m,1/normp,tz,z1);
            }
        }
        
        // Check III: prevent false convergence
        if ( relres < tol ) {
            
            if ( PrtLvl >= PRINT_MORE ) ITS_COMPRES(relres);
            
            // compute residual r = b - Ax again
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dcsr_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            // check convergence
            if ( relres < tol ) break;
            
            if ( more_step >= MaxRestartStep ) {
                if ( PrtLvl > PRINT_MIN ) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }
            
            // prepare for restarting method
            fasp_darray_set(m,p0,0.0);
            ++more_step;
            ++restart_step;
            
            // p1 = B(r)
            if ( pc != NULL )
                pc->fct(r,p1,pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m,r,p1); /* No preconditioner */
            
            // tp = A*p1
            fasp_blas_dcsr_mxv(A,p1,tp);
            
            // tz = B(tp)
            if ( pc != NULL )
                pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
            else
                fasp_darray_cp(m,tp,tz); /* No preconditioner */
            
            // p1 = p1/normp
            normp = fasp_blas_darray_dotprod(m,tz,tp);
            normp = sqrt(normp);
            fasp_darray_cp(m,p1,t);
            
            // t0 = A*p0 = 0
            fasp_darray_set(m,t0,0.0);
            fasp_darray_cp(m,t0,z0);
            fasp_darray_cp(m,t0,t1);
            fasp_darray_cp(m,t0,z1);
            fasp_darray_cp(m,t0,p1);
            
            fasp_blas_darray_axpy(m,1/normp,t,p1);
            
            // t1=tp/normp,z1=tz/normp
            fasp_blas_darray_axpy(m,1/normp,tp,t1);
            fasp_blas_darray_axpy(m,1/normp,tz,z1);
            
        } // end of convergence check
        
        // update relative residual here
        absres0 = absres;
        
    } // end of the main loop
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(m,b->val,r);
        fasp_blas_dcsr_aAxpy(-1.0,A,u_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
            case STOP_REL_PRECRES:
                if ( pc != NULL )
                    pc->fct(r,t,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,t); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(m,t,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(m,u_best,u->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:  // finish iterative method
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    // clean up temp memory
    fasp_mem_free(work); work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_solver_dblc_spminres (const dBLCmat *A, const dvector *b, dvector *u,
 *                                    precond *pc, const REAL tol, const INT MaxIt,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief A preconditioned minimal residual (Minres) method for solving Au=b with safety net
 *
 * \param A            Pointer to dBLCmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param u            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/09/2013
 */
INT fasp_solver_dblc_spminres (const dBLCmat  *A,
                               const dvector  *b,
                               dvector        *u,
                               precond        *pc,
                               const REAL      tol,
                               const INT       MaxIt,
                               const SHORT     StopType,
                               const SHORT     PrtLvl)
{
    const SHORT  MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT    m = b->row;
    const REAL   maxdiff = tol*STAG_RATIO; // staganation tolerance
    const REAL   sol_inf_tol = SMALLREAL; // infinity norm tolerance
    
    // local variables
    INT          iter = 0, stag = 1, more_step = 1, restart_step = 1;
    REAL         absres0 = BIGREAL, absres = BIGREAL;
    REAL         normr0  = BIGREAL, relres  = BIGREAL;
    REAL         normu2, normuu, normp, normuinf, factor;
    REAL         alpha, alpha0, alpha1, temp2;
    INT          iter_best = 0; // initial best known iteration
    REAL         absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need 12*m REAL)
    REAL *work=(REAL *)fasp_mem_calloc(12*m,sizeof(REAL));
    REAL *p0=work, *p1=work+m, *p2=p1+m, *z0=p2+m, *z1=z0+m, *t0=z1+m;
    REAL *t1=t0+m, *t=t1+m, *tp=t+m, *tz=tp+m, *r=tz+m, *u_best = r+m;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe MinRes solver (BLC) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    // p0 = 0
    fasp_darray_set(m,p0,0.0);
    
    // r = b-A*u
    fasp_darray_cp(m,b->val,r);
    fasp_blas_dblc_aAxpy(-1.0,A,u->val,r);
    
    // p1 = B(r)
    if ( pc != NULL )
        pc->fct(r,p1,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,r,p1); /* No preconditioner */
    
    // compute initial residuals
    switch ( StopType ) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m,r,p1));
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normu2  = MAX(SMALLREAL,fasp_blas_darray_norm2(m,u->val));
            relres  = absres0/normu2;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,iter,relres,absres0,0.0);
    
    // tp = A*p1
    fasp_blas_dblc_mxv(A,p1,tp);
    
    // tz = B(tp)
    if ( pc != NULL )
        pc->fct(tp,tz,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,tp,tz); /* No preconditioner */
    
    // p1 = p1/normp
    normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
    normp = sqrt(normp);
    fasp_darray_cp(m,p1,t);
    fasp_darray_set(m,p1,0.0);
    fasp_blas_darray_axpy(m,1/normp,t,p1);
    
    // t0 = A*p0 = 0
    fasp_darray_set(m,t0,0.0);
    fasp_darray_cp(m,t0,z0);
    fasp_darray_cp(m,t0,t1);
    fasp_darray_cp(m,t0,z1);
    
    // t1 = tp/normp, z1 = tz/normp
    fasp_blas_darray_axpy(m,1.0/normp,tp,t1);
    fasp_blas_darray_axpy(m,1.0/normp,tz,z1);
    
    // main MinRes loop
    while ( iter++ < MaxIt ) {
        
        // alpha = <r,z1>
        alpha=fasp_blas_darray_dotprod(m,r,z1);
        
        // u = u+alpha*p1
        fasp_blas_darray_axpy(m,alpha,p1,u->val);
        
        // r = r-alpha*Ap1
        fasp_blas_darray_axpy(m,-alpha,t1,r);
        
        // compute t = A*z1 alpha1 = <z1,t>
        fasp_blas_dblc_mxv(A,z1,t);
        alpha1=fasp_blas_darray_dotprod(m,z1,t);
        
        // compute t = A*z0 alpha0 = <z1,t>
        fasp_blas_dblc_mxv(A,z0,t);
        alpha0=fasp_blas_darray_dotprod(m,z1,t);
        
        // p2 = z1-alpha1*p1-alpha0*p0
        fasp_darray_cp(m,z1,p2);
        fasp_blas_darray_axpy(m,-alpha1,p1,p2);
        fasp_blas_darray_axpy(m,-alpha0,p0,p2);
        
        // tp = A*p2
        fasp_blas_dblc_mxv(A,p2,tp);
        
        // tz = B(tp)
        if ( pc != NULL )
            pc->fct(tp,tz,pc->data); /* Apply preconditioner */
        else
            fasp_darray_cp(m,tp,tz); /* No preconditioner */
        
        // p2 = p2/normp
        normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
        normp = sqrt(normp);
        fasp_darray_cp(m,p2,t);
        fasp_darray_set(m,p2,0.0);
        fasp_blas_darray_axpy(m,1/normp,t,p2);
        
        // prepare for next iteration
        fasp_darray_cp(m,p1,p0);
        fasp_darray_cp(m,p2,p1);
        fasp_darray_cp(m,t1,t0);
        fasp_darray_cp(m,z1,z0);
        
        // t1=tp/normp,z1=tz/normp
        fasp_darray_set(m,t1,0.0);
        fasp_darray_cp(m,t1,z1);
        fasp_blas_darray_axpy(m,1/normp,tp,t1);
        fasp_blas_darray_axpy(m,1/normp,tz,z1);
        
        normu2 = fasp_blas_darray_norm2(m,u->val);
        
        // compute residuals
        switch ( StopType ) {
            case STOP_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_REL_PRECRES:
                if (pc == NULL)
                    fasp_darray_cp(m,r,t);
                else
                    pc->fct(r,t,pc->data);
                temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_MOD_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normu2;
                break;
        }
        
        // compute reducation factor of residual ||r||
        factor = absres/absres0;
        
        // output iteration information if needed
        fasp_itinfo(PrtLvl,StopType,iter,relres,absres,factor);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(u) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(m,u->val,u_best);
        }
        
        // Check I: if soultion is close to zero, return ERROR_SOLVER_SOLSTAG
        normuinf = fasp_blas_darray_norminf(m, u->val);
        if (normuinf <= sol_inf_tol) {
            if ( PrtLvl > PRINT_MIN ) ITS_ZEROSOL;
            iter = ERROR_SOLVER_SOLSTAG;
            break;
        }
        
        // Check II: if staggenated, try to restart
        normuu = fasp_blas_darray_norm2(m,p1);
        normuu = ABS(alpha)*(normuu/normu2);
        
        if ( normuu < maxdiff ) {
            
            if ( stag < MaxStag ) {
                if ( PrtLvl >= PRINT_MORE ) {
                    ITS_DIFFRES(normuu,relres);
                    ITS_RESTART;
                }
            }
            
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dblc_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            if ( relres < tol )
                break;
            else {
                if ( stag >= MaxStag ) {
                    if ( PrtLvl > PRINT_MIN ) ITS_STAGGED;
                    iter = ERROR_SOLVER_STAG;
                    break;
                }
                fasp_darray_set(m,p0,0.0);
                ++stag;
                ++restart_step;
                
                // p1 = B(r)
                if ( pc != NULL )
                    pc->fct(r,p1,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,p1); /* No preconditioner */
                
                // tp = A*p1
                fasp_blas_dblc_mxv(A,p1,tp);
                
                // tz = B(tp)
                if ( pc != NULL )
                    pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
                else
                    fasp_darray_cp(m,tp,tz); /* No preconditioner */
                
                // p1 = p1/normp
                normp = fasp_blas_darray_dotprod(m,tz,tp);
                normp = sqrt(normp);
                fasp_darray_cp(m,p1,t);
                
                // t0 = A*p0=0
                fasp_darray_set(m,t0,0.0);
                fasp_darray_cp(m,t0,z0);
                fasp_darray_cp(m,t0,t1);
                fasp_darray_cp(m,t0,z1);
                fasp_darray_cp(m,t0,p1);
                
                fasp_blas_darray_axpy(m,1/normp,t,p1);
                
                // t1 = tp/normp, z1 = tz/normp
                fasp_blas_darray_axpy(m,1/normp,tp,t1);
                fasp_blas_darray_axpy(m,1/normp,tz,z1);
            }
        }
        
        // Check III: prevent false convergence
        if ( relres < tol ) {
            
            if ( PrtLvl >= PRINT_MORE ) ITS_COMPRES(relres);
            
            // compute residual r = b - Ax again
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dblc_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            // check convergence
            if ( relres < tol ) break;
            
            if ( more_step >= MaxRestartStep ) {
                if ( PrtLvl > PRINT_MIN ) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }
            
            // prepare for restarting method
            fasp_darray_set(m,p0,0.0);
            ++more_step;
            ++restart_step;
            
            // p1 = B(r)
            if ( pc != NULL )
                pc->fct(r,p1,pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m,r,p1); /* No preconditioner */
            
            // tp = A*p1
            fasp_blas_dblc_mxv(A,p1,tp);
            
            // tz = B(tp)
            if ( pc != NULL )
                pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
            else
                fasp_darray_cp(m,tp,tz); /* No preconditioner */
            
            // p1 = p1/normp
            normp = fasp_blas_darray_dotprod(m,tz,tp);
            normp = sqrt(normp);
            fasp_darray_cp(m,p1,t);
            
            // t0 = A*p0 = 0
            fasp_darray_set(m,t0,0.0);
            fasp_darray_cp(m,t0,z0);
            fasp_darray_cp(m,t0,t1);
            fasp_darray_cp(m,t0,z1);
            fasp_darray_cp(m,t0,p1);
            
            fasp_blas_darray_axpy(m,1/normp,t,p1);
            
            // t1=tp/normp,z1=tz/normp
            fasp_blas_darray_axpy(m,1/normp,tp,t1);
            fasp_blas_darray_axpy(m,1/normp,tz,z1);
            
        } // end of convergence check
        
        // update relative residual here
        absres0 = absres;
        
    } // end of the main loop
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(m,b->val,r);
        fasp_blas_dblc_aAxpy(-1.0,A,u_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
            case STOP_REL_PRECRES:
                if ( pc != NULL )
                    pc->fct(r,t,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,t); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(m,t,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(m,u_best,u->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:  // finish iterative method
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    // clean up temp memory
    fasp_mem_free(work); work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_solver_dstr_spminres (const dSTRmat *A, const dvector *b, dvector *u,
 *                                    precond *pc, const REAL tol, const INT MaxIt,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief A preconditioned minimal residual (Minres) method for solving Au=b with safety net
 *
 * \param A            Pointer to dSTRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param u            Pointer to dvector: unknowns
 * \param MaxIt        Maximal number of iterations
 * \param tol          Tolerance for stopping
 * \param pc           Pointer to structure of precondition (precond)
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/09/2013
 */
INT fasp_solver_dstr_spminres (const dSTRmat  *A,
                               const dvector  *b,
                               dvector        *u,
                               precond        *pc,
                               const REAL      tol,
                               const INT       MaxIt,
                               const SHORT     StopType,
                               const SHORT     PrtLvl)
{
    const SHORT  MaxStag = MAX_STAG, MaxRestartStep = MAX_RESTART;
    const INT    m = b->row;
    const REAL   maxdiff = tol*STAG_RATIO; // staganation tolerance
    const REAL   sol_inf_tol = SMALLREAL; // infinity norm tolerance
    
    // local variables
    INT          iter = 0, stag = 1, more_step = 1, restart_step = 1;
    REAL         absres0 = BIGREAL, absres = BIGREAL;
    REAL         normr0  = BIGREAL, relres  = BIGREAL;
    REAL         normu2, normuu, normp, normuinf, factor;
    REAL         alpha, alpha0, alpha1, temp2;
    INT          iter_best = 0; // initial best known iteration
    REAL         absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need 12*m REAL)
    REAL *work=(REAL *)fasp_mem_calloc(12*m,sizeof(REAL));
    REAL *p0=work, *p1=work+m, *p2=p1+m, *z0=p2+m, *z1=z0+m, *t0=z1+m;
    REAL *t1=t0+m, *t=t1+m, *tp=t+m, *tz=tp+m, *r=tz+m, *u_best = r+m;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe MinRes solver (STR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    // p0 = 0
    fasp_darray_set(m,p0,0.0);
    
    // r = b-A*u
    fasp_darray_cp(m,b->val,r);
    fasp_blas_dstr_aAxpy(-1.0,A,u->val,r);
    
    // p1 = B(r)
    if ( pc != NULL )
        pc->fct(r,p1,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,r,p1); /* No preconditioner */
    
    // compute initial residuals
    switch ( StopType ) {
        case STOP_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_REL_PRECRES:
            absres0 = sqrt(fasp_blas_darray_dotprod(m,r,p1));
            normr0  = MAX(SMALLREAL,absres0);
            relres  = absres0/normr0;
            break;
        case STOP_MOD_REL_RES:
            absres0 = fasp_blas_darray_norm2(m,r);
            normu2  = MAX(SMALLREAL,fasp_blas_darray_norm2(m,u->val));
            relres  = absres0/normu2;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,iter,relres,absres0,0.0);
    
    // tp = A*p1
    fasp_blas_dstr_mxv(A,p1,tp);
    
    // tz = B(tp)
    if ( pc != NULL )
        pc->fct(tp,tz,pc->data); /* Apply preconditioner */
    else
        fasp_darray_cp(m,tp,tz); /* No preconditioner */
    
    // p1 = p1/normp
    normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
    normp = sqrt(normp);
    fasp_darray_cp(m,p1,t);
    fasp_darray_set(m,p1,0.0);
    fasp_blas_darray_axpy(m,1/normp,t,p1);
    
    // t0 = A*p0 = 0
    fasp_darray_set(m,t0,0.0);
    fasp_darray_cp(m,t0,z0);
    fasp_darray_cp(m,t0,t1);
    fasp_darray_cp(m,t0,z1);
    
    // t1 = tp/normp, z1 = tz/normp
    fasp_blas_darray_axpy(m,1.0/normp,tp,t1);
    fasp_blas_darray_axpy(m,1.0/normp,tz,z1);
    
    // main MinRes loop
    while ( iter++ < MaxIt ) {
        
        // alpha = <r,z1>
        alpha=fasp_blas_darray_dotprod(m,r,z1);
        
        // u = u+alpha*p1
        fasp_blas_darray_axpy(m,alpha,p1,u->val);
        
        // r = r-alpha*Ap1
        fasp_blas_darray_axpy(m,-alpha,t1,r);
        
        // compute t = A*z1 alpha1 = <z1,t>
        fasp_blas_dstr_mxv(A,z1,t);
        alpha1=fasp_blas_darray_dotprod(m,z1,t);
        
        // compute t = A*z0 alpha0 = <z1,t>
        fasp_blas_dstr_mxv(A,z0,t);
        alpha0=fasp_blas_darray_dotprod(m,z1,t);
        
        // p2 = z1-alpha1*p1-alpha0*p0
        fasp_darray_cp(m,z1,p2);
        fasp_blas_darray_axpy(m,-alpha1,p1,p2);
        fasp_blas_darray_axpy(m,-alpha0,p0,p2);
        
        // tp = A*p2
        fasp_blas_dstr_mxv(A,p2,tp);
        
        // tz = B(tp)
        if ( pc != NULL )
            pc->fct(tp,tz,pc->data); /* Apply preconditioner */
        else
            fasp_darray_cp(m,tp,tz); /* No preconditioner */
        
        // p2 = p2/normp
        normp = ABS(fasp_blas_darray_dotprod(m,tz,tp));
        normp = sqrt(normp);
        fasp_darray_cp(m,p2,t);
        fasp_darray_set(m,p2,0.0);
        fasp_blas_darray_axpy(m,1/normp,t,p2);
        
        // prepare for next iteration
        fasp_darray_cp(m,p1,p0);
        fasp_darray_cp(m,p2,p1);
        fasp_darray_cp(m,t1,t0);
        fasp_darray_cp(m,z1,z0);
        
        // t1=tp/normp,z1=tz/normp
        fasp_darray_set(m,t1,0.0);
        fasp_darray_cp(m,t1,z1);
        fasp_blas_darray_axpy(m,1/normp,tp,t1);
        fasp_blas_darray_axpy(m,1/normp,tz,z1);
        
        normu2 = fasp_blas_darray_norm2(m,u->val);
        
        // compute residuals
        switch ( StopType ) {
            case STOP_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_REL_PRECRES:
                if (pc == NULL)
                    fasp_darray_cp(m,r,t);
                else
                    pc->fct(r,t,pc->data);
                temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                absres = sqrt(temp2);
                relres = absres/normr0;
                break;
            case STOP_MOD_REL_RES:
                temp2  = fasp_blas_darray_dotprod(m,r,r);
                absres = sqrt(temp2);
                relres = absres/normu2;
                break;
        }
        
        // compute reducation factor of residual ||r||
        factor = absres/absres0;
        
        // output iteration information if needed
        fasp_itinfo(PrtLvl,StopType,iter,relres,absres,factor);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(u) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(m,u->val,u_best);
        }
        
        // Check I: if soultion is close to zero, return ERROR_SOLVER_SOLSTAG
        normuinf = fasp_blas_darray_norminf(m, u->val);
        if (normuinf <= sol_inf_tol) {
            if ( PrtLvl > PRINT_MIN ) ITS_ZEROSOL;
            iter = ERROR_SOLVER_SOLSTAG;
            break;
        }
        
        // Check II: if staggenated, try to restart
        normuu = fasp_blas_darray_norm2(m,p1);
        normuu = ABS(alpha)*(normuu/normu2);
        
        if ( normuu < maxdiff ) {
            
            if ( stag < MaxStag ) {
                if ( PrtLvl >= PRINT_MORE ) {
                    ITS_DIFFRES(normuu,relres);
                    ITS_RESTART;
                }
            }
            
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dstr_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            if ( relres < tol )
                break;
            else {
                if ( stag >= MaxStag ) {
                    if ( PrtLvl > PRINT_MIN ) ITS_STAGGED;
                    iter = ERROR_SOLVER_STAG;
                    break;
                }
                fasp_darray_set(m,p0,0.0);
                ++stag;
                ++restart_step;
                
                // p1 = B(r)
                if ( pc != NULL )
                    pc->fct(r,p1,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,p1); /* No preconditioner */
                
                // tp = A*p1
                fasp_blas_dstr_mxv(A,p1,tp);
                
                // tz = B(tp)
                if ( pc != NULL )
                    pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
                else
                    fasp_darray_cp(m,tp,tz); /* No preconditioner */
                
                // p1 = p1/normp
                normp = fasp_blas_darray_dotprod(m,tz,tp);
                normp = sqrt(normp);
                fasp_darray_cp(m,p1,t);
                
                // t0 = A*p0=0
                fasp_darray_set(m,t0,0.0);
                fasp_darray_cp(m,t0,z0);
                fasp_darray_cp(m,t0,t1);
                fasp_darray_cp(m,t0,z1);
                fasp_darray_cp(m,t0,p1);
                
                fasp_blas_darray_axpy(m,1/normp,t,p1);
                
                // t1 = tp/normp, z1 = tz/normp
                fasp_blas_darray_axpy(m,1/normp,tp,t1);
                fasp_blas_darray_axpy(m,1/normp,tz,z1);
            }
        }
        
        // Check III: prevent false convergence
        if ( relres < tol ) {
            
            if ( PrtLvl >= PRINT_MORE ) ITS_COMPRES(relres);
            
            // compute residual r = b - Ax again
            fasp_darray_cp(m,b->val,r);
            fasp_blas_dstr_aAxpy(-1.0,A,u->val,r);
            
            // compute residuals
            switch (StopType) {
                case STOP_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if (pc == NULL)
                        fasp_darray_cp(m,r,t);
                    else
                        pc->fct(r,t,pc->data);
                    temp2  = ABS(fasp_blas_darray_dotprod(m,r,t));
                    absres = sqrt(temp2);
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    temp2  = fasp_blas_darray_dotprod(m,r,r);
                    absres = sqrt(temp2);
                    relres = absres/normu2;
                    break;
            }
            
            if ( PrtLvl >= PRINT_MORE ) ITS_REALRES(relres);
            
            // check convergence
            if ( relres < tol ) break;
            
            if ( more_step >= MaxRestartStep ) {
                if ( PrtLvl > PRINT_MIN ) ITS_ZEROTOL;
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }
            
            // prepare for restarting method
            fasp_darray_set(m,p0,0.0);
            ++more_step;
            ++restart_step;
            
            // p1 = B(r)
            if ( pc != NULL )
                pc->fct(r,p1,pc->data); /* Apply preconditioner */
            else
                fasp_darray_cp(m,r,p1); /* No preconditioner */
            
            // tp = A*p1
            fasp_blas_dstr_mxv(A,p1,tp);
            
            // tz = B(tp)
            if ( pc != NULL )
                pc->fct(tp,tz,pc->data); /* Apply rreconditioner */
            else
                fasp_darray_cp(m,tp,tz); /* No preconditioner */
            
            // p1 = p1/normp
            normp = fasp_blas_darray_dotprod(m,tz,tp);
            normp = sqrt(normp);
            fasp_darray_cp(m,p1,t);
            
            // t0 = A*p0 = 0
            fasp_darray_set(m,t0,0.0);
            fasp_darray_cp(m,t0,z0);
            fasp_darray_cp(m,t0,t1);
            fasp_darray_cp(m,t0,z1);
            fasp_darray_cp(m,t0,p1);
            
            fasp_blas_darray_axpy(m,1/normp,t,p1);
            
            // t1=tp/normp,z1=tz/normp
            fasp_blas_darray_axpy(m,1/normp,tp,t1);
            fasp_blas_darray_axpy(m,1/normp,tz,z1);
            
        } // end of convergence check
        
        // update relative residual here
        absres0 = absres;
        
    } // end of the main loop
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(m,b->val,r);
        fasp_blas_dstr_aAxpy(-1.0,A,u_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
            case STOP_REL_PRECRES:
                if ( pc != NULL )
                    pc->fct(r,t,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(m,r,t); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(m,t,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(m,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(m,u_best,u->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:  // finish iterative method
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    // clean up temp memory
    fasp_mem_free(work); work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
