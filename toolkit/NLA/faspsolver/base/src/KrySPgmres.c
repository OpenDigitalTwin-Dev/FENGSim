/*! \file  KrySPgmres.c
 *
 *  \brief Krylov subspace methods -- Preconditioned GMRes with safety net
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxVector.c, BlaArray.c,
 *         BlaSpmvBLC.c, BlaSpmvBSR.c, BlaSpmvCSR.c, and BlaSpmvSTR.c
 *
 *  \note  See also pgmres.c for a variable restarting version.
 *
 *  \note  The `best' iterative solution will be saved and used upon exit;
 *         See KryPgmres.c for a version without safety net
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

/*!
 * \fn INT fasp_solver_dcsr_spgmres (const dCSRmat *A, const dvector *b, dvector *x,
 *                                   precond *pc, const REAL tol, const INT MaxIt,
 *                                   SHORT restart, const SHORT StopType,
 *                                   const SHORT PrtLvl)
 *
 * \brief Preconditioned GMRES method for solving Au=b with safe-guard
 *
 * \param A            Pointer to dCSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/05/2013
 *
 * Modified by Chunsheng Feng on 07/22/2013: Add adapt memory allocate
 */
INT fasp_solver_dcsr_spgmres (const dCSRmat  *A,
                              const dvector  *b,
                              dvector        *x,
                              precond        *pc,
                              const REAL      tol,
                              const INT       MaxIt,
                              SHORT           restart,
                              const SHORT     StopType,
                              const SHORT     PrtLvl)
{
    const INT  n         = b->row;
    const INT  MIN_ITER  = 0;
    const REAL maxdiff   = tol*STAG_RATIO; // staganation tolerance
    const REAL epsmac    = SMALLREAL;
    
    // local variables
    INT      iter = 0;
    INT      restart1 = restart + 1;
    int      i, j, k; // must be signed! -zcs

    REAL     r_norm, r_normb, gamma, t;
    REAL     normr0 = BIGREAL, absres = BIGREAL;
    REAL     relres = BIGREAL, normu  = BIGREAL;
    
    INT      iter_best = 0; // initial best known iteration
    REAL     absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL    *c = NULL, *s = NULL, *rs = NULL;
    REAL    *norms = NULL, *r = NULL, *w = NULL;
    REAL    *work = NULL, *x_best = NULL;
    REAL    **p = NULL, **hh = NULL;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe GMRes solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    /* allocate memory and setup temp work space */
    work  = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
    
    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (restart > 5 ) ) {
        restart = restart - 5 ;
        work = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
        printf("### WARNING: GMRES restart number set to %d!\n", restart);
        restart1 = restart + 1;
    }
    
    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    
    p     = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    hh    = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    norms = (REAL *) fasp_mem_calloc(MaxIt+1, sizeof(REAL));
    
    r = work; w = r + n; rs = w + n; c = rs + restart1;
    x_best = c + restart; s = x_best + n;
    
    for ( i = 0; i < restart1; i++ ) p[i] = s + restart + i*n;
    
    for ( i = 0; i < restart1; i++ ) hh[i] = p[restart] + n + i*restart;
    
    // r = b-A*x
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dcsr_aAxpy(-1.0, A, x->val, p[0]);
    
    r_norm  = fasp_blas_darray_norm2(n,p[0]);
    
    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            normr0  = MAX(SMALLREAL,r_norm);
            relres  = r_norm/normr0;
            break;
        case STOP_REL_PRECRES:
            if ( pc == NULL )
                fasp_darray_cp(n, p[0], r);
            else
                pc->fct(p[0], r, pc->data);
            r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
            normr0  = MAX(SMALLREAL,r_normb);
            relres  = r_normb/normr0;
            break;
        case STOP_MOD_REL_RES:
            normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
            normr0  = r_norm;
            relres  = normr0/normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,0,relres,normr0,0.0);
    
    // store initial residual
    norms[0] = relres;
    
    /* outer iteration cycle */
    while ( iter < MaxIt ) {
        
        rs[0] = r_norm;
        
        t = 1.0 / r_norm;
        
        fasp_blas_darray_ax(n, t, p[0]);
        
        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while ( i < restart && iter < MaxIt ) {
            
            i++; iter++;
            
            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], r);
            else
                pc->fct(p[i-1], r, pc->data);
            
            fasp_blas_dcsr_mxv(A, r, p[i]);
            
            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if (t != 0.0) {
                t = 1.0/t;
                fasp_blas_darray_ax(n, t, p[i]);
            }
            
            for (j = 1; j < i; ++j) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t= hh[i][i-1]*hh[i][i-1];
            t+= hh[i-1][i-1]*hh[i-1][i-1];
            
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1]*rs[i-1];
            rs[i-1] =  c[i-1]*rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            
            absres = r_norm = fabs(rs[i]);
            
            relres = absres/normr0;
            
            norms[iter] = relres;
            
            // output iteration information if needed
            fasp_itinfo(PrtLvl, StopType, iter, relres, absres,
                        norms[iter]/norms[iter-1]);
            
            // should we exit restart cycle
            if ( relres <= tol && iter >= MIN_ITER ) break;
            
        } /* end of restart cycle */
        
        /* compute solution, first solve upper triangular system */
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];
            
            t += rs[k];
            rs[k] = t / hh[k][k];
        }
        
        fasp_darray_cp(n, p[i-1], w);
        
        fasp_blas_darray_ax(n, rs[i-1], w);
        
        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], w);
        
        /* apply preconditioner */
        if ( pc == NULL )
            fasp_darray_cp(n, w, r);
        else
            pc->fct(w, r, pc->data);
        
        fasp_blas_darray_axpy(n, 1.0, r, x->val);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(x) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(n,x->val,x_best);
        }
        
        // Check: prevent false convergence
        if ( relres <= tol && iter >= MIN_ITER ) {
            
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dcsr_aAxpy(-1.0, A, x->val, r);
            
            r_norm = fasp_blas_darray_norm2(n, r);
            
            switch ( StopType ) {
                case STOP_REL_RES:
                    absres = r_norm;
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL )
                        fasp_darray_cp(n, r, w);
                    else
                        pc->fct(r, w, pc->data);
                    absres = sqrt(fasp_blas_darray_dotprod(n,w,r));
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = r_norm;
                    normu  = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres = absres/normu;
                    break;
            }
            
            norms[iter] = relres;
            
            if ( relres <= tol ) {
                break;
            }
            else {
                // Need to restart
                fasp_darray_cp(n, r, p[0]); i = 0;
            }
            
        } /* end of convergence check */
        
        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }
        
        if ( i ) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);
        
        for ( j = i-1 ; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);
        
        if ( i ) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }
        
    } /* end of main while loop */
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(n,b->val,r);
        fasp_blas_dcsr_aAxpy(-1.0,A,x_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if ( pc != NULL )
                    pc->fct(r,w,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(n,r,w); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(n,w,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(n,x_best,x->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    /*-------------------------------------------
     * Clean up workspace
     *------------------------------------------*/
    fasp_mem_free(work);  work  = NULL;
    fasp_mem_free(p);     p     = NULL;
    fasp_mem_free(hh);    hh    = NULL;
    fasp_mem_free(norms); norms = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*!
 * \fn INT fasp_solver_dbsr_spgmres (const dBSRmat *A, const dvector *b, dvector *x,
 *                                   precond *pc, const REAL tol, const INT MaxIt,
 *                                   SHORT restart, const SHORT StopType,
 *                                   const SHORT PrtLvl)
 *
 * \brief Preconditioned GMRES method for solving Au=b with safe-guard
 *
 * \param A            Pointer to dBSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/05/2013
 */
INT fasp_solver_dbsr_spgmres (const dBSRmat  *A,
                              const dvector  *b,
                              dvector        *x,
                              precond        *pc,
                              const REAL      tol,
                              const INT       MaxIt,
                              SHORT           restart,
                              const SHORT     StopType,
                              const SHORT     PrtLvl)
{
    const INT  n         = b->row;
    const INT  MIN_ITER  = 0;
    const REAL maxdiff   = tol*STAG_RATIO; // staganation tolerance
    const REAL epsmac    = SMALLREAL;

    // local variables
    INT      iter = 0;
    INT      restart1 = restart + 1;
    int      i, j, k; // must be signed! -zcs

    REAL     r_norm, r_normb, gamma, t;
    REAL     normr0 = BIGREAL, absres = BIGREAL;
    REAL     relres = BIGREAL, normu  = BIGREAL;

    INT      iter_best = 0; // initial best known iteration
    REAL     absres_best = BIGREAL; // initial best known residual

    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL    *c = NULL, *s = NULL, *rs = NULL;
    REAL    *norms = NULL, *r = NULL, *w = NULL;
    REAL    *work = NULL, *x_best = NULL;
    REAL    **p = NULL, **hh = NULL;

    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe GMRes solver (BSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    /* allocate memory and setup temp work space */
    work  = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));

    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (restart > 5 ) ) {
        restart = restart - 5 ;
        work = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
        printf("### WARNING: GMRES restart number set to %d!\n", restart);
        restart1 = restart + 1;
    }

    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    p     = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    hh    = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    norms = (REAL *) fasp_mem_calloc(MaxIt+1, sizeof(REAL));

    r = work; w = r + n; rs = w + n; c = rs + restart1;
    x_best = c + restart; s = x_best + n;

    for ( i = 0; i < restart1; i++ ) p[i] = s + restart + i*n;

    for ( i = 0; i < restart1; i++ ) hh[i] = p[restart] + n + i*restart;

    // r = b-A*x
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dbsr_aAxpy(-1.0, A, x->val, p[0]);

    r_norm  = fasp_blas_darray_norm2(n,p[0]);

    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            normr0  = MAX(SMALLREAL,r_norm);
            relres  = r_norm/normr0;
            break;
        case STOP_REL_PRECRES:
            if ( pc == NULL )
                fasp_darray_cp(n, p[0], r);
            else
                pc->fct(p[0], r, pc->data);
            r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
            normr0  = MAX(SMALLREAL,r_normb);
            relres  = r_normb/normr0;
            break;
        case STOP_MOD_REL_RES:
            normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
            normr0  = r_norm;
            relres  = normr0/normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }

    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;

    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,0,relres,normr0,0.0);

    // store initial residual
    norms[0] = relres;

    /* outer iteration cycle */
    while ( iter < MaxIt ) {

        rs[0] = r_norm;

        t = 1.0 / r_norm;

        fasp_blas_darray_ax(n, t, p[0]);

        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while ( i < restart && iter < MaxIt ) {

            i++; iter++;

            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], r);
            else
                pc->fct(p[i-1], r, pc->data);

            fasp_blas_dbsr_mxv(A, r, p[i]);

            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if (t != 0.0) {
                t = 1.0/t;
                fasp_blas_darray_ax(n, t, p[i]);
            }

            for (j = 1; j < i; ++j) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t= hh[i][i-1]*hh[i][i-1];
            t+= hh[i-1][i-1]*hh[i-1][i-1];

            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1]*rs[i-1];
            rs[i-1] =  c[i-1]*rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];

            absres = r_norm = fabs(rs[i]);

            relres = absres/normr0;

            norms[iter] = relres;

            // output iteration information if needed
            fasp_itinfo(PrtLvl, StopType, iter, relres, absres,
                        norms[iter]/norms[iter-1]);

            // should we exit restart cycle
            if ( relres <= tol && iter >= MIN_ITER ) break;

        } /* end of restart cycle */

        /* compute solution, first solve upper triangular system */
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];

            t += rs[k];
            rs[k] = t / hh[k][k];
        }

        fasp_darray_cp(n, p[i-1], w);

        fasp_blas_darray_ax(n, rs[i-1], w);

        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], w);

        /* apply preconditioner */
        if ( pc == NULL )
            fasp_darray_cp(n, w, r);
        else
            pc->fct(w, r, pc->data);

        fasp_blas_darray_axpy(n, 1.0, r, x->val);

        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(x) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }

        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(n,x->val,x_best);
        }

        // Check: prevent false convergence
        if ( relres <= tol && iter >= MIN_ITER ) {

            fasp_darray_cp(n, b->val, r);
            fasp_blas_dbsr_aAxpy(-1.0, A, x->val, r);

            r_norm = fasp_blas_darray_norm2(n, r);

            switch ( StopType ) {
                case STOP_REL_RES:
                    absres = r_norm;
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL )
                        fasp_darray_cp(n, r, w);
                    else
                        pc->fct(r, w, pc->data);
                    absres = sqrt(fasp_blas_darray_dotprod(n,w,r));
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = r_norm;
                    normu  = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres = absres/normu;
                    break;
            }

            norms[iter] = relres;

            if ( relres <= tol ) {
                break;
            }
            else {
                // Need to restart
                fasp_darray_cp(n, r, p[0]); i = 0;
            }

        } /* end of convergence check */

        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }

        if ( i ) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);

        for ( j = i-1 ; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);

        if ( i ) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }

    } /* end of main while loop */

RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {

        // compute best residual
        fasp_darray_cp(n,b->val,r);
        fasp_blas_dbsr_aAxpy(-1.0,A,x_best,r);

        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if ( pc != NULL )
                    pc->fct(r,w,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(n,r,w); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(n,w,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
        }

        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(n,x_best,x->val);
            relres = absres_best / normr0;
        }
    }

FINISHED:
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);

    /*-------------------------------------------
     * Clean up workspace
     *------------------------------------------*/
    fasp_mem_free(work);  work  = NULL;
    fasp_mem_free(p);     p     = NULL;
    fasp_mem_free(hh);    hh    = NULL;
    fasp_mem_free(norms); norms = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_solver_dblc_spgmres (const dBLCmat *A, const dvector *b, dvector *x,
 *                                   precond *pc, const REAL tol, const INT MaxIt,
 *                                   SHORT restart, const SHORT StopType,
 *                                   const SHORT PrtLvl)
 *
 * \brief Preconditioned GMRES method for solving Au=b with safe-guard
 *
 * \param A            Pointer to dBLCmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/05/2013
 */
INT fasp_solver_dblc_spgmres (const dBLCmat  *A,
                              const dvector  *b,
                              dvector        *x,
                              precond        *pc,
                              const REAL      tol,
                              const INT       MaxIt,
                              SHORT           restart,
                              const SHORT     StopType,
                              const SHORT     PrtLvl)
{
    const INT  n         = b->row;
    const INT  MIN_ITER  = 0;
    const REAL maxdiff   = tol*STAG_RATIO; // staganation tolerance
    const REAL epsmac    = SMALLREAL;
    
    // local variables
    INT      iter = 0;
    INT      restart1 = restart + 1;
    int      i, j, k; // must be signed! -zcs

    REAL     r_norm, r_normb, gamma, t;
    REAL     normr0 = BIGREAL, absres = BIGREAL;
    REAL     relres = BIGREAL, normu  = BIGREAL;
    
    INT      iter_best = 0; // initial best known iteration
    REAL     absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL    *c = NULL, *s = NULL, *rs = NULL;
    REAL    *norms = NULL, *r = NULL, *w = NULL;
    REAL    *work = NULL, *x_best = NULL;
    REAL    **p = NULL, **hh = NULL;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe GMRes solver (BLC) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    /* allocate memory and setup temp work space */
    work  = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
    
    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (restart > 5 ) ) {
        restart = restart - 5 ;
        work = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
        printf("### WARNING: GMRES restart number set to %d!\n", restart);
        restart1 = restart + 1;
    }
    
    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    
    p     = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    hh    = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    norms = (REAL *) fasp_mem_calloc(MaxIt+1, sizeof(REAL));
    
    r = work; w = r + n; rs = w + n; c = rs + restart1;
    x_best = c + restart; s = x_best + n;
    
    for ( i = 0; i < restart1; i++ ) p[i] = s + restart + i*n;
    
    for ( i = 0; i < restart1; i++ ) hh[i] = p[restart] + n + i*restart;
    
    // r = b-A*x
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dblc_aAxpy(-1.0, A, x->val, p[0]);
    
    r_norm  = fasp_blas_darray_norm2(n,p[0]);
    
    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            normr0  = MAX(SMALLREAL,r_norm);
            relres  = r_norm/normr0;
            break;
        case STOP_REL_PRECRES:
            if ( pc == NULL )
                fasp_darray_cp(n, p[0], r);
            else
                pc->fct(p[0], r, pc->data);
            r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
            normr0  = MAX(SMALLREAL,r_normb);
            relres  = r_normb/normr0;
            break;
        case STOP_MOD_REL_RES:
            normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
            normr0  = r_norm;
            relres  = normr0/normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,0,relres,normr0,0.0);
    
    // store initial residual
    norms[0] = relres;
    
    /* outer iteration cycle */
    while ( iter < MaxIt ) {
        
        rs[0] = r_norm;
        
        t = 1.0 / r_norm;
        
        fasp_blas_darray_ax(n, t, p[0]);
        
        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while ( i < restart && iter < MaxIt ) {
            
            i++; iter++;
            
            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], r);
            else
                pc->fct(p[i-1], r, pc->data);
            
            fasp_blas_dblc_mxv(A, r, p[i]);
            
            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if (t != 0.0) {
                t = 1.0/t;
                fasp_blas_darray_ax(n, t, p[i]);
            }
            
            for (j = 1; j < i; ++j) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t= hh[i][i-1]*hh[i][i-1];
            t+= hh[i-1][i-1]*hh[i-1][i-1];
            
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1]*rs[i-1];
            rs[i-1] =  c[i-1]*rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            
            absres = r_norm = fabs(rs[i]);
            
            relres = absres/normr0;
            
            norms[iter] = relres;
            
            // output iteration information if needed
            fasp_itinfo(PrtLvl, StopType, iter, relres, absres,
                        norms[iter]/norms[iter-1]);
            
            // should we exit restart cycle
            if ( relres <= tol && iter >= MIN_ITER ) break;
            
        } /* end of restart cycle */
        
        /* compute solution, first solve upper triangular system */
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];
            
            t += rs[k];
            rs[k] = t / hh[k][k];
        }
        
        fasp_darray_cp(n, p[i-1], w);
        
        fasp_blas_darray_ax(n, rs[i-1], w);
        
        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], w);
        
        /* apply preconditioner */
        if ( pc == NULL )
            fasp_darray_cp(n, w, r);
        else
            pc->fct(w, r, pc->data);
        
        fasp_blas_darray_axpy(n, 1.0, r, x->val);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(x) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(n,x->val,x_best);
        }
        
        // Check: prevent false convergence
        if ( relres <= tol && iter >= MIN_ITER ) {
            
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dblc_aAxpy(-1.0, A, x->val, r);
            
            r_norm = fasp_blas_darray_norm2(n, r);
            
            switch ( StopType ) {
                case STOP_REL_RES:
                    absres = r_norm;
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL )
                        fasp_darray_cp(n, r, w);
                    else
                        pc->fct(r, w, pc->data);
                    absres = sqrt(fasp_blas_darray_dotprod(n,w,r));
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = r_norm;
                    normu  = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres = absres/normu;
                    break;
            }
            
            norms[iter] = relres;
            
            if ( relres <= tol ) {
                break;
            }
            else {
                // Need to restart
                fasp_darray_cp(n, r, p[0]); i = 0;
            }
            
        } /* end of convergence check */
        
        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }
        
        if ( i ) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);
        
        for ( j = i-1 ; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);
        
        if ( i ) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }
        
    } /* end of main while loop */
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(n,b->val,r);
        fasp_blas_dblc_aAxpy(-1.0,A,x_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if ( pc != NULL )
                    pc->fct(r,w,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(n,r,w); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(n,w,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(n,x_best,x->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    /*-------------------------------------------
     * Clean up workspace
     *------------------------------------------*/
    fasp_mem_free(work);  work  = NULL;
    fasp_mem_free(p);     p     = NULL;
    fasp_mem_free(hh);    hh    = NULL;
    fasp_mem_free(norms); norms = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*!
 * \fn INT fasp_solver_dstr_spgmres (const dSTRmat *A, const dvector *b, dvector *x,
 *                                   precond *pc, const REAL tol, const INT MaxIt,
 *                                   SHORT restart, const SHORT StopType,
 *                                   const SHORT PrtLvl)
 *
 * \brief Preconditioned GMRES method for solving Au=b with safe-guard
 *
 * \param A            Pointer to dSTRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to structure of precondition (precond)
 * \param tol          Tolerance for stopping
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/05/2013
 */
INT fasp_solver_dstr_spgmres (const dSTRmat  *A,
                              const dvector  *b,
                              dvector        *x,
                              precond        *pc,
                              const REAL      tol,
                              const INT       MaxIt,
                              SHORT           restart,
                              const SHORT     StopType,
                              const SHORT     PrtLvl)
{
    const INT  n         = b->row;
    const INT  MIN_ITER  = 0;
    const REAL maxdiff   = tol*STAG_RATIO; // staganation tolerance
    const REAL epsmac    = SMALLREAL;
    
    // local variables
    INT      iter = 0;
    INT      restart1 = restart + 1;
    int      i, j, k; // must be signed! -zcs

    REAL     r_norm, r_normb, gamma, t;
    REAL     normr0 = BIGREAL, absres = BIGREAL;
    REAL     relres = BIGREAL, normu  = BIGREAL;
    
    INT      iter_best = 0; // initial best known iteration
    REAL     absres_best = BIGREAL; // initial best known residual
    
    // allocate temp memory (need about (restart+4)*n REAL numbers)
    REAL    *c = NULL, *s = NULL, *rs = NULL;
    REAL    *norms = NULL, *r = NULL, *w = NULL;
    REAL    *work = NULL, *x_best = NULL;
    REAL    **p = NULL, **hh = NULL;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling Safe GMRes solver (STR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    /* allocate memory and setup temp work space */
    work  = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
    
    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (restart > 5 ) ) {
        restart = restart - 5 ;
        work = (REAL *) fasp_mem_calloc((restart+4)*(restart+n)+1, sizeof(REAL));
        printf("### WARNING: GMRES restart number set to %d!\n", restart);
        restart1 = restart + 1;
    }
    
    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    
    p     = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    hh    = (REAL **)fasp_mem_calloc(restart1, sizeof(REAL *));
    norms = (REAL *) fasp_mem_calloc(MaxIt+1, sizeof(REAL));
    
    r = work; w = r + n; rs = w + n; c = rs + restart1;
    x_best = c + restart; s = x_best + n;
    
    for ( i = 0; i < restart1; i++ ) p[i] = s + restart + i*n;
    
    for ( i = 0; i < restart1; i++ ) hh[i] = p[restart] + n + i*restart;
    
    // r = b-A*x
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dstr_aAxpy(-1.0, A, x->val, p[0]);
    
    r_norm  = fasp_blas_darray_norm2(n,p[0]);
    
    // compute initial residuals
    switch (StopType) {
        case STOP_REL_RES:
            normr0  = MAX(SMALLREAL,r_norm);
            relres  = r_norm/normr0;
            break;
        case STOP_REL_PRECRES:
            if ( pc == NULL )
                fasp_darray_cp(n, p[0], r);
            else
                pc->fct(p[0], r, pc->data);
            r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
            normr0  = MAX(SMALLREAL,r_normb);
            relres  = r_normb/normr0;
            break;
        case STOP_MOD_REL_RES:
            normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
            normr0  = r_norm;
            relres  = normr0/normu;
            break;
        default:
            printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
            goto FINISHED;
    }
    
    // if initial residual is small, no need to iterate!
    if ( relres < tol ) goto FINISHED;
    
    // output iteration information if needed
    fasp_itinfo(PrtLvl,StopType,0,relres,normr0,0.0);
    
    // store initial residual
    norms[0] = relres;
    
    /* outer iteration cycle */
    while ( iter < MaxIt ) {
        
        rs[0] = r_norm;
        
        t = 1.0 / r_norm;
        
        fasp_blas_darray_ax(n, t, p[0]);
        
        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while ( i < restart && iter < MaxIt ) {
            
            i++; iter++;
            
            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], r);
            else
                pc->fct(p[i-1], r, pc->data);
            
            fasp_blas_dstr_mxv(A, r, p[i]);
            
            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if (t != 0.0) {
                t = 1.0/t;
                fasp_blas_darray_ax(n, t, p[i]);
            }
            
            for (j = 1; j < i; ++j) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t= hh[i][i-1]*hh[i][i-1];
            t+= hh[i-1][i-1]*hh[i-1][i-1];
            
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1]*rs[i-1];
            rs[i-1] =  c[i-1]*rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            
            absres = r_norm = fabs(rs[i]);
            
            relres = absres/normr0;
            
            norms[iter] = relres;
            
            // output iteration information if needed
            fasp_itinfo(PrtLvl, StopType, iter, relres, absres,
                        norms[iter]/norms[iter-1]);
            
            // should we exit restart cycle
            if ( relres <= tol && iter >= MIN_ITER ) break;
            
        } /* end of restart cycle */
        
        /* compute solution, first solve upper triangular system */
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];
            
            t += rs[k];
            rs[k] = t / hh[k][k];
        }
        
        fasp_darray_cp(n, p[i-1], w);
        
        fasp_blas_darray_ax(n, rs[i-1], w);
        
        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], w);
        
        /* apply preconditioner */
        if ( pc == NULL )
            fasp_darray_cp(n, w, r);
        else
            pc->fct(w, r, pc->data);
        
        fasp_blas_darray_axpy(n, 1.0, r, x->val);
        
        // safety net check: save the best-so-far solution
        if ( fasp_dvec_isnan(x) ) {
            // If the solution is NAN, restore the best solution
            absres = BIGREAL;
            goto RESTORE_BESTSOL;
        }
        
        if ( absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best   = iter;
            fasp_darray_cp(n,x->val,x_best);
        }
        
        // Check: prevent false convergence
        if ( relres <= tol && iter >= MIN_ITER ) {
            
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dstr_aAxpy(-1.0, A, x->val, r);
            
            r_norm = fasp_blas_darray_norm2(n, r);
            
            switch ( StopType ) {
                case STOP_REL_RES:
                    absres = r_norm;
                    relres = absres/normr0;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL )
                        fasp_darray_cp(n, r, w);
                    else
                        pc->fct(r, w, pc->data);
                    absres = sqrt(fasp_blas_darray_dotprod(n,w,r));
                    relres = absres/normr0;
                    break;
                case STOP_MOD_REL_RES:
                    absres = r_norm;
                    normu  = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres = absres/normu;
                    break;
            }
            
            norms[iter] = relres;
            
            if ( relres <= tol ) {
                break;
            }
            else {
                // Need to restart
                fasp_darray_cp(n, r, p[0]); i = 0;
            }
            
        } /* end of convergence check */
        
        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }
        
        if ( i ) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);
        
        for ( j = i-1 ; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);
        
        if ( i ) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }
        
    } /* end of main while loop */
    
RESTORE_BESTSOL: // restore the best-so-far solution if necessary
    if ( iter != iter_best ) {
        
        // compute best residual
        fasp_darray_cp(n,b->val,r);
        fasp_blas_dstr_aAxpy(-1.0,A,x_best,r);
        
        switch ( StopType ) {
            case STOP_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                if ( pc != NULL )
                    pc->fct(r,w,pc->data); /* Apply preconditioner */
                else
                    fasp_darray_cp(n,r,w); /* No preconditioner */
                absres_best = sqrt(ABS(fasp_blas_darray_dotprod(n,w,r)));
                break;
            case STOP_MOD_REL_RES:
                absres_best = fasp_blas_darray_norm2(n,r);
                break;
        }
        
        if ( absres > absres_best + maxdiff || isnan(absres) ) {
            if ( PrtLvl > PRINT_NONE ) ITS_RESTORE(iter_best);
            fasp_darray_cp(n,x_best,x->val);
            relres = absres_best / normr0;
        }
    }
    
FINISHED:
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,relres);
    
    /*-------------------------------------------
     * Clean up workspace
     *------------------------------------------*/
    fasp_mem_free(work);  work  = NULL;
    fasp_mem_free(p);     p     = NULL;
    fasp_mem_free(hh);    hh    = NULL;
    fasp_mem_free(norms); norms = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
