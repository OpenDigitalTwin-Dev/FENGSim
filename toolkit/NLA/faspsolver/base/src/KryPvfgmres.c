/*! \file  KryPvfgmres.c
 *
 *  \brief Krylov subspace methods -- Preconditioned variable-restarting FGMRes
 *
 *  \note  This file contains Level-3 (Kry) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, BlaArray.c, BlaSpmvBLC.c,
 *         BlaSpmvBSR.c, and BlaSpmvCSR.c
 *
 *  \note  This file is modifed from KryPvgmres.c
 *
 *  Reference:
 *         A.H. Baker, E.R. Jessup, and Tz.V. Kolev
 *         A Simple Strategy for Varying the Restart Parameter in GMRES(m)
 *         Journal of Computational and Applied Mathematics, 230 (2009)
 *         pp. 751-761. UCRL-JRNL-235266.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
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
 * \fn INT fasp_solver_dcsr_pvfgmres (dCSRmat *A, dvector *b, dvector *x, precond *pc,
 *                                    const REAL tol, const REAL abstol,
 *                                    const INT MaxIt, const SHORT restart,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Solve "Ax=b" using PFGMRES(right preconditioned) iterative method in which
 *        the restart parameter can be adaptively modified during iteration and
 *        flexible preconditioner can be used.
 *
 * \param A            Pointer to dCSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to precond: structure of precondition
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type -- DO not support this parameter
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/04/2012
 *
 * Modified by Chunsheng Feng on 07/22/2013: Add adaptive memory allocate
 * Modified by Chensong Zhang on 05/09/2015: Clean up for stopping types
 */
INT fasp_solver_dcsr_pvfgmres(dCSRmat* A, dvector* b, dvector* x, precond* pc,
                              const REAL tol, const REAL abstol, const INT MaxIt,
                              const SHORT restart, const SHORT StopType,
                              const SHORT PrtLvl)
{
    const INT n                 = b->row;
    const INT min_iter          = 0;

    //--------------------------------------------//
    //   Newly added parameters to monitor when   //
    //   to change the restart parameter          //
    //--------------------------------------------//
    const REAL cr_max           = 0.99;    // = cos(8^o)  (experimental)
    const REAL cr_min           = 0.174;   // = cos(80^o) (experimental)

    // local variables
    INT    iter                 = 0;
    int    i, j, k; // must be signed! -zcs

    REAL   epsmac               = SMALLREAL;
    REAL   r_norm, b_norm, den_norm;
    REAL   epsilon, gamma, t;
    REAL   relres, normu, r_normb;

    REAL   *c = NULL, *s = NULL, *rs = NULL, *norms = NULL, *r = NULL;
    REAL   **p = NULL, **hh = NULL, **z=NULL;

    REAL   cr          = 1.0;     // convergence rate
    REAL   r_norm_old  = 0.0;     // save residual norm of previous restart cycle
    INT    d           = 3;       // reduction for restart parameter
    INT    restart_max = restart; // upper bound for restart in each restart cycle
    INT    restart_min = 3;       // lower bound for restart in each restart cycle

    INT  Restart  = restart; // real restart in some fixed restarted cycle
    INT  Restart1 = Restart + 1;
    LONG worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;

    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling VFGMRes solver (CSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    /* allocate memory and setup temp work space */
    REAL *work  = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));

    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (Restart > 5) ) {
        Restart = Restart - 5;
        worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
        work = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
        Restart1 = Restart + 1;
    }

    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__ );
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    if ( PrtLvl > PRINT_MIN && Restart < restart ) {
        printf("### WARNING: vFGMRES restart number set to %d!\n", Restart);
    }

    p  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    hh = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    z  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    norms = (REAL *)fasp_mem_calloc(MaxIt+1, sizeof(REAL));

    r = work; rs = r + n; c = rs + Restart1; s = c + Restart;
    for ( i = 0; i < Restart1; i++ ) p[i] = s + Restart + i*n;
    for ( i = 0; i < Restart1; i++ ) hh[i] = p[Restart] + n + i*Restart;
    for ( i = 0; i < Restart1; i++ ) z[i] = hh[Restart] + Restart + i*n;

    /* initialization */
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dcsr_aAxpy(-1.0, A, x->val, p[0]);

    b_norm = fasp_blas_darray_norm2(n, b->val);
    r_norm = fasp_blas_darray_norm2(n, p[0]);
    norms[0] = r_norm;

    if ( PrtLvl >= PRINT_SOME) {
        ITS_PUTNORM("right-hand side", b_norm);
        ITS_PUTNORM("residual", r_norm);
    }

    if ( b_norm > 0.0 ) den_norm = b_norm;
    else                den_norm = r_norm;

    epsilon = tol*den_norm;

    // if initial residual is small, no need to iterate!
    if ( r_norm < epsilon || r_norm < abstol ) goto FINISHED;

    if ( b_norm > 0.0 ) {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm, norms[iter], 0);
    }
    else {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter], 0);
    }

    /* outer iteration cycle */
    while ( iter < MaxIt ) {

        rs[0] = r_norm;
        r_norm_old = r_norm;
        if ( r_norm == 0.0 ) {
            fasp_mem_free(work);  work  = NULL;
            fasp_mem_free(p);     p     = NULL;
            fasp_mem_free(hh);    hh    = NULL;
            fasp_mem_free(norms); norms = NULL;
            fasp_mem_free(z);     z     = NULL;
            return iter;
        }

        //-----------------------------------//
        //   adjust the restart parameter    //
        //-----------------------------------//

        if ( cr > cr_max || iter == 0 ) {
            Restart = restart_max;
        }
        else if ( cr < cr_min ) {
            // Restart = Restart;
        }
        else {
            if ( Restart - d > restart_min ) Restart -= d;
            else Restart = restart_max;
        }

        // Enter the cycle at the first iteration for at least one iteration
        t = 1.0 / r_norm;
        fasp_blas_darray_ax(n, t, p[0]);
        i = 0;

        // RESTART CYCLE (right-preconditioning)
        while ( i < Restart && iter < MaxIt ) {

            i ++;  iter ++;

            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], z[i-1]);
            else
                pc->fct(p[i-1], z[i-1], pc->data);

            fasp_blas_dcsr_mxv(A, z[i-1], p[i]);

            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if ( t != 0.0 ) {
                t = 1.0 / t;
                fasp_blas_darray_ax(n, t, p[i]);
            }

            for ( j = 1; j < i; ++j ) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] =  s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t  = hh[i][i-1]   * hh[i][i-1];
            t += hh[i-1][i-1] * hh[i-1][i-1];
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1] * rs[i-1];
            rs[i-1] =  c[i-1] * rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];

            r_norm = fabs(rs[i]);
            norms[iter] = r_norm;

            if ( b_norm > 0 ) {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm,
                            norms[iter], norms[iter]/norms[iter-1]);
            }
            else {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter],
                            norms[iter]/norms[iter-1]);
            }

            /* Check: Exit the restart cycle? */
            if (r_norm <= epsilon && iter >= min_iter) break;

        } /* end of restart cycle */

        /* now compute solution, first solve upper triangular system */

        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j ++) t -= hh[k][j]*rs[j];

            t += rs[k];
            rs[k] = t / hh[k][k];
        }

        fasp_darray_cp(n, z[i-1], r);
        fasp_blas_darray_ax(n, rs[i-1], r);

        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], z[j], r);

        fasp_blas_darray_axpy(n, 1.0, r, x->val);

        if ( r_norm <= epsilon && iter >= min_iter ) {
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dcsr_aAxpy(-1.0, A, x->val, r);
            r_norm = fasp_blas_darray_norm2(n, r);

            switch (StopType) {
                case STOP_REL_RES:
                    relres  = r_norm/den_norm;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL ) fasp_darray_cp(n, r, p[0]);
                    else pc->fct(r, p[0], pc->data);
                    r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
                    relres  = r_normb/den_norm;
                    break;
                case STOP_MOD_REL_RES:
                    normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres  = r_norm/normu;
                    break;
                default:
                    printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
                    goto FINISHED;
            }

            if ( relres <= tol ) {
                break;
            }
            else {
                if ( PrtLvl >= PRINT_SOME ) ITS_FACONV;
                fasp_darray_cp(n, r, p[0]); i = 0;
            }

        } /* end of convergence check */

        /* compute residual vector and continue loop */
        for ( j = i; j > 0; j-- ) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }

        if (i) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);

        for ( j = i-1; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);

        if (i) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }

        //-----------------------------------//
        //   compute the convergence rate    //
        //-----------------------------------//
        cr = r_norm / r_norm_old;

    } /* end of iteration while loop */

    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,r_norm/den_norm);

FINISHED:
    /*-------------------------------------------
     * Free some stuff
     *------------------------------------------*/
    fasp_mem_free(work);   work  = NULL;
    fasp_mem_free(p);      p     = NULL;
    fasp_mem_free(hh);     hh    = NULL;
    fasp_mem_free(norms);  norms = NULL;
    fasp_mem_free(z);      z     = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*!
 * \fn INT fasp_solver_dbsr_pvfgmres (dBSRmat *A, dvector *b, dvector *x, precond *pc,
 *                                    const REAL tol, const REAL abstol,
 *                                    const INT MaxIt, const SHORT restart,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Solve "Ax=b" using PFGMRES(right preconditioned) iterative method in which
 *        the restart parameter can be adaptively modified during iteration and
 *        flexible preconditioner can be used.
 *
 * \param A            Pointer to dCSRmat: coefficient matrix
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to precond: structure of precondition
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type -- DO not support this parameter
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   02/05/2012
 *
 * Modified by Chunsheng Feng on 07/22/2013: Add adaptive memory allocate
 * Modified by Chensong Zhang on 05/09/2015: Clean up for stopping types
 */
INT fasp_solver_dbsr_pvfgmres(dBSRmat* A, dvector* b, dvector* x, precond* pc,
                              const REAL tol, const REAL abstol, const INT MaxIt,
                              const SHORT restart, const SHORT StopType,
                              const SHORT PrtLvl)
{
    const INT n                 = b->row;
    const INT min_iter          = 0;

    //--------------------------------------------//
    //   Newly added parameters to monitor when   //
    //   to change the restart parameter          //
    //--------------------------------------------//
    const REAL cr_max           = 0.99;    // = cos(8^o)  (experimental)
    const REAL cr_min           = 0.174;   // = cos(80^o) (experimental)

    // local variables
    INT    iter                 = 0;
    int    i, j, k; // must be signed! -zcs

    REAL   epsmac               = SMALLREAL;
    REAL   r_norm, b_norm, den_norm;
    REAL   epsilon, gamma, t;
    REAL   relres, normu, r_normb;

    REAL   *c = NULL, *s = NULL, *rs = NULL, *norms = NULL, *r = NULL;
    REAL   **p = NULL, **hh = NULL, **z=NULL;

    REAL   cr          = 1.0;     // convergence rate
    REAL   r_norm_old  = 0.0;     // save residual norm of previous restart cycle
    INT    d           = 3;       // reduction for restart parameter
    INT    restart_max = restart; // upper bound for restart in each restart cycle
    INT    restart_min = 3;       // lower bound for restart in each restart cycle

    INT  Restart  = restart; // real restart in some fixed restarted cycle
    INT  Restart1 = Restart + 1;
    LONG worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;

    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling VFGMRes solver (BSR) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif

    /* allocate memory and setup temp work space */
    REAL *work  = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));

    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (Restart > 5) ) {
        Restart = Restart - 5;
        worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
        work = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
        Restart1 = Restart + 1;
    }

    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__ );
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    if ( PrtLvl > PRINT_MIN && Restart < restart ) {
        printf("### WARNING: vFGMRES restart number set to %d!\n", Restart);
    }

    p  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    hh = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    z  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    norms = (REAL *)fasp_mem_calloc(MaxIt+1, sizeof(REAL));

    r = work; rs = r + n; c = rs + Restart1; s = c + Restart;
    for ( i = 0; i < Restart1; i++ ) p[i] = s + Restart + i*n;
    for ( i = 0; i < Restart1; i++ ) hh[i] = p[Restart] + n + i*Restart;
    for ( i = 0; i < Restart1; i++ ) z[i] = hh[Restart] + Restart + i*n;

    /* initialization */
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dbsr_aAxpy(-1.0, A, x->val, p[0]);

    b_norm = fasp_blas_darray_norm2(n, b->val);
    r_norm = fasp_blas_darray_norm2(n, p[0]);
    norms[0] = r_norm;

    if ( PrtLvl >= PRINT_SOME) {
        ITS_PUTNORM("right-hand side", b_norm);
        ITS_PUTNORM("residual", r_norm);
    }

    if ( b_norm > 0.0 ) den_norm = b_norm;
    else                den_norm = r_norm;

    epsilon = tol*den_norm;

    // if initial residual is small, no need to iterate!
    if (r_norm < epsilon || r_norm < abstol) goto FINISHED;

    if ( b_norm > 0.0 ) {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm, norms[iter], 0);
    }
    else {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter], 0);
    }

    /* outer iteration cycle */
    while ( iter < MaxIt ) {

        rs[0] = r_norm;
        r_norm_old = r_norm;
        if ( r_norm == 0.0 ) {
            fasp_mem_free(work);  work  = NULL;
            fasp_mem_free(p);     p     = NULL;
            fasp_mem_free(hh);    hh    = NULL;
            fasp_mem_free(norms); norms = NULL;
            fasp_mem_free(z);     z     = NULL;
            return iter;
        }

        //-----------------------------------//
        //   adjust the restart parameter    //
        //-----------------------------------//

        if ( cr > cr_max || iter == 0 ) {
            Restart = restart_max;
        }
        else if ( cr < cr_min ) {
            // Restart = Restart;
        }
        else {
            if ( Restart - d > restart_min ) Restart -= d;
            else Restart = restart_max;
        }

        // Enter the cycle at the first iteration for at least one iteration
        t = 1.0 / r_norm;
        fasp_blas_darray_ax(n, t, p[0]);
        i = 0;

        // RESTART CYCLE (right-preconditioning)
        while ( i < Restart && iter < MaxIt ) {

            i ++;  iter ++;

            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], z[i-1]);
            else
                pc->fct(p[i-1], z[i-1], pc->data);

            fasp_blas_dbsr_mxv(A, z[i-1], p[i]);

            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if ( t != 0.0 ) {
                t = 1.0 / t;
                fasp_blas_darray_ax(n, t, p[i]);
            }

            for ( j = 1; j < i; ++j ) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] =  s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t  = hh[i][i-1]   * hh[i][i-1];
            t += hh[i-1][i-1] * hh[i-1][i-1];
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1] * rs[i-1];
            rs[i-1] =  c[i-1] * rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];

            r_norm = fabs(rs[i]);
            norms[iter] = r_norm;

            if ( b_norm > 0 ) {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm,
                            norms[iter], norms[iter]/norms[iter-1]);
            }
            else {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter],
                            norms[iter]/norms[iter-1]);
            }

            /* Check: Exit the restart cycle? */
            if (r_norm <= epsilon && iter >= min_iter) break;

        } /* end of restart cycle */

        /* now compute solution, first solve upper triangular system */

        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j ++) t -= hh[k][j]*rs[j];

            t += rs[k];
            rs[k] = t / hh[k][k];
        }

        fasp_darray_cp(n, z[i-1], r);
        fasp_blas_darray_ax(n, rs[i-1], r);

        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], z[j], r);

        fasp_blas_darray_axpy(n, 1.0, r, x->val);

        if ( r_norm <= epsilon && iter >= min_iter ) {
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dbsr_aAxpy(-1.0, A, x->val, r);
            r_norm = fasp_blas_darray_norm2(n, r);

            switch (StopType) {
                case STOP_REL_RES:
                    relres  = r_norm/den_norm;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL ) fasp_darray_cp(n, r, p[0]);
                    else pc->fct(r, p[0], pc->data);
                    r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
                    relres  = r_normb/den_norm;
                    break;
                case STOP_MOD_REL_RES:
                    normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres  = r_norm/normu;
                    break;
                default:
                    printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
                    goto FINISHED;
            }

            if ( relres <= tol ) {
                break;
            }
            else {
                if ( PrtLvl >= PRINT_SOME ) ITS_FACONV;
                fasp_darray_cp(n, r, p[0]); i = 0;
            }

        } /* end of convergence check */

        /* compute residual vector and continue loop */
        for ( j = i; j > 0; j-- ) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }

        if (i) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);

        for ( j = i-1; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);

        if (i) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }

        //-----------------------------------//
        //   compute the convergence rate    //
        //-----------------------------------//
        cr = r_norm / r_norm_old;

    } /* end of iteration while loop */

    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,r_norm/den_norm);

FINISHED:
    /*-------------------------------------------
     * Free some stuff
     *------------------------------------------*/
    fasp_mem_free(work);   work  = NULL;
    fasp_mem_free(p);      p     = NULL;
    fasp_mem_free(hh);     hh    = NULL;
    fasp_mem_free(norms);  norms = NULL;
    fasp_mem_free(z);      z     = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*!
 * \fn INT fasp_solver_dblc_pvfgmres (dBLCmat *A, dvector *b, dvector *x, precond *pc,
 *                                    const REAL tol, const REAL abstol,
 *                                    const INT MaxIt, const SHORT restart,
 *                                    const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Solve "Ax=b" using PFGMRES (right preconditioned) iterative method in which
 *        the restart parameter can be adaptively modified during iteration and
 *        flexible preconditioner can be used.
 *
 * \param A            Pointer to coefficient matrix
 * \param b            Pointer to right hand side vector
 * \param x            Pointer to solution vector
 * \param MaxIt        Maximal iteration number allowed
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param pc           Pointer to preconditioner data
 * \param PrtLvl       How much information to print out
 * \param StopType     Stopping criterion, i.e.||r_k||/||r_0||<tol
 * \param restart      Number of restart for GMRES
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/04/2012
 *
 * \note   Based on Zhiyang Zhou's pvgmres.c
 *
 * Modified by Chunsheng Feng on 07/22/2013: Add adaptive memory allocate
 * Modified by Chensong Zhang on 05/09/2015: Clean up for stopping types
 */
INT fasp_solver_dblc_pvfgmres(dBLCmat* A, dvector* b, dvector* x, precond* pc,
                              const REAL tol, const REAL abstol, const INT MaxIt,
                              const SHORT restart, const SHORT StopType,
                              const SHORT PrtLvl)
{
    const INT n                 = b->row;
    const INT min_iter          = 0;
    
    //--------------------------------------------//
    //   Newly added parameters to monitor when   //
    //   to change the restart parameter          //
    //--------------------------------------------//
    const REAL cr_max           = 0.99;    // = cos(8^o)  (experimental)
    const REAL cr_min           = 0.174;   // = cos(80^o) (experimental)
    
    // local variables
    INT    iter                 = 0;
    int    i, j, k; // must be signed! -zcs
    
    REAL   epsmac               = SMALLREAL;
    REAL   r_norm, b_norm, den_norm;
    REAL   epsilon, gamma, t;
    REAL   relres, normu, r_normb;
    
    REAL   *c = NULL, *s = NULL, *rs = NULL, *norms = NULL, *r = NULL;
    REAL   **p = NULL, **hh = NULL, **z=NULL;
    
    REAL   cr          = 1.0;     // convergence rate
    REAL   r_norm_old  = 0.0;     // save residual norm of previous restart cycle
    INT    d           = 3;       // reduction for restart parameter
    INT    restart_max = restart; // upper bound for restart in each restart cycle
    INT    restart_min = 3;       // lower bound for restart in each restart cycle
    
    INT  Restart  = restart; // real restart in some fixed restarted cycle
    INT  Restart1 = Restart + 1;
    LONG worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling VFGMRes solver (BLC) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    /* allocate memory and setup temp work space */
    REAL *work  = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
    
    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (Restart > 5) ) {
        Restart = Restart - 5;
        worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
        work = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
        Restart1 = Restart + 1;
    }
    
    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__ );
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    
    if ( PrtLvl > PRINT_MIN && Restart < restart ) {
        printf("### WARNING: vFGMRES restart number set to %d!\n", Restart);
    }
    
    p  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    hh = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    z  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    norms = (REAL *)fasp_mem_calloc(MaxIt+1, sizeof(REAL));
    
    r = work; rs = r + n; c = rs + Restart1; s = c + Restart;
    for ( i = 0; i < Restart1; i++ ) p[i] = s + Restart + i*n;
    for ( i = 0; i < Restart1; i++ ) hh[i] = p[Restart] + n + i*Restart;
    for ( i = 0; i < Restart1; i++ ) z[i] = hh[Restart] + Restart + i*n;
    
    /* initialization */
    fasp_darray_cp(n, b->val, p[0]);
    fasp_blas_dblc_aAxpy(-1.0, A, x->val, p[0]);
    
    b_norm = fasp_blas_darray_norm2(n, b->val);
    r_norm = fasp_blas_darray_norm2(n, p[0]);
    norms[0] = r_norm;
    
    if ( PrtLvl >= PRINT_SOME) {
        ITS_PUTNORM("right-hand side", b_norm);
        ITS_PUTNORM("residual", r_norm);
    }
    
    if ( b_norm > 0.0 ) den_norm = b_norm;
    else                den_norm = r_norm;
    
    epsilon = tol*den_norm;

    // if initial residual is small, no need to iterate!
    if (r_norm < epsilon || r_norm < abstol) goto FINISHED;

    if ( b_norm > 0.0 ) {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm, norms[iter], 0);
    }
    else {
        fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter], 0);
    }
    
    /* outer iteration cycle */
    while ( iter < MaxIt ) {
        
        rs[0] = r_norm;
        r_norm_old = r_norm;
        if ( r_norm == 0.0 ) {
            fasp_mem_free(work);  work  = NULL;
            fasp_mem_free(p);     p     = NULL;
            fasp_mem_free(hh);    hh    = NULL;
            fasp_mem_free(norms); norms = NULL;
            fasp_mem_free(z);     z     = NULL;
            return iter;
        }
        
        //-----------------------------------//
        //   adjust the restart parameter    //
        //-----------------------------------//
        
        if ( cr > cr_max || iter == 0 ) {
            Restart = restart_max;
        }
        else if ( cr < cr_min ) {
            // Restart = Restart;
        }
        else {
            if ( Restart - d > restart_min ) Restart -= d;
            else Restart = restart_max;
        }
        
        // Enter the cycle at the first iteration for at least one iteration
        t = 1.0 / r_norm;
        fasp_blas_darray_ax(n, t, p[0]);
        i = 0;
        
        // RESTART CYCLE (right-preconditioning)
        while ( i < Restart && iter < MaxIt ) {
            
            i ++;  iter ++;
            
            /* apply preconditioner */
            if ( pc == NULL )
                fasp_darray_cp(n, p[i-1], z[i-1]);
            else
                pc->fct(p[i-1], z[i-1], pc->data);
            
            fasp_blas_dblc_mxv(A, z[i-1], p[i]);
            
            /* modified Gram_Schmidt */
            for ( j = 0; j < i; j++ ) {
                hh[j][i-1] = fasp_blas_darray_dotprod(n, p[j], p[i]);
                fasp_blas_darray_axpy(n, -hh[j][i-1], p[j], p[i]);
            }
            t = fasp_blas_darray_norm2(n, p[i]);
            hh[i][i-1] = t;
            if ( t != 0.0 ) {
                t = 1.0 / t;
                fasp_blas_darray_ax(n, t, p[i]);
            }
            
            for ( j = 1; j < i; ++j ) {
                t = hh[j-1][i-1];
                hh[j-1][i-1] =  s[j-1]*hh[j][i-1] + c[j-1]*t;
                hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t  = hh[i][i-1]   * hh[i][i-1];
            t += hh[i-1][i-1] * hh[i-1][i-1];
            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;
            c[i-1]  = hh[i-1][i-1] / gamma;
            s[i-1]  = hh[i][i-1] / gamma;
            rs[i]   = -s[i-1] * rs[i-1];
            rs[i-1] =  c[i-1] * rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            
            r_norm = fabs(rs[i]);
            norms[iter] = r_norm;
            
            if ( b_norm > 0 ) {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm,
                            norms[iter], norms[iter]/norms[iter-1]);
            }
            else {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter],
                            norms[iter]/norms[iter-1]);
            }
            
            /* Check: Exit the restart cycle? */
            if (r_norm <= epsilon && iter >= min_iter) break;
            
        } /* end of restart cycle */
        
        /* now compute solution, first solve upper triangular system */
        
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for ( k = i-2; k >= 0; k-- ) {
            t = 0.0;
            for (j = k+1; j < i; j ++) t -= hh[k][j]*rs[j];
            
            t += rs[k];
            rs[k] = t / hh[k][k];
        }
        
        fasp_darray_cp(n, z[i-1], r);
        fasp_blas_darray_ax(n, rs[i-1], r);
        
        for ( j = i-2; j >= 0; j-- ) fasp_blas_darray_axpy(n, rs[j], z[j], r);
        
        fasp_blas_darray_axpy(n, 1.0, r, x->val);
        
        if ( r_norm <= epsilon && iter >= min_iter ) {
            fasp_darray_cp(n, b->val, r);
            fasp_blas_dblc_aAxpy(-1.0, A, x->val, r);
            r_norm = fasp_blas_darray_norm2(n, r);
            
            switch (StopType) {
                case STOP_REL_RES:
                    relres  = r_norm/den_norm;
                    break;
                case STOP_REL_PRECRES:
                    if ( pc == NULL ) fasp_darray_cp(n, r, p[0]);
                    else pc->fct(r, p[0], pc->data);
                    r_normb = sqrt(fasp_blas_darray_dotprod(n,p[0],r));
                    relres  = r_normb/den_norm;
                    break;
                case STOP_MOD_REL_RES:
                    normu   = MAX(SMALLREAL,fasp_blas_darray_norm2(n,x->val));
                    relres  = r_norm/normu;
                    break;
                default:
                    printf("### ERROR: Unknown stopping type! [%s]\n", __FUNCTION__);
                    goto FINISHED;
            }
            
            if ( relres <= tol ) {
                break;
            }
            else {
                if ( PrtLvl >= PRINT_SOME ) ITS_FACONV;
                fasp_darray_cp(n, r, p[0]); i = 0;
            }
            
        } /* end of convergence check */
        
        /* compute residual vector and continue loop */
        for ( j = i; j > 0; j-- ) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }
        
        if (i) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);
        
        for ( j = i-1; j > 0; j-- ) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);
        
        if (i) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }
        
        //-----------------------------------//
        //   compute the convergence rate    //
        //-----------------------------------//
        cr = r_norm / r_norm_old;
        
    } /* end of iteration while loop */
    
    if ( PrtLvl > PRINT_NONE ) ITS_FINAL(iter,MaxIt,r_norm/den_norm);
    
FINISHED:
    /*-------------------------------------------
     * Free some stuff
     *------------------------------------------*/
    fasp_mem_free(work);   work  = NULL;
    fasp_mem_free(p);      p     = NULL;
    fasp_mem_free(hh);     hh    = NULL;
    fasp_mem_free(norms);  norms = NULL;
    fasp_mem_free(z);      z     = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if ( iter >= MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*!
 * \fn INT fasp_solver_pvfgmres (mxv_matfree *mf, dvector *b, dvector *x, precond *pc,
 *                               const REAL tol, const REAL abstol,
 *                               const INT MaxIt, const SHORT restart,
 *                               const SHORT StopType, const SHORT PrtLvl)
 *
 * \brief Solve "Ax=b" using PFGMRES(right preconditioned) iterative method in which
 *        the restart parameter can be adaptively modified during iteration and
 *        flexible preconditioner can be used.
 *
 * \param mf           Pointer to mxv_matfree: spmv operation
 * \param b            Pointer to dvector: right hand side
 * \param x            Pointer to dvector: unknowns
 * \param pc           Pointer to precond: structure of precondition
 * \param tol          Tolerance for relative residual
 * \param abstol       Tolerance for absolute residual
 * \param MaxIt        Maximal number of iterations
 * \param restart      Restarting steps
 * \param StopType     Stopping criteria type -- DO not support this parameter
 * \param PrtLvl       How much information to print out
 *
 * \return             Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/04/2012
 *
 * Modified by Feiteng Huang on 09/26/2012: matrix free
 * Modified by Chunsheng Feng on 07/22/2013: Add adapt memory allocate
 */
INT fasp_solver_pvfgmres(mxv_matfree* mf, dvector* b, dvector* x, precond* pc,
                         const REAL tol, const REAL abstol, const INT MaxIt,
                         const SHORT restart, const SHORT StopType, const SHORT PrtLvl)
{
    const INT n                 = b->row;
    const INT min_iter          = 0;
    
    //--------------------------------------------//
    //   Newly added parameters to monitor when   //
    //   to change the restart parameter          //
    //--------------------------------------------//
    const REAL cr_max           = 0.99;    // = cos(8^o)  (experimental)
    const REAL cr_min           = 0.174;   // = cos(80^o) (experimental)
    
    // local variables
    INT    iter                 = 0;
    int    i, j, k; // must be signed! -zcs
    
    REAL   epsmac               = SMALLREAL;
    REAL   r_norm, b_norm, den_norm;
    REAL   epsilon, gamma, t;
    
    REAL  *c = NULL, *s = NULL, *rs = NULL;
    REAL  *norms = NULL, *r = NULL;
    REAL  **p = NULL, **hh = NULL, **z=NULL;
    REAL  *work = NULL;
    
    REAL   cr          = 1.0;     // convergence rate
    REAL   r_norm_old  = 0.0;     // save residual norm of previous restart cycle
    INT    d           = 3;       // reduction for restart parameter
    INT    restart_max = restart; // upper bound for restart in each restart cycle
    INT    restart_min = 3;       // lower bound for restart in each restart cycle
    
    INT  Restart  = restart; // real restart in some fixed restarted cycle
    INT  Restart1 = Restart + 1;
    LONG worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
    
    // Output some info for debuging
    if ( PrtLvl > PRINT_NONE ) printf("\nCalling VFGMRes solver (MatFree) ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: maxit = %d, tol = %.4le\n", MaxIt, tol);
#endif
    
    /* allocate memory and setup temp work space */
    work  = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
    
    /* check whether memory is enough for GMRES */
    while ( (work == NULL) && (Restart > 5) ) {
        Restart = Restart - 5;
        worksize = (Restart+4)*(Restart+n)+1-n+Restart*n;
        work = (REAL *) fasp_mem_calloc(worksize, sizeof(REAL));
        Restart1 = Restart + 1;
    }
    
    if ( work == NULL ) {
        printf("### ERROR: No enough memory! [%s:%d]\n", __FILE__, __LINE__ );
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    
    if ( PrtLvl > PRINT_MIN && Restart < restart ) {
        printf("### WARNING: vFGMRES restart number set to %d!\n", Restart);
    }
    
    p  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    hh = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    z  = (REAL **)fasp_mem_calloc(Restart1, sizeof(REAL *));
    norms = (REAL *)fasp_mem_calloc(MaxIt+1, sizeof(REAL));
    
    r = work; rs = r + n; c = rs + Restart1; s = c + Restart;
    for (i = 0; i < Restart1; i ++) p[i] = s + Restart + i*n;
    for (i = 0; i < Restart1; i ++) hh[i] = p[Restart] + n + i*Restart;
    for (i = 0; i < Restart1; i ++) z[i] = hh[Restart] + Restart + i*n;
    
    /* initialization */
    mf->fct(mf->data, x->val, p[0]);
    fasp_blas_darray_axpby(n, 1.0, b->val, -1.0, p[0]);
    
    b_norm = fasp_blas_darray_norm2(n, b->val);
    r_norm = fasp_blas_darray_norm2(n, p[0]);
    norms[0] = r_norm;
    
    if ( PrtLvl >= PRINT_SOME) {
        ITS_PUTNORM("right-hand side", b_norm);
        ITS_PUTNORM("residual", r_norm);
    }
    
    if (b_norm > 0.0)  den_norm = b_norm;
    else               den_norm = r_norm;
    
    epsilon = tol*den_norm;
    
    /* outer iteration cycle */
    while (iter < MaxIt) {
        rs[0] = r_norm;
        r_norm_old = r_norm;
        if (r_norm == 0.0) {
            fasp_mem_free(work);  work  = NULL;
            fasp_mem_free(p);     p     = NULL;
            fasp_mem_free(hh);    hh    = NULL;
            fasp_mem_free(norms); norms = NULL;
            fasp_mem_free(z);     z     = NULL;
            return iter;
        }
        
        //-----------------------------------//
        //   adjust the restart parameter    //
        //-----------------------------------//
        
        if (cr > cr_max || iter == 0) {
            Restart = restart_max;
        }
        else if (cr < cr_min) {
            // Restart = Restart;
        }
        else {
            if ( Restart - d > restart_min ) Restart -= d;
            else Restart = restart_max;
        }
        
        if (r_norm <= epsilon && iter >= min_iter) {
            mf->fct(mf->data, x->val, r);
            fasp_blas_darray_axpby(n, 1.0, b->val, -1.0, r);
            r_norm = fasp_blas_darray_norm2(n, r);
            
            if (r_norm <= epsilon) {
                break;
            }
            else {
                if (PrtLvl >= PRINT_SOME) ITS_FACONV;
            }
        }
        
        t = 1.0 / r_norm;
        fasp_blas_darray_ax(n, t, p[0]);
        
        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while (i < Restart && iter < MaxIt) {
            
            i ++;  iter ++;
            
            /* apply preconditioner */
            if (pc == NULL) fasp_darray_cp(n, p[i-1], z[i-1]);
            else pc->fct(p[i-1], z[i-1], pc->data);
            
            mf->fct(mf->data, z[i-1], p[i]);
            
            /* modified Gram_Schmidt */
            for (j = 0; j < i; j ++) {
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
            rs[i-1] = c[i-1]*rs[i-1];
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            
            r_norm = fabs(rs[i]);
            norms[iter] = r_norm;
            
            if (b_norm > 0 ) {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter]/b_norm,
                            norms[iter], norms[iter]/norms[iter-1]);
            }
            else {
                fasp_itinfo(PrtLvl, StopType, iter, norms[iter], norms[iter],
                            norms[iter]/norms[iter-1]);
            }
            
            /* Check: Exit restart cycle? */
            if (r_norm <= epsilon && iter >= min_iter) break;
            
        } /* end of restart cycle */
        
        /* now compute solution, first solve upper triangular system */
        
        rs[i-1] = rs[i-1] / hh[i-1][i-1];
        for (k = i-2; k >= 0; k --) {
            t = 0.0;
            for (j = k+1; j < i; j ++)  t -= hh[k][j]*rs[j];
            
            t += rs[k];
            rs[k] = t / hh[k][k];
        }
        
        fasp_darray_cp(n, z[i-1], r);
        fasp_blas_darray_ax(n, rs[i-1], r);
        for (j = i-2; j >= 0; j --)  fasp_blas_darray_axpy(n, rs[j], z[j], r);
        
        fasp_blas_darray_axpy(n, 1.0, r, x->val);
        
        if (r_norm  <= epsilon && iter >= min_iter) {
            mf->fct(mf->data, x->val, r);
            fasp_blas_darray_axpby(n, 1.0, b->val, -1.0, r);
            r_norm = fasp_blas_darray_norm2(n, r);
            
            if (r_norm  <= epsilon) {
                break;
            }
            else {
                if (PrtLvl >= PRINT_SOME) ITS_FACONV;
                fasp_darray_cp(n, r, p[0]); i = 0;
            }
        } /* end of convergence check */
        
        
        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
        }
        
        if (i) fasp_blas_darray_axpy(n, rs[i]-1.0, p[i], p[i]);
        
        for (j = i-1 ; j > 0; j --) fasp_blas_darray_axpy(n, rs[j], p[j], p[i]);
        
        if (i) {
            fasp_blas_darray_axpy(n, rs[0]-1.0, p[0], p[0]);
            fasp_blas_darray_axpy(n, 1.0, p[i], p[0]);
        }
        
        //-----------------------------------//
        //   compute the convergence rate    //
        //-----------------------------------//
        cr = r_norm / r_norm_old;
        
    } /* end of iteration while loop */
    
    if (PrtLvl > PRINT_NONE) ITS_FINAL(iter,MaxIt,r_norm);
    
    /*-------------------------------------------
     * Free some stuff
     *------------------------------------------*/
    fasp_mem_free(work);  work  = NULL;
    fasp_mem_free(p);     p     = NULL;
    fasp_mem_free(hh);    hh    = NULL;
    fasp_mem_free(norms); norms = NULL;
    fasp_mem_free(z);     z     = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    if (iter>=MaxIt)
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
