/*! \file  PreMGSolve.c
 *
 *  \brief Algebraic multigrid iterations: SOLVE phase.
 *
 *  \note  Solve Ax=b using multigrid method. This is SOLVE phase only and is
 *         independent of SETUP method used! Should be called after multigrid
 *         hierarchy has been generated!
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxMessage.c, AuxTiming.c, AuxVector.c, BlaSpmvCSR.c, BlaVector.c,
 *         PreMGCycle.c, PreMGCycleFull.c, and PreMGRecurAMLI.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

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
 * \fn INT fasp_amg_solve (AMG_data *mgl, AMG_param *param)
 *
 * \brief AMG -- SOLVE phase
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       Iteration number if converges; ERROR otherwise.
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   04/02/2010
 *
 * Modified by Chensong 04/21/2013: Fix an output typo
 */
INT fasp_amg_solve (AMG_data   *mgl,
                    AMG_param  *param)
{
    dCSRmat      *ptrA = &mgl[0].A;
    dvector      *b = &mgl[0].b, *x = &mgl[0].x, *r = &mgl[0].w;
    
    const SHORT   prtlvl = param->print_level;
    const INT     MaxIt  = param->maxit;
    const REAL    tol    = param->tol;
    const REAL    sumb   = fasp_blas_dvec_norm2(b); // L2norm(b)
    
    // local variables
    REAL  solve_start, solve_end;
    REAL  relres1 = 1.0, absres0 = sumb, absres, factor;
    INT   iter = 0;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nrow = %d, ncol = %d, nnz = %d\n",
           mgl[0].A.row, mgl[0].A.col, mgl[0].A.nnz);
#endif

    fasp_gettime(&solve_start);
    
    // Print iteration information if needed
    fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, sumb, 0.0);

    // If b = 0, set x = 0 to be a trivial solution
    if ( sumb <= SMALLREAL ) fasp_dvec_set(x->row, x, 0.0);

    // MG solver here
    while ( (iter++ < MaxIt) & (sumb > SMALLREAL) ) {
        
#if TRUE
        // Call one multigrid cycle -- non recursive version
        fasp_solver_mgcycle(mgl, param);
#else
        // Call one multigrid cycle -- recursive version
        fasp_solver_mgrecur(mgl, param, 0);
#endif
        
        // Form residual r = b - A*x
        fasp_dvec_cp(b, r);
        fasp_blas_dcsr_aAxpy(-1.0, ptrA, x->val, r->val);
        
        // Compute norms of r and convergence factor
        absres  = fasp_blas_dvec_norm2(r);     // residual ||r||
        relres1 = absres/MAX(SMALLREAL,sumb);  // relative residual ||r||/||b||
        factor  = absres/absres0;              // contraction factor
        absres0 = absres;                      // prepare for next iteration
        
        // Print iteration information if needed
        fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, absres, factor);
        
        // Check convergence
        if ( relres1 < tol ) break;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        ITS_FINAL(iter, MaxIt, relres1);
        fasp_gettime(&solve_end);
        fasp_cputime("AMG solve", solve_end - solve_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;}

/**
 * \fn INT fasp_amg_solve_amli (AMG_data *mgl, AMG_param *param)
 *
 * \brief AMLI -- SOLVE phase
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   01/23/2011
 *
 * Modified by Chensong 04/21/2013: Fix an output typo
 *
 * \note AMLI polynomial computed by the best approximation of 1/x. 
 *       Refer to Johannes K. Kraus, Panayot S. Vassilevski,
 *       Ludmil T. Zikatanov, "Polynomial of best uniform approximation to $x^{-1}$ 
 *       and smoothing in two-level methods", 2013.
 */
INT fasp_amg_solve_amli (AMG_data   *mgl,
                         AMG_param  *param)
{
    dCSRmat     *ptrA = &mgl[0].A;
    dvector     *b = &mgl[0].b, *x = &mgl[0].x, *r = &mgl[0].w;
    
    const INT    MaxIt  = param->maxit;
    const SHORT  prtlvl = param->print_level;
    const REAL   tol    = param->tol;
    const REAL   sumb   = fasp_blas_dvec_norm2(b); // L2norm(b)
    
    // local variables
    REAL         solve_start, solve_end;
    REAL         relres1 = 1.0, absres0 = sumb, absres, factor;
    INT          iter = 0;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nrow = %d, ncol = %d, nnz = %d\n",
           mgl[0].A.row, mgl[0].A.col, mgl[0].A.nnz);
#endif
    
    fasp_gettime(&solve_start);

    // Print iteration information if needed
    fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, sumb, 0.0);

    // If b = 0, set x = 0 to be a trivial solution
    if ( sumb <= SMALLREAL ) fasp_dvec_set(x->row, x, 0.0);

    // MG solver here
    while ( (iter++ < MaxIt) & (sumb > SMALLREAL) ) {
        
        // Call one AMLI cycle
        fasp_solver_amli(mgl, param, 0);
        
        // Form residual r = b-A*x
        fasp_dvec_cp(b, r);
        fasp_blas_dcsr_aAxpy(-1.0, ptrA, x->val, r->val);
        
        // Compute norms of r and convergence factor
        absres  = fasp_blas_dvec_norm2(r);     // residual ||r||
        relres1 = absres/MAX(SMALLREAL,sumb);  // relative residual ||r||/||b||
        factor  = absres/absres0;              // contraction factor
        absres0 = absres;                      // prepare for next iteration
        
        // Print iteration information if needed
        fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, absres, factor);
        
        // Check convergence
        if ( relres1 < tol ) break;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        ITS_FINAL(iter, MaxIt, relres1);
        fasp_gettime(&solve_end);
        fasp_cputime("AMLI solve", solve_end - solve_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn INT fasp_amg_solve_namli (AMG_data *mgl, AMG_param *param)
 *
 * \brief Nonlinear AMLI -- SOLVE phase
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   04/30/2011
 *
 * Modified by Chensong 04/21/2013: Fix an output typo
 *
 * \note Nonlinear AMLI-cycle.  
 *       Refer to Xiazhe Hu, Panayot S. Vassilevski, Jinchao Xu
 *       "Comparative Convergence Analysis of Nonlinear AMLI-cycle Multigrid", 2013.
 */
INT fasp_amg_solve_namli (AMG_data   *mgl,
                          AMG_param  *param)
{
    dCSRmat      *ptrA = &mgl[0].A;
    dvector      *b = &mgl[0].b, *x = &mgl[0].x, *r = &mgl[0].w;
    
    const INT     MaxIt  = param->maxit;
    const SHORT   prtlvl = param->print_level;
    const REAL    tol    = param->tol;
    const REAL    sumb   = fasp_blas_dvec_norm2(b); // L2norm(b)
    
    // local variables
    REAL          solve_start, solve_end;
    REAL          relres1 = 1.0, absres0 = sumb, absres, factor;
    INT           iter = 0;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nrow = %d, ncol = %d, nnz = %d\n",
           mgl[0].A.row, mgl[0].A.col, mgl[0].A.nnz);
#endif
    
    fasp_gettime(&solve_start);
    
    // Print iteration information if needed
    fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, sumb, 0.0);

    // If b = 0, set x = 0 to be a trivial solution
    if ( sumb <= SMALLREAL ) fasp_dvec_set(x->row, x, 0.0);

    while ( (iter++ < MaxIt) & (sumb > SMALLREAL) ) // MG solver here
    {
        // one multigrid cycle
        fasp_solver_namli(mgl, param, 0, mgl[0].num_levels);
        
        // r = b-A*x
        fasp_dvec_cp(b, r);
        fasp_blas_dcsr_aAxpy(-1.0, ptrA, x->val, r->val);
        
        absres  = fasp_blas_dvec_norm2(r);     // residual ||r||
        relres1 = absres/MAX(SMALLREAL,sumb);  // relative residual ||r||/||b||
        factor  = absres/absres0;              // contraction factor
        
        // output iteration information if needed
        fasp_itinfo(prtlvl, STOP_REL_RES, iter, relres1, absres, factor);
        
        if ( relres1 < tol ) break; // early exit condition
        
        absres0 = absres;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        ITS_FINAL(iter, MaxIt, relres1);
        fasp_gettime(&solve_end);
        fasp_cputime("Nonlinear AMLI solve", solve_end - solve_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if ( iter > MaxIt )
        return ERROR_SOLVER_MAXIT;
    else
        return iter;
}

/**
 * \fn void fasp_famg_solve (AMG_data *mgl, AMG_param *param)
 *
 * \brief FMG -- SOLVE phase
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \author Chensong Zhang
 * \date   01/10/2012
 */
void fasp_famg_solve (AMG_data   *mgl,
                      AMG_param  *param)
{
    dCSRmat     *ptrA = &mgl[0].A;
    dvector     *b = &mgl[0].b, *x = &mgl[0].x, *r = &mgl[0].w;
    
    const SHORT  prtlvl = param->print_level;
    const REAL   sumb   = fasp_blas_dvec_norm2(b); // L2norm(b)
    
    // local variables
    REAL         solve_start, solve_end;
    REAL         relres1 = 1.0, absres;
        
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nrow = %d, ncol = %d, nnz = %d\n",
           mgl[0].A.row, mgl[0].A.col, mgl[0].A.nnz);
#endif
    
    fasp_gettime(&solve_start);

    // If b = 0, set x = 0 to be a trivial solution
    if ( sumb <= SMALLREAL ) fasp_dvec_set(x->row, x, 0.0);

    // Call one full multigrid cycle
    fasp_solver_fmgcycle(mgl, param);
    
    // Form residual r = b-A*x
    fasp_dvec_cp(b, r);
    fasp_blas_dcsr_aAxpy(-1.0, ptrA, x->val, r->val);
    
    // Compute norms of r and convergence factor
    absres  = fasp_blas_dvec_norm2(r);    // residual ||r||
    relres1 = absres/MAX(SMALLREAL,sumb); // relative residual ||r||/||b||
    
    if ( prtlvl > PRINT_NONE ) {
        printf("FMG finishes with relative residual %e.\n", relres1);
        fasp_gettime(&solve_end);
        fasp_cputime("FMG solve", solve_end - solve_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
