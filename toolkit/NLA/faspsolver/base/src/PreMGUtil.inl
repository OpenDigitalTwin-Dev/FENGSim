/*! \file  PreMGUtil.inl
 *
 *  \brief Routines for multigrid coarsest level solver
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreBSR.c, PreCSR.c, PreMGCycle.c, PreMGCycleFull.c, PreMGRecur.c,
 *         and PreMGRecurAMLI.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  \warning This file is also used in FASP4BLKOIL!!!
 */

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static inline void fasp_coarse_itsolver (cosnt dCSRmat *A, cosnt dvector *b,
 *                                              dvector *x, const REAL ctol,
 *                                              const SHORT PrtLvl)
 *
 * \brief Iterative solver on the coarset level of multigrid methods
 *
 * \param  A         pointer to matrix data
 * \param  b         pointer to rhs data
 * \param  x         pointer to sol data
 * \param  ctol      tolerance for the coarsest level
 * \param  PrtLvl    level of output
 *
 * \author Chensong Zhang
 * \date   01/10/2012
 */
static inline void fasp_coarse_itsolver (const dCSRmat  *A,
                                         const dvector  *b,
                                         dvector        *x,
                                         const REAL      ctol,
                                         const SHORT     PrtLvl)
{
    const INT n     = A->row;
    const INT maxit = MAX(250,MIN(n*n, 1000)); // Should NOT be less!

    // Default solver is SPCG for safty purposes
    INT status = fasp_solver_dcsr_spcg(A, b, x, NULL, ctol, maxit, 1, PrtLvl - 4);
    
    // If CG fails to converge, use SPVGMRES as a safety net
    if ( status < 0 ) {
        status = fasp_solver_dcsr_spvgmres(A, b, x, NULL, ctol, maxit, 20, 1, PrtLvl - 4);
    }
    
    if ( status < 0 && PrtLvl >= PRINT_MORE ) {
        printf("### WARNING: Coarse level solver did not converge!\n");
        printf("### WARNING: Consider to increase maxit to %d!\n", 2*maxit);
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
