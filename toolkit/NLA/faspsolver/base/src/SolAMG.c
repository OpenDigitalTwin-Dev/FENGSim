/*! \file  SolAMG.c
 *
 *  \brief AMG method as an iterative solver
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxMessage.c, AuxTiming.c, AuxVector.c, BlaSparseCheck.c,
 *         BlaSparseCSR.c, KrySPgmres.c, PreAMGSetupRS.c, PreAMGSetupSA.c,
 *         PreAMGSetupUA.c, PreDataInit.c, and PreMGSolve.c
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
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_amg (dCSRmat *A, dvector *b, dvector *x,
 *                          AMG_param *param)
 *
 * \brief Solve Ax = b by algebraic multigrid methods
 *
 * \param A      Pointer to dCSRmat: the coefficient matrix
 * \param b      Pointer to dvector: the right hand side
 * \param x      Pointer to dvector: the unknowns
 * \param param  Pointer to AMG_param: AMG parameters
 *
 * \return       Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 *
 * \note Refer to "Multigrid"
 *       by U. Trottenberg, C. W. Oosterlee and A. Schuller
 *       Appendix A.7 (by A. Brandt, P. Oswald and K. Stuben)
 *       Academic Press Inc., San Diego, CA, 2001.
 *
 * Modified by Chensong Zhang on 07/26/2014: Add error handling for AMG setup
 * Modified by Chensong Zhang on 02/01/2021: Add return value
 */
INT fasp_solver_amg(dCSRmat* A, dvector* b, dvector* x, AMG_param* param)
{
    const REAL  tol        = param->tol;
    const SHORT max_levels = param->max_levels;
    const SHORT prtlvl     = param->print_level;
    const SHORT amg_type   = param->AMG_type;
    const SHORT cycle_type = param->cycle_type;
    const INT   maxit      = param->maxit;
    const INT   nnz = A->nnz, m = A->row, n = A->col;

    // local variables
    SHORT     status;
    INT       iter      = 0;
    AMG_data* mgl       = fasp_amg_data_create(max_levels);
    REAL      AMG_start = 0, AMG_end;

#if MULTI_COLOR_ORDER
    A->color = 0;
    A->IC    = NULL;
    A->ICMAP = NULL;
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    if (prtlvl > PRINT_NONE) fasp_gettime(&AMG_start);

    // check matrix data
    fasp_check_dCSRmat(A);

    // Step 0: initialize mgl[0] with A, b and x
    mgl[0].A = fasp_dcsr_create(m, n, nnz);
    fasp_dcsr_cp(A, &mgl[0].A);

    mgl[0].b = fasp_dvec_create(n);
    fasp_dvec_cp(b, &mgl[0].b);

    mgl[0].x = fasp_dvec_create(n);
    fasp_dvec_cp(x, &mgl[0].x);

    // Step 1: AMG setup phase
    switch (amg_type) {

        case SA_AMG: // Smoothed Aggregation AMG setup
            status = fasp_amg_setup_sa(mgl, param);
            break;

        case UA_AMG: // Unsmoothed Aggregation AMG setup
            status = fasp_amg_setup_ua(mgl, param);
            break;

        default: // Classical AMG setup
            status = fasp_amg_setup_rs(mgl, param);
            break;
    }

    // Step 2: AMG solve phase
    if (status == FASP_SUCCESS) { // call a multilevel cycle

        switch (cycle_type) {

            case AMLI_CYCLE: // AMLI-cycle
                iter = fasp_amg_solve_amli(mgl, param);
                break;

            case NL_AMLI_CYCLE: // Nonlinear AMLI-cycle
                iter = fasp_amg_solve_namli(mgl, param);
                break;

            default: // V,W-cycles or hybrid cycles (determined by param)
                iter = fasp_amg_solve(mgl, param);
                break;
        }

        fasp_dvec_cp(&mgl[0].x, x);

    }

    else { // call a backup solver

        if (prtlvl > PRINT_MIN) {
            printf("### WARNING: AMG setup failed!\n");
            printf("### WARNING: Use a backup solver instead!\n");
        }
        fasp_solver_dcsr_spgmres(A, b, x, NULL, tol, maxit, 20, 1, prtlvl);
    }

    // clean-up memory
    fasp_amg_data_free(mgl, param);

    // print out CPU time if needed
    if (prtlvl > PRINT_NONE) {
        fasp_gettime(&AMG_end);
        fasp_cputime("AMG totally", AMG_end - AMG_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
