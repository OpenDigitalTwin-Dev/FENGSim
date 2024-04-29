/*! \file  PreMGSmoother.inl
 *
 *  \brief Routines for multigrid smoothers
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreMGCycle.c, PreMGCycleFull.c, PreMGRecur.c, and PreMGRecurAMLI.c
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
 * \fn static void fasp_dcsr_presmoothing (const SHORT smoother, dCSRmat *A,
 *                                         dvector *b, dvector *x,
 *                                         const INT nsweeps, const INT istart,
 *                                         const INT iend, const INT istep,
 *                                         const REAL relax, const SHORT ndeg,
 *                                         const SHORT order, INT *ordering)
 *
 * \brief Multigrid presmoothing
 *
 * \param  smoother  type of smoother
 * \param  A         pointer to matrix data
 * \param  b         pointer to rhs data
 * \param  x         pointer to sol data
 * \param  nsweeps   number of smoothing sweeps
 * \param  istart    starting index
 * \param  iend      ending index
 * \param  istep     step size
 * \param  relax     relaxation parameter or weight for smoothers
 * \param  ndeg      degree of the polynomial smoother
 * \param  order     order for smoothing sweeps
 * \param  ordering  user defined ordering
 *
 * \author Chensong Zhang
 * \date   01/10/2012
 *
 * Modified by Xiaozhe on 06/04/2012: add ndeg as input
 * Modified by Chensong on 02/16/2013: GS -> SMOOTHER_GS, etc
 */
static void fasp_dcsr_presmoothing(const SHORT smoother,
                                   dCSRmat*    A,
                                   dvector*    b,
                                   dvector*    x,
                                   const INT   nsweeps,
                                   const INT   istart,
                                   const INT   iend,
                                   const INT   istep,
                                   const REAL  relax,
                                   const SHORT ndeg,
                                   const SHORT order,
                                   INT*        ordering)
{
    switch (smoother) {

        case SMOOTHER_JACOBIF:
            fasp_smoother_dcsr_jacobi_ff(x, A, b, nsweeps, ordering, relax);
            break;

        case SMOOTHER_GS:
            if (order == NO_ORDER || ordering == NULL)
                fasp_smoother_dcsr_gs(x, istart, iend, istep, A, b, nsweeps);
            else if (order == CF_ORDER)
                fasp_smoother_dcsr_gs_cf(x, A, b, nsweeps, ordering, 1);
            break;

        case SMOOTHER_GSF:
            fasp_smoother_dcsr_gs_ff(x, A, b, nsweeps, ordering);
            break;

        case SMOOTHER_SGS:
            fasp_smoother_dcsr_sgs(x, A, b, nsweeps);
            break;

        case SMOOTHER_JACOBI:
            fasp_smoother_dcsr_jacobi(x, istart, iend, istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_L1DIAG:
            fasp_smoother_dcsr_L1diag(x, istart, iend, istep, A, b, nsweeps);
            break;

        case SMOOTHER_POLY:
            fasp_smoother_dcsr_poly(A, b, x, iend + 1, ndeg, nsweeps);
            break;

        case SMOOTHER_SOR:
            fasp_smoother_dcsr_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_SSOR:
            fasp_smoother_dcsr_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_GSOR:
            fasp_smoother_dcsr_gs(x, istart, iend, istep, A, b, nsweeps);
            fasp_smoother_dcsr_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_SGSOR:
            fasp_smoother_dcsr_gs(x, istart, iend, istep, A, b, nsweeps);
            fasp_smoother_dcsr_gs(x, iend, istart, -istep, A, b, nsweeps);
            fasp_smoother_dcsr_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_CG:
            fasp_solver_dcsr_pcg(A, b, x, NULL, 1e-3, 1e-15, nsweeps, 1, PRINT_NONE);
            break;

        default:
            printf("### ERROR: Unknown smoother type %d!\n", smoother);
            fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
    }
}

/**
 * \fn static void fasp_dcsr_postsmoothing (const SHORT smoother, dCSRmat *A,
 *                                          dvector *b, dvector *x,
 *                                          const INT nsweeps, const INT istart,
 *                                          const INT iend, const INT istep,
 *                                          const REAL relax, const SHORT ndeg,
 *                                          const SHORT order, INT *ordering)
 *
 * \brief Multigrid presmoothing
 *
 * \param  smoother  type of smoother
 * \param  A         pointer to matrix data
 * \param  b         pointer to rhs data
 * \param  x         pointer to sol data
 * \param  nsweeps   number of smoothing sweeps
 * \param  istart    starting index
 * \param  iend      ending index
 * \param  istep     step size
 * \param  relax     relaxation parameter or weight for smoothers
 * \param  ndeg      degree of the polynomial smoother
 * \param  order     order for smoothing sweeps
 * \param  ordering  user defined ordering
 *
 * \author Chensong Zhang
 * \date   01/10/2012
 *
 * Modified by Xiaozhe Hu on 06/04/2012: add ndeg as input
 * Modified by Chensong on 02/16/2013: GS -> SMOOTHER_GS, etc
 */
static void fasp_dcsr_postsmoothing(const SHORT smoother,
                                    dCSRmat*    A,
                                    dvector*    b,
                                    dvector*    x,
                                    const INT   nsweeps,
                                    const INT   istart,
                                    const INT   iend,
                                    const INT   istep,
                                    const REAL  relax,
                                    const SHORT ndeg,
                                    const SHORT order,
                                    INT*        ordering)
{
    switch (smoother) {

        case SMOOTHER_JACOBIF:
            fasp_smoother_dcsr_jacobi_ff(x, A, b, nsweeps, ordering, relax);
            break;
            
        case SMOOTHER_GS:
            if (order == NO_ORDER || ordering == NULL) {
                fasp_smoother_dcsr_gs(x, iend, istart, istep, A, b, nsweeps);
            } else if (order == CF_ORDER)
                fasp_smoother_dcsr_gs_cf(x, A, b, nsweeps, ordering, -1);
            break;

        case SMOOTHER_GSF:
            fasp_smoother_dcsr_gs_ff(x, A, b, nsweeps, ordering);
            break;

        case SMOOTHER_SGS:
            fasp_smoother_dcsr_sgs(x, A, b, nsweeps);
            break;

        case SMOOTHER_JACOBI:
            fasp_smoother_dcsr_jacobi(x, iend, istart, istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_L1DIAG:
            fasp_smoother_dcsr_L1diag(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SMOOTHER_POLY:
            fasp_smoother_dcsr_poly(A, b, x, iend + 1, ndeg, nsweeps);
            break;

        case SMOOTHER_SOR:
            fasp_smoother_dcsr_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_SSOR:
            fasp_smoother_dcsr_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            break;

        case SMOOTHER_GSOR:
            fasp_smoother_dcsr_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_gs(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SMOOTHER_SGSOR:
            fasp_smoother_dcsr_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            fasp_smoother_dcsr_gs(x, istart, iend, -istep, A, b, nsweeps);
            fasp_smoother_dcsr_gs(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SMOOTHER_CG:
            fasp_solver_dcsr_pcg(A, b, x, NULL, 1e-3, 1e-15, nsweeps, 1, PRINT_NONE);
            break;

        default:
            printf("### ERROR: Unknown smoother type %d!\n", smoother);
            fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
