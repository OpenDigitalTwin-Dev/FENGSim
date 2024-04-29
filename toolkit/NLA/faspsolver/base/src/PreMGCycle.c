/*! \file  PreMGCycle.c
 *
 *  \brief Abstract multigrid cycle -- non-recursive version
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMessage.c, AuxVector.c, BlaArray.c, BlaSchwarzSetup.c,
 *         BlaSpmvBSR.c, BlaSpmvCSR.c, ItrSmootherBSR.c, ItrSmootherCSR.c,
 *         ItrSmootherCSRpoly.c, KryPcg.c, KryPvgmres.c, KrySPcg.c,
 *         and KrySPvgmres.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "PreMGSmoother.inl"
#include "PreMGUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_solver_mgcycle (AMG_data *mgl, AMG_param *param)
 *
 * \brief Solve Ax=b with non-recursive multigrid cycle
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \author Chensong Zhang
 * \date   10/06/2010
 *
 * Modified by Chensong Zhang on 02/27/2013: update direct solvers.
 * Modified by Chensong Zhang on 12/30/2014: update Schwarz smoothers.
 */
void fasp_solver_mgcycle(AMG_data* mgl, AMG_param* param)
{
    const SHORT prtlvl        = param->print_level;
    const SHORT amg_type      = param->AMG_type;
    const SHORT smoother      = param->smoother;
    const SHORT smooth_order  = param->smooth_order;
    const SHORT cycle_type    = param->cycle_type;
    const SHORT coarse_solver = param->coarse_solver;
    const SHORT nl            = mgl[0].num_levels;
    const REAL  relax         = param->relaxation;
    const REAL  tol           = param->tol * 1e-4;
    const SHORT ndeg          = param->polynomial_degree;

    // Schwarz parameters
    SWZ_param swzparam;
    if (param->SWZ_levels > 0) {
        swzparam.SWZ_blksolver = param->SWZ_blksolver;
    }

    // local variables
    REAL alpha                = 1.0;
    INT  num_lvl[MAX_AMG_LVL] = {0}, l = 0;

    // more general cycling types on each level --zcs 05/07/2020
    INT   ncycles[MAX_AMG_LVL] = {1};
    SHORT i;
    for (i = 0; i < MAX_AMG_LVL; ++i) ncycles[i] = 1; // initially V-cycle
    switch (cycle_type) {
        case 12:
            for (i = MAX_AMG_LVL - 2; i > 0; i -= 2) ncycles[i] = 2;
            break;
        case 21:
            for (i = MAX_AMG_LVL - 1; i > 0; i -= 2) ncycles[i] = 2;
            break;
        default:
            for (i = 0; i < MAX_AMG_LVL; i += 1) ncycles[i] = cycle_type;
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: n=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.nnz);
#endif

#if DEBUG_MODE > 1
    printf("### DEBUG: AMG_level = %d, ILU_level = %d\n", nl, mgl->ILU_levels);
#endif

ForwardSweep:
    while (l < nl - 1) {

        num_lvl[l]++;

        // pre-smoothing with ILU method
        if (l < mgl->ILU_levels) {
            fasp_smoother_dcsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
        }

        // or pre-smoothing with Schwarz method
        else if (l < mgl->SWZ_levels) {
            switch (mgl[l].Schwarz.SWZ_type) {
                case SCHWARZ_SYMMETRIC:
                    fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                          &mgl[l].b);
                    fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                           &mgl[l].b);
                    break;
                default:
                    fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                          &mgl[l].b);
                    break;
            }
        }

        // or pre-smoothing with standard smoother
        else {
#if MULTI_COLOR_ORDER
            // printf("fasp_smoother_dcsr_gs_multicolor, %s, %d\n",  __FUNCTION__,
            // __LINE__);
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->presmooth_iter, 1);
#else
            fasp_dcsr_presmoothing(smoother, &mgl[l].A, &mgl[l].b, &mgl[l].x,
                                   param->presmooth_iter, 0, mgl[l].A.row - 1, 1, relax,
                                   ndeg, smooth_order, mgl[l].cfmark.val);
#endif
        }

        // form residual r = b - A x
        fasp_darray_cp(mgl[l].A.row, mgl[l].b.val, mgl[l].w.val);
        fasp_blas_dcsr_aAxpy(-1.0, &mgl[l].A, mgl[l].x.val, mgl[l].w.val);

        // restriction r1 = R*r0
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_mxv_agg(&mgl[l].R, mgl[l].w.val, mgl[l + 1].b.val);
                break;
            default:
                fasp_blas_dcsr_mxv(&mgl[l].R, mgl[l].w.val, mgl[l + 1].b.val);
                break;
        }

        // prepare for the next level
        ++l;
        fasp_dvec_set(mgl[l].A.row, &mgl[l].x, 0.0);
    }

    // If AMG only has one level or we have arrived at the coarsest level,
    // call the coarse space solver:
    switch (coarse_solver) {

#if WITH_PARDISO
        case SOLVER_PARDISO:
            {
                /* use Intel MKL PARDISO direct solver on the coarsest level */
                fasp_pardiso_solve(&mgl[nl - 1].A, &mgl[nl - 1].b, &mgl[nl - 1].x,
                                   &mgl[nl - 1].pdata, 0);
                break;
            }
#endif

#if WITH_MUMPS
        case SOLVER_MUMPS:
            {
                // use MUMPS direct solver on the coarsest level
                mgl[nl - 1].mumps.job = 2;
                fasp_solver_mumps_steps(&mgl[nl - 1].A, &mgl[nl - 1].b, &mgl[nl - 1].x,
                                        &mgl[nl - 1].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        case SOLVER_UMFPACK:
            {
                // use UMFPACK direct solver on the coarsest level
                fasp_umfpack_solve(&mgl[nl - 1].A, &mgl[nl - 1].b, &mgl[nl - 1].x,
                                   mgl[nl - 1].Numeric, 0);
                break;
            }
#endif

#if WITH_SuperLU
        case SOLVER_SUPERLU:
            {
                // use SuperLU direct solver on the coarsest level
                fasp_solver_superlu(&mgl[nl - 1].A, &mgl[nl - 1].b, &mgl[nl - 1].x, 0);
                break;
            }
#endif

        default:
            // use iterative solver on the coarsest level
            fasp_coarse_itsolver(&mgl[nl - 1].A, &mgl[nl - 1].b, &mgl[nl - 1].x, tol,
                                 prtlvl);
    }

    // BackwardSweep:
    while (l > 0) {

        --l;

        // find the optimal scaling factor alpha
        if (param->coarse_scaling == ON) {
            alpha =
                fasp_blas_darray_dotprod(mgl[l + 1].A.row, mgl[l + 1].x.val,
                                         mgl[l + 1].b.val) /
                fasp_blas_dcsr_vmv(&mgl[l + 1].A, mgl[l + 1].x.val, mgl[l + 1].x.val);
            alpha = MIN(alpha, 1.0); // Add this for safety! --Chensong on 10/04/2014
        }

        // prolongation u = u + alpha*P*e1
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_aAxpy_agg(alpha, &mgl[l].P, mgl[l + 1].x.val,
                                         mgl[l].x.val);
                break;
            default:
                fasp_blas_dcsr_aAxpy(alpha, &mgl[l].P, mgl[l + 1].x.val, mgl[l].x.val);
                break;
        }

        // post-smoothing with ILU method
        if (l < mgl->ILU_levels) {
            fasp_smoother_dcsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
        }

        // post-smoothing with Schwarz method
        else if (l < mgl->SWZ_levels) {
            switch (mgl[l].Schwarz.SWZ_type) {
                case SCHWARZ_SYMMETRIC:
                    fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                           &mgl[l].b);
                    fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                          &mgl[l].b);
                    break;
                default:
                    fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam, &mgl[l].x,
                                           &mgl[l].b);
                    break;
            }
        }

        // post-smoothing with standard methods
        else {
#if MULTI_COLOR_ORDER
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->postsmooth_iter, -1);
#else
            fasp_dcsr_postsmoothing(smoother, &mgl[l].A, &mgl[l].b, &mgl[l].x,
                                    param->postsmooth_iter, 0, mgl[l].A.row - 1, -1,
                                    relax, ndeg, smooth_order, mgl[l].cfmark.val);
#endif
        }

        // General cycling on each level --zcs
        if (num_lvl[l] < ncycles[l])
            break;
        else
            num_lvl[l] = 0;
    }

    if (l > 0) goto ForwardSweep;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * \fn void fasp_solver_mgcycle_bsr (AMG_data_bsr *mgl, AMG_param *param)
 *
 * \brief Solve Ax=b with non-recursive multigrid cycle
 *
 * \param mgl    Pointer to AMG data: AMG_data_bsr
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \author Xiaozhe Hu
 * \date   08/07/2011
 */
void fasp_solver_mgcycle_bsr(AMG_data_bsr* mgl, AMG_param* param)
{
    const SHORT prtlvl        = param->print_level;
    const SHORT nl            = mgl[0].num_levels;
    const SHORT smoother      = param->smoother;
    const SHORT cycle_type    = param->cycle_type;
    const SHORT coarse_solver = param->coarse_solver;
    const REAL  relax         = param->relaxation;
    INT         steps         = param->presmooth_iter;

    // local variables
    INT  nu_l[MAX_AMG_LVL] = {0}, l = 0;
    REAL alpha = 1.0;
    INT  i;

    dvector r_nk, z_nk;

    if (mgl[0].A_nk != NULL) {
        fasp_dvec_alloc(mgl[0].A_nk->row, &r_nk);
        fasp_dvec_alloc(mgl[0].A_nk->row, &z_nk);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

#if DEBUG_MODE > 1
    printf("### DEBUG: AMG_level = %d, ILU_level = %d\n", nl, mgl->ILU_levels);
#endif

ForwardSweep:
    while (l < nl - 1) {
        nu_l[l]++;
        // pre smoothing
        if (l < mgl->ILU_levels) {
            fasp_smoother_dbsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
            for (i = 0; i < steps; i++)
                fasp_smoother_dbsr_gs_ascend(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                             mgl[l].diaginv.val);
        } else {
            if (steps > 0) {
                switch (smoother) {
                    case SMOOTHER_JACOBI:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_jacobi1(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                                       mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_GS:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_gs_ascend(&mgl[l].A, &mgl[l].b,
                                                         &mgl[l].x, mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_SGS:
                        for (i = 0; i < steps; i++) {
                            fasp_smoother_dbsr_gs_ascend(&mgl[l].A, &mgl[l].b,
                                                         &mgl[l].x, mgl[l].diaginv.val);
                            fasp_smoother_dbsr_gs_descend(
                                &mgl[l].A, &mgl[l].b, &mgl[l].x, mgl[l].diaginv.val);
                        }
                        break;
                    case SMOOTHER_SOR:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_sor_ascend(&mgl[l].A, &mgl[l].b,
                                                          &mgl[l].x, mgl[l].diaginv.val,
                                                          relax);
                        break;
                    case SMOOTHER_SSOR:
                        for (i = 0; i < steps; i++) {
                            fasp_smoother_dbsr_sor_ascend(&mgl[l].A, &mgl[l].b,
                                                          &mgl[l].x, mgl[l].diaginv.val,
                                                          relax);
                        }
                        fasp_smoother_dbsr_sor_descend(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                                       mgl[l].diaginv.val, relax);
                        break;
                    default:
                        printf("### ERROR: Unknown smoother type %d!\n", smoother);
                        fasp_chkerr(ERROR_SOLVER_TYPE, __FUNCTION__);
                }
            }
        }

        // extra kernel solve
        if (mgl[l].A_nk != NULL) {

            //--------------------------------------------
            // extra kernel solve
            //--------------------------------------------
            // form residual r = b - A x
            fasp_darray_cp(mgl[l].A.ROW * mgl[l].A.nb, mgl[l].b.val, mgl[l].w.val);
            fasp_blas_dbsr_aAxpy(-1.0, &mgl[l].A, mgl[l].x.val, mgl[l].w.val);

            // r_nk = R_nk*r
            fasp_blas_dcsr_mxv(mgl[l].R_nk, mgl[l].w.val, r_nk.val);

            // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
            fasp_solver_umfpack(mgl[l].A_nk, &r_nk, &z_nk, 0);
#else
            fasp_coarse_itsolver(mgl[l].A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif

            // z = z + P_nk*z_nk;
            fasp_blas_dcsr_aAxpy(1.0, mgl[l].P_nk, z_nk.val, mgl[l].x.val);
        }

        // form residual r = b - A x
        fasp_darray_cp(mgl[l].A.ROW * mgl[l].A.nb, mgl[l].b.val, mgl[l].w.val);
        fasp_blas_dbsr_aAxpy(-1.0, &mgl[l].A, mgl[l].x.val, mgl[l].w.val);

        // restriction r1 = R*r0
        fasp_blas_dbsr_mxv(&mgl[l].R, mgl[l].w.val, mgl[l + 1].b.val);

        // prepare for the next level
        ++l;
        fasp_dvec_set(mgl[l].A.ROW * mgl[l].A.nb, &mgl[l].x, 0.0);
    }

    // If AMG only has one level or we have arrived at the coarsest level,
    // call the coarse space solver:
    switch (coarse_solver) {

#if WITH_PARDISO
        case SOLVER_PARDISO:
            {
                /* use Intel MKL PARDISO direct solver on the coarsest level */
                fasp_pardiso_solve(&mgl[nl - 1].Ac, &mgl[nl - 1].b, &mgl[nl - 1].x,
                                   &mgl[nl - 1].pdata, 0);
                break;
            }
#endif

#if WITH_MUMPS
        case SOLVER_MUMPS:
            /* use MUMPS direct solver on the coarsest level */
            mgl[nl - 1].mumps.job = 2;
            fasp_solver_mumps_steps(&mgl[nl - 1].Ac, &mgl[nl - 1].b, &mgl[nl - 1].x,
                                    &mgl[nl - 1].mumps);
            break;
#endif

#if WITH_UMFPACK
        case SOLVER_UMFPACK:
            /* use UMFPACK direct solver on the coarsest level */
            fasp_umfpack_solve(&mgl[nl - 1].Ac, &mgl[nl - 1].b, &mgl[nl - 1].x,
                               mgl[nl - 1].Numeric, 0);
            break;
#endif

#if WITH_SuperLU
        case SOLVER_SUPERLU:
            /* use SuperLU direct solver on the coarsest level */
            fasp_solver_superlu(&mgl[nl - 1].Ac, &mgl[nl - 1].b, &mgl[nl - 1].x, 0);
            break;
#endif

        default:
            {
                /* use iterative solver on the coarsest level */
                const INT  csize  = mgl[nl - 1].A.ROW * mgl[nl - 1].A.nb;
                const INT  cmaxit = MIN(csize * csize, 200);
                const REAL ctol   = param->tol;  // coarse level tolerance
                const REAL atol   = ctol * 1e-8; // coarse level tolerance
                if (fasp_solver_dbsr_pvgmres(&mgl[nl - 1].A, &mgl[nl - 1].b,
                                             &mgl[nl - 1].x, NULL, ctol, atol, cmaxit,
                                             25, 1, 0) < 0) {
                    if (prtlvl > PRINT_MIN) {
                        printf("### WARNING: Coarse level solver did not converge!\n");
                        printf("### WARNING: Consider to increase maxit to %d!\n",
                               2 * cmaxit);
                    }
                }
            }
    }

    // BackwardSweep:
    while (l > 0) {
        --l;

        // prolongation u = u + alpha*P*e1
        if (param->coarse_scaling == ON) {
            dvector PeH, Aeh;
            PeH.row = Aeh.row = mgl[l].b.row;
            PeH.val           = mgl[l].w.val + mgl[l].b.row;
            Aeh.val           = PeH.val + mgl[l].b.row;

            fasp_blas_dbsr_mxv(&mgl[l].P, mgl[l + 1].x.val, PeH.val);
            fasp_blas_dbsr_mxv(&mgl[l].A, PeH.val, Aeh.val);

            alpha = (fasp_blas_darray_dotprod(mgl[l].b.row, Aeh.val, mgl[l].w.val)) /
                    (fasp_blas_darray_dotprod(mgl[l].b.row, Aeh.val, Aeh.val));
            alpha = MIN(alpha, 1.0); // Add this for safety! --Chensong on 10/04/2014
            fasp_blas_darray_axpy(mgl[l].b.row, alpha, PeH.val, mgl[l].x.val);
        } else {
            fasp_blas_dbsr_aAxpy(alpha, &mgl[l].P, mgl[l + 1].x.val, mgl[l].x.val);
        }

        // extra kernel solve
        if (mgl[l].A_nk != NULL) {
            //--------------------------------------------
            // extra kernel solve
            //--------------------------------------------
            // form residual r = b - A x
            fasp_darray_cp(mgl[l].A.ROW * mgl[l].A.nb, mgl[l].b.val, mgl[l].w.val);
            fasp_blas_dbsr_aAxpy(-1.0, &mgl[l].A, mgl[l].x.val, mgl[l].w.val);

            // r_nk = R_nk*r
            fasp_blas_dcsr_mxv(mgl[l].R_nk, mgl[l].w.val, r_nk.val);

            // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
            fasp_solver_umfpack(mgl[l].A_nk, &r_nk, &z_nk, 0);
#else
            fasp_coarse_itsolver(mgl[l].A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif

            // z = z + P_nk*z_nk;
            fasp_blas_dcsr_aAxpy(1.0, mgl[l].P_nk, z_nk.val, mgl[l].x.val);
        }

        // post-smoothing
        if (l < mgl->ILU_levels) {
            fasp_smoother_dbsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
            for (i = 0; i < steps; i++)
                fasp_smoother_dbsr_gs_descend(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                              mgl[l].diaginv.val);
        } else {
            if (steps > 0) {
                switch (smoother) {
                    case SMOOTHER_JACOBI:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_jacobi1(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                                       mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_GS:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_gs_descend(
                                &mgl[l].A, &mgl[l].b, &mgl[l].x, mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_SGS:
                        for (i = 0; i < steps; i++) {
                            fasp_smoother_dbsr_gs_ascend(&mgl[l].A, &mgl[l].b,
                                                         &mgl[l].x, mgl[l].diaginv.val);
                            fasp_smoother_dbsr_gs_descend(
                                &mgl[l].A, &mgl[l].b, &mgl[l].x, mgl[l].diaginv.val);
                        }
                        break;
                    case SMOOTHER_SOR:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_sor_descend(&mgl[l].A, &mgl[l].b,
                                                           &mgl[l].x,
                                                           mgl[l].diaginv.val, relax);
                        break;
                    case SMOOTHER_SSOR:
                        for (i = 0; i < steps; i++)
                            fasp_smoother_dbsr_sor_ascend(&mgl[l].A, &mgl[l].b,
                                                          &mgl[l].x, mgl[l].diaginv.val,
                                                          relax);
                        fasp_smoother_dbsr_sor_descend(&mgl[l].A, &mgl[l].b, &mgl[l].x,
                                                       mgl[l].diaginv.val, relax);
                        break;
                    default:
                        printf("### ERROR: Unknown smoother type %d!\n", smoother);
                        fasp_chkerr(ERROR_SOLVER_TYPE, __FUNCTION__);
                }
            }
        }

        if (nu_l[l] < cycle_type)
            break;
        else
            nu_l[l] = 0;
    }

    if (l > 0) goto ForwardSweep;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
