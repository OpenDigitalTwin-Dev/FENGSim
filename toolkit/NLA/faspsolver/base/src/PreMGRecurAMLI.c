/*! \file  PreMGRecurAMLI.c
 *
 *  \brief Abstract AMLI multilevel iteration -- recursive version
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxParam.c, AuxVector.c,
 *         BlaSchwarzSetup.c, BlaArray.c, BlaSpmvBSR.c, BlaSpmvCSR.c,
 *         ItrSmootherBSR.c, ItrSmootherCSR.c, ItrSmootherCSRpoly.c, KryPcg.c,
 *         KryPvfgmres.c, KrySPcg.c, KrySPvgmres.c, PreBSR.c, and PreCSR.c
 *
 *  \note  This file includes both AMLI and non-linear AMLI cycles
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

#include "PreMGRecurAMLI.inl"
#include "PreMGSmoother.inl"
#include "PreMGUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_solver_amli (AMG_data *mgl, AMG_param *param, INT l)
 *
 * \brief Solve Ax=b with recursive AMLI-cycle
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 * \param l      Current level
 *
 * \author Xiaozhe Hu
 * \date   01/23/2011
 *
 * \note AMLI polynomial computed by the best approximation of 1/x.
 *       Refer to Johannes K. Kraus, Panayot S. Vassilevski, Ludmil T. Zikatanov,
 *       "Polynomial of best uniform approximation to $x^{-1}$ and smoothing in
 *        two-level methods", 2013.
 *
 * Modified by Chensong Zhang on 02/27/2013: update direct solvers.
 * Modified by Zheng Li on 11/10/2014: update direct solvers.
 * Modified by Hongxuan Zhang on 12/15/2015: update direct solvers.
 */
void fasp_solver_amli(AMG_data* mgl, AMG_param* param, INT l)
{
    const SHORT amg_type      = param->AMG_type;
    const SHORT prtlvl        = param->print_level;
    const SHORT smoother      = param->smoother;
    const SHORT smooth_order  = param->smooth_order;
    const SHORT coarse_solver = param->coarse_solver;
    const SHORT degree        = param->amli_degree;
    const REAL  relax         = param->relaxation;
    const REAL  tol           = param->tol * 1e-4;
    const SHORT ndeg          = param->polynomial_degree;

    // local variables
    REAL  alpha = 1.0;
    REAL* coef  = param->amli_coef;

    dvector *b0 = &mgl[l].b, *e0 = &mgl[l].x;         // fine level b and x
    dvector *b1 = &mgl[l + 1].b, *e1 = &mgl[l + 1].x; // coarse level b and x

    dCSRmat* A0 = &mgl[l].A;                          // fine level matrix
    dCSRmat* A1 = &mgl[l + 1].A;                      // coarse level matrix

    const INT m0 = A0->row, m1 = A1->row;

    INT*      ordering = mgl[l].cfmark.val;     // smoother ordering
    ILU_data* LU_level = &mgl[l].LU;            // fine level ILU decomposition
    REAL*     r        = mgl[l].w.val;          // work array for residual
    REAL*     r1       = mgl[l + 1].w.val + m1; // work array for residual

    // Schwarz parameters
    SWZ_param swzparam;
    if (param->SWZ_levels > 0) {
        swzparam.SWZ_blksolver = param->SWZ_blksolver;
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: n=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.nnz);
#endif

    if (prtlvl >= PRINT_MOST) printf("AMLI level %d, smoother %d.\n", l, smoother);

    if (l < mgl[l].num_levels - 1) {

        // pre smoothing
        if (l < mgl[l].ILU_levels) {

            fasp_smoother_dcsr_ilu(A0, b0, e0, LU_level);

        }

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

        else {
#if MULTI_COLOR_ORDER
            // printf("fasp_smoother_dcsr_gs_multicolor, %s, %d\n",  __FUNCTION__,
            // __LINE__);
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->presmooth_iter, 1);
#else
            fasp_dcsr_presmoothing(smoother, A0, b0, e0, param->presmooth_iter, 0,
                                   m0 - 1, 1, relax, ndeg, smooth_order, ordering);
#endif
        }

        // form residual r = b - A x
        fasp_darray_cp(m0, b0->val, r);
        fasp_blas_dcsr_aAxpy(-1.0, A0, e0->val, r);

        // restriction r1 = R*r0
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_mxv_agg(&mgl[l].R, r, b1->val);
                break;
            default:
                fasp_blas_dcsr_mxv(&mgl[l].R, r, b1->val);
                break;
        }

        // coarse grid correction
        {
            INT i;

            fasp_darray_cp(m1, b1->val, r1);

            for (i = 1; i <= degree; i++) {
                fasp_dvec_set(m1, e1, 0.0);
                fasp_solver_amli(mgl, param, l + 1);

                // b1 = (coef[degree-i]/coef[degree])*r1 + A1*e1;
                // First, compute b1 = A1*e1
                fasp_blas_dcsr_mxv(A1, e1->val, b1->val);
                // Then, compute b1 = b1 + (coef[degree-i]/coef[degree])*r1
                fasp_blas_darray_axpy(m1, coef[degree - i] / coef[degree], r1, b1->val);
            }

            fasp_dvec_set(m1, e1, 0.0);
            fasp_solver_amli(mgl, param, l + 1);
        }

        // find the optimal scaling factor alpha
        fasp_blas_darray_ax(m1, coef[degree], e1->val);
        if (param->coarse_scaling == ON) {
            alpha = fasp_blas_darray_dotprod(m1, e1->val, r1) /
                    fasp_blas_dcsr_vmv(A1, e1->val, e1->val);
            alpha = MIN(alpha, 1.0);
        }

        // prolongation e0 = e0 + alpha * P * e1
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_aAxpy_agg(alpha, &mgl[l].P, e1->val, e0->val);
                break;
            default:
                fasp_blas_dcsr_aAxpy(alpha, &mgl[l].P, e1->val, e0->val);
                break;
        }

        // post smoothing
        if (l < mgl[l].ILU_levels) {

            fasp_smoother_dcsr_ilu(A0, b0, e0, LU_level);

        }

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

        else {
#if MULTI_COLOR_ORDER
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->postsmooth_iter, -1);
#else
            fasp_dcsr_postsmoothing(smoother, A0, b0, e0, param->postsmooth_iter, 0,
                                    m0 - 1, -1, relax, ndeg, smooth_order, ordering);
#endif
        }

    }

    else { // coarsest level solver

        switch (coarse_solver) {

#if WITH_PARDISO
            case SOLVER_PARDISO:
                {
                    /* use Intel MKL PARDISO direct solver on the coarsest level */
                    fasp_pardiso_solve(A0, b0, e0, &mgl[l].pdata, 0);
                    break;
                }
#endif

#if WITH_SuperLU
            case SOLVER_SUPERLU:
                /* use SuperLU direct solver on the coarsest level */
                fasp_solver_superlu(A0, b0, e0, 0);
                break;
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                /* use UMFPACK direct solver on the coarsest level */
                fasp_umfpack_solve(A0, b0, e0, mgl[l].Numeric, 0);
                break;
#endif

#if WITH_MUMPS
            case SOLVER_MUMPS:
                /* use MUMPS direct solver on the coarsest level */
                mgl[l].mumps.job = 2;
                fasp_solver_mumps_steps(A0, b0, e0, &mgl[l].mumps);
                break;
#endif

            default:
                /* use iterative solver on the coarsest level */
                fasp_coarse_itsolver(A0, b0, e0, tol, prtlvl);
        }
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * \fn void fasp_solver_namli (AMG_data *mgl, AMG_param *param, INT l, INT num_levels)
 *
 * \brief Solve Ax=b with recursive nonlinear AMLI-cycle
 *
 * \param mgl         Pointer to AMG_data data
 * \param param       Pointer to AMG parameters
 * \param l           Current level
 * \param num_levels  Total number of levels
 *
 * \author Xiaozhe Hu
 * \date   04/06/2010
 *
 * \note Refer to Xiazhe Hu, Panayot S. Vassilevski, Jinchao Xu
 *       "Comparative Convergence Analysis of Nonlinear AMLI-cycle Multigrid", 2013.
 *
 * Modified by Chensong Zhang on 02/27/2013: update direct solvers.
 * Modified by Zheng Li on 11/10/2014: update direct solvers.
 * Modified by Hongxuan Zhang on 12/15/2015: update direct solvers.
 */
void fasp_solver_namli(AMG_data* mgl, AMG_param* param, INT l, INT num_levels)
{
    const SHORT amg_type      = param->AMG_type;
    const SHORT prtlvl        = param->print_level;
    const SHORT smoother      = param->smoother;
    const SHORT smooth_order  = param->smooth_order;
    const SHORT coarse_solver = param->coarse_solver;
    const REAL  relax         = param->relaxation;
    const REAL  tol           = param->tol * 1e-4;
    const SHORT ndeg          = param->polynomial_degree;

    dvector *b0 = &mgl[l].b, *e0 = &mgl[l].x;         // fine level b and x
    dvector *b1 = &mgl[l + 1].b, *e1 = &mgl[l + 1].x; // coarse level b and x

    dCSRmat* A0 = &mgl[l].A;                          // fine level matrix
    dCSRmat* A1 = &mgl[l + 1].A;                      // coarse level matrix

    const INT m0 = A0->row, m1 = A1->row;

    INT*      ordering = mgl[l].cfmark.val; // smoother ordering
    ILU_data* LU_level = &mgl[l].LU;        // fine level ILU decomposition
    REAL*     r        = mgl[l].w.val;      // work array for residual

    dvector uH;                             // for coarse level correction
    uH.row = m1;
    uH.val = mgl[l + 1].w.val + m1;

    // Schwarz parameters
    SWZ_param swzparam;
    if (param->SWZ_levels > 0) swzparam.SWZ_blksolver = param->SWZ_blksolver;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: n=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.nnz);
#endif

    if (prtlvl >= PRINT_MOST)
        printf("Nonlinear AMLI level %d, smoother %d.\n", num_levels, smoother);

    if (l < num_levels - 1) {

        // pre smoothing
        if (l < mgl[l].ILU_levels) {

            fasp_smoother_dcsr_ilu(A0, b0, e0, LU_level);

        }

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

        else {
#if MULTI_COLOR_ORDER
            // printf("fasp_smoother_dcsr_gs_multicolor, %s, %d\n",  __FUNCTION__,
            // __LINE__);
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->presmooth_iter, 1);
#else
            fasp_dcsr_presmoothing(smoother, A0, b0, e0, param->presmooth_iter, 0,
                                   m0 - 1, 1, relax, ndeg, smooth_order, ordering);
#endif
        }

        // form residual r = b - A x
        fasp_darray_cp(m0, b0->val, r);
        fasp_blas_dcsr_aAxpy(-1.0, A0, e0->val, r);

        // restriction r1 = R*r0
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_mxv_agg(&mgl[l].R, r, b1->val);
                break;
            default:
                fasp_blas_dcsr_mxv(&mgl[l].R, r, b1->val);
        }

        // call nonlinear AMLI-cycle recursively
        {
            fasp_dvec_set(m1, e1, 0.0);

            // V-cycle will be enforced when needed !!!
            if (mgl[l + 1].cycle_type <= 1) {

                fasp_solver_namli(&mgl[l + 1], param, 0, num_levels - 1);

            }

            else { // recursively call preconditioned Krylov method on coarse grid

                precond_data pcdata;

                fasp_param_amg_to_prec(&pcdata, param);
                pcdata.maxit      = 1;
                pcdata.max_levels = num_levels - 1;
                pcdata.mgl_data   = &mgl[l + 1];

                precond pc;
                pc.data = &pcdata;
                pc.fct  = fasp_precond_namli;

                fasp_darray_cp(m1, e1->val, uH.val);

                switch (param->nl_amli_krylov_type) {
                    case SOLVER_GCG: // Use GCG
                        Kcycle_dcsr_pgcg(A1, b1, &uH, &pc);
                        break;
                    default: // Use GCR
                        Kcycle_dcsr_pgcr(A1, b1, &uH, &pc);
                }

                fasp_darray_cp(m1, uH.val, e1->val);
            }
        }

        // prolongation e0 = e0 + P*e1
        switch (amg_type) {
            case UA_AMG:
                fasp_blas_dcsr_aAxpy_agg(1.0, &mgl[l].P, e1->val, e0->val);
                break;
            default:
                fasp_blas_dcsr_aAxpy(1.0, &mgl[l].P, e1->val, e0->val);
        }

        // post smoothing
        if (l < mgl[l].ILU_levels) {

            fasp_smoother_dcsr_ilu(A0, b0, e0, LU_level);

        } else if (l < mgl->SWZ_levels) {

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
            }

        }

        else {
#if MULTI_COLOR_ORDER
            fasp_smoother_dcsr_gs_multicolor(&mgl[l].x, &mgl[l].A, &mgl[l].b,
                                             param->postsmooth_iter, -1);
#else
            fasp_dcsr_postsmoothing(smoother, A0, b0, e0, param->postsmooth_iter, 0,
                                    m0 - 1, -1, relax, ndeg, smooth_order, ordering);
#endif
        }

    }

    else { // coarsest level solver

        switch (coarse_solver) {

#if WITH_PARDISO
            case SOLVER_PARDISO:
                {
                    /* use Intel MKL PARDISO direct solver on the coarsest level */
                    fasp_pardiso_solve(A0, b0, e0, &mgl[l].pdata, 0);
                    break;
                }
#endif

#if WITH_SuperLU
            case SOLVER_SUPERLU:
                /* use SuperLU direct solver on the coarsest level */
                fasp_solver_superlu(A0, b0, e0, 0);
                break;
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                /* use UMFPACK direct solver on the coarsest level */
                fasp_umfpack_solve(A0, b0, e0, mgl[l].Numeric, 0);
                break;
#endif

#if WITH_MUMPS
            case SOLVER_MUMPS:
                /* use MUMPS direct solver on the coarsest level */
                mgl[l].mumps.job = 2;
                fasp_solver_mumps_steps(A0, b0, e0, &mgl[l].mumps);
                break;
#endif

            default:
                /* use iterative solver on the coarsest level */
                fasp_coarse_itsolver(A0, b0, e0, tol, prtlvl);
        }
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * \fn void fasp_solver_namli_bsr (AMG_data_bsr *mgl, AMG_param *param,
 *                                 INT l, INT num_levels)
 *
 * \brief Solve Ax=b with recursive nonlinear AMLI-cycle
 *
 * \param mgl         Pointer to AMG data: AMG_data
 * \param param       Pointer to AMG parameters: AMG_param
 * \param l           Current level
 * \param num_levels  Total number of levels
 *
 * \author Xiaozhe Hu
 * \date   04/06/2010
 *
 * \note Nonlinear AMLI-cycle.
 *       Refer to Xiazhe Hu, Panayot S. Vassilevski, Jinchao Xu
 *       "Comparative Convergence Analysis of Nonlinear AMLI-cycle Multigrid", 2013.
 *
 * Modified by Chensong Zhang on 02/27/2013: update direct solvers.
 * Modified by Hongxuan Zhang on 12/15/2015: update direct solvers.
 * Modified by Li Zhao on 05/01/2023: update direct solvers and smoothers.
 */
void fasp_solver_namli_bsr(AMG_data_bsr* mgl, AMG_param* param, INT l, INT num_levels)
{
    const SHORT prtlvl        = param->print_level;
    const SHORT smoother      = param->smoother;
    const SHORT coarse_solver = param->coarse_solver;
    const REAL  relax         = param->relaxation;
    const REAL  tol           = param->tol;
    INT         i;

    dvector *b0 = &mgl[l].b, *e0 = &mgl[l].x;         // fine level b and x
    dvector *b1 = &mgl[l + 1].b, *e1 = &mgl[l + 1].x; // coarse level b and x

    dBSRmat*  A0 = &mgl[l].A;                         // fine level matrix
    dBSRmat*  A1 = &mgl[l + 1].A;                     // coarse level matrix
    const INT m0 = A0->ROW * A0->nb, m1 = A1->ROW * A1->nb;

    ILU_data* LU_level = &mgl[l].LU;   // fine level ILU decomposition
    REAL*     r        = mgl[l].w.val; // for residual

    dvector uH, bH;                    // for coarse level correction
    uH.row = m1;
    uH.val = mgl[l + 1].w.val + m1;
    bH.row = m1;
    bH.val = mgl[l + 1].w.val + 2 * m1;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: n=%d, nnz=%d\n", mgl[0].A.ROW, mgl[0].A.NNZ);
    // printf("### DEBUG: prtlvl=%d\n", prtlvl);
    // exit(0);
#endif

    // REAL start_time, end_time; //! zhaoli

    if (prtlvl >= PRINT_MOST)
        printf("Nonlinear AMLI: level %d, smoother %d.\n", l, smoother);

    if (l < num_levels - 1) {

        // fasp_gettime(&start_time); //! zhaoli

        // pre smoothing
        if (l < param->ILU_levels) {
            fasp_smoother_dbsr_ilu(A0, b0, e0, LU_level);
        } else {
            SHORT steps = param->presmooth_iter;

            if (steps > 0) {
                switch (smoother) {
                    case SMOOTHER_JACOBI:
                        for (i = 0; i < steps; i++)
                            //! 该函数内部生成了对角块的逆, zhaoli, 2023.05.01
                            // fasp_smoother_dbsr_jacobi (A0, b0, e0);
                            //! 该函数直接使用对角块的逆(在setup阶段生成), zhaoli,
                            //! 2023.05.01
                            fasp_smoother_dbsr_jacobi1(A0, b0, e0, mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_GS:
                        if (l == 0) {
                            for (i = 0; i < steps; i++)
#if BAMG_GS0_DiagInv || 1
                                fasp_smoother_dbsr_gs1(A0, b0, e0, ASCEND, NULL,
                                                       mgl[l].diaginv.val);
#else
                                fasp_smoother_dbsr_gs_ascend1(A0, b0, e0);
#endif
                        } else {
                            for (i = 0; i < steps; i++)
                                fasp_smoother_dbsr_gs1(A0, b0, e0, ASCEND, NULL,
                                                       mgl[l].diaginv.val);
                        }

                        break;
                    case SMOOTHER_SOR:
                        for (i = 0; i < steps; i++)
                            // fasp_smoother_dbsr_sor(A0, b0, e0, ASCEND, NULL, relax);
                            fasp_smoother_dbsr_sor1(A0, b0, e0, ASCEND, NULL,
                                                    mgl[l].diaginv.val, relax);
                        break;
                    default:
                        printf("### ERROR: Unknown smoother type %d!\n", smoother);
                        fasp_chkerr(ERROR_SOLVER_TYPE, __FUNCTION__);
                }
            }
        }

        // fasp_gettime(&end_time);                      //! zhaoli
        // PreSmoother_time_zl += end_time - start_time; //! zhaoli

        // form residual r = b - A x
        fasp_darray_cp(m0, b0->val, r);
        fasp_blas_dbsr_aAxpy(-1.0, A0, e0->val, r);

        fasp_blas_dbsr_mxv(&mgl[l].R, r, b1->val);

        // call nonlinear AMLI-cycle recursively
        {
            fasp_dvec_set(m1, e1, 0.0);

            // The coarsest problem is solved exactly.
            // No need to call Krylov method on second coarsest level
            if (l == num_levels - 2) {
                fasp_solver_namli_bsr(&mgl[l + 1], param, 0, num_levels - 1);
            } else { // recursively call preconditioned Krylov method on coarse grid
                precond_data_bsr pcdata;

                fasp_param_amg_to_precbsr(&pcdata, param);
                pcdata.maxit      = 1;
                pcdata.max_levels = num_levels - 1;
                pcdata.mgl_data   = &mgl[l + 1];

                precond pc;
                pc.data = &pcdata;
                pc.fct  = fasp_precond_dbsr_namli;

                fasp_darray_cp(m1, b1->val, bH.val);
                fasp_darray_cp(m1, e1->val, uH.val);

                const INT maxit = param->amli_degree + 1;

                // fasp_gettime(&start_time); //! zhaoli

                // fasp_solver_dbsr_pcg(A1, &bH, &uH, &pc, param->tol, param->tol *
                // 1e-8,
                //                      maxit, 1, PRINT_NONE);

                // printf("tol = %e\n", param->tol);
                // exit(0);
                // fasp_solver_dbsr_pbcgs(A1, &bH, &uH, &pc, param->tol, param->tol *
                // 1e-8,
                //                        maxit, 1, PRINT_NONE);

                fasp_solver_dbsr_pvfgmres(A1, &bH, &uH, &pc, param->tol,
                                          param->tol * 1e-8, maxit, MIN(maxit, 30), 1,
                                          PRINT_NONE);

                // fasp_gettime(&end_time);                 //! zhaoli
                // Krylov_time_zl += end_time - start_time; //! zhaoli

                fasp_darray_cp(m1, bH.val, b1->val);
                fasp_darray_cp(m1, uH.val, e1->val);
            }
        }

        fasp_blas_dbsr_aAxpy(1.0, &mgl[l].P, e1->val, e0->val);

        // fasp_gettime(&start_time); //! zhaoli

        // post smoothing
        if (l < param->ILU_levels) {
            fasp_smoother_dbsr_ilu(A0, b0, e0, LU_level);
        } else {
            SHORT steps = param->postsmooth_iter;

            if (steps > 0) {
                switch (smoother) {
                    case SMOOTHER_JACOBI:
                        for (i = 0; i < steps; i++)
                            //! 该函数内部生成了对角块的逆, zhaoli, 2023.05.01
                            // fasp_smoother_dbsr_jacobi(A0, b0, e0);
                            //! 该函数直接使用对角块的逆(在setup阶段生成), zhaoli,
                            //! 2023.05.01
                            fasp_smoother_dbsr_jacobi1(A0, b0, e0, mgl[l].diaginv.val);
                        break;
                    case SMOOTHER_GS:
                        // fasp_smoother_dbsr_gs(A0, b0, e0, ASCEND, NULL);
                        if (l == 0) {
                            for (i = 0; i < steps; i++)
#if BAMG_GS0_DiagInv || 1
                                fasp_smoother_dbsr_gs1(A0, b0, e0, DESCEND, NULL,
                                                       mgl[l].diaginv.val);
#else
                                fasp_smoother_dbsr_gs_descend1(A0, b0, e0);
#endif
                        } else {
                            for (i = 0; i < steps; i++)
                                fasp_smoother_dbsr_gs1(A0, b0, e0, DESCEND, NULL,
                                                       mgl[l].diaginv.val);
                        }

                        break;
                    case SMOOTHER_SOR:
                        for (i = 0; i < steps; i++)
                            // fasp_smoother_dbsr_sor(A0, b0, e0, ASCEND, NULL, relax);
                            fasp_smoother_dbsr_sor1(A0, b0, e0, DESCEND, NULL,
                                                    mgl[l].diaginv.val, relax);
                        break;
                    default:
                        printf("### ERROR: Unknown smoother type %d!\n", smoother);
                        fasp_chkerr(ERROR_SOLVER_TYPE, __FUNCTION__);
                }
            }
        }

        // fasp_gettime(&end_time);                       //! zhaoli
        // PostSmoother_time_zl += end_time - start_time; //! zhaoli

    }

    else { // coarsest level solver
        // fasp_gettime(&start_time); //! zhaoli

        switch (coarse_solver) {

#if WITH_PARDISO
            case SOLVER_PARDISO:
                {
                    /* use Intel MKL PARDISO direct solver on the coarsest level */
                    fasp_pardiso_solve(&mgl[l].Ac, b0, e0, &mgl[l].pdata, 0);
                    break;
                }
#endif

#if WITH_SuperLU
            case SOLVER_SUPERLU:
                /* use SuperLU direct solver on the coarsest level */
                fasp_solver_superlu(&mgl[l].Ac, b0, e0, 0);
                break;
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                /* use UMFPACK direct solver on the coarsest level */
                fasp_umfpack_solve(&mgl[l].Ac, b0, e0, mgl[l].Numeric, 0);
                break;
#endif

#if WITH_MUMPS
            case SOLVER_MUMPS:
                /* use MUMPS direct solver on the coarsest level */
                mgl[l].mumps.job = 2;
                fasp_solver_mumps_steps(&mgl[l].Ac, b0, e0, &mgl[l].mumps);
                break;
#endif

            default:
                /* use iterative solver on the coarsest level */
                fasp_coarse_itsolver(&mgl[l].Ac, b0, e0, tol, prtlvl);
        }

        // fasp_gettime(&end_time);                  //! zhaoli
        // Coarsen_time_zl += end_time - start_time; //! zhaoli
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * \fn void fasp_amg_amli_coef (const REAL lambda_max, const REAL lambda_min,
 *                              const INT degree, REAL *coef)
 *
 * \brief Compute the coefficients of the polynomial used by AMLI-cycle
 *
 * \param lambda_max  Maximal lambda
 * \param lambda_min  Minimal lambda
 * \param degree      Degree of polynomial approximation
 * \param coef        Coefficient of AMLI (output)
 *
 * \author Xiaozhe Hu
 * \date   01/23/2011
 */
void fasp_amg_amli_coef(const REAL lambda_max,
                        const REAL lambda_min,
                        const INT  degree,
                        REAL*      coef)
{
    const REAL mu0 = 1.0 / lambda_max, mu1 = 1.0 / lambda_min;
    const REAL c = (sqrt(mu0) + sqrt(mu1)) * (sqrt(mu0) + sqrt(mu1));
    const REAL a = (4 * mu0 * mu1) / (c);

    const REAL kappa = lambda_max / lambda_min; // condition number
    const REAL delta = (sqrt(kappa) - 1.0) / (sqrt(kappa) + 1.0);
    const REAL b     = delta * delta;

    if (degree == 0) {
        coef[0] = 0.5 * (mu0 + mu1);
    }

    else if (degree == 1) {
        coef[0] = 0.5 * c;
        coef[1] = -1.0 * mu0 * mu1;
    }

    else if (degree > 1) {
        INT i;

        // allocate memory
        REAL* work = (REAL*)fasp_mem_calloc(2 * degree - 1, sizeof(REAL));
        REAL *coef_k, *coef_km1;
        coef_k   = work;
        coef_km1 = work + degree;

        // get q_k
        fasp_amg_amli_coef(lambda_max, lambda_min, degree - 1, coef_k);
        // get q_km1
        fasp_amg_amli_coef(lambda_max, lambda_min, degree - 2, coef_km1);

        // get coef
        coef[0] = a - b * coef_km1[0] + (1 + b) * coef_k[0];

        for (i = 1; i < degree - 1; i++) {
            coef[i] = -b * coef_km1[i] + (1 + b) * coef_k[i] - a * coef_k[i - 1];
        }

        coef[degree - 1] = (1 + b) * coef_k[degree - 1] - a * coef_k[degree - 2];

        coef[degree] = -a * coef_k[degree - 1];

        // clean memory
        fasp_mem_free(work);
        work = NULL;
    }

    else {
        printf("### ERROR: Wrong AMLI degree %d!\n", degree);
        fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
    }

    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
