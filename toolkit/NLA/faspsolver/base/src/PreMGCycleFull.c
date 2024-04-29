/*! \file  PreMGCycleFull.c
 *
 *  \brief Abstract non-recursive full multigrid cycle
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMessage.c, AuxVector.c, BlaSchwarzSetup.c, BlaArray.c,
 *         BlaSpmvCSR.c, BlaVector.c, ItrSmootherCSR.c, ItrSmootherCSRpoly.c, 
 *         KryPcg.c, KrySPcg.c, and KrySPvgmres.c
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

#include "PreMGUtil.inl"
#include "PreMGSmoother.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_solver_fmgcycle (AMG_data *mgl, AMG_param *param)
 *
 * \brief Solve Ax=b with non-recursive full multigrid K-cycle
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \author Chensong Zhang
 * \date   02/27/2011
 *
 * Modified by Chensong Zhang on 06/01/2012: fix a bug when there is only one level.
 * Modified by Hongxuan Zhang on 12/15/2015: update direct solvers.
 */
void fasp_solver_fmgcycle (AMG_data   *mgl,
                           AMG_param  *param)
{
    const SHORT  maxit = 3; // Max allowed V-cycles in each level
    const SHORT  amg_type = param->AMG_type;
    const SHORT  prtlvl = param->print_level;
    const SHORT  nl = mgl[0].num_levels;
    const SHORT  smoother = param->smoother;
    const SHORT  smooth_order = param->smooth_order;
    const SHORT  coarse_solver = param->coarse_solver;

    const REAL   relax = param->relaxation;
    const SHORT  ndeg = param->polynomial_degree;
    const REAL   tol = param->tol*1e-4;

    // local variables
    INT l, i, lvl, num_cycle;
    REAL alpha = 1.0, relerr;

    // Schwarz parameters
    SWZ_param swzparam;
    if ( param->SWZ_levels > 0 ) {
        swzparam.SWZ_blksolver = param->SWZ_blksolver;
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: n=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.nnz);
#endif

    if ( prtlvl >= PRINT_MOST )
        printf("FMG_level = %d, ILU_level = %d\n", nl, param->ILU_levels);

    // restriction r1 = R*r0
    switch (amg_type) {

        case UA_AMG:
            for (l=0;l<nl-1;l++)
                fasp_blas_dcsr_mxv_agg(&mgl[l].R, mgl[l].b.val, mgl[l+1].b.val);
            break;

        default:
            for (l=0;l<nl-1;l++)
                fasp_blas_dcsr_mxv(&mgl[l].R, mgl[l].b.val, mgl[l+1].b.val);
            break;

    }

    fasp_dvec_set(mgl[l].A.row, &mgl[l].x, 0.0); // initial guess

    // If only one level, just direct solver
    if ( nl==1 ) {

        switch (coarse_solver) {

#if WITH_PARDISO
            case SOLVER_PARDISO: {
                /* use Intel MKL PARDISO direct solver on the coarsest level */
                fasp_pardiso_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].pdata, 0);
                break;
            }
#endif

#if WITH_SuperLU
            case SOLVER_SUPERLU:
                /* use SuperLU direct solver on the coarsest level */
                fasp_solver_superlu(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, 0);
                break;
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                /* use UMFPACK direct solver on the coarsest level */
                fasp_umfpack_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, mgl[nl-1].Numeric, 0);
                break;
#endif

#if WITH_MUMPS
            case SOLVER_MUMPS:
                /* use MUMPS direct solver on the coarsest level */
                mgl[nl-1].mumps.job = 2;
                fasp_solver_mumps_steps(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].mumps);
                break;
#endif

            default:
                /* use iterative solver on the coarest level */
                fasp_coarse_itsolver(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, tol, prtlvl);

        }

        return;

    }

    for ( i=1; i<nl; i++ ) {

        // Coarse Space Solver:
        switch (coarse_solver) {

#if WITH_PARDISO
            case SOLVER_PARDISO: {
                /* use Intel MKL PARDISO direct solver on the coarsest level */
                fasp_pardiso_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].pdata, 0);
                break;
            }
#endif

#if WITH_SuperLU
            case SOLVER_SUPERLU:
                /* use SuperLU direct solver on the coarsest level */
                fasp_solver_superlu(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, 0);
                break;
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                /* use UMFPACK direct solver on the coarsest level */
                fasp_umfpack_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, mgl[nl-1].Numeric, 0);
                break;
#endif

#if WITH_MUMPS
            case SOLVER_MUMPS:
                /* use MUMPS direct solver on the coarsest level */
                mgl[nl-1].mumps.job = 2;
                fasp_solver_mumps_steps(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].mumps);
                break;
#endif

            default:
                /* use iterative solver on the coarest level */
                fasp_coarse_itsolver(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, tol, prtlvl);

        }

        // Slash part: /-cycle
        {
            --l; // go back to finer level

            // find the optimal scaling factor alpha
            if ( param->coarse_scaling == ON ) {
                alpha = fasp_blas_darray_dotprod(mgl[l+1].A.row, mgl[l+1].x.val, mgl[l+1].b.val)
                      / fasp_blas_dcsr_vmv(&mgl[l+1].A, mgl[l+1].x.val, mgl[l+1].x.val);
                alpha = MIN(alpha, 1.0); // Add this for safty! --Chensong on 10/04/2014
            }

            // prolongation u = u + alpha*P*e1
            switch (amg_type) {
                case UA_AMG:
                    fasp_blas_dcsr_aAxpy_agg(alpha, &mgl[l].P, mgl[l+1].x.val, mgl[l].x.val); break;
                default:
                    fasp_blas_dcsr_aAxpy(alpha, &mgl[l].P, mgl[l+1].x.val, mgl[l].x.val); break;
            }
        }

        // initialzie rel error
        num_cycle = 0; relerr = BIGREAL;

        while ( relerr > param->tol && num_cycle < maxit) {

            ++num_cycle;

            // form residual r = b - A x
            fasp_darray_cp(mgl[l].A.row, mgl[l].b.val, mgl[l].w.val);
            fasp_blas_dcsr_aAxpy(-1.0,&mgl[l].A, mgl[l].x.val, mgl[l].w.val);
            relerr = fasp_blas_dvec_norm2(&mgl[l].w) / fasp_blas_dvec_norm2(&mgl[l].b);

            // Forward Sweep
            for ( lvl=0; lvl<i; lvl++ ) {

                // pre smoothing
                if (l<param->ILU_levels) {
                    fasp_smoother_dcsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
                }
                else if (l<mgl->SWZ_levels) {
                    switch (mgl[l].Schwarz.SWZ_type) {
                        case SCHWARZ_SYMMETRIC:
                            fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x, &mgl[l].b);
                            fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam,&mgl[l].x, &mgl[l].b);
                            break;
                        default:
                            fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x, &mgl[l].b);
                            break;
                    }
                }

                else {
                    fasp_dcsr_presmoothing(smoother,&mgl[l].A,&mgl[l].b,&mgl[l].x,param->presmooth_iter,
                                           0,mgl[l].A.row-1,1,relax,ndeg,smooth_order,mgl[l].cfmark.val);
                }

                // form residual r = b - A x
                fasp_darray_cp(mgl[l].A.row, mgl[l].b.val, mgl[l].w.val);
                fasp_blas_dcsr_aAxpy(-1.0,&mgl[l].A, mgl[l].x.val, mgl[l].w.val);

                // restriction r1 = R*r0
                switch (amg_type) {
                    case UA_AMG:
                        fasp_blas_dcsr_mxv_agg(&mgl[l].R, mgl[l].w.val, mgl[l+1].b.val);
                        break;
                    default:
                        fasp_blas_dcsr_mxv(&mgl[l].R, mgl[l].w.val, mgl[l+1].b.val);
                        break;
                }

                ++l;

                // prepare for the next level
                fasp_dvec_set(mgl[l].A.row, &mgl[l].x, 0.0);

            }    // end for lvl

            // CoarseSpaceSolver:
            switch (coarse_solver) {

#if WITH_PARDISO
                case SOLVER_PARDISO: {
                    /* use Intel MKL PARDISO direct solver on the coarsest level */
                    fasp_pardiso_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].pdata, 0);
                    break;
                }
#endif

#if WITH_SuperLU
                case SOLVER_SUPERLU:
                    /* use SuperLU direct solver on the coarsest level */
                    fasp_solver_superlu(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, 0);
                    break;
#endif

#if WITH_UMFPACK
                case SOLVER_UMFPACK:
                    /* use UMFPACK direct solver on the coarsest level */
                    fasp_umfpack_solve(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, mgl[nl-1].Numeric, 0);
                    break;
#endif

#if WITH_MUMPS
                case SOLVER_MUMPS:
                    /* use MUMPS direct solver on the coarsest level */
                    mgl[nl-1].mumps.job = 2;
                    fasp_solver_mumps_steps(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, &mgl[nl-1].mumps);
                    break;
#endif

                default:
                    /* use iterative solver on the coarest level */
                    fasp_coarse_itsolver(&mgl[nl-1].A, &mgl[nl-1].b, &mgl[nl-1].x, tol, prtlvl);

            }

            // Backward Sweep
            for ( lvl=0; lvl<i; lvl++ ) {

                --l;

                // find the optimal scaling factor alpha
                if ( param->coarse_scaling == ON ) {
                    alpha = fasp_blas_darray_dotprod(mgl[l+1].A.row, mgl[l+1].x.val, mgl[l+1].b.val)
                          / fasp_blas_dcsr_vmv(&mgl[l+1].A, mgl[l+1].x.val, mgl[l+1].x.val);
                    alpha = MIN(alpha, 1.0); // Add this for safty! --Chensong on 10/04/2014
                }

                // prolongation u = u + alpha*P*e1
                switch (amg_type)
                {
                    case UA_AMG:
                        fasp_blas_dcsr_aAxpy_agg(alpha, &mgl[l].P, mgl[l+1].x.val, mgl[l].x.val);
                        break;
                    default:
                        fasp_blas_dcsr_aAxpy(alpha, &mgl[l].P, mgl[l+1].x.val, mgl[l].x.val);
                        break;
                }

                // post-smoothing
                if (l<param->ILU_levels) {
                    fasp_smoother_dcsr_ilu(&mgl[l].A, &mgl[l].b, &mgl[l].x, &mgl[l].LU);
                }
                else if (l<mgl->SWZ_levels) {
                    switch (mgl[l].Schwarz.SWZ_type) {
                        case SCHWARZ_SYMMETRIC:
                            fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam,&mgl[l].x, &mgl[l].b);
                            fasp_dcsr_swz_forward(&mgl[l].Schwarz, &swzparam, &mgl[l].x, &mgl[l].b);
                            break;
                        default:
                            fasp_dcsr_swz_backward(&mgl[l].Schwarz, &swzparam,&mgl[l].x, &mgl[l].b);
                            break;
                    }
                }

                else {
                    fasp_dcsr_postsmoothing(smoother,&mgl[l].A,&mgl[l].b,&mgl[l].x,param->postsmooth_iter,
                                            0,mgl[l].A.row-1,-1,relax,ndeg,smooth_order,mgl[l].cfmark.val);
                }

            } // end while

        } //end while

    } // end for

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
