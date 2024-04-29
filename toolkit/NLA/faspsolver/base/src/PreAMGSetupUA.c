/*! \file  PreAMGSetupUA.c
 *
 *  \brief Unsmoothed aggregation AMG: SETUP phase
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxTiming.c, AuxVector.c,
 *         BlaILUSetupCSR.c, BlaSchwarzSetup.c, BlaSparseCSR.c, BlaSpmvCSR.c,
 *         and PreMGRecurAMLI.c
 *
 *  \note  Setup A, P, PT and levels using the unsmoothed aggregation algorithm
 *
 *  Reference:
 *         A. Napov and Y. Notay
 *         An Algebraic Multigrid Method with Guaranteed Convergence Rate, 2012
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2011--Present by the FASP team. All rights reserved.
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

#include "PreAMGAggregation.inl"
#include "PreAMGAggregationCSR.inl"
#include "PreAMGAggregationUA.inl"

static SHORT amg_setup_unsmoothP_unsmoothR(AMG_data*, AMG_param*);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_amg_setup_ua (AMG_data *mgl, AMG_param *param)
 *
 * \brief Set up phase of unsmoothed aggregation AMG
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Xiaozhe Hu
 * \date   12/28/2011
 */
SHORT fasp_amg_setup_ua(AMG_data* mgl, AMG_param* param)
{
    const SHORT prtlvl = param->print_level;

    // Output some info for debuging
    if (prtlvl > PRINT_NONE) printf("\nSetting up UA AMG ...\n");

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.col,
           mgl[0].A.nnz);
#endif

    SHORT status = amg_setup_unsmoothP_unsmoothR(mgl, param);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static SHORT amg_setup_unsmoothP_unsmoothR (AMG_data *mgl, AMG_param *param)
 *
 * \brief Setup phase of plain aggregation AMG, using unsmoothed P and unsmoothed A
 *
 * \param mgl    Pointer to AMG_data
 * \param param  Pointer to AMG_param
 *
 * \return       FASP_SUCCESS if succeed, error otherwise
 *
 * \author Xiaozhe Hu
 * \date   02/21/2011
 *
 * Modified by Chensong Zhang on 05/10/2013: adjust the structure.
 * Modified by Chensong Zhang on 09/23/2014: check coarse spaces.
 * Modified by Zheng Li on 01/13/2015: adjust coarsening stop criterion.
 * Modified by Zheng Li on 03/22/2015: adjust coarsening ratio.
 * Modified by Chunsheng Feng on 10/17/2020: if NPAIR fail auto switch aggregation type
 * to VBM.
 */
static SHORT amg_setup_unsmoothP_unsmoothR(AMG_data* mgl, AMG_param* param)
{
    const SHORT prtlvl     = param->print_level;
    const SHORT cycle_type = param->cycle_type;
    const SHORT csolver    = param->coarse_solver;
    const SHORT min_cdof   = MAX(param->coarse_dof, 50);
    const INT   m          = mgl[0].A.row;

    // empiric value
    const REAL cplxmax = 3.0;
    const REAL xsi     = 0.6;
    INT        icum    = 1;
    REAL       eta, fracratio;

    // local variables
    SHORT     max_levels = param->max_levels, lvl = 0, status = FASP_SUCCESS;
    INT       i;
    REAL      setup_start, setup_end;
    ILU_param iluparam;
    SWZ_param swzparam;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", mgl[0].A.row, mgl[0].A.col,
           mgl[0].A.nnz);
#endif

    fasp_gettime(&setup_start);

    // level info (fine: 0; coarse: 1)
    ivector* vertices = (ivector*)fasp_mem_calloc(max_levels, sizeof(ivector));

    // each level stores the information of the number of aggregations
    INT* num_aggs = (INT*)fasp_mem_calloc(max_levels, sizeof(INT));

    // each level stores the information of the strongly coupled neighborhoods
    dCSRmat* Neighbor = (dCSRmat*)fasp_mem_calloc(max_levels, sizeof(dCSRmat));

    // Initialize level information
    for (i = 0; i < max_levels; ++i) num_aggs[i] = 0;

    mgl[0].near_kernel_dim = 1;
    mgl[0].near_kernel_basis =
        (REAL**)fasp_mem_calloc(mgl->near_kernel_dim, sizeof(REAL*));

    for (i = 0; i < mgl->near_kernel_dim; ++i) {
        mgl[0].near_kernel_basis[i] = (REAL*)fasp_mem_calloc(m, sizeof(REAL));
        fasp_darray_set(m, mgl[0].near_kernel_basis[i], 1.0);
    }

    // Initialize ILU parameters
    mgl->ILU_levels = param->ILU_levels;
    if (param->ILU_levels > 0) {
        iluparam.print_level = param->print_level;
        iluparam.ILU_lfil    = param->ILU_lfil;
        iluparam.ILU_droptol = param->ILU_droptol;
        iluparam.ILU_relax   = param->ILU_relax;
        iluparam.ILU_type    = param->ILU_type;
    }

    // Initialize Schwarz parameters
    mgl->SWZ_levels = param->SWZ_levels;
    if (param->SWZ_levels > 0) {
        swzparam.SWZ_mmsize    = param->SWZ_mmsize;
        swzparam.SWZ_maxlvl    = param->SWZ_maxlvl;
        swzparam.SWZ_type      = param->SWZ_type;
        swzparam.SWZ_blksolver = param->SWZ_blksolver;
    }

    // Initialize AMLI coefficients
    if (cycle_type == AMLI_CYCLE) {
        const INT amlideg = param->amli_degree;
        param->amli_coef  = (REAL*)fasp_mem_calloc(amlideg + 1, sizeof(REAL));
        REAL lambda_max = 2.0, lambda_min = lambda_max / 4;
        fasp_amg_amli_coef(lambda_max, lambda_min, amlideg, param->amli_coef);
    }

#if DIAGONAL_PREF
    fasp_dcsr_diagpref(&mgl[0].A); // reorder each row to make diagonal appear first
#endif

    /*----------------------------*/
    /*--- checking aggregation ---*/
    /*----------------------------*/

    // Pairwise matching algorithm requires diagonal preference ordering
    if (param->aggregation_type == PAIRWISE) {
        param->pair_number = MIN(param->pair_number, max_levels);
        fasp_dcsr_diagpref(&mgl[0].A);
    }

    // Main AMG setup loop
    while ((mgl[lvl].A.row > min_cdof) && (lvl < max_levels - 1)) {

#if DEBUG_MODE > 1
        printf("### DEBUG: level = %d, row = %d, nnz = %d\n", lvl, mgl[lvl].A.row,
               mgl[lvl].A.nnz);
#endif

        /*-- Setup ILU decomposition if necessary */
        if (lvl < param->ILU_levels) {
            status = fasp_ilu_dcsr_setup(&mgl[lvl].A, &mgl[lvl].LU, &iluparam);
            if (status < 0) {
                if (prtlvl > PRINT_MIN) {
                    printf("### WARNING: ILU setup on level-%d failed!\n", lvl);
                    printf("### WARNING: Disable ILU for level >= %d.\n", lvl);
                }
                param->ILU_levels = lvl;
            }
        }

        /*-- Setup Schwarz smoother if necessary */
        if (lvl < param->SWZ_levels) {
            mgl[lvl].Schwarz.A = fasp_dcsr_sympart(&mgl[lvl].A);
            fasp_dcsr_shift(&(mgl[lvl].Schwarz.A), 1);
            status = fasp_swz_dcsr_setup(&mgl[lvl].Schwarz, &swzparam);
            if (status < 0) {
                if (prtlvl > PRINT_MIN) {
                    printf("### WARNING: Schwarz on level-%d failed!\n", lvl);
                    printf("### WARNING: Disable Schwarz for level >= %d.\n", lvl);
                }
                param->SWZ_levels = lvl;
            }
        }

        /*-- Aggregation --*/
        switch (param->aggregation_type) {

            case VMB: // VMB aggregation

                status = aggregation_vmb(&mgl[lvl].A, &vertices[lvl], param, lvl + 1,
                                         &Neighbor[lvl], &num_aggs[lvl]);

                /*-- Choose strength threshold adaptively --*/
                if (num_aggs[lvl] * 4.0 > mgl[lvl].A.row)
                    param->strong_coupled /= 2.0;
                else if (num_aggs[lvl] * 1.25 < mgl[lvl].A.row)
                    param->strong_coupled *= 2.0;

                break;

            case NPAIR: // non-symmetric pairwise matching aggregation
                status =
                    aggregation_nsympair(mgl, param, lvl, vertices, &num_aggs[lvl]);
                /*-- Modified by Chunsheng Feng on 10/17/2020, ZCS on 01/15/2021:
                     if NPAIR fail, switch aggregation type to VBM. --*/
                if (status != FASP_SUCCESS || num_aggs[lvl] * 2.0 > mgl[lvl].A.row) {
                    if (prtlvl > PRINT_MORE) {
                        printf("### WARNING: Non-symmetric pairwise matching failed on "
                               "level %d!\n",
                               lvl);
                        printf("### WARNING: Switch to VMB aggregation on level %d!\n",
                               lvl);
                    }
                    param->aggregation_type = VMB;
                    status = aggregation_vmb(&mgl[lvl].A, &vertices[lvl], param,
                                             lvl + 1, &Neighbor[lvl], &num_aggs[lvl]);
                }

                break;

            default: // symmetric pairwise matching aggregation
                status =
                    aggregation_symmpair(mgl, param, lvl, vertices, &num_aggs[lvl]);
        }

        // Check 1: Did coarsening step succeed?
        if (status < 0) {
            // When error happens, stop at the current multigrid level!
            if (prtlvl > PRINT_MIN) {
                printf("### WARNING: Stop coarsening on level %d!\n", lvl);
            }
            status = FASP_SUCCESS;
            fasp_ivec_free(&vertices[lvl]);
            fasp_dcsr_free(&Neighbor[lvl]);
            break;
        }

        /*-- Form Prolongation --*/
        form_tentative_p(&vertices[lvl], &mgl[lvl].P, mgl[0].near_kernel_basis,
                         num_aggs[lvl]);

        // Check 2: Is coarse sparse too small?
        if (mgl[lvl].P.col < MIN_CDOF) {
            fasp_ivec_free(&vertices[lvl]);
            fasp_dcsr_free(&Neighbor[lvl]);
            break;
        }

        // Check 3: Does this coarsening step too aggressive?
        if (mgl[lvl].P.row > mgl[lvl].P.col * MAX_CRATE) {
            if (prtlvl > PRINT_MIN) {
                printf("### WARNING: Coarsening might be too aggressive!\n");
                printf("### WARNING: Fine level = %d, coarse level = %d. Discard!\n",
                       mgl[lvl].P.row, mgl[lvl].P.col);
            }
            fasp_ivec_free(&vertices[lvl]);
            fasp_dcsr_free(&Neighbor[lvl]);
            break;
        }

        /*-- Form restriction --*/
        fasp_dcsr_trans(&mgl[lvl].P, &mgl[lvl].R);

        /*-- Form coarse level stiffness matrix --*/
        fasp_blas_dcsr_rap_agg(&mgl[lvl].R, &mgl[lvl].A, &mgl[lvl].P, &mgl[lvl + 1].A);

        fasp_dcsr_free(&Neighbor[lvl]);
        fasp_ivec_free(&vertices[lvl]);

        ++lvl;

#if DIAGONAL_PREF
        fasp_dcsr_diagpref(
            &mgl[lvl].A); // reorder each row to make diagonal appear first
#endif

        // Check 4: Is this coarsening ratio too small?
        if ((REAL)mgl[lvl].P.col > mgl[lvl].P.row * MIN_CRATE) {
            param->quality_bound *= 2.0;
        }

    } // end of the main while loop

    // Setup coarse level systems for direct solvers
    switch (csolver) {

#if WITH_MUMPS
        case SOLVER_MUMPS:
            {
                // Setup MUMPS direct solver on the coarsest level
                mgl[lvl].mumps.job = 1;
                fasp_solver_mumps_steps(&mgl[lvl].A, &mgl[lvl].b, &mgl[lvl].x,
                                        &mgl[lvl].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        case SOLVER_UMFPACK:
            {
                // Need to sort the matrix A for UMFPACK to work
                dCSRmat Ac_tran;
                Ac_tran =
                    fasp_dcsr_create(mgl[lvl].A.row, mgl[lvl].A.col, mgl[lvl].A.nnz);
                fasp_dcsr_transz(&mgl[lvl].A, NULL, &Ac_tran);
                // It is equivalent to do transpose and then sort
                //     fasp_dcsr_trans(&mgl[lvl].A, &Ac_tran);
                //     fasp_dcsr_sort(&Ac_tran);
                fasp_dcsr_cp(&Ac_tran, &mgl[lvl].A);
                fasp_dcsr_free(&Ac_tran);
                mgl[lvl].Numeric = fasp_umfpack_factorize(&mgl[lvl].A, 0);
                break;
            }
#endif

#if WITH_PARDISO
        case SOLVER_PARDISO:
            {
                fasp_dcsr_sort(&mgl[lvl].A);
                fasp_pardiso_factorize(&mgl[lvl].A, &mgl[lvl].pdata, prtlvl);
                break;
            }
#endif

        default: // Do nothing!
            break;
    }

    // setup total level number and current level
    mgl[0].num_levels = max_levels = lvl + 1;
    mgl[0].w                       = fasp_dvec_create(m);

    for (lvl = 1; lvl < max_levels; ++lvl) {
        INT mm              = mgl[lvl].A.row;
        mgl[lvl].num_levels = max_levels;
        mgl[lvl].b          = fasp_dvec_create(mm);
        mgl[lvl].x          = fasp_dvec_create(mm);

        mgl[lvl].cycle_type = cycle_type;              // initialize cycle type!
        mgl[lvl].ILU_levels = param->ILU_levels - lvl; // initialize ILU levels!
        mgl[lvl].SWZ_levels = param->SWZ_levels - lvl; // initialize Schwarz!

        if (cycle_type == NL_AMLI_CYCLE)
            mgl[lvl].w = fasp_dvec_create(3 * mm);
        else
            mgl[lvl].w = fasp_dvec_create(2 * mm);
    }

    // setup for cycle type of unsmoothed aggregation
    eta                            = xsi / ((1 - xsi) * (cplxmax - 1));
    mgl[0].cycle_type              = 1;
    mgl[max_levels - 1].cycle_type = 0;

    for (lvl = 1; lvl < max_levels - 1; ++lvl) {
        fracratio = (REAL)mgl[lvl].A.nnz / mgl[0].A.nnz;
        mgl[lvl].cycle_type =
            (INT)(pow((REAL)xsi, (REAL)lvl) / (eta * fracratio * icum));
        // safe-guard step: make cycle type >= 1 and <= 2
        mgl[lvl].cycle_type = MAX(1, MIN(2, mgl[lvl].cycle_type));
        icum                = icum * mgl[lvl].cycle_type;
    }

#if MULTI_COLOR_ORDER
    INT Colors, rowmax;
#ifdef _OPENMP
    int threads = fasp_get_num_threads();
    if (threads > max_levels - 1) threads = max_levels - 1;
#pragma omp parallel for private(lvl, rowmax, Colors) schedule(static, 1)              \
    num_threads(threads)
#endif
    for (lvl = 0; lvl < max_levels - 1; lvl++) {

#if 1
        dCSRmat_Multicoloring(&mgl[lvl].A, &rowmax, &Colors);
#else
        dCSRmat_Multicoloring_Theta(&mgl[lvl].A, mgl[lvl].GS_Theta, &rowmax, &Colors);
#endif
        if (prtlvl > 1)
            printf("mgl[%3d].A.row = %12d, rowmax = %5d, rowavg = %7.2lf, colors = "
                   "%5d, Theta = %le.\n",
                   lvl, mgl[lvl].A.row, rowmax, (double)mgl[lvl].A.nnz / mgl[lvl].A.row,
                   mgl[lvl].A.color, mgl[lvl].GS_Theta);
    }
#endif

    if (prtlvl > PRINT_NONE) {
        fasp_gettime(&setup_end);
        fasp_amgcomplexity(mgl, prtlvl);
        fasp_cputime("Unsmoothed aggregation setup", setup_end - setup_start);
    }

    fasp_mem_free(Neighbor);
    Neighbor = NULL;
    fasp_mem_free(vertices);
    vertices = NULL;
    fasp_mem_free(num_aggs);
    num_aggs = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
