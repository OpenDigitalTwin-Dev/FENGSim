/*! \file  PreAMGSetupUABSR.c
 *
 *  \brief Unsmoothed aggregation AMG: SETUP phase (for BSR matrices)
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxTiming.c, AuxVector.c,
 *         BlaFormat.c, BlaILUSetupBSR.c, BlaSparseBLC.c, BlaSparseBSR.c,
 *         BlaSparseCSR.c, BlaSpmvBSR.c, BlaSpmvCSR.c, and PreDataInit.c
 *
 *  \note  Setup A, P, PT and levels using the unsmoothed aggregation algorithm
 *
 *  Reference:
 *         P. Vanek, J. Madel and M. Brezina
 *         Algebraic Multigrid on Unstructured Meshes, 1994
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
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
#include "PreAMGAggregationBSR.inl"
#include "PreAMGAggregationUA.inl"

static SHORT amg_setup_unsmoothP_unsmoothR_bsr(AMG_data_bsr*, AMG_param*);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_amg_setup_ua_bsr (AMG_data_bsr *mgl, AMG_param *param)
 *
 * \brief Set up phase of unsmoothed aggregation AMG (BSR format)
 *
 * \param mgl    Pointer to AMG data: AMG_data_bsr
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Xiaozhe Hu
 * \date   03/16/2012
 */
SHORT fasp_amg_setup_ua_bsr(AMG_data_bsr* mgl, AMG_param* param)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    SHORT status = amg_setup_unsmoothP_unsmoothR_bsr(mgl, param);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static SHORT amg_setup_unsmoothP_unsmoothR_bsr (AMG_data_bsr *mgl,
 *                                                     AMG_param *param)
 *
 * \brief Set up phase of plain aggregation AMG, using unsmoothed P and unsmoothed A
 *        in BSR format
 *
 * \param mgl    Pointer to AMG data: AMG_data_bsr
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       FASP_SUCCESS if succeed, error otherwise
 *
 * \author Xiaozhe Hu
 * \date   03/16/2012
 *
 * Modified by Chensong Zhang on 05/10/2013: adjust the structure.
 */
static SHORT amg_setup_unsmoothP_unsmoothR_bsr(AMG_data_bsr* mgl, AMG_param* param)
{
    const SHORT CondType = 2; // Condensation method used for AMG

    const SHORT prtlvl   = param->print_level;
    const SHORT csolver  = param->coarse_solver;
    const SHORT min_cdof = MAX(param->coarse_dof, 50);
    const INT   m        = mgl[0].A.ROW;
    const INT   nb       = mgl[0].A.nb;

    SHORT max_levels = param->max_levels;
    SHORT i, lvl = 0, status = FASP_SUCCESS;
    REAL  setup_start, setup_end;

    AMG_data* mgl_csr = fasp_amg_data_create(max_levels);

    dCSRmat temp1, temp2;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", mgl[0].A.ROW, mgl[0].A.COL,
           mgl[0].A.NNZ);
#endif

    fasp_gettime(&setup_start);

    /*-----------------------*/
    /*--local working array--*/
    /*-----------------------*/

    // level info (fine: 0; coarse: 1)
    ivector* vertices = (ivector*)fasp_mem_calloc(max_levels, sizeof(ivector));

    // each elvel stores the information of the number of aggregations
    INT* num_aggs = (INT*)fasp_mem_calloc(max_levels, sizeof(INT));

    // each level stores the information of the strongly coupled neighborhoods
    dCSRmat* Neighbor = (dCSRmat*)fasp_mem_calloc(max_levels, sizeof(dCSRmat));

    for (i = 0; i < max_levels; ++i) num_aggs[i] = 0;

    /*------------------------------------------*/
    /*-- setup null spaces for whole Jacobian --*/
    /*------------------------------------------*/

    /*
     mgl[0].near_kernel_dim   = 1;
     mgl[0].near_kernel_basis = (REAL **)fasp_mem_calloc(mgl->near_kernel_dim,
     sizeof(REAL*));

     for ( i=0; i < mgl->near_kernel_dim; ++i ) mgl[0].near_kernel_basis[i] = NULL;
     */

    /*-----------------------*/
    /*-- setup ILU param   --*/
    /*-----------------------*/

    // initialize ILU parameters
    mgl->ILU_levels = param->ILU_levels;
    ILU_param iluparam;

    if (param->ILU_levels > 0) {
        iluparam.print_level = param->print_level;
        iluparam.ILU_lfil    = param->ILU_lfil;
        iluparam.ILU_droptol = param->ILU_droptol;
        iluparam.ILU_relax   = param->ILU_relax;
        iluparam.ILU_type    = param->ILU_type;
    }

    /*----------------------------*/
    /*--- checking aggregation ---*/
    /*----------------------------*/
    if (param->aggregation_type == PAIRWISE)
        param->pair_number = MIN(param->pair_number, max_levels);

    // Main AMG setup loop
    while ((mgl[lvl].A.ROW > min_cdof) && (lvl < max_levels - 1)) {

        /*-- setup ILU decomposition if necessary */
        if (lvl < param->ILU_levels) {
            status = fasp_ilu_dbsr_setup(&mgl[lvl].A, &mgl[lvl].LU, &iluparam);
            if (status < 0) {
                if (prtlvl > PRINT_MIN) {
                    printf("### WARNING: ILU setup on level-%d failed!\n", lvl);
                    printf("### WARNING: Disable ILU for level >= %d.\n", lvl);
                }
                param->ILU_levels = lvl;
            }
        }

        /*-- get the diagonal inverse --*/
        mgl[lvl].diaginv = fasp_dbsr_getdiaginv(&mgl[lvl].A);

        switch (CondType) {
            case 2:
                mgl[lvl].PP = condenseBSR(&mgl[lvl].A);
                break;
            default:
                mgl[lvl].PP = condenseBSRLinf(&mgl[lvl].A);
                break;
        }

        /*-- Aggregation --*/
        switch (param->aggregation_type) {

            case VMB: // VMB aggregation

                status = aggregation_vmb(&mgl[lvl].PP, &vertices[lvl], param, lvl + 1,
                                         &Neighbor[lvl], &num_aggs[lvl]);

                /*-- Choose strength threshold adaptively --*/
                if (num_aggs[lvl] * 4 > mgl[lvl].PP.row)
                    // param->strong_coupled /= 4.0;
                    param->strong_coupled /= 8.0;
                else if (num_aggs[lvl] * 1.25 < mgl[lvl].PP.row)
                    param->strong_coupled *= 1.5;

                break;

            case NPAIR: // non-symmetric pairwise matching aggregation

                mgl_csr[lvl].A = mgl[lvl].PP;
                status =
                    aggregation_nsympair(mgl_csr, param, lvl, vertices, &num_aggs[lvl]);

                break;

            default: // symmetric pairwise matching aggregation

                mgl_csr[lvl].A = mgl[lvl].PP;
                status =
                    aggregation_symmpair(mgl_csr, param, lvl, vertices, &num_aggs[lvl]);

                // TODO: Need to design better algorithm for pairwise BSR -- Xiaozhe
                // TODO: Unsymmetric pairwise aggregation not finished -- Chensong
                // TODO: Check why this fails for BSR --Chensong

                break;
        }

        if (status < 0) {
            // When error happens, force solver to use the current multigrid levels!
            if (prtlvl > PRINT_MIN) {
                printf("### WARNING: Forming aggregates on level-%d failed!\n", lvl);
            }
            status = FASP_SUCCESS;
            break;
        }

        /* -- Form Prolongation --*/
        if (lvl == 0 && mgl[0].near_kernel_dim > 0) {
            form_tentative_p_bsr1(&vertices[lvl], &mgl[lvl].P, &mgl[0], num_aggs[lvl],
                                  mgl[0].near_kernel_dim, mgl[0].near_kernel_basis);
        } else {
            form_boolean_p_bsr(&vertices[lvl], &mgl[lvl].P, &mgl[0], num_aggs[lvl]);
        }

        /*-- Form resitriction --*/
        fasp_dbsr_trans(&mgl[lvl].P, &mgl[lvl].R);

        /*-- Form coarse level stiffness matrix --*/
        fasp_blas_dbsr_rap(&mgl[lvl].R, &mgl[lvl].A, &mgl[lvl].P, &mgl[lvl + 1].A);

        /* -- Form extra near kernal space if needed --*/
        if (mgl[lvl].A_nk != NULL) {

            mgl[lvl + 1].A_nk = (dCSRmat*)fasp_mem_calloc(1, sizeof(dCSRmat));
            mgl[lvl + 1].P_nk = (dCSRmat*)fasp_mem_calloc(1, sizeof(dCSRmat));
            mgl[lvl + 1].R_nk = (dCSRmat*)fasp_mem_calloc(1, sizeof(dCSRmat));

            temp1 = fasp_format_dbsr_dcsr(&mgl[lvl].R);
            fasp_blas_dcsr_mxm(&temp1, mgl[lvl].P_nk, mgl[lvl + 1].P_nk);
            fasp_dcsr_trans(mgl[lvl + 1].P_nk, mgl[lvl + 1].R_nk);
            temp2 = fasp_format_dbsr_dcsr(&mgl[lvl + 1].A);
            fasp_blas_dcsr_rap(mgl[lvl + 1].R_nk, &temp2, mgl[lvl + 1].P_nk,
                               mgl[lvl + 1].A_nk);
            fasp_dcsr_free(&temp1);
            fasp_dcsr_free(&temp2);
        }

        fasp_dcsr_free(&Neighbor[lvl]);
        fasp_ivec_free(&vertices[lvl]);

        ++lvl;
    }

    // Setup coarse level systems for direct solvers (BSR version)
    switch (csolver) {

#if WITH_MUMPS
        case SOLVER_MUMPS:
            {
                // Setup MUMPS direct solver on the coarsest level
                mgl[lvl].Ac        = fasp_format_dbsr_dcsr(&mgl[lvl].A);
                mgl[lvl].mumps.job = 1;
                fasp_solver_mumps_steps(&mgl[lvl].Ac, &mgl[lvl].b, &mgl[lvl].x,
                                        &mgl[lvl].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        case SOLVER_UMFPACK:
            {
                // Need to sort the matrix A for UMFPACK to work
                mgl[lvl].Ac = fasp_format_dbsr_dcsr(&mgl[lvl].A);
                dCSRmat Ac_tran;
                fasp_dcsr_trans(&mgl[lvl].Ac, &Ac_tran);
                fasp_dcsr_sort(&Ac_tran);
                fasp_dcsr_cp(&Ac_tran, &mgl[lvl].Ac);
                fasp_dcsr_free(&Ac_tran);
                mgl[lvl].Numeric = fasp_umfpack_factorize(&mgl[lvl].Ac, 0);
                break;
            }
#endif

#if WITH_SuperLU
        case SOLVER_SUPERLU:
            {
                /* Setup SuperLU direct solver on the coarsest level */
                mgl[lvl].Ac = fasp_format_dbsr_dcsr(&mgl[lvl].A);
            }
#endif

#if WITH_PARDISO
        case SOLVER_PARDISO:
            {
                mgl[lvl].Ac = fasp_format_dbsr_dcsr(&mgl[lvl].A);
                fasp_dcsr_sort(&mgl[lvl].Ac);
                fasp_pardiso_factorize(&mgl[lvl].Ac, &mgl[lvl].pdata, prtlvl);
                break;
            }
#endif

        default:
            // Do nothing!
            break;
    }

    // setup total level number and current level
    mgl[0].num_levels = max_levels = lvl + 1;
    mgl[0].w                       = fasp_dvec_create(3 * m * nb);

    if (mgl[0].A_nk != NULL) {

#if WITH_UMFPACK
        // Need to sort the matrix A_nk for UMFPACK
        fasp_dcsr_trans(mgl[0].A_nk, &temp1);
        fasp_dcsr_sort(&temp1);
        fasp_dcsr_cp(&temp1, mgl[0].A_nk);
        fasp_dcsr_free(&temp1);
#endif
    }

    for (lvl = 1; lvl < max_levels; lvl++) {
        const INT mm        = mgl[lvl].A.ROW * nb;
        mgl[lvl].num_levels = max_levels;
        mgl[lvl].b          = fasp_dvec_create(mm);
        mgl[lvl].x          = fasp_dvec_create(mm);
        mgl[lvl].w          = fasp_dvec_create(3 * mm);
        mgl[lvl].ILU_levels = param->ILU_levels - lvl; // initialize ILU levels!

        if (mgl[lvl].A_nk != NULL) {

#if WITH_UMFPACK
            // Need to sort the matrix A_nk for UMFPACK
            fasp_dcsr_trans(mgl[lvl].A_nk, &temp1);
            fasp_dcsr_sort(&temp1);
            fasp_dcsr_cp(&temp1, mgl[lvl].A_nk);
            fasp_dcsr_free(&temp1);
#endif
        }
    }

    if (prtlvl > PRINT_NONE) {
        fasp_gettime(&setup_end);
        fasp_amgcomplexity_bsr(mgl, prtlvl);
        fasp_cputime("Unsmoothed aggregation (BSR) setup", setup_end - setup_start);
    }

    fasp_mem_free(vertices);
    vertices = NULL;
    fasp_mem_free(num_aggs);
    num_aggs = NULL;
    fasp_mem_free(Neighbor);
    Neighbor = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
