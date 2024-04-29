/*! \file  PreDataInit.c
 *
 *  \brief Initialize important data structures
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxMemory.c, AuxVector.c, BlaSparseBSR.c, and BlaSparseCSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  \warning Every structures should be initialized before usage.
 */

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_precond_data_init (precond_data *pcdata)
 *
 * \brief Initialize precond_data
 *
 * \param pcdata   Preconditioning data structure
 *
 * \author Chensong Zhang
 * \date   2010/03/23
 */
void fasp_precond_data_init(precond_data* pcdata)
{
    pcdata->AMG_type            = CLASSIC_AMG;
    pcdata->print_level         = PRINT_NONE;
    pcdata->maxit               = 500;
    pcdata->max_levels          = 20;
    pcdata->tol                 = 1e-8;
    pcdata->cycle_type          = V_CYCLE;
    pcdata->smoother            = SMOOTHER_GS;
    pcdata->smooth_order        = CF_ORDER;
    pcdata->presmooth_iter      = 1;
    pcdata->postsmooth_iter     = 1;
    pcdata->relaxation          = 1.1;
    pcdata->coarsening_type     = 1;
    pcdata->coarse_scaling      = ON;
    pcdata->amli_degree         = 1;
    pcdata->nl_amli_krylov_type = SOLVER_GCG;
}

/**
 * \fn AMG_data * fasp_amg_data_create (SHORT max_levels)
 *
 * \brief Create and initialize AMG_data for classical and SA AMG
 *
 * \param max_levels   Max number of levels allowed
 *
 * \return Pointer to the AMG_data data structure
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 */
AMG_data* fasp_amg_data_create(SHORT max_levels)
{
    max_levels = MAX(1, max_levels); // at least allocate one level

    AMG_data* mgl = (AMG_data*)fasp_mem_calloc(max_levels, sizeof(AMG_data));

    INT i;
    for (i = 0; i < max_levels; ++i) {
        mgl[i].max_levels        = max_levels;
        mgl[i].num_levels        = 0;
        mgl[i].near_kernel_dim   = 0;
        mgl[i].near_kernel_basis = NULL;
        mgl[i].cycle_type        = 0;
#if MULTI_COLOR_ORDER
        mgl[i].GS_Theta = 0.0E-2; // 0.0; //1.0E-2;
#endif
    }

    return (mgl);
}

/**
 * \fn void fasp_amg_data_free (AMG_data *mgl, AMG_param *param)
 *
 * \brief Free AMG_data data memeory space
 *
 * \param mgl    Pointer to the AMG_data
 * \param param  Pointer to AMG parameters
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *
 * Modified by Chensong Zhang on 05/05/2013: Clean up param as well!
 * Modified by Hongxuan Zhang on 12/15/2015: Free memory for Intel MKL PARDISO
 * Modified by Chunsheng Feng on 02/12/2017: Permute A back to its origin for ILUtp
 * Modified by Chunsheng Feng on 08/11/2017: Check for max_levels == 1
 */
void fasp_amg_data_free(AMG_data* mgl, AMG_param* param)
{
    const INT max_levels = MAX(1, mgl[0].num_levels);

    INT i;

    switch (param->coarse_solver) {

#if WITH_MUMPS
        /* Destroy MUMPS direct solver on the coarsest level */
        case SOLVER_MUMPS:
            {
                mgl[max_levels - 1].mumps.job = 3;
                fasp_solver_mumps_steps(&mgl[max_levels - 1].A, &mgl[max_levels - 1].b,
                                        &mgl[max_levels - 1].x,
                                        &mgl[max_levels - 1].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        /* Destroy UMFPACK direct solver on the coarsest level */
        case SOLVER_UMFPACK:
            {
                fasp_mem_free(mgl[max_levels - 1].Numeric);
                mgl[max_levels - 1].Numeric = NULL;
                break;
            }
#endif

#if WITH_PARDISO
        /* Destroy PARDISO direct solver on the coarsest level */
        case SOLVER_PARDISO:
            {
                fasp_pardiso_free_internal_mem(&mgl[max_levels - 1].pdata);
                break;
            }

#endif
        default: // Do nothing!
            break;
    }

    for (i = 0; i < max_levels; ++i) {
        fasp_ilu_data_free(&mgl[i].LU);
        fasp_dcsr_free(&mgl[i].A);
        if (max_levels > 1) {
            fasp_dcsr_free(&mgl[i].P);
            fasp_dcsr_free(&mgl[i].R);
        }
        fasp_dvec_free(&mgl[i].b);
        fasp_dvec_free(&mgl[i].x);
        fasp_dvec_free(&mgl[i].w);
        fasp_ivec_free(&mgl[i].cfmark);
        fasp_swz_data_free(&mgl[i].Schwarz);
    }

    for (i = 0; i < mgl->near_kernel_dim; ++i) {
        fasp_mem_free(mgl->near_kernel_basis[i]);
        mgl->near_kernel_basis[i] = NULL;
    }

    fasp_mem_free(mgl->near_kernel_basis);
    mgl->near_kernel_basis = NULL;
    fasp_mem_free(mgl);
    mgl = NULL;

    if (param == NULL) return; // exit if no param given

    if (param->cycle_type == AMLI_CYCLE) {
        fasp_mem_free(param->amli_coef);
        param->amli_coef = NULL;
    }
}

/**
 * \fn void fasp_amg_data_free1 (AMG_data *mgl, AMG_param *param)
 *
 * \brief Free AMG_data data memeory space
 *
 * \param mgl    Pointer to the AMG_data
 * \param param  Pointer to AMG parameters
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *
 * Modified by Chensong Zhang on 05/05/2013: Clean up param as well!
 * Modified by Hongxuan Zhang on 12/15/2015: Free memory for Intel MKL PARDISO
 * Modified by Chunsheng Feng on 02/12/2017: Permute A back to its origin for ILUtp
 * Modified by Chunsheng Feng on 08/11/2017: Check for max_levels == 1
 *
 * The difference with "fasp_amg_data_free1" is that matrix mgl[i].A does not belong to
 * itself and cannot be destroyed here. Li Zhao, 05/20/2023
 */
void fasp_amg_data_free1(AMG_data* mgl, AMG_param* param)
{
    const INT max_levels = MAX(1, mgl[0].num_levels);

    INT i;

    switch (param->coarse_solver) {

#if WITH_MUMPS
        /* Destroy MUMPS direct solver on the coarsest level */
        case SOLVER_MUMPS:
            {
                mgl[max_levels - 1].mumps.job = 3;
                fasp_solver_mumps_steps(&mgl[max_levels - 1].A, &mgl[max_levels - 1].b,
                                        &mgl[max_levels - 1].x,
                                        &mgl[max_levels - 1].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        /* Destroy UMFPACK direct solver on the coarsest level */
        case SOLVER_UMFPACK:
            {
                fasp_mem_free(mgl[max_levels - 1].Numeric);
                mgl[max_levels - 1].Numeric = NULL;
                break;
            }
#endif

#if WITH_PARDISO
        /* Destroy PARDISO direct solver on the coarsest level */
        case SOLVER_PARDISO:
            {
                fasp_pardiso_free_internal_mem(&mgl[max_levels - 1].pdata);
                break;
            }

#endif
        default: // Do nothing!
            break;
    }

    for (i = 0; i < max_levels; ++i) {
        fasp_ilu_data_free(&mgl[i].LU);
        // mgl[i].A is pointer variable, here don't free, Li Zhao, 05/20/2023
        // fasp_dcsr_free(&mgl[i].A);
        if (max_levels > 1) {
            fasp_dcsr_free(&mgl[i].P);
            fasp_dcsr_free(&mgl[i].R);
        }
        fasp_dvec_free(&mgl[i].b);
        fasp_dvec_free(&mgl[i].x);
        fasp_dvec_free(&mgl[i].w);
        fasp_ivec_free(&mgl[i].cfmark);
        fasp_swz_data_free(&mgl[i].Schwarz);
    }

    for (i = 0; i < mgl->near_kernel_dim; ++i) {
        fasp_mem_free(mgl->near_kernel_basis[i]);
        mgl->near_kernel_basis[i] = NULL;
    }

    fasp_mem_free(mgl->near_kernel_basis);
    mgl->near_kernel_basis = NULL;
    fasp_mem_free(mgl);
    mgl = NULL;

    if (param == NULL) return; // exit if no param given

    if (param->cycle_type == AMLI_CYCLE) {
        fasp_mem_free(param->amli_coef);
        param->amli_coef = NULL;
    }
}

/**
 * \fn AMG_data_bsr * fasp_amg_data_bsr_create (SHORT max_levels)
 *
 * \brief Create and initialize AMG_data data sturcture for AMG/SAMG (BSR format)
 *
 * \param max_levels   Max number of levels allowed
 *
 * \return Pointer to the AMG_data data structure
 *
 * \author Xiaozhe Hu
 * \date   08/07/2011
 */
AMG_data_bsr* fasp_amg_data_bsr_create(SHORT max_levels)
{
    max_levels = MAX(1, max_levels); // at least allocate one level

    AMG_data_bsr* mgl =
        (AMG_data_bsr*)fasp_mem_calloc(max_levels, sizeof(AMG_data_bsr));

    INT i;
    for (i = 0; i < max_levels; ++i) {
        mgl[i].max_levels        = max_levels;
        mgl[i].num_levels        = 0;
        mgl[i].near_kernel_dim   = 0;
        mgl[i].near_kernel_basis = NULL;
        mgl[i].A_nk              = NULL;
        mgl[i].P_nk              = NULL;
        mgl[i].R_nk              = NULL;
    }

    return (mgl);
}

/**
 * \fn void fasp_amg_data_bsr_free (AMG_data_bsr *mgl)
 *
 * \brief Free AMG_data_bsr data memeory space
 *
 * \param mgl  Pointer to the AMG_data_bsr
 *
 * \author Xiaozhe Hu, Chensong Zhang
 * \date   2013/02/13
 *
 * Modified by Chensong Zhang on 08/14/2017: Check for max_levels == 1
 */
void fasp_amg_data_bsr_free(AMG_data_bsr* mgl, AMG_param* param)
{
    const INT max_levels = MAX(1, mgl[0].num_levels);

    INT i;
    switch (param->coarse_solver) {

#if WITH_MUMPS
        /* Destroy MUMPS direct solver on the coarsest level */
        case SOLVER_MUMPS:
            {
                mgl[max_levels - 1].mumps.job = 3;
                fasp_solver_mumps_steps(&mgl[max_levels - 1].A, &mgl[max_levels - 1].b,
                                        &mgl[max_levels - 1].x,
                                        &mgl[max_levels - 1].mumps);
                break;
            }
#endif

#if WITH_UMFPACK
        /* Destroy UMFPACK direct solver on the coarsest level */
        case SOLVER_UMFPACK:
            {
                fasp_mem_free(mgl[max_levels - 1].Numeric);
                mgl[max_levels - 1].Numeric = NULL;
                break;
            }
#endif

#if WITH_PARDISO
        /* Destroy PARDISO direct solver on the coarsest level */
        case SOLVER_PARDISO:
            {
                fasp_pardiso_free_internal_mem(&mgl[max_levels - 1].pdata);
                break;
            }

#endif
        default: // Do nothing!
            break;
    }

    for (i = 0; i < max_levels; ++i) {

        fasp_ilu_data_free(&mgl[i].LU);
        fasp_dbsr_free(&mgl[i].A);
        if (max_levels > 1) {
            fasp_dbsr_free(&mgl[i].P);
            fasp_dbsr_free(&mgl[i].R);
        }
        fasp_dvec_free(&mgl[i].b);
        fasp_dvec_free(&mgl[i].x);
        fasp_dvec_free(&mgl[i].diaginv);
        fasp_dvec_free(&mgl[i].diaginv_SS);
        fasp_dcsr_free(&mgl[i].Ac);

        fasp_ilu_data_free(&mgl[i].PP_LU);
        fasp_dcsr_free(&mgl[i].PP);
        fasp_dcsr_free(&mgl[i].TT); // added by LiZhao, 05/23/2023
        fasp_dbsr_free(&mgl[i].PT); // added by LiZhao, 05/19/2023
        fasp_dbsr_free(&mgl[i].SS);
        // fasp_dvec_free(&mgl[i].diaginv_SS);
        fasp_dvec_free(&mgl[i].w);
        fasp_ivec_free(&mgl[i].cfmark);

        fasp_mem_free(mgl[i].pw);
        mgl[i].pw = NULL;
        fasp_mem_free(mgl[i].sw);
        mgl[i].sw = NULL;
    }

    for (i = 0; i < mgl->near_kernel_dim; ++i) {
        fasp_mem_free(mgl->near_kernel_basis[i]);
        mgl->near_kernel_basis[i] = NULL;
    }
    fasp_mem_free(mgl->near_kernel_basis);
    mgl->near_kernel_basis = NULL;
    fasp_mem_free(mgl);
    mgl = NULL;
}

/**
 * \fn void fasp_ilu_data_create (const INT iwk, const INT nwork, ILU_data *iludata)
 *
 * \brief Allocate workspace for ILU factorization
 *
 * \param iwk       Size of the index array
 * \param nwork     Size of the work array
 * \param iludata   Pointer to the ILU_data
 *
 * \author Chensong Zhang
 * \date   2010/04/06
 *
 * Modified by Chunsheng Feng on 02/12/2017: add iperm array for ILUtp
 */
void fasp_ilu_data_create(const INT iwk, const INT nwork, ILU_data* iludata)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: iwk=%d, nwork=%d \n", iwk, nwork);
#endif

    iludata->ijlu = (INT*)fasp_mem_calloc(iwk, sizeof(INT));

    if (iludata->type == ILUtp)
        iludata->iperm = (INT*)fasp_mem_calloc(iludata->row * 2, sizeof(INT));

    iludata->luval = (REAL*)fasp_mem_calloc(iwk, sizeof(REAL));

    iludata->work = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
#if DEBUG_MODE > 0
    printf("### DEBUG: %s ...... %d [End]\n", __FUNCTION__, __LINE__);
#endif

    return;
}

/**
 * \fn void fasp_ilu_data_free (ILU_data *iludata)
 *
 * \brief Create ILU_data sturcture
 *
 * \param iludata   Pointer to ILU_data
 *
 * \author Chensong Zhang
 * \date   2010/04/03
 *
 * Modified by Chunsheng Feng on 02/12/2017: add iperm array for ILUtp
 */
void fasp_ilu_data_free(ILU_data* iludata)
{
    if (iludata == NULL) return; // There is nothing to do!

    fasp_mem_free(iludata->ijlu);
    iludata->ijlu = NULL;
    fasp_mem_free(iludata->luval);
    iludata->luval = NULL;
    fasp_mem_free(iludata->work);
    iludata->work = NULL;
    fasp_mem_free(iludata->ilevL);
    iludata->ilevL = NULL;
    fasp_mem_free(iludata->jlevL);
    iludata->jlevL = NULL;
    fasp_mem_free(iludata->ilevU);
    iludata->ilevU = NULL;
    fasp_mem_free(iludata->jlevU);
    iludata->jlevU = NULL;

    if (iludata->type == ILUtp) {

        if (iludata->A != NULL) {
            // To permute the matrix back to its original state use the loop:
            INT        k;
            const INT  nnz   = iludata->A->nnz;
            const INT* iperm = iludata->iperm;
            for (k = 0; k < nnz; k++) {
                // iperm is in Fortran array format
                iludata->A->JA[k] = iperm[iludata->A->JA[k]] - 1;
            }
        }

        fasp_mem_free(iludata->iperm);
        iludata->iperm = NULL;
    }

    iludata->row = iludata->col = iludata->nzlu = iludata->nwork = iludata->nb =
        iludata->nlevL = iludata->nlevU = 0;
}

/**
 * \fn void fasp_swz_data_free (SWZ_data *swzdata)
 * \brief Free SWZ_data data memeory space
 *
 * \param swzdata      Pointer to the SWZ_data for Schwarz methods
 *
 * \author Xiaozhe Hu
 * \date   2010/04/06
 */
void fasp_swz_data_free(SWZ_data* swzdata)
{
    INT i;

    if (swzdata == NULL) return; // There is nothing to do!

    fasp_dcsr_free(&swzdata->A);

    for (i = 0; i < swzdata->nblk; ++i) fasp_dcsr_free(&((swzdata->blk_data)[i]));

    swzdata->nblk = 0;

    fasp_mem_free(swzdata->iblock);
    swzdata->iblock = NULL;
    fasp_mem_free(swzdata->jblock);
    swzdata->jblock = NULL;

    fasp_dvec_free(&swzdata->rhsloc1);
    fasp_dvec_free(&swzdata->xloc1);

    swzdata->memt = 0;
    fasp_mem_free(swzdata->mask);
    swzdata->mask = NULL;
    fasp_mem_free(swzdata->maxa);
    swzdata->maxa = NULL;

#if WITH_MUMPS
    if (swzdata->mumps == NULL) return;

    for (i = 0; i < swzdata->nblk; ++i) fasp_mumps_free(&((swzdata->mumps)[i]));
#endif
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
