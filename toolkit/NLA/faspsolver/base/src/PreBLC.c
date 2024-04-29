/*! \file  PreBLC.c
 *
 *  \brief Preconditioners for dBLCmat matrices
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxVector.c, BlaSpmvCSR.c, and PreMGCycle.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  TODO: Separate solve and setup phases for direct solvers!!! --Chensong
 */

#include "fasp.h"
#include "fasp_block.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_precond_dblc_diag_3 (REAL *r, REAL *z, void *data)
 *
 * \brief Block diagonal preconditioner (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_diag_3 (REAL *r,
                               REAL *z,
                               void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!

    precond_data_blc *precdata = (precond_data_blc *)data;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr  = &(precdata->r);
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N  = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
#elif WITH_SuperLU
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
#endif
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);

#endif
}

/**
 * \fn void fasp_precond_dblc_diag_3_amg (REAL *r, REAL *z, void *data)
 *
 * \brief Block diagonal preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved by AMG
 */
void fasp_precond_dblc_diag_3_amg (REAL *r,
                                   REAL *z,
                                   void *data)
{
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A     = precdata->Ablc;
    dvector *tempr = &(precdata->r);
    
    AMG_param *amgparam = precdata->amgparam;
    AMG_data **mgl      = precdata->mgl;
    
    const INT N0 = A->blocks[0]->row;
    const INT N1 = A->blocks[4]->row;
    const INT N2 = A->blocks[8]->row;
    const INT N  = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    r0.row = N0; z0.row = N0; r1.row = N1; z1.row = N1; r2.row = N2; z2.row = N2;
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]); z0.val = z;
    z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A00 block
    mgl[0]->b.row=N0; fasp_darray_cp(N0, r0.val, mgl[0]->b.val);
    mgl[0]->x.row=N0; fasp_dvec_set(N0, &mgl[0]->x, 0.0);
    
    fasp_solver_mgcycle(mgl[0], amgparam);
    fasp_darray_cp(N0, mgl[0]->x.val, z0.val);
    
    // Preconditioning A11 block
    mgl[1]->b.row=N1; fasp_darray_cp(N1, r1.val, mgl[1]->b.val);
    mgl[1]->x.row=N1; fasp_dvec_set(N1, &mgl[1]->x,0.0);
    
    fasp_solver_mgcycle(mgl[1], amgparam);
    fasp_darray_cp(N1, mgl[1]->x.val, z1.val);
    
    // Preconditioning A22 block
    mgl[2]->b.row=N2; fasp_darray_cp(N2, r2.val, mgl[2]->b.val);
    mgl[2]->x.row=N2; fasp_dvec_set(N2, &mgl[2]->x,0.0);
    
    fasp_solver_mgcycle(mgl[2], amgparam);
    fasp_darray_cp(N2, mgl[2]->x.val, z2.val);
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
}

/**
 * \fn void fasp_precond_dblc_diag_4 (REAL *r, REAL *z, void *data)
 *
 * \brief Block diagonal preconditioning (4x4 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_diag_4 (REAL *r,
                               REAL *z,
                               void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!

    precond_data_blc *precdata=(precond_data_blc *)data;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr = &(precdata->r);
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N3 = A_diag[3].row;
    const INT N = N0 + N1 + N2 + N3;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
    dvector r0, r1, r2, r3, z0, z1, z2, z3;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    r3.row = N3; z3.row = N3;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]); r3.val = &(r[N0+N1+N2]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]); z3.val = &(z[N0+N1+N2]);
#elif WITH_SuperLU
    dvector r0, r1, r2, r3, z0, z1, z2, z3;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    r3.row = N3; z3.row = N3;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]); r3.val = &(r[N0+N1+N2]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]); z3.val = &(z[N0+N1+N2]);
#endif
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // Preconditioning A33 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[3], &r3, &z3, LU_diag[3], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[3], &r3, &z3, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
    
#endif
}

/**
 * \fn void fasp_precond_dblc_lower_3 (REAL *r, REAL *z, void *data)
 *
 * \brief block lower triangular preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_lower_3 (REAL *r,
                                REAL *z,
                                void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!

    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr = &(precdata->r);
    
#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
#endif

    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N  = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // r1 = r1 - A3*z0
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[3], z0.val, r1.val);
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // r2 = r2 - A6*z0 - A7*z1
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[6], z0.val, r2.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[7], z1.val, r2.val);
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
    
#endif
}

/**
 * \fn void fasp_precond_dblc_lower_3_amg (REAL *r, REAL *z, void *data)
 *
 * \brief block lower triangular preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved by AMG
 */
void fasp_precond_dblc_lower_3_amg (REAL *r,
                                    REAL *z,
                                    void *data)
{
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dvector *tempr = &(precdata->r);
    
    AMG_param *amgparam = precdata->amgparam;
    AMG_data **mgl = precdata->mgl;
    
    const INT N0 = A->blocks[0]->row;
    const INT N1 = A->blocks[4]->row;
    const INT N2 = A->blocks[8]->row;
    const INT N = N0 + N1 + N2;
    
    INT i;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    r0.row = N0; z0.row = N0; r1.row = N1; z1.row = N1; r2.row = N2; z2.row = N2;
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]); z0.val = z;
    z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A00 block
    mgl[0]->b.row=N0; fasp_darray_cp(N0, r0.val, mgl[0]->b.val);
    mgl[0]->x.row=N0; fasp_dvec_set(N0, &mgl[0]->x, 0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[0], amgparam);
    fasp_darray_cp(N0, mgl[0]->x.val, z0.val);
    
    // r1 = r1 - A10*z0
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[3], z0.val, r1.val);
    
    // Preconditioning A11 block
    mgl[1]->b.row=N1; fasp_darray_cp(N1, r1.val, mgl[1]->b.val);
    mgl[1]->x.row=N1; fasp_dvec_set(N1, &mgl[1]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[1], amgparam);
    fasp_darray_cp(N1, mgl[1]->x.val, z1.val);
    
    // r2 = r2 - A20*z0 - A21*z1
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[6], z0.val, r2.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[7], z1.val, r2.val);
    
    // Preconditioning A22 block
    mgl[2]->b.row=N2; fasp_darray_cp(N2, r2.val, mgl[2]->b.val);
    mgl[2]->x.row=N2; fasp_dvec_set(N2, &mgl[2]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[2], amgparam);
    fasp_darray_cp(N2, mgl[2]->x.val, z2.val);
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
}

/**
 * \fn void fasp_precond_dblc_lower_4 (REAL *r, REAL *z, void *data)
 *
 * \brief block lower triangular preconditioning (4x4 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   07/10/2014
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_lower_4 (REAL *r,
                                REAL *z,
                                void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!
    
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr = &(precdata->r);

#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
#endif
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N3 = A_diag[3].row;
    const INT N  = N0 + N1 + N2 + N3;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, r3, z0, z1, z2, z3;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    r3.row = N3; z3.row = N3;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]); r3.val = &(r[N0+N1+N2]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]); z3.val = &(z[N0+N1+N2]);
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // r1 = r1 - A4*z0
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[4], z0.val, r1.val);
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // r2 = r2 - A8*z0 - A9*z1
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[8], z0.val, r2.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[9], z1.val, r2.val);
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // r3 = r3 - A12*z0 - A13*z1-A14*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[12], z0.val, r3.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[13], z1.val, r3.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[14], z2.val, r3.val);
    
    // Preconditioning A33 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[3], &r3, &z3, LU_diag[3], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[3], &r3, &z3, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);

#endif
}

/**
 * \fn void fasp_precond_dblc_upper_3 (REAL *r, REAL *z, void *data)
 *
 * \brief block upper triangular preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/18/2015
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_upper_3 (REAL *r,
                                REAL *z,
                                void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!
    
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr = &(precdata->r);

#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
#endif
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N  = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // r1 = r1 - A5*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[5], z2.val, r1.val);
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // r0 = r0 - A1*z1 - A2*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[1], z1.val, r0.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[2], z2.val, r0.val);
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);

#endif
}

/**
 * \fn void fasp_precond_dblc_upper_3_amg (REAL *r, REAL *z, void *data)
 *
 * \brief block upper triangular preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/19/2015
 *
 * \note   Each diagonal block is solved by AMG
 */
void fasp_precond_dblc_upper_3_amg (REAL *r,
                                    REAL *z,
                                    void *data)
{
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    
    AMG_param *amgparam = precdata->amgparam;
    AMG_data **mgl = precdata->mgl;
    
    dvector *tempr = &(precdata->r);
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N  = N0 + N1 + N2;
    
    INT i;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A22 block
    mgl[2]->b.row=N2; fasp_darray_cp(N2, r2.val, mgl[2]->b.val);
    mgl[2]->x.row=N2; fasp_dvec_set(N2, &mgl[2]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[2], amgparam);
    fasp_darray_cp(N2, mgl[2]->x.val, z2.val);
    
    // r1 = r1 - A5*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[5], z2.val, r1.val);
    
    // Preconditioning A11 block
    mgl[1]->b.row=N1; fasp_darray_cp(N1, r1.val, mgl[1]->b.val);
    mgl[1]->x.row=N1; fasp_dvec_set(N1, &mgl[1]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[1], amgparam);
    fasp_darray_cp(N1, mgl[1]->x.val, z1.val);
    
    // r0 = r0 - A1*z1 - A2*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[1], z1.val, r0.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[2], z2.val, r0.val);
    
    // Preconditioning A00 block
    mgl[0]->b.row=N0; fasp_darray_cp(N0, r0.val, mgl[0]->b.val);
    mgl[0]->x.row=N0; fasp_dvec_set(N0, &mgl[0]->x, 0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[0], amgparam);
    fasp_darray_cp(N0, mgl[0]->x.val, z0.val);
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
}

/**
 * \fn void fasp_precond_dblc_SGS_3 (REAL *r, REAL *z, void *data)
 *
 * \brief Block symmetric GS preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/19/2015
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_SGS_3 (REAL *r,
                              REAL *z,
                              void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!

    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    dvector *tempr = &(precdata->r);

#if WITH_UMFPACK
    void **LU_diag = precdata->LU_diag;
#endif
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // r1 = r1 - A3*z0
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[3], z0.val, r1.val);
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // r2 = r2 - A6*z0 - A7*z1
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[6], z0.val, r2.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[7], z1.val, r2.val);
    
    // Preconditioning A22 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[2], &r2, &z2, LU_diag[2], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[2], &r2, &z2, 0);
#endif
    
    // r1 = r1 - A5*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[5], z2.val, r1.val);
    
    // Preconditioning A11 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[1], &r1, &z1, LU_diag[1], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[1], &r1, &z1, 0);
#endif
    
    // r0 = r0 - A1*z1 - A2*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[1], z1.val, r0.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[2], z2.val, r0.val);
    
    // Preconditioning A00 block
#if WITH_UMFPACK
    /* use UMFPACK direct solver */
    fasp_umfpack_solve(&A_diag[0], &r0, &z0, LU_diag[0], 0);
#elif WITH_SuperLU
    /* use SuperLU direct solver */
    fasp_solver_superlu(&A_diag[0], &r0, &z0, 0);
#endif
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);

#endif
}

/**
 * \fn void fasp_precond_dblc_SGS_3_amg (REAL *r, REAL *z, void *data)
 *
 * \brief Block symmetric GS preconditioning (3x3 blocks)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/19/2015
 *
 * \note   Each diagonal block is solved by AMG
 */
void fasp_precond_dblc_SGS_3_amg (REAL *r,
                                  REAL *z,
                                  void *data)
{
    precond_data_blc *precdata = (precond_data_blc *)data;
    dBLCmat *A = precdata->Ablc;
    dCSRmat *A_diag = precdata->A_diag;
    
    AMG_param *amgparam = precdata->amgparam;
    AMG_data **mgl = precdata->mgl;
    
    INT i;
    
    dvector *tempr = &(precdata->r);
    
    const INT N0 = A_diag[0].row;
    const INT N1 = A_diag[1].row;
    const INT N2 = A_diag[2].row;
    const INT N = N0 + N1 + N2;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, tempr->val);
    fasp_darray_set(N, z, 0.0);
    
    // prepare
    dvector r0, r1, r2, z0, z1, z2;
    
    r0.row = N0; z0.row = N0;
    r1.row = N1; z1.row = N1;
    r2.row = N2; z2.row = N2;
    
    r0.val = r; r1.val = &(r[N0]); r2.val = &(r[N0+N1]);
    z0.val = z; z1.val = &(z[N0]); z2.val = &(z[N0+N1]);
    
    // Preconditioning A00 block
    mgl[0]->b.row=N0; fasp_darray_cp(N0, r0.val, mgl[0]->b.val);
    mgl[0]->x.row=N0; fasp_dvec_set(N0, &mgl[0]->x, 0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[0], amgparam);
    fasp_darray_cp(N0, mgl[0]->x.val, z0.val);
    
    // r1 = r1 - A3*z0
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[3], z0.val, r1.val);
    
    // Preconditioning A11 block
    mgl[1]->b.row=N1; fasp_darray_cp(N1, r1.val, mgl[1]->b.val);
    mgl[1]->x.row=N1; fasp_dvec_set(N1, &mgl[1]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[1], amgparam);
    fasp_darray_cp(N1, mgl[1]->x.val, z1.val);
    
    // r2 = r2 - A6*z0 - A7*z1
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[6], z0.val, r2.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[7], z1.val, r2.val);
    
    // Preconditioning A22 block
    mgl[2]->b.row=N2; fasp_darray_cp(N2, r2.val, mgl[2]->b.val);
    mgl[2]->x.row=N2; fasp_dvec_set(N2, &mgl[2]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[2], amgparam);
    fasp_darray_cp(N2, mgl[2]->x.val, z2.val);
    
    // r1 = r1 - A5*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[5], z2.val, r1.val);
    
    // Preconditioning A11 block
    mgl[1]->b.row=N1; fasp_darray_cp(N1, r1.val, mgl[1]->b.val);
    mgl[1]->x.row=N1; fasp_dvec_set(N1, &mgl[1]->x,0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[1], amgparam);
    fasp_darray_cp(N1, mgl[1]->x.val, z1.val);
    
    // r0 = r0 - A1*z1 - A2*z2
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[1], z1.val, r0.val);
    fasp_blas_dcsr_aAxpy(-1.0, A->blocks[2], z2.val, r0.val);
    
    // Preconditioning A00 block
    mgl[0]->b.row=N0; fasp_darray_cp(N0, r0.val, mgl[0]->b.val);
    mgl[0]->x.row=N0; fasp_dvec_set(N0, &mgl[0]->x, 0.0);
    
    for(i=0;i<1;++i) fasp_solver_mgcycle(mgl[0], amgparam);
    fasp_darray_cp(N0, mgl[0]->x.val, z0.val);
    
    // restore r
    fasp_darray_cp(N, tempr->val, r);
}

/**
 * \fn void fasp_precond_dblc_sweeping (REAL *r, REAL *z, void *data)
 *
 * \brief Sweeping preconditioner for Maxwell equations
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   05/01/2014
 *
 * \note   Each diagonal block is solved exactly
 */
void fasp_precond_dblc_sweeping (REAL *r,
                                 REAL *z,
                                 void *data)
{
#if WITH_UMFPACK || WITH_SuperLU // Must use direct solvers for this method!

    precond_data_sweeping *precdata = (precond_data_sweeping *)data;
    
    INT NumLayers = precdata->NumLayers;
    dBLCmat *A = precdata->A;
    dBLCmat *Ai = precdata->Ai;
    dCSRmat *local_A = precdata->local_A;
    ivector *local_index = precdata->local_index;

#if WITH_UMFPACK
    void **local_LU = precdata->local_LU;
#endif
    
    dvector *r_backup = &(precdata->r);
    REAL *w = precdata->w;
    
    // local veriables
    INT i,l;
    dvector temp_r;
    dvector temp_e;
    
    dvector *local_r = (dvector *)fasp_mem_calloc(NumLayers, sizeof(dvector));
    dvector *local_e = (dvector *)fasp_mem_calloc(NumLayers, sizeof(dvector));
    
    // calculate the size and generate block local_r and local_z
    INT N=0;
    
    for (l=0;l<NumLayers; l++) {
        
        local_r[l].row = A->blocks[l*NumLayers+l]->row;
        local_r[l].val = r+N;
        
        local_e[l].row = A->blocks[l*NumLayers+l]->col;
        local_e[l].val = z+N;
        
        N = N+A->blocks[l*NumLayers+l]->col;
        
    }
    
    temp_r.val = w;
    temp_e.val = w+N;
    
    // back up r, setup z;
    fasp_darray_cp(N, r, r_backup->val);
    fasp_darray_cp(N, r, z);
    
    // L^{-1}r
    for (l=0; l<NumLayers-1; l++){
        
        temp_r.row = local_A[l].row;
        temp_e.row = local_A[l].row;
        
        fasp_dvec_set(local_A[l].row, &temp_r, 0.0);
        
        for (i=0; i<local_e[l].row; i++){
            temp_r.val[local_index[l].val[i]] = local_e[l].val[i];
        }
        
#if WITH_UMFPACK
        /* use UMFPACK direct solver */
        fasp_umfpack_solve(&local_A[l], &temp_r, &temp_e, local_LU[l], 0);
#elif WITH_SuperLU
        /* use SuperLU direct solver */
        fasp_solver_superlu(&local_A[l], &temp_r, &temp_e, 0);
#endif
        
        for (i=0; i<local_r[l].row; i++){
            local_r[l].val[i] = temp_e.val[local_index[l].val[i]];
        }
        
        fasp_blas_dcsr_aAxpy(-1.0, Ai->blocks[(l+1)*NumLayers+l], local_r[l].val,
                             local_e[l+1].val);
        
    }
    
    // D^{-1}L^{-1}r
    for (l=0; l<NumLayers; l++){
        
        temp_r.row = local_A[l].row;
        temp_e.row = local_A[l].row;
        
        fasp_dvec_set(local_A[l].row, &temp_r, 0.0);
        
        for (i=0; i<local_e[l].row; i++){
            temp_r.val[local_index[l].val[i]] = local_e[l].val[i];
        }
        
#if WITH_UMFPACK
        /* use UMFPACK direct solver */
        fasp_umfpack_solve(&local_A[l], &temp_r, &temp_e, local_LU[l], 0);
#elif WITH_SuperLU
        /* use SuperLU direct solver */
        fasp_solver_superlu(&local_A[l], &temp_r, &temp_e, 0);
#endif
        
        for (i=0; i<local_e[l].row; i++) {
            local_e[l].val[i] = temp_e.val[local_index[l].val[i]];
        }
        
    }
    
    // L^{-t}D^{-1}L^{-1}u
    for (l=NumLayers-2; l>=0; l--){
        
        temp_r.row = local_A[l].row;
        temp_e.row = local_A[l].row;
        
        fasp_dvec_set(local_A[l].row, &temp_r, 0.0);
        
        fasp_blas_dcsr_mxv (Ai->blocks[l*NumLayers+l+1], local_e[l+1].val, local_r[l].val);
        
        for (i=0; i<local_r[l].row; i++){
            temp_r.val[local_index[l].val[i]] = local_r[l].val[i];
        }
        
#if WITH_UMFPACK
        /* use UMFPACK direct solver */
        fasp_umfpack_solve(&local_A[l], &temp_r, &temp_e, local_LU[l], 0);
#elif WITH_SuperLU
        /* use SuperLU direct solver */
        fasp_solver_superlu(&local_A[l], &temp_r, &temp_e, 0);
#endif
        
        for (i=0; i<local_e[l].row; i++){
            local_e[l].val[i] = local_e[l].val[i] - temp_e.val[local_index[l].val[i]];
        }
        
    }
    
    // restore r
    fasp_darray_cp(N, r_backup->val, r);

#endif
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
