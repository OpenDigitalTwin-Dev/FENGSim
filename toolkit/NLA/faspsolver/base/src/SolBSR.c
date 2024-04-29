/*! \file  SolBSR.c
 *
 *  \brief Iterative solvers for dBSRmat matrices
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxMemory.c, AuxMessage.c, AuxThreads.c, AuxTiming.c, AuxVector.c,
 *         BlaSmallMatInv.c, BlaILUSetupBSR.c, BlaSparseBSR.c, BlaSparseCheck.c,
 *         KryPbcgs.c, KryPcg.c, KryPgmres.c, KryPvfgmres.c, KryPvgmres.c,
 *         PreAMGSetupSA.c, PreAMGSetupUA.c, PreBSR.c, and PreDataInit.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "KryUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_dbsr_itsolver (dBSRmat *A, dvector *b, dvector *x,
 *                                    precond *pc, ITS_param *itparam)
 *
 * \brief Solve Ax=b by preconditioned Krylov methods for BSR matrices
 *
 * \param A        Pointer to the coeff matrix in dBSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param pc       Pointer to the preconditioning action
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Zhiyang Zhou, Xiaozhe Hu
 * \date   10/26/2010
 *
 * Modified by Chunsheng Feng on 03/04/2016: add VBiCGstab solver
 */
INT fasp_solver_dbsr_itsolver(
    dBSRmat* A, dvector* b, dvector* x, precond* pc, ITS_param* itparam)
{
    const SHORT prtlvl        = itparam->print_level;
    const SHORT itsolver_type = itparam->itsolver_type;
    const SHORT stop_type     = itparam->stop_type;
    const SHORT restart       = itparam->restart;
    const INT   MaxIt         = itparam->maxit;
    const REAL  tol           = itparam->tol;
    const REAL  abstol        = itparam->abstol;

    // Local variables
    INT  iter = ERROR_SOLVER_TYPE;
    REAL solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    /* Safe-guard checks on parameters */
    ITS_CHECK(MaxIt, tol);

    switch (itsolver_type) {

        case SOLVER_CG:
            iter = fasp_solver_dbsr_pcg(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                        prtlvl);
            break;

        case SOLVER_BiCGstab:
            iter = fasp_solver_dbsr_pbcgs(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                          prtlvl);
            break;

        case SOLVER_GMRES:
            iter = fasp_solver_dbsr_pgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                           stop_type, prtlvl);
            break;

        case SOLVER_VGMRES:
            iter = fasp_solver_dbsr_pvgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                            stop_type, prtlvl);
            break;

        case SOLVER_VFGMRES:
            iter = fasp_solver_dbsr_pvfgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                             stop_type, prtlvl);
            break;

        default:
            printf("### ERROR: Unknown iterative solver type %d! [%s]\n", itsolver_type,
                   __FUNCTION__);
            return ERROR_SOLVER_TYPE;
    }

    if ((prtlvl > PRINT_MIN) && (iter >= 0)) {
        fasp_gettime(&solve_end);
        fasp_cputime("Iterative method", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return iter;
}

/**
 * \fn INT fasp_solver_dbsr_krylov (dBSRmat *A, dvector *b, dvector *x,
 *                                  ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods for BSR matrices
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Zhiyang Zhou, Xiaozhe Hu
 * \date   10/26/2010
 */
INT fasp_solver_dbsr_krylov(dBSRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;
    INT         status = FASP_SUCCESS;
    REAL        solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // solver part
    fasp_gettime(&solve_start);

    status = fasp_solver_dbsr_itsolver(A, b, x, NULL, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl > PRINT_NONE)
        fasp_cputime("Krylov method totally", solve_end - solve_start);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dbsr_krylov_diag (dBSRmat *A, dvector *b, dvector *x,
 *                                       ITS_param *itparam)
 *
 * \brief Solve Ax=b by diagonal preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Zhiyang Zhou, Xiaozhe Hu
 * \date   10/26/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/15/2012
 */
INT fasp_solver_dbsr_krylov_diag(dBSRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;
    INT         status = FASP_SUCCESS;
    REAL        solve_start, solve_end;

    INT nb  = A->nb, i, k;
    INT nb2 = nb * nb;
    INT ROW = A->ROW;

#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    // setup preconditioner
    precond_diag_bsr diag;
    fasp_dvec_alloc(ROW * nb2, &(diag.diag));

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // get all the diagonal sub-blocks
#ifdef _OPENMP
    if (ROW > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, k)
        for (myid = 0; myid < nthreads; ++myid) {
            fasp_get_start_end(myid, nthreads, ROW, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
                    if (A->JA[k] == i)
                        memcpy(diag.diag.val + i * nb2, A->val + k * nb2,
                               nb2 * sizeof(REAL));
                }
            }
        }
    } else {
#endif
        for (i = 0; i < ROW; ++i) {
            for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
                if (A->JA[k] == i)
                    memcpy(diag.diag.val + i * nb2, A->val + k * nb2,
                           nb2 * sizeof(REAL));
            }
        }
#ifdef _OPENMP
    }
#endif

    diag.nb = nb;

#ifdef _OPENMP
#pragma omp parallel for if (ROW > OPENMP_HOLDS)
#endif
    for (i = 0; i < ROW; ++i) {
        fasp_smat_inv(&(diag.diag.val[i * nb2]), nb);
    }

    precond* pc = (precond*)fasp_mem_calloc(1, sizeof(precond));
    pc->data    = &diag;
    pc->fct     = fasp_precond_dbsr_diag;

    // solver part
    fasp_gettime(&solve_start);

    status = fasp_solver_dbsr_itsolver(A, b, x, pc, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl > PRINT_NONE)
        fasp_cputime("Diag_Krylov method totally", solve_end - solve_start);

    // clean up
    fasp_dvec_free(&(diag.diag));

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dbsr_krylov_ilu (dBSRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, ILU_param *iluparam)
 *
 * \brief Solve Ax=b by ILUs preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param iluparam  Pointer to parameters of ILU
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Shiquang Zhang, Xiaozhe Hu
 * \date   10/26/2010
 */
INT fasp_solver_dbsr_krylov_ilu(
    dBSRmat* A, dvector* b, dvector* x, ITS_param* itparam, ILU_param* iluparam)
{
    const SHORT prtlvl = itparam->print_level;
    REAL        solve_start, solve_end;
    INT         status = FASP_SUCCESS;

    ILU_data LU;
    precond  pc;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->ROW, A->COL, A->NNZ);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // ILU setup for whole matrix
    if ((status = fasp_ilu_dbsr_setup(A, &LU, iluparam)) < 0) goto FINISHED;

    // check iludata
    if ((status = fasp_mem_iludata_check(&LU)) < 0) goto FINISHED;

    // set preconditioner
    pc.data = &LU;
    pc.fct  = fasp_precond_dbsr_ilu;

    // solve
    status = fasp_solver_dbsr_itsolver(A, b, x, &pc, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl > PRINT_NONE)
        fasp_cputime("ILUk_Krylov method totally", solve_end - solve_start);

FINISHED:
    fasp_ilu_data_free(&LU);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dbsr_krylov_amg (dBSRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, AMG_param *amgparam)
 *
 * \brief Solve Ax=b by AMG preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param amgparam  Pointer to parameters of AMG
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   03/16/2012
 */
INT fasp_solver_dbsr_krylov_amg(
    dBSRmat* A, dvector* b, dvector* x, ITS_param* itparam, AMG_param* amgparam)
{
    //--------------------------------------------------------------
    // Part 1: prepare
    // --------------------------------------------------------------

    //! parameters of iterative method
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;

    // return variable
    INT status = FASP_SUCCESS;

    // data of AMG
    AMG_data_bsr* mgl = fasp_amg_data_bsr_create(max_levels);

    // timing
    REAL setup_start, setup_end, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    //--------------------------------------------------------------
    // Part 2: set up the preconditioner
    //--------------------------------------------------------------
    fasp_gettime(&setup_start);

    // initialize A, b, x for mgl[0]
    mgl[0].A = fasp_dbsr_create(A->ROW, A->COL, A->NNZ, A->nb, A->storage_manner);
    mgl[0].b = fasp_dvec_create(mgl[0].A.ROW * mgl[0].A.nb);
    mgl[0].x = fasp_dvec_create(mgl[0].A.COL * mgl[0].A.nb);

    fasp_dbsr_cp(A, &(mgl[0].A));

    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa_bsr(mgl, amgparam);
            break;

        default:
            status = fasp_amg_setup_ua_bsr(mgl, amgparam);
            break;
    }

    if (status < 0) goto FINISHED;

    precond_data_bsr precdata;
    precdata.print_level      = amgparam->print_level;
    precdata.maxit            = amgparam->maxit;
    precdata.tol              = amgparam->tol;
    precdata.cycle_type       = amgparam->cycle_type;
    precdata.smoother         = amgparam->smoother;
    precdata.presmooth_iter   = amgparam->presmooth_iter;
    precdata.postsmooth_iter  = amgparam->postsmooth_iter;
    precdata.coarsening_type  = amgparam->coarsening_type;
    precdata.relaxation       = amgparam->relaxation;
    precdata.coarse_scaling   = amgparam->coarse_scaling;
    precdata.amli_degree      = amgparam->amli_degree;
    precdata.amli_coef        = amgparam->amli_coef;
    precdata.tentative_smooth = amgparam->tentative_smooth;
    precdata.max_levels       = mgl[0].num_levels;
    precdata.mgl_data         = mgl;
    precdata.A                = A;

    precond prec;
    prec.data = &precdata;
    switch (amgparam->cycle_type) {
        case NL_AMLI_CYCLE: // Nonlinear AMLI AMG
            prec.fct = fasp_precond_dbsr_namli;
            break;
        default: // V,W-Cycle AMG
            prec.fct = fasp_precond_dbsr_amg;
            break;
    }

    fasp_gettime(&setup_end);

    if (prtlvl >= PRINT_MIN) fasp_cputime("BSR AMG setup", setup_end - setup_start);

    //--------------------------------------------------------------
    // Part 3: solver
    //--------------------------------------------------------------
    status = fasp_solver_dbsr_itsolver(A, b, x, &prec, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl >= PRINT_MIN) fasp_cputime("BSR Krylov method", solve_end - setup_start);

FINISHED:
    fasp_amg_data_bsr_free(mgl, amgparam);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (status == ERROR_ALLOC_MEM) goto MEMORY_ERROR;
    return status;

MEMORY_ERROR:
    printf("### ERROR: Cannot allocate memory! [%s]\n", __FUNCTION__);
    exit(status);
}

/**
 * \fn INT fasp_solver_dbsr_krylov_amg_nk (dBSRmat *A, dvector *b, dvector *x,
 *                                         ITS_param *itparam, AMG_param *amgparam,
 *                                         dCSRmat *A_nk, dCSRmat *P_nk, dCSRmat *R_nk)
 *
 * \brief Solve Ax=b by AMG with extra near kernel solve preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param amgparam  Pointer to parameters of AMG
 * \param A_nk      Pointer to the coeff matrix for near kernel space in dBSRmat format
 * \param P_nk      Pointer to the prolongation for near kernel space in dBSRmat format
 * \param R_nk      Pointer to the restriction for near kernel space in dBSRmat format
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   05/26/2012
 */
INT fasp_solver_dbsr_krylov_amg_nk(dBSRmat*   A,
                                   dvector*   b,
                                   dvector*   x,
                                   ITS_param* itparam,
                                   AMG_param* amgparam,
                                   dCSRmat*   A_nk,
                                   dCSRmat*   P_nk,
                                   dCSRmat*   R_nk)
{
    //--------------------------------------------------------------
    // Part 1: prepare
    // --------------------------------------------------------------
    // parameters of iterative method
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;

    // return variable
    INT status = FASP_SUCCESS;

    // data of AMG
    AMG_data_bsr* mgl = fasp_amg_data_bsr_create(max_levels);

    // timing
    REAL setup_start, setup_end, setup_time;
    REAL solve_start, solve_end, solve_time;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    //--------------------------------------------------------------
    // Part 2: set up the preconditioner
    //--------------------------------------------------------------
    fasp_gettime(&setup_start);

    // initialize A, b, x for mgl[0]
    mgl[0].A = fasp_dbsr_create(A->ROW, A->COL, A->NNZ, A->nb, A->storage_manner);
    fasp_dbsr_cp(A, &(mgl[0].A));
    mgl[0].b = fasp_dvec_create(mgl[0].A.ROW * mgl[0].A.nb);
    mgl[0].x = fasp_dvec_create(mgl[0].A.COL * mgl[0].A.nb);

    // near kernel space
    mgl[0].A_nk = NULL;
    mgl[0].P_nk = P_nk;
    mgl[0].R_nk = R_nk;

    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa_bsr(mgl, amgparam);
            break;

        default:
            status = fasp_amg_setup_ua_bsr(mgl, amgparam);
            break;
    }

    if (status < 0) goto FINISHED;

    precond_data_bsr precdata;
    precdata.print_level      = amgparam->print_level;
    precdata.maxit            = amgparam->maxit;
    precdata.tol              = amgparam->tol;
    precdata.cycle_type       = amgparam->cycle_type;
    precdata.smoother         = amgparam->smoother;
    precdata.presmooth_iter   = amgparam->presmooth_iter;
    precdata.postsmooth_iter  = amgparam->postsmooth_iter;
    precdata.coarsening_type  = amgparam->coarsening_type;
    precdata.relaxation       = amgparam->relaxation;
    precdata.coarse_scaling   = amgparam->coarse_scaling;
    precdata.amli_degree      = amgparam->amli_degree;
    precdata.amli_coef        = amgparam->amli_coef;
    precdata.tentative_smooth = amgparam->tentative_smooth;
    precdata.max_levels       = mgl[0].num_levels;
    precdata.mgl_data         = mgl;
    precdata.A                = A;

#if WITH_UMFPACK // use UMFPACK directly
    dCSRmat A_tran;
    A_tran = fasp_dcsr_create(A_nk->row, A_nk->col, A_nk->nnz);
    fasp_dcsr_transz(A_nk, NULL, &A_tran);
    // It is equivalent to do transpose and then sort
    //     fasp_dcsr_trans(A_nk, &A_tran);
    //     fasp_dcsr_sort(&A_tran);
    precdata.A_nk = &A_tran;
#else
    precdata.A_nk = A_nk;
#endif

    precdata.P_nk = P_nk;
    precdata.R_nk = R_nk;

    if (status < 0) goto FINISHED;

    precond prec;
    prec.data = &precdata;

    prec.fct = fasp_precond_dbsr_amg_nk;

    fasp_gettime(&setup_end);

    setup_time = setup_end - setup_start;

    if (prtlvl >= PRINT_MIN) fasp_cputime("BSR AMG setup", setup_time);

    //--------------------------------------------------------------
    // Part 3: solver
    //--------------------------------------------------------------
    fasp_gettime(&solve_start);

    status = fasp_solver_dbsr_itsolver(A, b, x, &prec, itparam);

    fasp_gettime(&solve_end);

    solve_time = solve_end - solve_start;

    if (prtlvl >= PRINT_MIN) {
        fasp_cputime("BSR Krylov method", setup_time + solve_time);
    }

FINISHED:
    fasp_amg_data_bsr_free(mgl, amgparam);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

#if WITH_UMFPACK // use UMFPACK directly
    fasp_dcsr_free(&A_tran);
#endif
    if (status == ERROR_ALLOC_MEM) goto MEMORY_ERROR;
    return status;

MEMORY_ERROR:
    printf("### ERROR: Cannot allocate memory! [%s]\n", __FUNCTION__);
    exit(status);
}

/**
 * \fn INT fasp_solver_dbsr_krylov_nk_amg (dBSRmat *A, dvector *b, dvector *x,
 *                                         ITS_param *itparam, AMG_param *amgparam,
 *                                         const INT nk_dim, dvector *nk)
 *
 * \brief Solve Ax=b by AMG preconditioned Krylov methods with extra kernal space
 *
 * \param A         Pointer to the coeff matrix in dBSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param amgparam  Pointer to parameters of AMG
 * \param nk_dim    Dimension of the near kernel spaces
 * \param nk        Pointer to the near kernal spaces
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   05/27/2012
 */
INT fasp_solver_dbsr_krylov_nk_amg(dBSRmat*   A,
                                   dvector*   b,
                                   dvector*   x,
                                   ITS_param* itparam,
                                   AMG_param* amgparam,
                                   const INT  nk_dim,
                                   dvector*   nk)
{
    //--------------------------------------------------------------
    // Part 1: prepare
    // --------------------------------------------------------------
    //! parameters of iterative method
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;

    // local variable
    INT i;

    // return variable
    INT status = FASP_SUCCESS;

    // data of AMG
    AMG_data_bsr* mgl = fasp_amg_data_bsr_create(max_levels);

    // timing
    REAL setup_start, setup_end, setup_time;
    REAL solve_start, solve_end, solve_time;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    //--------------------------------------------------------------
    // Part 2: set up the preconditioner
    //--------------------------------------------------------------
    fasp_gettime(&setup_start);

    // initialize A, b, x for mgl[0]
    mgl[0].A = fasp_dbsr_create(A->ROW, A->COL, A->NNZ, A->nb, A->storage_manner);
    fasp_dbsr_cp(A, &(mgl[0].A));
    mgl[0].b = fasp_dvec_create(mgl[0].A.ROW * mgl[0].A.nb);
    mgl[0].x = fasp_dvec_create(mgl[0].A.COL * mgl[0].A.nb);

    /*-----------------------*/
    /*-- setup null spaces --*/
    /*-----------------------*/

    // null space for whole Jacobian
    mgl[0].near_kernel_dim = nk_dim;
    mgl[0].near_kernel_basis =
        (REAL**)fasp_mem_calloc(mgl->near_kernel_dim, sizeof(REAL*));

    for (i = 0; i < mgl->near_kernel_dim; ++i) mgl[0].near_kernel_basis[i] = nk[i].val;

    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa_bsr(mgl, amgparam);
            break;

        default:
            status = fasp_amg_setup_ua_bsr(mgl, amgparam);
            break;
    }

    if (status < 0) goto FINISHED;

    precond_data_bsr precdata;
    precdata.print_level      = amgparam->print_level;
    precdata.maxit            = amgparam->maxit;
    precdata.tol              = amgparam->tol;
    precdata.cycle_type       = amgparam->cycle_type;
    precdata.smoother         = amgparam->smoother;
    precdata.presmooth_iter   = amgparam->presmooth_iter;
    precdata.postsmooth_iter  = amgparam->postsmooth_iter;
    precdata.coarsening_type  = amgparam->coarsening_type;
    precdata.relaxation       = amgparam->relaxation;
    precdata.coarse_scaling   = amgparam->coarse_scaling;
    precdata.amli_degree      = amgparam->amli_degree;
    precdata.amli_coef        = amgparam->amli_coef;
    precdata.tentative_smooth = amgparam->tentative_smooth;
    precdata.max_levels       = mgl[0].num_levels;
    precdata.mgl_data         = mgl;
    precdata.A                = A;

    if (status < 0) goto FINISHED;

    precond prec;
    prec.data = &precdata;
    switch (amgparam->cycle_type) {
        case NL_AMLI_CYCLE: // Nonlinear AMLI AMG
            prec.fct = fasp_precond_dbsr_namli;
            break;
        default: // V,W-Cycle AMG
            prec.fct = fasp_precond_dbsr_amg;
            break;
    }

    fasp_gettime(&setup_end);

    setup_time = setup_end - setup_start;

    if (prtlvl >= PRINT_MIN) fasp_cputime("BSR AMG setup", setup_time);

    //--------------------------------------------------------------
    // Part 3: solver
    //--------------------------------------------------------------
    fasp_gettime(&solve_start);

    status = fasp_solver_dbsr_itsolver(A, b, x, &prec, itparam);

    fasp_gettime(&solve_end);

    solve_time = solve_end - solve_start;

    if (prtlvl >= PRINT_MIN) {
        fasp_cputime("BSR Krylov method", setup_time + solve_time);
    }

FINISHED:
    fasp_amg_data_bsr_free(mgl, amgparam);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    if (status == ERROR_ALLOC_MEM) goto MEMORY_ERROR;
    return status;

MEMORY_ERROR:
    printf("### ERROR: Cannot allocate memory! [%s]\n", __FUNCTION__);
    exit(status);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
