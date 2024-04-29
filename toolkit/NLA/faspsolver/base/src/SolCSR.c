/*! \file  SolCSR.c
 *
 *  \brief Iterative solvers for dCSRmat matrices
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxMemory.c, AuxMessage.c, AuxParam.c, AuxTiming.c, AuxVector.c,
 *         BlaILUSetupCSR.c, BlaSchwarzSetup.c, BlaSparseCheck.c, BlaSparseCSR.c,
 *         KryPbcgs.c, KryPcg.c, KryPgcg.c, KryPgcr.c, KryPgmres.c, KryPminres.c,
 *         KryPvfgmres.c, KryPvgmres.c, PreAMGSetupRS.c, PreAMGSetupSA.c,
 *         PreAMGSetupUA.c, PreCSR.c, and PreDataInit.c
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
 * \fn INT fasp_solver_dcsr_itsolver (dCSRmat *A, dvector *b, dvector *x,
 *                                    precond *pc, ITS_param *itparam)
 *
 * \brief Solve Ax=b by preconditioned Krylov methods for CSR matrices
 *
 * \note  This is an abstract interface for iterative methods.
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param pc       Pointer to the preconditioning action
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   09/25/2009
 */
INT fasp_solver_dcsr_itsolver(dCSRmat* A, dvector* b, dvector* x, precond* pc,
                              ITS_param* itparam)
{
    const SHORT prtlvl        = itparam->print_level;
    const SHORT itsolver_type = itparam->itsolver_type;
    const SHORT stop_type     = itparam->stop_type;
    const SHORT restart       = itparam->restart;
    const INT   MaxIt         = itparam->maxit;
    const REAL  tol           = itparam->tol;
    const REAL  abstol        = itparam->abstol;

    /* Local Variables */
    REAL solve_start, solve_end;
    INT  iter;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    /* check matrix data */
    fasp_check_dCSRmat(A);

    /* Safe-guard checks on parameters */
    ITS_CHECK(MaxIt, tol);

    /* Choose a desirable Krylov iterative solver */
    switch (itsolver_type) {
        case SOLVER_CG:
            iter = fasp_solver_dcsr_pcg(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                        prtlvl);
            break;

        case SOLVER_BiCGstab:
            iter = fasp_solver_dcsr_pbcgs(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                          prtlvl);
            break;

        case SOLVER_MinRes:
            iter = fasp_solver_dcsr_pminres(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                            prtlvl);
            break;

        case SOLVER_GMRES:
            iter = fasp_solver_dcsr_pgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                           stop_type, prtlvl);
            break;

        case SOLVER_VGMRES:
            iter = fasp_solver_dcsr_pvgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                            stop_type, prtlvl);
            break;

        case SOLVER_VFGMRES:
            iter = fasp_solver_dcsr_pvfgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                             stop_type, prtlvl);
            break;

        case SOLVER_GCG:
            iter = fasp_solver_dcsr_pgcg(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                         prtlvl);
            break;

        case SOLVER_GCR:
            iter = fasp_solver_dcsr_pgcr(A, b, x, pc, tol, abstol, MaxIt, restart,
                                         stop_type, prtlvl);
            break;

        default:
            printf("### ERROR: Unknown iterative solver type %d! [%s]\n", itsolver_type,
                   __FUNCTION__);
            return ERROR_SOLVER_TYPE;
    }

    if ((prtlvl >= PRINT_SOME) && (iter >= 0)) {
        fasp_gettime(&solve_end);
        fasp_cputime("Iterative method", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return iter;
}

/**
 * \fn INT fasp_solver_dcsr_itsolver_s (dCSRmat *A, dvector *b, dvector *x,
 *                                      precond *pc, ITS_param *itparam)
 *
 * \brief Solve Ax=b by preconditioned Krylov methods with safe-net for CSR matrices
 *
 * \note  This is an abstract interface for iterative methods.
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param pc       Pointer to the preconditioning action
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   10/21/2017
 */
INT fasp_solver_dcsr_itsolver_s(dCSRmat* A, dvector* b, dvector* x, precond* pc,
                                ITS_param* itparam)
{
    const SHORT prtlvl        = itparam->print_level;
    const SHORT itsolver_type = itparam->itsolver_type;
    const SHORT stop_type     = itparam->stop_type;
    const SHORT restart       = itparam->restart;
    const INT   MaxIt         = itparam->maxit;
    const REAL  tol           = itparam->tol;

    /* Local Variables */
    REAL solve_start, solve_end;
    INT  iter;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    /* check matrix data */
    fasp_check_dCSRmat(A);

    /* Safe-guard checks on parameters */
    ITS_CHECK(MaxIt, tol);

    /* Choose a desirable Krylov iterative solver */
    switch (itsolver_type) {
        case SOLVER_CG:
            iter = fasp_solver_dcsr_spcg(A, b, x, pc, tol, MaxIt, stop_type, prtlvl);
            break;

        case SOLVER_BiCGstab:
            iter = fasp_solver_dcsr_spbcgs(A, b, x, pc, tol, MaxIt, stop_type, prtlvl);
            break;

        case SOLVER_MinRes:
            iter =
                fasp_solver_dcsr_spminres(A, b, x, pc, tol, MaxIt, stop_type, prtlvl);
            break;

        case SOLVER_GMRES:
            iter = fasp_solver_dcsr_spgmres(A, b, x, pc, tol, MaxIt, restart, stop_type,
                                            prtlvl);
            break;

        case SOLVER_VGMRES:
            iter = fasp_solver_dcsr_spvgmres(A, b, x, pc, tol, MaxIt, restart,
                                             stop_type, prtlvl);
            break;

        default:
            printf("### ERROR: Unknown iterative solver type %d! [%s]\n", itsolver_type,
                   __FUNCTION__);
            return ERROR_SOLVER_TYPE;
    }

    if ((prtlvl >= PRINT_SOME) && (iter >= 0)) {
        fasp_gettime(&solve_end);
        fasp_cputime("Iterative method", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return iter;
}

/**
 * \fn INT fasp_solver_dcsr_krylov (dCSRmat *A, dvector *b, dvector *x,
 *                                  ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods for CSR matrices
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang, Shiquan Zhang
 * \date   09/25/2009
 */
INT fasp_solver_dcsr_krylov(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    status = fasp_solver_dcsr_itsolver(A, b, x, NULL, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        fasp_cputime("Krylov method totally", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_s (dCSRmat *A, dvector *b, dvector *x,
 *                                    ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods with safe-net for CSR matrices
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   10/22/2017
 */
INT fasp_solver_dcsr_krylov_s(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    status = fasp_solver_dcsr_itsolver_s(A, b, x, NULL, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        fasp_cputime("Krylov method totally", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_diag (dCSRmat *A, dvector *b, dvector *x,
 *                                       ITS_param *itparam)
 *
 * \brief Solve Ax=b by diagonal preconditioned Krylov methods
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang, Shiquan Zhang
 * \date   09/25/2009
 */
INT fasp_solver_dcsr_krylov_diag(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // setup preconditioner
    dvector diag;
    fasp_dcsr_getdiag(0, A, &diag);

    precond pc;
    pc.data = &diag;
    pc.fct  = fasp_precond_diag;

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        fasp_cputime("Diag_Krylov method totally", solve_end - solve_start);
    }

    fasp_dvec_free(&diag);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_swz (dCSRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, SWZ_param *schparam)
 *
 * \brief Solve Ax=b by overlapping Schwarz Krylov methods
 *
 * \param A        Pointer to the coeff matrix in dCSRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param itparam  Pointer to parameters for iterative solvers
 * \param schparam Pointer to parameters for Schwarz methods
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   03/21/2011
 *
 * Modified by Chensong on 07/02/2012: change interface
 */
INT fasp_solver_dcsr_krylov_swz(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam,
                                SWZ_param* schparam)
{
    SWZ_param swzparam;
    swzparam.SWZ_mmsize    = schparam->SWZ_mmsize;
    swzparam.SWZ_maxlvl    = schparam->SWZ_maxlvl;
    swzparam.SWZ_type      = schparam->SWZ_type;
    swzparam.SWZ_blksolver = schparam->SWZ_blksolver;

    const SHORT prtlvl = itparam->print_level;

    REAL setup_start, setup_end;
    REAL solve_start, solve_end;
    INT  status = FASP_SUCCESS;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);
    fasp_gettime(&setup_start);

    // setup preconditioner
    SWZ_data SWZ_data;

    // symmetrize the matrix (get rid of this later)
    SWZ_data.A = fasp_dcsr_sympart(A);

    // construct Schwarz precondtioner
    fasp_dcsr_shift(&SWZ_data.A, 1);
    fasp_swz_dcsr_setup(&SWZ_data, &swzparam);

    fasp_gettime(&setup_end);
    printf("SWZ_Krylov method setup %f seconds.\n", setup_end - setup_start);

    precond prec;
    prec.data = &SWZ_data;
    prec.fct  = fasp_precond_swz;

    // solver part
    status = fasp_solver_dcsr_itsolver(A, b, x, &prec, itparam);

    if (prtlvl > PRINT_NONE) {
        fasp_gettime(&solve_end);
        fasp_cputime("SWZ_Krylov method totally", solve_end - solve_start);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    fasp_swz_data_free(&SWZ_data);

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_amg (dCSRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, AMG_param *amgparam)
 *
 * \brief Solve Ax=b by AMG preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dCSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param amgparam  Pointer to parameters for AMG methods
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   09/25/2009
 */
INT fasp_solver_dcsr_krylov_amg(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam,
                                AMG_param* amgparam)
{
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;
    const INT   nnz = A->nnz, m = A->row, n = A->col;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;

#if MULTI_COLOR_ORDER
    A->color = 0;
    A->IC    = NULL;
    A->ICMAP = NULL;
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // initialize A, b, x for mgl[0]
    AMG_data* mgl = fasp_amg_data_create(max_levels);
    mgl[0].A      = fasp_dcsr_create(m, n, nnz);
    fasp_dcsr_cp(A, &mgl[0].A);
    mgl[0].b = fasp_dvec_create(n);
    mgl[0].x = fasp_dvec_create(n);

    // setup preconditioner
    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa(mgl, amgparam);
            break;

        case UA_AMG: // Unsmoothed Aggregation AMG
            status = fasp_amg_setup_ua(mgl, amgparam);
            break;

        default: // Classical AMG
            status = fasp_amg_setup_rs(mgl, amgparam);
    }

#if DEBUG_MODE > 1
    fasp_mem_usage();
#endif

    if (status < 0) goto FINISHED;

    // setup preconditioner
    precond_data pcdata;
    fasp_param_amg_to_prec(&pcdata, amgparam);
    pcdata.max_levels = mgl[0].num_levels;
    pcdata.mgl_data   = mgl;

    precond pc;
    pc.data = &pcdata;

    if (itparam->precond_type == PREC_FMG) {
        pc.fct = fasp_precond_famg; // Full AMG
    } else {
        switch (amgparam->cycle_type) {
            case AMLI_CYCLE: // AMLI cycle
                pc.fct = fasp_precond_amli;
                break;
            case NL_AMLI_CYCLE: // Nonlinear AMLI
                pc.fct = fasp_precond_namli;
                break;
            default: // V,W-cycles or hybrid cycles
                pc.fct = fasp_precond_amg;
        }
    }

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        fasp_cputime("AMG_Krylov method totally", solve_end - solve_start);
    }

FINISHED:
    fasp_amg_data_free(mgl, amgparam);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_ilu (dCSRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, ILU_param *iluparam)
 *
 * \brief Solve Ax=b by ILUs preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dCSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param iluparam  Pointer to parameters for ILU
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang, Shiquan Zhang
 * \date   09/25/2009
 */
INT fasp_solver_dcsr_krylov_ilu(dCSRmat* A, dvector* b, dvector* x, ITS_param* itparam,
                                ILU_param* iluparam)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end, solve_time;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // ILU setup for whole matrix
    ILU_data LU;
    if ((status = fasp_ilu_dcsr_setup(A, &LU, iluparam)) < 0) goto FINISHED;

    // check iludata
    if ((status = fasp_mem_iludata_check(&LU)) < 0) goto FINISHED;

    // set preconditioner
    precond pc;
    pc.data = &LU;
    pc.fct  = fasp_precond_ilu;

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        solve_time = solve_end - solve_start;

        switch (iluparam->ILU_type) {
            case ILUt:
                fasp_cputime("ILUt_Krylov method totally", solve_time);
                break;
            case ILUtp:
                fasp_cputime("ILUtp_Krylov method totally", solve_time);
                break;
            default: // ILUk
                fasp_cputime("ILUk_Krylov method totally", solve_time);
        }
    }

FINISHED:
    fasp_ilu_data_free(&LU);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_ilu_M (dCSRmat *A, dvector *b, dvector *x,
 *                                        ITS_param *itparam, ILU_param *iluparam,
 *                                        dCSRmat *M)
 *
 * \brief Solve Ax=b by ILUs preconditioned Krylov methods: ILU of M as preconditioner
 *
 * \param A         Pointer to the coeff matrix in dCSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param iluparam  Pointer to parameters for ILU
 * \param M         Pointer to the preconditioning matrix in dCSRmat format
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date 09/25/2009
 *
 * \note This function is specially designed for reservoir simulation.
 *       Have not been tested in any other places.
 */
INT fasp_solver_dcsr_krylov_ilu_M(dCSRmat* A, dvector* b, dvector* x,
                                  ITS_param* itparam, ILU_param* iluparam, dCSRmat* M)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    REAL solve_start, solve_end, solve_time;
    INT  status = FASP_SUCCESS;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // ILU setup for M
    ILU_data LU;
    if ((status = fasp_ilu_dcsr_setup(M, &LU, iluparam)) < 0) goto FINISHED;

    // check iludata
    if ((status = fasp_mem_iludata_check(&LU)) < 0) goto FINISHED;

    // set precondtioner
    precond pc;
    pc.data = &LU;
    pc.fct  = fasp_precond_ilu;

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        solve_time = solve_end - solve_start;

        switch (iluparam->ILU_type) {
            case ILUt:
                fasp_cputime("ILUt_Krylov method", solve_time);
                break;
            case ILUtp:
                fasp_cputime("ILUtp_Krylov method", solve_time);
                break;
            default: // ILUk
                fasp_cputime("ILUk_Krylov method", solve_time);
        }
    }

FINISHED:
    fasp_ilu_data_free(&LU);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dcsr_krylov_amg_nk (dCSRmat *A, dvector *b, dvector *x,
 *                                         ITS_param *itparam, AMG_param *amgparam,
 *                                         dCSRmat *A_nk, dCSRmat *P_nk, dCSRmat *R_nk)
 *
 * \brief Solve Ax=b by AMG preconditioned Krylov methods with an extra near kernel
 * solve
 *
 * \param A         Pointer to the coeff matrix in dCSRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param amgparam  Pointer to parameters for AMG methods
 * \param A_nk      Pointer to the coeff matrix of near kernel space in dCSRmat format
 * \param P_nk      Pointer to the prolongation of near kernel space in dCSRmat format
 * \param R_nk      Pointer to the restriction of near kernel space in dCSRmat format
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   05/26/2014
 */
INT fasp_solver_dcsr_krylov_amg_nk(dCSRmat* A, dvector* b, dvector* x,
                                   ITS_param* itparam, AMG_param* amgparam,
                                   dCSRmat* A_nk, dCSRmat* P_nk, dCSRmat* R_nk)
{
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;
    const INT   nnz = A->nnz, m = A->row, n = A->col;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end, solve_time;

#if MULTI_COLOR_ORDER
    A->color = 0;
    A->IC    = NULL;
    A->ICMAP = NULL;
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // initialize A, b, x for mgl[0]
    AMG_data* mgl = fasp_amg_data_create(max_levels);
    mgl[0].A      = fasp_dcsr_create(m, n, nnz);
    fasp_dcsr_cp(A, &mgl[0].A);
    mgl[0].b = fasp_dvec_create(n);
    mgl[0].x = fasp_dvec_create(n);

    // setup preconditioner
    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa(mgl, amgparam);
            break;

        case UA_AMG: // Unsmoothed Aggregation AMG
            status = fasp_amg_setup_ua(mgl, amgparam);
            break;

        default: // Classical AMG
            status = fasp_amg_setup_rs(mgl, amgparam);
    }

#if DEBUG_MODE > 1
    fasp_mem_usage();
#endif

    if (status < 0) goto FINISHED;

    // setup preconditioner
    precond_data pcdata;
    fasp_param_amg_to_prec(&pcdata, amgparam);
    pcdata.max_levels = mgl[0].num_levels;
    pcdata.mgl_data   = mgl;

    // near kernel space
#if WITH_UMFPACK // use UMFPACK directly
    dCSRmat A_tran;
    A_tran = fasp_dcsr_create(A_nk->row, A_nk->col, A_nk->nnz);
    fasp_dcsr_transz(A_nk, NULL, &A_tran);
    // It is equivalent to do transpose and then sort
    //     fasp_dcsr_trans(A_nk, &A_tran);
    //     fasp_dcsr_sort(&A_tran);
    pcdata.A_nk = &A_tran;
#else
    pcdata.A_nk = A_nk;
#endif

    pcdata.P_nk = P_nk;
    pcdata.R_nk = R_nk;

    precond pc;
    pc.data = &pcdata;
    pc.fct  = fasp_precond_amg_nk;

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        solve_time = solve_end - solve_start;
        fasp_cputime("AMG_NK_Krylov method", solve_time);
    }

FINISHED:
    fasp_amg_data_free(mgl, amgparam);

#if WITH_UMFPACK // use UMFPACK directly
    fasp_dcsr_free(&A_tran);
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
