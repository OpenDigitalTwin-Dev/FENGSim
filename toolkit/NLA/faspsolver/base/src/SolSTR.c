/*! \file  SolSTR.c
 *
 *  \brief Iterative solvers for dSTRmat matrices
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxTiming.c, AuxVector.c,
 *         BlaSmallMatInv.c, BlaILUSetupSTR.c, BlaSparseSTR.c, ItrSmootherSTR.c,
 *         KryPbcgs.c, KryPcg.c, KryPgmres.c, KryPvgmres.c, and PreSTR.c
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

#include "KryUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_dstr_itsolver (dSTRmat *A, dvector *b, dvector *x,
 *                                    precond *pc, ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods
 *
 * \param A        Pointer to the coeff matrix in dSTRmat format
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param pc       Pointer to the preconditioning action
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Iteration number if converges; ERROR otherwise.
 *
 * \author Chensong Zhang
 * \date   09/25/2009
 *
 * Modified by Chunsheng Feng on 03/04/2016: add VBiCGstab solver
 */
INT fasp_solver_dstr_itsolver(dSTRmat* A, dvector* b, dvector* x, precond* pc,
                              ITS_param* itparam)
{
    const SHORT prtlvl        = itparam->print_level;
    const SHORT itsolver_type = itparam->itsolver_type;
    const SHORT stop_type     = itparam->stop_type;
    const SHORT restart       = itparam->restart;
    const INT   MaxIt         = itparam->maxit;
    const REAL  tol           = itparam->tol;
    const REAL  abstol        = itparam->abstol;

    // local variables
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
            iter = fasp_solver_dstr_pcg(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                        prtlvl);
            break;

        case SOLVER_BiCGstab:
            iter = fasp_solver_dstr_pbcgs(A, b, x, pc, tol, abstol, MaxIt, stop_type,
                                          prtlvl);
            break;

        case SOLVER_GMRES:
            iter = fasp_solver_dstr_pgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
                                           stop_type, prtlvl);
            break;

        case SOLVER_VGMRES:
            iter = fasp_solver_dstr_pvgmres(A, b, x, pc, tol, abstol, MaxIt, restart,
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
 * \fn INT fasp_solver_dstr_krylov (dSTRmat *A, dvector *b, dvector *x,
 *                                  ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dSTRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Zhiyang Zhou
 * \date   04/25/2010
 */
INT fasp_solver_dstr_krylov(dSTRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;
    INT         status = FASP_SUCCESS;
    REAL        solve_start, solve_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // solver part
    fasp_gettime(&solve_start);

    status = fasp_solver_dstr_itsolver(A, b, x, NULL, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl >= PRINT_MIN)
        fasp_cputime("Krylov method totally", solve_end - solve_start);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dstr_krylov_diag (dSTRmat *A, dvector *b, dvector *x,
 *                                       ITS_param *itparam)
 *
 * \brief Solve Ax=b by diagonal preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dSTRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Zhiyang Zhou
 * \date   4/23/2010
 */
INT fasp_solver_dstr_krylov_diag(dSTRmat* A, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;
    const INT   ngrid  = A->ngrid;

    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;
    INT  nc = A->nc, nc2 = nc * nc, i;

    fasp_gettime(&solve_start);

    // setup preconditioner
    precond_diag_str diag;
    fasp_dvec_alloc(ngrid * nc2, &(diag.diag));
    fasp_darray_cp(ngrid * nc2, A->diag, diag.diag.val);

    diag.nc = nc;

    for (i = 0; i < ngrid; ++i) fasp_smat_inv(&(diag.diag.val[i * nc2]), nc);

    precond* pc = (precond*)fasp_mem_calloc(1, sizeof(precond));

    pc->data = &diag;
    pc->fct  = fasp_precond_dstr_diag;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // solver part
    status = fasp_solver_dstr_itsolver(A, b, x, pc, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl >= PRINT_MIN)
        fasp_cputime("Diag_Krylov method totally", solve_end - solve_start);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dstr_krylov_ilu (dSTRmat *A, dvector *b, dvector *x,
 *                                      ITS_param *itparam, ILU_param *iluparam)
 *
 * \brief Solve Ax=b by structured ILU preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dSTRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param iluparam  Pointer to parameters for ILU
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   05/01/2010
 */
INT fasp_solver_dstr_krylov_ilu(dSTRmat* A, dvector* b, dvector* x, ITS_param* itparam,
                                ILU_param* iluparam)
{
    const SHORT prtlvl   = itparam->print_level;
    const INT   ILU_lfil = iluparam->ILU_lfil;

    INT  status = FASP_SUCCESS;
    REAL setup_start, setup_end, setup_time;
    REAL solve_start, solve_end, solve_time;

    // set up
    dSTRmat LU;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    fasp_gettime(&setup_start);

    if (ILU_lfil == 0) {
        fasp_ilu_dstr_setup0(A, &LU);
    } else if (ILU_lfil == 1) {
        fasp_ilu_dstr_setup1(A, &LU);
    } else {
        printf("### ERROR: Illegal level of ILU fill-in (>1)! [%s]\n", __FUNCTION__);
        return ERROR_MISC;
    }

    fasp_gettime(&setup_end);

    setup_time = setup_end - setup_start;

    if (prtlvl > PRINT_NONE)
        printf("Structrued ILU(%d) setup costs %f seconds.\n", ILU_lfil, setup_time);

    precond pc;
    pc.data = &LU;
    if (ILU_lfil == 0) {
        pc.fct = fasp_precond_dstr_ilu0;
    } else if (ILU_lfil == 1) {
        pc.fct = fasp_precond_dstr_ilu1;
    } else {
        printf("### ERROR: Illegal level of ILU fill-in (>1)! [%s]\n", __FUNCTION__);
        return ERROR_MISC;
    }

    // solver part
    fasp_gettime(&solve_start);

    status = fasp_solver_dstr_itsolver(A, b, x, &pc, itparam);

    fasp_gettime(&solve_end);

    if (prtlvl >= PRINT_MIN) {
        solve_time = solve_end - solve_start;
        printf("Iterative solver costs %f seconds.\n", solve_time);
        fasp_cputime("ILU_Krylov method totally", setup_time + solve_time);
    }

    fasp_dstr_free(&LU);

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_solver_dstr_krylov_blockgs (dSTRmat *A, dvector *b, dvector *x,
 *                                          ITS_param *itparam, ivector *neigh,
 *                                          ivector *order)
 *
 * \brief Solve Ax=b by diagonal preconditioned Krylov methods
 *
 * \param A         Pointer to the coeff matrix in dSTRmat format
 * \param b         Pointer to the right hand side in dvector format
 * \param x         Pointer to the approx solution in dvector format
 * \param itparam   Pointer to parameters for iterative solvers
 * \param neigh     Pointer to neighbor vector
 * \param order     Pointer to solver ordering
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Xiaozhe Hu
 * \date   10/10/2010
 */
INT fasp_solver_dstr_krylov_blockgs(dSTRmat* A, dvector* b, dvector* x,
                                    ITS_param* itparam, ivector* neigh, ivector* order)
{
    // Parameter for iterative method
    const SHORT prtlvl = itparam->print_level;

    // Information about matrices
    INT ngrid = A->ngrid;

    // return parameter
    INT status = FASP_SUCCESS;

    // local parameter
    REAL setup_start, setup_end, setup_time = 0;
    REAL solve_start, solve_end, solve_time = 0;

    dvector* diaginv;
    ivector* pivot;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // setup preconditioner
    fasp_gettime(&setup_start);

    diaginv = (dvector*)fasp_mem_calloc(ngrid, sizeof(dvector));
    pivot   = (ivector*)fasp_mem_calloc(ngrid, sizeof(ivector));
    fasp_generate_diaginv_block(A, neigh, diaginv, pivot);

    precond_data_str pcdata;
    pcdata.A_str   = A;
    pcdata.diaginv = diaginv;
    pcdata.pivot   = pivot;
    pcdata.order   = order;
    pcdata.neigh   = neigh;

    precond pc;
    pc.data = &pcdata;
    pc.fct  = fasp_precond_dstr_blockgs;

    fasp_gettime(&setup_end);

    if (prtlvl > PRINT_NONE) {
        setup_time = setup_end - setup_start;
        printf("Preconditioner setup costs %f seconds.\n", setup_time);
    }

    // solver part
    fasp_gettime(&solve_start);

    status = fasp_solver_dstr_itsolver(A, b, x, &pc, itparam);

    fasp_gettime(&solve_end);
    solve_time = solve_end - solve_start;

    if (prtlvl >= PRINT_MIN) {
        fasp_cputime("Iterative solver", solve_time);
        fasp_cputime("BlockGS_Krylov method totally", setup_time + solve_time);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
