/*! \file  SolMatFree.c
 *
 *  \brief Iterative solvers using MatFree spmv operations
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxMessage.c, AuxTiming.c, BlaSpmvBLC.c, BlaSpmvBSR.c, BlaSpmvCSR.c,
 *         BlaSpmvCSRL.c, BlaSpmvSTR.c, KryPbcgs.c, KryPcg.c, KryPgcg.c,
 *         KryPgmres.c, KryPminres.c, KryPvfgmres.c, and KryPvgmres.c
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
#include "fasp_block.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "BlaSpmvMatFree.inl"
#include "KryUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_itsolver (mxv_matfree *mf, dvector *b, dvector *x,
 *                               precond *pc, ITS_param *itparam)
 *
 * \brief Solve Ax=b by preconditioned Krylov methods for CSR matrices
 *
 * \note This is an abstract interface for iterative methods.
 *
 * \param mf       Pointer to mxv_matfree MatFree spmv operation
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
 * Modified by Feiteng Huang on 09/19/2012: matrix free
 */
INT fasp_solver_itsolver(mxv_matfree* mf, dvector* b, dvector* x, precond* pc,
                         ITS_param* itparam)
{
    const SHORT prtlvl        = itparam->print_level;
    const SHORT itsolver_type = itparam->itsolver_type;
    const SHORT stop_type     = itparam->stop_type;
    const INT   restart       = itparam->restart;
    const INT   MaxIt         = itparam->maxit;
    const REAL  tol           = itparam->tol;
    const REAL  abstol        = itparam->abstol;

    /* Local Variables */
    REAL solve_start, solve_end, solve_time;
    INT  iter = ERROR_SOLVER_TYPE;

    fasp_gettime(&solve_start);

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    /* Safe-guard checks on parameters */
    ITS_CHECK(MaxIt, tol);

    /* Choose a desirable Krylov iterative solver */
    switch (itsolver_type) {

        case SOLVER_CG:
            iter = fasp_solver_pcg(mf, b, x, pc, tol, abstol, MaxIt, stop_type, prtlvl);
            break;

        case SOLVER_BiCGstab:
            iter =
                fasp_solver_pbcgs(mf, b, x, pc, tol, abstol, MaxIt, stop_type, prtlvl);
            break;

        case SOLVER_MinRes:
            iter = fasp_solver_pminres(mf, b, x, pc, tol, abstol, MaxIt, stop_type,
                                       prtlvl);
            break;

        case SOLVER_GMRES:
            iter = fasp_solver_pgmres(mf, b, x, pc, tol, abstol, MaxIt, restart,
                                      stop_type, prtlvl);
            break;

        case SOLVER_VGMRES:
            iter = fasp_solver_pvgmres(mf, b, x, pc, tol, abstol, MaxIt, restart,
                                       stop_type, prtlvl);
            break;

        case SOLVER_VFGMRES:
            iter = fasp_solver_pvfgmres(mf, b, x, pc, tol, abstol, MaxIt, restart,
                                        stop_type, prtlvl);
            break;

        case SOLVER_GCG:
            iter =
                fasp_solver_pgcg(mf, b, x, pc, tol, abstol, MaxIt, stop_type, prtlvl);
            break;

        default:
            printf("### ERROR: Unknown iterative solver type %d! [%s]\n", itsolver_type,
                   __FUNCTION__);
            return ERROR_SOLVER_TYPE;
    }

    if ((prtlvl >= PRINT_SOME) && (iter >= 0)) {
        fasp_gettime(&solve_end);
        solve_time = solve_end - solve_start;
        fasp_cputime("Iterative method", solve_time);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return iter;
}

/**
 * \fn INT fasp_solver_krylov (mxv_matfree *mf, dvector *b, dvector *x,
 *                             ITS_param *itparam)
 *
 * \brief Solve Ax=b by standard Krylov methods -- without preconditioner
 *
 * \param mf       Pointer to mxv_matfree MatFree spmv operation
 * \param b        Pointer to the right hand side in dvector format
 * \param x        Pointer to the approx solution in dvector format
 * \param itparam  Pointer to parameters for iterative solvers
 *
 * \return         Number of iterations if succeed
 *
 * \author Chensong Zhang, Shiquan Zhang
 * \date   09/25/2009
 *
 * Modified by Feiteng Huang on 09/20/2012: matrix free
 */
INT fasp_solver_krylov(mxv_matfree* mf, dvector* b, dvector* x, ITS_param* itparam)
{
    const SHORT prtlvl = itparam->print_level;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end, solve_time;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    status = fasp_solver_itsolver(mf, b, x, NULL, itparam);

    if (prtlvl >= PRINT_MIN) {
        fasp_gettime(&solve_end);
        solve_time = solve_end - solve_start;
        fasp_cputime("Krylov method totally", solve_time);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn void fasp_solver_matfree_init (INT matrix_format, mxv_matfree *mf, void *A)
 *
 * \brief Initialize MatFree (or non-specified format) itsovlers
 *
 * \param matrix_format    matrix format
 * \param mf               Pointer to mxv_matfree MatFree spmv operation
 * \param A                void pointer to the coefficient matrix
 *
 * \author Feiteng Huang
 * \date   09/18/2012
 *
 * Modified by Chensong Zhang on 05/10/2013: Change interface of mat-free mv
 */
void fasp_solver_matfree_init(INT matrix_format, mxv_matfree* mf, void* A)
{
    switch (matrix_format) {

        case MAT_CSR:
            mf->fct = fasp_blas_mxv_csr;
            break;

        case MAT_BSR:
            mf->fct = fasp_blas_mxv_bsr;
            break;

        case MAT_STR:
            mf->fct = fasp_blas_mxv_str;
            break;

        case MAT_BLC:
            mf->fct = fasp_blas_mxv_blc;
            break;

        case MAT_CSRL:
            mf->fct = fasp_blas_mxv_csrl;
            break;

        default:
            printf("### ERROR: Unknown matrix format %d!\n", matrix_format);
            exit(ERROR_DATA_STRUCTURE);
    }

    mf->data = A;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
