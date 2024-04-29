/*! \file  XtrStrumpack.c
 *
 *  \brief Interface to STRUMPACK direct solvers
 *
 *  Reference for STRUMPACK:
 *  https://portal.nersc.gov/project/sparse/strumpack/
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2022--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

#if WITH_STRUMPACK
#include "StrumpackSparseSolver.h"
#endif

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_strumpack (dCSRmat *ptrA, dvector *b, dvector *u,
 *                                const SHORT prtlvl)
 *
 * \brief Solve Au=b by UMFpack
 *
 * \param ptrA         Pointer to a dCSRmat matrix
 * \param b            Pointer to the dvector of right-hand side term
 * \param u            Pointer to the dvector of solution
 * \param prtlvl       Output level
 *
 * \author Chensong Zhang
 * \date   10/03/2022
 */
INT fasp_solver_strumpack(dCSRmat* ptrA, dvector* b, dvector* u, const SHORT prtlvl)
{

#if WITH_STRUMPACK

    const INT n       = ptrA->col;
    INT*      row_ptr = ptrA->IA;
    INT*      col_ind = ptrA->JA;
    double*   val     = ptrA->val;
    INT       status  = FASP_SUCCESS;

    int verbose = 0;
    if (prtlvl > PRINT_MORE) verbose = 1;

#if DEBUG_MODE
    const INT m   = ptrA->row;
    const INT nnz = ptrA->nnz;
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    REAL start_time, end_time;
    fasp_gettime(&start_time);

    /* Set Strumpack */
    STRUMPACK_SparseSolver S;
    STRUMPACK_init_mt(&S, STRUMPACK_DOUBLE, STRUMPACK_MT, 0, NULL, verbose);
    STRUMPACK_set_Krylov_solver(S, STRUMPACK_DIRECT);
    STRUMPACK_set_matching(S, STRUMPACK_MATCHING_NONE);
    STRUMPACK_set_reordering_method(S, STRUMPACK_METIS);
    STRUMPACK_set_compression(S, STRUMPACK_NONE);
    STRUMPACK_set_csr_matrix(S, &n, row_ptr, col_ind, val, 0);

    /* Call STRUMPACK as a solver */
    status = STRUMPACK_reorder(S);
    // status = STRUMPACK_factor(S);
    status = STRUMPACK_solve(S, b->val, u->val, 0);

    if (prtlvl > PRINT_MIN) {
        fasp_gettime(&end_time);
        fasp_cputime("STRUMPACK costs", end_time - start_time);
    }

    /* Clean up factorization */
    STRUMPACK_destroy(&S);
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return status;

#else

    printf("### ERROR: STRUMPACK is not available!\n");
    return ERROR_SOLVER_EXIT;

#endif
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
