/*! \file  XtrSuperlu.c
 *
 *  \brief Interface to SuperLU direct solvers
 *
 *  Reference for SuperLU:
 *  http://crd-legacy.lbl.gov/~xiaoye/SuperLU/
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

#if WITH_SuperLU
#include "slu_ddefs.h"
#endif

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn int fasp_solver_superlu (dCSRmat *ptrA, dvector *b, dvector *u,
 *                              const SHORT prtlvl)
 *
 * \brief Solve Au=b by SuperLU
 *
 * \param ptrA      Pointer to a dCSRmat matrix
 * \param b         Pointer to the dvector of right-hand side term
 * \param u         Pointer to the dvector of solution
 * \param prtlvl    Output level
 *
 * \author Xiaozhe Hu
 * \date   11/05/2009
 *
 * Modified by Chensong Zhang on 02/27/2013 for new FASP function names.
 *
 * \note  Factorization and solution are combined together!!! Not efficient!!!
 */
int fasp_solver_superlu(dCSRmat* ptrA, dvector* b, dvector* u, const SHORT prtlvl)
{

#if WITH_SuperLU

    SuperMatrix A, L, U, B;

    int* perm_r; /* row permutations from partial pivoting */
    int* perm_c; /* column permutation vector */
    int  nrhs = 1, info, m = ptrA->row, n = ptrA->col, nnz = ptrA->nnz;

    if (prtlvl > PRINT_NONE) printf("superlu: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);

    REAL start_time, end_time;
    fasp_gettime(&start_time);
    
    dCSRmat tempA = fasp_dcsr_create(m, n, nnz);
    fasp_dcsr_cp(ptrA, &tempA);

    dvector tempb = fasp_dvec_create(n);
    fasp_dvec_cp(b, &tempb);

    /* Create matrix A in the format expected by SuperLU. */
    dCreate_CompCol_Matrix(&A, m, n, nnz, tempA.val, tempA.JA, tempA.IA, SLU_NR, SLU_D,
                           SLU_GE);

    /* Create right-hand side B. */
    dCreate_Dense_Matrix(&B, m, nrhs, tempb.val, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m))) ABORT("Malloc fails for perm_r[].");
    if (!(perm_c = intMalloc(n))) ABORT("Malloc fails for perm_c[].");

    /* Set the default input options. */
    superlu_options_t options;
    set_default_options(&options);
    options.ColPerm = COLAMD; // MMD_AT_PLUS_A; MMD_ATA; NATURAL;

    /* Initialize the statistics variables. */
    SuperLUStat_t stat;
    StatInit(&stat);

    /* SuperLU */
    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

    DNformat* BB = (DNformat*)B.Store;
    u->val       = (double*)BB->nzval;
    u->row       = n;

    if (prtlvl > PRINT_MIN) {
        fasp_gettime(&end_time);
        fasp_cputime("SUPERLU solver", end_time - start_time);
    }

    /* De-allocate storage */
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(perm_c);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
    StatFree(&stat);

    return FASP_SUCCESS;

#else

    printf("### ERROR: SuperLU is not available!\n");
    return ERROR_SOLVER_EXIT;

#endif
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
