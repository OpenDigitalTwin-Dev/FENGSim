/*! \file  testkrylov.c
 *
 *  \brief The test function for Krylov solvers
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp_functs.h"
#include <stdio.h>
#include <time.h>

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function to test speed of Krylov solvers.
 *
 * \author Kailei Zhang, Chensong Zhang
 * \date   06/30/2020
 */
int main(int argc, char** argv)
{
    int     j;
    dCSRmat A;

    fasp_dmtx_read("../../data/Hardesty1.mtx", &A);
    // fasp_dcsr_read("../../data/fdm_1023X1023.csr", &A);

    INT     n = A.row;
    dvector b, x;

    b.row = n;
    x.row = n;

    b.val = (REAL*)malloc(n * sizeof(REAL));
    for (j = 0; j < n; ++j) b.val[j] = 1.0;

    const REAL  tol     = 1e-5;
    const REAL  abstol  = 1e-18;
    const INT   maxit   = 500;
    const INT   restart = 10;
    const SHORT prtlvl  = PRINT_MIN;
    REAL        start;

    x.val = (REAL*)malloc(n * sizeof(REAL)); // solution vector

    printf("Test CG method ...\n");
    for (j = 0; j < n; ++j) x.val[j] = 1.0; // initial guess
    start = clock();
    fasp_solver_dcsr_pcg(&A, &b, &x, NULL, tol, abstol, maxit, 1, prtlvl);
    printf("CPU Time: %lf seconds\n", (clock() - start) / CLOCKS_PER_SEC);

    printf("Test GMRES method ...\n");
    for (j = 0; j < n; ++j) x.val[j] = 1.0; // initial guess
    start = clock();
    fasp_solver_dcsr_pgmres(&A, &b, &x, NULL, tol, abstol, maxit, restart, 1, prtlvl);
    printf("CPU Time: %lf seconds\n", (clock() - start) / CLOCKS_PER_SEC);

    printf("Test vGMRES method ...\n");
    for (j = 0; j < n; ++j) x.val[j] = 1.0; // initial guess
    start = clock();
    fasp_solver_dcsr_pvgmres(&A, &b, &x, NULL, tol, abstol, maxit, restart, 1, prtlvl);
    printf("CPU Time: %lf seconds\n", (clock() - start) / CLOCKS_PER_SEC);

    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
