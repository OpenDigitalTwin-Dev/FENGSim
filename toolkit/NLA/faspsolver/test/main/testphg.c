/*! \file  testphg.c
 *
 *  \brief Test FASP solvers for PHG
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2021--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for a few simple tests.
 *
 * \author Chensong Zhang
 * \date   03/22/2021
 */
int main (int argc, const char * argv[]) 
{
    int status = FASP_SUCCESS;
        
    //---------------------------------------------------//
    // Step 1. Read stiffness matrix and right-hand side //
    //---------------------------------------------------//

    // Read A and b -- P1 FE discretization for Poisson.
    const char *datafile1 = "../data/csrmat_FE.dat";
    const char *datafile2 = "../data/rhs_FE.dat";
    dCSRmat A;
    dvector b, x;
    fasp_dcsrvec_read2(datafile1, datafile2, &A, &b);

    // Print problem size
    printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
    printf("b: n = %d\n", b.row);

    //--------------------------//
    // Step 2. Solve the system //
    //--------------------------//

    // Set solver parameters
    ITS_param itspar; // parameters for itsolver
    AMG_param amgpar; // parameters for AMG

    // Initialize solver parameters
    fasp_param_amg_init(&amgpar);
    fasp_param_solver_init(&itspar);

    // Set any parameters locally, for example:
    itspar.print_level = 3;

    // Print out solver parameters
    fasp_param_solver_print(&itspar);
    fasp_param_amg_print(&amgpar);

    // Set initial guess
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);

    // Preconditioned Krylov methods
    status = fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itspar, &amgpar);
    if (status<0) {
        printf("\n### ERROR: Solver failed! Exit status = %d.\n\n", status);
    }
        
    //--------------------------//
    // Step 3. Clean up memory  //
    //--------------------------//
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/