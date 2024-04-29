/*! \file  poisson-amg.c
 *
 *  \brief The first test example for FASP: using AMG to solve 
 *         the discrete Poisson equation from P1 finite element.
 *         C version.
 *
 *  \note  Solving the Poisson equation (P1 FEM) with AMG: C version
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2011--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for the first example.
 *
 * \author Chensong Zhang
 * \date   12/21/2011
 * 
 * Modified by Chensong Zhang on 09/22/2012
 */
int main (int argc, const char * argv[]) 
{
    input_param     inparam;  // parameters from input files
    AMG_param       amgparam; // parameters for AMG
    
    printf("\n========================================");
    printf("\n||   FASP: AMG example -- C version   ||");
    printf("\n========================================\n\n");

    // Step 0. Set parameters: We can use ini/amg.dat
    fasp_param_set(argc, argv, &inparam);
    fasp_param_init(&inparam, NULL, &amgparam, NULL, NULL);
    
    // Set local parameters using the input values
    const int print_level = inparam.print_level;
    
    // Step 1. Get stiffness matrix and right-hand side
    // Read A and b -- P1 FE discretization for Poisson. The location
    // of the data files is given in "ini/amg.dat".
    dCSRmat A;
    dvector b, x;
    char filename1[512], *datafile1;
    char filename2[512], *datafile2;
    
    // Read the stiffness matrix from matFE.dat
    memcpy(filename1, inparam.workdir, STRLEN);
    datafile1="csrmat_FE.dat"; strcat(filename1, datafile1);
    
    // Read the RHS from rhsFE.dat
    memcpy(filename2, inparam.workdir, STRLEN);
    datafile2="rhs_FE.dat"; strcat(filename2, datafile2);
    
    fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    
    // Step 2. Print problem size and AMG parameters
    if (print_level>PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        fasp_param_amg_print(&amgparam);
    }
    
    // Step 3. Solve the system with AMG as an iterative solver
    // Set the initial guess to be zero and then solve it
    // with AMG method as an iterative procedure
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);
    
    fasp_solver_amg(&A, &b, &x, &amgparam);
    
    // Step 4. Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
