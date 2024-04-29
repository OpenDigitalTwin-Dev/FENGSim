/*! \file  poisson-its.c
 *
 *  \brief The second test example for FASP: using ITS to solve 
 *         the discrete Poisson equation from P1 finite element.
 *
 *  \note  Solving the Poisson equation (P1 FEM) with iterative methods: C version
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for the second example.
 *
 * \author Feiteng Huang
 * \date   04/13/2012
 *
 * Modified by Chensong Zhang on 09/22/2012
 */
int main (int argc, const char * argv[]) 
{
    input_param     inparam;  // parameters from input files
    ITS_param       itparam;  // parameters for itsolver

    printf("\n========================================");
    printf("\n||   FASP: ITS example -- C version   ||");
    printf("\n========================================\n\n");
    
    // Step 0. Set parameters: We can use ini/its.dat
    fasp_param_set(argc, argv, &inparam);
    fasp_param_init(&inparam, &itparam, NULL, NULL, NULL);

    // Set local parameters
    const int print_level = inparam.print_level;
    
    // Step 1. Get stiffness matrix and right-hand side
    // Read A and b -- P1 FE discretization for Poisson. The location
    // of the data files is given in "ini/its.dat".
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
    
    // Step 2. Print problem size and ITS parameters
    if (print_level>PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        fasp_param_solver_print(&itparam);
    }
    
    // Step 3. Solve the system with ITS as an iterative solver
    // Set the initial guess to be zero and then solve it using standard
    // iterative methods, without applying any preconditioners
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);
    
    fasp_solver_dcsr_itsolver(&A, &b, &x, NULL, &itparam);
    
    // Step 4. Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
