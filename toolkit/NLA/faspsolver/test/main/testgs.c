/*! \file  testgs.c
 *
 *  \brief The test function for FASP GS smoother
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
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
 * \date   09/24/2017
 */
int main (int argc, const char * argv[]) 
{
    dCSRmat A;
    dvector b, x;
    int     status = FASP_SUCCESS;
    int     i;
    double  absres;
    
    //------------------------//
    // Step 0. Set parameters //
    //------------------------//
    input_param  inipar; // parameters from input files

    // Set solver parameters
    fasp_param_set(argc, argv, &inipar);

    // Set local parameters
    const int print_level   = inipar.print_level;
    const int problem_num   = inipar.problem_num;
    const int output_type   = inipar.output_type;

    // Set output device
    if (output_type) {
        char *outputfile = "out/test.out";
        printf("Redirecting outputs to file: %s ...\n", outputfile);
        if ( freopen(outputfile,"w",stdout) == NULL ) // open a file for stdout
        fprintf(stderr, "Output redirecting stdout\n");
    }
    
    printf("Test Problem %d\n", problem_num);
    
    //----------------------------------------------------//
    // Step 1. Input stiffness matrix and right-hand side //
    //----------------------------------------------------//
    char filename1[512], *datafile1;
    char filename2[512], *datafile2;
    
    memcpy(filename1,inipar.workdir,STRLEN);
    memcpy(filename2,inipar.workdir,STRLEN);
    
    if (problem_num == 10) {
        
        // Read A and b -- P1 FE discretization for Poisson.
        datafile1="csrmat_FE.dat";
        strcat(filename1,datafile1);
        
        datafile2="rhs_FE.dat";
        strcat(filename2,datafile2);
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);

    }
    
    else if (problem_num == 11) {

        // Read A and b -- P1 FE discretization for Poisson, 1M DoF
        datafile1="coomat_1046529.dat"; // This file is NOT in ../data!
        strcat(filename1,datafile1);
        fasp_dcoo_read(filename1, &A);
        
        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
        
    }
    
    else if (problem_num == 12) {

        // Read A and b -- FD discretization for Poisson, 1M DoF
        datafile1="csrmat_1023X1023.dat"; // This file is NOT in ../data!
        strcat(filename1,datafile1);
        
        datafile2="rhs_1023X1023.dat";
        strcat(filename2,datafile2);
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
        
    }

    else {
        printf("### ERROR: Unrecognised problem number %d\n", problem_num);
        return ERROR_INPUT_PAR;
    }
    
    // Print problem size
    if (print_level > PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
    }

    // Set initial guess
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row,&x,0.0);

    // Compute norm of residual
    absres = fasp_blas_dvec_norm2(&b); // residual ||r||
    printf("Initial residual in L2 norm is %e\n", absres);

    for ( i = 0; i < 5000; i++ ) {
        fasp_smoother_dcsr_gs(&x, 0, A.col-1, 1, &A, &b, 1);
    }

    // Compute norm of residual
    fasp_blas_dcsr_aAxpy(-1.0, &A, x.val, b.val);
    absres = fasp_blas_dvec_norm2(&b); // residual ||r||
    printf("Final residual in L2 norm is %e\n", absres);

    if (status<0) {
        printf("\n### ERROR: Solver failed! Exit status = %d.\n\n", status);
    }
    
    if (output_type) fclose (stdout);
    
    // Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
