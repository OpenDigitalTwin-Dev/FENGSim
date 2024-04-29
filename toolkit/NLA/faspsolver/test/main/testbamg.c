/*! \file  testbamg.c
 *
 *  \brief The main test function for FASP Block AMG solvers
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2013--Present by the FASP team. All rights reserved.
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
 * \author Xiaozhe Hu
 * \date   01/01/2013
 * 
 * Modified by Chensong Zhang on 01/24/2013
 * Modified by Chensong Zhang on 10/10/2015: reformat a little bit
 */
int main (int argc, const char * argv[]) 
{
    dBSRmat Absr;
    dvector b, uh;
    
    int status = FASP_SUCCESS;
    
    // Step 0. Set parameters
    input_param   inpar;  // parameters from input files
    ITS_param     itpar;  // parameters for itsolver
    AMG_param     amgpar; // parameters for AMG
    ILU_param     ilupar; // parameters for ILU
    
    // Set solver parameters: use ./ini/bamg.dat
    fasp_param_set(argc, argv, &inpar);
    fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);

    // Set local parameters
    const int print_level   = inpar.print_level;
    const int problem_num   = inpar.problem_num;
    const int itsolver_type = inpar.solver_type;
    const int precond_type  = inpar.precond_type;
    const int output_type   = inpar.output_type;
    
    // Set output device
    if (output_type) {
        char *outputfile = "out/test.out";
        printf("Redirecting outputs to file: %s ...\n", outputfile);
        if ( freopen(outputfile,"w",stdout) == NULL ) // open a file for stdout
            fprintf(stderr, "Output redirecting stdout\n");
    }
    
    printf("Test Problem %d\n", problem_num);
    
    // Step 1. Input stiffness matrix and right-hand side
    char filename1[512], *datafile1;
    char filename2[512], *datafile2;
    
    memcpy(filename1,inpar.workdir,STRLEN);
    memcpy(filename2,inpar.workdir,STRLEN);
    
    // Default test problem from black-oil benchmark: SPE01
    if (problem_num == 10) {                
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);    
        datafile1="bsrmat_SPE01.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="rhs_SPE01.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
    }
    
    // Test problem 1
    else if (problem_num == 11) {               
        dCSRmat A;

        datafile1="PNNL/test1/A.dat";
        strcat(filename1,datafile1);
        datafile2="PNNL/test1/b.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        
        Absr = fasp_format_dcsr_dbsr(&A, 6);        

        fasp_dcsr_free(&A);
    }   
    
    // Test problem 2
    else if (problem_num == 12) {
        dCSRmat A;
        
        datafile1="PNNL/test2/A.dat";
        strcat(filename1,datafile1);
        datafile2="PNNL/test3/b.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 6);        
        
#if 0
        // Form the right-hand-side b = A*sol
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_set(A.row,&sol,1);
        fasp_dvec_rand(A.row, &sol);
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);             
        fasp_dvec_free(&sol);
#endif 

        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    // Test problem 3
    else if (problem_num == 13) {               
        dCSRmat A;

        datafile1="PNNL/test3/A.dat";
        strcat(filename1,datafile1);
        datafile2="PNNL/test3/b.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 6);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }   
    
    // Erhan problem 
    else if (problem_num == 20) {
        dCSRmat A;
        
        datafile1="Erhan/erhan_A1.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/erhan_b1.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 21) {
        dCSRmat A;
        
        datafile1="Erhan/erhan_A2.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/erhan_b2.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 22) {
        dCSRmat A;
        
        datafile1="Erhan/erhan_A3.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/erhan_b3.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 23) {
        dCSRmat A;
        
        datafile1="Erhan/new/1/erhan1.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/1/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 24) {
        dCSRmat A;
        
        datafile1="Erhan/new/2/erhan2.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/2/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 25) {
        dCSRmat A;
        
        datafile1="Erhan/new/3/erhan3.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/3/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 26) {
        dCSRmat A;
        
        datafile1="Erhan/new/4/erhan4.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/4/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 27) {
        dCSRmat A;
        
        datafile1="Erhan/new/5/erhan5.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/5/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 28) {
        dCSRmat A;
        
        datafile1="Erhan/new/6/erhan6.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/6/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else if (problem_num == 29) {
        dCSRmat A;
        
        datafile1="Erhan/new/7/erhan7.dat";
        strcat(filename1,datafile1);
        datafile2="Erhan/new/7/LB.txt";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        Absr = fasp_format_dcsr_dbsr(&A, 4);
        
        fasp_dcsr_free(&A); // free up temp matrix
    }
    
    else {
        printf("### ERROR: Unrecognised problem number %d\n", problem_num);
        return ERROR_INPUT_PAR;
    }
    
    // Print problem size
    if (print_level>PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", Absr.ROW, Absr.COL, Absr.NNZ);
        printf("b: n = %d\n", b.row);
    }
    
    // Step 2. Solve the system
    
    // Print out solver parameters
    if (print_level>PRINT_NONE) fasp_param_solver_print(&itpar);
    
    // Set initial guess
    fasp_dvec_alloc(b.row, &uh); 
    fasp_dvec_set(b.row, &uh, 0.0);
    
    // Preconditioned Krylov methods
    if ( itsolver_type > 0 && itsolver_type < 20 ) {
        
        // Using no preconditioner for Krylov iterative methods
        if (precond_type == PREC_NULL) {
            status = fasp_solver_dbsr_krylov(&Absr, &b, &uh, &itpar);
        }   
        
        // Using diag(A) as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_DIAG) {
            status = fasp_solver_dbsr_krylov_diag(&Absr, &b, &uh, &itpar);
        }
        
        // Using AMG as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_AMG || precond_type == PREC_FMG) {
            if (print_level>PRINT_NONE) fasp_param_amg_print(&amgpar);
            status = fasp_solver_dbsr_krylov_amg(&Absr, &b, &uh, &itpar, &amgpar); 
        }
        
        // Using ILU as preconditioner for Krylov iterative methods Q: Need to change!
        else if (precond_type == PREC_ILU) {
            if (print_level>PRINT_NONE) fasp_param_ilu_print(&ilupar);
            status = fasp_solver_dbsr_krylov_ilu(&Absr, &b, &uh, &itpar, &ilupar);
        }
        
        else {
            printf("### ERROR: Unknown preconditioner type %d!!!\n", precond_type);       
            exit(ERROR_SOLVER_PRECTYPE);
        }
        
    }
    
    else {
        printf("### ERROR: Unknown solver type %d!!!\n", itsolver_type);      
        status = ERROR_SOLVER_TYPE;
        goto FINISHED;
    }
    
    if (status<0) {
        printf("\n### ERROR: Solver failed! Exit status = %d.\n\n", status);
    }
    
    if (output_type) fclose (stdout);
    
 FINISHED:
    // Clean up memory
    fasp_dbsr_free(&Absr);
    fasp_dvec_free(&b);
    fasp_dvec_free(&uh);
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
