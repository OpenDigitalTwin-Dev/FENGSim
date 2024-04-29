/*! \file  testbsr.c
 *
 *  \brief The main test function for FASP solvers -- BSR format
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

#include "math.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for a few simple tests.
 *
 * \author Chensong Zhang
 * \date   03/31/2009
 * 
 * Modified by Chensong Zhang on 09/09/2011
 */
int main (int argc, const char * argv[]) 
{
    dBSRmat Absr;
    dvector b, uh;

    int status = FASP_SUCCESS;
    
    // Step 0. Set parameters
    input_param  inpar;  // parameters from input files
    ITS_param    itpar;  // parameters for itsolver
    AMG_param    amgpar; // parameters for AMG
    ILU_param    ilupar; // parameters for ILU
    
    // Set solver parameters: Should use ./ini/bsr.dat
    fasp_param_set(argc, argv, &inpar);
    fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
    
    // Set local parameters
    const int print_level   = inpar.print_level;
    const int problem_num   = inpar.problem_num;
    const int itsolver_type = inpar.solver_type;
    const int precond_type  = inpar.precond_type;
    const int output_type   = inpar.output_type;
    
    // Set output devices
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
    
    else if (problem_num == 20) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_fasp_bsr.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b_fasp.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
    }
    
    else if (problem_num == 21) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_fasp_bsr_1.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b_fasp_1.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
    }
    
    else if (problem_num == 31) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_bsr004.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b004.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
    }
    
    else if (problem_num == 32) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_bsr_zy.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b_bsr_zy.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
    }

    else if (problem_num == 33) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_bsr_spe9.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b_bsr_spe9.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
    }
    
    else if (problem_num == 40) {
        
        // Read the stiffness matrix from bsrmat_SPE01.dat
        memcpy(filename1,inpar.workdir,STRLEN);
        datafile1="A_fasp_bsr.dat"; strcat(filename1,datafile1);
        fasp_dbsr_read(filename1, &Absr);
        
        // Read the RHS from rhs_SPE01.dat
        memcpy(filename2,inpar.workdir,STRLEN);
        datafile2="b_fasp.dat"; strcat(filename2,datafile2);
        fasp_dvec_read(filename2, &b);
        
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
