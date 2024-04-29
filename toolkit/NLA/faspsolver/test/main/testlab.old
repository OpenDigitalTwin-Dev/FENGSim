/*! \file  testlab.c
 *
 *  \brief The experimental test function for FASP solvers
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2013--2017 by the FASP team. All rights reserved.
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
 * \date   12/29/2013 
 */
int main (int argc, const char * argv[]) 
{
    dCSRmat A;
    dvector b, x;
    int status = FASP_SUCCESS;
    
    //------------------------//
    // Step 0. Set parameters //
    //------------------------//
    input_param  inpar;  // parameters from input files
    ITS_param    itpar;  // parameters for itsolver
    AMG_param    amgpar; // parameters for AMG
    ILU_param    ilupar; // parameters for ILU
    SWZ_param    swzpar; // parameters for Schwarz method
    
    // Set solver parameters
    fasp_param_set(argc, argv, &inpar);
    fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, &swzpar);
    
    // Set local parameters
    const int print_level   = inpar.print_level;
    const int problem_num   = inpar.problem_num;
    const int solver_type   = inpar.solver_type;
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
    
    //----------------------------------------------------//
    // Step 1. Input stiffness matrix and right-hand side //
    //----------------------------------------------------//
    char filename1[512], *datafile1;
    char filename2[512], *datafile2;
    
    strncpy(filename1,inpar.workdir,128);
    strncpy(filename2,inpar.workdir,128);
    
    // Read A and b -- P1 FE discretization for Poisson.
    if (problem_num == 10) {                
        datafile1="csrmat_FE.dat";
        strcat(filename1,datafile1);
        
        datafile2="rhs_FE.dat";
        strcat(filename2,datafile2);        
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    }   
    
    // Read A and b -- P1 FE discretization for Poisson, 1M DoF    
    else if (problem_num == 11) {
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
    
    // Read A and b -- FD discretization for Poisson, 1M DoF    
    else if (problem_num == 12) {
        datafile1="csrmat_1023X1023.dat"; // This file is NOT in ../data!
        strcat(filename1,datafile1);
        
        datafile2="rhs_1023X1023.dat";
        strcat(filename2,datafile2);
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    }
    
    else if (problem_num == 20){
        
        FILE *fid;
        int i;
        
        fid = fopen("../data/PowerGrid/matrix.bin", "r");
        if ( fid==NULL ) {
            printf("### ERROR: Opening file ...\n");
        }
        
        status = fread(&A.row, sizeof(int), 1, fid);
        A.col = A.row;
        b.row = A.row;
        
        status = fread(&A.nnz, sizeof(int), 1, fid);
        
        A.val = (double *)fasp_mem_calloc(A.nnz, sizeof(double));
        for (i=0;i<A.nnz; i++){
            status = fread(&A.val[i], sizeof(double), 1, fid);
        }
        
        b.val = (double *)fasp_mem_calloc(b.row, sizeof(double));
        for (i=0;i<b.row; i++){
            status = fread(&b.val[i], sizeof(double), 1, fid);
        }
        
        A.IA = (int *)fasp_mem_calloc(A.row+1, sizeof(int));
        for (i=0; i<A.row+1; i++){
            status = fread(&A.IA[i], sizeof(int), 1, fid);
            A.IA[i] = A.IA[i] - 1;
        }
        
        A.JA = (int *)fasp_mem_calloc(A.nnz, sizeof(int));
        for (i=0; i<A.nnz; i++){
            status = fread(&A.JA[i], sizeof(int), 1, fid);
            A.JA[i] = A.JA[i] - 1;
        }
        
        fclose(fid);
        
    }
    
    else if (problem_num == 30) {
        datafile1="Pan_mat_small.dat";
        strcat(filename1,datafile1);
        
        datafile2="Pan_rhs_small.dat";
        strcat(filename2,datafile2);
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    }
    
    else if (problem_num == 31) {
        datafile1="Pan_mat_big.dat";
        strcat(filename1,datafile1);
        
        datafile2="Pan_rhs_big.dat";
        strcat(filename2,datafile2);
        
        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    }
    
    else if (problem_num == 32) {
        datafile1="Pan_mech_mat_1.dat";
        strcat(filename1,datafile1);
        
        datafile2="Pan_mech_rhs_1.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
        
        //fasp_dcsrvec_read2(filename1, filename2, &A, &b);
        
        //fasp_dcoo_write("Pan_mech_mat_coo.dat", &A);
    }
    
    else if (problem_num == 40 ) {
        datafile1="JumpData/mat128_p4_k8.dat";
        strcat(filename1,datafile1);
        
        datafile2="JumpData/rhs128_p4_k8.dat";
        strcat(filename2,datafile2);
        
        fasp_dcoo_read(filename1, &A);
        fasp_dvec_read(filename2, &b);
    }
    
    else if (problem_num == 50){
        
        datafile1="spe10-uncong/SPE10120.amg";
        strcat(filename1,datafile1);
    
        datafile2="spe10-uncong/SPE10120.rhs";
        strcat(filename2,datafile2);
        
        fasp_matrix_read_bin(filename1, &A);
        //fasp_dvec_alloc(A.row, &b);
        //fasp_dvec_set(A.row,&b,1.0);
        fasp_dvec_read(filename2, &b);
    }
    
    //
    else if (problem_num == 60) {
        
        // read in matrix
        datafile1="/pnp-data/set-2/A.dat";
        strcat(filename1,datafile1);
        
        fasp_dcoo_read(filename1, &A);
        
        // read in b
        datafile2="/pnp-data/set-2/b.dat";
        strcat(filename2,datafile2);
        
        fasp_dvec_read(filename2, &b);
    }

    else {
        printf("### ERROR: Unrecognised problem number %d\n", problem_num);
        return ERROR_INPUT_PAR;
    }
    
    // Print problem size
    if (print_level>PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        fasp_mem_usage();
    }
    
    // Print out solver parameters
    if (print_level>PRINT_NONE) fasp_param_solver_print(&itpar);
    
    //--------------------------//
    // Step 2. Solve the system //
    //--------------------------//
    
    // Set initial guess
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row,&x,0.0);
    //fasp_dvec_rand(A.row,&x);

    // Preconditioned Krylov methods
    if ( solver_type >= 1 && solver_type <= 20) {
        
        // Using no preconditioner for Krylov iterative methods
        if (precond_type == PREC_NULL) {
            status = fasp_solver_dcsr_krylov(&A, &b, &x, &itpar);
        }   
        
        // Using diag(A) as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_DIAG) {
            status = fasp_solver_dcsr_krylov_diag(&A, &b, &x, &itpar);
        }
        
        // Using AMG as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_AMG || precond_type == PREC_FMG) {
            if (print_level>PRINT_NONE) fasp_param_amg_print(&amgpar);
            //amgpar.smooth_order=NO_ORDER;
            status = fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itpar, &amgpar);
            //fasp_dvec_print(10, &x);
            //fasp_dvec_write("solu.dat", &x);
        }
        
        // Using ILU as preconditioner for Krylov iterative methods Q: Need to change!
        else if (precond_type == PREC_ILU) {
            if (print_level>PRINT_NONE) fasp_param_ilu_print(&ilupar);
            status = fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itpar, &ilupar);
        }
        
        // Using Schwarz as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_SCHWARZ){
            if (print_level>PRINT_NONE) fasp_param_swz_print(&swzpar);
            status = fasp_solver_dcsr_krylov_swz(&A, &b, &x, &itpar, &swzpar);
        }
        
        else {
            printf("### ERROR: Unknown preconditioner type %d!!!\n", precond_type);       
            status = ERROR_SOLVER_PRECTYPE;
        }
        
    }
    
    // AMG as the iterative solver
    else if (solver_type == SOLVER_AMG) {
        if (print_level>PRINT_NONE) fasp_param_amg_print(&amgpar);
        fasp_solver_amg(&A, &b, &x, &amgpar); 
        
    }

    // Full AMG as the iterative solver 
    else if (solver_type == SOLVER_FMG) {
        if (print_level>PRINT_NONE) fasp_param_amg_print(&amgpar);
        fasp_solver_famg(&A, &b, &x, &amgpar);
    }
    
#if With_SuperLU // use SuperLU directly
    else if (solver_type == SOLVER_SUPERLU) {
        status = superlu(&A, &b, &x, print_level);   
    }
#endif   
    
#if With_UMFPACK // use UMFPACK directly
    else if (solver_type == SOLVER_UMFPACK) {
        status = umfpack(&A, &b, &x, print_level);   
    }
#endif   
    
    else {
        printf("### ERROR: Unknown solver type %d!!!\n", solver_type);        
        status = ERROR_SOLVER_TYPE;
    }
        
    if (status<0) {
        printf("\n### ERROR: Solver failed! Exit status = %d.\n\n", status);
    }
    
    //fasp_dvec_write("solu.dat", &x);
    //fasp_dvec_print(10, &x);
    
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
