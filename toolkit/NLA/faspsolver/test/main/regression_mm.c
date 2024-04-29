/*! \file  regression_mm.c
 *
 *  \brief Regression tests with Matrix-Market problems
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

#define num_prob      10                /**< how many problems to be used */
#define num_solvers    8                /**< how many methods  to be used */

unsigned int  ntest[num_solvers];       /**< number of tests without preconditioner */
unsigned int  nfail[num_solvers];       /**< number of failed tests without preconditioner */
unsigned int  ntest_diag[num_solvers];  /**< number of tests using diag preconditioner */
unsigned int  nfail_diag[num_solvers];  /**< number of failed tests for diag preconditioner */
unsigned int  ntest_iluk[num_solvers];  /**< number of tests using ILUk preconditioner */
unsigned int  nfail_iluk[num_solvers];  /**< number of failed tests for ILUk preconditioner */
unsigned int  ntest_ilut[num_solvers];  /**< number of tests using ILUt preconditioner */
unsigned int  nfail_ilut[num_solvers];  /**< number of failed tests for ILUt preconditioner */
unsigned int  ntest_ilutp[num_solvers]; /**< number of tests using ILUtp preconditioner */
unsigned int  nfail_ilutp[num_solvers]; /**< number of failed tests for ILUtp preconditioner */
unsigned int  ntest_amg[num_solvers];   /**< number of tests using AMG preconditioner */
unsigned int  nfail_amg[num_solvers];   /**< number of failed tests for AMG preconditioner */
unsigned int  ntest_amg_solver;         /**< number of tests using AMG solver */
unsigned int  nfail_amg_solver;         /**< number of failed tests for AMG solver */

/**
 * \fn static void check_solu(dvector *x, dvector *sol, double tol)
 *
 * This function compares x and sol to a given tolerance tol.
 */
static void check_solu(dvector *x, dvector *sol, double tol, unsigned int *nt, unsigned int *nf)
{
    double diff_u = fasp_dvec_maxdiff(x, sol);
    (*nt)++;
    
    if ( diff_u < tol ) {
        printf("Max diff %.4e smaller than tolerance................. [PASS]\n", diff_u);
    }
    else {
        (*nf)++;
        printf("### WARNING: Max diff %.4e BIGGER than tolerance..... [ATTENTION!!!]\n", diff_u);
    }
}

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for solver test.
 *
 * We pick a few test problems from the Matrix Market randomly.
 * Some are symmetric and some are non-symmetric. We use these
 * test problems to check the package.
 *
 * \author Feiteng Huang
 * \date   06/12/2012
 *
 * Modified by Chunsheng Feng on 02/16/2017: changed the reading format for matrix 9
 */
int main (int argc, const char * argv[])
{
    const INT  print_level = 1;    // how much information to print out
    const REAL tolerance   = 1e-4; // tolerance for accepting the solution
    const char solvers[num_solvers][128] = {"CG", "BiCGstab", "MinRes",
                                            "GMRES", "VGMRES", "VFGMRES",
                                            "GCG", "GCR"};
    
    /* Local Variables */
    ITS_param  itparam;      // input parameters for iterative solvers
    ILU_param  iluparam;     // input parameters for AMG
    AMG_param  amgparam;     // input parameters for AMG
    dCSRmat    A;            // coefficient matrix
    dvector    b, x, sol;    // rhs, numerical sol, exact sol
    int        indp;         // index for test problems
    int        indm;         // index for test methods
    
    time_t     lt  = time(NULL);
    
    printf("\n\n");
    printf("------------------------- Test starts at -------------------------\n");
    printf("%s",asctime(localtime(&lt))); // output starting local time
    printf("------------------------------------------------------------------\n");
    
    memset(ntest_diag, 0x0, num_solvers*sizeof(int));
    memset(nfail_diag, 0x0, num_solvers*sizeof(int));
    memset(ntest_iluk, 0x0, num_solvers*sizeof(int));
    memset(nfail_ilut, 0x0, num_solvers*sizeof(int));
    memset(ntest_amg,  0x0, num_solvers*sizeof(int));
    memset(nfail_amg,  0x0, num_solvers*sizeof(int));
    ntest_amg_solver = 0;
    nfail_amg_solver = 0;
    
    for ( indp = 1; indp <= num_prob; indp++ ) {
        
        /*******************************************/
        /* Step 1. Get matrix and right-hand side  */
        /*******************************************/
        printf("\n==================================================================\n");
        printf("Test Problem Number %d ...\n", indp);
        
        switch (indp) {
                
            case 1:
                // - Problem 1. MatrixMarket Driven cavity E05R0500.
                // driven cavity, 5x5 elements, Re=500
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/e05r0500.mtx", &A);
                
                printf("MatrixMarket Driven cavity E05R0500\n");
                printf("||  Condition Number:      4.8e+6   ||\n");
                printf("||        Unsymmetric               ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 2:
                // - Problem 2. MatrixMarket Finite element analysis of cylindrical shells.
                // Cylindrical shell, uniform 30x30 quadrilateral mesh,
                // stabilized MITC4 elements, R/t=100
                
                // Read A in MatrixMarket COO format.
                fasp_dmtxsym_read("../data/s2rmq4m1.mtx", &A);
                
                printf("MatrixMarket Finite element analysis of cylindrical shells\n");
                printf("||  Condition Number:     1.15e+8   ||\n");
                printf("||   Symmetric positive definite    ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 3:
                // - Problem 3. MatrixMarket Oil reservoir simulation - generated problems.
                // oil reservoir simulation for 21x21x5 full grid
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/orsreg_1.mtx", &A);
                
                printf("MatrixMarket Oil reservoir simulation - generated problems\n");
                printf("||  Condition Number:        1e+2   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 4:
                // - Problem 4. MatrixMarket Enhanced oil recovery.
                // 3D steam model of oil res. -5x5x6 -4 DOF
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/steam2.mtx", &A);
                
                printf("MatrixMarket Enhanced oil recovery\n");
                printf("||  Condition Number:      3.5e+6   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 5:
                // - Problem 5. MatrixMarket BCS Structural Engineering Matrices.
                // U.S. Army Corps of Engineers dam
                
                // Read A in MatrixMarket COO format.
                fasp_dmtxsym_read("../data/bcsstk16.mtx", &A);
                
                printf("MatrixMarket BCS Structural Engineering Matrices\n");
                printf("||  Condition Number:          65   ||\n");
                printf("||   Symmetric positive definite    ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 6:
                // - Problem 6. MatrixMarket Circuit physics modeling.
                // Computer random simulation of a circuit physics model
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/jpwh_991.mtx", &A);
                
                printf("MatrixMarket Circuit physics modeling\n");
                printf("||  Condition Number:      7.3e+2   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 7:
                // - Problem 7. MatrixMarket Simulation of computer systems.
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/gre__115.mtx", &A);
                
                printf("MatrixMarket Simulation of computer systems\n");
                printf("||  Condition Number:      1.5e+2   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 8:
                // - Problem 8. MatrixMarket Computer component design.
                // 32-bit adder
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/add32.mtx", &A);
                
                printf("MatrixMarket Computer component design\n");
                printf("||  Condition Number:     2.14e+2   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 9:
                // - Problem 9. MatrixMarket Oil reservoir simulation challenge matrics.
                // Black oil simulation, shale barriers(NX = NY = NZ = 10, NC = 1)
                
                // Read A in MatrixMarket COO format.
               fasp_dmtx_read("../data/sherman1.mtx", &A); // Chunsheng, 02/16/2017
                
                printf("MatrixMarket Oil reservoir simulation challenge matrics\n");
                printf("||  Condition Number:      2.3e+4   ||\n");
                printf("||            Symmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
                
            case 10:
                // - Problem 10. MatrixMarket Petroleum engineering.
                
                // Read A in MatrixMarket COO format.
                fasp_dmtx_read("../data/watt__1.mtx", &A);
                
                printf("MatrixMarket Petroleum engineering\n");
                printf("||  Condition Number:     5.38e+9   ||\n");
                printf("||          Unsymmetric             ||\n");
                printf("|| row:%5d, col:%5d, nnz:%6d ||\n", A.row, A.col, A.nnz);
                printf("==================================================================\n");
                
                // Generate an exact solution randomly
                sol = fasp_dvec_create(A.row);
                fasp_dvec_rand(A.row, &sol);
                
                // Form the right-hand-side b = A*sol
                b = fasp_dvec_create(A.row);
                fasp_blas_dcsr_mxv(&A, sol.val, b.val);
                
                break;
        }
        
        /************************************/
        /* Step 2. Check matrix properties  */
        /************************************/
        fasp_check_symm(&A);     // check symmetry
        fasp_check_diagpos(&A);  // check sign of diagonal entries
        fasp_check_diagdom(&A);  // check diagonal dominance
        
        /*****************************/
        /* Step 3. Solve the system  */
        /*****************************/
        fasp_dvec_alloc(b.row, &x);  // allocate mem for numerical solution
        
        if (TRUE) {
            /* Using no preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("Krylov solver ...\n");
            
            fasp_param_solver_init(&itparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
                check_solu(&x, &sol, tolerance, &(ntest[indm]), &(nfail[indm]));
            }
        }

        if (TRUE) {
            /* Using diagonal preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("Diagonal preconditioned Krylov solver ...\n");
            
            fasp_param_solver_init(&itparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov_diag(&A, &b, &x, &itparam);
                check_solu(&x, &sol, tolerance, &(ntest_diag[indm]), &(nfail_diag[indm]));
            }
        }
        
        if (TRUE) {
            /* Using ILUk as preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("ILUk preconditioned Krylov solver ...\n");
            
            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            iluparam.ILU_type     = ILUk;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
                check_solu(&x, &sol, tolerance, &(ntest_iluk[indm]), &(nfail_iluk[indm]));
            }
        }
        
        if (TRUE) {
            /* Using ILUt as preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("ILUt preconditioned Krylov solver ...\n");
            
            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            iluparam.ILU_type     = ILUt;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
                check_solu(&x, &sol, tolerance, &(ntest_ilut[indm]), &(nfail_ilut[indm]));
            }
        }

        if (TRUE) {
            /* Using ILUtp as preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("ILUtp preconditioned Krylov solver ...\n");

            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            iluparam.ILU_type     = ILUtp;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
                check_solu(&x, &sol, tolerance, &(ntest_ilutp[indm]), &(nfail_ilutp[indm]));
            }
        }

        if (TRUE) {
            /* Using classical AMG as preconditioner for Krylov methods */
            printf("\n------------------------------------------------------------------\n");
            printf("AMG preconditioned Krylov solver ...\n");
            
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.maxit         = 100;
            itparam.tol           = 1e-15;
            itparam.print_level   = print_level;
            for (indm = 0; indm<num_solvers; indm++) {
                fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
                itparam.itsolver_type = indm+1;
                fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
                check_solu(&x, &sol, tolerance, &(ntest_amg[indm]), &(nfail_amg[indm]));
            }
        }
        
        if (TRUE) {
            /* Using classical AMG as a solver */
            printf("\n------------------------------------------------------------------\n");
            printf("AMG as iterative solver ...\n");
            
            amgparam.maxit        = 20;
            amgparam.tol          = 1e-10;
            amgparam.print_level  = print_level;
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_solver_amg(&A, &b, &x,&amgparam);
            check_solu(&x, &sol, tolerance, &ntest_amg_solver, &nfail_amg_solver);
        }
        
        /* clean up memory */
        fasp_dcsr_free(&A);
        fasp_dvec_free(&b);
        fasp_dvec_free(&x);
        fasp_dvec_free(&sol);
        
    } // end of for indp
    
    /* all done */
    lt = time(NULL);
    printf("\n---------------------- All test finished at ----------------------\n");
    printf("%s",asctime(localtime(&lt))); // output ending local time
    printf("------------------------------------------------------------------\n\n");
    
    printf("========================No preconditioner=========================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest[indm], nfail[indm], ntest[indm]-nfail[indm]);
    }
    printf("=======================Diagonal preconditioner====================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest_diag[indm], nfail_diag[indm], ntest_diag[indm]-nfail_diag[indm]);
    }
    printf("========================ILUk preconditioner======================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest_iluk[indm], nfail_iluk[indm], ntest_iluk[indm]-nfail_iluk[indm]);
    }
    printf("========================ILUt preconditioner======================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest_ilut[indm], nfail_ilut[indm], ntest_ilut[indm]-nfail_ilut[indm]);
    }
    printf("========================ILUtp preconditioner=====================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest_ilutp[indm], nfail_ilutp[indm], ntest_ilutp[indm]-nfail_ilutp[indm]);
    }
    printf("=========================AMG preconditioner======================\n");
    for (indm=0; indm<num_solvers; indm++) {
        printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n",
               solvers[indm], ntest_amg[indm], nfail_amg[indm], ntest_amg[indm]-nfail_amg[indm]);
    }
    printf("=============================AMG==================================\n");
    printf("Solver %10s:  %2d tests finished: %2d failed, %2d succeeded!\n", "AMG",
           ntest_amg_solver, nfail_amg_solver, ntest_amg_solver-nfail_amg_solver);
    printf("==================================================================\n");
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
