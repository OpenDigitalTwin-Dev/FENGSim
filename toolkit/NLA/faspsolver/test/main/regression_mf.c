/*! \file  regression_mf.c
 *
 *  \brief Regression tests for matrix-free iterative solvers
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

unsigned int  ntest;    /**< number of tests all together */
unsigned int  nfail;    /**< number of failed tests */

/**
 * \fn static void check_solu(dvector *x, dvector *sol, double tol)
 *
 * This function compares x and sol to a given tolerance tol.
 */
static void check_solu(dvector *x, dvector *sol, double tol)
{
    double diff_u = fasp_dvec_maxdiff(x, sol);
    ntest++;
    
    if ( diff_u < tol ) {
        printf("Max diff %.4e smaller than tolerance................. [PASS]\n", diff_u);
    }
    else {
        nfail++;
        printf("### WARNING: Max diff %.4e BIGGER than tolerance..... [ATTENTION!!!]\n", diff_u);
    }
}

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for regression test.
 *
 * \author Chensong Zhang
 * \date   03/20/2009
 *
 * Modified by Chensong Zhang on 09/02/2011
 * Modified by Chensong Zhang on 12/03/2011
 * Modified by Chensong Zhang on 03/20/2012
 * Modified by Feiteng Huang on 09/19/2012
 * Modified by Chunsheng Feng on 03/04/2016
 */
int main (int argc, const char * argv[])
{
    const INT  print_level = 1;    // how much information to print out
    const INT  num_prob    = 2;    // how many problems to be used
    const REAL tolerance   = 1e-4; // tolerance for accepting the solution
    
    /* Local Variables */
    ITS_param   itparam;      // input parameters for iterative solvers
    dCSRmat     A;            // coefficient matrix
    dvector     b, x, sol;    // rhs, numerical sol, exact sol
    INT         indp;         // index for test problems
    
    time_t      lt  = time(NULL);
    
    printf("\n\n");
    printf("------------------------- Test starts at -------------------------\n");
    printf("%s",asctime(localtime(&lt))); // output starting local time
    printf("------------------------------------------------------------------\n");
    
    ntest = nfail = 0;
    
    /*******************************************/
    /* Step 1. Get matrix and right-hand side  */
    /*******************************************/
    for ( indp = 1; indp <= num_prob; indp++ ) {
        
        printf("\n=====================================================\n");
        printf("Test Problem Number %d ...\n", indp);
        
        switch (indp) {
                
            case 1: //     - Problem 1. 10X10 5pt FD for Poisson
                
                printf("10X10 5-point finite difference for Poisson");
                printf("\n=====================================================\n");
                
                // Read A and b from two files in IJ format.
                fasp_dcsrvec_read2("../data/csrmat_FD.dat", "../data/rhs_FD.dat", &A, &b);
                
                // Read ref. sol. from a non-indexed vec file.
                fasp_dvecind_read("../data/sol_FD.dat", &sol);
                
                break;
                
            case 2: //     - Problem 2. P1 FE for Poisson.
                
                printf("P1 finite element for Poisson");
                printf("\n=====================================================\n");
                
                // Read A and b from two files in IJ format.
                fasp_dcsrvec_read2("../data/csrmat_FE.dat", "../data/rhs_FE.dat", &A, &b);
                
                // Read ref. sol. from an indexed vec file.
                fasp_dvecind_read("../data/sol_FE.dat", &sol);
                
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
        
        dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* CG */
            printf("------------------------------------------------------------------\n");
            printf("CG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free CG */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free CG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free CG for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free CG solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* BiCGstab */
            printf("------------------------------------------------------------------\n");
            printf("BiCGstab solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_BiCGstab;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free BiCGstab */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free BiCGstab solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_BiCGstab;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free BiCGstab for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free BiCGstab solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_BiCGstab;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* MinRes */
            printf("------------------------------------------------------------------\n");
            printf("MinRes solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_MinRes;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free MinRes */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free MinRes solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_MinRes;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free MinRes for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free MinRes solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_MinRes;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* GMRES */
            printf("------------------------------------------------------------------\n");
            printf("GMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free GMRES */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free GMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free GMRES for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free GMRES solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* VGMRES */
            printf("------------------------------------------------------------------\n");
            printf("VGMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free VGMRES */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free VGMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free VGMRES for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free VGMRES solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* VFGMRES */
            printf("------------------------------------------------------------------\n");
            printf("VFGMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VFGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free VFGMRES */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free VFGMRES solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VFGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free VFGMRES for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free VFGMRES solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VFGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* GCG */
            printf("------------------------------------------------------------------\n");
            printf("GCG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GCG;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free GCG */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free GCG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GCG;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_CSR, &mf, &A);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Matrix-free GCG for BSR */
            printf("------------------------------------------------------------------\n");
            printf("Matrix-free GCG solver for BSR ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GCG;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            mxv_matfree mf;
            fasp_solver_matfree_init(MAT_BSR, &mf, &A_bsr);
            fasp_solver_krylov(&mf, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        /* clean up memory */
        fasp_dcsr_free(&A);
        fasp_dbsr_free(&A_bsr);
        fasp_dvec_free(&b);
        fasp_dvec_free(&x);
        fasp_dvec_free(&sol);
        
    } // end of for indp
    
    /* all done */
    lt = time(NULL);
    printf("---------------------- All test finished at ----------------------\n");
    printf("%d tests finished: %d failed, %d succeeded!\n", ntest, nfail, ntest-nfail);
    printf("%s",asctime(localtime(&lt))); // output ending local time
    printf("------------------------------------------------------------------\n");
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
