/*! \file  regression.c
 *
 *  \brief Regression tests for iterative solvers
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
 * \brief This function compares x and sol to a given tolerance tol. 
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
 * Modified by Chunsheng Feng on 03/04/2016
 * Modified by Chensong Zhang on 01/22/2017
 */
int main (int argc, const char * argv[]) 
{
    const INT  print_level = 1;    // how much information to print out
    const INT  num_prob    = 3;    // how many problems to be used
    const REAL tolerance   = 1e-4; // tolerance for accepting the solution
    
    /* Local Variables */
    ITS_param  itparam;      // input parameters for iterative solvers
    AMG_param  amgparam;     // input parameters for AMG
    dCSRmat    A;            // coefficient matrix
    dvector    b, x, sol;    // rhs, numerical sol, exact sol
    INT        indp;         // index for test problems
    int        status = FASP_SUCCESS;
    time_t     lt = time(NULL);
    
    printf("\n\n");
    printf("------------------------- Test starts at -------------------------\n");
    printf("%s",asctime(localtime(&lt))); // output starting local time
    printf("------------------------------------------------------------------\n");
    
    ntest = nfail = 0;
    
    for ( indp = 1; indp <= num_prob; indp++ ) {
        
        /*******************************************/
        /* Step 1. Get matrix and right-hand side  */
        /*******************************************/
        printf("\n=====================================================\n");
        printf("Test Problem Number %d ...\n", indp);   
        
        switch (indp) {
                
            case 1: // - Problem 1. 10X10 5pt FD for Poisson
                
                printf("10X10 5-point finite difference for Poisson");  
                printf("\n=====================================================\n");        
                
                // Read A and b from two files in IJ format. 
                fasp_dcsrvec_read2("../data/csrmat_FD.dat", "../data/rhs_FD.dat", &A, &b);
                
                // Read ref. sol. from a non-indexed vec file.
                fasp_dvecind_read("../data/sol_FD.dat", &sol);

                break;
                
            case 2: // - Problem 2. P1 FE for Poisson.
                
                printf("P1 finite element for Poisson");    
                printf("\n=====================================================\n");        
                
                // Read A and b from two files in IJ format. 
                fasp_dcsrvec_read2("../data/csrmat_FE.dat", "../data/rhs_FE.dat", &A, &b);
                
                // Read ref. sol. from an indexed vec file.
                fasp_dvecind_read("../data/sol_FE.dat", &sol);
                
                break;
                
            case 3: // - Problem 3. MatrixMarket finite element analysis NOS7.
                // Finite difference approximation to diffusion equation with varying
                // diffusivity in a 3D unit cube with Dirichlet boundary conditions.
                
                printf("MatrixMarket finite element analysis NOS7");    
                printf("\n=====================================================\n");        
                
                // Read A in MatrixMarket SYM COO format. 
                fasp_dmtxsym_read("../data/nos7.mtx", &A);
                
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
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle (Direct interpolation) with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG (direct interp) V-cycle as iterative solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);

            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle (Standard interpolation) with GS smoother as a solver */         
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG (standard interp) V-cycle as iterative solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            amgparam.interpolation_type = INTERP_STD;
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle (EM interpolation) with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG (energy-min interp) V-cycle as iterative solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            amgparam.interpolation_type = INTERP_ENG;
            amgparam.maxit       = 30;
            amgparam.tol         = 1e-11;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* FMG V-cycle (Direct interpolation) with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("FAMG (direct interp) V-cycle as iterative solver ...\n");   
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.print_level = print_level;
            fasp_solver_famg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG W-cycle with GS smoother as a solver */          
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG W-cycle as iterative solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.cycle_type  = W_CYCLE;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG AMLI-cycle with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG AMLI-cycle as iterative solver ...\n");   
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.cycle_type  = AMLI_CYCLE;
            amgparam.amli_degree = 3;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG Nonlinear AMLI-cycle with GS smoother as a solver */         
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG Nonlinear AMLI-cycle as iterative solver ...\n"); 
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.cycle_type  = NL_AMLI_CYCLE;
            amgparam.amli_degree = 3;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle with SGS smoother as a solver */         
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG V-cycle with SGS smoother as iterative solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.smoother    = SMOOTHER_SGS;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle with L1_DIAG smoother as a solver */         
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG V-cycle with L1_DIAG smoother as iterative solver ...\n");    
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 500;
            amgparam.tol         = 1e-10;
            amgparam.smoother    = SMOOTHER_L1DIAG;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* AMG V-cycle with SOR smoother as a solver */         
            printf("------------------------------------------------------------------\n");
            printf("Classical AMG V-cycle with SOR smoother as iterative solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 20;
            amgparam.tol         = 1e-10;
            amgparam.smoother    = SMOOTHER_SOR;
            amgparam.print_level = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* SA AMG V-cycle with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("SA AMG V-cycle with GS smoother as iterative solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit          = 500;
            amgparam.tol            = 1e-10;
            amgparam.strong_coupled = 0.15; // cannot be too big
            amgparam.AMG_type       = SA_AMG;
            amgparam.smoother       = SMOOTHER_GS;
            amgparam.print_level    = print_level;
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* UA AMG V-cycle with GS smoother as a solver */           
            printf("------------------------------------------------------------------\n");
            printf("UA AMG V-cycle with GS smoother as iterative solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_amg_init(&amgparam);
            amgparam.maxit       = 500;
            amgparam.tol         = 1e-10;
            amgparam.AMG_type    = UA_AMG;
            amgparam.smoother    = SMOOTHER_GS;
            amgparam.print_level = print_level;        
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* CG */
            printf("------------------------------------------------------------------\n");
            printf("CG solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;  
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* CG safe */
            printf("------------------------------------------------------------------\n");
            printf("CG solver with safe-net ...\n");

            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_s(&A, &b, &x, &itparam);

            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
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

        if ( indp==1 || indp==2 ) {
            /* MINRES */
            printf("------------------------------------------------------------------\n");
            printf("MINRES solver ...\n");
            
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

        if ( indp==1 || indp==2 ) {
            /* MINRES safe */
            printf("------------------------------------------------------------------\n");
            printf("MINRES solver with safe-net ...\n");

            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_MinRes;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_s(&A, &b, &x, &itparam);

            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
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

        if ( indp==1 || indp==2 ) {
            /* GMRES safe */
            printf("------------------------------------------------------------------\n");
            printf("GMRES solver with safe-net ...\n");

            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_s(&A, &b, &x, &itparam);

            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
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
        
        if ( indp==1 || indp==2 ) {
            /* VGMRES safe */
            printf("------------------------------------------------------------------\n");
            printf("VGMRES solver with safe-net ...\n");

            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_s(&A, &b, &x, &itparam);

            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
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
        
        if ( indp==1 || indp==2 ) {
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

        if ( indp==1 || indp==2 ) {
            /* GCR */
            printf("------------------------------------------------------------------\n");
            printf("GCR solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GCR;
            itparam.maxit         = 5000;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
            /* CG in BSR */
            dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
            
            printf("------------------------------------------------------------------\n");
            printf("CG solver in BSR format ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_CG;
            itparam.maxit         = 500;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dbsr_krylov(&A_bsr, &b, &x, &itparam);
            fasp_dbsr_free(&A_bsr);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
            /* BiCGstab in BSR */
            dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
            
            printf("------------------------------------------------------------------\n");
            printf("BiCGstab solver in BSR format ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;  
            itparam.itsolver_type = SOLVER_BiCGstab;
            itparam.maxit         = 500;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dbsr_krylov(&A_bsr, &b, &x, &itparam);
            fasp_dbsr_free(&A_bsr);

            check_solu(&x, &sol, tolerance);            
        }
        
        if ( indp==1 || indp==2 ) {
            /* GMRES in BSR */
            dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
            
            printf("------------------------------------------------------------------\n");
            printf("GMRES solver in BSR format ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-8;
            itparam.print_level   = print_level;
            fasp_solver_dbsr_krylov(&A_bsr, &b, &x, &itparam);
            fasp_dbsr_free(&A_bsr);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
            /* VGMRES in BSR */
            dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
            
            printf("------------------------------------------------------------------\n");
            printf("VGMRES solver in BSR format ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-8;
            itparam.print_level   = print_level;
            fasp_solver_dbsr_krylov(&A_bsr, &b, &x, &itparam);
            fasp_dbsr_free(&A_bsr);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 ) {
            /* VFGMRES in BSR */
            dBSRmat A_bsr = fasp_format_dcsr_dbsr (&A, 1);
            
            printf("------------------------------------------------------------------\n");
            printf("VFGMRES solver in BSR format ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_NULL;
            itparam.itsolver_type = SOLVER_VFGMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-8;
            itparam.print_level   = print_level;
            fasp_solver_dbsr_krylov(&A_bsr, &b, &x, &itparam);
            fasp_dbsr_free(&A_bsr);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using diag(A) as preconditioner for CG */
            printf("------------------------------------------------------------------\n");
            printf("Diagonal preconditioned CG solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            itparam.precond_type  = PREC_DIAG;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_diag(&A, &b, &x, &itparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for CG */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned CG solver ...\n");   
            fasp_dvec_set(b.row,&x,0.0);
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for BiCGstab */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned BiCGstab solver ...\n"); 
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_BiCGstab;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for MinRes */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned MinRes solver ...\n");   
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_MinRes;
            itparam.maxit         = 500;
            itparam.print_level   = print_level;

            // This is special. If 1e-10, cost a lot more iterations
            itparam.tol           = 1e-9;
            // We need to use 2 smoothing steps to make test 3 to converge --Chensong
            amgparam.presmooth_iter  = amgparam.postsmooth_iter = 2;
            amgparam.strong_threshold = 0.5;

            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for GMRes */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned GMRes solver ...\n");    
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_GMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for vGMRes */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned vGMRes solver ...\n");   
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_VGMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-12;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for VFGMRes */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned VFGMRes solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_VFGMRES;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for GCG */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned GCG solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_GCG;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using classical AMG as preconditioner for GCR */
            printf("------------------------------------------------------------------\n");
            printf("AMG preconditioned GCR solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_amg_init(&amgparam);
            itparam.itsolver_type = SOLVER_GCR;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itparam, &amgparam);
            
            check_solu(&x, &sol, tolerance);
        }
        
        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using ILUk as preconditioner for CG */
            ILU_param      iluparam;
            printf("------------------------------------------------------------------\n");
            printf("ILUk preconditioned CG solver ...\n");  
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            itparam.maxit         = 500;
            itparam.tol           = 1e-8;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using ILUt as preconditioner for CG */
            ILU_param      iluparam;
            printf("------------------------------------------------------------------\n");
            printf("ILUt preconditioned CG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            iluparam.ILU_type     = ILUt;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
            
            check_solu(&x, &sol, tolerance);
        }

        if ( indp==1 || indp==2 || indp==3 ) {
            /* Using ILUtp as preconditioner for CG */
            ILU_param      iluparam;
            printf("------------------------------------------------------------------\n");
            printf("ILUtp preconditioned CG solver ...\n");
            
            fasp_dvec_set(b.row, &x, 0.0); // reset initial guess
            fasp_param_solver_init(&itparam);
            fasp_param_ilu_init(&iluparam);
            iluparam.ILU_type     = ILUtp;
            itparam.maxit         = 500;
            itparam.tol           = 1e-10;
            itparam.print_level   = print_level;
            fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itparam, &iluparam);
            
            check_solu(&x, &sol, tolerance);
        }

        /* clean up memory */
        fasp_dcsr_free(&A);
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
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
