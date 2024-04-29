/*! \file  test.c
 *
 *  \brief The main test function for FASP solvers
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
 * \date   03/31/2009
 *
 * Modified by Chensong Zhang on 09/09/2011
 * Modified by Chensong Zhang on 06/21/2012
 * Modified by Chensong Zhang on 10/15/2012: revise along with testfem.c
 * Modified by Chensong Zhang on 12/29/2013: clean up non-test problems
 * Modified by Xiaozhe Hu on 05/02/2014: umfpack--separate factorization and solve
 * Modified by Chensong Zhang on 08/28/2022: add mumps and pardiso
 */
int main(int argc, const char* argv[])
{
    //---------------------//
    // Step 0. Preparation //
    //---------------------//

    // Set parameters
    input_param inipar; // parameters from input files
    ITS_param   itspar; // parameters for itsolver
    AMG_param   amgpar; // parameters for AMG
    ILU_param   ilupar; // parameters for ILU
    SWZ_param   swzpar; // parameters for Schwarz method

    // Set solver parameters
    fasp_param_set(argc, argv, &inipar);
    fasp_param_init(&inipar, &itspar, &amgpar, &ilupar, &swzpar);

    // Set local parameters
    const int print_level  = inipar.print_level;
    const int problem_num  = inipar.problem_num;
    const int solver_type  = inipar.solver_type;
    const int precond_type = inipar.precond_type;
    const int output_type  = inipar.output_type;

    // Set output device
    if (output_type) {
        char* outputfile = "out/test.out";
        printf("Redirecting outputs to file: %s ...\n", outputfile);
        if (freopen(outputfile, "w", stdout) == NULL) // open a file for stdout
            fprintf(stderr, "Output redirecting stdout\n");
    }

    // Data filenames
    char filename1[512], *datafile1;
    char filename2[512], *datafile2;
    char filename3[512], *datafile3;
    char filename4[512], *datafile4;

    memcpy(filename1, inipar.workdir, STRLEN);
    memcpy(filename2, inipar.workdir, STRLEN);
    memcpy(filename3, inipar.workdir, STRLEN);
    memcpy(filename4, inipar.workdir, STRLEN);

    // Matrix and vector
    dCSRmat A;
    dvector b, x;
    int     status = FASP_SUCCESS;

    //----------------------------------------------------//
    // Step 1. Input stiffness matrix and right-hand side //
    //----------------------------------------------------//
    printf("Test Problem Number %d\n", problem_num);

    if (problem_num == 10) {
        // Read A and b -- P1 FE discretization for Poisson.
        datafile1 = "csrmat_FE.dat";
        strcat(filename1, datafile1);

        datafile2 = "rhs_FE.dat";
        strcat(filename2, datafile2);

        fasp_dcsrvec_read2(filename1, filename2, &A, &b);
    }

    else if (problem_num == 11) {
        // Read A -- P1 FE discretization for Poisson, 1M DoF
        datafile1 = "Poisson/coomat_1046529.dat"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
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
        // Read A -- P1 FE discretization for Poisson, 0.25M DoF
        datafile1 = "Poisson/coomat_261121.dat"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcoo_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 13) {
        // Read A -- P1 FE discretization for Poisson, 65K DoF
        datafile1 = "Poisson/coomat_65025.dat"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcoo_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 14) {
        // Read A -- 5pt FD stencil for Poisson, 1M DoF
        datafile1 = "Poisson/fdm_1023X1023.csr"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 21) {
        // Read A and b -- Elasticity problem from Huawei (large)
        datafile1 = "case21/COOMatrix_1Row.bcoo";
        strcat(filename1, datafile1);

        datafile2 = "case21/COOMatrix_1Col.bcoo";
        strcat(filename2, datafile2);

        datafile3 = "case21/COOMatrix_1Value.bcoo";
        strcat(filename3, datafile3);

        datafile4 = "case21/rhx_1.bvec";
        strcat(filename4, datafile4);

        fasp_dcoovec_bin_read(filename1, filename2, filename3, filename4, &A, &b);
    }

    else if (problem_num == 22) {
        // Read A and b -- Elasticity problem from Huawei (middle)
        datafile1 = "case22/COOMatrix_1Row.bcoo";
        strcat(filename1, datafile1);

        datafile2 = "case22/COOMatrix_1Col.bcoo";
        strcat(filename2, datafile2);

        datafile3 = "case22/COOMatrix_1Value.bcoo";
        strcat(filename3, datafile3);

        datafile4 = "case22/rhx_1.bvec";
        strcat(filename4, datafile4);

        fasp_dcoovec_bin_read(filename1, filename2, filename3, filename4, &A, &b);
    }

    else if (problem_num == 27) {
        // Read A and b -- Elasticity problem from Huawei (small)
        datafile1 = "case27/COOMatrix_1Row.bcoo";
        strcat(filename1, datafile1);

        datafile2 = "case27/COOMatrix_1Col.bcoo";
        strcat(filename2, datafile2);

        datafile3 = "case27/COOMatrix_1Value.bcoo";
        strcat(filename3, datafile3);

        datafile4 = "case27/rhx_1.bvec";
        strcat(filename4, datafile4);

        fasp_dcoovec_bin_read(filename1, filename2, filename3, filename4, &A, &b);
    }

    else if (problem_num == 31) {
        // Read A -- FE discretization for DLD
        datafile1 = "DLD/A11.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcoo_read1(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 41) {
        // Read A -- FE discretization for DLD
        datafile1 = "femlevels/A1.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 42) {
        // Read A -- FE discretization for DLD
        datafile1 = "femlevels/A2.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 43) {
        // Read A -- FE discretization for DLD
        datafile1 = "femlevels/A3.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 44) {
        // Read A -- FE discretization for DLD
        datafile1 = "femlevels/A4.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
    }

    else if (problem_num == 45) {
        // Read A -- FE discretization for DLD
        datafile1 = "femlevels/A5.coo"; // This file is NOT in ../data!
        strcat(filename1, datafile1);
        fasp_dcsr_read(filename1, &A);

        // Generate a random solution
        dvector sol = fasp_dvec_create(A.row);
        fasp_dvec_rand(A.row, &sol);

        // Form the right-hand-side b = A*sol
        b = fasp_dvec_create(A.row);
        fasp_blas_dcsr_mxv(&A, sol.val, b.val);
        fasp_dvec_free(&sol);
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

    // Print out solver parameters
    if (print_level > PRINT_NONE) fasp_param_solver_print(&itspar);

    //--------------------------//
    // Step 2. Solve the system //
    //--------------------------//

    // Set initial guess
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);

    // Preconditioned Krylov methods
    if (solver_type >= 1 && solver_type <= 20) {

        // Using no preconditioner for Krylov iterative methods
        if (precond_type == PREC_NULL) {
            status = fasp_solver_dcsr_krylov(&A, &b, &x, &itspar);
        }

        // Using diag(A) as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_DIAG) {
            status = fasp_solver_dcsr_krylov_diag(&A, &b, &x, &itspar);
        }

        // Using AMG as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_AMG || precond_type == PREC_FMG) {
            if (print_level > PRINT_NONE) fasp_param_amg_print(&amgpar);
            status = fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itspar, &amgpar);
        }

        // Using ILU as preconditioner for Krylov iterative methods Q: Need to change!
        else if (precond_type == PREC_ILU) {
            if (print_level > PRINT_NONE) fasp_param_ilu_print(&ilupar);
            status = fasp_solver_dcsr_krylov_ilu(&A, &b, &x, &itspar, &ilupar);
        }

        // Using Schwarz as preconditioner for Krylov iterative methods
        else if (precond_type == PREC_SCHWARZ) {
            if (print_level > PRINT_NONE) fasp_param_swz_print(&swzpar);
            status = fasp_solver_dcsr_krylov_swz(&A, &b, &x, &itspar, &swzpar);
        }

        else {
            printf("### ERROR: Unknown preconditioner %d!!!\n", precond_type);
            status = ERROR_SOLVER_PRECTYPE;
        }
    }

    // AMG as the iterative solver
    else if (solver_type == SOLVER_AMG) {
        if (print_level > PRINT_NONE) fasp_param_amg_print(&amgpar);
        fasp_solver_amg(&A, &b, &x, &amgpar);
    }

    // Full AMG as the iterative solver
    else if (solver_type == SOLVER_FMG) {
        if (print_level > PRINT_NONE) fasp_param_amg_print(&amgpar);
        fasp_solver_famg(&A, &b, &x, &amgpar);
    }

#if WITH_SuperLU // Call SuperLU if linked with -DUSE_SUPERLU=ON
    else if (solver_type == SOLVER_SUPERLU) {
        status = fasp_solver_superlu(&A, &b, &x, print_level);
    }
#endif

#if WITH_UMFPACK // Call UMFPACK if linked with -DUSE_UMFPACK=ON
    else if (solver_type == SOLVER_UMFPACK) {
        dCSRmat A_tran;
        fasp_dcsr_trans(&A, &A_tran);
        fasp_dcsr_sort(&A_tran);
        void* Numeric = fasp_umfpack_factorize(&A_tran, print_level);

        status = fasp_umfpack_solve(&A_tran, &b, &x, Numeric, print_level);
        fasp_umfpack_free_numeric(Numeric);
        fasp_dcsr_free(&A_tran);
    }
#endif

#if WITH_MUMPS // Call MUMPS if linked with -DUSE_MUMPS=ON
    else if (solver_type == SOLVER_MUMPS) {
        status = fasp_solver_mumps(&A, &b, &x, print_level);
    }
#endif

#if WITH_PARDISO // Call PARDISO if linked with -DUSE_PARDISO=ON
    else if (solver_type == SOLVER_PARDISO) {
        fasp_dcsr_sort(&A);
        status = fasp_solver_pardiso(&A, &b, &x, print_level);
    }
#endif

#if WITH_STRUMPACK // Call STRUMPACK if linked with -DUSE_STRUMPACK=ON
    else if (solver_type == SOLVER_STRUMPACK) {
        fasp_dcsr_sort(&A);
        status = fasp_solver_strumpack(&A, &b, &x, print_level);
    }
#endif

    else {
        printf("### ERROR: Unknown solver %d!!!\n", solver_type);
        status = ERROR_SOLVER_TYPE;
    }

    if (status < 0) {
        printf("\n### ERROR: Solver failed! Exit status = %d.\n\n", status);
    }

    if (output_type) fclose(stdout);

    //------------------------//
    // Step 3. Check solution //
    //------------------------//
    fasp_blas_dcsr_aAxpy(-1.0, &A, x.val, b.val); // compute residual

    printf("L2 norm of residual = %.10e\n\n", fasp_blas_dvec_norm2(&b));

    //-------------------------//
    // Step 4. Clean up memory //
    //-------------------------//
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
