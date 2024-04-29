/*! \file  poisson-pcg.c
 *
 *  \brief The third test example for FASP: using PCG to solve
 *         the discrete Poisson equation from P1 finite element.
 *         C version.
 *
 *  \note  Solving the Poisson equation (P1 FEM) with PCG: C version
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
 * \brief This is the main function for the third example.
 *
 * \author Feiteng Huang
 * \date   05/17/2012
 *
 * Modified by Chensong Zhang on 09/22/2012
 * Modified by Chensong Zhang on 12/23/2018: Fix memory leakage
 */
int main(int argc, const char* argv[])
{
    input_param inparam;  // parameters from input files
    ITS_param   itparam;  // parameters for itsolver
    AMG_param   amgparam; // parameters for AMG
    ILU_param   iluparam; // parameters for ILU

    printf("\n========================================");
    printf("\n||   FASP: PCG example -- C version   ||");
    printf("\n========================================\n\n");

    // Step 0. Set parameters: We can use ini/pcg.dat
    fasp_param_set(argc, argv, &inparam);
    fasp_param_init(&inparam, &itparam, &amgparam, &iluparam, NULL);

    // Set local parameters
    const SHORT prtlvl    = itparam.print_level;
    const SHORT pc_type   = itparam.precond_type;
    const SHORT stop_type = itparam.stop_type;
    const INT   maxit     = itparam.maxit;
    const REAL  tol       = itparam.tol;
    const REAL  abstol    = itparam.abstol;

    // Step 1. Get stiffness matrix and right-hand side
    // Read A and b -- P1 FE discretization for Poisson. The location
    // of the data files is given in "ini/pcg.dat".
    dCSRmat A;
    dvector b, x;
    char    filename1[512], *datafile1;
    char    filename2[512], *datafile2;

    // Read the stiffness matrix from matFE.dat
    memcpy(filename1, inparam.workdir, STRLEN);
    datafile1 = "csrmat_FE.dat";
    strcat(filename1, datafile1);

    // Read the RHS from rhsFE.dat
    memcpy(filename2, inparam.workdir, STRLEN);
    datafile2 = "rhs_FE.dat";
    strcat(filename2, datafile2);

    fasp_dcsrvec_read2(filename1, filename2, &A, &b);

    // Step 2. Print problem size and PCG parameters
    if (prtlvl > PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        fasp_param_solver_print(&itparam);
    }

    // Setp 3. Setup preconditioner
    // Preconditioner type is determined by pc_type
    precond* pc = fasp_precond_setup(pc_type, &amgparam, &iluparam, &A);

    // Step 4. Solve the system with PCG as an iterative solver
    // Set the initial guess to be zero and then solve it using PCG solver
    // Note that we call PCG interface directly. There is another way which
    // calls the abstract iterative method interface; see possion-its.c for
    // more details.
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);

    fasp_solver_dcsr_pcg(&A, &b, &x, pc, tol, abstol, maxit, stop_type, prtlvl);

    // Step 5. Clean up memory
    // First clean up the AMG data if you are using AMG as preconditioner
    fasp_amg_data_free(((precond_data*)pc->data)->mgl_data, &amgparam);

    // Clean up the pcdata
    if (pc_type != PREC_NULL) fasp_mem_free(pc->data);
    fasp_mem_free(pc);

    // Clean up coefficient matrix, right-hand side, and solution
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);

    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
