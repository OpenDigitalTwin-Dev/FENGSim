/*! \file  SolWrapper.c
 *
 *  \brief Wrappers for accessing functions by advanced users
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxParam.c, BlaFormat.c, BlaSparseBSR.c, BlaSparseCSR.c, SolAMG.c,
 *         SolBSR.c, and SolCSR.c
 *
 *  \note  IMPORTANT: The wrappers DO NOT change the original matrix data. Users
 *         should shift the matrix indices in order to make the IA and JA to start
 *         from 0 instead of 1.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_block.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_fwrapper_dcsr_pardiso_ (INT *n, INT *nnz, INT *ia, INT *ja, REAL *a,
 *                                       REAL *b, REAL *u, INT *ptrlvl)
 *
 * \brief Solve Ax=b by the Pardiso direct solver
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   01/09/2020
 */
#if WITH_PARDISO
void fasp_fwrapper_dcsr_pardiso_(
    INT* n, INT* nnz, INT* ia, INT* ja, REAL* a, REAL* b, REAL* u, INT* ptrlvl)
{
    dCSRmat mat;      // coefficient matrix
    dvector rhs, sol; // right-hand-side, solution

    // set up coefficient matrix
    mat.row = *n;
    mat.col = *n;
    mat.nnz = *nnz;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n;
    rhs.val = b;
    sol.row = *n;
    sol.val = u;

    fasp_dcsr_sort(&mat);

    fasp_solver_pardiso(&mat, &rhs, &sol, *ptrlvl);
}
#endif

/**
 * \fn void fasp_fwrapper_dcsr_strumpack_ (INT *n, INT *nnz, INT *ia, INT *ja, REAL *a,
 *                                         REAL *b, REAL *u, INT *ptrlvl)
 *
 * \brief Solve Ax=b by the STRUMPACK
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Yumiao Zhang
 * \date   10/17/2022
 */
#if WITH_UMFPACK
void fasp_fwrapper_dcsr_strumpack_(
    INT* n, INT* nnz, INT* ia, INT* ja, REAL* a, REAL* b, REAL* u, INT* ptrlvl)
{
    dCSRmat mat;      // coefficient matrix
    dvector rhs, sol; // right-hand-side, solution

    // set up coefficient matrix
    mat.row = *n;
    mat.col = *n;
    mat.nnz = *nnz;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n;
    rhs.val = b;
    sol.row = *n;
    sol.val = u;

    fasp_dcsr_sort(&mat);

    fasp_solver_strumpack(&mat, &rhs, &sol, *ptrlvl);
}
#endif

/**
 * \fn void fasp_fwrapper_dcsr_amg_ (INT *n, INT *nnz, INT *ia, INT *ja, REAL *a,
 *                                   REAL *b, REAL *u, REAL *tol, INT *maxit,
 *                                   INT *ptrlvl)
 *
 * \brief Solve Ax=b by Ruge and Stuben's classic AMG
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param tol     Tolerance for iterative solvers
 * \param maxit   Max number of iterations
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   09/16/2010
 */
void fasp_fwrapper_dcsr_amg_(INT*  n,
                             INT*  nnz,
                             INT*  ia,
                             INT*  ja,
                             REAL* a,
                             REAL* b,
                             REAL* u,
                             REAL* tol,
                             INT*  maxit,
                             INT*  ptrlvl)
{
    dCSRmat   mat;      // coefficient matrix
    dvector   rhs, sol; // right-hand-side, solution
    AMG_param amgparam; // parameters for AMG

    // setup AMG parameters
    fasp_param_amg_init(&amgparam);

    amgparam.tol         = *tol;
    amgparam.print_level = *ptrlvl;
    amgparam.maxit       = *maxit;

    // set up coefficient matrix
    mat.row = *n;
    mat.col = *n;
    mat.nnz = *nnz;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n;
    rhs.val = b;
    sol.row = *n;
    sol.val = u;

    fasp_solver_amg(&mat, &rhs, &sol, &amgparam);
}

/**
 * \fn void fasp_fwrapper_dcsr_krylov_ilu_ (INT *n, INT *nnz, INT *ia, INT *ja,
 *                                          REAL *a, REAL *b, REAL *u, REAL *tol,
 *                                          INT *maxit, INT *ptrlvl)
 *
 * \brief Solve Ax=b by Krylov method preconditioned by ILUk
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param tol     Tolerance for iterative solvers
 * \param maxit   Max number of iterations
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   03/24/2018
 */
void fasp_fwrapper_dcsr_krylov_ilu_(INT*  n,
                                    INT*  nnz,
                                    INT*  ia,
                                    INT*  ja,
                                    REAL* a,
                                    REAL* b,
                                    REAL* u,
                                    REAL* tol,
                                    INT*  maxit,
                                    INT*  ptrlvl)
{
    dCSRmat   mat;      // coefficient matrix
    dvector   rhs, sol; // right-hand-side, solution
    ILU_param iluparam; // parameters for ILU
    ITS_param itsparam; // parameters for itsolver

    // setup ILU parameters
    fasp_param_ilu_init(&iluparam);

    iluparam.print_level = *ptrlvl;

    // setup Krylov method parameters
    fasp_param_solver_init(&itsparam);

    itsparam.itsolver_type = SOLVER_VFGMRES;
    itsparam.tol           = *tol;
    itsparam.maxit         = *maxit;
    itsparam.print_level   = *ptrlvl;

    // set up coefficient matrix
    mat.row = *n;
    mat.col = *n;
    mat.nnz = *nnz;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n;
    rhs.val = b;
    sol.row = *n;
    sol.val = u;

    fasp_solver_dcsr_krylov_ilu(&mat, &rhs, &sol, &itsparam, &iluparam);
}

/**
 * \fn void fasp_fwrapper_dcsr_krylov_amg_ (INT *n, INT *nnz, INT *ia, INT *ja,
 *                                          REAL *a, REAL *b, REAL *u, REAL *tol,
 *                                          INT *maxit, INT *ptrlvl)
 *
 * \brief Solve Ax=b by Krylov method preconditioned by classic AMG
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param tol     Tolerance for iterative solvers
 * \param maxit   Max number of iterations
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   09/16/2010
 */
void fasp_fwrapper_dcsr_krylov_amg_(INT*  n,
                                    INT*  nnz,
                                    INT*  ia,
                                    INT*  ja,
                                    REAL* a,
                                    REAL* b,
                                    REAL* u,
                                    REAL* tol,
                                    INT*  maxit,
                                    INT*  ptrlvl)
{
    dCSRmat     mat;      // coefficient matrix
    dvector     rhs, sol; // right-hand-side, solution
    input_param inparam;  // parameters from input files
    AMG_param   amgparam; // parameters for AMG
    ITS_param   itsparam; // parameters for itsolver
    ILU_param   iluparam; // parameters for ILU

    /** Step 0. Read input parameters */
    char* inputfile = "ini/amg.dat"; // Added for fasp4ns 2022.04.08 --zcs
    fasp_param_input(inputfile, &inparam);
    fasp_param_init(&inparam, &itsparam, &amgparam, &iluparam, NULL);

    itsparam.tol         = *tol;
    itsparam.maxit       = *maxit;
    itsparam.print_level = *ptrlvl;

    // set up coefficient matrix
    mat.row = *n;
    mat.col = *n;
    mat.nnz = *nnz;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n;
    rhs.val = b;
    sol.row = *n;
    sol.val = u;

    fasp_solver_dcsr_krylov_amg(&mat, &rhs, &sol, &itsparam, &amgparam);
}

/**
 * \fn void fasp_fwrapper_dbsr_krylov_ilu_ (INT *n, INT *nnz, INT *nb, INT *ia, INT *ja,
 *                                          REAL *a, REAL *b, REAL *u, REAL *tol,
 *                                          INT *maxit, INT *ptrlvl)
 *
 * \brief Solve Ax=b by Krylov method preconditioned by block ILU in BSR format
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param nb      Size of each small block
 * \param ia      IA of A in BSR format
 * \param ja      JA of A in BSR format
 * \param a       VAL of A in BSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param tol     Tolerance for iterative solvers
 * \param maxit   Max number of iterations
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   03/25/2018
 */
void fasp_fwrapper_dbsr_krylov_ilu_(INT*  n,
                                    INT*  nnz,
                                    INT*  nb,
                                    INT*  ia,
                                    INT*  ja,
                                    REAL* a,
                                    REAL* b,
                                    REAL* u,
                                    REAL* tol,
                                    INT*  maxit,
                                    INT*  ptrlvl)
{
    dBSRmat mat;        // coefficient matrix in BSR format
    dvector rhs, sol;   // right-hand-side, solution

    ILU_param iluparam; // parameters for ILU
    ITS_param itsparam; // parameters for itsolver

    // setup ILU parameters
    fasp_param_ilu_init(&iluparam);
    iluparam.ILU_lfil    = 0;
    iluparam.print_level = *ptrlvl;

    // setup Krylov method parameters
    fasp_param_solver_init(&itsparam);

    itsparam.itsolver_type = SOLVER_VFGMRES;
    itsparam.tol           = *tol;
    itsparam.maxit         = *maxit;
    itsparam.print_level   = *ptrlvl;

    // set up coefficient matrix
    mat.ROW = *n;
    mat.COL = *n;
    mat.NNZ = *nnz;
    mat.nb  = *nb;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n * *nb;
    rhs.val = b;
    sol.row = *n * *nb;
    sol.val = u;

    // solve
    fasp_solver_dbsr_krylov_ilu(&mat, &rhs, &sol, &itsparam, &iluparam);
}

/**
 * \fn void fasp_fwrapper_dbsr_krylov_amg_ (INT *n, INT *nnz, INT *nb, INT *ia, INT *ja,
 *                                          REAL *a, REAL *b, REAL *u, REAL *tol,
 *                                          INT *maxit, INT *ptrlvl)
 *
 * \brief Solve Ax=b by Krylov method preconditioned by block AMG in BSR format
 *
 * \param n       Number of cols of A
 * \param nnz     Number of nonzeros of A
 * \param nb      Size of each small block
 * \param ia      IA of A in CSR format
 * \param ja      JA of A in CSR format
 * \param a       VAL of A in CSR format
 * \param b       RHS vector
 * \param u       Solution vector
 * \param tol     Tolerance for iterative solvers
 * \param maxit   Max number of iterations
 * \param ptrlvl  Print level for iterative solvers
 *
 * \author Chensong Zhang
 * \date   04/05/2018
 */
void fasp_fwrapper_dbsr_krylov_amg_(INT*  n,
                                    INT*  nnz,
                                    INT*  nb,
                                    INT*  ia,
                                    INT*  ja,
                                    REAL* a,
                                    REAL* b,
                                    REAL* u,
                                    REAL* tol,
                                    INT*  maxit,
                                    INT*  ptrlvl)
{
    dBSRmat mat;        // coefficient matrix in CSR format
    dvector rhs, sol;   // right-hand-side, solution

    AMG_param amgparam; // parameters for AMG
    ITS_param itsparam; // parameters for itsolver

    // setup AMG parameters
    fasp_param_amg_init(&amgparam);
    amgparam.AMG_type    = UA_AMG;
    amgparam.print_level = *ptrlvl;

    // setup Krylov method parameters
    fasp_param_solver_init(&itsparam);
    itsparam.tol           = *tol;
    itsparam.print_level   = *ptrlvl;
    itsparam.maxit         = *maxit;
    itsparam.itsolver_type = SOLVER_VFGMRES;

    // set up coefficient matrix
    mat.ROW = *n;
    mat.COL = *n;
    mat.NNZ = *nnz;
    mat.nb  = *nb;
    mat.IA  = ia;
    mat.JA  = ja;
    mat.val = a;

    rhs.row = *n * *nb;
    rhs.val = b;
    sol.row = *n * *nb;
    sol.val = u;

    // solve
    fasp_solver_dbsr_krylov_amg(&mat, &rhs, &sol, &itsparam, &amgparam);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
