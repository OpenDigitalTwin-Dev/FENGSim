/*! \file  XtrUmfpack.c
 *
 *  \brief Interface to UMFPACK direct solvers
 *
 *  Reference for SuiteSparse:
 *  http://faculty.cse.tamu.edu/davis/suitesparse.html
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

#if WITH_UMFPACK
#include "umfpack.h"
#endif

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_solver_umfpack (dCSRmat *ptrA, dvector *b, dvector *u,
 *                              const SHORT prtlvl)
 *
 * \brief Solve Au=b by UMFpack
 *
 * \param ptrA         Pointer to a dCSRmat matrix
 * \param b            Pointer to the dvector of right-hand side term
 * \param u            Pointer to the dvector of solution
 * \param prtlvl       Output level
 *
 * \author Chensong Zhang
 * \date   05/20/2010
 *
 * Modified by Chensong Zhang on 02/27/2013 for new FASP function names.
 * Modified by Chensong Zhang on 08/14/2022 for checking return status.
 */
INT fasp_solver_umfpack(dCSRmat* ptrA, dvector* b, dvector* u, const SHORT prtlvl)
{

#if WITH_UMFPACK

    const INT n  = ptrA->col;
    INT*      Ap = ptrA->IA;
    INT*      Ai = ptrA->JA;
    double*   Ax = ptrA->val;

    INT    status = FASP_SUCCESS;
    void*  Symbolic;
    void*  Numeric;
    double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];

    if (prtlvl > PRINT_SOME) {
        Control[UMFPACK_PRL] = prtlvl;
    } else {
        Control[UMFPACK_PRL] = 0;
    }

#if DEBUG_MODE
    const INT m   = ptrA->row;
    const INT nnz = ptrA->nnz;
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    REAL start_time, end_time;
    fasp_gettime(&start_time);

    /* Symbolic factorization */
    status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK symbolic factorization failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    /* Numeric factorization */
    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK numeric factorization failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    /* Clean up symbolic factorization */
    umfpack_di_free_symbolic(&Symbolic);

    /* Solve using after numeric factorization */
    status =
        umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, u->val, b->val, Numeric, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK solve failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    /* Clean up numeric factorization */
    umfpack_di_free_numeric(&Numeric);

    if (prtlvl > PRINT_MIN) {
        fasp_gettime(&end_time);
        fasp_cputime("UMFPACK costs", end_time - start_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return status;

#else

    printf("### ERROR: UMFPACK is not available!\n");
    return ERROR_SOLVER_EXIT;

#endif
}

#if WITH_UMFPACK
/**
 * \fn void* fasp_umfpack_factorize (dCSRmat *ptrA, const SHORT prtlvl)
 * \brief factorize A by UMFpack
 *
 * \param ptrA      Pointer to stiffness matrix of levelNum levels
 * \param Numeric   Pointer to the numerical factorization
 *
 * \author Xiaozhe Hu
 * \date   10/20/2010
 */
void* fasp_umfpack_factorize(dCSRmat* ptrA, const SHORT prtlvl)
{
    INT       status = FASP_SUCCESS;
    const INT n      = ptrA->col;
    INT*      Ap     = ptrA->IA;
    INT*      Ai     = ptrA->JA;
    REAL*     Ax     = ptrA->val;
    void*     Symbolic;
    void*     Numeric;
    double    Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];

    if (prtlvl > PRINT_SOME) {
        Control[UMFPACK_PRL] = prtlvl;
    } else {
        Control[UMFPACK_PRL] = 0;
    }

#if DEBUG_MODE
    const INT m   = ptrA->row;
    const INT nnz = ptrA->nnz;
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    REAL start_time, end_time;
    fasp_gettime(&start_time);

    /* Symbolic factorization */
    status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK symbolic factorization failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    /* Numeric factorization */
    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK numeric factorization failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    umfpack_di_free_symbolic(&Symbolic);

    if (prtlvl > PRINT_MIN) {
        fasp_gettime(&end_time);
        fasp_cputime("UMFPACK setup", end_time - start_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return Numeric;
}

/**
 * \fn INT fasp_umfpack_solve (dCSRmat *ptrA, dvector *b, dvector *u,
 *                             void *Numeric, const SHORT prtlvl)
 * \brief Solve Au=b by UMFpack, numerical factorization is given
 *
 * \param ptrA      Pointer to stiffness matrix of levelNum levels
 * \param b         Pointer to the dvector of right hand side term
 * \param u         Pointer to the dvector of dofs
 * \param Numeric   Pointer to the numerical factorization
 * \param prtlvl    Output level
 *
 * \author Xiaozhe Hu
 * \date   10/20/2010
 */
INT fasp_umfpack_solve(dCSRmat* ptrA, dvector* b, dvector* u, void* Numeric,
                       const SHORT prtlvl)
{
    INT     status = FASP_SUCCESS;
    INT*    Ap     = ptrA->IA;
    INT*    Ai     = ptrA->JA;
    double* Ax     = ptrA->val;
    double  Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];

    if (prtlvl > PRINT_SOME) {
        Control[UMFPACK_PRL] = prtlvl;
    } else {
        Control[UMFPACK_PRL] = 0;
    }

#if DEBUG_MODE
    const INT m   = ptrA->row;
    const INT n   = ptrA->col;
    const INT nnz = ptrA->nnz;
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    REAL start_time, end_time;
    fasp_gettime(&start_time);

    /* Solve using after numeric factorization */
    status =
        umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, u->val, b->val, Numeric, Control, Info);
    if (prtlvl > PRINT_MORE) umfpack_di_report_status(Control, status);
    if (prtlvl > PRINT_MOST) umfpack_di_report_control(Control);
    if (prtlvl > PRINT_MORE) umfpack_di_report_info(Control, Info);
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: UMFPACK solve failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    if (prtlvl > PRINT_NONE) {
        fasp_gettime(&end_time);
        fasp_cputime("UMFPACK solve", end_time - start_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn INT fasp_umfpack_free_numeric (void *Numeric)
 * \brief Solve Au=b by UMFpack
 *
 * \param Numeric   Pointer to the numerical factorization
 *
 * \author Xiaozhe Hu
 * \date   10/20/2010
 */
INT fasp_umfpack_free_numeric(void* Numeric)
{
    INT status = FASP_SUCCESS;

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
#endif

    umfpack_di_free_numeric(&Numeric);

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return status;
}

#endif

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
