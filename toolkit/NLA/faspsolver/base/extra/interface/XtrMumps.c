/*! \file  XtrMumps.c
 *
 *  \brief Interface to MUMPS direct solvers
 *
 *  Reference for MUMPS:
 *  http://mumps.enseeiht.fr/
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2013--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

#if WITH_MUMPS
#include "dmumps_c.h"
#endif

#define ICNTL(I) icntl[(I)-1] /**< macro s.t. indices match documentation */

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn int fasp_solver_mumps (dCSRmat *ptrA, dvector *b, dvector *u,
 *                            const SHORT prtlvl)
 *
 * \brief Solve Ax=b by MUMPS directly
 *
 * \param ptrA      Pointer to a dCSRmat matrix
 * \param b         Pointer to the dvector of right-hand side term
 * \param u         Pointer to the dvector of solution
 * \param prtlvl    Output level
 *
 * \author Chunsheng Feng
 * \date   02/27/2013
 *
 * Modified by Chensong Zhang on 02/27/2013 for new FASP function names.
 */
int fasp_solver_mumps(dCSRmat* ptrA, dvector* b, dvector* u, const SHORT prtlvl)
{

#if WITH_MUMPS

    DMUMPS_STRUC_C id;

    int       status = FASP_SUCCESS;
    const int n      = ptrA->row;
    const int nz     = ptrA->nnz;
    int*      IA     = ptrA->IA;
    int*      JA     = ptrA->JA;
    double*   AA     = ptrA->val;
    double*   f      = b->val;
    double*   x      = u->val;

    int*    irn;
    int*    jcn;
    double* a;
    double* rhs;
    int     i, j;
    int     begin_row, end_row;

#if DEBUG_MODE
    printf("### DEBUG: fasp_solver_mumps ... [Start]\n");
    printf("### DEBUG: nr=%d,  nnz=%d\n", n, nz);
#endif

    // First check the matrix format
    if (IA[0] != 0 && IA[0] != 1) {
        printf("### ERROR: Matrix format is wrong -- IA[0] = %d\n", IA[0]);
        return ERROR_SOLVER_EXIT;
    }

    REAL start_time, end_time;
    fasp_gettime(&start_time);

    /* Define A and rhs */
    irn = (int*)malloc(sizeof(int) * nz);
    jcn = (int*)malloc(sizeof(int) * nz);
    a   = (double*)malloc(sizeof(double) * nz);
    rhs = (double*)malloc(sizeof(double) * n);

    if (IA[0] == 0) { // C-convention
        for (i = 0; i < n; i++) {
            begin_row = IA[i];
            end_row   = IA[i + 1];
            for (j = begin_row; j < end_row; j++) {
                irn[j] = i + 1;
                jcn[j] = JA[j] + 1;
                a[j]   = AA[j];
            }
        }
    } else { // F-convention
        for (i = 0; i < n; i++) {
            begin_row = IA[i] - 1;
            end_row   = IA[i + 1] - 1;
            for (j = begin_row; j < end_row; j++) {
                irn[j] = i + 1;
                jcn[j] = JA[j];
                a[j]   = AA[j];
            }
        }
    }

    /* Initialize a MUMPS instance. */
    id.job          = -1;
    id.par          = 1; // host involved in factorization/solve
    id.sym          = 0; // 0: general, 1: spd, 2: sym
    id.comm_fortran = 0;
    dmumps_c(&id);

    /* Define the problem on the host */
    id.n   = n;
    id.nz  = nz;
    id.irn = irn;
    id.jcn = jcn;
    id.a   = a;
    id.rhs = rhs;

    if (prtlvl > PRINT_MORE) { // enable debug output
        id.ICNTL(1) = 6;       // err output stream
        id.ICNTL(2) = 6;       // warn/info output stream
        id.ICNTL(3) = 6;       // global output stream
        id.ICNTL(4) = 3;       // 0:none, 1:err, 2:warn/stats, 3:diagnos, 4:parameters
        if (prtlvl > PRINT_MOST) id.ICNTL(11) = 1; // 0:none, 1:all, 2:main
    } else {
        id.ICNTL(1) = -1;
        id.ICNTL(2) = -1;
        id.ICNTL(3) = -1;
        id.ICNTL(4) = 0;
    }

    /* Call the MUMPS package */
    for (i = 0; i < n; i++) rhs[i] = f[i];

    id.job = 6;    /* Combines phase 1, 2, and 3 */
    dmumps_c(&id); /* Sometimes segmentation faults in MUMPS-5.0.0 */
    status = id.info[0];
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: MUMPS solve failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    for (i = 0; i < n; i++) x[i] = id.rhs[i];

    id.job = -2;
    dmumps_c(&id); /* Terminate instance */

    free(irn);
    free(jcn);
    free(a);
    free(rhs);

    if (prtlvl > PRINT_MIN) {
        fasp_gettime(&end_time);
        fasp_cputime("MUMPS solver", end_time - start_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: fasp_solver_mumps ... [Finish]\n");
#endif
    return status;

#else

    printf("### ERROR: MUMPS is not available!\n");
    return ERROR_SOLVER_EXIT;

#endif
}

/**
 * \fn int fasp_solver_mumps_steps (dCSRmat *ptrA, dvector *b, dvector *u,
 *                                  Mumps_data *mumps)
 *
 * \brief Solve Ax=b by MUMPS in three steps
 *
 * \param ptrA   Pointer to a dCSRmat matrix
 * \param b      Pointer to the dvector of right-hand side term
 * \param u      Pointer to the dvector of solution
 * \param mumps  Pointer to MUMPS data
 *
 * \author Chunsheng Feng
 * \date   02/27/2013
 *
 * Modified by Chensong Zhang on 02/27/2013 for new FASP function names.
 * Modified by Zheng Li on 10/10/2014 to adjust input parameters.
 * Modified by Chunsheng Feng on 08/11/2017 for debug information.
 */
int fasp_solver_mumps_steps(dCSRmat* ptrA, dvector* b, dvector* u, Mumps_data* mumps)
{
#if WITH_MUMPS

    DMUMPS_STRUC_C id;

    int        status   = FASP_SUCCESS;
    int        job      = mumps->job;
    static int job_stat = 0;
    int        i, j;
    int*       irn;
    int*       jcn;
    double*    a;
    double*    rhs;

    switch (job) {

        case 1:
            {
#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Start]\n",
                       __FUNCTION__, job, job_stat);
#endif
                int       begin_row, end_row;
                const int n  = ptrA->row;
                const int nz = ptrA->nnz;
                int*      IA = ptrA->IA;
                int*      JA = ptrA->JA;
                double*   AA = ptrA->val;

                irn = id.irn = (int*)malloc(sizeof(int) * nz);
                jcn = id.jcn = (int*)malloc(sizeof(int) * nz);
                a = id.a = (double*)malloc(sizeof(double) * nz);
                rhs = id.rhs = (double*)malloc(sizeof(double) * n);

                // First check the matrix format
                if (IA[0] != 0 && IA[0] != 1) {
                    printf("### ERROR: Matrix format is wrong, IA[0] = %d!\n", IA[0]);
                    return ERROR_SOLVER_EXIT;
                }

                // Define A and rhs
                if (IA[0] == 0) { // C-convention
                    for (i = 0; i < n; i++) {
                        begin_row = IA[i];
                        end_row   = IA[i + 1];
                        for (j = begin_row; j < end_row; j++) {
                            irn[j] = i + 1;
                            jcn[j] = JA[j] + 1;
                            a[j]   = AA[j];
                        }
                    }
                } else { // F-convention
                    for (i = 0; i < n; i++) {
                        begin_row = IA[i] - 1;
                        end_row   = IA[i + 1] - 1;
                        for (j = begin_row; j < end_row; j++) {
                            irn[j] = i + 1;
                            jcn[j] = JA[j];
                            a[j]   = AA[j];
                        }
                    }
                }

                /* Initialize a MUMPS instance. */
                id.job          = -1;
                id.par          = 1;
                id.sym          = 0;
                id.comm_fortran = 0;
                dmumps_c(&id);

                /* Define the problem on the host */
                id.n   = n;
                id.nz  = nz;
                id.irn = irn;
                id.jcn = jcn;
                id.a   = a;
                id.rhs = rhs;

                /* No outputs */
                id.ICNTL(1) = -1;
                id.ICNTL(2) = -1;
                id.ICNTL(3) = -1;
                id.ICNTL(4) = 0;

                id.job = 4;
                dmumps_c(&id);
                status = id.info[0];
                if (status < 0) {
                    printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
                    printf("### ERROR: MUMPS factorization failed!\n");
                    exit(ERROR_SOLVER_MISC);
                }
                job_stat = 1;

                mumps->id = id;

#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Finish]\n",
                       __FUNCTION__, job, job_stat);
#endif
                break;
            }

        case 2:
            {
#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Start]\n",
                       __FUNCTION__, job, job_stat);
#endif
                id = mumps->id;

                if (job_stat != 1)
                    printf("### ERROR: %s setup failed!\n", __FUNCTION__);

                /* Call the MUMPS package. */
                for (i = 0; i < id.n; i++) id.rhs[i] = b->val[i];

                id.job = 3;
                dmumps_c(&id);
                status = id.info[0];
                if (status < 0) {
                    printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
                    printf("### ERROR: MUMPS solve failed!\n");
                    exit(ERROR_SOLVER_MISC);
                }

                for (i = 0; i < id.n; i++) u->val[i] = id.rhs[i];

#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Finish]\n",
                       __FUNCTION__, job, job_stat);
#endif
                break;
            }

        case 3:
            {
#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Start]\n",
                       __FUNCTION__, job, job_stat);
#endif
                id = mumps->id;

                if (job_stat != 1)
                    printf("### ERROR: %s setup failed!\n", __FUNCTION__);

                free(id.irn);
                free(id.jcn);
                free(id.a);
                free(id.rhs);
                id.job = -2;
                dmumps_c(&id); /* Terminate instance */

#if DEBUG_MODE
                printf("### DEBUG: %s, step %d, job_stat = %d... [Finish]\n",
                       __FUNCTION__, job, job_stat);
#endif

                break;
            }

        default:
            printf("### ERROR: job = %d. Should be 1, 2, or 3!\n", job);
            return ERROR_SOLVER_EXIT;
    }

    return status;

#else

    printf("### ERROR: MUMPS is not available!\n");
    return ERROR_SOLVER_EXIT;

#endif
}

#if WITH_MUMPS
/**
 * \fn DMUMPS_STRUC_C fasp_mumps_factorize (dCSRmat *ptrA, dvector *b, dvector *u,
 *                                          const SHORT prtlvl)
 * \brief Factorize A by MUMPS
 *
 * \param ptrA     Pointer to stiffness matrix of levelNum levels
 * \param b        Pointer to the dvector of right hand side term
 * \param u        Pointer to the dvector of dofs
 * \param prtlvl   output level
 *
 * \author Zheng Li
 * \date   10/09/2014
 */
Mumps_data fasp_mumps_factorize(dCSRmat* ptrA, dvector* b, dvector* u,
                                const SHORT prtlvl)
{
    Mumps_data     mumps;
    DMUMPS_STRUC_C id;

    int       status = FASP_SUCCESS;
    const int n      = ptrA->col;
    const int nz     = ptrA->nnz;
    int*      IA     = ptrA->IA;
    int*      JA     = ptrA->JA;
    double*   AA     = ptrA->val;
    int       i, j;

    int*    irn = id.irn = (int*)malloc(sizeof(int) * nz);
    int*    jcn = id.jcn = (int*)malloc(sizeof(int) * nz);
    double* a = id.a = (double*)malloc(sizeof(double) * nz);
    double* rhs = id.rhs = (double*)malloc(sizeof(double) * n);

    int begin_row, end_row;

#if DEBUG_MODE
    printf("### DEBUG: %s ... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", n, n, nz);
#endif

    clock_t start_time = clock();

    if (IA[0] == 0) { // C-convention
        for (i = 0; i < n; i++) {
            begin_row = IA[i];
            end_row   = IA[i + 1];
            for (j = begin_row; j < end_row; j++) {
                irn[j] = i + 1;
                jcn[j] = JA[j] + 1;
                a[j]   = AA[j];
            }
        }
    } else { // F-convention
        for (i = 0; i < n; i++) {
            begin_row = IA[i] - 1;
            end_row   = IA[i + 1] - 1;
            for (j = begin_row; j < end_row; j++) {
                irn[j] = i + 1;
                jcn[j] = JA[j];
                a[j]   = AA[j];
            }
        }
    }

    /* Initialize a MUMPS instance. */
    id.job          = -1;
    id.par          = 1;
    id.sym          = 0;
    id.comm_fortran = 0;
    dmumps_c(&id);

    /* Define the problem on the host */
    id.n   = n;
    id.nz  = nz;
    id.irn = irn;
    id.jcn = jcn;
    id.a   = a;
    id.rhs = rhs;

    if (prtlvl > PRINT_MORE) { // enable debug output
        id.ICNTL(1) = 6;       // err output stream
        id.ICNTL(2) = 6;       // warn/info output stream
        id.ICNTL(3) = 6;       // global output stream
        id.ICNTL(4) = 3;       // 0:none, 1:err, 2:warn/stats, 3:diagnos, 4:parameters
        if (prtlvl > PRINT_MOST) id.ICNTL(11) = 1; // 0:none, 1:all, 2:main
    } else {
        id.ICNTL(1) = -1;
        id.ICNTL(2) = -1;
        id.ICNTL(3) = -1;
        id.ICNTL(4) = 0;
    }

    id.job = 4;
    dmumps_c(&id);
    status = id.info[0];
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: MUMPS factorization failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    if (prtlvl > PRINT_MIN) {
        clock_t end_time = clock();
        double  fac_time = (double)(end_time - start_time) / (double)(CLOCKS_PER_SEC);
        printf("MUMPS factorize costs %f seconds.\n", fac_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: %s ... [Finish]\n", __FUNCTION__);
#endif

    mumps.id = id;

    return mumps;
}
#endif

#if WITH_MUMPS
/**
 * \fn void fasp_mumps_solve (dCSRmat *ptrA, dvector *b, dvector *u,
 *                            Mumps_data mumps, const SHORT prtlvl)
 * \brief Solve A by MUMPS
 *
 * \param ptrA      Pointer to stiffness matrix of levelNum levels
 * \param b         Pointer to the dvector of right hand side term
 * \param u         Pointer to the dvector of dofs
 * \param mumps     Pointer to mumps data
 * \param prtlvl    Output level
 *
 * \author Zheng Li
 * \date   10/09/2014
 */
void fasp_mumps_solve(dCSRmat* ptrA, dvector* b, dvector* u, Mumps_data mumps,
                      const SHORT prtlvl)
{
    int            i;
    DMUMPS_STRUC_C id     = mumps.id;
    int            status = FASP_SUCCESS;
    double*        rhs    = id.rhs;

#if DEBUG_MODE
    printf("### DEBUG: %s ... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nz);
#endif

    clock_t start_time = clock();

    double* f = b->val;
    double* x = u->val;

    /* Call the MUMPS package. */
    for (i = 0; i < id.n; i++) rhs[i] = f[i];

    if (prtlvl > PRINT_MORE) { // enable debug output
        id.ICNTL(1) = 6;       // err output stream
        id.ICNTL(2) = 6;       // warn/info output stream
        id.ICNTL(3) = 6;       // global output stream
        id.ICNTL(4) = 3;       // 0:none, 1:err, 2:warn/stats, 3:diagnos, 4:parameters
        if (prtlvl > PRINT_MOST) id.ICNTL(11) = 1; // 0:none, 1:all, 2:main
    } else {
        id.ICNTL(1) = -1;
        id.ICNTL(2) = -1;
        id.ICNTL(3) = -1;
        id.ICNTL(4) = 0;
    }

    id.job = 3;
    dmumps_c(&id);
    status = id.info[0];
    if (status < 0) {
        printf("### ERROR: %d, %s %d\n", status, __FUNCTION__, __LINE__);
        printf("### ERROR: MUMPS solve failed!\n");
        exit(ERROR_SOLVER_MISC);
    }

    for (i = 0; i < id.n; i++) x[i] = id.rhs[i];

    if (prtlvl > PRINT_NONE) {
        clock_t end_time   = clock();
        double  solve_time = (double)(end_time - start_time) / (double)(CLOCKS_PER_SEC);
        printf("MUMPS costs %f seconds.\n", solve_time);
    }

#if DEBUG_MODE
    printf("### DEBUG: %s ... [Finish]\n", __FUNCTION__);
#endif
}
#endif

#if WITH_MUMPS
/**
 * \fn void fasp_mumps_free (Mumps_data *mumps)
 *
 * \brief Free MUMPS memory
 *
 * \param mumps   Pointer to mumps data
 *
 * \author Zheng Li
 * \date   10/09/2014
 */
void fasp_mumps_free(Mumps_data* mumps)
{
    DMUMPS_STRUC_C id = mumps->id;

    free(id.irn);
    free(id.jcn);
    free(id.a);
    free(id.rhs);
}
#endif

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
