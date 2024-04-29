/*! \file  SolGMGPoisson.c
 *
 *  \brief GMG method as an iterative solver for Poisson Problem
 *
 *  \note  This file contains Level-5 (Sol) functions. It requires:
 *         AuxArray.c, AuxMessage.c, and AuxTiming.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>
#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "PreGMG.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_poisson_gmg1d (REAL *u, REAL *b, const INT nx, const INT maxlevel,
 *                             const REAL rtol, const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 1D equation by Geometric Multigrid Method
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmg1d (REAL         *u,
                        REAL         *b,
                        const INT     nx,
                        const INT     maxlevel,
                        const REAL    rtol,
                        const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL      *u0, *r0, *b0;
    REAL       norm_r, norm_r0, norm_r1, factor, error = BIGREAL;
    INT        i, *level, count = 0;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, maxlevel=%d\n", nx, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", nx+1);
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = nx+1;
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(level[i]-level[i-1]+1)/2;
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    
    fasp_darray_cp(nx, u, u0);
    fasp_darray_cp(nx, b, b0);
    
    // compute initial l2 norm of residue
    fasp_darray_set(level[1], r0, 0.0);
    residual1d(u0, b0, r0, 0, level);
    norm_r0 = l2norm(r0, level, 0);
    norm_r1 = norm_r0;
    if (norm_r0 < atol) goto FINISHED;
    
    if ( prtlvl > PRINT_SOME ){
        printf("-----------------------------------------------------------\n");
        printf("It Num |   ||r||/||b||   |     ||r||      |  Conv. Factor\n");
        printf("-----------------------------------------------------------\n");
    }
    
    // GMG solver of V-cycle
    while (count < max_itr_num) {
        count++;
        mg1d(u0, b0, level, 0, maxlevel);
        residual1d(u0, b0, r0, 0, level);
        norm_r = l2norm(r0, level, 0);
        factor = norm_r/norm_r1;
        error = norm_r / norm_r0;
        norm_r1 = norm_r;
        if ( prtlvl > PRINT_SOME ){
            printf("%6d | %13.6e   | %13.6e  | %10.4f\n",count,error,norm_r,factor);
        }
        if (error < rtol || norm_r < atol) break;
    }
    
    if ( prtlvl > PRINT_NONE ){
        if (count >= max_itr_num) {
            printf("### WARNING: V-cycle failed to converge.\n");
        }
        else {
            printf("Num of V-cycle's: %d, Relative Residual = %e.\n", count, error);
        }
    }
    
    // Update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG totally", AMG_end - AMG_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
FINISHED:
    free(level);
    free(r0);
    free(u0);
    free(b0);
    
    return count;
}

/**
 * \fn INT fasp_poisson_gmg2d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                             const INT maxlevel, const REAL rtol,
 *                             const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 2D equation by Geometric Multigrid Method
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmg2d (REAL         *u,
                        REAL         *b,
                        const INT     nx,
                        const INT     ny,
                        const INT     maxlevel,
                        const REAL    rtol,
                        const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL *u0, *b0, *r0;
    REAL norm_r, norm_r0, norm_r1, factor, error = BIGREAL;
    INT i, k, count = 0, *nxk, *nyk, *level;
    REAL AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, maxlevel=%d\n", nx, ny, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1));
    }
    
    // set nxk, nyk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    nxk[0] = nx+1; nyk[0] = ny+1;
    for (k=1;k<maxlevel;k++) {
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0
    u0 = (REAL *)malloc(level[maxlevel+1]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel+1]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel+1]*sizeof(REAL));
    
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    residual2d(u0, b0, r0, 0, level, nxk, nyk);
    norm_r0 = l2norm(r0, level, 0);
    norm_r1 = norm_r0;
    if (norm_r0 < atol) goto FINISHED;
    
    if ( prtlvl > PRINT_SOME ){
        printf("-----------------------------------------------------------\n");
        printf("It Num |   ||r||/||b||   |     ||r||      |  Conv. Factor\n");
        printf("-----------------------------------------------------------\n");
    }
    
    // GMG solver of V-cycle
    while ( count < max_itr_num ) {
        count++;
        mg2d(u0, b0, level, 0, maxlevel, nxk, nyk);
        residual2d(u0, b0, r0, 0, level, nxk, nyk);
        norm_r = l2norm(r0, level, 0);
        error = norm_r / norm_r0;
        factor = norm_r/norm_r1;
        norm_r1 = norm_r;
        if ( prtlvl > PRINT_SOME ){
            printf("%6d | %13.6e   | %13.6e  | %10.4f\n",count,error,norm_r,factor);
        }
        if ( error < rtol || norm_r < atol ) break;
    }
    
    if ( prtlvl > PRINT_NONE ){
        if (count >= max_itr_num) {
            printf("### WARNING: V-cycle failed to converge.\n");
        }
        else {
            printf("Num of V-cycle's: %d, Relative Residual = %e.\n", count, error);
        }
    }
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG totally", AMG_end - AMG_start);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(u0);
    free(b0);
    free(r0);
    
    return count;
}

/**
 * \fn INT fasp_poisson_gmg3d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                             const INT nz, const INT maxlevel,
 *                             const REAL rtol, const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 3D equation by Geometric Multigrid Method
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param nz        Number of grids in z direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmg3d (REAL         *u,
                        REAL         *b,
                        const INT     nx,
                        const INT     ny,
                        const INT     nz,
                        const INT     maxlevel,
                        const REAL    rtol,
                        const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL       *u0, *r0, *b0;
    REAL       norm_r,norm_r0,norm_r1, factor, error = BIGREAL;
    INT        i, k, count = 0, *nxk, *nyk, *nzk, *level;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, nz=%d, maxlevel=%d\n",
           nx, ny, nz, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1)*(nz+1));
    }
    
    // set nxk, nyk, nzk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    nzk = (INT *)malloc(maxlevel*sizeof(INT));
    nxk[0] = nx+1; nyk[0] = ny+1; nzk[0] = nz+1;
    for(k=1;k<maxlevel;k++){
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
        nzk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1)*(nz+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1)*(nz/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    residual3d(u0, b0, r0, 0, level, nxk, nyk, nzk);
    norm_r0 = l2norm(r0, level, 0);
    norm_r1 = norm_r0;
    if (norm_r0 < atol) goto FINISHED;
    
    if ( prtlvl > PRINT_SOME ){
        printf("-----------------------------------------------------------\n");
        printf("It Num |   ||r||/||b||   |     ||r||      |  Conv. Factor\n");
        printf("-----------------------------------------------------------\n");
    }
    
    // GMG solver of V-cycle
    while (count < max_itr_num) {
        count++;
        mg3d(u0, b0, level, 0, maxlevel, nxk, nyk, nzk);
        residual3d(u0, b0, r0, 0, level, nxk, nyk, nzk);
        norm_r = l2norm(r0, level, 0);
        factor = norm_r/norm_r1;
        error = norm_r / norm_r0;
        norm_r1 = norm_r;
        if ( prtlvl > PRINT_SOME ){
            printf("%6d | %13.6e   | %13.6e  | %10.4f\n",count,error,norm_r,factor);
        }
        if (error < rtol || norm_r < atol) break;
    }
    
    if ( prtlvl > PRINT_NONE ){
        if (count >= max_itr_num) {
            printf("### WARNING: V-cycle failed to converge.\n");
        }
        else {
            printf("Num of V-cycle's: %d, Relative Residual = %e.\n", count, error);
        }
    }
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG totally", AMG_end - AMG_start);
    }
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(nzk);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return count;
}

/**
 * \fn void fasp_poisson_fgmg1d (REAL *u, REAL *b, const INT nx, const INT maxlevel,
 *                               const REAL rtol, const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 1D equation by Geometric Multigrid Method (FMG)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
void fasp_poisson_fgmg1d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL  atol = 1.0E-15;
    REAL        *u0, *r0, *b0;
    REAL        norm_r0, norm_r;
    INT         *level;
    REAL        AMG_start = 0, AMG_end;
    int         i;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, maxlevel=%d\n", nx, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1));
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = nx+1;
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(level[i]-level[i-1]+1)/2;
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(nx, u, u0);
    fasp_darray_cp(nx, b, b0);
    
    // compute initial l2 norm of residue
    fasp_darray_set(level[1], r0, 0.0);
    residual1d(u0, b0, r0, 0, level);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    //  Full GMG solver
    fmg1d(u0, b0, level, maxlevel, nx);
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out Relative Residual and CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("FGMG totally", AMG_end - AMG_start);
        residual1d(u0, b0, r0, 0, level);
        norm_r = l2norm(r0, level, 0);
        printf("Relative Residual = %e.\n", norm_r/norm_r0);
    }
    
FINISHED:
    free(level);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
}

/**
 * \fn void fasp_poisson_fgmg2d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                               const INT maxlevel, const REAL rtol, 
 *                               const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 2D equation by Geometric Multigrid Method (FMG)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in Y direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
void fasp_poisson_fgmg2d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     ny,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    REAL       *u0, *r0, *b0;
    REAL       norm_r0, norm_r;
    INT        *nxk, *nyk, *level;
    int        i, k;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, maxlevel=%d\n", nx, ny, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1));
    }
    
    // set nxk, nyk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    
    nxk[0] = nx+1; nyk[0] = ny+1;
    for(k=1;k<maxlevel;k++) {
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel] + 1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    fasp_darray_set(level[1], r0, 0.0);
    residual2d(u0, b0, r0, 0, level, nxk, nyk);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    // FMG solver
    fmg2d(u0, b0, level, maxlevel, nxk, nyk);
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out Relative Residual and CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("FGMG totally", AMG_end - AMG_start);
        residual2d(u0, b0, r0, 0, level, nxk, nyk);
        norm_r = l2norm(r0, level, 0);
        printf("Relative Residual = %e.\n", norm_r/norm_r0);
    }
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
}

/**
 * \fn void fasp_poisson_fgmg3d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                               const INT nz, const INT maxlevel, const REAL rtol,
 *                               const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 3D equation by Geometric Multigrid Method (FMG)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        NUmber of grids in y direction
 * \param nz        NUmber of grids in z direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
void fasp_poisson_fgmg3d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     ny,
                          const INT     nz,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL  atol = 1.0E-15;
    REAL        *u0, *r0, *b0;
    REAL        norm_r0, norm_r;
    INT         *nxk, *nyk, *nzk, *level;
    int         i, k;
    REAL        AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, nz=%d, maxlevel=%d\n",
           nx, ny, nz, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1)*(nz+1));
    }
    // set nxk, nyk, nzk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    nzk = (INT *)malloc(maxlevel*sizeof(INT));
    
    nxk[0] = nx+1; nyk[0] = ny+1; nzk[0] = nz+1;
    for(k=1;k<maxlevel;k++){
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
        nzk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1)*(nz+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1)*(nz/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    residual3d(u0, b0, r0, 0, level, nxk, nyk, nzk);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    // FMG
    fmg3d(u0, b0, level, maxlevel, nxk, nyk, nzk);
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("FGMG totally", AMG_end - AMG_start);
        residual3d(u0, b0, r0, 0, level, nxk, nyk, nzk);
        norm_r = l2norm(r0, level, 0);
        printf("Relative Residual = %e.\n", norm_r/norm_r0);
    }
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(nzk);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
}

/**
 * \fn INT fasp_poisson_gmgcg1d (REAL *u, REAL *b, const INT nx, const INT maxlevel,
 *                               const REAL rtol, const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 1D equation by Geometric Multigrid Method
 *        (GMG preconditioned Conjugate Gradient method)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmgcg1d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL       *u0, *r0, *b0;
    REAL       norm_r0;
    INT        *level;
    int        i, iter = 0;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, maxlevel=%d\n", nx, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1));
    }
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = nx+1;
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(level[i]-level[i-1]+1)/2;
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(nx, u, u0);
    fasp_darray_cp(nx, b, b0);
    
    // compute initial l2 norm of residue
    fasp_darray_set(level[1], r0, 0.0);
    residual1d(u, b, r0, 0, level);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    // Preconditioned CG method
    iter = pcg1d(u0, b0, level, maxlevel, nx, rtol, max_itr_num, prtlvl);
    
    // Update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG_PCG totally", AMG_end - AMG_start);
    }
    
FINISHED:
    free(level);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return iter;
}

/**
 * \fn INT fasp_poisson_gmgcg2d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                               const INT maxlevel, const REAL rtol,
 *                               const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 2D equation by Geometric Multigrid Method
 *        (GMG preconditioned Conjugate Gradient method)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmgcg2d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     ny,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL       *u0,*r0,*b0;
    REAL       norm_r0;
    INT        *nxk, *nyk, *level;
    int        i, k, iter = 0;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, maxlevel=%d\n", nx, ny, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1));
    }
    // set nxk, nyk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    
    nxk[0] = nx+1; nyk[0] = ny+1;
    for (k=1;k<maxlevel;k++) {
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0, r0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    fasp_darray_set(level[1], r0, 0.0);
    residual2d(u0, b0, r0, 0, level, nxk, nyk);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    // Preconditioned CG method
    iter = pcg2d(u0, b0, level, maxlevel, nxk,
                 nyk, rtol, max_itr_num, prtlvl);
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG_PCG totally", AMG_end - AMG_start);
    }
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return iter;
}

/**
 * \fn INT fasp_poisson_gmgcg3d (REAL *u, REAL *b, const INT nx, const INT ny,
 *                               const INT nz, const INT maxlevel, const REAL rtol,
 *                               const SHORT prtlvl)
 *
 * \brief Solve Ax=b of Poisson 3D equation by Geometric Multigrid Method
 *        (GMG preconditioned Conjugate Gradient method)
 *
 * \param u         Pointer to the vector of dofs
 * \param b         Pointer to the vector of right hand side
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param nz        Number of grids in z direction
 * \param maxlevel  Maximum levels of the multigrid
 * \param rtol      Relative tolerance to judge convergence
 * \param prtlvl    Print level for output
 *
 * \return          Iteration number if converges; ERROR otherwise.
 *
 * \author Ziteng Wang, Chensong Zhang
 * \date   06/07/2013
 */
INT fasp_poisson_gmgcg3d (REAL         *u,
                          REAL         *b,
                          const INT     nx,
                          const INT     ny,
                          const INT     nz,
                          const INT     maxlevel,
                          const REAL    rtol,
                          const SHORT   prtlvl)
{
    const REAL atol = 1.0E-15;
    const INT  max_itr_num = 100;
    
    REAL       *u0,*r0,*b0;
    REAL       norm_r0;
    INT        *nxk, *nyk, *nzk, *level;
    int        i, k, iter = 0;
    REAL       AMG_start = 0, AMG_end;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nx=%d, ny=%d, nz=%d, maxlevel=%d\n",
           nx, ny, nz, maxlevel);
#endif
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_start);
        printf("Num of DOF's: %d\n", (nx+1)*(ny+1)*(nz+1));
    }
    
    // set nxk, nyk, nzk
    nxk = (INT *)malloc(maxlevel*sizeof(INT));
    nyk = (INT *)malloc(maxlevel*sizeof(INT));
    nzk = (INT *)malloc(maxlevel*sizeof(INT));
    
    nxk[0] = nx+1; nyk[0] = ny+1; nzk[0] = nz+1;
    for (k = 1; k < maxlevel; k++ ) {
        nxk[k] = (int) (nxk[k-1]+1)/2;
        nyk[k] = (int) (nyk[k-1]+1)/2;
        nzk[k] = (int) (nyk[k-1]+1)/2;
    }
    
    // set level
    level = (INT *)malloc((maxlevel+2)*sizeof(INT));
    level[0] = 0; level[1] = (nx+1)*(ny+1)*(nz+1);
    for (i = 1; i < maxlevel; i++) {
        level[i+1] = level[i]+(nx/pow(2.0,i)+1)*(ny/pow(2.0,i)+1)*(nz/pow(2.0,i)+1);
    }
    level[maxlevel+1] = level[maxlevel]+1;
    
    // set u0, b0
    u0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    b0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    r0 = (REAL *)malloc(level[maxlevel]*sizeof(REAL));
    fasp_darray_set(level[maxlevel], u0, 0.0);
    fasp_darray_set(level[maxlevel], b0, 0.0);
    fasp_darray_set(level[maxlevel], r0, 0.0);
    fasp_darray_cp(level[1], u, u0);
    fasp_darray_cp(level[1], b, b0);
    
    // compute initial l2 norm of residue
    residual3d(u0, b0, r0, 0, level, nxk, nyk, nzk);
    norm_r0 = l2norm(r0, level, 0);
    if (norm_r0 < atol) goto FINISHED;
    
    // Preconditioned CG method
    iter = pcg3d(u0, b0, level, maxlevel, nxk,
                 nyk, nzk, rtol, max_itr_num, prtlvl);
    
    // update u
    fasp_darray_cp(level[1], u0, u);
    
    // print out CPU time if needed
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&AMG_end);
        fasp_cputime("GMG_PCG totally", AMG_end - AMG_start);
    }
    
FINISHED:
    free(level);
    free(nxk);
    free(nyk);
    free(nzk);
    free(r0);
    free(u0);
    free(b0);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return iter;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
