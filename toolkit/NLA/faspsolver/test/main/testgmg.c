/*! \file  testgmg.c
 *
 *  \brief The test function for FASP GMG ssolvers
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2013--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>
#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

const REAL pi = 3.14159265;

/**
 * \fn static REAL f1d (INT i, INT nx)
 *
 * \brief Setting f in Poisson equation, where
 *        f = sin(pi x)                 in 1D
 *
 * \param i      i-th position in x direction
 * \param nx     Number of grids in x direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 */
static REAL f1d (INT i,
                 INT nx)
{
    return sin(pi *(((REAL) i)/((REAL) nx)));
}

/**
 * \fn static REAL f2d (INT i, INT j, INT nx, INT ny)
 *
 * \brief Setting f in Poisson equation, where
 *        f = sin(pi x)*sin(pi y)       in 2D
 *
 * \param i      i-th position in x direction
 * \param j      j-th position in y direction
 * \param nx     Number of grids in x direction
 * \param ny     Number of grids in y direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 */
static REAL f2d (INT i,
                 INT j,
                 INT nx,
                 INT ny)
{
    return sin(pi *(((REAL) i)/((REAL) nx)))
          *sin(pi *(((REAL) j)/((REAL) ny)));
}

/**
 * \fn static REAL f3d (INT i, INT j, INT k, INT nx, INT ny, INT nz)
 * \brief Setting f in Poisson equation, where
 *        f = sin(pi x*sin(pi y)*sin(pi z) in 3D
 *
 * \param i      i-th position in x direction
 * \param j      j-th position in y direction
 * \param k      k-th position in y direction
 * \param nx     Number of grids in x direction
 * \param ny     Number of grids in y direction
 * \param nz     Number of grids in z direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 */
static REAL f3d (INT i,
                 INT j,
                 INT k,
                 INT nx,
                 INT ny,
                 INT nz)
{
    return sin(pi *(((REAL) i)/((REAL) nx)))
          *sin(pi *(((REAL) j)/((REAL) ny)))
          *sin(pi *(((REAL) k)/((REAL) nz)));
}

/**
 * \fn static REAL L2NormError1d (REAL *u, INT nx)
 *
 * \brief Computing Discretization Error, where exact solution
 *        u = sin(pi x)/(pi*pi)                         1D
 *
 * \param u      Vector of DOFs
 * \param nx     Number of grids in x direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 *
 * Modified by Chensong Zhang on 06/07/2013: bug fixed and revise the structure
 */
static REAL L2NormError1d (REAL *u,
                           INT nx)
{
    const REAL h = 1.0/nx;
    REAL l2norm  = 0.0, uexact;
    
    INT i;
    for ( i = 1; i < nx; i++ ) {
        uexact  = sin(pi*i*h)/(pi*pi);
        //uexact  = ((REAL) i)*h*(1-((REAL) i)*h)/2;
        l2norm += pow((u[i] - uexact), 2);
    }
    l2norm = sqrt(l2norm*h);
    
    return l2norm;
}

/**
 * \fn static REAL L2NormError2d(REAL *u, INT nx, INT ny)
 *
 * \brief Computing Discretization Error, where exact solution
 *        u = sin(pi x)*sin(pi y)/(2*pi*pi)             2D
 *
 * \param u      Vector of DOFs
 * \param nx     Number of grids in x direction
 * \param ny     Number of grids in y direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 *
 * Modified by Chensong Zhang on 06/07/2013: bug fixed and revise the structure
 */
static REAL L2NormError2d (REAL *u,
                           INT nx,
                           INT ny)
{
    const REAL h = 1.0/nx;
    REAL l2norm  = 0.0, uexact;
    
    INT i, j;
    for ( i = 1; i < ny; i++ ) {
        for ( j = 1; j < nx; j++ ) {
            uexact  = sin(pi*i*h)*sin(pi*j*h)/(pi*pi*2.0);
            l2norm += pow((u[i*(nx+1)+j] - uexact), 2);
        }
    }
    l2norm = sqrt(l2norm*h*h);
    
    return l2norm;
}

/**
 * \fn static REAL L2NormError3d (REAL *u, INT nx, INT ny, INT nz)
 *
 * \brief Computing Discretization Error, where exact solution
 *        u = sin(pi x)*sin(pi y)*sin(pi z)/(3*pi*pi)   3D
 *
 * \param u      Vector of DOFs
 * \param nx     Number of grids in x direction
 * \param ny     Number of grids in y direction
 * \param nz     Number of grids in z direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 *
 * Modified by Chensong Zhang on 06/07/2013: bug fixed and revise the structure
 */
static REAL L2NormError3d (REAL *u,
                           INT nx,
                           INT ny,
                           INT nz)
{
    const REAL h = 1.0/nx;
    REAL l2norm  = 0.0, uexact;
    
    INT i, j, k;
    for ( i = 1; i < nz; i++ ) {
        for ( j = 1; j < ny; j++ ) {
            for ( k = 1; k < nx; k++ ) {
                uexact  = sin(pi*i*h)*sin(pi*j*h)*sin(pi*k*h)/(pi*pi*3.0);
                l2norm += pow((u[i*(nx+1)*(ny+1)+j*(nx+1)+k] - uexact), 2);
            }
        }
    }
    l2norm = sqrt(l2norm*h*h*h);
    
    return l2norm;
}

/**
 * \brief An example of GMG method: V-cycle = 1, FMG = 2, PCG = 3
 *
 * \author Ziteng Wang
 * \date   06/07/2013
 *
 * \note   Number of grids of nx, ny, ny should be all equal to 2^maxlevel.
 *
 * Modified by Chensong Zhang on 06/07/2013: bug fixed and revise the structure
 * Modified by Ziteng Wang and Chensong on 07/04/2013: print level control
 */
int main (int argc, const char *argv[])
{
    const REAL rtol = 1.0e-6;
    const INT  prtlvl = PRINT_MORE;
    
    INT        maxlevel = 3, dim = 3, method = 2;
    INT        i, j, k, nx = 8, ny = 8, nz = 8;
    REAL      *u, *b, h, error0;
    
    printf("Enter spatial dimension (1, 2 or 3):   ");
    
    if ( scanf("%d", &dim) > 1 ) {
        printf("### ERROR: Did not get a valid input !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    if ( dim > 3 || dim < 1) {
        printf("### ERROR: Wrong dimension number !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    printf("Choosing solver (V-cycle=1, FMG=2, PCG=3):   ");
    
    if ( scanf("%d", &method) > 1 ) {
        printf("### ERROR: Did not get a valid input !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    if ( method > 3 || method < 1) {
        printf("### ERROR: Unknown solver type !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    printf("Enter the desired number of levels:   ");
    if ( scanf("%d", &maxlevel) > 1 ) {
        printf("### ERROR: Did not get a valid input !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    nx = (int) pow(2.0, maxlevel);
    if ( dim > 1 ) ny = (int) pow(2.0, maxlevel);
    if ( dim > 2 ) nz = (int) pow(2.0, maxlevel);
    h = 1.0/((REAL) nx);
    
    switch (dim) {
            
        case 1: // 1 dimension
            
            u = (REAL *)malloc((nx+1)*sizeof(REAL));
            fasp_darray_set(nx+1, u, 0.0);
            
            b = (REAL *)malloc((nx+1)*sizeof(REAL));
            for (i = 0; i <= nx; i++) b[i] = h*h*f1d(i, nx);
            
            switch (method) {
                    
                case 1: // V-cycle
                    fasp_poisson_gmg1d(u, b, nx, maxlevel, rtol, prtlvl);
                    break;
                    
                case 2: // FMG
                    fasp_poisson_fgmg1d(u, b, nx, maxlevel, rtol, prtlvl);
                    break;
                    
                case 3: // PCG
                    fasp_poisson_gmgcg1d(u, b, nx, maxlevel, rtol, prtlvl);
                    break;
                    
            }
            
            break;
            
        case 2: // 2 dimension
            
            u = (REAL *)malloc((nx+1)*(ny+1)*sizeof(REAL));
            fasp_darray_set((nx+1)*(ny+1), u, 0.0);
            
            b = (REAL *)malloc((nx+1)*(ny+1)*sizeof(REAL));
            for (i = 0; i <= nx; i++) {
                for (j = 0; j <= ny; j++) {
                    b[j*(nx+1)+i] = h*h*f2d(i, j, nx, ny);
                }
            }
            
            switch (method) {
                    
                case 1: // V-cycle
                    fasp_poisson_gmg2d(u, b, nx, ny, maxlevel, rtol, prtlvl);
                    break;
                    
                case 2: // FMG
                    fasp_poisson_fgmg2d(u, b, nx, ny, maxlevel, rtol, prtlvl);
                    break;
                    
                case 3: // PCG
                    fasp_poisson_gmgcg2d(u, b, nx, ny, maxlevel, rtol, prtlvl);
                    break;
                    
            }
            
            break;
            
        default: // 3 dimension
            
            u = (REAL *)malloc((nx+1)*(ny+1)*(nz+1)*sizeof(REAL));
            fasp_darray_set((nx+1)*(ny+1)*(nz+1), u, 0.0);
            
            b = (REAL *)malloc((nx+1)*(ny+1)*(nz+1)*sizeof(REAL));
            for (i = 0; i <= nx; i++) {
                for (j = 0; j <= ny; j++) {
                    for (k = 0; k <= nz; k++) {
                        b[i+j*(nx+1)+k*(nx+1)*(ny+1)] = h*h*f3d(i,j,k,nx,ny,nz);
                    }
                }
            }
            
            switch (method) {
                    
                case 1: // V-cycle
                    fasp_poisson_gmg3d(u, b, nx, ny, nz, maxlevel, rtol, prtlvl);
                    break;
                    
                case 2: // FMG
                    fasp_poisson_fgmg3d(u, b, nx, ny, nz, maxlevel, rtol, prtlvl);
                    break;
                    
                case 3: // PCG
                    fasp_poisson_gmgcg3d(u, b, nx, ny, nz, maxlevel, rtol, prtlvl);
                    break;
                    
            }
            
            break;
            
    }
    
    if ( prtlvl >= PRINT_SOME){
        switch (dim) {
                
            case 1: // 1 dimension
                error0 = L2NormError1d(u, nx); break;
                
            case 2: // 2 dimension
                error0 = L2NormError2d(u, nx, ny); break;
                
            default: // 3 dimension
                error0 = L2NormError3d(u, nx, ny, nz); break;
                
        }
        printf("||u-u'|| = %e\n",error0);
    }
    
    free(u);
    free(b);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
