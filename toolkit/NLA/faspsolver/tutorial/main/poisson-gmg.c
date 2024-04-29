/*! \file  poisson-gmg.c
 *
 *  \brief The fourth test example for FASP: using GMG to solve
 *         the discrete Poisson equation from five-point finite
 *         difference stencil. C version.
 *
 *  \note  Solving the Poisson equation (FDM) with GMG: C version
 *
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
 * \fn static REAL f2d(INT i, INT j, INT nx, INT ny)
 *
 * \brief Setting f in Poisson equation, where
 *        f = sin(pi x)*sin(pi y)
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
    return sin(pi *(((REAL) j)/((REAL) ny)))
          *sin(pi *(((REAL) i)/((REAL) nx)));
}

/**
 * \fn static REAL L2NormError2d(REAL *u, INT nx, INT ny)
 *
 * \brief Computing Discretization Error, where exact solution
 *        u = sin(pi x)*sin(pi y)/(2*pi*pi)
 *
 * \param u      Vector of DOFs
 * \param nx     Number of grids in x direction
 * \param ny     Number of grids in y direction
 *
 * \author Ziteng Wang
 * \date   06/07/2013
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
    
    return sqrt(l2norm*h*h);
}

/**
 * \brief An example of GMG method using Full Multigrid cycle
 *
 * \author Chensong Zhang
 * \date   10/12/2015
 *
 * \note   Number of grids of nx = ny should be equal to 2^maxlevel.
 */
int main (int argc, const char *argv[])
{
    const REAL rtol   = 1.0e-6;
    const INT  prtlvl = PRINT_MORE;
    
    INT        i, j, nx, maxlevel;
    REAL      *u, *b, h, error0;
    
    // Step 0. Set number of levels for GMG
    printf("Enter the desired number of levels:   ");
    if ( scanf("%d", &maxlevel) > 1 ) {
        printf("### ERROR: Did not get a valid input !!!\n");
        return ERROR_INPUT_PAR;
    }
    
    // Step 1. Compute right-hand side b and set approximate solution u
    nx = (int) pow(2.0, maxlevel);
    h = 1.0/((REAL) nx);
    
    u = (REAL *)malloc((nx+1)*(nx+1)*sizeof(REAL));
    fasp_darray_set((nx+1)*(nx+1), u, 0.0);
    
    b = (REAL *)malloc((nx+1)*(nx+1)*sizeof(REAL));
    for (i = 0; i <= nx; i++) {
        for (j = 0; j <= nx; j++) {
            b[j*(nx+1)+i] = h*h*f2d(i, j, nx, nx);
        }
    }
    
    // Step 2. Solve the Poisson system in 2D with full Multigrid cycle
    fasp_poisson_fgmg2d(u, b, nx, nx, maxlevel, rtol, prtlvl);
    
    // Step 3. Compute error in L2 norm
    error0 = L2NormError2d(u, nx, nx);
    
    printf("L2 error ||u-u'|| = %e\n",error0);
    
    // Step 4. Clean up memory
    free(u);
    free(b);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
