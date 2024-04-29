/*! \file  FdmPoisson.c
 *
 *  \brief Setup FDM for the Poisson's equation
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "poisson_fdm.h"

/*!
 * \brief Generate the coefficient matrix, right hand side vector and true
 *        solution vector for the following Poisson Problem
 
 * Consider a two-dimensional Poisson equation
 * \f[
 *    \frac{du}{dt} - xcoeff * u_{xx} - ycoeff * u_{yy} = f(x,y)
 *    (x,y)\ in\ \Omega = (0,1)\times(0,1)
 * \f]
 * \f[
 *    u(x,y,0) = 0\ \ \ \ \ \ in\ \Omega = (0,1)\times(0,1)
 * \f]
 * \f[
 *    u = 0\ \ \ \ \ \ \ \ \ on\  \partial\Omega
 * \f]
 *
 *  where f(x,y,t) = \f$ 2*\pi^2*sin(\pi*x)*sin(\pi*y)*t + sin(\pi*x)*sin(\pi*y), \f$
 *  and the solution function can be expressed by
 *
 *             \f$ u(x,y,t) = sin(pi*x)*sin(pi*y)*t \f$
 *
 *  Sample grid: nx = 6; ny = 6
 *
 *           y
 *           |
 *           |_________________________________________________(1,1)
 *           |      |      |      |      |      |      |      |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |30    |31    |32    |33    |34    |35    |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |24    |25    |26    |27    |28    |29    |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |18    |19    |20    |21    |22    |23    |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |12    |13    |14    |15    |16    |17    |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |6     |7     |8     |9     |10    |11    |
 *           |      |      |      |      |      |      |      |
 *           |______|______|______|______|______|______|______|
 *           |      |0     |1     |2     |3     |4     |5     |
 *           |      |      |      |      |      |      |      |
 *     (0,0) |______|______|______|______|______|______|______|_______x
 *
 * \param nt number of nodes in t-direction (nt == 0 ==> the Poisson's equation)
 * \param nx number of nodes in x-direction (excluding the boundary nodes)
 * \param ny number of nodes in y-direction (excluding the boundary nodes)
 * \param A_ptr pointer to pointer to the coefficient matrix
 * \param f_ptr pointer to pointer to the right hand side vector
 * \param u_ptr pointer to pointer to the true solution vector
 *
 * \author Zhiyang Zhou
 * \date   2010/07/14
 *
 * Modified by Chensong Zhang on 05/01/2018: Add coefficients
 */
void
fsls_BuildLinearSystem_5pt2d (int               nt,
                              int               nx,
                              int               ny,
                              fsls_BandMatrix **A_ptr,
                              fsls_XVector    **f_ptr,
                              fsls_XVector    **u_ptr )
{
    fsls_BandMatrix *A = NULL;
    fsls_XVector    *f = NULL;
    fsls_XVector    *u = NULL;
    
    int      ngrid   = nx*ny;
    int      ngridxt = ngrid*nt;
    int      nband   = 8;
    int      nt_flag = 1;
    
    int     *offset  = NULL;
    double  *diag    = NULL;
    double **offdiag = NULL;
    
    double  *f_data  = NULL;
    double  *u_data  = NULL;
    
    int      nxplus1  = nx + 1;
    int      nyplus1  = ny + 1;
    int      nxminus1 = nx - 1;
    
    int      ngridminus1  = ngrid - 1;
    int      ngridminusnx = ngrid - nx;
    
    int      i,j,k,m;
    double   hx,hy,ht,x,y,s,t;
    double   hx2,hy2;
    double   dd;
    double   factorx  = 0.0;
    double   factory  = 0.0;
    double   constant = 2*PI*PI;
    double   xcoeff   = 1.0;
    double   ycoeff   = 1.0;

    A = fsls_BandMatrixCreate(ngrid, nband);
    fsls_BandMatrixInitialize(A);
    fsls_BandMatrixNx(A) = nx;
    fsls_BandMatrixNy(A) = ny;
    
    offset  = fsls_BandMatrixOffsets(A);
    diag    = fsls_BandMatrixDiag(A);
    offdiag = fsls_BandMatrixOffdiag(A);
    
    offset[0] = -1;
    offset[1] =  1;
    offset[2] = -nx;
    offset[3] =  nx;
    offset[4] = -nxplus1;
    offset[5] = -nxminus1;
    offset[6] =  nxminus1;
    offset[7] =  nxplus1;
    
    /* Generate matrix without BC processing */
    if (nt == 0) {
        ngridxt = ngrid;
        nt = 1;
        nt_flag = 0;
    }
    
    hx  = 1.0 / (double)nxplus1;
    hy  = 1.0 / (double)nyplus1;
    ht  = 1.0 / (double)nt;
    hx2 = hx*hx;
    hy2 = hy*hy;
    factorx = xcoeff / hx2;
    factory = ycoeff / hy2;
    dd = 2.0 * (factorx + factory);
    
    for (i = 0; i < ngrid; i ++) {
        diag[i] = dd;
        offdiag[0][i] = -factorx;
        offdiag[1][i] = -factorx;
        offdiag[2][i] = -factory;
        offdiag[3][i] = -factory;
    }
    
    /* zero-out some entries */
    offdiag[0][0] = 0.0;
    offdiag[2][0] = 0.0;
    offdiag[1][nxminus1] = 0.0;
    offdiag[2][nxminus1] = 0.0;
    offdiag[0][ngridminusnx] = 0.0;
    offdiag[3][ngridminusnx] = 0.0;
    offdiag[1][ngridminus1] = 0.0;
    offdiag[3][ngridminus1] = 0.0; /* the four vtx */
    
    for (i = 1; i < nxminus1; i ++) offdiag[2][i] = 0.0;

    for (i = ngrid-nxminus1; i < ngridminus1; i ++) offdiag[3][i] = 0.0;

    for (i = nx; i < ngridminusnx; i += nx) offdiag[0][i] = 0.0;

    for (i = 2*nx-1; i < ngridminus1; i += nx) offdiag[1][i] = 0.0;

    // generate the rhs and sol vector
    f = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(f);
    u = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(u);
    f_data = fsls_XVectorData(f);
    u_data = fsls_XVectorData(u);
    
    k = 0;
    double tmp;
    for (m = 0; m < nt; m ++) {
        t = ht*(m + 1);
        for (j = 0; j < ny; j ++) {
            y = hy*(j + 1);
            s = sin(PI*y);
            for (i = 0; i < nx; i ++) {
                x = hx*(i + 1);
                tmp = s*sin(PI*x);
                u_data[k] = tmp*t;
                f_data[k] = tmp*nt_flag + constant*u_data[k];
                k ++;
            }
        }
    }
    
    *A_ptr = A;
    *f_ptr = f;
    *u_ptr = u;
}

/*!
 * \brief Generate the coefficient matrix, right hand side vector and true
 *        solution vector for the Poisson Problem with red-black ordering
 *
 * \param nt number of nodes in t-direction, if nt == 0, it turn to be the Poisson system
 * \param nx number of nodes in x-direction (excluding the boundary nodes)
 * \param ny number of nodes in y-direction (excluding the boundary nodes)
 * \param A_ptr pointer to pointer to the coefficient matrix
 * \param f_ptr pointer to pointer to the right hand side vector
 * \param u_ptr pointer to pointer to the true solution vector
 *
 * \author peghoty
 * \date   2010/07/14
 */
void
fsls_BuildLinearSystem_5pt2d_rb (int               nt,
                                 int               nx,
                                 int               ny,
                                 fsls_BandMatrix **A_ptr,
                                 fsls_XVector    **f_ptr,
                                 fsls_XVector    **u_ptr )
{
    fsls_BandMatrix *A = NULL;
    fsls_XVector    *f = NULL;
    fsls_XVector    *u = NULL;
    
    int      ngrid   = nx*ny;
    int      ngridxt = ngrid*nt;
    int      nband   = 8;
    int      nt_flag = 1;
    
    int     *offset  = NULL;
    double  *diag    = NULL;
    double **offdiag = NULL;
    
    double  *f_data  = NULL;
    double  *u_data  = NULL;
    
    int      nxplus1  = nx + 1;
    int      nyplus1  = ny + 1;
    
    int      i,j,m;
    double   hx,hy,ht,x,y,s,t;
    double   hx2,hy2;
    double   dd;
    double   factorx  = 0.0;
    double   factory  = 0.0;
    double   constant = 2*PI*PI;
    double   xcoeff   = 1.0;
    double   ycoeff   = 1.0;

    A = fsls_BandMatrixCreate(ngrid, nband);
    fsls_BandMatrixInitialize(A);
    fsls_BandMatrixNx(A) = nx;
    fsls_BandMatrixNy(A) = ny;
    
    offset  = fsls_BandMatrixOffsets(A);
    diag    = fsls_BandMatrixDiag(A);
    offdiag = fsls_BandMatrixOffdiag(A);
    
    int n_red_odd, n_red_even, n_red, n_black_odd, n_black_even, n_black, n_odd, n_even;
    n_red_odd  = n_black_even = (nx + nx%2)/2;
    n_red_even = n_black_odd  = (nx - nx%2)/2;
    n_odd      = (ny + ny%2)/2;
    n_even     = (ny - ny%2)/2;
    n_red      = n_red_odd*n_odd + n_red_even*n_even;
    n_black    = n_black_odd*n_odd + n_black_even*n_even;
    
    offset[0] =  n_red - 1;
    offset[1] =  n_red;
    offset[2] =  n_red - n_red_odd;
    offset[3] =  n_red + n_red_even;
    offset[4] = -n_red;
    offset[5] = -n_red + 1;
    offset[6] = -n_red - n_black_odd;
    offset[7] = -n_red + n_black_even;
    
    /* Generate matrix without BC processing */
    if (nt == 0) {
        ngridxt = ngrid;
        nt = 1;
        nt_flag = 0;
    }
    
    hx  = 1.0 / (double)nxplus1;
    hy  = 1.0 / (double)nyplus1;
    ht  = 1.0 / (double)nt;
    hx2 = hx*hx;
    hy2 = hy*hy;
    factorx = xcoeff / hx2;
    factory = ycoeff / hy2;
    dd = 2*(factorx + factory);
    
    for (i = 0; i < n_red; i++) {
        diag[i] = dd;
        offdiag[0][i] = -factorx;
        offdiag[1][i] = -factorx;
        offdiag[2][i] = -factory;
        offdiag[3][i] = -factory;
    }
    
    for (i = n_red; i < n_red+n_black; i++) {
        diag[i] = dd;
        offdiag[4][i] = -factorx;
        offdiag[5][i] = -factorx;
        offdiag[6][i] = -factory;
        offdiag[7][i] = -factory;
    }
    
    /* zero-out some entries */
    offdiag[0][0] = 0.0;
    offdiag[2][0] = 0.0;
    offdiag[1][n_red_odd-1] = 0.0;
    offdiag[2][n_red_odd-1] = 0.0;
    offdiag[0][n_red-n_red_odd] = 0.0;
    offdiag[3][n_red-n_red_odd] = 0.0;
    offdiag[1][n_red-1] = 0.0;
    offdiag[3][n_red-1] = 0.0;/* the four vtx */
    
    for (i = 1; i < n_red_odd-1; i ++) {
        offdiag[2][i] = 0.0;
    }
    
    for (i = n_red; i < n_red+n_black_odd; i ++) {
        offdiag[6][i] = 0.0;
    }
    
    for (i = n_red-n_red_odd+1; i < n_red-1; i ++) {
        offdiag[3][i] = 0.0;
    }
    
    for (i = n_red+n_black-n_black_odd; i < n_red+n_black; i ++) {
        offdiag[7][i] = 0.0;
    }
    
    for (i = n_red_even+n_red_odd; i < n_red-n_red_odd; i += n_red_even+n_red_odd) {
        offdiag[0][i] = 0.0;
    }
    
    for (i = n_red+n_black_odd; i < n_red+n_black-n_black_odd-n_black_even+1; i += n_black_even+n_black_odd) {
        offdiag[4][i] = 0.0;
    }
    
    for (i = 2*n_red_odd+n_red_even-1; i < n_red-1; i += n_red_even+n_red_odd) {
        offdiag[1][i] = 0.0;
    }
    
    for (i = n_red+n_black_odd+n_black_even-1; i < n_red+n_black-n_black_odd+1; \
         i += n_black_even+n_black_odd) {
        offdiag[5][i] = 0.0;
    } /* the four lines */
    
    // generate the rhs and sol vector
    f = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(f);
    u = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(u);
    f_data = fsls_XVectorData(f);
    u_data = fsls_XVectorData(u);
    
    double tmp;
    for (m = 0; m < nt; m ++) {
        t = ht*(m + 1);
        for (j = 0; j < n_odd; j++) {
            y = 2*hy*j + hy;
            s = sin(PI*y);
            for (i = 0; i < n_red_odd; i++) {
                x = 2*hx*i + hx;
                tmp = s*sin(PI*x);
                u_data[i+j*(n_red_odd+n_red_even)+m*ngrid] = tmp*t;
                f_data[i+j*(n_red_odd+n_red_even)+m*ngrid] = tmp*nt_flag + constant*tmp*t;
            }
            for (i = 0; i < n_black_odd; i++) {
                x = 2*hx*(i + 1);
                tmp = s*sin(PI*x);
                u_data[n_red+i+j*(n_black_odd+n_black_even)+m*ngrid] = tmp*t;
                f_data[n_red+i+j*(n_black_odd+n_black_even)+m*ngrid] = tmp*nt_flag + constant*tmp*t;
            }
        }
        for (j = 0; j < n_even; j++) {
            y = 2*hy*(j + 1);
            s = sin(PI*y);
            for (i = 0; i < n_red_even; i++) {
                x = 2*hx*(i + 1);
                tmp = s*sin(PI*x);
                u_data[n_red_odd+i+j*(n_red_odd+n_red_even)+m*ngrid] = tmp*t;
                f_data[n_red_odd+i+j*(n_red_odd+n_red_even)+m*ngrid] = tmp*nt_flag + constant*tmp*t;
            }
            for (i = 0; i < n_black_even; i++) {
                x = 2*hx*i + hx;
                tmp = s*sin(PI*x);
                u_data[n_red+n_black_odd+i+j*(n_black_odd+n_black_even)+m*ngrid] = tmp*t;
                f_data[n_red+n_black_odd+i+j*(n_black_odd+n_black_even)+m*ngrid] = tmp*nt_flag + constant*tmp*t;
            }
        }
    }
    
    *A_ptr = A;
    *f_ptr = f;
    *u_ptr = u;
}

/*!
 * \brief Generate the coefficient matrix, right hand side vector and
 *        true solution vector for the following Poisson Problem
 
 * Consider a three-dimensional Poisson equation
 *
 * \f[
 *   \frac{du}{dt}-u_{xx}-u_{yy}-u_{zz} = f(x,y,t)\ \ in\ \Omega = (0,1)\times(0,1)\times(0,1)
 * \f]
 * \f[
 *                 u(x,y,z,0) = 0\ \ \ \ \ \ in\ \Omega
 * \f]
 * \f[
 *                        u = 0\ \ \ \ \ \ \ \ \ on\  \partial\Omega
 * \f]
 *
 *  where f(x,y,z,t) = \f$3*\pi^2*u(x,y,z,t) + sin(\pi*x)*sin(\pi*y)*sin(\pi*z)\f$,
 *  and the solution function can be expressed by
 *
 *             \f$u(x,y,z,t) = sin(\pi*x)*sin(\pi*y)*sin(\pi*z)\f$
 *
 * \param nt number of nodes in t-direction, if nt == 0, it turn to be the Poisson system
 * \param nx number of nodes in x-direction (excluding the boundary nodes)
 * \param ny number of nodes in y-direction (excluding the boundary nodes)
 * \param nz number of nodes in z-direction (excluding the boundary nodes)
 * \param A_ptr pointer to pointer to the coefficient matrix
 * \param f_ptr pointer to pointer to the right hand side vector
 * \param u_ptr pointer to pointer to the true solution vector
 *
 * \author peghoty
 * \date   2010/08/05
 */
void
fsls_BuildLinearSystem_7pt3d (int               nt,
                              int               nx,
                              int               ny,
                              int               nz,
                              fsls_BandMatrix **A_ptr,
                              fsls_XVector    **f_ptr,
                              fsls_XVector    **u_ptr )
{
    fsls_BandMatrix *A = NULL;
    fsls_XVector    *f = NULL;
    fsls_XVector    *u = NULL;
    
    int      ngrid   = nx*ny*nz;
    int      ngridxt = ngrid*nt;
    int      nplane  = nx*ny;
    int      nband   = 6;
    int      nt_flag = 1;
    
    int     *offset  = NULL;
    double  *diag    = NULL;
    double **offdiag = NULL;
    
    double  *f_data  = NULL;
    double  *u_data  = NULL;
    
    int      i,j,k,cnt,m;
    double   x,y,z,t;
    double   hx,hy,hz,ht,s,ss;
    double   hx2,hy2,hz2;
    double   dd;
    
    int      nxplus1 = nx + 1;
    int      nyplus1 = ny + 1;
    int      nzplus1 = nz + 1;
    
    int      corner1 = 0;
    int      corner2 = nx-1;
    int      corner3 = nplane-nx;
    int      corner4 = nplane-1;
    int      corner5 = nx*ny*(nz-1);
    int      corner6 = nx*ny*(nz-1)+nx-1;
    int      corner7 = ngrid-nx;
    int      corner8 = ngrid-1;
    
    double   factorx  = 0.0;
    double   factory  = 0.0;
    double   factorz  = 0.0;
    double   constant = 3.0*PI*PI;
    double   xcoeff   = 1.0;
    double   ycoeff   = 1.0;
    double   zcoeff   = 1.0;

    A = fsls_BandMatrixCreate(ngrid, nband);
    fsls_BandMatrixInitialize(A);
    fsls_BandMatrixNx(A) = nx;
    fsls_BandMatrixNy(A) = ny;
    fsls_BandMatrixNz(A) = nz;
    
    offset  = fsls_BandMatrixOffsets(A);
    diag    = fsls_BandMatrixDiag(A);
    offdiag = fsls_BandMatrixOffdiag(A);
    
    offset[0] = -1;
    offset[1] =  1;
    offset[2] = -nx;
    offset[3] =  nx;
    offset[4] = -nplane;
    offset[5] =  nplane;
    
    /*----------------------------------------------------
     * Generate matrix without BC processing
     *--------------------------------------------------*/
    
    if (nt == 0) {
        ngridxt = ngrid;
        nt = 1;
        nt_flag = 0;
    }
    
    hx  = 1.0 / (double)nxplus1;
    hy  = 1.0 / (double)nyplus1;
    hz  = 1.0 / (double)nzplus1;
    ht  = 1.0 / (double)nt;
    
    hx2 = hx*hx;
    hy2 = hy*hy;
    hz2 = hz*hz;
    
    factorx = xcoeff / hx2;
    factory = ycoeff / hy2;
    factorz = zcoeff / hz2;
    
    dd = 2.0*(factorx + factory + factorz);
    
    for (i = 0; i < ngrid; i ++) {
        diag[i]       = dd;
        offdiag[0][i] = -factorx;
        offdiag[1][i] = -factorx;
        offdiag[2][i] = -factory;
        offdiag[3][i] = -factory;
        offdiag[4][i] = -factorz;
        offdiag[5][i] = -factorz;
    }
    
    /*----------------------------------------------------
     * zero-out some entries
     *--------------------------------------------------*/
    
    /* 8 corner points */
    
    offdiag[0][corner1] = 0.0;
    offdiag[2][corner1] = 0.0;
    offdiag[4][corner1] = 0.0;
    
    offdiag[1][corner2] = 0.0;
    offdiag[2][corner2] = 0.0;
    offdiag[4][corner2] = 0.0;
    
    offdiag[0][corner3] = 0.0;
    offdiag[3][corner3] = 0.0;
    offdiag[4][corner3] = 0.0;
    
    offdiag[1][corner4] = 0.0;
    offdiag[3][corner4] = 0.0;
    offdiag[4][corner4] = 0.0;
    
    offdiag[0][corner5] = 0.0;
    offdiag[2][corner5] = 0.0;
    offdiag[5][corner5] = 0.0;
    
    offdiag[1][corner6] = 0.0;
    offdiag[2][corner6] = 0.0;
    offdiag[5][corner6] = 0.0;
    
    offdiag[0][corner7] = 0.0;
    offdiag[3][corner7] = 0.0;
    offdiag[5][corner7] = 0.0;
    
    offdiag[1][corner8] = 0.0;
    offdiag[3][corner8] = 0.0;
    offdiag[5][corner8] = 0.0;
    
    /* 12 edges */
    
    for (i = corner1+1; i < corner2; i ++) {
        offdiag[2][i] = 0.0;
        offdiag[4][i] = 0.0;
    }
    
    for (i = corner3+1; i < corner4; i ++) {
        offdiag[3][i] = 0.0;
        offdiag[4][i] = 0.0;
    }
    
    for (i = corner5+1; i < corner6; i ++) {
        offdiag[2][i] = 0.0;
        offdiag[5][i] = 0.0;
    }
    
    for (i = corner7+1; i < corner8; i ++) {
        offdiag[3][i] = 0.0;
        offdiag[5][i] = 0.0;
    }
    
    for (i = corner1+nx; i < corner3; i += nx) {
        offdiag[0][i] = 0.0;
        offdiag[4][i] = 0.0;
    }
    
    for (i = corner2+nx; i < corner4; i += nx) {
        offdiag[1][i] = 0.0;
        offdiag[4][i] = 0.0;
    }
    
    for (i = corner5+nx; i < corner7; i += nx) {
        offdiag[0][i] = 0.0;
        offdiag[5][i] = 0.0;
    }
    
    for (i = corner6+nx; i < corner8; i += nx) {
        offdiag[1][i] = 0.0;
        offdiag[5][i] = 0.0;
    }
    
    for (i = corner1+nplane; i < corner5; i += nplane) {
        offdiag[0][i] = 0.0;
        offdiag[2][i] = 0.0;
    }
    
    for (i = corner2+nplane; i < corner6; i += nplane) {
        offdiag[1][i] = 0.0;
        offdiag[2][i] = 0.0;
    }
    
    for (i = corner3+nplane; i < corner7; i += nplane) {
        offdiag[0][i] = 0.0;
        offdiag[3][i] = 0.0;
    }
    
    for (i = corner4+nplane; i < corner8; i += nplane) {
        offdiag[1][i] = 0.0;
        offdiag[3][i] = 0.0;
    }
    
    /* 6 planes */
    
    // Left
    for (i = corner1+nplane; i < corner5; i += nplane) {
        k = corner3 + i;
        for (j = i+nx; j < k; j += nx) {
            offdiag[0][j] = 0.0;
        }
    }
    
    // Right
    for (i = corner2+nplane; i < corner6; i += nplane) {
        k = corner3 + i;
        for (j = i+nx; j < k; j += nx) {
            offdiag[1][j] = 0.0;
        }
    }
    
    // Front
    for (i = corner1+nplane; i < corner5; i += nplane) {
        k = corner2 + i;
        for (j = i+1; j < k; j ++) {
            offdiag[2][j] = 0.0;
        }
    }
    
    // Back
    for (i = corner3+nplane; i < corner7; i += nplane) {
        k = corner2 + i;
        for (j = i+1; j < k; j ++) {
            offdiag[3][j] = 0.0;
        }
    }
    
    //Down
    for (i = corner1+nx; i < corner3; i += nx) {
        k = corner2 + i;
        for (j = i+1; j < k; j ++) {
            offdiag[4][j] = 0.0;
        }
    }
    
    // Up
    for (i = corner5+nx; i < corner7; i += nx) {
        k = corner2 + i;
        for (j = i+1; j < k; j ++) {
            offdiag[5][j] = 0.0;
        }
    }
    
    /*----------------------------------------------------
     * generate the rhs and sol vector
     *--------------------------------------------------*/
    
    f = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(f);
    u = fsls_XVectorCreate(ngridxt);
    fsls_XVectorInitialize(u);
    f_data = fsls_XVectorData(f);
    u_data = fsls_XVectorData(u);
    
    cnt = 0;
    double tmp;
    for (m = 0; m < nt; m ++) {
        t = ht*(m + 1);
        for (k = 0; k < nz; k ++) {
            z = hz*(k + 1);
            ss = sin(PI*z);
            for (j = 0; j < ny; j ++) {
                y = hy*(j + 1);
                s = ss*sin(PI*y);
                for (i = 0; i < nx; i ++) {
                    x = hx*(i + 1);
                    tmp = s*sin(PI*x);
                    u_data[cnt] = tmp*t;
                    f_data[cnt] = tmp*nt_flag + constant*u_data[cnt];
                    cnt ++;
                }
            }
        }
    }
    
    *A_ptr = A;
    *f_ptr = f;
    *u_ptr = u;
}

int
fsls_Band2CSRMatrix (fsls_BandMatrix *B,
                     fsls_CSRMatrix **A_ptr)
{
    int      n       = fsls_BandMatrixN(B);
    int      nband   = fsls_BandMatrixNband(B);
    int     *offsets = fsls_BandMatrixOffsets(B);
    double  *diag    = fsls_BandMatrixDiag(B);
    double **offdiag = fsls_BandMatrixOffdiag(B);
    
    fsls_CSRMatrix *A = NULL;
    int *ia = NULL;
    int *ja = NULL;
    double *a = NULL;
    
    int i;
    int col;
    int band,offset;
    int begin,end;
    int nplus1 = n + 1;
    int nzpr = nband + 1;
    
    /*---------------------------------------------
     * Create a CSR matrix
     *--------------------------------------------*/
    A = fsls_CSRMatrixCreate(n,n,nzpr*n);
    fsls_CSRMatrixInitialize(A);
    ia = fsls_CSRMatrixI(A);
    ja = fsls_CSRMatrixJ(A);
    a  = fsls_CSRMatrixData(A);
    
    /*---------------------------------------------
     * Generate the 'ia' array
     *--------------------------------------------*/
    for (i = 0; i < nplus1; i ++) {
        ia[i] = i*nzpr;
    }
    
    /*---------------------------------------------
     * fill the diagonal entries
     *--------------------------------------------*/
    for (i = 0; i < n; i ++) {
        a[ia[i]]  = diag[i];
        ja[ia[i]] = i;
        ia[i] ++;
    }
    
    /*---------------------------------------------
     * fill the offdiagonal entries
     *--------------------------------------------*/
    for (band = 0; band < nband; band ++) {
        offset = offsets[band];
        if (offset < 0) {
            begin = abs(offset);
            for (i = begin,col = 0; i < n; i++, col++) {
                a[ia[i]]  = offdiag[band][i];
                ja[ia[i]] = col;
                ia[i] ++;
            }
        }
        else {
            end = n - offset;
            for (i = 0,col = offset; i < end; i++, col++) {
                a[ia[i]]  = offdiag[band][i];
                ja[ia[i]] = col;
                ia[i] ++;
            }
        }
    }
    
    /*---------------------------------------------
     * regenerate the 'ia' array
     *--------------------------------------------*/
    for (i = 0; i < nplus1; i ++) {
        ia[i] = i*nzpr;
    }
    
    /*---------------------------------------------
     * delete zero entries in A
     *--------------------------------------------*/
    A = fsls_CSRMatrixDeleteZeros(A, 0.0);
    
    *A_ptr = A;
    
    return 0;
}

int
fsls_CSRMatrixPrint (fsls_CSRMatrix *matrix,
                     char *file_name )
{
    FILE    *fp;
    
    double  *matrix_data;
    int     *matrix_i;
    int     *matrix_j;
    int      num_rows;
    
    int      file_base = 1;
    
    int      j;
    
    int      ierr = 0;
    
    /*----------------------------------------------
     * Print the matrix data
     *---------------------------------------------*/
    
    matrix_data = fsls_CSRMatrixData(matrix);
    matrix_i    = fsls_CSRMatrixI(matrix);
    matrix_j    = fsls_CSRMatrixJ(matrix);
    num_rows    = fsls_CSRMatrixNumRows(matrix);
    
    fp = fopen(file_name, "w");
    
    fprintf(fp, "%d\n", num_rows);
    
    for (j = 0; j <= num_rows; j++) {
        fprintf(fp, "%d\n", matrix_i[j] + file_base);
    }
    
    for (j = 0; j < matrix_i[num_rows]; j++) {
        fprintf(fp, "%d\n", matrix_j[j] + file_base);
    }
    
    if (matrix_data) {
        for (j = 0; j < matrix_i[num_rows]; j++) {
            fprintf(fp, "%.15le\n", matrix_data[j]); // we always use "%.15le\n"
        }
    }
    else {
        fprintf(fp, "### WARNING: No matrix data!\n");
    }
    
    fclose(fp);
    
    return ierr;
}

int
fsls_CSRMatrixDestroy ( fsls_CSRMatrix *matrix )
{
    int  ierr=0;
    if (matrix) {
        fsls_TFree(fsls_CSRMatrixI(matrix));
        if (fsls_CSRMatrixRownnz(matrix)) {
            fsls_TFree(fsls_CSRMatrixRownnz(matrix));
        }
        if ( fsls_CSRMatrixOwnsData(matrix) ) {
            fsls_TFree(fsls_CSRMatrixData(matrix));
            fsls_TFree(fsls_CSRMatrixJ(matrix));
        }
        fsls_TFree(matrix);
    }
    
    return ierr;
}

int
fsls_CSRMatrixInitialize ( fsls_CSRMatrix *matrix )
{
    int  num_rows     = fsls_CSRMatrixNumRows(matrix);
    int  num_nonzeros = fsls_CSRMatrixNumNonzeros(matrix);
    int  ierr=0;
    
    if ( ! fsls_CSRMatrixData(matrix) && num_nonzeros )
        fsls_CSRMatrixData(matrix) = fsls_CTAlloc(double, num_nonzeros);
    if ( ! fsls_CSRMatrixI(matrix) )
        fsls_CSRMatrixI(matrix)    = fsls_CTAlloc(int, num_rows + 1);
    if ( ! fsls_CSRMatrixJ(matrix) && num_nonzeros )
        fsls_CSRMatrixJ(matrix)    = fsls_CTAlloc(int, num_nonzeros);
    
    return ierr;
}

fsls_CSRMatrix *
fsls_CSRMatrixCreate (int num_rows,
                      int num_cols,
                      int num_nonzeros)
{
    fsls_CSRMatrix  *matrix;
    
    matrix = fsls_CTAlloc(fsls_CSRMatrix, 1);
    
    fsls_CSRMatrixData(matrix) = NULL;
    fsls_CSRMatrixI(matrix)    = NULL;
    fsls_CSRMatrixJ(matrix)    = NULL;
    fsls_CSRMatrixRownnz(matrix) = NULL;
    fsls_CSRMatrixNumRows(matrix) = num_rows;
    fsls_CSRMatrixNumCols(matrix) = num_cols;
    fsls_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
    
    /* set defaults */
    fsls_CSRMatrixOwnsData(matrix) = 1;
    fsls_CSRMatrixNumRownnz(matrix) = num_rows;
    
    return matrix;
}

fsls_CSRMatrix *
fsls_CSRMatrixDeleteZeros (fsls_CSRMatrix *A,
                           double tol)
{
    double     *A_data   = fsls_CSRMatrixData(A);
    int        *A_i      = fsls_CSRMatrixI(A);
    int        *A_j      = fsls_CSRMatrixJ(A);
    int         nrows_A  = fsls_CSRMatrixNumRows(A);
    int         ncols_A  = fsls_CSRMatrixNumCols(A);
    int         num_nonzeros  = fsls_CSRMatrixNumNonzeros(A);
    
    fsls_CSRMatrix *B;
    double         *B_data;
    int            *B_i;
    int            *B_j;
    
    int zeros;
    int i,j;
    int nzB;
    
    /* get the total number of zeros in matrix A */
    zeros = 0;
    for (i = 0; i < num_nonzeros; i++) {
        if (fabs(A_data[i]) <= tol) zeros++;
    }
    
    /* there exists zeros in the matrix A */
    if (zeros) {
        B = fsls_CSRMatrixCreate(nrows_A,ncols_A,num_nonzeros-zeros);
        fsls_CSRMatrixInitialize(B);
        B_i = fsls_CSRMatrixI(B);
        B_j = fsls_CSRMatrixJ(B);
        B_data = fsls_CSRMatrixData(B);
        B_i[0] = 0;
        nzB = 0;
        for (i=0; i < nrows_A; i++) {
            for (j = A_i[i]; j < A_i[i+1]; j++) {
                /* modified by peghoty 2009/12/06 */
                if (fabs(A_data[j]) > tol) {
                    B_data[nzB] = A_data[j];
                    B_j[nzB] = A_j[j];
                    nzB++;
                }
            }
            B_i[i+1] = nzB;
        }
        fsls_CSRMatrixDestroy(A); /* added by peghoty 2009/12/06 */
        A = NULL;
        return B;
    }
    else {
        /* there doesn't exist zeros in the matrix A */
        return A; /* modified by peghoty 2009/12/06 */
    }
}

void
fsls_Free ( char *ptr )
{
    if (ptr) free(ptr);
}

char *
fsls_CAlloc ( size_t count, size_t elt_size )
{
    char *ptr;
    long long   size = count*elt_size;
    
    if (size > 0) {
        ptr = (char*)calloc(count, elt_size);
        if (ptr == NULL) {
            fsls_OutOfMemory(size);
        }
    }
    else {
        ptr = NULL;
    }
    
    return ptr;
}

int
fsls_OutOfMemory ( size_t size )
{
    printf("### ERROR: Out of memory trying to allocate %d bytes\n", (int) size);
    fflush(stdout);
    return 0;
}

fsls_BandMatrix *
fsls_BandMatrixCreate ( int n, int nband )
{
    fsls_BandMatrix *matrix = NULL;
    
    matrix = fsls_CTAlloc(fsls_BandMatrix, 1);
    
    fsls_BandMatrixN(matrix)       = n;
    fsls_BandMatrixNband(matrix)   = nband;
    fsls_BandMatrixOffsets(matrix) = NULL;
    fsls_BandMatrixDiag(matrix)    = NULL;
    fsls_BandMatrixOffdiag(matrix) = NULL;
    fsls_BandMatrixDataExt(matrix) = NULL;
    
    return matrix;
}

void
fsls_BandMatrixInitialize ( fsls_BandMatrix *matrix )
{
    int      n        = fsls_BandMatrixN(matrix);
    int      nband    = fsls_BandMatrixNband(matrix);
    int     *offsets  = NULL;
    double  *diag     = NULL;
    double **offdiag  = NULL;
    double  *data_ext = NULL;
    
    int i;
    
    offsets  = fsls_CTAlloc(int, nband);
    offdiag  = fsls_CTAlloc(double*, nband);
    data_ext = fsls_CTAlloc(double, (nband+1)*(n+2));
    
    /* reset the pointer */
    diag = &data_ext[1];
    diag[-1] = 1.0;
    diag[n]  = 1.0;
    
    for (i = 0; i < nband; i ++) {
        offdiag[i] = &data_ext[(i+1)*(n+2)+1];
    }
    
    fsls_BandMatrixOffsets(matrix) = offsets;
    fsls_BandMatrixDiag(matrix)    = diag;
    fsls_BandMatrixOffdiag(matrix) = offdiag;
    fsls_BandMatrixDataExt(matrix) = data_ext;
}

int
fsls_XVectorPrint (fsls_XVector *vector,
                   char *file_name)
{
    /* information of vector */
    int      size = fsls_XVectorSize(vector);
    double  *data = fsls_XVectorData(vector);
    
    FILE    *fp = NULL;
    int      i;
    
    /*-----------------------------------------
     * Print in the data
     *---------------------------------------*/
    fp = fopen(file_name, "w");
    
    fprintf(fp, "%d\n", size);
    
    for (i = 0; i < size; i ++) {
        fprintf(fp, "%.15le\n", data[i]);
    }
    
    fclose(fp);
    
    return (0);
}

fsls_XVector *
fsls_XVectorCreate (int size)
{
    fsls_XVector  *vector = NULL;
    
    vector = fsls_CTAlloc(fsls_XVector, 1);
    
    fsls_XVectorData(vector)    = NULL;
    fsls_XVectorDataExt(vector) = NULL;
    fsls_XVectorSize(vector)    = size;
    
    return (vector);
}

int
fsls_XVectorInitialize (fsls_XVector *vector)
{
    int     size     = fsls_XVectorSize(vector);
    double *data     = NULL;
    double *data_ext = NULL;
    
    if ( ! data_ext ) {
        data_ext = fsls_CTAlloc(double, size+2);
    }
    
    data = &data_ext[1];
    
    fsls_XVectorData(vector)    = data;
    fsls_XVectorDataExt(vector) = data_ext;
    
    return 0;
}

int
fsls_XVectorDestroy (fsls_XVector *vector)
{
    if (vector) {
        if (fsls_XVectorDataExt(vector)) {
            fsls_TFree(fsls_XVectorDataExt(vector));
        }
        fsls_TFree(vector);
    }
    
    return 0;
}

void
fsls_BandMatrixDestroy (fsls_BandMatrix *matrix)
{
    if (matrix) {
        if (fsls_BandMatrixOffsets(matrix))
            fsls_TFree(fsls_BandMatrixOffsets(matrix));
        if (fsls_BandMatrixOffdiag(matrix))
            fsls_TFree(fsls_BandMatrixOffdiag(matrix));
        if (fsls_BandMatrixDataExt(matrix))
            fsls_TFree(fsls_BandMatrixDataExt(matrix));
        fsls_TFree(matrix);
    }
}

int
fsls_WriteSAMGData (fsls_CSRMatrix *A,
                    fsls_XVector *b,
                    fsls_XVector *u)
{
    FILE    *fp = NULL;
    
    double  *b_data = fsls_XVectorData(b);
    double  *u_data = fsls_XVectorData(u);
    double  *matrix_data;
    int     *matrix_i;
    int     *matrix_j;
    int      num_rows;
    int      num_nonzeros;
    
    int      file_base = 1;
    
    int      j;
    
    int      ierr = 0;
    char    *file_name01 = NULL;
    char    *file_name02 = NULL;
    char    *file_name03 = NULL;
    char    *file_name04 = NULL;
    
    matrix_data  = fsls_CSRMatrixData(A);
    matrix_i     = fsls_CSRMatrixI(A);
    matrix_j     = fsls_CSRMatrixJ(A);
    num_rows     = fsls_CSRMatrixNumRows(A);
    num_nonzeros = matrix_i[num_rows];
    
    file_name01 = "./SAMGDATA/poisson.frm";
    file_name02 = "./SAMGDATA/poisson.amg";
    file_name03 = "./SAMGDATA/poisson.rhs";
    file_name04 = "./SAMGDATA/poisson.sol";
    
    /* write the .frm file */
    fp = fopen(file_name01, "w");
    fprintf(fp, "%s   %d\n", "f", 4);
    fprintf(fp, "%d %d %d %d %d\n", num_nonzeros, num_rows, 12, 1, 0);
    fclose(fp);
    
    /* write the .amg file */
    fp = fopen(file_name02, "w");
    for (j = 0; j <= num_rows; j++) {
        fprintf(fp, "%d\n", matrix_i[j] + file_base);
    }
    for (j = 0; j < num_nonzeros; j++) {
        fprintf(fp, "%d\n", matrix_j[j] + file_base);
    }
    if (matrix_data) {
        for (j = 0; j < num_nonzeros; j++) {
            fprintf(fp, "%.15le\n", matrix_data[j]); // we always use "%.15le\n"
        }
    }
    else {
        fprintf(fp, "### WARNING: No matrix data!\n");
    }
    fclose(fp);
    
    /* write the .rhs file */
    fp = fopen(file_name03, "w");
    for (j = 0; j < num_rows; j++) {
        fprintf(fp, "%.15le\n", b_data[j]);
    }
    fclose(fp);
    
    /* write the .sol file */
    fp = fopen(file_name04, "w");
    for (j = 0; j < num_rows; j++) {
        fprintf(fp, "%.15le\n", u_data[j]);
    }
    fclose(fp);
    
    return ierr;
}

int
fsls_CSR2FullMatrix (fsls_CSRMatrix *A,
                     double **full_ptr)
{
    int ierr = 0, row = 0, col = 0, i;
    int num_rows = fsls_CSRMatrixNumRows(A);
    int num_cols = fsls_CSRMatrixNumCols(A);
    int *matrix_i = fsls_CSRMatrixI(A);
    int *matrix_j = fsls_CSRMatrixJ(A);
    double *matrix_data = fsls_CSRMatrixData(A);
    int nnz = fsls_CSRMatrixNumNonzeros(A);
    double *full_A = fsls_CTAlloc(double, num_rows*num_cols);
    memset(full_A, 0X0, num_rows*num_cols*sizeof(double));
    
    for (i = 0; i < nnz; ++i) {
        if (i==matrix_i[row+1])
            row = row+1;
        col = matrix_j[i];
        full_A[row+num_cols*col]=matrix_data[i];
    }
    *full_ptr = full_A;
    return ierr;
}

int
fsls_dtMatrix (double dt,
               int n_rows,
               int n_cols,
               double *A_full)
{
    int ierr = 0;
    int i,j;
    
    if(n_rows != n_cols) printf("...\n");
    
    for( i = 0; i < n_rows; ++i ) {
        for( j = 0; j < n_cols; ++j ) {
            A_full[i+n_cols*j] *= dt;
        }
    }
    for( i = 0; i < n_rows; ++i ) {
        A_full[i+n_cols*i] += 1;
    }
    return ierr;
}

int
fsls_COOMatrixPrint (fsls_CSRMatrix *matrix,
                     char *file_name )
{
    int ierr = 0;
    
    FILE    *fp;
    
    double  *matrix_data;
    int     *matrix_i;
    int     *matrix_j;
    int      num_rows;
    int      num_cols;
    int      nnz, row = 0, col = 0, i;
    double   data;
    
    /*----------------------------------------------
     * Print the matrix data
     *---------------------------------------------*/
    
    matrix_data = fsls_CSRMatrixData(matrix);
    matrix_i    = fsls_CSRMatrixI(matrix);
    matrix_j    = fsls_CSRMatrixJ(matrix);
    num_rows    = fsls_CSRMatrixNumRows(matrix);
    num_cols    = fsls_CSRMatrixNumCols(matrix);
    nnz                  = fsls_CSRMatrixNumNonzeros(matrix);
    
    fp = fopen(file_name, "w");
    
    fprintf(fp, "%d          ", num_rows);
    fprintf(fp, "%d          ", num_cols);
    fprintf(fp, "%d\n", nnz);
    
    for (i = 0; i < nnz; ++i) {
        if (i==matrix_i[row+1])
            row = row+1;
        col = matrix_j[i];
        data = matrix_data[i];
        fprintf(fp, "%d          %d          %.15le\n", row, col, data);
    }
    fclose(fp);
    
    return ierr;
}

int
fsls_MatrixSPGnuplot (fsls_CSRMatrix *matrix,
                      char *file_name )
{
    int ierr = 0;
    
    FILE    *fp;
    
    int     *matrix_i;
    int     *matrix_j;
    int      nnz, row = 0, col = 0, i;
    
    /*----------------------------------------------
     * Print the matrix data
     *---------------------------------------------*/
    
    matrix_i    = fsls_CSRMatrixI(matrix);
    matrix_j    = fsls_CSRMatrixJ(matrix);
    nnz            = fsls_CSRMatrixNumNonzeros(matrix);
    
    fp = fopen(file_name, "w");
    
    for (i = 0; i < nnz; ++i) {
        if (i==matrix_i[row+1])
            row = row+1;
        col = matrix_j[i];
        fprintf(fp, "%d   %d\n", col, -row);
    }
    fclose(fp);
    
    return ierr;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
