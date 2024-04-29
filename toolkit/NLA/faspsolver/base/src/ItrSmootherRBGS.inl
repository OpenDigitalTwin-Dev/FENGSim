/*! \file  ItrSmootherRBGS.inl
 *
 *  \brief Routines for Red-Black Gauss-Seidel smoother.
 *
 *  \note  This file contains Level-2 (Itr) functions, which WAS used in:
 *         ItrSmootherCSR.c. Currently NOT used!
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn void fasp_smoother_dcsr_rbgs3d (dvector *u, dCSRmat *A, dvector *b, INT L,
 *                                     const INT order, INT *mark, const INT maximap,
 *                                     const INT nx, const INT ny, const INT nz)
 *
 * \brief       Red-black Gauss-Seidel smoother for Au=b in 3D
 *
 * \param u        Initial guess and the new approximation to the solution
 * \param A        Pointer to stiffness matrix
 * \param b        Pointer to right hand side
 * \param L        Number of iterations
 * \param order    Ordering: -1 = forward, 1 = backward
 * \param mark     Marker for C/F points
 * \param maximap  Size of IMAP
 * \param nx       Number vertex of X direction
 * \param ny       Number vertex of Y direction
 * \param nz       Number vertex of Z direction
 *
 * \author Chunsheng Feng
 * \date   02/08/2012
 */
void fasp_smoother_dcsr_rbgs3d (dvector    *u,
                                dCSRmat    *A,
                                dvector    *b,
                                INT         L,
                                const INT   order,
                                INT        *mark,
                                const INT   maximap,
                                const INT   nx,
                                const INT   ny,
                                const INT   nz)
{
    const INT   nrow = b->row; // number of rows
    INT        *ia = A->IA,  *ja   = A->JA;
    REAL       *aa = A->val, *bval = b->val;
    REAL       *uval = u->val;

    INT         i,j,k,begin_row,end_row;
    REAL        t, d = 0.0;

    // forward
    if (order == 1) {
        while (L--) {
            rb0f3d( ia, ja, aa, uval, bval, mark, nx,  ny,  nz, 1);
#if 1 // TODO: Check! Why? --Chensong
            for ( i = maximap; i < nrow; i++ ) {
                t = bval[i];
                begin_row = ia[i]; end_row = ia[i+1];
                for ( k = begin_row; k < end_row; k++ ) {
                    j = ja[k];
                    if (i!=j) t -= aa[k]*uval[j];
                    else d = aa[k];
                } // end for k
                if (ABS(d) > SMALLREAL) uval[i] = t/d;
            } // end for i
#endif
        } // end while
    }

    // backward
    else {
        while (L--) {
#if 1 // TODO: Check! Why? --Chensong
            for ( i = nrow-1; i >= maximap; i-- ) {
                t = bval[i];
                begin_row = ia[i]; end_row = ia[i+1];
                for ( k = begin_row; k < end_row; k ++ ) {
                    j = ja[k];
                    if (i!=j) t -= aa[k]*uval[j];
                    else d = aa[k];
                } // end for k
                if (ABS(d) > SMALLREAL) uval[i] = t/d;
            } // end for i
#endif
            rb0b3d( ia, ja, aa, uval, bval, mark, nx,  ny,  nz, 1);
        } // end while
    }
    return;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void swep2db (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f, INT nbegx,
 *                          INT nbegy, INT *mark, INT nx, INT ny)
 *
 * \brief      Gauss-Seidel backward smoother for certain color
 *
 * \param ia     Pointer to start location of each row
 * \param ja     Pointer to column index of nonzero elements
 * \param aa     Pointer to nonzero elements of
 * \param u      Pointer to initial guess
 * \param f      Pointer to right hand
 * \param nbegx  The stride between the same color nodes in x direction
 * \param nbegy  The stride between the same color nodes in y direction
 * \param mark   Pointer to order of nodes
 * \param nx     Number of nodes in x direction
 * \param ny     Number of nodes in y direction
 *
 * \author  Chunsheng Feng, Zheng Li
 * \date    02/06/2012
 *
 * \note The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void swep2db (INT   *ia,
                     INT   *ja,
                     REAL  *aa,
                     REAL  *u,
                     REAL  *f,
                     INT    nbegx,
                     INT    nbegy,
                     INT   *mark,
                     INT    nx,
                     INT    ny)
{
    INT j, j0, i, i0;
    INT begin_row, end_row, ii, jj;
    REAL t, d = 1.0;

    nbegx = nx + nbegx;
    nbegy = ny + nbegy;

#ifdef _OPENMP
#pragma omp parallel for private(j,j0,i,i0,t,begin_row,end_row,ii,jj,d)
#endif
    for (j = nbegy; j >=0; j-=2) {
        j0= j*nx;
        for (i = nbegx; i >=0; i-=2) {
            i0 = i + j0;
            i0 = mark[i0]-1; // Fortran to C
            if (i0>=0 ) {
                t = f[i0];
                begin_row = ia[i0]; end_row = ia[i0+1];
                for (ii = begin_row; ii < end_row; ii ++) {
                    jj = ja[ii];
                    if (i0!=jj) t -= aa[ii]*u[jj];
                    else d = aa[ii];
                } // end for ii
                if (ABS(d) > SMALLREAL) u[i0] = t/d;
            } //if (i0>=0 )
        }
    }
}

/**
 * \fn static void swep3db (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f, INT nbegx,
 *                          INT nbegy, INT nbegz, INT *mark, INT nx, INT ny, INT nz)
 *
 * \brief      Gauss-Seidel backward smoother for certain color
 *
 * \param ia     Pointer to start location of each row
 * \param ja     Pointer to column index of nonzero elements
 * \param aa     Pointer to nonzero elements of
 * \param u      Pointer to initial guess
 * \param f      Pointer to right hand
 * \param nbegx  The stride between the same color nodes in x direction
 * \param nbegy  The stride between the same color nodes in y direction
 * \param nbegz  The stride between the same color nodes in z direction
 * \param mark   Pointer to order of nodes
 * \param nx     Number of nodes in x direction
 * \param ny     Number of nodes in y direction
 * \param nz     Number of nodes in z direction
 *
 * \author  Chunsheng Feng, Zheng Li
 * \date    02/06/2012
 *
 * \note The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void swep3db (INT   *ia,
                     INT   *ja,
                     REAL  *aa,
                     REAL  *u,
                     REAL  *f,
                     INT    nbegx,
                     INT    nbegy,
                     INT    nbegz,
                     INT   *mark,
                     INT    nx,
                     INT    ny,
                     INT    nz)
{
    INT nxy, k, k0, j, j0, i, i0;
    INT begin_row, end_row, ii, jj;
    REAL t, d = 1.0;

    nxy=nx*ny;
    nbegx = nx + nbegx;
    nbegy = ny + nbegy;
    nbegz = nz + nbegz;

#ifdef _OPENMP
#pragma omp parallel for private(k,k0,j,j0,i,i0,t,begin_row,end_row,ii,jj,d)
#endif
    for (k=nbegz; k >=0; k-=2) {
        k0= k*nxy;
        for (j = nbegy; j >=0; j-=2) {
            j0= j*nx;
            for (i = nbegx; i >=0; i-=2) {
                i0 = i   +  j0    + k0;
                i0 = mark[i0]-1;  // Fortran to C
                if (i0>=0 ) {
                    t = f[i0];
                    begin_row = ia[i0]; end_row = ia[i0+1];
                    for (ii = begin_row; ii < end_row; ii ++) {
                        jj = ja[ii];
                        if (i0!=jj) t -= aa[ii]*u[jj];
                        else d = aa[ii];
                    } // end for ii

                    if (ABS(d) > SMALLREAL) u[i0] = t/d;
                } //if (i0>=0 )
            }
        }
    }
}

/*
 * \fn static void rb0b2d (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f,
 *                         INT *mark, INT nx, INT ny, INT nsweeps)
 *
 * \brief Colored Gauss-Seidel backward smoother for Au=b
 *
 * \param ia       Pointer to start location of each row
 * \param ja       Pointer to column index of nonzero elements
 * \param aa       Pointer to nonzero elements of
 * \param u        Pointer to initial guess
 * \param f        Pointer to right hand
 * \param mark     Pointer to order of nodes
 * \param nx       Number of nodes in x direction
 * \param ny       Number of nodes in y direction
 * \param nsweeps  Number of relaxation sweeps
 *
 * \author  Chunsheng Feng, Zheng Li
 * \date    02/06/2012
 *
 * Modified by Chensong Zhang on 09/21/2017: Remove unused code.
 *
 * \note This subroutine is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void rb0b2d (INT   *ia,
                    INT   *ja,
                    REAL  *aa,
                    REAL  *u,
                    REAL  *f,
                    INT   *mark,
                    INT    nx,
                    INT    ny,
                    INT    nsweeps)
{
    const INT  n0e = 0, n0o = 1;
    INT        isweep;

    for (isweep = 1; isweep <= nsweeps; isweep++) {
        /*...  e-e */
        swep2df(ia,ja,aa,u,f,n0e,n0e,mark,nx,ny);
        /*...  e-o */
        swep2df(ia,ja,aa,u,f,n0e,n0o,mark,nx,ny);
        /*...  o-e */
        swep2df(ia,ja,aa,u,f,n0o,n0e,mark,nx,ny);
        /*...  o-o */
        swep2df(ia,ja,aa,u,f,n0o,n0o,mark,nx,ny);
    }
}

/*
 * \fn static void rb0b3d (INT *ia, INT *ja, REAL *aa,REAL *u, REAL *f,
 *                         INT *mark, INT nx, INT ny, INT nz, INT nsweeps)
 *
 * \brief  Colores Gauss-Seidel backward smoother for Au=b
 *
 * \param ia       Pointer to start location of each row
 * \param ja       Pointer to column index of nonzero elements
 * \param aa       Pointer to nonzero elements of
 * \param u        Pointer to initial guess
 * \param f        Pointer to right hand
 * \param mark     Pointer to order of nodes
 * \param nx       Number of nodes in x direction
 * \param ny       Number of nodes in y direction
 * \param nz       Number of nodes in z direction
 * \param nsweeps  Number of relaxation sweeps
 *
 * \author  Chunsheng Feng
 * \date    02/06/2012
 *
 * \note This subroutine is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void rb0b3d (INT   *ia,
                    INT   *ja,
                    REAL  *aa,
                    REAL  *u,
                    REAL  *f,
                    INT   *mark,
                    INT    nx,
                    INT    ny,
                    INT    nz,
                    INT    nsweeps)
{
    const INT  n0e = 0, n0o = 1;
    INT        isweep;

    for (isweep = 1; isweep <= nsweeps; isweep++) {
        /*...  e-e-e */
        swep3df(ia,ja,aa,u,f,n0e,n0e,n0e,mark,nx,ny,nz);
        /*...  e-e-o */
        swep3df(ia,ja,aa,u,f,n0e,n0e,n0o,mark,nx,ny,nz);
        /*...  e-o-e */
        swep3df(ia,ja,aa,u,f,n0e,n0o,n0e,mark,nx,ny,nz);
        /*...  e-o-o */
        swep3df(ia,ja,aa,u,f,n0e,n0o,n0o,mark,nx,ny,nz);
        /*...  o-e-e */
        swep3df(ia,ja,aa,u,f,n0o,n0e,n0e,mark,nx,ny,nz);
        /*...  o-e-o */
        swep3df(ia,ja,aa,u,f,n0o,n0e,n0o,mark,nx,ny,nz);
        /*...  o-o-e */
        swep3df(ia,ja,aa,u,f,n0o,n0o,n0e,mark,nx,ny,nz);
        /*...  o-o-o */
        swep3df(ia,ja,aa,u,f,n0o,n0o,n0o,mark,nx,ny,nz);
    }
}

/**
 * \fn static void swep2df (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f, INT nbegx,
 *                          INT nbegy, INT *mark, INT nx, INT ny)
 *
 * \brief      Gauss-Seidel forward smoother for certain color
 *
 * \param ia     Pointer to start location of each row
 * \param ja     Pointer to column index of nonzero elements
 * \param aa     Pointer to nonzero elements of
 * \param u      Pointer to initial guess
 * \param f      Pointer to right hand
 * \param nbegx  The stride between the same color nodes in x direction
 * \param nbegy  The stride between the same color nodes in y direction
 * \param mark   Pointer to order of nodes
 * \param nx     Number of nodes in x direction
 * \param ny     Number of nodes in y direction
 *
 * \author  Chunsheng Feng, Zheng Li
 * \date    02/06/2012
 *
 * \note The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void swep2df (INT   *ia,
                     INT   *ja,
                     REAL  *aa,
                     REAL  *u,
                     REAL  *f,
                     INT    nbegx,
                     INT    nbegy,
                     INT   *mark,
                     INT    nx,
                     INT    ny)
{
    INT j,j0,i,i0;
    INT begin_row,end_row,ii,jj;
    REAL t,d=0;

#ifdef _OPENMP
#pragma omp parallel for private(j,j0,i,i0,t,begin_row,end_row,ii,jj,d)
#endif
    for (j = nbegy; j < ny; j+=2) {
        j0= j*nx;
        for (i = nbegx; i < nx; i+=2)    /*!*/
        {
            i0 = i + j0;
            i0 = mark[i0]-1; //Fortran to C
            if (i0>=0 ) {
                t = f[i0];
                begin_row = ia[i0]; end_row = ia[i0+1];
                for (ii = begin_row; ii < end_row; ii ++) {
                    jj = ja[ii];
                    if (i0!=jj) t -= aa[ii]*u[jj];
                    else d = aa[ii];
                } // end for ii
                if (ABS(d) > SMALLREAL) u[i0] = t/d;
            } //    if (i0>=0 )
        }
    }

}


/**
 * \fn static void swep3df (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f, INT nbegx,
 *                          INT nbegy, INT nbegz, INT *mark, INT nx, INT ny, INT nz)
 *
 * \brief      Gauss-Seidel forward smoother for certain color
 *
 * \param ia     Pointer to start location of each row
 * \param ja     Pointer to column index of nonzero elements
 * \param aa     Pointer to nonzero elements of
 * \param u      Pointer to initial guess
 * \param f      Pointer to right hand
 * \param nbegx  The stride between the same color nodes in x direction
 * \param nbegy  The stride between the same color nodes in y direction
 * \param nbegz  The stride between the same color nodes in z direction
 * \param mark   Pointer to order of nodes
 * \param nx     Number of nodes in x direction
 * \param ny     Number of nodes in y direction
 * \param nz     Number of nodes in z direction
 *
 * \author  Chunsheng Feng, Zheng Li
 * \date    02/06/2012
 *
 * Note: The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void swep3df (INT   *ia,
                     INT   *ja,
                     REAL  *aa,
                     REAL  *u,
                     REAL  *f,
                     INT    nbegx,
                     INT    nbegy,
                     INT    nbegz,
                     INT   *mark,
                     INT    nx,
                     INT    ny,
                     INT    nz)
{
    INT nxy=nx*ny,k,k0,j,j0,i,i0;
    INT begin_row,end_row,ii,jj;
    REAL t,d=0;

#ifdef _OPENMP
#pragma omp parallel for private(k,k0,j,j0,i,i0,t,begin_row,end_row,ii,jj,d)
#endif
    for (k=nbegz; k < nz; k+=2) {
        k0= k*nxy;
        for (j = nbegy; j < ny; j+=2) {
            j0= j*nx;
            for (i = nbegx; i < nx; i+=2)    /*!*/
            {
                i0 = i   +  j0    + k0;
                i0 = mark[i0]-1; //Fortran to C

                if (i0>=0 ) {
                    t = f[i0];
                    begin_row = ia[i0]; end_row = ia[i0+1];
                    for (ii = begin_row; ii < end_row; ii ++) {
                        jj = ja[ii];
                        if (i0!=jj) t -= aa[ii]*u[jj];
                        else d = aa[ii];
                    } // end for ii

                    if (ABS(d) > SMALLREAL) u[i0] = t/d;
                } //    if (i0>=0 )
            }
        }
    }

}

/*
 * \fn static void rb0f2d (INT *ia, INT *ja, REAL *aa, REAL *u, REAL *f,
 *                         INT *mark, INT nx, INT ny, INT nsweeps)
 *
 * \brief  Colores Gauss-Seidel forward smoother for Au = b
 *
 * \param ia       Pointer to the start location to of row
 * \param ja       Pointer to the column index of nonzero elements
 * \param aa       Pointer to the values of the nonzero elements
 * \param u        Pointer to initial value
 * \param f        Pointer to right hand
 * \param mark     Pointer to the order index of nodes
 * \param nx       Number of nodes in x direction
 * \param ny       Number of nodes in y direction
 * \param nsweeps  Number of relaxation sweeps
 *
 * \author  Chunsheng Feng, Zheng Li
 * \data    02/06/2012
 *
 * NOTE: The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void rb0f2d (INT   *ia,
                    INT   *ja,
                    REAL  *aa,
                    REAL  *u,
                    REAL  *f,
                    INT   *mark,
                    INT    nx,
                    INT    ny,
                    INT    nsweeps)
{
    INT n0e,n0o,isweep;

    n0e=0;
    n0o=1;

    for (isweep = 1; isweep <= nsweeps; isweep++) {
        /*...  o-o */
        swep2df(ia,ja,aa,u,f,n0o,n0o,mark,nx,ny);
        /*...  o-e */
        swep2df(ia,ja,aa,u,f,n0o,n0e,mark,nx,ny);
        /*...  e-o */
        swep2df(ia,ja,aa,u,f,n0e,n0o,mark,nx,ny);
        /*...  e-e */
        swep2df(ia,ja,aa,u,f,n0e,n0e,mark,nx,ny);
    }
}

/*
 * \fn static void rb0f3d (INT *ia, INT *ja, REAL *aa,REAL *u, REAL *f, INT *mark,
 *                         INT nx, INT ny, INT nz, INT nsweeps)
 *
 * \brief  Colores Gauss-Seidel forward smoother for Au=b
 *
 * \param ia       Pointer to the start location to of row
 * \param ja       Pointer to the column index of nonzero elements
 * \param aa       Pointer to the values of the nonzero elements
 * \param u        Pointer to initial value
 * \param f        Pointer to right hand
 * \param mark     Pointer to the order index of nodes
 * \param nx       Number of nodes in x direction
 * \param ny       Number of nodes in y direction
 * \param nz       Number of nodes in z direction
 * \param nsweeps  Number of relaxation sweeps
 *
 * \author      Chunsheng Feng, Zheng Li
 * \data        02/06/2012
 *
 * NOTE: The following code is based on SiPSMG (Simple Poisson Solver based on MultiGrid)
 * (c) 2008 Johannes Kraus, Jinchao Xu, Yunrong Zhu, Ludmil Zikatanov
 */
static void rb0f3d (INT   *ia,
                    INT   *ja,
                    REAL  *aa,
                    REAL  *u,
                    REAL  *f,
                    INT   *mark,
                    INT    nx,
                    INT    ny,
                    INT    nz,
                    INT    nsweeps)
{
    INT n0e,n0o,isweep;

    n0e=0;
    n0o=1;

    for (isweep = 1; isweep <= nsweeps; isweep++) {
        /*...  o-o-o */
        swep3df(ia,ja,aa,u,f,n0o,n0o,n0o,mark,nx,ny,nz);
        /*...  o-o-e */
        swep3df(ia,ja,aa,u,f,n0o,n0o,n0e,mark,nx,ny,nz);
        /*...  o-e-o */
        swep3df(ia,ja,aa,u,f,n0o,n0e,n0o,mark,nx,ny,nz);
        /*...  o-e-e */
        swep3df(ia,ja,aa,u,f,n0o,n0e,n0e,mark,nx,ny,nz);
        /*...  e-o-o */
        swep3df(ia,ja,aa,u,f,n0e,n0o,n0o,mark,nx,ny,nz);
        /*...  e-o-e */
        swep3df(ia,ja,aa,u,f,n0e,n0o,n0e,mark,nx,ny,nz);
        /*...  e-e-o */
        swep3df(ia,ja,aa,u,f,n0e,n0e,n0o,mark,nx,ny,nz);
        /*...  e-e-e */
        swep3df(ia,ja,aa,u,f,n0e,n0e,n0e,mark,nx,ny,nz);
        /*...  o-o-o */
        swep3df(ia,ja,aa,u,f,n0o,n0o,n0o,mark,nx,ny,nz);

    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
