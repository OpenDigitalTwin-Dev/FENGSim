/*! \file  BlaSparseSTR.c
 *
 *  \brief Sparse matrix operations for dSTRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn dSTRmat fasp_dstr_create (const INT nx, const INT ny, const INT nz, 
 *                               const INT nc, const INT nband, INT *offsets)
 *
 * \brief Create STR sparse matrix data memory space
 *
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param nz        Number of grids in z direction
 * \param nc        Number of components 
 * \param nband     Number of off-diagonal bands 
 * \param offsets   Shift from diagonal
 *
 * \return          The dSTRmat matrix
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   05/17/2010 
 */
dSTRmat fasp_dstr_create (const INT  nx,
                          const INT  ny,
                          const INT  nz,
                          const INT  nc,
                          const INT  nband,
                          INT       *offsets)
{    
    dSTRmat A;
    
    INT i;
    
    A.nx=nx; A.ny=ny; A.nz=nz;
    A.nc=nc;
    A.nxy=A.nx*A.ny;
    A.ngrid=A.nxy*A.nz;
    A.nband=nband;
    
    A.offsets=(INT*)fasp_mem_calloc(nband, sizeof(INT));
    
    for (i=0;i<nband;++i) A.offsets[i]=offsets[i];
    
    A.diag=(REAL*)fasp_mem_calloc(A.ngrid*A.nc*A.nc, sizeof(REAL));
    
    A.offdiag=(REAL**)fasp_mem_calloc(nband, sizeof(REAL*));
    
    for (i=0;i<A.nband;++i) {
        A.offdiag[i]=(REAL*)fasp_mem_calloc((A.ngrid-ABS(A.offsets[i]))*A.nc*A.nc, sizeof(REAL));
    }
    
    return(A);
}

/**
 * \fn void fasp_dstr_alloc (const INT nx, const INT ny, const INT nz, const INT nxy, 
 *                           const INT ngrid, const INT nband, const INT nc, 
 *                           INT *offsets, dSTRmat *A)
 *
 * \brief Allocate STR sparse matrix memory space
 *
 * \param nx        Number of grids in x direction
 * \param ny        Number of grids in y direction
 * \param nz        Number of grids in z direction
 * \param nxy       Number of grids in x-y plane
 * \param ngrid     Number of grids
 * \param nband     Number of off-diagonal bands 
 * \param nc        Number of components 
 * \param offsets   Shift from diagonal
 * \param A         Pointer to the dSTRmat matrix
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   05/17/2010  
 */
void fasp_dstr_alloc (const INT  nx,
                      const INT  ny,
                      const INT  nz,
                      const INT  nxy,
                      const INT  ngrid,
                      const INT  nband,
                      const INT  nc,
                      INT       *offsets,
                      dSTRmat   *A)
{    
    INT i;
    
    A->nx=nx;
    A->ny=ny;
    A->nz=nz;
    A->nxy=nxy;
    A->ngrid=ngrid;
    A->nband=nband;
    A->nc=nc;
    
    A->offsets=(INT*)fasp_mem_calloc(nband, sizeof(INT));
    
    for (i=0;i<nband;++i) A->offsets[i]=offsets[i];
    
    A->diag=(REAL*)fasp_mem_calloc(ngrid*nc*nc, sizeof(REAL));
    
    A->offdiag = (REAL **)fasp_mem_calloc(A->nband, sizeof(REAL*));
    
    for (i=0;i<nband;++i) {
        A->offdiag[i]=(REAL*)fasp_mem_calloc((ngrid-ABS(offsets[i]))*nc*nc, sizeof(REAL));
    }
}

/**
 * \fn void fasp_dstr_free (dSTRmat *A)
 *
 * \brief Free STR sparse matrix data memeory space
 *
 * \param A   Pointer to the dSTRmat matrix
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   05/17/2010 
 */
void fasp_dstr_free (dSTRmat *A)
{    
    INT i;
    
    fasp_mem_free(A->offsets); A->offsets = NULL;
    fasp_mem_free(A->diag);    A->diag    = NULL;
    
    for ( i = 0; i < A->nband; ++i ) {
        fasp_mem_free(A->offdiag[i]); A->offdiag[i] = NULL;
    }

    A->nx = A->ny = A->nz = A->nxy=0;
    A->ngrid = A->nband = A->nc=0;
}

/**
 * \fn void fasp_dstr_cp (const dSTRmat *A, dSTRmat *B)
 *
 * \brief Copy a dSTRmat to a new one B=A
 *
 * \param A   Pointer to the dSTRmat matrix
 * \param B   Pointer to the dSTRmat matrix
 * 
 * \author Zhiyang Zhou
 * \date   04/21/2010  
 */
void fasp_dstr_cp (const dSTRmat *A,
                   dSTRmat       *B)
{    
    const INT nc2 = (A->nc)*(A->nc);

    INT i;
    B->nx    = A->nx;
    B->ny    = A->ny;
    B->nz    = A->nz;
    B->nxy   = A->nxy;
    B->ngrid = A->ngrid;
    B->nc    = A->nc;
    B->nband = A->nband;
    
    memcpy(B->offsets,A->offsets,(A->nband)*sizeof(INT));
    memcpy(B->diag,A->diag,(A->ngrid*nc2)*sizeof(REAL));
    for (i=0;i<A->nband;++i) {
        memcpy(B->offdiag[i],A->offdiag[i],
               ((A->ngrid - ABS(A->offsets[i]))*nc2)*sizeof(REAL));
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
