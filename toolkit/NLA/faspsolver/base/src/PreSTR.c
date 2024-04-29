/*! \file  PreSTR.c
 *
 *  \brief Preconditioners for dSTRmat matrices
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxVector.c, BlaSmallMat.c, BlaArray.c,
 *         and ItrSmootherSTR.c
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
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static inline void fasp_darray_cp_nc3 (const REAL *x, REAL *y);
static inline void fasp_darray_cp_nc5 (const REAL *x, REAL *y);
static inline void fasp_darray_cp_nc7 (const REAL *x, REAL *y);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_precond_dstr_diag (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/06/2010
 */
void fasp_precond_dstr_diag (REAL *r, 
                             REAL *z, 
                             void *data)
{
    const precond_diag_str *diag = (precond_diag_str *)data;
    const REAL *diagptr = diag->diag.val;
    const INT   nc = diag->nc, nc2 = nc*nc;
    const INT   m = diag->diag.row/nc2;
    
    INT i;
    for ( i=0; i<m; ++i ) {
        fasp_blas_smat_mxv(&(diagptr[i*nc2]),&(r[i*nc]),&(z[i*nc]),nc);
    }    
}

/**
 * \fn void fasp_precond_dstr_ilu0 (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(0) decomposition
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/21/2010
 */
void fasp_precond_dstr_ilu0 (REAL *r, 
                             REAL *z, 
                             void *data)
{
    INT i, ic, ic2;
    REAL *zz,*zr,*tc;
    INT nline, nplane;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif    
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    zr=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    memcpy(zr,r,(size)*sizeof(REAL));
    
    if (nc == 1) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0]=zr[0];
        for (i=1;i<m;++i) {
            zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            if (i>=nline) zz[i]=zz[i]-ILU_data->offdiag[2][i-nline]*zz[i-nline];
            if (i>=nplane) zz[i]=zz[i]-ILU_data->offdiag[4][i-nplane]*zz[i-nplane];
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
        z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        for (i=m-2;i>=0;i--) {
            zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            if (i<m-nline) zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nline];
            if (i<m-nplane) zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nplane];
            z[i]=zz[i]*ILU_data->diag[i];
        }
    
    } // end if (nc == 1)   
    
    else if (nc == 3) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc3(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;
    
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc3(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc3(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc3(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 3)
    
    else if (nc == 5) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc5(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;

            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc5(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc5(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc5(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 5)
    
    
    else if (nc == 7) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc7(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;

            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc7(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc7(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc7(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 7)
    
    else {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp(nc,&(zr[0]),&(zz[0]));
        for (i=1;i<m;++i) {
            ic=i*nc;

            fasp_blas_smat_mxv(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc,nc);
            fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
    
            if (i>=nline) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp(nc,&(zr[ic]),&(zz[ic]));
    
        } // end for (i=1; i<m; ++i)
    
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]),nc);
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc,nc);
            fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]),nc);
            
        }// end for (i=m-2;i>=0;i--)
    } // end else   
    
    fasp_mem_free(zr); zr = NULL;
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif    
    
    return;
}

/**
 * \fn void fasp_precond_dstr_ilu1 (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(1) decomposition
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/21/2010
 */
void fasp_precond_dstr_ilu1 (REAL *r, 
                             REAL *z, 
                             void *data)
{
    REAL *zz,*zr,*tc;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT i,ic, ic2;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    INT nline, nplane;
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    zr=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    for (i=0;i<size;++i) zr[i]=r[i];
    if (nc == 1) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0]=zr[0];
        for (i=1;i<m;++i) {
    
            zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            if (i>=nline-1)
                zz[i]=zz[i]-ILU_data->offdiag[2][i-nline+1]*zz[i-nline+1];
    
            if (i>=nline)
                zz[i]=zz[i]-ILU_data->offdiag[4][i-nline]*zz[i-nline];
            if (i>=nplane-nline)
                zz[i]=zz[i]-ILU_data->offdiag[6][i-nplane+nline]*zz[i-nplane+nline];
            if (i>=nplane-1)
                zz[i]=zz[i]-ILU_data->offdiag[8][i-nplane+1]*zz[i-nplane+1];
            if (i>=nplane)
                zz[i]=zz[i]-ILU_data->offdiag[10][i-nplane]*zz[i-nplane];    
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
    
        z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        for (i=m-2;i>=0;i--) {
    
            zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            if (i+nline-1<m)
                zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nline-1];
            if (i+nline<m)
                zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nline];
            if (i+nplane-nline<m)
                zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nplane-nline];
            if (i+nplane-1<m)
                zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nplane-1]; 
            if (i+nplane<m)
                zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nplane];
    
            z[i]=ILU_data->diag[i]*zz[i];
    
        }
    
    }     // end if (nc == 1)
    
    else if (nc == 3) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc3(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;

            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc3(&(zr[ic]),&(zz[ic]));          
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc3(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc3(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 3)
    
    else if (nc == 5) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc5(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;

            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc5(&(zr[ic]),&(zz[ic]));          
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc5(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc5(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 5)
    
    else if (nc == 7) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc7(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;

            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc7(&(zr[ic]),&(zz[ic]));          
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc7(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc7(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 7)
        
    else {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp(nc,&(zr[0]),&(zz[0]));
         
        for (i=1;i<m;++i) {
            ic=i*nc;
            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc,nc);           
            fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            } 

            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }

            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }

            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
            fasp_darray_cp(nc,&(zr[ic]),&(zz[ic]));          
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]),nc);
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc,nc);          
            fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]),nc);
        }
    }  // end else
    
    fasp_mem_free(zr); zr = NULL;
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
    return;
}

/**
 * \fn void fasp_precond_dstr_ilu0_forward (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(0) decomposition: Lz = r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   06/07/2010
 */
void fasp_precond_dstr_ilu0_forward (REAL *r, 
                                     REAL *z, 
                                     void *data)
{
    INT i, ic;
    REAL *zz,*zr,*tc;
    INT nline, nplane;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    zr=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    memcpy(zr,r,(size)*sizeof(REAL));
    if (nc == 1) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0]=zr[0];
        for (i=1;i<m;++i) {
            zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            if (i>=nline) zz[i]=zz[i]-ILU_data->offdiag[2][i-nline]*zz[i-nline];
            if (i>=nplane) zz[i]=zz[i]-ILU_data->offdiag[4][i-nplane]*zz[i-nplane];
        }
    } // end if (nc == 1)   
    
    else if (nc == 3) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc3(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc3(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
    } // end else if (nc == 3)
    
    else if (nc == 5) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc5(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc5(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
    } // end else if (nc == 5)
    
    
    else if (nc == 7) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc7(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);
            fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            if (i>=nline) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            if (i>=nplane) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            fasp_darray_cp_nc7(&(zr[ic]),&(zz[ic]));
        } // end for (i=1;i<m;++i) 
    
    } // end else if (nc == 7)
    
    
    else {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp(nc,&(zr[0]),&(zz[0]));
        for (i=1;i<m;++i) {
            ic=i*nc;    
            fasp_blas_smat_mxv(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc,nc);
            fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
    
            if (i>=nline) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[2][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                fasp_blas_smat_mxv(&(ILU_data->offdiag[4][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp(nc,&(zr[ic]),&(zz[ic]));
    
        } // end for (i=1; i<m; ++i)
    
    } // end else   
    
    memcpy(z,zz,(size)*sizeof(REAL));
    
    fasp_mem_free(zr); zr = NULL;
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
    return;
}

/**
 * \fn void fasp_precond_dstr_ilu0_backward (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(0) decomposition: Uz = r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   06/07/2010
 */
void fasp_precond_dstr_ilu0_backward (REAL *r, 
                                      REAL *z, 
                                      void *data)
{
    INT i, ic, ic2;
    REAL *zz,*tc;
    INT nline, nplane;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    memcpy(zz,r,(size)*sizeof(REAL));
    if (nc == 1) {
        // backward sweep: solve upper matrix equation U*z=zz
    
        z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        for (i=m-2;i>=0;i--) {
            zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            if (i<m-nline) zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nline];
            if (i<m-nplane) zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nplane];
            z[i]=zz[i]*ILU_data->diag[i];
        }
    
    } // end if (nc == 1)   
    
    else if (nc == 3) {
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc3(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc3(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 3)
    
    else if (nc == 5) {
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc5(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc5(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 5)
    
    else if (nc == 7) {
        // backward sweep: solve upper matrix equation U*z=zz
        fasp_blas_smat_mxv_nc7(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
    
            ic=i*nc;
            ic2=i*nc2;
    
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);
            fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
    
            if (i<m-nline) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i<m-nplane) {
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc);
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            fasp_blas_smat_mxv_nc7(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for for (i=m-2;i>=0;i--) 
    
    } // end else if (nc == 7)
    
    
    else 
        {
            // backward sweep: solve upper matrix equation U*z=zz
            fasp_blas_smat_mxv(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]),nc);
    
            for (i=m-2;i>=0;i--) {
                ic=i*nc;
                ic2=i*nc2;
    
                fasp_blas_smat_mxv(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc,nc);
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
    
                if (i<m-nline) {
                    fasp_blas_smat_mxv(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline)*nc]),tc,nc);
                    fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
                }
    
                if (i<m-nplane) {
                    fasp_blas_smat_mxv(&(ILU_data->offdiag[5][ic2]),&(z[(i+nplane)*nc]),tc,nc);
                    fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
                }
    
                fasp_blas_smat_mxv(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]),nc);
            
            }// end for (i=m-2;i>=0;i--)
        } // end else   
    
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
    return;
}

/**
 * \fn void fasp_precond_dstr_ilu1_forward (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(1) decomposition: Lz = r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/21/2010
 */
void fasp_precond_dstr_ilu1_forward (REAL *r, 
                                     REAL *z, 
                                     void *data)
{
    REAL *zz,*zr,*tc;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT i,ic;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    INT nline, nplane;
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    zr=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    //for (i=0;i<size;++i) zr[i]=r[i];
    memcpy(zr,r,(size)*sizeof(REAL));
    if (nc == 1) {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0]=zr[0];
        for (i=1;i<m;++i) {
    
            zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            if (i>=nline-1)
                zz[i]=zz[i]-ILU_data->offdiag[2][i-nline+1]*zz[i-nline+1];
    
            if (i>=nline)
                zz[i]=zz[i]-ILU_data->offdiag[4][i-nline]*zz[i-nline];
            if (i>=nplane-nline)
                zz[i]=zz[i]-ILU_data->offdiag[6][i-nplane+nline]*zz[i-nplane+nline];
            if (i>=nplane-1)
                zz[i]=zz[i]-ILU_data->offdiag[8][i-nplane+1]*zz[i-nplane+1];
            if (i>=nplane)
                zz[i]=zz[i]-ILU_data->offdiag[10][i-nplane]*zz[i-nplane];    
        }
    
    }     // end if (nc == 1)
    
    else if (nc == 3) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc3(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc3(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc3(&(zr[ic]),&(zz[ic]));          
        }
    
    }  // end if (nc == 3)
    
    else if (nc == 5) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc5(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc5(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc5(&(zr[ic]),&(zz[ic]));          
        }
    
    }  // end if (nc == 5)
    
    else if (nc == 7) {
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp_nc7(&(zr[0]),&(zz[0]));
    
        for (i=1;i<m;++i) {
            ic=i*nc;    
            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc);           
            fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            } 
    
            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc);           
                fasp_blas_darray_axpy_nc7(-1,tc,&(zr[ic]));
            }
    
            fasp_darray_cp_nc7(&(zr[ic]),&(zz[ic]));          
        }
    
    }  // end if (nc == 7)
    
    else {
        // forward sweep: solve unit lower matrix equation L*zz=zr
        fasp_darray_cp(nc,&(zr[0]),&(zz[0]));
        for (i=1;i<m;++i) {
            ic=i*nc;
            //zz[i]=zr[i]-ILU_data->offdiag[0][i-1]*zz[i-1];
            fasp_blas_smat_mxv(&(ILU_data->offdiag[0][(i-1)*nc2]),&(zz[(i-1)*nc]),tc,nc);           
            fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
    
            if (i>=nline-1) {
                //zz[i]=zz[i]-ILU_data->offdiag[2][i-nx+1]*zz[i-nx+1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[2][(i-nline+1)*nc2]),&(zz[(i-nline+1)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
            
            if (i>=nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[4][i-nx]*zz[i-nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[4][(i-nline)*nc2]),&(zz[(i-nline)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            } 

            if (i>=nplane-nline) {
                //zz[i]=zz[i]-ILU_data->offdiag[6][i-nxy+nx]*zz[i-nxy+nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[6][(i-nplane+nline)*nc2]),&(zz[(i-nplane+nline)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }

            if (i>=nplane-1) {
                // zz[i]=zz[i]-ILU_data->offdiag[8][i-nxy+1]*zz[i-nxy+1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[8][(i-nplane+1)*nc2]),&(zz[(i-nplane+1)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }

            if (i>=nplane) {
                //zz[i]=zz[i]-ILU_data->offdiag[10][i-nxy]*zz[i-nxy];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[10][(i-nplane)*nc2]),&(zz[(i-nplane)*nc]),tc,nc);           
                fasp_blas_darray_axpy(nc,-1,tc,&(zr[ic]));
            }
            fasp_darray_cp(nc,&(zr[ic]),&(zz[ic]));          
        }
    }  // end else
    
    memcpy(z,zz,(size)*sizeof(REAL));
    
    fasp_mem_free(zr); zr = NULL;
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
    return;
}

/**
 * \fn void fasp_precond_dstr_ilu1_backward (REAL *r, REAL *z, void *data)
 *
 * \brief Preconditioning using STR_ILU(1) decomposition: Uz = r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/21/2010
 */
void fasp_precond_dstr_ilu1_backward (REAL *r, 
                                      REAL *z, 
                                      void *data)
{
    REAL *zz,*tc;
    
    dSTRmat *ILU_data=(dSTRmat *)data;
    INT i,ic, ic2;
    INT m=ILU_data->ngrid;
    INT nc=ILU_data->nc;
    INT nc2=nc*nc;
    INT nx=ILU_data->nx;
    INT ny=ILU_data->ny;
    INT nz=ILU_data->nz;
    INT nxy=ILU_data->nxy;
    INT size=m*nc;
    INT nline, nplane;
    
    if (nx == 1) {
        nline = ny;
        nplane = m;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = m;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = m;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    tc=(REAL*)fasp_mem_calloc(nc, sizeof(REAL)); 
    
    zz=(REAL*)fasp_mem_calloc(size, sizeof(REAL)); 
    
    // copy residual r to zr, to save r
    //for (i=0;i<size;++i) zr[i]=r[i];
    memcpy(zz,r,(size)*sizeof(REAL));
    if (nc == 1) {
        // backward sweep: solve upper matrix equation U*z=zz
    
        z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        for (i=m-2;i>=0;i--) {
    
            zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            if (i+nline-1<m)
                zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nline-1];
            if (i+nline<m)
                zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nline];
            if (i+nplane-nline<m)
                zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nplane-nline];
            if (i+nplane-1<m)
                zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nplane-1]; 
            if (i+nplane<m)
                zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nplane];
    
            z[i]=ILU_data->diag[i]*zz[i];
    
        }
    
    }     // end if (nc == 1)
    
    else if (nc == 3) {
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc3(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc3(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc3(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc3(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 3)
    
    else if (nc == 5) {
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc5(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc5(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc5(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc5(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 5)
    
    else if (nc == 7) {
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv_nc7(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]));
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
    
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc);          
            fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv_nc7(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc);          
                fasp_blas_darray_axpy_nc7(-1,tc,&(zz[ic]));
            }
    
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv_nc7(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]));
        } // end for (i=m-2;i>=0;i--)
    
    }  // end if (nc == 7)
        
    else {
        // backward sweep: solve upper matrix equation U*z=zz
    
        // z[m-1]=zz[m-1]*ILU_data->diag[m-1];
        fasp_blas_smat_mxv(&(ILU_data->diag[(m-1)*nc2]),&(zz[(m-1)*nc]),&(z[(m-1)*nc]),nc);
    
        for (i=m-2;i>=0;i--) {
            ic=i*nc;
            ic2=ic*nc;
            //zz[i]=zz[i]-ILU_data->offdiag[1][i]*z[i+1];
            fasp_blas_smat_mxv(&(ILU_data->offdiag[1][ic2]),&(z[(i+1)*nc]),tc,nc);          
            fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
    
            if (i+nline-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[3][i]*z[i+nx-1];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[3][ic2]),&(z[(i+nline-1)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }
    
            if (i+nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[5][i]*z[i+nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[5][ic2]),&(z[(i+nline)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane-nline<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[7][i]*z[i+nxy-nx];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[7][ic2]),&(z[(i+nplane-nline)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane-1<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[9][i]*z[i+nxy-1]; 
                fasp_blas_smat_mxv(&(ILU_data->offdiag[9][ic2]),&(z[(i+nplane-1)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }

            if (i+nplane<m) {
                //zz[i]=zz[i]-ILU_data->offdiag[11][i]*z[i+nxy];
                fasp_blas_smat_mxv(&(ILU_data->offdiag[11][ic2]),&(z[(i+nplane)*nc]),tc,nc);          
                fasp_blas_darray_axpy(nc,-1,tc,&(zz[ic]));
            }
            //z[i]=ILU_data->diag[i]*zz[i];
            fasp_blas_smat_mxv(&(ILU_data->diag[ic2]),&(zz[ic]),&(z[ic]),nc);
        }
    }  // end else
    
    fasp_mem_free(zz); zz = NULL;
    fasp_mem_free(tc); tc = NULL;
    
    return;
}

/**
 * \fn void fasp_precond_dstr_blockgs (REAL *r, REAL *z, void *data)
 *
 * \brief CPR-type preconditioner (STR format)
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   10/17/2010
 */
void fasp_precond_dstr_blockgs (REAL *r, 
                                REAL *z, 
                                void *data)
{
    precond_data_str *predata=(precond_data_str *)data;
    dSTRmat *A = predata->A_str;
    dvector *diaginv = predata->diaginv;
    ivector *pivot = predata->pivot;
    ivector *order = predata->order;
    ivector *neigh = predata->neigh;
    
    INT i;
    const INT nc = A->nc;
    const INT ngrid = A->ngrid;
    const INT n   = nc*ngrid;  // whole size
    
    dvector zz, rr;
    zz.row=rr.row=n; zz.val=z; rr.val=r;
    fasp_dvec_set(n,&zz,0.0);
    
    for (i=0; i<1; ++i)
        fasp_smoother_dstr_swz(A, &rr, &zz, diaginv, pivot, neigh, order);
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static inline void fasp_darray_cp_nc3 (const REAL *x, REAL *y)
 *
 * \brief Copy an array to the other y=x, the length is 3
 *
 * \param x    Pointer to the original vector
 * \param y    Pointer to the destination vector
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
static inline void fasp_darray_cp_nc3 (const REAL  *x,
                                       REAL        *y)
{
    memcpy(y, x, 3*sizeof(REAL));
}

/**
 * \fn static inline void fasp_darray_cp_nc5 (const REAL *x, REAL *y)
 *
 * \brief Copy an array to the other y=x, the length is 5
 *
 * \param x    Pointer to the original vector
 * \param y    Pointer to the destination vector
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
static inline void fasp_darray_cp_nc5 (const REAL  *x,
                                       REAL        *y)
{
    memcpy(y, x, 5*sizeof(REAL));
}

/**
 * \fn static inline void fasp_darray_cp_nc7 (const REAL *x, REAL *y)
 *
 * \brief Copy an array to the other y=x, the length is 7
 *
 * \param x    Pointer to the original vector
 * \param y    Pointer to the destination vector
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
static inline void fasp_darray_cp_nc7 (const REAL  *x,
                                       REAL        *y)
{
    memcpy(y, x, 7*sizeof(REAL));
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
