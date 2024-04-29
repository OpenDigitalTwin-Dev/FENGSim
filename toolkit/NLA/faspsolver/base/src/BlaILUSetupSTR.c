/*! \file  BlaILUSetupSTR.c
 *
 *  \brief Setup incomplete LU decomposition for dSTRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c, BlaSmallMat.c, BlaSmallMatInv.c, BlaSparseSTR.c,
 *         and BlaArray.c
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
 * \fn void fasp_ilu_dstr_setup0 (dSTRmat *A, dSTRmat *LU)
 *
 * \brief Get ILU(0) decomposition of a structured matrix A 
 *
 * \param A   Pointer to dSTRmat
 * \param LU  Pointer to ILU structured matrix of REAL type
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   11/08/2010
 *
 * \note Only works for 5 bands 2D and 7 bands 3D matrix with default offsets 
 *       (order can be arbitrary)!
 */
void fasp_ilu_dstr_setup0 (dSTRmat  *A,
                           dSTRmat  *LU)
{
    // local variables 
    INT i,i1,ix,ixy,ii;     
    INT *LUoffsets;
    INT nline, nplane;
    
    // information of A
    INT nc = A->nc;
    INT nc2 = nc*nc;
    INT nx = A->nx;
    INT ny = A->ny;
    INT nz = A->nz;
    INT nxy = A->nxy;
    INT ngrid = A->ngrid;
    INT nband = A->nband;
    
    INT  *offsets = A->offsets;
    REAL *smat=(REAL *)fasp_mem_calloc(nc2,sizeof(REAL));
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL;
    REAL *offdiag3=NULL, *offdiag4=NULL, *offdiag5=NULL;
    
    // initialize    
    if (nx == 1) {
        nline = ny;
        nplane = ngrid;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = ngrid;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = ngrid;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    // check number of bands
    if (nband == 4) {
        LUoffsets=(INT *)fasp_mem_calloc(4,sizeof(INT));
        LUoffsets[0]=-1; LUoffsets[1]=1; LUoffsets[2]=-nline; LUoffsets[3]=nline; 
    }
    else if (nband == 6) {
        LUoffsets=(INT *)fasp_mem_calloc(6,sizeof(INT));
        LUoffsets[0]=-1; LUoffsets[1]=1; LUoffsets[2]=-nline; 
        LUoffsets[3]=nline; LUoffsets[4]=-nplane; LUoffsets[5]=nplane;
    } 
    else {
        printf("%s: number of bands for structured ILU is illegal!\n", __FUNCTION__);
        return;
    }
    
    // allocate memory to store LU decomposition
    fasp_dstr_alloc(nx, ny, nz, nxy, ngrid, nband, nc, offsets, LU);
    
    // copy diagonal 
    memcpy(LU->diag, diag, (ngrid*nc2)*sizeof(REAL));
    
    // check offsets and copy off-diagonals
    for (i=0; i<nband; ++i) {
        if (offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
            memcpy(LU->offdiag[0],offdiag0,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }
        else if (offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
            memcpy(LU->offdiag[1],offdiag1,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }    
        else if (offsets[i] == -nline) {
            offdiag2 = A->offdiag[i];
            memcpy(LU->offdiag[2],offdiag2,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }    
        else if (offsets[i] == nline) {
            offdiag3 = A->offdiag[i];
            memcpy(LU->offdiag[3],offdiag3,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }
        else if (offsets[i] == -nplane) {
            offdiag4 = A->offdiag[i];
            memcpy(LU->offdiag[4],offdiag4,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }    
        else if (offsets[i] == nplane) {
            offdiag5 = A->offdiag[i];
            memcpy(LU->offdiag[5],offdiag5,((ngrid - ABS(offsets[i]))*nc2)*sizeof(REAL));
        }    
        else {
            printf("### ERROR: Illegal offset for ILU! [%s]\n", __FUNCTION__);
            return;
        }
    }
    
    // Setup
    if (nc == 1) {

        LU->diag[0]=1.0/(LU->diag[0]);
    
        for (i=1;i<ngrid;++i) {
            
            LU->offdiag[0][i-1]=(offdiag0[i-1])*(LU->diag[i-1]);
            if (i>=nline)
                LU->offdiag[2][i-nline]=(offdiag2[i-nline])*(LU->diag[i-nline]);
            if (i>=nplane)
                LU->offdiag[4][i-nplane]=(offdiag0[i-nplane])*(LU->diag[i-nplane]);
    
            LU->diag[i]=diag[i]-(LU->offdiag[0][i-1])*(LU->offdiag[1][i-1]);
    
            if (i>=nline)
                LU->diag[i]=LU->diag[i]-(LU->offdiag[2][i-nline])*(LU->offdiag[3][i-nline]);
            if (i>=nplane)
                LU->diag[i]=LU->diag[i]-(LU->offdiag[4][i-nplane])*(LU->offdiag[5][i-nplane]);
    
            LU->diag[i]=1.0/(LU->diag[i]);
    
        } // end for (i=1; i<ngrid; ++i)
    
    } // end if (nc == 1)
    
    else if (nc == 3) {       
    
        fasp_smat_inv_nc3(LU->diag);
    
        for (i=1;i<ngrid;++i) {
    
            i1=(i-1)*9;    
            ix=(i-nline)*9;
            ixy=(i-nplane)*9;
            ii=i*9;
    
            fasp_blas_smat_mul_nc3(&(offdiag0[i1]),&(LU->diag[i1]),&(LU->offdiag[0][i1]));
    
            if (i>=nline)
                fasp_blas_smat_mul_nc3(&(offdiag2[ix]),&(LU->diag[ix]),&(LU->offdiag[2][ix]));
            if (i>=nplane)
                fasp_blas_smat_mul_nc3(&(offdiag4[ixy]),&(LU->diag[ixy]),&(LU->offdiag[4][ixy]));
    
            fasp_blas_smat_mul_nc3(&(LU->offdiag[0][i1]),&(LU->offdiag[1][i1]),smat);
    
            fasp_blas_darray_axpyz_nc3(-1,smat,&(diag[ii]),&(LU->diag[ii]));
            
            if (i>=nline) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[2][ix]),&(LU->offdiag[3][ix]),smat);
                fasp_blas_darray_axpy_nc3(-1.0,smat,&(LU->diag[ii]));
            } //end if (i>=nline)
    
            if (i>=nplane) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[4][ixy]),&(LU->offdiag[5][ixy]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ii]));
            } // end if (i>=nplane)
    
            fasp_smat_inv_nc3(&(LU->diag[ii]));
    
        } // end for(i=1;i<A->ngrid;++i)
    
    }  // end if (nc == 3)
    
    else if (nc == 5) {       
    
        fasp_smat_inv_nc5(LU->diag);
    
        for (i=1;i<ngrid;++i) {
    
            i1=(i-1)*25;    
            ix=(i-nline)*25;
            ixy=(i-nplane)*25;
            ii=i*25;
    
            fasp_blas_smat_mul_nc5(&(offdiag0[i1]),&(LU->diag[i1]),&(LU->offdiag[0][i1]));
    
            if (i>=nline)
                fasp_blas_smat_mul_nc5(&(offdiag2[ix]),&(LU->diag[ix]),&(LU->offdiag[2][ix]));
            if (i>=nplane)
                fasp_blas_smat_mul_nc5(&(offdiag4[ixy]),&(LU->diag[ixy]),&(LU->offdiag[4][ixy]));
    
            fasp_blas_smat_mul_nc5(&(LU->offdiag[0][i1]),&(LU->offdiag[1][i1]),smat);
    
            fasp_blas_darray_axpyz_nc5(-1.0,smat,&(diag[ii]),&(LU->diag[ii]));
            
            if (i>=nline) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[2][ix]),&(LU->offdiag[3][ix]),smat);
                fasp_blas_darray_axpy_nc5(-1.0,smat,&(LU->diag[ii]));
            } //end if (i>=nline)
    
            if (i>=nplane) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[4][ixy]),&(LU->offdiag[5][ixy]),smat);
                fasp_blas_darray_axpy_nc5(-1.0,smat,&(LU->diag[ii]));
            } // end if (i>=nplane)
    
            fasp_smat_inv_nc5(&(LU->diag[ii]));
    
        } // end for(i=1;i<A->ngrid;++i)
    
    }  // end if (nc == 5)
    
    else if (nc == 7) {       
    
        fasp_smat_inv_nc7(LU->diag);
    
        for (i=1;i<ngrid;++i) {
    
            i1=(i-1)*49;    
            ix=(i-nline)*49;
            ixy=(i-nplane)*49;
            ii=i*49;
    
            fasp_blas_smat_mul_nc7(&(offdiag0[i1]),&(LU->diag[i1]),&(LU->offdiag[0][i1]));
    
            if (i>=nline)
                fasp_blas_smat_mul_nc7(&(offdiag2[ix]),&(LU->diag[ix]),&(LU->offdiag[2][ix]));
            if (i>=nplane)
                fasp_blas_smat_mul_nc7(&(offdiag4[ixy]),&(LU->diag[ixy]),&(LU->offdiag[4][ixy]));
    
            fasp_blas_smat_mul_nc7(&(LU->offdiag[0][i1]),&(LU->offdiag[1][i1]),smat);
    
            fasp_blas_darray_axpyz_nc7(-1.0,smat,&(diag[ii]),&(LU->diag[ii]));
            
            if (i>=nline) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[2][ix]),&(LU->offdiag[3][ix]),smat);
                fasp_blas_darray_axpy_nc7(-1.0,smat,&(LU->diag[ii]));
            } //end if (i>=nline)
    
            if (i>=nplane) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[4][ixy]),&(LU->offdiag[5][ixy]),smat);
                fasp_blas_darray_axpy_nc7(-1.0,smat,&(LU->diag[ii]));
            } // end if (i>=nplane)
    
            fasp_smat_inv_nc7(&(LU->diag[ii]));
    
        } // end for(i=1;i<A->ngrid;++i)
    
    }  // end if (nc == 7)
    
    else {
        
        fasp_smat_inv(LU->diag,nc);
    
        for (i=1;i<ngrid;++i) {
    
            i1=(i-1)*nc2;
            ix=(i-nline)*nc2;
            ixy=(i-nplane)*nc2;
            ii=i*nc2;
    
            fasp_blas_smat_mul(&(offdiag0[i1]),&(LU->diag[i1]),&(LU->offdiag[0][i1]),nc);
    
            if (i>=nline)
                fasp_blas_smat_mul(&(offdiag2[ix]),&(LU->diag[ix]),&(LU->offdiag[2][ix]),nc);
            if (i>=nplane)
                fasp_blas_smat_mul(&(offdiag4[ixy]),&(LU->diag[ixy]),&(LU->offdiag[4][ixy]),nc);
    
            fasp_blas_smat_mul(&(LU->offdiag[0][i1]),&(LU->offdiag[1][i1]),smat,nc);
    
            fasp_blas_darray_axpyz(nc2,-1,smat,&(diag[ii]),&(LU->diag[ii]));
            
            if (i>=nline) {
                fasp_blas_smat_mul(&(LU->offdiag[2][ix]),&(LU->offdiag[3][ix]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ii]));
            } //end if (i>=nline)
    
            if (i>=nplane) {
                fasp_blas_smat_mul(&(LU->offdiag[4][ixy]),&(LU->offdiag[5][ixy]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ii]));
            } // end if (i>=nplane)
    
            fasp_smat_inv(&(LU->diag[ii]),nc);
    
        } // end for(i=1;i<A->ngrid;++i)
    
    }
    
    fasp_mem_free(smat); smat = NULL;
    
    return;
}

/**
 * \fn void fasp_ilu_dstr_setup1 (dSTRmat *A, dSTRmat *LU)
 *
 * \brief Get ILU(1) decoposition of a structured matrix A
 *
 * \param A    Pointer to oringinal structured matrix of REAL type
 * \param LU   Pointer to ILU structured matrix of REAL type
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   11/08/2010
 *
 * \note Put L and U in a STR matrix and it has the following structure:
 *       the diag is d, the offdiag of L are alpha1 to alpha6, the offdiag 
 *       of U are beta1 to beta6
 *
 * \note Only works for 5 bands 2D and 7 bands 3D matrix with default offsets
 */
void fasp_ilu_dstr_setup1 (dSTRmat  *A,
                           dSTRmat  *LU)
{
    const INT LUnband = 12;
    INT LUoffsets[12];

    const INT nc=A->nc, nc2=nc*nc;
    const INT nx=A->nx;
    const INT ny=A->ny;
    const INT nz=A->nz;
    const INT nxy=A->nxy;
    const INT nband=A->nband;
    const INT ngrid=A->ngrid;
    INT nline, nplane;

    INT i,j,i1,ix,ixy,ixyx,ix1,ixy1,ic,i1c,ixc,ix1c,ixyc,ixy1c,ixyxc;
    register REAL *smat,t,*tc;
    
    if (nx == 1) {
        nline = ny;
        nplane = ngrid;
    }
    else if (ny == 1) {
        nline = nx;
        nplane = ngrid;
    }
    else if (nz == 1) {
        nline = nx;
        nplane = ngrid;
    }
    else {
        nline = nx;
        nplane = nxy;
    }
    
    smat=(REAL *)fasp_mem_calloc(nc2,sizeof(REAL));
    
    tc=(REAL *)fasp_mem_calloc(nc2,sizeof(REAL));
    
    LUoffsets[0] = -1;
	LUoffsets[1] = 1;
	LUoffsets[2] = 1-nline;
	LUoffsets[3] = nline-1;
	LUoffsets[4] = -nline;
	LUoffsets[5] = nline;
	LUoffsets[6] = nline-nplane;
	LUoffsets[7] = nplane-nline;
	LUoffsets[8] = 1-nplane;
	LUoffsets[9] = nplane-1;
	LUoffsets[10] = -nplane;
	LUoffsets[11] = nplane;
    
    fasp_dstr_alloc(nx,A->ny,A->nz,nxy,ngrid,LUnband,nc,LUoffsets,LU);
    
    if (nband == 6) memcpy(LU->offdiag[11],A->offdiag[5],((ngrid-nxy)*nc2)*sizeof(REAL));
    memcpy(LU->diag,A->diag,nc2*sizeof(REAL));
    
    if (nc == 1) {    
        // comput the first row
        LU->diag[0]=1.0/(LU->diag[0]);
        LU->offdiag[1][0]=A->offdiag[1][0];
        LU->offdiag[5][0]=A->offdiag[3][0];
        LU->offdiag[3][0]=0;
        LU->offdiag[7][0]=0;
        LU->offdiag[9][0]=0;
    
        for (i=1;i<ngrid;++i) {
            
            i1=i-1;ix=i-nline;ixy=i-nplane;ix1=ix+1;ixyx=ixy+nline;ixy1=ixy+1;
    
            // comput alpha6[i-nxy]
            if (ixy>=0)
                LU->offdiag[10][ixy]=A->offdiag[4][ixy]*LU->diag[ixy];
    
            // comput alpha5[ixy1]
            if (ixy1>=0) {
                t=0;
    
                if (ixy>=0) t-=LU->offdiag[10][ixy]*LU->offdiag[1][ixy];     
    
                LU->offdiag[8][ixy1]=t*(LU->diag[ixy1]);
            }
    
            // comput alpha4[ixyx]
            if (ixyx>=0) {
                t=0;
    
                if (ixy>=0) t-=LU->offdiag[10][ixy]*LU->offdiag[5][ixy];
                if (ixy1>=0) t-=LU->offdiag[8][ixy1]*LU->offdiag[3][ixy1];
    
                LU->offdiag[6][ixyx]=t*(LU->diag[ixyx]);
            }
    
            // comput alpha3[ix]
            if (ix>=0) {
                t=A->offdiag[2][ix];
                
                if (ixy>=0) t-=LU->offdiag[10][ixy]*LU->offdiag[7][ixy];
    
                LU->offdiag[4][ix]=t*(LU->diag[ix]);
            }
    
            // comput alpha2[i-nx+1]
            if (ix1>=0) {
                t=0;
    
                if (ix>=0) t-=LU->offdiag[4][ix]*LU->offdiag[1][ix];
                if (ixy1>=0) t-=LU->offdiag[8][ixy1]*LU->offdiag[7][ixy1];
    
                LU->offdiag[2][ix1]=t*(LU->diag[ix1]);
            }
    
            // comput alpha1[i-1]
            t=A->offdiag[0][i1];
    
            if (ix>=0) t-=LU->offdiag[4][ix]*LU->offdiag[3][ix];
            if (ixy>=0) t-=LU->offdiag[10][ixy]*LU->offdiag[9][ixy];
    
            LU->offdiag[0][i1]=t*(LU->diag[i1]);
    
            // comput beta1[i]
            if (i+1<ngrid) {
                t=A->offdiag[1][i];
    
                if (ix1>=0) t-=LU->offdiag[2][ix1]*LU->offdiag[5][ix1];
                if (ixy1>=0) t-=LU->offdiag[8][ixy1]*LU->offdiag[11][ixy1];
    
                LU->offdiag[1][i]=t;
            }
    
            // comput beta2[i]
            if (i+nline-1<ngrid) {
                t=-LU->offdiag[0][i1]*LU->offdiag[5][i1];
    
                if (ixyx>=0) t-=LU->offdiag[6][ixyx]*LU->offdiag[9][ixyx];
    
                LU->offdiag[3][i]=t;
            }
    
            // comput beta3[i]
            if (i+nline<ngrid) {
                t=A->offdiag[3][i];
    
                if (ixyx>=0) t-=LU->offdiag[6][ixyx]*LU->offdiag[11][ixyx];
    
                LU->offdiag[5][i]=t;
            }
    
            // comput beta4[i]
            if (i+nplane-nline<ngrid) {
                t=0;
    
                if (ix1>=0) t-=LU->offdiag[2][ix1]*LU->offdiag[9][ix1];
                if (ix>=0) t-=LU->offdiag[4][ix]*LU->offdiag[11][ix];
    
                LU->offdiag[7][i]=t;
            }
    
            // comput beta5[i]
            if (i+nplane-1<ngrid) LU->offdiag[9][i]=-LU->offdiag[0][i1]*LU->offdiag[11][i1];
    
            // comput d[i]       
            LU->diag[i]=A->diag[i]-(LU->offdiag[0][i1])*(LU->offdiag[1][i1]);
    
            if (ix1>=0) LU->diag[i]-=(LU->offdiag[2][ix1])*(LU->offdiag[3][ix1]);
            if (ix>=0) LU->diag[i]-=(LU->offdiag[4][ix])*(LU->offdiag[5][ix]);
            if (ixyx>=0) LU->diag[i]-=(LU->offdiag[6][ixyx])*(LU->offdiag[7][ixyx]);
            if (ixy1>=0) LU->diag[i]-=(LU->offdiag[8][ixy1])*(LU->offdiag[9][ixy1]);
            if (ixy>=0) LU->diag[i]-=(LU->offdiag[10][ixy])*(LU->offdiag[11][ixy]);
    
            LU->diag[i]=1.0/(LU->diag[i]);
    
        } // end for (i=1; i<ngrid; ++i)
    
    }  // end if (nc == 1)
    
    else if (nc == 3) {
        
        // comput the first row
        fasp_smat_inv_nc3(LU->diag);
        memcpy(LU->offdiag[1],A->offdiag[1],9*sizeof(REAL));
        memcpy(LU->offdiag[5],A->offdiag[3],9*sizeof(REAL));
    
        for (i=1;i<ngrid;++i) {
            i1=i-1;ix=i-nline;ixy=i-nplane;ix1=ix+1;ixyx=ixy+nline;ixy1=ixy+1;
            ic=i*nc2;i1c=i1*nc2;ixc=ix*nc2;ix1c=ix1*nc2;ixyc=ixy*nc2;
            ixy1c=ixy1*nc2;ixyxc=ixyx*nc2;
    
            // comput alpha6[i-nxy]
            if (ixy>=0) fasp_blas_smat_mul_nc3(&(A->offdiag[4][ixyc]),&(LU->diag[ixyc]),&(LU->offdiag[10][ixyc]));
    
            // comput alpha5[ixy1]
            if (ixy1>=0) {
                for (j=0;j<9;++j) tc[j]=0;
    
                if (ixy>=0) {    
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[10][ixyc]),&(LU->offdiag[1][ixyc]),smat);  
                    fasp_blas_darray_axpy_nc3(-1,smat,tc); 
                }  
    
                fasp_blas_smat_mul_nc3(tc,&(LU->diag[ixy1c]),&(LU->offdiag[8][ixy1c]));
            }
    
            // comput alpha4[ixyx]
            if (ixyx>=0) {
                for (j=0;j<9;++j) tc[j]=0;
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[10][ixyc]),&(LU->offdiag[5][ixyc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[3][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc3(tc,&(LU->diag[ixyxc]),&(LU->offdiag[6][ixyxc]));
            }
    
            // comput alpha3[ix]
            if (ix>=0) {
    
                memcpy(tc,&(A->offdiag[2][ixc]),9*sizeof(REAL));
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[10][ixyc]),&(LU->offdiag[7][ixyc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc3(tc,&(LU->diag[ixc]),&(LU->offdiag[4][ixc]));
            }
    
            // comput alpha2[i-nx+1]
            if (ix1>=0) {
    
                for (j=0;j<9;++j) tc[j]=0;
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[4][ixc]),&(LU->offdiag[1][ixc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[7][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc3(tc,&(LU->diag[ix1c]),&(LU->offdiag[2][ix1c]));
    
            } // end if (ix1 >= 0)
    
            // comput alpha1[i-1]
    
            memcpy(tc,&(A->offdiag[0][i1c]),9*sizeof(REAL));
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[4][ixc]),&(LU->offdiag[3][ixc]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,tc);
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[10][ixyc]),&(LU->offdiag[9][ixyc]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,tc);
            }
    
            fasp_blas_smat_mul_nc3(tc,&(LU->diag[i1c]),&(LU->offdiag[0][i1c]));
    
            // comput beta1[i]
            if (i+1<ngrid) {
    
                memcpy(&(LU->offdiag[1][ic]),&(A->offdiag[1][ic]),9*sizeof(REAL));
                
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[2][ix1c]),&(LU->offdiag[5][ix1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[1][ic]));
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[11][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[1][ic]));
                }
    
            }
    
            // comput beta2[i]
            if (i+nline-1<ngrid) {
    
                {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[0][i1c]),&(LU->offdiag[5][i1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[3][ic]));
                }
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[9][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[3][ic]));
                }
    
            }
    
            // comput beta3[i]
            if (i+nline<ngrid) {
                
                memcpy(&(LU->offdiag[5][ic]),&(A->offdiag[3][ic]),9*sizeof(REAL));
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[11][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[5][ic]));
                }
    
            }
    
            // comput beta4[i]
            if (i+nplane-nline<ngrid) {
    
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[2][ix1c]),&(LU->offdiag[9][ix1c]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[7][ic]));
                }
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc3(&(LU->offdiag[4][ixc]),&(LU->offdiag[11][ixc]),smat);
                    fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[7][ic]));
                }
    
            }
    
            // comput beta5[i]
            if (i+nplane-1<ngrid) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[0][i1c]),&(LU->offdiag[11][i1c]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->offdiag[9][ic]));
            }
    
            // comput d[i]
            {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[0][i1c]),&(LU->offdiag[1][i1c]),smat);
                fasp_blas_darray_axpyz_nc3(-1,smat,&(A->diag[ic]),&(LU->diag[ic]));
            }
    
            if (ix1>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[2][ix1c]),&(LU->offdiag[3][ix1c]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ic]));
            }
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[4][ixc]),&(LU->offdiag[5][ixc]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixyx>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[7][ixyxc]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy1>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[9][ixy1c]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc3(&(LU->offdiag[10][ixyc]),&(LU->offdiag[11][ixyc]),smat);
                fasp_blas_darray_axpy_nc3(-1,smat,&(LU->diag[ic]));
            }
    
            fasp_smat_inv_nc3(&(LU->diag[ic]));
    
        } // end for(i=1;i<ngrid;++i)
    
    }  // end if (nc == 3)
    
    else if (nc == 5) {
        // comput the first row
        // fasp_smat_inv_nc5(LU->diag);
        fasp_smat_inv(LU->diag,5);
        memcpy(LU->offdiag[1],A->offdiag[1], 25*sizeof(REAL));
        memcpy(LU->offdiag[5],A->offdiag[3], 25*sizeof(REAL));
    
        for(i=1;i<ngrid;++i) {
            i1=i-1;ix=i-nline;ixy=i-nplane;ix1=ix+1;ixyx=ixy+nline;ixy1=ixy+1;
            ic=i*nc2;i1c=i1*nc2;ixc=ix*nc2;ix1c=ix1*nc2;ixyc=ixy*nc2;ixy1c=ixy1*nc2;ixyxc=ixyx*nc2;
    
            // comput alpha6[i-nxy]
            if (ixy>=0) fasp_blas_smat_mul_nc5(&(A->offdiag[4][ixyc]),&(LU->diag[ixyc]),&(LU->offdiag[10][ixyc]));
    
            // comput alpha5[ixy1]
            if (ixy1>=0) {
                for (j=0;j<25;++j) tc[j]=0;
    
                if (ixy>=0) {    
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[10][ixyc]),&(LU->offdiag[1][ixyc]),smat);  
                    fasp_blas_darray_axpy_nc5(-1.0,smat,tc); 
                }  
    
                fasp_blas_smat_mul_nc5(tc,&(LU->diag[ixy1c]),&(LU->offdiag[8][ixy1c]));
            }
    
            // comput alpha4[ixyx]
            if (ixyx>=0) {
                for (j=0;j<25;++j) tc[j]=0;
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[10][ixyc]),&(LU->offdiag[5][ixyc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[3][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc5(tc,&(LU->diag[ixyxc]),&(LU->offdiag[6][ixyxc]));
            }
    
            // comput alpha3[ix]
            if (ix>=0) {
    
                memcpy(tc,&(A->offdiag[2][ixc]),25*sizeof(REAL));
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[10][ixyc]),&(LU->offdiag[7][ixyc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc5(tc,&(LU->diag[ixc]),&(LU->offdiag[4][ixc]));
            }
    
            // comput alpha2[i-nx+1]
            if (ix1>=0) {
    
                for (j=0;j<25;++j) tc[j]=0;
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[4][ixc]),&(LU->offdiag[1][ixc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[7][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc5(tc,&(LU->diag[ix1c]),&(LU->offdiag[2][ix1c]));
    
            } // end if (ix1 >= 0)
    
            // comput alpha1[i-1]
    
            memcpy(tc,&(A->offdiag[0][i1c]), 25*sizeof(REAL));
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[4][ixc]),&(LU->offdiag[3][ixc]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,tc);
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[10][ixyc]),&(LU->offdiag[9][ixyc]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,tc);
            }
    
            fasp_blas_smat_mul_nc5(tc,&(LU->diag[i1c]),&(LU->offdiag[0][i1c]));
    
            // comput beta1[i]
            if (i+1<ngrid) {
    
                memcpy(&(LU->offdiag[1][ic]),&(A->offdiag[1][ic]), 25*sizeof(REAL));
                
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[2][ix1c]),&(LU->offdiag[5][ix1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[1][ic]));
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[11][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[1][ic]));
                }
    
            }
    
            // comput beta2[i]
            if (i+nline-1<ngrid) {
    
                {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[0][i1c]),&(LU->offdiag[5][i1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[3][ic]));
                }
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[9][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[3][ic]));
                }
    
            }
    
            // comput beta3[i]
            if (i+nline<ngrid) {
                
                memcpy(&(LU->offdiag[5][ic]),&(A->offdiag[3][ic]), 25*sizeof(REAL));
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[11][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[5][ic]));
                }
    
            }
    
            // comput beta4[i]
            if (i+nplane-nline<ngrid) {
    
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[2][ix1c]),&(LU->offdiag[9][ix1c]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[7][ic]));
                }
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc5(&(LU->offdiag[4][ixc]),&(LU->offdiag[11][ixc]),smat);
                    fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[7][ic]));
                }
    
            }
    
            // comput beta5[i]
            if (i+nplane-1<ngrid) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[0][i1c]),&(LU->offdiag[11][i1c]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->offdiag[9][ic]));
            }
    
            // comput d[i]       
            {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[0][i1c]),&(LU->offdiag[1][i1c]),smat);
                fasp_blas_darray_axpyz_nc5(-1,smat,&(A->diag[ic]),&(LU->diag[ic]));
            }
    
            if (ix1>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[2][ix1c]),&(LU->offdiag[3][ix1c]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->diag[ic]));
            }
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[4][ixc]),&(LU->offdiag[5][ixc]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixyx>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[7][ixyxc]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy1>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[9][ixy1c]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc5(&(LU->offdiag[10][ixyc]),&(LU->offdiag[11][ixyc]),smat);
                fasp_blas_darray_axpy_nc5(-1,smat,&(LU->diag[ic]));
            }
    
            //fasp_smat_inv_nc5(&(LU->diag[ic]));
            fasp_smat_inv(&(LU->diag[ic]), 5);
    
        } // end for(i=1;i<ngrid;++i)
    
    }  // end if (nc == 5)
    
    else if (nc == 7) {
        // comput the first row
        //fasp_smat_inv_nc5(LU->diag);
        fasp_smat_inv(LU->diag,7);
        memcpy(LU->offdiag[1],A->offdiag[1], 49*sizeof(REAL));
        memcpy(LU->offdiag[5],A->offdiag[3], 49*sizeof(REAL));
    
        for(i=1;i<ngrid;++i) {   
            i1=i-1;ix=i-nline;ixy=i-nplane;ix1=ix+1;ixyx=ixy+nline;ixy1=ixy+1;
            ic=i*nc2;i1c=i1*nc2;ixc=ix*nc2;ix1c=ix1*nc2;ixyc=ixy*nc2;ixy1c=ixy1*nc2;ixyxc=ixyx*nc2;
    
            // comput alpha6[i-nxy]
            if (ixy>=0) fasp_blas_smat_mul_nc7(&(A->offdiag[4][ixyc]),&(LU->diag[ixyc]),&(LU->offdiag[10][ixyc]));
    
            // comput alpha5[ixy1]
            if (ixy1>=0) {
                for (j=0;j<49;++j) tc[j]=0;
    
                if (ixy>=0) {    
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[10][ixyc]),&(LU->offdiag[1][ixyc]),smat);  
                    fasp_blas_darray_axpy_nc7(-1.0,smat,tc); 
                }  
    
                fasp_blas_smat_mul_nc7(tc,&(LU->diag[ixy1c]),&(LU->offdiag[8][ixy1c]));
            }
    
            // comput alpha4[ixyx]
            if (ixyx>=0) {
                for (j=0;j<49;++j) tc[j]=0;
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[10][ixyc]),&(LU->offdiag[5][ixyc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[3][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc7(tc,&(LU->diag[ixyxc]),&(LU->offdiag[6][ixyxc]));
            }
    
            // comput alpha3[ix]
            if (ix>=0) {
    
                memcpy(tc,&(A->offdiag[2][ixc]),49*sizeof(REAL));
    
                if (ixy>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[10][ixyc]),&(LU->offdiag[7][ixyc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc7(tc,&(LU->diag[ixc]),&(LU->offdiag[4][ixc]));
            }
    
            // comput alpha2[i-nx+1]
            if (ix1>=0) {
    
                for (j=0;j<49;++j) tc[j]=0;
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[4][ixc]),&(LU->offdiag[1][ixc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[7][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,tc);
                }
    
                fasp_blas_smat_mul_nc7(tc,&(LU->diag[ix1c]),&(LU->offdiag[2][ix1c]));
    
            } // end if (ix1 >= 0)
    
            // comput alpha1[i-1]
    
            memcpy(tc,&(A->offdiag[0][i1c]), 49*sizeof(REAL));
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[4][ixc]),&(LU->offdiag[3][ixc]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,tc);
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[10][ixyc]),&(LU->offdiag[9][ixyc]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,tc);
            }
    
            fasp_blas_smat_mul_nc7(tc,&(LU->diag[i1c]),&(LU->offdiag[0][i1c]));
    
            // comput beta1[i]
            if (i+1<ngrid) {
    
                memcpy(&(LU->offdiag[1][ic]),&(A->offdiag[1][ic]), 49*sizeof(REAL));
                
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[2][ix1c]),&(LU->offdiag[5][ix1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[1][ic]));
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[11][ixy1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[1][ic]));
                }
    
            }
    
            // comput beta2[i]
            if (i+nline-1<ngrid) {
    
                {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[0][i1c]),&(LU->offdiag[5][i1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[3][ic]));
                }
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[9][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[3][ic]));
                }
    
            }
    
            // comput beta3[i]
            if (i+nline<ngrid) {
                
                memcpy(&(LU->offdiag[5][ic]),&(A->offdiag[3][ic]), 49*sizeof(REAL));
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[11][ixyxc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[5][ic]));
                }
    
            }
    
            // comput beta4[i]
            if (i+nplane-nline<ngrid) {
    
                if (ix1>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[2][ix1c]),&(LU->offdiag[9][ix1c]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[7][ic]));
                }
    
                if (ix>=0) {
                    fasp_blas_smat_mul_nc7(&(LU->offdiag[4][ixc]),&(LU->offdiag[11][ixc]),smat);
                    fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[7][ic]));
                }
    
            }
    
            // comput beta5[i]
            if (i+nplane-1<ngrid) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[0][i1c]),&(LU->offdiag[11][i1c]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->offdiag[9][ic]));
            }
    
            // comput d[i]       
            {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[0][i1c]),&(LU->offdiag[1][i1c]),smat);
                fasp_blas_darray_axpyz_nc7(-1,smat,&(A->diag[ic]),&(LU->diag[ic]));
            }
    
            if (ix1>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[2][ix1c]),&(LU->offdiag[3][ix1c]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->diag[ic]));
            }
    
            if (ix>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[4][ixc]),&(LU->offdiag[5][ixc]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixyx>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[7][ixyxc]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy1>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[9][ixy1c]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->diag[ic]));
            }
    
            if (ixy>=0) {
                fasp_blas_smat_mul_nc7(&(LU->offdiag[10][ixyc]),&(LU->offdiag[11][ixyc]),smat);
                fasp_blas_darray_axpy_nc7(-1,smat,&(LU->diag[ic]));
            }
    
            //fasp_smat_inv_nc5(&(LU->diag[ic]));
            fasp_smat_inv(&(LU->diag[ic]), 7);
    
        } // end for(i=1;i<ngrid;++i)
    
    }  // end if (nc == 7)
    
    else {
        // comput the first row
        fasp_smat_inv(LU->diag,nc);
        memcpy(LU->offdiag[1],A->offdiag[1],nc2*sizeof(REAL));
        memcpy(LU->offdiag[5],A->offdiag[3],nc2*sizeof(REAL));
    
        for(i=1;i<ngrid;++i) {   
    
            i1=i-1;ix=i-nline;ixy=i-nplane;ix1=ix+1;ixyx=ixy+nline;ixy1=ixy+1;
            ic=i*nc2;i1c=i1*nc2;ixc=ix*nc2;ix1c=ix1*nc2;ixyc=ixy*nc2;ixy1c=ixy1*nc2;ixyxc=ixyx*nc2;
            // comput alpha6[i-nxy]
            if (ixy>=0)
                fasp_blas_smat_mul(&(A->offdiag[4][ixyc]),&(LU->diag[ixyc]),&(LU->offdiag[10][ixyc]),nc);
    
            // comput alpha5[ixy1]
            if (ixy1>=0) {
                for (j=0;j<nc2;++j) tc[j]=0;
                if (ixy>=0) {    
                    fasp_blas_smat_mul(&(LU->offdiag[10][ixyc]),&(LU->offdiag[1][ixyc]),smat,nc);  
                    fasp_blas_darray_axpy(nc2,-1,smat,tc); 
                }  
    
                fasp_blas_smat_mul(tc,&(LU->diag[ixy1c]),&(LU->offdiag[8][ixy1c]),nc);
            }
    
            // comput alpha4[ixyx]
            if (ixyx>=0) {
                for (j=0;j<nc2;++j) tc[j]=0;
                if (ixy>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[10][ixyc]),&(LU->offdiag[5][ixyc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,tc);
                }
                if (ixy1>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[3][ixy1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,tc);
                }
    
                fasp_blas_smat_mul(tc,&(LU->diag[ixyxc]),&(LU->offdiag[6][ixyxc]),nc);
            }
    
            // comput alpha3[ix]
            if (ix>=0) {
    
                memcpy(tc,&(A->offdiag[2][ixc]),nc2*sizeof(REAL));
                if (ixy>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[10][ixyc]),&(LU->offdiag[7][ixyc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,tc);
                }
    
                fasp_blas_smat_mul(tc,&(LU->diag[ixc]),&(LU->offdiag[4][ixc]),nc);
            }
    
            // comput alpha2[i-nx+1]
            if (ix1>=0) {
    
                for (j=0;j<nc2;++j) tc[j]=0;
    
                if (ix>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[4][ixc]),&(LU->offdiag[1][ixc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,tc);
                }
    
                if (ixy1>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[7][ixy1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,tc);
                }
    
                fasp_blas_smat_mul(tc,&(LU->diag[ix1c]),&(LU->offdiag[2][ix1c]),nc);
            }
    
            // comput alpha1[i-1]
    
            memcpy(tc,&(A->offdiag[0][i1c]),nc2*sizeof(REAL));
            if (ix>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[4][ixc]),&(LU->offdiag[3][ixc]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,tc);
            }
            if (ixy>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[10][ixyc]),&(LU->offdiag[9][ixyc]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,tc);
            }
    
            fasp_blas_smat_mul(tc,&(LU->diag[i1c]),&(LU->offdiag[0][i1c]),nc);
    
            // comput beta1[i]
            if (i+1<ngrid) {
    
                memcpy(&(LU->offdiag[1][ic]),&(A->offdiag[1][ic]),nc2*sizeof(REAL));
                if (ix1>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[2][ix1c]),&(LU->offdiag[5][ix1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[1][ic]));
                }
                if (ixy1>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[11][ixy1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[1][ic]));
                }
    
            }
    
            // comput beta2[i]
            if (i+nline-1<ngrid) {
    
                {
                    fasp_blas_smat_mul(&(LU->offdiag[0][i1c]),&(LU->offdiag[5][i1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[3][ic]));
                }
    
                if (ixyx>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[9][ixyxc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[3][ic]));
                }
    
            }
    
            // comput beta3[i]
            if (i+nline<ngrid) {
                
                memcpy(&(LU->offdiag[5][ic]),&(A->offdiag[3][ic]),nc2*sizeof(REAL));
                if (ixyx>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[11][ixyxc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[5][ic]));
                }
    
            }
    
            // comput beta4[i]
            if (i+nplane-nline<ngrid) {
    
                if (ix1>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[2][ix1c]),&(LU->offdiag[9][ix1c]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[7][ic]));
                }
    
                if (ix>=0) {
                    fasp_blas_smat_mul(&(LU->offdiag[4][ixc]),&(LU->offdiag[11][ixc]),smat,nc);
                    fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[7][ic]));
                }
            }
    
            // comput beta5[i]
            if (i+nplane-1<ngrid) {
                fasp_blas_smat_mul(&(LU->offdiag[0][i1c]),&(LU->offdiag[11][i1c]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->offdiag[9][ic]));
            }
    
            // comput d[i]           
            {
                fasp_blas_smat_mul(&(LU->offdiag[0][i1c]),&(LU->offdiag[1][i1c]),smat,nc);
                fasp_blas_darray_axpyz(nc2,-1,smat,&(A->diag[ic]),&(LU->diag[ic]));
            }
    
            if (ix1>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[2][ix1c]),&(LU->offdiag[3][ix1c]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ic]));
            }
        
            if (ix>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[4][ixc]),&(LU->offdiag[5][ixc]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ic]));
            }
        
            if (ixyx>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[6][ixyxc]),&(LU->offdiag[7][ixyxc]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ic]));
            }
    
    
            if (ixy1>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[8][ixy1c]),&(LU->offdiag[9][ixy1c]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ic]));
            }
        
            if (ixy>=0) {
                fasp_blas_smat_mul(&(LU->offdiag[10][ixyc]),&(LU->offdiag[11][ixyc]),smat,nc);
                fasp_blas_darray_axpy(nc2,-1,smat,&(LU->diag[ic]));
            }
    
            fasp_smat_inv(&(LU->diag[ic]),nc);
        
        }
    
    } // end else
    
    fasp_mem_free(smat); smat = NULL;
    fasp_mem_free(tc);   tc   = NULL;
    
    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
