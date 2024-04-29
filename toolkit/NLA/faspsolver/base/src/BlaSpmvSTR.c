/*! \file  BlaSpmvSTR.c
 *
 *  \brief Linear algebraic operations for dSTRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxThreads.c, BlaSmallMatInv.c, BlaSmallMat.c,
 *         and BlaSparseSTR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static inline void smat_amxv_nc3(const REAL, const REAL *, const REAL *, REAL *);
static inline void smat_amxv_nc5(const REAL, const REAL *, const REAL *, REAL *);
static inline void smat_amxv(const REAL, const REAL *, const REAL *, const INT, REAL *);
static inline void str_spaAxpy_2D_nc1(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_2D_nc3(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_2D_nc5(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_2D_blk(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_3D_nc1(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_3D_nc3(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_3D_nc5(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy_3D_blk(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void str_spaAxpy(const REAL, const dSTRmat *, const REAL *, REAL *);
static inline void blkcontr_str(const INT, const INT, const INT, const INT,
                                const REAL *, const REAL *, REAL *);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_blas_dstr_aAxpy (const REAL alpha, const dSTRmat *A,
 *                                const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Zhiyang Zhou, Xiaozhe Hu, Shiquan Zhang
 * \date   2010/10/15
 */
void fasp_blas_dstr_aAxpy (const REAL      alpha,
                           const dSTRmat  *A,
                           const REAL     *x,
                           REAL           *y)
{
    
    switch (A->nband) {
            
        case 4:
            
            switch (A->nc) {
                case 1:
                    str_spaAxpy_2D_nc1(alpha, A, x, y);
                    break;
                    
                case 3:
                    str_spaAxpy_2D_nc3(alpha, A, x, y);
                    break;
                    
                case 5:
                    str_spaAxpy_2D_nc5(alpha, A, x, y);
                    break;
                    
                default:
                    str_spaAxpy_2D_blk(alpha, A, x, y);
                    break;
            }
            
            break;
            
        case 6:
            
            switch (A->nc) {
                case 1:
                    str_spaAxpy_3D_nc1(alpha, A, x, y);
                    break;
                    
                case 3:
                    str_spaAxpy_3D_nc3(alpha, A, x, y);
                    break;
                    
                case 5:
                    str_spaAxpy_3D_nc5(alpha, A, x, y);
                    break;
                    
                default:
                    str_spaAxpy_3D_blk(alpha, A, x, y);
                    break;
            }
            break;
            
        default:
            str_spaAxpy(alpha, A, x, y);
            break;
    }
    
}

/**
 * \fn void fasp_blas_dstr_mxv (const dSTRmat *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Chensong Zhang
 * \date   04/27/2013
 */
void fasp_blas_dstr_mxv (const dSTRmat  *A,
                         const REAL     *x,
                         REAL           *y)
{
    int n = (A->ngrid)*(A->nc)*(A->nc);
    
    memset(y, 0, n*sizeof(REAL));
    
    fasp_blas_dstr_aAxpy(1.0, A, x, y);
}

/*!
 * \fn INT fasp_blas_dstr_diagscale (const dSTRmat *A, dSTRmat *B)
 *
 * \brief B=D^{-1}A
 *
 * \param A   Pointer to a 'dSTRmat' type matrix A
 * \param B   Pointer to a 'dSTRmat' type matrix B
 *
 * \author Shiquan Zhang
 * \date   2010/10/15
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/30/2012
 */
INT fasp_blas_dstr_diagscale (const dSTRmat  *A,
                              dSTRmat        *B)
{
    const INT ngrid=A->ngrid, nc=A->nc, nband=A->nband;
    const INT nc2=nc*nc, size=ngrid*nc2;
    INT i,j,ic2,nb,nb1;
    
#ifdef _OPENMP
    //variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
    REAL *diag=(REAL *)fasp_mem_calloc(size,sizeof(REAL));
    
    fasp_darray_cp(size,A->diag,diag);
    
    fasp_dstr_alloc(A->nx, A->ny, A->nz,A->nxy,ngrid, nband,nc,A->offsets, B);
    
    //compute diagnal elements of B
#ifdef _OPENMP
    if (ngrid > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, ic2, j)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, ngrid, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                ic2=i*nc2;
                for (j=0; j<nc2; j++) {
                    if (j/nc == j%nc) B->diag[ic2+j]=1;
                    else B->diag[ic2+j]=0;
                }
            }
        }
    }
    else {
#endif
        for (i=0;i<ngrid;++i) {
            ic2=i*nc2;
            for (j=0;j<nc2;++j) {
                if (j/nc == j%nc) B->diag[ic2+j]=1;
                else B->diag[ic2+j]=0;
            }
        }
#ifdef _OPENMP
    }
#endif
    
    for (i=0;i<ngrid;++i) fasp_smat_inv(&(diag[i*nc2]),nc);
    
    for (i=0;i<nband;++i) {
        nb=A->offsets[i];
        nb1=abs(nb);
        if (nb<0) {
            for (j=0;j<ngrid-nb1;++j)
                fasp_blas_smat_mul(&(diag[(j+nb1)*nc2]),&(A->offdiag[i][j*nc2]),&(B->offdiag[i][j*nc2]),nc);
        }
        else {
            for (j=0;j<ngrid-nb1;++j)
                fasp_blas_smat_mul(&(diag[j*nc2]),&(A->offdiag[i][j*nc2]),&(B->offdiag[i][j*nc2]),nc);
        }
        
    }
    
    fasp_mem_free(diag); diag = NULL;
    
    return (0);
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static inline void smat_amxv_nc3 (const REAL alpha, const REAL *a, 
 *                                       const REAL *b, REAL *c)
 *
 * \brief Matrix-vector multiplication c = alpha*a*b + c where a is a 3*3 full matrix
 *
 * \param alpha   REAL factor alpha
 * \param a       Pointer to the REAL vector which stands a 3*3 matrix
 * \param b       Pointer to the REAL vector with length 3
 * \param c       Pointer to the REAL vector with length 3
 *
 * \author Shiquan Zhang
 * \date   2010/10/15
 */
static inline void smat_amxv_nc3 (const REAL   alpha,
                                  const REAL  *a,
                                  const REAL  *b,
                                  REAL        *c)
{
    c[0] += alpha*(a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
    c[1] += alpha*(a[3]*b[0] + a[4]*b[1] + a[5]*b[2]);
    c[2] += alpha*(a[6]*b[0] + a[7]*b[1] + a[8]*b[2]);
}

/**
 * \fn static inline void smat_amxv_nc5 (const REAL alpha, const REAL *a, 
 *                                       const REAL *b, REAL *c)
 *
 * \brief  Matrix-vector multiplication c = alpha*a*b + c where a is a 5*5 full matrix
 *
 * \param alpha   REAL factor alpha
 * \param a       Pointer to the REAL vector which stands a 5*5 matrix
 * \param b       Pointer to the REAL vector with length 5
 * \param c       Pointer to the REAL vector with length 5
 *
 * \author Shiquan Zhang
 * \date   2010/10/15
 */
static inline void smat_amxv_nc5 (const REAL   alpha,
                                  const REAL  *a,
                                  const REAL  *b,
                                  REAL        *c)
{
    c[0] += alpha*(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3] * b[3] + a[4] * b[4]);
    c[1] += alpha*(a[5]*b[0] + a[6]*b[1] + a[7]*b[2] + a[8] * b[3] + a[9] * b[4]);
    c[2] += alpha*(a[10]*b[0] + a[11]*b[1] + a[12]*b[2] + a[13] * b[3] + a[14] * b[4]);
    c[3] += alpha*(a[15]*b[0] + a[16]*b[1] + a[17]*b[2] + a[18] * b[3] + a[19] * b[4]);
    c[4] += alpha*(a[20]*b[0] + a[21]*b[1] + a[22]*b[2] + a[23] * b[3] + a[24] * b[4]);
}

/**
 * \fn static inline void smat_amxv (const REAL alpha, const REAL *a, const REAL *b,
 *                                   const INT n, REAL *c)
 *
 * \brief  Matrix-vector multiplication c = alpha*a*b + c where a is a n*n full matrix
 *
 * \param alpha   REAL factor alpha
 * \param a       Pointer to the REAL vector which stands a n*n matrix
 * \param b       Pointer to the REAL vector with length n
 * \param c       Pointer to the REAL vector with length n
 * \param n the dimension of the matrix
 *
 * \author Shiquan Zhang
 * \date   2010/10/15
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/30/2012
 */
static inline void smat_amxv (const REAL   alpha,
                              const REAL  *a,
                              const REAL  *b,
                              const INT    n,
                              REAL       *c)
{
    INT i,j;
    INT in;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
#ifdef _OPENMP
    if (n > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, in, j)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                in = i*n;
                for (j=0; j<n; j++)
                    c[i] += alpha*a[in+j]*b[j];
            }
        }
    }
    else {
#endif
        for (i=0;i<n;++i) {
            in = i*n;
            for (j=0;j<n;++j)
                c[i] += alpha*a[in+j]*b[j];
        }
#ifdef _OPENMP
    }
#endif
    return;
}

/**
 * \fn static inline void blkcontr_str (const INT start_data, const INT start_vecx,
 *                                      const INT start_vecy, const INT nc, 
 *                                      const REAL *data, const REAL *x, REAL *y)
 *
 * \brief contribute the block computation 'P*z' to 'y', where 'P' is a nc*nc
 *        full matrix stored in 'data' from the address 'start_data', and 'z'
 *        is a nc*1 vector stored in 'x' from the address 'start_vect'.
 *
 * \param start_data Starting position of data
 * \param start_vecx Starting position of vecx
 * \param start_vecy Starting position of vecy
 * \param nc         Dimension of the submatrix
 * \param data       Pointer to matrix data
 * \param x          Pointer to REAL array x
 * \param y          Pointer to REAL array y
 *
 * \author Shiquan Zhang
 * \date   2010/04/24
 *
 * Modified by Chunsheng Feng, Zheng Li on 08/30/2012
 */
static inline void blkcontr_str (const INT   start_data,
                                 const INT   start_vecx,
                                 const INT   start_vecy,
                                 const INT   nc,
                                 const REAL *data,
                                 const REAL *x,
                                 REAL *y)
{
    INT i,j,k,m;
    
#ifdef _OPENMP
    //variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
#ifdef _OPENMP
    if (nc > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, k, m, j)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, nc, &mybegin, &myend);
            for (i = mybegin; i < myend; i ++) {
                k = start_data + i*nc;
                m = start_vecy + i;
                for (j = 0; j < nc; j ++) {
                    y[m] += data[k+j]*x[start_vecx+j];
                }
            }
        }
    }
    else {
#endif
        for (i = 0; i < nc; i ++) {
            k = start_data + i*nc;
            m = start_vecy + i;
            for (j = 0; j < nc; j ++) {
                y[m] += data[k+j]*x[start_vecx+j];
            }
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn static inline void str_spaAxpy_2D_nc1 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 5 banded 2D
 *        structured matrix (nc = 1)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang
 * \date   2010/10/15
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/08/28
 *
 * \note The offsets of the five bands have to be (-1, +1, -nx, +nx) for nx != 1
 *       and (-1,+1,-ny,+ny) for nx = 1, but the order can be arbitrary.
 */
static inline void str_spaAxpy_2D_nc1 (const REAL     alpha,
                                       const dSTRmat *A,
                                       const REAL    *x,
                                       REAL          *y)
{
    INT i;
    INT idx1, idx2;
    INT end1, end2;
    INT nline;
    
#ifdef _OPENMP
    //variables for OpenMP
    INT myid, mybegin, myend, idx;
    INT nthreads = fasp_get_num_threads();
#endif
    
    // information of A
    INT nx = A->nx;
    INT ngrid = A->ngrid;  // number of grids
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL, *offdiag3=NULL;
    
    if (nx == 1) {
        nline = A->ny;
    }
    else {
        nline = nx;
    }
    
    for (i=0; i<nband; ++i) {
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nline) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nline) {
            offdiag3 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nline;
    
    y[0] += alpha*(diag[0]*x[0] + offdiag1[0]*x[1] + offdiag3[0]*x[nline]);
    
#ifdef _OPENMP
    if (nline-1 > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, idx1, idx)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, nline-1, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx1 = i;
                idx  = i+1;
                y[idx] += alpha*(offdiag0[idx1]*x[idx1] + diag[idx]*x[idx] +
                                 offdiag1[idx]*x[idx+1] + offdiag3[idx]*x[idx+nline]);
            }
        }
    }
    else {
#endif
        for (i=1; i<nline; ++i) {
            idx1 = i-1;
            y[i] += alpha*(offdiag0[idx1]*x[idx1] + diag[i]*x[i] +
                           offdiag1[i]*x[i+1] + offdiag3[i]*x[i+nline]);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    if (end2-nline > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, i, mybegin, myend, idx1, idx2, idx)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, end2-nline, &mybegin, &myend);
            for (i=mybegin; i<myend; ++i) {
                idx  = i+nline;
                idx1 = idx-1; //idx1 = i-1+nline;
                idx2 = i;
                y[idx] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1] +
                                 diag[idx]*x[idx] + offdiag1[idx]*x[idx+1] +
                                 offdiag3[idx]*x[idx+nline]);
            }
        }
    }
    else {
#endif
        for (i=nline; i<end2; ++i) {
            idx1 = i-1;
            idx2 = i-nline;
            y[i] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1] +
                           diag[i]*x[i] + offdiag1[i]*x[i+1] + offdiag3[i]*x[i+nline]);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    if (end1-end2 > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, i, mybegin, myend, idx1, idx2, idx)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, end1-end2, &mybegin, &myend);
            for (i=mybegin; i<myend; ++i) {
                idx  = i+end2;
                idx1 = idx-1;     //idx1 = i-1+end2;
                idx2 = idx-nline; //idx2 = i-nline+end2;
                y[idx] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1] +
                                 diag[idx]*x[idx] + offdiag1[idx]*x[idx+1]);
            }
        }
    }
    else {
#endif
        for (i=end2; i<end1; ++i) {
            idx1 = i-1;
            idx2 = i-nline;
            y[i] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1] +
                           diag[i]*x[i] + offdiag1[i]*x[i+1]);
        }
#ifdef _OPENMP
    }
#endif
    
    idx1 = end1-1;
    idx2 = end1-nline;
    y[end1] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1] + diag[end1]*x[end1]);
    
    return;
    
}

/**
 * \fn static inline void str_spaAxpy_2D_nc3 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 5 banded
 *        2D structured matrix (nc = 3)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * Modified by Chunsheng Feng, Zheng Li on 2012/08/30
 *
 * \note the offsets of the five bands have to be (-1, +1, -nx, +nx) for nx != 1
 *       and (-1,+1,-ny,+ny) for nx = 1, but the order can be arbitrary.
 */
static inline void str_spaAxpy_2D_nc3 (const REAL     alpha,
                                       const dSTRmat *A,
                                       const REAL    *x,
                                       REAL          *y)
{
    INT i;
    INT idx,idx1,idx2;
    INT matidx, matidx1, matidx2;
    INT end1, end2;
    INT nline, nlinenc;
    
    // information of A
    INT nx = A->nx;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nband = A->nband;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend, up;
    INT nthreads = fasp_get_num_threads();
#endif
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL, *offdiag3=NULL;
    
    if (nx == 1) {
        nline = A->ny;
    }
    else {
        nline = nx;
    }
    nlinenc = nline*nc;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nline) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nline) {
            offdiag3 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
        
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nline;
    
    smat_amxv_nc3(alpha, diag, x, y);
    smat_amxv_nc3(alpha, offdiag1, x+nc, y);
    smat_amxv_nc3(alpha, offdiag3, x+nlinenc, y);
    
#ifdef _OPENMP
    up = nline - 1;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, idx, matidx, idx1, matidx1)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+1)*nc;
                matidx = idx*nc;
                idx1 = i*nc;
                matidx1 = idx1*nc;
                smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
                smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
                smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=1; i<nline; ++i) {
            idx = i*nc;
            matidx = idx*nc;
            idx1 = idx - nc;
            matidx1 = idx1*nc;
            smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    up = end2 - nx;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, idx, idx1, idx2, matidx, matidx1, matidx2)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+nx)*nc;
                idx1 = idx-nc;
                idx2 = idx-nlinenc;
                matidx = idx*nc;
                matidx1 = idx1*nc;
                matidx2 = idx2*nc;
                smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
                smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc3(alpha, diag+matidx,      x+idx,  y+idx);
                smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
                smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=nx; i<end2; ++i) {
            idx = i*nc;
            idx1 = idx-nc;
            idx2 = idx-nlinenc;
            matidx = idx*nc;
            matidx1 = idx1*nc;
            matidx2 = idx2*nc;
            smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
            smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    up = end1 - end2;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, idx, idx1, idx2, matidx, matidx1, matidx2)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+end2)*nc;
                idx1 = idx-nc;
                idx2 = idx-nlinenc;
                matidx = idx*nc;
                matidx1 = idx1*nc;
                matidx2 = idx2*nc;
                smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
                smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
                smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=end2; i<end1; ++i) {
            idx = i*nc;
            idx1 = idx-nc;
            idx2 = idx-nlinenc;
            matidx = idx*nc;
            matidx1 = idx1*nc;
            matidx2 = idx2*nc;
            smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
            smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nlinenc;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
    smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
    smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
    
    return;
}

/**
 * \fn static inline void str_spaAxpy_2D_nc5 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 5 banded
 *        2D structured matrix (nc = 5)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Chensheng Feng, Zheng Li
 * \date   2012/09/01
 *
 * \note the offsets of the five bands have to be (-1, +1, -nx, +nx) for nx != 1
 *       and (-1,+1,-ny,+ny) for nx = 1, but the order can be arbitrary.
 */
static inline void str_spaAxpy_2D_nc5 (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx,idx1,idx2;
    INT matidx, matidx1, matidx2;
    INT end1, end2;
    INT nline, nlinenc;
    
    // information of A
    INT nx = A->nx;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nband = A->nband;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend, up;
    INT nthreads = fasp_get_num_threads();
#endif
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL, *offdiag3=NULL;
    
    if (nx == 1) {
        nline = A->ny;
    }
    else {
        nline = nx;
    }
    nlinenc = nline*nc;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nline) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nline) {
            offdiag3 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nline;
    
    smat_amxv_nc5(alpha, diag, x, y);
    smat_amxv_nc5(alpha, offdiag1, x+nc, y);
    smat_amxv_nc5(alpha, offdiag3, x+nlinenc, y);
    
#ifdef _OPENMP
    up = nline - 1;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, i, idx, matidx, idx1, matidx1)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+1)*nc;
                matidx = idx*nc;
                idx1 = i*nc; // idx1 = idx - nc;
                matidx1 = idx1*nc;
                smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
                smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
                smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=1; i<nline; ++i) {
            idx = i*nc;
            matidx = idx*nc;
            idx1 = idx - nc;
            matidx1 = idx1*nc;
            smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    up = end2 - nx;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, idx, idx1, idx2, matidx, matidx1, matidx2)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+nx)*nc;
                idx1 = idx-nc;
                idx2 = idx-nlinenc;
                matidx = idx*nc;
                matidx1 = idx1*nc;
                matidx2 = idx2*nc;
                smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
                smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc5(alpha, diag+matidx,      x+idx,  y+idx);
                smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
                smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=nx; i<end2; ++i) {
            idx = i*nc;
            idx1 = idx-nc;
            idx2 = idx-nlinenc;
            matidx = idx*nc;
            matidx1 = idx1*nc;
            matidx2 = idx2*nc;
            smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
            smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nlinenc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    
#ifdef _OPENMP
    up = end1 - end2;
    if (up > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend, idx, idx1, idx2, matidx, matidx1, matidx2)
        for (myid=0; myid<nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, up, &mybegin, &myend);
            for (i=mybegin; i<myend; i++) {
                idx = (i+end2)*nc;
                idx1 = idx-nc;
                idx2 = idx-nlinenc;
                matidx = idx*nc;
                matidx1 = idx1*nc;
                matidx2 = idx2*nc;
                smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
                smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
                smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
                smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
            }
        }
    }
    else {
#endif
        for (i=end2; i<end1; ++i) {
            idx = i*nc;
            idx1 = idx-nc;
            idx2 = idx-nlinenc;
            matidx = idx*nc;
            matidx1 = idx1*nc;
            matidx2 = idx2*nc;
            smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
            smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
            smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
            smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        }
#ifdef _OPENMP
    }
#endif
    
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nlinenc;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
    smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
    smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
    
    return;
    
}

/**
 * \fn static inline void str_spaAxpy_2D_blk (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 5 banded
 *        2D structured matrix (nc != 1)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * \note the offsets of the five bands have to be (-1, +1, -nx, +nx) for nx != 1
 *       and (-1,+1,-ny,+ny) for nx = 1, but the order can be arbitrary.
 */
static inline void str_spaAxpy_2D_blk (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx,idx1,idx2;
    INT matidx, matidx1, matidx2;
    INT end1, end2;
    INT nline, nlinenc;
    
    // information of A
    INT nx = A->nx;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL, *offdiag3=NULL;
    
    if (nx == 1) {
        nline = A->ny;
    }
    else {
        nline = nx;
    }
    nlinenc = nline*nc;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nline) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nline) {
            offdiag3 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
        
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nline;
    
    smat_amxv(alpha, diag, x, nc, y);
    smat_amxv(alpha, offdiag1, x+nc, nc, y);
    smat_amxv(alpha, offdiag3, x+nlinenc, nc, y);
    
    for (i=1; i<nline; ++i) {
        idx = i*nc;
        matidx = idx*nc;
        idx1 = idx - nc;
        matidx1 = idx1*nc;
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nlinenc, nc, y+idx);
    }
    
    for (i=nx; i<end2; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nlinenc;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nlinenc, nc, y+idx);
    }
    
    for (i=end2; i<end1; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nlinenc;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
    }
    
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nlinenc;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
    smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
    smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
    
    return;
}

/**
 * \fn static inline void str_spaAxpy_3D_nc1 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 7 banded
 *        3D structured matrix (nc = 1)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * \note the offsetsoffsets of the five bands have to be -1, +1, -nx, +nx, -nxy
 *       and +nxy, but the order can be arbitrary.
 */
static inline void str_spaAxpy_3D_nc1 (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx1,idx2,idx3;
    INT end1, end2, end3;
    // information of A
    INT nx = A->nx;
    INT nxy = A->nxy;
    INT ngrid = A->ngrid;  // number of grids
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL,
    *offdiag3=NULL, *offdiag4=NULL, *offdiag5=NULL;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nx) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nx) {
            offdiag3 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nxy) {
            offdiag4 = A->offdiag[i];
        }
        else if (A->offsets[i] == nxy) {
            offdiag5 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 3D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nx;
    end3 = ngrid-nxy;
    
    y[0] += alpha*(diag[0]*x[0] + offdiag1[0]*x[1] + offdiag3[0]*x[nx] + offdiag5[0]*x[nxy]);
    
    for (i=1; i<nx; ++i) {
        idx1 = i-1;
        y[i] += alpha*(offdiag0[idx1]*x[idx1] + diag[i]*x[i] + offdiag1[i]*x[i+1] +
                       offdiag3[i]*x[i+nx] + offdiag5[i]*x[i+nxy]);
    }
    
    for (i=nx; i<nxy; ++i) {
        idx1 = i-1;
        idx2 = i-nx;
        y[i] += alpha*(offdiag2[idx2]*x[idx2] + offdiag0[idx1]*x[idx1]
                       + diag[i]*x[i] + offdiag1[i]*x[i+1] + offdiag3[i]*x[i+nx]
                       + offdiag5[i]*x[i+nxy]);
    }
    
    for (i=nxy; i<end3; ++i) {
        idx1 = i-1;
        idx2 = i-nx;
        idx3 = i-nxy;
        y[i] += alpha*(offdiag4[idx3]*x[idx3] + offdiag2[idx2]*x[idx2]
                       + offdiag0[idx1]*x[idx1] + diag[i]*x[i] + offdiag1[i]*x[i+1]
                       + offdiag3[i]*x[i+nx] + offdiag5[i]*x[i+nxy]);
    }
    
    for (i=end3; i<end2; ++i) {
        idx1 = i-1;
        idx2 = i-nx;
        idx3 = i-nxy;
        y[i] += alpha*(offdiag4[idx3]*x[idx3] + offdiag2[idx2]*x[idx2]
                       + offdiag0[idx1]*x[idx1] + diag[i]*x[i]
                       + offdiag1[i]*x[i+1] + offdiag3[i]*x[i+nx]);
    }
    
    for (i=end2; i<end1; ++i) {
        idx1 = i-1;
        idx2 = i-nx;
        idx3 = i-nxy;
        y[i] += alpha*(offdiag4[idx3]*x[idx3] + offdiag2[idx2]*x[idx2]
                       + offdiag0[idx1]*x[idx1] + diag[i]*x[i]
                       + offdiag1[i]*x[i+1]);
    }
    
    idx1 = end1-1;
    idx2 = end1-nx;
    idx3 = end1-nxy;
    y[end1] += alpha*(offdiag4[idx3]*x[idx3] + offdiag2[idx2]*x[idx2] +
                      offdiag0[idx1]*x[idx1] + diag[end1]*x[end1]);
    
    return;
}

/**
 * \fn static inline void str_spaAxpy_3D_nc3 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 7 banded
 *        3D structured matrix (nc = 3)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * \note the offsetsoffsets of the five bands have to be -1, +1, -nx, +nx, -nxy
 *       and +nxy, but the order can be arbitrary.
 */
static inline void str_spaAxpy_3D_nc3 (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx,idx1,idx2,idx3;
    INT matidx, matidx1, matidx2, matidx3;
    INT end1, end2, end3;
    // information of A
    INT nx = A->nx;
    INT nxy = A->nxy;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nxnc = nx*nc;
    INT nxync = nxy*nc;
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL,
    *offdiag3=NULL, *offdiag4=NULL, *offdiag5=NULL;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nx) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nx) {
            offdiag3 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nxy) {
            offdiag4 = A->offdiag[i];
        }
        else if (A->offsets[i] == nxy) {
            offdiag5 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nx;
    end3 = ngrid-nxy;
    
    smat_amxv_nc3(alpha, diag, x, y);
    smat_amxv_nc3(alpha, offdiag1, x+nc, y);
    smat_amxv_nc3(alpha, offdiag3, x+nxnc, y);
    smat_amxv_nc3(alpha, offdiag5, x+nxync, y);
    
    for (i=1; i<nx; ++i) {
        idx = i*nc;
        matidx = idx*nc;
        idx1 = idx - nc;
        matidx1 = idx1*nc;
        smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc3(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
    }
    
    for (i=nx; i<nxy; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc3(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
        
    }
    
    for (i=nxy; i<end3; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc3(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc3(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
    }
    
    for (i=end3; i<end2; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc3(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc3(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
    }
    
    for (i=end2; i<end1; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc3(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc3(alpha, offdiag1+matidx, x+idx+nc, y+idx);
    }
    
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nxnc;
    idx3 = idx-nxync;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    matidx3 = idx3*nc;
    smat_amxv_nc3(alpha, offdiag4+matidx3, x+idx3, y+idx);
    smat_amxv_nc3(alpha, offdiag2+matidx2, x+idx2, y+idx);
    smat_amxv_nc3(alpha, offdiag0+matidx1, x+idx1, y+idx);
    smat_amxv_nc3(alpha, diag+matidx, x+idx, y+idx);
    
    return;
    
}

/**
 * \fn static inline void str_spaAxpy_3D_nc5 (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 7 banded
 *        3D structured matrix (nc = 5)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * \note the offsetsoffsets of the five bands have to be -1, +1, -nx, +nx, -nxy
 *       and +nxy, but the order can be arbitrary.
 */
static inline void str_spaAxpy_3D_nc5 (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx,idx1,idx2,idx3;
    INT matidx, matidx1, matidx2, matidx3;
    INT end1, end2, end3;
    // information of A
    INT nx = A->nx;
    INT nxy = A->nxy;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nxnc = nx*nc;
    INT nxync = nxy*nc;
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL,
    *offdiag3=NULL, *offdiag4=NULL, *offdiag5=NULL;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nx) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nx) {
            offdiag3 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nxy) {
            offdiag4 = A->offdiag[i];
        }
        else if (A->offsets[i] == nxy) {
            offdiag5 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nx;
    end3 = ngrid-nxy;
    
    smat_amxv_nc5(alpha, diag, x, y);
    smat_amxv_nc5(alpha, offdiag1, x+nc, y);
    smat_amxv_nc5(alpha, offdiag3, x+nxnc, y);
    smat_amxv_nc5(alpha, offdiag5, x+nxync, y);
    
    for (i=1; i<nx; ++i) {
        idx = i*nc;
        matidx = idx*nc;
        idx1 = idx - nc;
        matidx1 = idx1*nc;
        smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc5(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
    }
    
    for (i=nx; i<nxy; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc5(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
        
    }
    
    for (i=nxy; i<end3; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc5(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
        smat_amxv_nc5(alpha, offdiag5+matidx, x+idx+nxync, y+idx);
    }
    
    for (i=end3; i<end2; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc5(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
        smat_amxv_nc5(alpha, offdiag3+matidx, x+idx+nxnc, y+idx);
    }
    
    for (i=end2; i<end1; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv_nc5(alpha, offdiag4+matidx3, x+idx3, y+idx);
        smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
        smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
        smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
        smat_amxv_nc5(alpha, offdiag1+matidx, x+idx+nc, y+idx);
    }
    
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nxnc;
    idx3 = idx-nxync;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    matidx3 = idx3*nc;
    smat_amxv_nc5(alpha, offdiag4+matidx3, x+idx3, y+idx);
    smat_amxv_nc5(alpha, offdiag2+matidx2, x+idx2, y+idx);
    smat_amxv_nc5(alpha, offdiag0+matidx1, x+idx1, y+idx);
    smat_amxv_nc5(alpha, diag+matidx, x+idx, y+idx);
    
    return;
    
}

/**
 * \fn static inline void str_spaAxpy_3D_blk (const REAL alpha, const dSTRmat *A,
 *                                            const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y, where A is a 7 banded
 *        3D structured matrix (nc != 1)
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 *
 * \note the offsetsoffsets of the five bands have to be -1, +1, -nx, +nx, -nxy
 *       and +nxy, but the order can be arbitrary.
 */
static inline void str_spaAxpy_3D_blk (const REAL      alpha,
                                       const dSTRmat  *A,
                                       const REAL     *x,
                                       REAL           *y)
{
    INT i;
    INT idx,idx1,idx2,idx3;
    INT matidx, matidx1, matidx2, matidx3;
    INT end1, end2, end3;
    // information of A
    INT nx = A->nx;
    INT nxy = A->nxy;
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;
    INT nxnc = nx*nc;
    INT nxync = nxy*nc;
    INT nband = A->nband;
    
    REAL *diag = A->diag;
    REAL *offdiag0=NULL, *offdiag1=NULL, *offdiag2=NULL,
    *offdiag3=NULL, *offdiag4=NULL, *offdiag5=NULL;
    
    for (i=0; i<nband; ++i) {
        
        if (A->offsets[i] == -1) {
            offdiag0 = A->offdiag[i];
        }
        else if (A->offsets[i] == 1) {
            offdiag1 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nx) {
            offdiag2 = A->offdiag[i];
        }
        else if (A->offsets[i] == nx) {
            offdiag3 = A->offdiag[i];
        }
        else if (A->offsets[i] == -nxy) {
            offdiag4 = A->offdiag[i];
        }
        else if (A->offsets[i] == nxy) {
            offdiag5 = A->offdiag[i];
        }
        else {
            printf("### WARNING: offsets for 2D scalar is illegal! %s\n", __FUNCTION__);
            str_spaAxpy(alpha, A, x, y);
            return;
        }
    }
    
    end1 = ngrid-1;
    end2 = ngrid-nx;
    end3 = ngrid-nxy;
    
    smat_amxv(alpha, diag, x, nc, y);
    smat_amxv(alpha, offdiag1, x+nc, nc, y);
    smat_amxv(alpha, offdiag3, x+nxnc, nc, y);
    smat_amxv(alpha, offdiag5, x+nxync, nc, y);
    
    for (i=1; i<nx; ++i) {
        idx = i*nc;
        matidx = idx*nc;
        idx1 = idx - nc;
        matidx1 = idx1*nc;
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nxnc, nc, y+idx);
        smat_amxv(alpha, offdiag5+matidx, x+idx+nxync, nc, y+idx);
    }
    
    for (i=nx; i<nxy; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nxnc, nc, y+idx);
        smat_amxv(alpha, offdiag5+matidx, x+idx+nxync, nc, y+idx);
        
    }
    
    for (i=nxy; i<end3; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv(alpha, offdiag4+matidx3, x+idx3, nc, y+idx);
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nxnc, nc, y+idx);
        smat_amxv(alpha, offdiag5+matidx, x+idx+nxync, nc, y+idx);
    }
    
    for (i=end3; i<end2; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv(alpha, offdiag4+matidx3, x+idx3, nc, y+idx);
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
        smat_amxv(alpha, offdiag3+matidx, x+idx+nxnc, nc, y+idx);
    }
    
    for (i=end2; i<end1; ++i) {
        idx = i*nc;
        idx1 = idx-nc;
        idx2 = idx-nxnc;
        idx3 = idx-nxync;
        matidx = idx*nc;
        matidx1 = idx1*nc;
        matidx2 = idx2*nc;
        matidx3 = idx3*nc;
        smat_amxv(alpha, offdiag4+matidx3, x+idx3, nc, y+idx);
        smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
        smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
        smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
        smat_amxv(alpha, offdiag1+matidx, x+idx+nc, nc, y+idx);
    }
    
    i=end1;
    idx = i*nc;
    idx1 = idx-nc;
    idx2 = idx-nxnc;
    idx3 = idx-nxync;
    matidx = idx*nc;
    matidx1 = idx1*nc;
    matidx2 = idx2*nc;
    matidx3 = idx3*nc;
    smat_amxv(alpha, offdiag4+matidx3, x+idx3, nc, y+idx);
    smat_amxv(alpha, offdiag2+matidx2, x+idx2, nc, y+idx);
    smat_amxv(alpha, offdiag0+matidx1, x+idx1, nc, y+idx);
    smat_amxv(alpha, diag+matidx, x+idx, nc, y+idx);
    
    return;
    
}

/**
 * \fn static inline void str_spaAxpy (const REAL alpha, const dSTRmat *A, 
 *                                     const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y for general cases
 *
 * \param alpha   REAL factor alpha
 * \param A       Pointer to dSTRmat matrix
 * \param x       Pointer to REAL array
 * \param y       Pointer to REAL array
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   2010/10/15
 */
static inline void str_spaAxpy (const REAL      alpha,
                                const dSTRmat  *A,
                                const REAL     *x,
                                REAL           *y)
{
    // information of A
    INT ngrid = A->ngrid;  // number of grids
    INT nc = A->nc;        // size of each block (number of components)
    INT nband = A->nband ; // number of off-diag band
    INT *offsets = A->offsets; // offsets of the off-diagals
    REAL  *diag = A->diag;       // Diagonal entries
    REAL **offdiag = A->offdiag; // Off-diagonal entries
    
    // local variables
    INT k;
    INT block = 0;
    INT point = 0;
    INT band  = 0;
    INT width = 0;
    INT size = nc*ngrid;
    INT nc2  = nc*nc;
    INT ncw  = 0;
    INT start_data = 0;
    INT start_vecx = 0;
    INT start_vecy = 0;
    INT start_vect = 0;
    REAL beta = 0.0;
    
    if (alpha == 0) {
        return; // nothing should be done
    }
    
    beta = 1.0/alpha;
    
    // y: = beta*y
    for (k = 0; k < size; ++k) {
        y[k] *= beta;
    }
    
    // y: = y + A*x
    if (nc > 1) {
        // Deal with the diagonal band
        for (block = 0; block < ngrid; ++block) {
            start_data = nc2*block;
            start_vect = nc*block;
            blkcontr_str(start_data,start_vect,start_vect,nc,diag,x,y);
        }
        
        // Deal with the off-diagonal bands
        for (band = 0; band < nband; band ++) {
            width = offsets[band];
            ncw   = nc*width;
            if (width < 0) {
                for (block = 0; block < ngrid+width; ++block) {
                    start_data = nc2*block;
                    start_vecx = nc*block;
                    start_vecy = start_vecx - ncw;
                    blkcontr_str(start_data,start_vecx,start_vecy,nc,offdiag[band],x,y);
                }
            }
            else {
                for (block = 0; block < ngrid-width; ++block) {
                    start_data = nc2*block;
                    start_vecy = nc*block;
                    start_vecx = start_vecy + ncw;
                    blkcontr_str(start_data,start_vecx,start_vecy,nc,offdiag[band],x,y);
                }
            }
        }
    }
    else if (nc == 1) {
        // Deal with the diagonal band
        for (point = 0; point < ngrid; point ++) {
            y[point] += diag[point]*x[point];
        }
        
        // Deal with the off-diagonal bands
        for (band = 0; band < nband; band ++) {
            width = offsets[band];
            if (width < 0) {
                for (point = 0; point < ngrid+width; point ++) {
                    y[point-width] += offdiag[band][point]*x[point];
                }
            }
            else {
                for (point = 0; point < ngrid-width; point ++) {
                    y[point] += offdiag[band][point]*x[point+width];
                }
            }
        }
    }
    else {
        printf("### WARNING: nc is illegal! %s\n", __FUNCTION__);
        return;
    }
    
    // y: = alpha*y
    for (k = 0; k < size; ++k) {
        y[k] *= alpha;
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
