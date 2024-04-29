/*! \file  PreAMGInterpEM.c
 *
 *  \brief Interpolation operators for AMG based on energy-min
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxThreads.c, AuxVector.c, BlaSmallMatLU.c,
 *         BlaSparseCSR.c, KryPcg.c, and PreCSR.c
 *
 *  Reference: 
 *         J. Xu and L. Zikatanov
 *         On An Energy Minimizing Basis in Algebraic Multigrid Methods,
 *         Computing and visualization in sciences, 2003
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static SHORT getiteval(dCSRmat *, dCSRmat *);
static SHORT invden(INT, REAL *, REAL *);
static SHORT get_block(dCSRmat *, INT, INT, INT *, INT *, REAL *, INT *);
static SHORT gentisquare_nomass(dCSRmat *, INT, INT *, REAL *, INT *);
static SHORT getinonefull(INT **, REAL **, INT *, INT, INT *, REAL *);
static SHORT orderone(INT **, REAL **, INT *);
static SHORT genintval(dCSRmat *, INT **, REAL **, INT, INT *, INT, INT, INT);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_amg_interp_em (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                              AMG_param *param)
 *
 * \brief Energy-min interpolation
 *
 * \param A          Pointer to dCSRmat: the coefficient matrix (index starts from 0)
 * \param vertices   Pointer to the indicator of CF splitting on fine or coarse grid
 * \param P          Pointer to the dCSRmat matrix of resulted interpolation
 * \param param      Pointer to AMG_param: AMG parameters
 *
 * \author Shuo Zhang, Xuehai Huang
 * \date   04/04/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012: add OMP support
 * Modified by Chensong Zhang on 05/14/2013: reconstruct the code
 */
void fasp_amg_interp_em (dCSRmat    *A,
                         ivector    *vertices,
                         dCSRmat    *P,
                         AMG_param  *param)
{
    INT    *vec = vertices->val;
    INT    *CoarseIndex = (INT *)fasp_mem_calloc(vertices->row, sizeof(INT));
    INT     i, j, index;
    
    // generate indices for C-points
    for ( index = i = 0; i < vertices->row; ++i ) {
        if ( vec[i] == 1 ) {
            CoarseIndex[i] = index;
            index++;
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for private(i,j) if(P->nnz>OPENMP_HOLDS)
#endif
    for ( i = 0; i < P->nnz; ++i ) {
        j        = P->JA[i];
        P->JA[i] = CoarseIndex[j];
    }
    
    // clean up memory
    fasp_mem_free(CoarseIndex); CoarseIndex = NULL;
    
    // main part
    getiteval(A, P);
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static SHORT invden (INT nn, REAL *mat, REAL *invmat)
 *
 * \brief the routine is to find the inverse of a dense matrix
 *
 * \param nn      Size of the matrix
 * \param mat     pointer to the full matrix
 * \param invmat  pointer to the full inverse matrix
 *
 * \return        FASP_SUCCESS or error message
 *
 * \note this routine works for symmetric matrix.
 *
 * \author Xuehai Huang
 * \date   04/04/2009
 */
static SHORT invden (INT    nn,
                     REAL  *mat,
                     REAL  *invmat)
{
    INT    i,j;
    SHORT  status = FASP_SUCCESS;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
    INT  *pivot=(INT *)fasp_mem_calloc(nn,sizeof(INT));
    REAL *rhs=(REAL *)fasp_mem_calloc(nn,sizeof(REAL));
    REAL *sol=(REAL *)fasp_mem_calloc(nn,sizeof(REAL));
    
    fasp_smat_lu_decomp(mat,pivot,nn);
    
#ifdef _OPENMP
#pragma omp parallel for private(myid,mybegin,myend,i,j) if(nn>OPENMP_HOLDS)
    for (myid=0; myid<nthreads; ++myid) {
        fasp_get_start_end(myid, nthreads, nn, &mybegin, &myend);
        for (i=mybegin; i<myend; ++i) {
#else
            for (i=0;i<nn;++i) {
#endif
                for (j=0;j<nn;++j) rhs[j]=0.;
                rhs[i]=1.;
                fasp_smat_lu_solve(mat,rhs,pivot,sol,nn);
                for (j=0;j<nn;++j) invmat[i*nn+j]=sol[j];
#ifdef _OPENMP
            }
        }
#else
    }
#endif
    
    fasp_mem_free(pivot); pivot = NULL;
    fasp_mem_free(rhs);   rhs   = NULL;
    fasp_mem_free(sol);   sol   = NULL;
    
    return status;
}

/**
 * \fn static SHORT get_block (dCSRmat *A, INT m, INT n, INT *rows, INT *cols, 
 *                             REAL *Aloc, INT *mask)
 *
 * \brief Get a local block from a CSR sparse matrix
 *
 * \param A     Pointer to dCSRmat matrix: the coefficient matrix
 * \param m     Number of rows of the local block matrix
 * \param n     Number of columns of the local block matrix
 * \param rows  Indices to local rows
 * \param cols  Indices to local columns
 * \param Aloc  Local dense matrix saved as an array
 * \param mask  Working array, which should be a negative number initially
 *
 * \return      FASP_SUCCESS or error message
 *
 * \author Xuehai Huang
 * \date   04/04/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012
 */
static SHORT get_block (dCSRmat  *A,
                        INT       m,
                        INT       n,
                        INT      *rows,
                        INT      *cols,
                        REAL     *Aloc,
                        INT      *mask)
{
    INT i, j, k, iloc;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
    memset(Aloc, 0x0, sizeof(REAL)*m*n);
    
#ifdef _OPENMP
#pragma omp parallel for if(n>OPENMP_HOLDS) private(j)
#endif
    for ( j=0; j<n; ++j ) {
        mask[cols[j]] = j; // initialize mask, mask stores C indices 0,1,...
    }
    
#ifdef _OPENMP
#pragma omp parallel for private(myid,mybegin,myend,i,j,k,iloc) if(m>OPENMP_HOLDS)
    for ( myid=0; myid<nthreads; ++myid ) {
        fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
        for ( i=mybegin; i<myend; ++i ) {
#else
            for ( i=0; i<m; ++i ) {
#endif
                iloc=rows[i];
                for ( k=A->IA[iloc]; k<A->IA[iloc+1]; ++k ) {
                    j = A->JA[k];
                    if (mask[j]>=0) Aloc[i*n+mask[j]]=A->val[k];
                } /* end for k */
#ifdef _OPENMP
            }
        }
#else
    } /* enf for i */
#endif
    
#ifdef _OPENMP
#pragma omp parallel for if(n>OPENMP_HOLDS) private(j)
#endif
    for ( j=0; j<n; ++j ) mask[cols[j]] = -1; // re-initialize mask
    
    return FASP_SUCCESS;
}

/**
 * \fn static SHORT gentisquare_nomass (dCSRmat *A, INT mm, INT *Ii, REAL *ima, INT *mask)
 *
 * \brief Given the row indices and col indices, find a block sub-matrix and get its inverse
 *
 * \param A     Pointer to dCSRmat matrix: the coefficient matrix
 * \param mm    Size of the sub-matrix
 * \param Ii    Integer array, to store the indices of row (also col)
 * \param ima   Pointer to the inverse of the full sub-matrix, the storage is row by row
 * \param mask  Working array
 *
 * \return      FASP_SUCCESS or error message
 *
 * \author Xuehai Huang
 * \date   04/04/2010
 */
static SHORT gentisquare_nomass (dCSRmat  *A,
                                 INT       mm,
                                 INT      *Ii,
                                 REAL     *ima,
                                 INT      *mask)
{
    SHORT status = FASP_SUCCESS;
    
    REAL *ms = (REAL *)fasp_mem_calloc(mm*mm,sizeof(REAL));
    
    get_block(A,mm,mm,Ii,Ii,ms,mask);

    status = invden(mm,ms,ima);
    
    fasp_mem_free(ms); ms = NULL;

    return status;
}

/**
 * \fn static SHORT getinonefull (INT **mat, REAL **matval, INT *lengths, INT mm, INT *Ii, REAL *ima)
 *
 * \brief Add a small sub-matrix to a big matrix with respect to its row and cols in the big matrix
 *
 * \param mat      Pointer pointing to the structure of the matrix
 * \param matval   Pointer pointing to the values according to the structure
 * \param lengths  2d array, the second entry is the lengths of matval
 * \param mm       Number of the rows (also the columns)
 * \param Ii       Pointer to the array to store the relative position of the rows and cols
 * \param ima      Pointer to the full sub-matrix, the sequence is row by row
 *
 * \return         FASP_SUCCESS or error message
 *
 * \author Xuehai Huang
 * \date   04/04/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/14/2012
 */
static SHORT getinonefull (INT   **mat,
                           REAL  **matval,
                           INT    *lengths,
                           INT     mm,
                           INT    *Ii,
                           REAL   *ima)
{
    INT tniz,i,j;
    
#ifdef _OPENMP
    // variables for OpenMP
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
    
    tniz=lengths[1];
    
#ifdef _OPENMP
#pragma omp parallel for private(myid,mybegin,myend,i,j) if(mm>OPENMP_HOLDS)
    for (myid=0; myid<nthreads; ++myid) {
        fasp_get_start_end(myid, nthreads, mm, &mybegin, &myend);
        for (i=mybegin; i<myend; ++i) {
#else
            for (i=0;i<mm;++i) {
#endif
                for (j=0;j<mm;++j) {
                    mat[0][tniz+i*mm+j]=Ii[i];
                    mat[1][tniz+i*mm+j]=Ii[j];
                    matval[0][tniz+i*mm+j]=ima[i*mm+j];
                }
#ifdef _OPENMP
            }
        }
#else
    }
#endif
    lengths[1]=tniz+mm*mm;
    
    return FASP_SUCCESS;
}

/**
 * \fn static SHORT orderone (INT **mat, REAL **matval, INT *lengths)
 *
 * \brief Order a cluster of entries in a sequence
 *
 * \param mat       Pointer to the relative position of the entries
 * \param matval    Pointer to the values corresponding to the position
 * \param lengths   Pointer to the numbers of rows, cols and nonzeros
 *
 * \return          FASP_SUCCESS or error message
 *
 * \author Xuehai Huang
 * \date   04/04/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012
 */
static SHORT orderone (INT   **mat,
                       REAL  **matval,
                       INT    *lengths)
//    lengths[0] for the number of rows
//    lengths[1] for the number of cols
//    lengths[2] for the number of nonzeros
{
    INT *rows[2],*cols[2],nns[2],tnizs[2];
    REAL *vals[2];
    SHORT status = FASP_SUCCESS;
    INT tniz,i;
    
    nns[0]=lengths[0];
    nns[1]=lengths[1];
    tnizs[0]=lengths[2];
    tniz=lengths[2];
    
    rows[0]=(INT *)fasp_mem_calloc(tniz,sizeof(INT));
    cols[0]=(INT *)fasp_mem_calloc(tniz,sizeof(INT));
    vals[0]=(REAL *)fasp_mem_calloc(tniz,sizeof(REAL));
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        rows[0][i]=mat[0][i];
        cols[0][i]=mat[1][i];
        vals[0][i]=matval[0][i];
    }
    
    rows[1]=(INT *)fasp_mem_calloc(tniz,sizeof(INT));
    cols[1]=(INT *)fasp_mem_calloc(tniz,sizeof(INT));
    vals[1]=(REAL *)fasp_mem_calloc(tniz,sizeof(REAL));
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
    // all the nonzeros with same col are gathering together
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        rows[0][i]=rows[1][i];
        cols[0][i]=cols[1][i];
        vals[0][i]=vals[1][i];
    }
    tnizs[1]=nns[0];
    nns[0]=nns[1];
    nns[1]=tnizs[1];
    tnizs[1]=tnizs[0];
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
    // all the nonzeros with same col and row are gathering together
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        rows[0][i]=rows[1][i];
        cols[0][i]=cols[1][i];
        vals[0][i]=vals[1][i];
    }
    tnizs[1]=nns[0];
    nns[0]=nns[1];
    nns[1]=tnizs[1];
    tnizs[1]=tnizs[0];
    
    tniz=tnizs[0];
    for (i=0;i<tniz-1;++i) {
        if (rows[0][i]==rows[0][i+1]&&cols[0][i]==cols[0][i+1]) {
            vals[0][i+1]+=vals[0][i];
            rows[0][i]=nns[0];
            cols[0][i]=nns[1];
        }
    }
    nns[0]=nns[0]+1;
    nns[1]=nns[1]+1;
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        rows[0][i]=rows[1][i];
        cols[0][i]=cols[1][i];
        vals[0][i]=vals[1][i];
    }
    tnizs[1]=nns[0];
    nns[0]=nns[1];
    nns[1]=tnizs[1];
    tnizs[1]=tnizs[0];
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        rows[0][i]=rows[1][i];
        cols[0][i]=cols[1][i];
        vals[0][i]=vals[1][i];
    }
    tnizs[1]=nns[0];
    nns[0]=nns[1];
    nns[1]=tnizs[1];
    tnizs[1]=tnizs[0];
    
    tniz=0;
    for (i=0;i<tnizs[0];++i)
        if (rows[0][i]<nns[0]-1) tniz++;
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) {
        mat[0][i]=rows[0][i];
        mat[1][i]=cols[0][i];
        matval[0][i]=vals[0][i];
    }
    nns[0]=nns[0]-1;
    nns[1]=nns[1]-1;
    lengths[0]=nns[0];
    lengths[1]=nns[1];
    lengths[2]=tniz;
    
    fasp_mem_free(rows[0]); rows[0] = NULL;
    fasp_mem_free(rows[1]); rows[1] = NULL;
    fasp_mem_free(cols[0]); cols[0] = NULL;
    fasp_mem_free(cols[1]); cols[1] = NULL;
    fasp_mem_free(vals[0]); vals[0] = NULL;
    fasp_mem_free(vals[1]); vals[1] = NULL;
    
    return(status);
}

/**
 * \fn static SHORT genintval (dCSRmat *A, INT **itmat, REAL **itmatval, INT ittniz,
 *                             INT *isol, INT numiso, INT nf, INT nc)
 *
 * \brief Given the structure of the interpolation, construct interpolation
 *
 * \param A         Pointer to dCSRmat matrix: the coefficient matrix
 * \param itmat     Pointer to the structure of the interpolation
 * \param itmatval  Pointer to the evaluation of the interpolation
 * \param isol      Pointer to the isolated points
 * \param numiso    Number of the isolated points
 * \param ittniz    Length of interpolation
 * \param nf        Number of fine-level nodes
 * \param nc        Number of coarse-level nodes
 *
 * \return          FASP_SUCCESS or error message
 *
 * \author Xuehai Huang
 * \date   10/29/2010
 *
 * \note
 *  nf=number fine, nc= n coarse
 *  Suppose that the structure of the interpolation is known.
 *  It is N*m matrix, N>m, recorded in itmat.
 *  We record its row index and col index, note that the same col indices gather together
 *  the itma and itmatval have a special data structure
 *  to be exact, the same columns gather together
 *  itmat[0] record the column number, and itmat[1] record the row number.
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012
 */
static SHORT genintval (dCSRmat  *A,
                        INT     **itmat,
                        REAL    **itmatval,
                        INT       ittniz,
                        INT      *isol,
                        INT       numiso,
                        INT       nf,
                        INT       nc)
{
    INT  *Ii=NULL, *mask=NULL;
    REAL *ima=NULL, *pex=NULL, **imas=NULL;
    INT **mat=NULL;
    REAL **matval;
    INT lengths[3];
    dCSRmat T;
    INT tniz;
    dvector sol, rhs;
    
    INT mm,sum,i,j,k;
    INT *iz,*izs,*izt,*izts;
    SHORT status=FASP_SUCCESS;
    
    mask=(INT *)fasp_mem_calloc(nf,sizeof(INT));
    iz=(INT *)fasp_mem_calloc(nc,sizeof(INT));
    izs=(INT *)fasp_mem_calloc(nc,sizeof(INT));
    izt=(INT *)fasp_mem_calloc(nf,sizeof(INT));
    izts=(INT *)fasp_mem_calloc(nf,sizeof(INT));
    
    fasp_iarray_set(nf, mask, -1);
    
    memset(iz, 0, sizeof(INT)*nc);
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<ittniz;++i) iz[itmat[0][i]]++;
    
    izs[0]=0;
    for (i=1;i<nc;++i) izs[i]=izs[i-1]+iz[i-1];
    
    sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) if(nc>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<nc;++i) sum+=iz[i]*iz[i];
    
    imas=(REAL **)fasp_mem_calloc(nc,sizeof(REAL *));
    
    for (i=0;i<nc;++i) {
        imas[i]=(REAL *)fasp_mem_calloc(iz[i]*iz[i],sizeof(REAL));
    }
    
    mat=(INT **)fasp_mem_calloc(2,sizeof(INT *));
    mat[0]=(INT *)fasp_mem_calloc((sum+numiso),sizeof(INT));
    mat[1]=(INT *)fasp_mem_calloc((sum+numiso),sizeof(INT));
    matval=(REAL **)fasp_mem_calloc(1,sizeof(REAL *));
    matval[0]=(REAL *)fasp_mem_calloc(sum+numiso,sizeof(REAL));
    
    lengths[1]=0;
    
    for (i=0;i<nc;++i) {
        
        mm=iz[i];
        Ii=(INT *)fasp_mem_realloc(Ii,mm*sizeof(INT));
        
#ifdef _OPENMP
#pragma omp parallel for if(mm>OPENMP_HOLDS) private(j)
#endif
        for (j=0;j<mm;++j) Ii[j]=itmat[1][izs[i]+j];
        
        ima=(REAL *)fasp_mem_realloc(ima,mm*mm*sizeof(REAL));
        
        gentisquare_nomass(A,mm,Ii,ima,mask);
        
        getinonefull(mat,matval,lengths,mm,Ii,ima);
        
#ifdef _OPENMP
#pragma omp parallel for if(mm*mm>OPENMP_HOLDS) private(j)
#endif
        for (j=0;j<mm*mm;++j) imas[i][j]=ima[j];
    }
    
#ifdef _OPENMP
#pragma omp parallel for if(numiso>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<numiso;++i) {
        mat[0][sum+i]=isol[i];
        mat[1][sum+i]=isol[i];
        matval[0][sum+i]=1.0;
    }
    
    lengths[0]=nf;
    lengths[2]=lengths[1]+numiso;
    lengths[1]=nf;
    orderone(mat,matval,lengths);
    tniz=lengths[2];
    
    sol.row=nf;
    sol.val=(REAL*)fasp_mem_calloc(nf,sizeof(REAL));
    
    memset(izt, 0, sizeof(INT)*nf);
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<tniz;++i) izt[mat[0][i]]++;
    
    T.IA=(INT*)fasp_mem_calloc((nf+1),sizeof(INT));
    
    T.row=nf;
    T.col=nf;
    T.nnz=tniz;
    T.IA[0]=0;
    for (i=1;i<nf+1;++i) T.IA[i]=T.IA[i-1]+izt[i-1];
    
    T.JA=(INT*)fasp_mem_calloc(tniz,sizeof(INT));
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(j)
#endif
    for (j=0;j<tniz;++j) T.JA[j]=mat[1][j];
    
    T.val=(REAL*)fasp_mem_calloc(tniz,sizeof(REAL));
    
#ifdef _OPENMP
#pragma omp parallel for if(tniz>OPENMP_HOLDS) private(j)
#endif
    for (j=0;j<tniz;++j) T.val[j]=matval[0][j];
    
    rhs.val=(REAL*)fasp_mem_calloc(nf,sizeof(REAL));
    
#ifdef _OPENMP
#pragma omp parallel for if(nf>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<nf;++i) rhs.val[i]=1.0;
    rhs.row=nf;
    
    // setup preconditioner
    dvector diag; fasp_dcsr_getdiag(0,&T,&diag);
    
    precond pc;
    pc.data = &diag;
    pc.fct  = fasp_precond_diag;
    
    status = fasp_solver_dcsr_pcg(&T,&rhs,&sol,&pc,1e-3,1e-15,100,STOP_REL_RES,PRINT_NONE);

    for (i=0;i<nc;++i) {
        mm=iz[i];
        
        ima=(REAL *)fasp_mem_realloc(ima,mm*mm*sizeof(REAL));
        
        pex=(REAL *)fasp_mem_realloc(pex,mm*sizeof(REAL));
        
        Ii=(INT *)fasp_mem_realloc(Ii,mm*sizeof(INT));
        
#ifdef _OPENMP
#pragma omp parallel for if(mm>OPENMP_HOLDS) private(j)
#endif
        for (j=0;j<mm;++j) Ii[j]=itmat[1][izs[i]+j];
        
#ifdef _OPENMP
#pragma omp parallel for if(mm*mm>OPENMP_HOLDS) private(j)
#endif
        for (j=0;j<mm*mm;++j) ima[j]=imas[i][j];
        
#ifdef _OPENMP
#pragma omp parallel for if(mm>OPENMP_HOLDS) private(k,j)
#endif
        for (k=0;k<mm;++k) {
            for (pex[k]=j=0;j<mm;++j) pex[k]+=ima[k*mm+j]*sol.val[Ii[j]];
        }
#ifdef _OPENMP
#pragma omp parallel for if(mm>OPENMP_HOLDS) private(j)
#endif
        for (j=0;j<mm;++j) itmatval[0][izs[i]+j]=pex[j];
        
    }
    
    fasp_mem_free(ima); ima = NULL;
    fasp_mem_free(pex); pex = NULL;
    fasp_mem_free(Ii); Ii = NULL;
    fasp_mem_free(mask); mask = NULL;
    fasp_mem_free(iz); iz = NULL;
    fasp_mem_free(izs); izs = NULL;
    fasp_mem_free(izt); izt = NULL;
    fasp_mem_free(izts); izts = NULL;
    fasp_mem_free(mat[0]); mat[0] = NULL;
    fasp_mem_free(mat[1]); mat[1] = NULL;
    fasp_mem_free(mat); mat = NULL;
    fasp_mem_free(matval[0]); matval[0] = NULL;
    fasp_mem_free(matval); matval = NULL;
    for ( i=0; i<nc; ++i ) {fasp_mem_free(imas[i]); imas[i] = NULL;}
    fasp_mem_free(imas); imas = NULL;

    fasp_dcsr_free(&T);
    fasp_dvec_free(&rhs);
    fasp_dvec_free(&sol);
    fasp_dvec_free(&diag);
    
    return status;
}

/**
 * \fn static SHORT getiteval (dCSRmat *A, dCSRmat *it)
 *
 * \brief Given a coarsening (in the form of an interpolation operator), inherit the 
 *        structure, get new evaluation
 *
 * \param A    Pointer to dCSRmat matrix: the coefficient matrix
 * \param it   Pointer to dCSRmat matrix: the interpolation matrix
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   10/29/2010
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012
 */
static SHORT getiteval (dCSRmat  *A,
                        dCSRmat  *it)
{
    INT nf,nc,ittniz;
    INT *itmat[2];
    REAL **itmatval;
    INT *rows[2],*cols[2];
    REAL *vals[2];
    INT nns[2],tnizs[2];
    INT i,j,numiso;
    INT *isol;
    SHORT status = FASP_SUCCESS;
    
    nf=A->row;
    nc=it->col;
    ittniz=it->IA[nf];
    
    itmat[0]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    itmat[1]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    itmatval=(REAL **)fasp_mem_calloc(1,sizeof(REAL *));
    itmatval[0]=(REAL *)fasp_mem_calloc(ittniz,sizeof(REAL));
    isol=(INT *)fasp_mem_calloc(nf,sizeof(INT));
    
    numiso=0;
    for (i=0;i<nf;++i) {
        if (it->IA[i]==it->IA[i+1]) {
            isol[numiso]=i;
            numiso++;
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for if(nf>OPENMP_HOLDS) private(i,j)
#endif
    for (i=0;i<nf;++i) {
        for (j=it->IA[i];j<it->IA[i+1];++j) itmat[0][j]=i;
    }
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(j)
#endif
    for (j=0;j<ittniz;++j) {
        itmat[1][j]=it->JA[j];
        itmatval[0][j]=it->val[j];
    }
    
    rows[0]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    cols[0]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    vals[0]=(REAL *)fasp_mem_calloc(ittniz,sizeof(REAL));
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<ittniz;++i) {
        rows[0][i]=itmat[0][i];
        cols[0][i]=itmat[1][i];
        vals[0][i]=itmat[0][i];
    }
    
    nns[0]=nf;
    nns[1]=nc;
    tnizs[0]=ittniz;
    
    rows[1]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    cols[1]=(INT *)fasp_mem_calloc(ittniz,sizeof(INT));
    vals[1]=(REAL *)fasp_mem_calloc(ittniz,sizeof(REAL));
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<ittniz;++i) {
        itmat[0][i]=rows[1][i];
        itmat[1][i]=cols[1][i];
        itmatval[0][i]=vals[1][i];
    }
    genintval(A,itmat,itmatval,ittniz,isol,numiso,nf,nc);
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<ittniz;++i) {
        rows[0][i]=itmat[0][i];
        cols[0][i]=itmat[1][i];
        vals[0][i]=itmatval[0][i];
    }
    nns[0]=nc;
    nns[1]=nf;
    tnizs[0]=ittniz;
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
#ifdef _OPENMP
#pragma omp parallel for if(ittniz>OPENMP_HOLDS) private(i)
#endif
    for (i=0;i<ittniz;++i) it->val[i]=vals[1][i];
    
    fasp_mem_free(isol); isol = NULL;
    fasp_mem_free(itmat[0]); itmat[0] = NULL;
    fasp_mem_free(itmat[1]); itmat[1] = NULL;
    fasp_mem_free(itmatval[0]); itmatval[0] = NULL;
    fasp_mem_free(itmatval); itmatval = NULL;
    fasp_mem_free(rows[0]); rows[0] = NULL;
    fasp_mem_free(rows[1]); rows[1] = NULL;
    fasp_mem_free(cols[0]); cols[0] = NULL;
    fasp_mem_free(cols[1]); cols[1] = NULL;
    fasp_mem_free(vals[0]); vals[0] = NULL;
    fasp_mem_free(vals[1]); vals[1] = NULL;
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
