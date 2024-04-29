/*! \file  BlaSparseUtil.c
 *
 *  \brief Routines for sparse matrix operations
 *
 *  \note  Most algorithms work as follows:
 *         (a) Boolean operations (to determine the nonzero structure);
 *         (b) Numerical part, where the result is calculated.
 *
 *  \note  Parameter notation
 *         :I: is input; :O: is output; :IO: is both
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/
/**
 * \fn void fasp_sparse_abybms_ (INT *ia, INT *ja, INT *ib, INT *jb, INT *nap,
 *                               INT *map, INT *mbp, INT *ic, INT *jc)
 *
 * \brief Multiplication of two sparse matrices: calculating the nonzero
 *        structure of the result if jc is not null. If jc is null only
 *        finds num of nonzeroes.
 *
 * \param ia   array of row pointers 1st multiplicand
 * \param ja   array of column indices 1st multiplicand
 * \param ib   array of row pointers 2nd multiplicand
 * \param jb   array of column indices 2nd multiplicand
 * \param nap  number of rows of A
 * \param map  number of cols of A
 * \param mbp  number of cols of b
 * \param ic   array of row pointers in the result (this is also computed here again,
 *               so that we can have a stand alone call of this routine, if for some
 *               reason the number of nonzeros in the result is known)
 * \param jc   array of column indices in the result c=a*b
 *
 * Modified by Chensong Zhang on 09/11/2012
 */
void fasp_sparse_abybms_ (INT *ia,
                          INT *ja,
                          INT *ib,
                          INT *jb,
                          INT *nap,
                          INT *map,
                          INT *mbp,
                          INT *ic,
                          INT *jc)
{
    /*  FORM ic when jc is null and both when jc is not null for
        the ic and jc are for c=a*b, a and b sparse  */
    /*  na = number of rows of a    */
    /*  mb = number of columns of b */
    unsigned int jcform=0;
    INT na,mb,icpp,iastrt,ibstrt,iaend,ibend,i,j,k,jia,jib;
    INT *icp;
    if (jc) jcform=1;
    na=*nap;
    mb=*mbp;
    icpp = 1;
    icp=(INT *) calloc(mb,sizeof(INT));

    for (i = 0; i < mb; ++i) icp[i] = 0;

    for (i = 0; i < na; ++i) {
        ic[i] = icpp;
        iastrt = ia[i]-1;
        iaend = ia[i+1]-1;
        if (iaend > iastrt) {
            for (jia = iastrt; jia < iaend; ++jia) {
                j = ja[jia]-1;
                ibstrt = ib[j]-1;
                ibend = ib[j+1]-1;
                if (ibend > ibstrt) {
                    for (jib = ibstrt; jib< ibend; ++jib) {
                        k = jb[jib]-1;
                        if (icp[k] != i+1) {
                            if (jcform) jc[icpp-1] = k+1;
                            ++icpp;
                            icp[k] = i+1;
                        } //if
                    } //for
                } //if
            } //for
        } //if
    } //for (i...
    ic[na] = icpp;

    if (icp) free(icp);

    return;
}

/**
 * \fn void fasp_sparse_abyb_ (INT *ia,INT *ja, REAL *a, INT *ib, INT *jb, REAL *b,
 *                             INT *nap, INT *map, INT *mbp, INT *ic, INT *jc, REAL *c)
 *
 * \brief Multiplication of two sparse matrices
 *
 * \param ia   array of row pointers 1st multiplicand
 * \param ja   array of column indices 1st multiplicand
 * \param a    entries of the 1st multiplicand
 * \param ib   array of row pointers 2nd multiplicand
 * \param jb   array of column indices 2nd multiplicand
 * \param b    entries of the 2nd multiplicand
 * \param ic   array of row pointers in c=a*b
 * \param jc   array of column indices in c=a*b
 * \param c    entries of the result: c= a*b
 * \param nap  number of rows in the 1st multiplicand
 * \param map  number of columns in the 1st multiplicand
 * \param mbp  number of columns in the 2nd multiplicand
 *
 * Modified by Chensong Zhang on 09/11/2012
 */
void fasp_sparse_abyb_ (INT  *ia,
                        INT  *ja,
                        REAL *a,
                        INT  *ib,
                        INT  *jb,
                        REAL *b,
                        INT  *nap,
                        INT  *map,
                        INT  *mbp,
                        INT  *ic,
                        INT  *jc,
                        REAL *c)
{
    INT na,mb,iastrt,ibstrt,iaend,ibend,icstrt,icend,i,j,k,ji,jia,jib;
    REAL *x;
    REAL x0;
    /*
     C--------------------------------------------------------------------
     C...  C = A*B
     C--------------------------------------------------------------------
     */
    na=*nap;
    mb=*mbp;
    x=(REAL *)calloc(mb,sizeof(REAL));
    for (i = 0; i < na; ++i) {
        icstrt = ic[i]-1;
        icend = ic[i+1]-1;
        if (icend > icstrt) {
            for (ji = icstrt;ji < icend;++ji) {
                k=jc[ji]-1;
                x[k] = 0e+0;
            }
            iastrt = ia[i]-1;
            iaend = ia[i+1]-1;
            if (iaend > iastrt) {
                for (jia = iastrt; jia < iaend ; ++jia) {
                    j = ja[jia]-1;
                    x0 = a[jia];
                    ibstrt = ib[j]-1;
                    ibend = ib[j+1]-1;
                    if (ibend > ibstrt) {
                        for (jib = ibstrt; jib < ibend; ++jib) {
                            k = jb[jib]-1;
                            x[k] += x0*b[jib];
                        }
                    }  // end if
                } //  end for
            }
            for (ji = icstrt; ji < icend; ++ji) {
                k=jc[ji]-1;
                c[ji]=x[k];
            } // end for
        } // end if
    }//end do
    if (x) free(x);
    return;
}

/**
 * \fn void fasp_sparse_iit_ (INT *ia, INT *ja, INT *na, INT *ma, INT *iat, INT *jat)
 *
 * \brief Transpose a boolean matrix (only given by ia, ja)
 *
 * \param ia   array of row pointers (as usual in CSR)
 * \param ja   array of column indices
 * \param na   number of rows
 * \param ma   number of cols
 * \param iat  array of row pointers in the result
 * \param jat  array of column indices
 */
void fasp_sparse_iit_ (INT *ia,
                       INT *ja,
                       INT *na,
                       INT *ma,
                       INT *iat,
                       INT *jat)
{
    /*C====================================================================*/
    INT i,j,jp,n,m,mh,nh,iaa,iab,k;
    /*
     C--------------------------------------------------------------------
     C...  Transposition of a graph (or the matrix) symbolically.
     C...
     C...  Input:
     C...    IA, JA   - given graph (or matrix).
     C...    N        - number of rows of the matrix.
     C...    M        - number of columns of the matrix.
     C...
     C...  Output:
     C...    IAT, JAT - transposed graph (or matrix).
     C...
     C...  Note:
     C...    N+1 is the dimension of IA.
     C...    M+1 is the dimension of IAT.
     C--------------------------------------------------------------------
     */
    n=*na;
    m=*ma;
    mh = m + 1;
    nh = n + 1;
    for (i = 1; i < mh; ++i) {
        iat[i] = 0;
    }
    iab = ia[nh-1] - 1;
    for (i = 1; i <= iab; ++i) {
        j = ja[i-1] + 2;
        if (j <= mh)
            iat[j-1] = iat[j-1] + 1;
    }
    iat[0] = 1;
    iat[1] = 1;
    if (m != 1) {
        for (i= 2; i< mh; ++i) {
            iat[i] = iat[i] + iat[i-1];
        }
    }
    for (i = 1; i <= n; ++i) {
        iaa = ia[i-1];
        iab = ia[i] - 1;
        if (iab >= iaa) {
            for (jp = iaa; jp <= iab; ++jp) {
                j = ja[jp-1] + 1;
                k = iat[j-1];
                jat[k-1] = i;
                iat[j-1] = k + 1;
            }
        }
    }
    return;
}

/**
 * \fn void fasp_sparse_aat_ (INT *ia, INT *ja, REAL *a, INT *na, INT *ma, INT *iat,
 *                            INT *jat, REAL *at)
 *
 * \brief Transpose a boolean matrix (only given by ia, ja)
 *
 * \param ia   array of row pointers (as usual in CSR)
 * \param ja   array of column indices
 * \param a    array of entries of teh input
 * \param na   number of rows of A
 * \param ma   number of cols of A
 * \param iat  array of row pointers in the result
 * \param jat  array of column indices
 * \param at   array of entries of the result
 */
void fasp_sparse_aat_ (INT  *ia,
                       INT  *ja,
                       REAL *a,
                       INT  *na,
                       INT  *ma,
                       INT  *iat,
                       INT  *jat,
                       REAL *at)
{
    /*C====================================================================*/
    INT i,j,jp,n,m,mh,nh,iaa,iab,k;
    /*
     C--------------------------------------------------------------------
     C...  Transposition of a matrix.
     C...
     C...  Input:
     C...    IA, JA   - given graph (or matrix).
     C...    N        - number of rows of the matrix.
     C...    M        - number of columns of the matrix.
     C...
     C...  Output:
     C...    IAT, JAT, AT - transposed matrix
     C...
     C...  Note:
     C...    N+1 is the dimension of IA.
     C...    M+1 is the dimension of IAT.
     C--------------------------------------------------------------------
     */
    n=*na;
    m=*ma;
    mh = m + 1;
    nh = n + 1;
    
    for (i = 1; i < mh; ++i) {
        iat[i] = 0;
    }
    iab = ia[nh-1] - 1; /* Size of ja */
    for (i = 1;i<=iab; ++i) {
        j = ja[i-1] + 2;
        if (j <= mh) {
            iat[j-1] = iat[j-1] + 1;
        }
    }
    iat[0] = 1;
    iat[1] = 1;
    if (m != 1) {
        for (i= 2; i< mh; ++i) {
            iat[i] = iat[i] + iat[i-1];
        }
    }
    
    for (i=1; i<=n; ++i) {
        iaa = ia[i-1];
        iab = ia[i] - 1;
        if (iab >= iaa) {
            for (jp = iaa; jp <= iab; ++jp) {
                j = ja[jp-1] + 1;
                k = iat[j-1];
                jat[k-1] = i;
                at[k-1] = a[jp-1];
                iat[j-1] = k + 1;
            }
        }
    }
    
    return;
}

/**
 * \fn void void fasp_sparse_aplbms_ (INT *ia, INT *ja, INT *ib, INT *jb,
 *                                    INT *nab, INT *mab, INT *ic, INT *jc)
 *
 * \brief Addition of two sparse matrices: calculating the nonzero structure of the
 *        result if jc is not null. if jc is null only finds num of nonzeroes.
 *
 * \param ia   array of row pointers 1st summand
 * \param ja   array of column indices 1st summand
 * \param ib   array of row pointers 2nd summand
 * \param jb   array of column indices 2nd summand
 * \param nab  number of rows
 * \param mab  number of cols
 * \param ic   array of row pointers in the result (this is also computed here again,
 *               so that we can have a stand alone call of this routine, if for some
 *               reason the number of nonzeros in the result is known)
 * \param jc   array of column indices in the result c=a+b
 */
void fasp_sparse_aplbms_ (INT *ia,
                          INT *ja,
                          INT *ib,
                          INT *jb,
                          INT *nab,
                          INT *mab,
                          INT *ic,
                          INT *jc)
{
    unsigned int jcform=0;
    INT icpp,i1,i,j,jp,n,m,iastrt,iaend,ibstrt,ibend;
    INT *icp;
    /*
     c...  addition of two general sparse matricies (symbolic part) :
     c= a + b.
     */
    if (jc) jcform=1;
    n=*nab;
    m=*mab;
    icp=(INT *) calloc(m,sizeof(INT));
    for (i=0; i< m; ++i) icp[i] = 0;
    icpp = 1;
    for (i=0; i< n; ++i) {
        ic[i] = icpp;
        i1=i+1;
        iastrt = ia[i]-1;
        iaend = ia[i1]-1;
        if (iaend > iastrt) {
            for (jp = iastrt; jp < iaend; ++jp) {
                j = ja[jp];
                if (jcform) jc[icpp-1] = j;
                ++icpp;
                icp[j-1] = i1;
            }
        }
        ibstrt = ib[i] - 1;
        ibend = ib[i1] - 1;
        if (ibend > ibstrt) {
            for (jp = ibstrt; jp < ibend; ++jp) {
                j = jb[jp];
                if (icp[j-1] != i1) {
                    if (jcform) jc[icpp-1] = j;
                    ++icpp;
                }
            }
        }
    }  // // loop i=0; i< n
    ic[n] = icpp;
    if (icp) free(icp);
    return;
}

/**
 * \fn void fasp_sparse_aplusb_ (INT *ia, INT *ja, REAL *a,
 *                               INT *ib, INT *jb, REAL *b,
 *                               INT *nab, INT *mab,
 *                               INT *ic, INT *jc, REAL *c)
 *
 * \brief Addition of two sparse matrices
 *
 * \param ia   array of row pointers 1st summand
 * \param ja   array of column indices 1st summand
 * \param a    entries of the 1st summand
 * \param ib   array of row pointers 2nd summand
 * \param jb   array of column indices 2nd summand
 * \param b    entries of the 2nd summand
 * \param nab  number of rows
 * \param mab  number of cols
 * \param ic   array of row pointers in c=a+b
 * \param jc   array of column indices in c=a+b
 * \param c    entries of the result: c=a+b
 */
void fasp_sparse_aplusb_ (INT  *ia,
                          INT  *ja,
                          REAL *a,
                          INT  *ib,
                          INT  *jb,
                          REAL *b,
                          INT  *nab,
                          INT  *mab,
                          INT  *ic,
                          INT  *jc,
                          REAL *c)
{
    INT n,m,icpp,i1,i,j,iastrt,iaend,ibstrt,ibend,icstrt,icend;
    REAL *x;
    /*
     c...  addition of two general sparse matricies (numerical part) :
     c= a + b
     */
    n=*nab;
    m=*mab;
    x=(REAL *)calloc(m,sizeof(REAL));
    for (i=0;i<n;++i) {
        i1=i+1;
        icstrt = ic[i]-1;
        icend = ic[i1]-1;
        if (icend > icstrt) {
            for (icpp = icstrt;icpp<icend;++icpp) {
                j=jc[icpp]-1;
                x[j] = 0e+00;
            }
            iastrt = ia[i]-1;
            iaend = ia[i1]-1;
            if (iaend > iastrt) {
                for (icpp = iastrt;icpp<iaend;++icpp) {
                    j=ja[icpp]-1;
                    x[j] = a[icpp];
                }
            }
            ibstrt = ib[i]-1;
            ibend = ib[i1]-1;
            if (ibend > ibstrt) {
                for (icpp = ibstrt;icpp<ibend;++icpp) {
                    j = jb[icpp]-1;
                    x[j] = x[j] + b[icpp];
                }
            }
            for (icpp = icstrt;icpp<icend;++icpp) {
                j=jc[icpp]-1;
                c[icpp] = x[j];
            }
        } // if (icstrt > icend)...
    } // loop i=0; i< n
    if (x) free(x);
    return;
}

/**
 * \fn void fasp_sparse_rapms_(INT *ir, INT *jr, INT *ia, INT *ja, INT *ip, INT *jp,
 *                             INT *nin, INT *ncin, INT *iac, INT *jac, INT *maxrout)
 *
 * \brief Calculates the nonzero structure of R*A*P, if jac is not null.
 *        If jac is null only finds num of nonzeroes.
 *
 * \note  :I: is input :O: is output :IO: is both
 *
 * \param ir :I: array of row pointers for R
 * \param jr :I: array of column indices for R
 * \param ia :I: array of row pointers for A
 * \param ja :I: array of column indices for A
 * \param ip :I: array of row pointers for P
 * \param jp :I: array of column indices for P
 * \param nin :I: number of rows in R
 * \param ncin :I: number of columns in R
 * \param iac :O: array of row pointers for Ac
 * \param jac :O: array of column indices for Ac
 * \param maxrout :O: the maximum nonzeroes per row for R
 *
 * \note Computes the sparsity pattern of R*A*P.  maxrout is output and is
 *         the maximum nonzeroes per row for r.  On output we also have is
 *         iac (if jac is null) and jac (if jac entry is not null).
 *         R is (nc,n) A is (n,n) and P is (n,nc)!
 *
 * Modified by Chensong Zhang on 09/11/2012
 */
void fasp_sparse_rapms_ (INT *ir,
                         INT *jr,
                         INT *ia,
                         INT *ja,
                         INT *ip,
                         INT *jp,
                         INT *nin,
                         INT *ncin,
                         INT *iac,
                         INT *jac,
                         INT *maxrout)
{
    INT i,jk,jak,jpk,ic,jc,nc,icp1,ira,irb,ipa,ipb;
    INT maxri,maxr,iaa,iab,iacp,if1,jf1,jacform=0;
    INT *ix;
    
    nc = *ncin;
    ix=(INT *) calloc(nc,sizeof(INT));
    if (jac) jacform=1;
    maxr = 0;
    for (i =0;i<nc; ++i) {
        ix[i]=0;
        ira=ir[i];
        irb=ir[i+1];
        maxri=irb-ira;
        if (maxr < maxri) maxr=maxri;
    }
    iac[0] = 1;
    iacp = iac[0]-1;
    for (ic = 0;ic<nc;ic++) {
        ira=ir[ic]-1;
        icp1=ic+1;
        irb=ir[icp1]-1;
        for (jk = ira;jk<irb;jk++) {
            if1 = jr[jk]-1;
            iaa = ia[if1]-1;
            iab = ia[if1+1]-1;
            for (jak = iaa;jak < iab;jak++) {
                jf1 = ja[jak]-1;
                ipa = ip[jf1]-1;
                ipb = ip[jf1+1]-1;
                for (jpk = ipa;jpk < ipb;jpk++) {
                    jc = jp[jpk]-1;
                    if (ix[jc] != icp1) {
                        ix[jc]=icp1;
                        if (jacform) jac[iacp] = jc+1;
                        iacp++;
                    }
                }
            }
        }
        iac[icp1] = iacp+1;
    }
    *maxrout=maxr;
    if (ix) free(ix);
    return;
}

/**
 * \fn void fasp_sparse_wtams_(INT *jw, INT *ia, INT *ja, INT *nwp,INT *map,
 *                               INT *jv, INT *nvp, INT *icp)
 *
 * \brief Finds the nonzeroes in the result of v^t = w^t A, where w
 *        is a sparse vector and A is sparse matrix. jv is an integer
 *        array containing the indices of the nonzero elements in the
 *        result.
 *
 * :I: is input :O: is output :IO: is both
 * \param jw :I: indices such that w[jw] is nonzero
 * \param ia :I: array of row pointers for A
 * \param ja :I: array of column indices for A
 * \param nwp :I: number of nonzeroes in w (the length of w)
 * \param map :I: number of columns in A
 * \param jv :O: indices such that v[jv] is nonzero
 * \param nvp :I: number of nonzeroes in v
 * \param icp :IO: is a working array of length (*map) which on
 *                   output satisfies icp[jv[k]-1]=k; Values of icp[] at
 *                   positions * other than (jv[k]-1) remain unchanged.
 *
 * Modified by Chensong Zhang on 09/11/2012
 */
void fasp_sparse_wtams_ (INT *jw,
                         INT *ia,
                         INT *ja,
                         INT *nwp,
                         INT *map,
                         INT *jv,
                         INT *nvp,
                         INT *icp)
{
    INT nw,nv,iastrt,iaend,j,k,jiw,jia;
    if (*nwp<=0) {*nvp=0; return;}
    nw=*nwp;
    nv = 0;
    for (jiw = 0;jiw < nw; ++jiw) {
        j = jw[jiw]-1;
        iastrt = ia[j]-1;
        iaend  = ia[j+1]-1;
        if (iaend > iastrt) {
            for (jia = iastrt ;jia< iaend;jia++) {
                k = ja[jia]-1;
                if (!icp[k]) {
                    jv[nv] = k+1;
                    nv++;
                    icp[k] = nv;
                }
            }
        }
    }
    *nvp=nv;
    return;
}

/**
 * \fn void fasp_sparse_wta_(INT *jw, REAL *w, INT *ia, INT *ja, REAL *a,
 *                           INT *nwp, INT *map, INT *jv, REAL *v, INT *nvp)
 *
 * \brief Calculate v^t = w^t A, where w is a sparse vector and A is sparse matrix.
 *        v is an array of dimension = number of columns in A.
 *
 * \note  :I: is input :O: is output :IO: is both
 *
 * \param jw :I: indices such that w[jw] is nonzero
 * \param w :I: the values of w
 * \param ia :I: array of row pointers for A
 * \param ja :I: array of column indices for A
 * \param a :I: entries of A
 * \param nwp :I: number of nonzeroes in w (the length of w)
 * \param map :I: number of columns in A
 * \param jv :O: indices such that v[jv] is nonzero
 * \param v :O: the result v^t=w^t A
 * \param nvp :I: number of nonzeroes in v
 */
void fasp_sparse_wta_ (INT  *jw,
                       REAL *w,
                       INT  *ia,
                       INT  *ja,
                       REAL *a,
                       INT  *nwp,
                       INT  *map,
                       INT  *jv,
                       REAL *v,
                       INT  *nvp)
{
    INT nw,nv,iastrt,iaend,j,k,ji,jiw,jia;
    REAL v0;
    
    if (*nwp<=0) {*nvp=-1; return;}
    nw=*nwp;
    nv=*nvp;
    for (ji = 0;ji < nv;++ji) {
        k=jv[ji]-1;
        v[k] = 0e+0;
    }
    for (jiw = 0;jiw<nw; ++jiw) {
        j = jw[jiw]-1;
        v0 = w[jiw];
        iastrt = ia[j]-1;
        iaend  = ia[j+1]-1;
        if (iaend > iastrt) {
            for (jia = iastrt;jia < iaend;jia++) {
                k = ja[jia]-1;
                v[k] += v0*a[jia];
            }
        }  // end if
    } //  end for
    return;
}

/**
 * \fn void fasp_sparse_ytxbig_ (INT *jy, REAL *y, INT *nyp, REAL *x, REAL *s)
 *
 * \brief Calculates s = y^t x. y-sparse, x - no
 *
 * \note  :I: is input :O: is output :IO: is both
 *
 * \param jy :I: indices such that y[jy] is nonzero
 * \param y :I: is a sparse vector
 * \param nyp :I: number of nonzeroes in v
 * \param x :I: also a vector assumed to have entry for any j=jy[i]-1;
 *              for i=1:nyp. This means that x here does not have to be
 *              sparse
 * \param s :O: s = y^t x
 */
void fasp_sparse_ytxbig_ (INT  *jy,
                          REAL *y,
                          INT  *nyp,
                          REAL *x,
                          REAL *s)
{
    INT i,ii;
    *s=0e+00;
    if (*nyp > 0) {
        for (i = 0;i< *nyp; ++i) {
            ii = jy[i]-1;
            *s += y[i]*x[ii];
        }
    }
    return;
}

/**
 * \fn void fasp_sparse_ytx_(INT *jy, REAL *y, INT *jx, REAL *x,
 *                           INT *nyp, INT *nxp, INT *icp, REAL *s)
 *
 * \brief Calculates s = y^t x. y is sparse, x is sparse
 *
 * \note  :I: is input :O: is output :IO: is both
 *
 * \param jy :I: indices such that y[jy] is nonzero
 * \param y :I: is a sparse vector.
 * \param nyp :I: number of nonzeroes in y
 * \param jx :I: indices such that x[jx] is nonzero
 * \param x :I: is a sparse vector.
 * \param nxp :I: number of nonzeroes in x
 * \param icp ???
 * \param s :O: s = y^t x.
 */
void fasp_sparse_ytx_ (INT  *jy,
                       REAL *y,
                       INT  *jx,
                       REAL *x,
                       INT  *nyp,
                       INT  *nxp,
                       INT  *icp,
                       REAL *s)
{// not tested
    INT i,j,i0,ii;
    *s=0e+00;
    if ((*nyp > 0) && (*nxp > 0)) {
        for (i = 0;i< *nyp; ++i) {
            j = jy[i]-1;
            i0=icp[j];
            if (i0) {
                ii=jx[i0]-1;
                *s += y[i]*x[ii];
            }
        }
    }
    return;
}

/**
 * \fn void fasp_sparse_rapcmp_ (INT *ir, INT *jr, REAL *r, INT *ia, INT *ja,
 *                               REAL *a, INT *ipt, INT *jpt, REAL *pt, INT *nin,
 *                               INT *ncin, INT *iac,INT *jac, REAL *ac, INT *idummy)
 *
 * \brief Calculates R*A*P after the nonzero structure
 *        of the result is known. iac,jac,ac have to be
 *        allocated before call to this function.
 *
 * \note :I: is input :O: is output :IO: is both
 *
 * \param ir :I: array of row pointers for R
 * \param jr :I: array of column indices for R
 * \param r :I: entries of R
 * \param ia :I: array of row pointers for A
 * \param ja :I: array of column indices for A
 * \param a :I: entries of A
 * \param ipt :I: array of row pointers for P
 * \param jpt :I: array of column indices for P
 * \param pt :I: entries of P
 * \param nin :I: number of rows in R
 * \param ncin :I: number of rows in
 * \param iac :O: array of row pointers for P
 * \param jac :O: array of column indices for P
 * \param ac :O: entries of P
 * \param idummy not changed
 *
 * \note  Compute R*A*P for known nonzero structure of the result
 *        the result is stored in iac,jac,ac!
 */
void fasp_sparse_rapcmp_ (INT  *ir,
                          INT  *jr,
                          REAL *r,
                          INT  *ia,
                          INT  *ja,
                          REAL *a,
                          INT  *ipt,
                          INT  *jpt,
                          REAL *pt,
                          INT  *nin,
                          INT  *ncin,
                          INT  *iac,
                          INT  *jac,
                          REAL *ac,
                          INT  *idummy)
{
    INT i,j,k,n,nc,nv,nw,nptjc,iacst,iacen,ic,jc,is,js,jkc,iastrt,iaend,ji,jia;
    REAL aij,v0;
    INT *icp=NULL, *jv=NULL,*jris=NULL, *jptjs=NULL;
    REAL *v=NULL, *ris=NULL, *ptjs=NULL;
    n=*nin;
    nc=*ncin;
    
    v  = (REAL *) calloc(n,sizeof(REAL));
    icp = (INT *) calloc(n,sizeof(INT));
    jv  = (INT *) calloc(n,sizeof(INT));
    if (!(icp && v && jv)) {
        fprintf(stderr,"### ERROR: Could not allocate memory!\n");
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }
    for (i=0;i<n;++i) {
        icp[i] = 0;
        jv[i] = 0;
        v[i]=0e+00;
    }
    for (ic = 0;ic<nc;ic++) {
        nw = ir[ic+1]-ir[ic];
        if (nw<=0) continue;
        is = ir[ic]-1;
        jris=jr+is;
        //    wtams_(jris,ia, ja, &nw,&n,jv, &nv, icp);
        //    void wtams_(INT *jw,INT *ia, INT *ja, INT *nwp,INT *map,
        //        INT *jv, INT *nvp, INT *icp)
        //    INT nw,ma,nv,iastrt,iaend,i,j,k,ji,jia;
        nv = 0;
        for (ji = 0;ji < nw; ++ji) {
            j = *(jris+ji)-1;
            iastrt = ia[j]-1;
            iaend  = ia[j+1]-1;
            if (iaend > iastrt) {
                for (jia = iastrt ;jia< iaend;jia++) {
                    k = ja[jia]-1;
                    if (!icp[k]) {
                        *(jv+nv) = k+1;
                        nv++;
                        icp[k] = nv;
                    } //end if
                } //end for
            } //end if
        } //end for loop for forming the nonz struct of (r_i)^t*A
        ris=r+is;
        //    wta_(jris, ris,ia, ja, a,&nw, &n, jv, v, &nv);
        for (ji = 0;ji < nv;++ji) {
            k=jv[ji]-1;
            v[k] = 0e+0;
        }
        for (ji = 0;ji<nw ; ++ji) {
            j = *(jris+ji)-1;
            v0 = *(ris+ji);
            iastrt = ia[j]-1;
            iaend  = ia[j+1]-1;
            if (iaend > iastrt) {
                for (jia = iastrt;jia < iaend;jia++) {
                    k = ja[jia]-1;
                    v[k] += v0*a[jia];
                }
            }  // end if
        } //end for loop for calculating the product (r_i)^t*A
        iacst=iac[ic]-1;
        iacen=iac[ic+1]-1;
        for (jkc = iacst; jkc<iacen;jkc++) {
            jc = jac[jkc]-1;
            nptjc = ipt[jc+1]-ipt[jc];
            js = ipt[jc]-1;
            jptjs = jpt+js;
            ptjs = pt+js;
            //      ytxbig_(jptjs,ptjs,&nptjc,v,&aij);
            aij=0e+00;
            if (nptjc > 0) {
                for (i = 0;i< nptjc; ++i) {
                    j = *(jptjs+i)-1;
                    aij += (*(ptjs+i))*(*(v+j));
                } //end for
            } //end if
            ac[jkc] = aij;
        } //end for
        // set nos the values of v and icp back to 0;
        for (i=0; i < nv; ++i) {
            j=jv[i]-1;
            icp[j]=0;
            v[j]=0e+00;
        } //end for
    } //end for
    
    if (v) free(v);
    if (icp) free(icp);
    if (jv) free(jv);
    
    return;
}

/**
 * \fn ivector fasp_sparse_mis (dCSRmat *A)
 *
 * \brief Get the maximal independet set of a CSR matrix
 *
 * \param A pointer to the matrix
 *
 * \note  Only use the sparsity of A, index starts from 1 (fortran)!!
 */
ivector fasp_sparse_mis (dCSRmat *A)
{
    // information of A
    INT n = A->row;
    INT *IA = A->IA;
    INT *JA = A->JA;
    
    // local variables
    INT i,j;
    INT row_begin, row_end;
    INT count=0;
    INT *flag;
    flag = (INT *)fasp_mem_calloc(n, sizeof(INT));
    //for (i=0;i<n;i++) flag[i]=0;
    memset(flag, 0, sizeof(INT)*n);
    
    // work space
    INT *work = (INT*)fasp_mem_calloc(n,sizeof(INT));
    
    // return vector
    ivector MIS;
    
    // main loop
    for (i=0;i<n;i++) {
        if (flag[i] == 0) {
            flag[i] = 1;
            row_begin = IA[i] - 1; row_end = IA[i+1] - 1;
            for (j = row_begin; j<row_end; j++) {
                if (flag[JA[j]-1] > 0) {
                    flag[i] = -1;
                    break;
                }
            }
            if (flag[i]) {
                work[count] = i; count++;
                for (j = row_begin; j<row_end; j++) {
                    flag[JA[j]-1] = -1;
                }
            }
        } // end if
    }// end for
    
    // form MIS
    MIS.row = count;
    work = (INT *)fasp_mem_realloc(work, count*sizeof(INT));
    MIS.val = work;
    
    // clean
    fasp_mem_free(flag); flag = NULL;
    
    //return
    return MIS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
