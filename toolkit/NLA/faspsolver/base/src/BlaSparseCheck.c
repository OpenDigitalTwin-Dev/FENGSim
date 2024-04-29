/*! \file  BlaSparseCheck.c
 *
 *  \brief Check properties of sparse matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c, AuxMessage.c, AuxVector.c, and BlaSparseCSR.c
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
 * \fn INT fasp_check_diagpos (const dCSRmat *A)
 *
 * \brief Check positivity of diagonal entries of a CSR sparse matrix.
 *
 * \param A   Pointer to dCSRmat matrix
 *
 * \return Number of negative diagonal entries 
 *
 * \author Shuo Zhang
 * \date   03/29/2009
 */
INT fasp_check_diagpos (const dCSRmat *A)
{
    const INT m = A->row;
    INT i, num_neg;
    
#if DEBUG_MODE > 1
    printf("### DEBUG: nr = %d, nc = %d, nnz = %d\n", A->row, A->col, A->nnz);
#endif
    
    // store diagonal of A 
    dvector        diag;
    fasp_dcsr_getdiag(m,A,&diag);
    
    // check positiveness of entries of diag
    for (num_neg=i=0;i<m;++i) {
        if (diag.val[i]<0) num_neg++;        
    }
    
    printf("Number of negative diagonal entries = %d\n", num_neg);
    
    fasp_dvec_free(&diag);
    
    return num_neg;
}

/**
 * \fn SHORT fasp_check_diagzero (const dCSRmat *A)
 *
 * \brief Check if a CSR sparse matrix has diagonal entries that are very close to zero.
 *
 * \param A pointer to the dCSRmat matrix
 * 
 * \return FASP_SUCCESS if no diagonal entry is close to zero, else ERROR
 *
 * \author Shuo Zhang
 * \date   03/29/2009
 */
SHORT fasp_check_diagzero (const dCSRmat *A)
{
    const INT    m  = A->row;
    const INT   *ia = A->IA, *ja = A->JA;
    const REAL  *aj = A->val;

    SHORT        status = FASP_SUCCESS;
    INT          i,j,k,begin_row,end_row;
    
    for ( i = 0; i < m; ++i ) {
        begin_row = ia[i]; end_row = ia[i+1];
        for ( k = begin_row; k < end_row; ++k ) {
            j = ja[k];
            if ( i == j ) {
                if ( ABS(aj[k]) < SMALLREAL ) {
                    printf("### ERROR: diag[%d] = %e, close to zero!\n", i, aj[k]);
                    status = ERROR_DATA_ZERODIAG;
                    goto FINISHED;
                }
            }    
        } // end for k
    } // end for i

 FINISHED:    
    return status;
}

/**
 * INT fasp_check_diagdom (const dCSRmat *A)
 *
 * \brief Check whether a matrix is diagonally dominant.
 *
 * \param A   Pointer to the dCSRmat matrix
 *
 * \return Number of the rows which are not diagonally dominant
 *
 * \note The routine chechs whether the sparse matrix is diagonally dominant each row.
 *       It will print out the percentage of the rows which are diagonally dominant.
 *
 * \author Shuo Zhang
 * \date   03/29/2009
 */
INT fasp_check_diagdom (const dCSRmat *A)
{
    const INT   nn  = A->row;
    const INT   nnz = A->IA[nn]-A->IA[0];
    INT         i, j, k;
    REAL        sum;
    
    INT *rowp = (INT *)fasp_mem_calloc(nnz,sizeof(INT));
    
    for ( i=0; i<nn; ++i ) {
        for ( j=A->IA[i]; j<A->IA[i+1]; ++j ) rowp[j]=i;
    }
    
    for ( k=0, i=0; i<nn; ++i ) {
        sum = 0.0;
        for ( j=A->IA[i]; j<A->IA[i+1]; ++j ) {
            if ( A->JA[j]==i ) sum += A->val[j];
            if ( A->JA[j]!=i ) sum -= fabs(A->val[j]);
        }
        if ( sum<-SMALLREAL ) ++k;
    }
    
    printf("Percentage of the diagonal-dominant rows is %3.2lf%s\n", 
           100.0*(REAL)(nn-k)/(REAL)nn,"%");
    
    fasp_mem_free(rowp); rowp = NULL;
    
    return k;
}

/**
 * \fn INT fasp_check_symm (const dCSRmat *A)
 *
 * \brief Check symmetry of a sparse matrix of CSR format.
 *
 * \param A   Pointer to the dCSRmat matrix
 *
 * \return 1 and 2 if the structure of the matrix is not symmetric;
 *         0 if the structure of the matrix is symmetric,
 * 
 * \note Print the maximal relative difference between matrix and its transpose.
 *
 * \author Shuo Zhang
 * \date   03/29/2009
 */
INT fasp_check_symm (const dCSRmat *A)
{
    const REAL symmetry_tol = 1.0e-12;
    
    INT  *rowp,*rows[2],*cols[2];
    INT   i,j,mdi,mdj;
    INT   nns[2],tnizs[2];
    INT   type=0;
    
    REAL  maxdif,dif;
    REAL *vals[2];
    
    const INT nn  = A->row;
    const INT nnz = A->IA[nn]-A->IA[0];
    
    if (nnz!=A->nnz) {
        printf("### ERROR: nnz=%d, ia[n]-ia[0]=%d, mismatch!\n",A->nnz,nnz);
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    rowp=(INT *)fasp_mem_calloc(nnz,sizeof(INT));
    
    for (i=0;i<nn;++i) {
        for (j=A->IA[i];j<A->IA[i+1];++j) rowp[j]=i;
    }
    
    rows[0]=(INT *)fasp_mem_calloc(nnz,sizeof(INT));
    cols[0]=(INT *)fasp_mem_calloc(nnz,sizeof(INT));
    vals[0]=(REAL *)fasp_mem_calloc(nnz,sizeof(REAL));
    
    memcpy(rows[0],rowp,nnz*sizeof(INT));
    memcpy(cols[0],A->JA,nnz*sizeof(INT));
    memcpy(vals[0],A->val,nnz*sizeof(REAL));
    
    nns[0]=nn;
    nns[1]=A->col;
    tnizs[0]=nnz;    
    
    rows[1]=(INT *)fasp_mem_calloc(nnz,sizeof(INT));    
    cols[1]=(INT *)fasp_mem_calloc(nnz,sizeof(INT));    
    vals[1]=(REAL *)fasp_mem_calloc(nnz,sizeof(REAL));
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
    memcpy(rows[0],rows[1],nnz*sizeof(INT));
    memcpy(cols[0],cols[1],nnz*sizeof(INT));
    memcpy(vals[0],vals[1],nnz*sizeof(REAL));
    nns[0]=A->col;
    nns[1]=nn;
    
    fasp_dcsr_transpose(rows,cols,vals,nns,tnizs);
    
    maxdif=0.;
    mdi=0;
    mdj=0;
    for (i=0;i<nnz;++i) {
        rows[0][i]=rows[1][i]-rows[0][i];
        if (rows[0][i]!=0) {
            type=-1;
            mdi=rows[1][i];
            break;
        }
        
        cols[0][i]=cols[1][i]-cols[0][i];
        if (cols[0][i]!=0) {
            type=-2;
            mdj=cols[1][i];
            break;
        }
        
        if (fabs(vals[0][i])>SMALLREAL||fabs(vals[1][i])>SMALLREAL) {
            dif=fabs(vals[1][i]-vals[0][i])/(fabs(vals[0][i])+fabs(vals[1][i]));
            if (dif>maxdif) {
                maxdif=dif;
                mdi=rows[0][i];
                mdj=cols[0][i];
            }
        }
    }
    
    if (maxdif>symmetry_tol) type=-3;
    
    switch (type) {
    case 0:
        printf("Matrix is symmetric with max relative difference is %1.3le\n",maxdif);
        break;
    case -1:
        printf("Matrix has nonsymmetric pattern, check the %d-th, %d-th and %d-th rows and cols\n",
               mdi-1,mdi,mdi+1);
        break;
    case -2:
        printf("Matrix has nonsymmetric pattern, check the %d-th, %d-th and %d-th cols and rows\n",
               mdj-1,mdj,mdj+1);
        break;
    case -3:
        printf("Matrix is nonsymmetric with max relative difference is %1.3le\n",maxdif);
        break;
    default:
        break;
    }
    
    fasp_mem_free(rowp);    rowp    = NULL;
    fasp_mem_free(rows[0]); rows[0] = NULL;
    fasp_mem_free(rows[1]); rows[1] = NULL;
    fasp_mem_free(cols[0]); cols[0] = NULL;
    fasp_mem_free(cols[1]); cols[1] = NULL;
    fasp_mem_free(vals[0]); vals[0] = NULL;
    fasp_mem_free(vals[1]); vals[1] = NULL;

    return type;
}

/**
 * \fn void fasp_check_dCSRmat (const dCSRmat *A)
 *
 * \brief Check whether an dCSRmat matrix is supported or not
 *
 * \param A   Pointer to the matrix in dCSRmat format
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_check_dCSRmat (const dCSRmat *A)
{    
    INT i;    
    
    if ( (A->IA == NULL) || (A->JA == NULL) || (A->val == NULL) ) {
        printf("### ERROR: Something is wrong with the matrix!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    if ( A->row != A->col ) {
        printf("### ERROR: Non-square CSR matrix!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }
    
    if ( ( A->nnz <= 0 ) || ( A->row == 0 ) || ( A->col == 0 ) ) {
        printf("### ERROR: Empty CSR matrix!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    for ( i = 0; i < A->nnz; ++i ) {
        if ( ( A->JA[i] < 0 ) || ( A->JA[i] >= A->col ) ) {
            printf("### ERROR: Wrong CSR matrix format!\n");
            fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
        }
    }
}

/**
 * \fn SHORT fasp_check_iCSRmat (const iCSRmat *A)
 *
 * \brief Check whether an iCSRmat matrix is valid or not
 *
 * \param A   Pointer to the matrix in iCSRmat format
 *
 * \author Shuo Zhang
 * \date   03/29/2009
 */
SHORT fasp_check_iCSRmat (const iCSRmat *A)
{    
    INT i;    
    
    if ( (A->IA == NULL) || (A->JA == NULL) || (A->val == NULL) ) {
        printf("### ERROR: Something is wrong with the matrix!\n");
        fasp_chkerr(ERROR_MAT_SIZE, __FUNCTION__);
    }

    if (A->row != A->col) {
        printf("### ERROR: Non-square CSR matrix!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    if ( (A->nnz==0) || (A->row==0) || (A->col==0) ) {
        printf("### ERROR: Empty CSR matrix!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    for (i=0;i<A->nnz;++i) {
        if ( (A->JA[i]<0) || (A->JA[i]-A->col>=0) ) {
            printf("### ERROR: Wrong CSR matrix format!\n");
            fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
        }
    }
    
    return FASP_SUCCESS;
}

/**
* \fn void fasp_check_ordering (dCSRmat *A)
*
* \brief Check whether each row of A is in ascending order w.r.t. column indices
*
* \param A   Pointer to the dCSRmat matrix
*
* \author Chensong Zhang
* \date   02/26/2019
*/
void fasp_check_ordering (dCSRmat *A)
{
	const INT n = A->col;
	INT i, j, j1, j2, start, end;

	for ( i=0; i<n; ++i ) {

		start = A->IA[i];
		end   = A->IA[i+1] - 1;

		for ( j=start; j<end-1; ++j ) {
			j1 = A->JA[j]; j2 = A->JA[j + 1];
			if ( j1 >= j2 ) {
				printf("### ERROR: Order in row %10d is wrong! %10d, %10d\n", i, j1, j2);
				fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
			}
		}

	}

}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
