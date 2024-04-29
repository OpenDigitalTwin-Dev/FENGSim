/*! \file  BlaSpmvBLC.c
 *
 *  \brief Linear algebraic operations for dBLCmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         BlaSpmvCSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_block.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_blas_dblc_aAxpy (const REAL alpha, const dBLCmat *A,
 *                                const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y
 *
 * \param alpha  REAL factor a
 * \param A      Pointer to dBLCmat matrix A
 * \param x      Pointer to array x
 * \param y      Pointer to array y
 *
 * \author Xiaozhe Hu
 * \date   06/04/2010
 */
void fasp_blas_dblc_aAxpy (const REAL      alpha,
                           const dBLCmat  *A,
                           const REAL     *x,
                           REAL           *y)
{
    // information of A
    const INT brow = A->brow;
    
    // local variables
    register dCSRmat *A11, *A12, *A21, *A22;
    register dCSRmat *A13, *A23, *A31, *A32, *A33;
    
    INT row1, col1;
    INT row2, col2;
    
    const register REAL *x1, *x2, *x3;
    register REAL       *y1, *y2, *y3;
    
    INT i,j;
    INT start_row, start_col;
    
    switch (brow) {
            
        case 2:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A21 = A->blocks[2];
            A22 = A->blocks[3];
            
            row1 = A11->row;
            col1 = A11->col;
            
            x1 = x;
            x2 = &(x[col1]);
            y1 = y;
            y2 = &(y[row1]);
            
            // y1 = alpha*A11*x1 + alpha*A12*x2 + y1
            if (A11) fasp_blas_dcsr_aAxpy(alpha, A11, x1, y1);
            if (A12) fasp_blas_dcsr_aAxpy(alpha, A12, x2, y1);
            
            // y2 = alpha*A21*x1 + alpha*A22*x2 + y2
            if (A21) fasp_blas_dcsr_aAxpy(alpha, A21, x1, y2);
            if (A22) fasp_blas_dcsr_aAxpy(alpha, A22, x2, y2);
            
            break;
            
        case 3:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A13 = A->blocks[2];
            A21 = A->blocks[3];
            A22 = A->blocks[4];
            A23 = A->blocks[5];
            A31 = A->blocks[6];
            A32 = A->blocks[7];
            A33 = A->blocks[8];
            
            row1 = A11->row;
            col1 = A11->col;
            row2 = A22->row;
            col2 = A22->col;
            
            x1 = x;
            x2 = &(x[col1]);
            x3 = &(x[col1+col2]);
            y1 = y;
            y2 = &(y[row1]);
            y3 = &(y[row1+row2]);
            
            // y1 = alpha*A11*x1 + alpha*A12*x2 + alpha*A13*x3 + y1
            if (A11) fasp_blas_dcsr_aAxpy(alpha, A11, x1, y1);
            if (A12) fasp_blas_dcsr_aAxpy(alpha, A12, x2, y1);
            if (A13) fasp_blas_dcsr_aAxpy(alpha, A13, x3, y1);
            
            // y2 = alpha*A21*x1 + alpha*A22*x2 + alpha*A23*x3 + y2
            if (A21) fasp_blas_dcsr_aAxpy(alpha, A21, x1, y2);
            if (A22) fasp_blas_dcsr_aAxpy(alpha, A22, x2, y2);
            if (A23) fasp_blas_dcsr_aAxpy(alpha, A23, x3, y2);
            
            // y3 = alpha*A31*x1 + alpha*A32*x2 + alpha*A33*x3 + y2
            if (A31) fasp_blas_dcsr_aAxpy(alpha, A31, x1, y3);
            if (A32) fasp_blas_dcsr_aAxpy(alpha, A32, x2, y3);
            if (A33) fasp_blas_dcsr_aAxpy(alpha, A33, x3, y3);
            
            break;
            
        default:
            
            start_row = 0;
            start_col = 0;
            
            for (i=0; i<brow; i++) {
                
                for (j=0; j<brow; j++) {
                    
                    if (A->blocks[i*brow+j]) {
                        fasp_blas_dcsr_aAxpy(alpha, A->blocks[i*brow+j],
                                             &(x[start_col]), &(y[start_row]));
                    }
                    start_col = start_col + A->blocks[j*brow+j]->col;

                }
                
                start_row = start_row + A->blocks[i*brow+i]->row;
                start_col = 0;
            }
            
            break;
            
    } // end of switch
    
}

/**
 * \fn void fasp_blas_dblc_mxv (const dBLCmat *A, const REAL *x, REAL *y)
 *
 * \brief Matrix-vector multiplication y = A*x
 *
 * \param A      Pointer to dBLCmat matrix A
 * \param x      Pointer to array x
 * \param y      Pointer to array y
 *
 * \author Chensong Zhang
 * \date   04/27/2013
 */
void fasp_blas_dblc_mxv (const dBLCmat  *A,
                         const REAL     *x,
                         REAL           *y)
{
    // information of A
    const INT brow = A->brow;
    
    // local variables
    register dCSRmat *A11, *A12, *A21, *A22;
    register dCSRmat *A13, *A23, *A31, *A32, *A33;
    
    INT row1, col1;
    INT row2, col2;
    
    const register REAL *x1, *x2, *x3;
    register REAL       *y1, *y2, *y3;
    
    INT i,j;
    INT start_row, start_col;
    
    switch (brow) {
            
        case 2:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A21 = A->blocks[2];
            A22 = A->blocks[3];
            
            row1 = A11->row;
            col1 = A11->col;
            
            x1 = x;
            x2 = &(x[col1]);
            y1 = y;
            y2 = &(y[row1]);
            
            // y1 = A11*x1 + A12*x2
            if (A11) fasp_blas_dcsr_mxv(A11, x1, y1);
            if (A12) fasp_blas_dcsr_aAxpy(1.0, A12, x2, y1);
            
            // y2 = A21*x1 + A22*x2
            if (A21) fasp_blas_dcsr_mxv(A21, x1, y2);
            if (A22) fasp_blas_dcsr_aAxpy(1.0, A22, x2, y2);
            
            break;
            
        case 3:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A13 = A->blocks[2];
            A21 = A->blocks[3];
            A22 = A->blocks[4];
            A23 = A->blocks[5];
            A31 = A->blocks[6];
            A32 = A->blocks[7];
            A33 = A->blocks[8];
            
            row1 = A11->row;
            col1 = A11->col;
            row2 = A22->row;
            col2 = A22->col;
            
            x1 = x;
            x2 = &(x[col1]);
            x3 = &(x[col1+col2]);
            y1 = y;
            y2 = &(y[row1]);
            y3 = &(y[row1+row2]);
            
            // y1 = A11*x1 + A12*x2 + A13*x3 + y1
            if (A11) fasp_blas_dcsr_mxv(A11, x1, y1);
            if (A12) fasp_blas_dcsr_aAxpy(1.0, A12, x2, y1);
            if (A13) fasp_blas_dcsr_aAxpy(1.0, A13, x3, y1);
            
            // y2 = A21*x1 + A22*x2 + A23*x3 + y2
            if (A21) fasp_blas_dcsr_mxv(A21, x1, y2);
            if (A22) fasp_blas_dcsr_aAxpy(1.0, A22, x2, y2);
            if (A23) fasp_blas_dcsr_aAxpy(1.0, A23, x3, y2);
            
            // y3 = A31*x1 + A32*x2 + A33*x3 + y2
            if (A31) fasp_blas_dcsr_mxv(A31, x1, y3);
            if (A32) fasp_blas_dcsr_aAxpy(1.0, A32, x2, y3);
            if (A33) fasp_blas_dcsr_aAxpy(1.0, A33, x3, y3);
            
            break;
            
        default:
            
            start_row = 0;
            start_col = 0;
            
            for (i=0; i<brow; i++) {
                
                for (j=0; j<brow; j++){
                    
                    if (j==0) {
                        if (A->blocks[i*brow+j]){
                            fasp_blas_dcsr_mxv(A->blocks[i*brow+j], &(x[start_col]), &(y[start_row]));
                        }
                    }
                    else {
                        if (A->blocks[i*brow+j]){
                            fasp_blas_dcsr_aAxpy(1.0, A->blocks[i*brow+j], &(x[start_col]), &(y[start_row]));
                        }
                    }
                    start_col = start_col + A->blocks[j*brow+j]->col;
                }
                
                start_row = start_row + A->blocks[i*brow+i]->row;
                start_col = 0;
            }
            
            break;
            
    } // end of switch
    
}

/**
 * \fn void fasp_blas_ldblc_aAxpy (const REAL alpha, const dBLCmat *A,
 *                                const LONGREAL  *x,  REAL *y)
 *
 * \brief Matrix-vector multiplication y = alpha*A*x + y
 *
 * \param alpha  REAL factor a
 * \param A      Pointer to dBLCmat matrix A
 * \param x      Pointer to array x
 * \param y      Pointer to array y
 *
 * \author Lai Ting
 * \date   08/01/2022
 */
void fasp_blas_ldblc_aAxpy (const REAL      alpha,
                           const dBLCmat    *A,
                           const LONGREAL   *x,
                           REAL             *y)
{
   // information of A
    const INT brow = A->brow;
    
    // local variables
    register dCSRmat *A11, *A12, *A21, *A22;
    register dCSRmat *A13, *A23, *A31, *A32, *A33;
    
    INT row1, col1;
    INT row2, col2;
    
    const register LONGREAL *x1, *x2, *x3;
    register REAL       *y1, *y2, *y3;
    
    INT i,j;
    INT start_row, start_col;
    
    switch (brow) {
            
        case 2:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A21 = A->blocks[2];
            A22 = A->blocks[3];
            
            row1 = A11->row;
            col1 = A11->col;
            
            x1 = x;
            x2 = &(x[col1]);
            y1 = y;
            y2 = &(y[row1]);
            
            // y1 = alpha*A11*x1 + alpha*A12*x2 + y1
            if (A11) fasp_blas_ldcsr_aAxpy(alpha, A11, x1, y1);
            if (A12) fasp_blas_ldcsr_aAxpy(alpha, A12, x2, y1);
            
            // y2 = alpha*A21*x1 + alpha*A22*x2 + y2
            if (A21) fasp_blas_ldcsr_aAxpy(alpha, A21, x1, y2);
            if (A22) fasp_blas_ldcsr_aAxpy(alpha, A22, x2, y2);
            
            break;
            
        case 3:
            A11 = A->blocks[0];
            A12 = A->blocks[1];
            A13 = A->blocks[2];
            A21 = A->blocks[3];
            A22 = A->blocks[4];
            A23 = A->blocks[5];
            A31 = A->blocks[6];
            A32 = A->blocks[7];
            A33 = A->blocks[8];
            
            row1 = A11->row;
            col1 = A11->col;
            row2 = A22->row;
            col2 = A22->col;
            
            x1 = x;
            x2 = &(x[col1]);
            x3 = &(x[col1+col2]);
            y1 = y;
            y2 = &(y[row1]);
            y3 = &(y[row1+row2]);
            
            // y1 = alpha*A11*x1 + alpha*A12*x2 + alpha*A13*x3 + y1
            if (A11) fasp_blas_ldcsr_aAxpy(alpha, A11, x1, y1);
            if (A12) fasp_blas_ldcsr_aAxpy(alpha, A12, x2, y1);
            if (A13) fasp_blas_ldcsr_aAxpy(alpha, A13, x3, y1);
            
            // y2 = alpha*A21*x1 + alpha*A22*x2 + alpha*A23*x3 + y2
            if (A21) fasp_blas_ldcsr_aAxpy(alpha, A21, x1, y2);
            if (A22) fasp_blas_ldcsr_aAxpy(alpha, A22, x2, y2);
            if (A23) fasp_blas_ldcsr_aAxpy(alpha, A23, x3, y2);
            
            // y3 = alpha*A31*x1 + alpha*A32*x2 + alpha*A33*x3 + y2
            if (A31) fasp_blas_ldcsr_aAxpy(alpha, A31, x1, y3);
            if (A32) fasp_blas_ldcsr_aAxpy(alpha, A32, x2, y3);
            if (A33) fasp_blas_ldcsr_aAxpy(alpha, A33, x3, y3);
            
            break;
            
        default:
            
            start_row = 0;
            start_col = 0;
            
            for (i=0; i<brow; i++) {
                
                for (j=0; j<brow; j++) {
                    
                    if (A->blocks[i*brow+j]) {
                        fasp_blas_ldcsr_aAxpy(alpha, A->blocks[i*brow+j],
                                             &(x[start_col]), &(y[start_row]));
                    }
                    start_col = start_col + A->blocks[j*brow+j]->col;

                }
                
                start_row = start_row + A->blocks[i*brow+i]->row;
                start_col = 0;
            }
            
            break;
            
    } // end of switch
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
