/*! \file  BlaSpmvCSRL.c
 *
 *  \brief Linear algebraic operations for dCSRLmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions.
 *
 *  Reference: 
 *         John Mellor-Crummey and John Garvin
 *         Optimizaing sparse matrix vector product computations using unroll and
 *         jam, Tech Report Rice Univ, Aug 2002.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_blas_dcsrl_mxv ( const dCSRLmat *A, const REAL *x, REAL *y )
 *
 * \brief Compute y = A*x for a sparse matrix in CSRL format
 *
 * \param A   Pointer to dCSRLmat matrix A
 * \param x   Pointer to REAL array of vector x
 * \param y   Pointer to REAL array of vector y
 *
 * \author Zhiyang Zhou, Chensong Zhang
 * \date   2011/01/07
 */
void fasp_blas_dcsrl_mxv (const dCSRLmat  *A,
                          const REAL      *x,
                          REAL            *y)
{
    const INT     dif      = A -> dif;
    const INT    *nz_diff  = A -> nz_diff;
    const INT    *rowindex = A -> index;
    const INT    *rowstart = A -> start;
    const INT    *ja       = A -> ja;
    const REAL   *a        = A -> val;
    
    INT i;
    INT row, col=0;
    INT len, rowlen;
    INT firstrow, lastrow;
    
    REAL val0, val1;
    
    for (len = 0; len < dif; len ++) {
        firstrow = rowstart[len];
        lastrow  = rowstart[len+1] - 1;
        rowlen   = nz_diff[len];
    
        if (lastrow > firstrow ) {
            //----------------------------------------------------------
            // Fully-unrolled code for special case (i.g.,rowlen = 5) 
            // Note: you can also set other special case
            //----------------------------------------------------------               
            if (rowlen == 5) {
                for (row = firstrow; row < lastrow; row += 2) {
                    val0 = a[col]*x[ja[col]];
                    val1 = a[col+5]*x[ja[col+5]];
                    col ++;
    
                    val0 += a[col]*x[ja[col]];
                    val1 += a[col+5]*x[ja[col+5]];
                    col ++;
    
                    val0 += a[col]*x[ja[col]];
                    val1 += a[col+5]*x[ja[col+5]];
                    col ++;
    
                    val0 += a[col]*x[ja[col]];
                    val1 += a[col+5]*x[ja[col+5]];
                    col ++;
    
                    val0 += a[col]*x[ja[col]];
                    val1 += a[col+5]*x[ja[col+5]];
                    col ++;
    
                    y[rowindex[row]] = val0;
                    y[rowindex[row+1]] = val1;
    
                    col += 5;
                }
            }
            else {
                //------------------------------------------------------------------
                // Unroll-and-jammed code for handling two rows at a time 
                //------------------------------------------------------------------
    
                for (row = firstrow; row < lastrow; row += 2) { 
                    val0 = 0.0;
                    val1 = 0.0;
                    for (i = 0; i < rowlen; i ++) {
                        val0 += a[col]*x[ja[col]];
                        val1 += a[col+rowlen]*x[ja[col+rowlen]];
                        col ++;
                    }
                    y[rowindex[row]] = val0;
                    y[rowindex[row+1]] = val1;
                    col += rowlen;               
                }  
            }
            firstrow = row;
        }
    
        //-----------------------------------------------------------
        // Handle leftover rows that can't be handled in bundles 
        // in the unroll-and -jammed loop 
        //-----------------------------------------------------------
    
        for (row = firstrow; row <= lastrow; row ++) {
            val0 = 0.0;
            for (i = 0; i < rowlen; i ++) {
                val0 += a[col]*x[ja[col]];
                col ++;
            }
            y[rowindex[row]] = val0;
        }
    
    }
    
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
