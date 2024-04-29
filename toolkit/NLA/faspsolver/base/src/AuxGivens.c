/*! \file  AuxGivens.c
 *
 *  \brief Givens transformation
 *
 *  \note  This file contains Level-0 (Aux) functions.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2008--Present by the FASP team. All rights reserved.
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
 * \fn void fasp_aux_givens (const REAL beta, const dCSRmat *H, dvector *y, 
 *                           REAL *work)
 *
 * \brief Perform Givens rotations to compute y |beta*e_1- H*y|
 *
 * \param beta   Norm of residual r_0
 * \param H      Upper Hessenberg dCSRmat matrix: (m+1)*m
 * \param y      Minimizer of |beta*e_1- H*y|
 * \param work   Temporary work array
 *
 * \author Xuehai Huang
 * \date   10/19/2008
 */
void fasp_aux_givens (const REAL      beta,
                      const dCSRmat  *H,
                      dvector        *y,
                      REAL           *work)
{
    const INT  Hsize = H->row;
    INT        i, j, istart, idiag, ip1start;
    REAL       h0, h1, r, c, s, tempi, tempip1, sum;
    
    memset(&work, 0x0, sizeof(REAL)*Hsize);
    work[0] = beta;

    for ( i=0; i<Hsize-1; ++i ) {
        istart   = H->IA[i];
        ip1start = H->IA[i+1];
        if (i==0) idiag = istart;
        else idiag = istart+1;
    
        h0 = H->val[idiag];      // h0=H[i][i]
        h1 = H->val[H->IA[i+1]]; // h1=H[i+1][i]
        r  = sqrt(h0*h0+h1*h1);
        c  = h0/r; s = h1/r;
    
        for ( j=idiag; j<ip1start; ++j ) {
            tempi   = H->val[j];
            tempip1 = H->val[ip1start+(j-idiag)];
            H->val[j] = c*tempi+s*tempip1;
            H->val[ip1start+(j-idiag)] = c*tempip1-s*tempi;
        }
    
        tempi   = c*work[i]+s*work[i+1];
        tempip1 = c*work[i+1]-s*work[i];
    
        work[i] = tempi; work[i+1]=tempip1;
    }
    
    for ( i = Hsize-2; i >= 0; --i ) {
        sum = work[i];
        istart = H->IA[i];
        if (i==0) idiag = istart;
        else idiag = istart+1;
    
        for ( j=Hsize-2; j>i; --j ) sum-=H->val[idiag+j-i]*y->val[j];
    
        y->val[i] = sum/H->val[idiag];
    }
    
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
