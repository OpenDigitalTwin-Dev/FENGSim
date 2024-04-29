/*! \file  BlaILU.c
 *
 *  \brief Incomplete LU decomposition: ILUk, ILUt, ILUtp
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c
 *
 *  Translated from SparseKit (Fortran code) by Chunsheng Feng, 09/03/2016
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2016--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static void fasp_qsplit  (REAL *a, INT *ind, INT n, INT ncut);
static void fasp_sortrow (INT num,INT *q);
static void fasp_check_col_index (INT row, INT num, INT  *q);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_iluk (INT n, REAL *a,INT *ja, INT *ia, INT lfil,
 *                     REAL *alu, INT *jlu, INT iwk, INT *ierr, INT *nzlu)
 *
 * \brief Get ILU factorization with level of fill-in k (ilu(k)) for a CSR matrix A
 *
 * \param n    row number of A
 * \param a    nonzero entries of A
 * \param ja   integer array of column for A
 * \param ia   integer array of row pointers for A
 * \param lfil integer. The fill-in parameter. Each row of L and each row
 *             of U will have a maximum of lfil elements (excluding the diagonal
 *             element). lfil must be .ge. 0.
 * \param alu  matrix stored in Modified Sparse Row (MSR) format containing
 *             the L and U factors together. The diagonal (stored in
 *             alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
 *             contains the i-th row of L (excluding the diagonal entry=1)
 *             followed by the i-th row of U.
 * \param jlu  integer array of length n containing the pointers to
 *             the beginning of each row of U in the matrix alu,jlu.
 * \param iwk  integer. The minimum length of arrays alu, jlu, and levs.
 * \param ierr integer pointer. Return error message with the following meaning.
 *                0  --> successful return.
 *               >0  --> zero pivot encountered at step number ierr.
 *               -1  --> Error. input matrix may be wrong.
 *                       (The elimination process has generated a
 *                       row in L or U whose length is .gt.  n.)
 *               -2  --> The matrix L overflows the array al.
 *               -3  --> The matrix U overflows the array alu.
 *               -4  --> Illegal value for lfil.
 *               -5  --> zero row encountered.
 * \param nzlu  integer pointer. Return number of nonzero entries for alu and jlu
 *
 * \note:  All the diagonal elements of the input matrix must be nonzero.
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
void fasp_iluk (INT    n,
                REAL  *a,
                INT   *ja,
                INT   *ia,
                INT    lfil,
                REAL  *alu,
                INT   *jlu,
                INT    iwk,
                INT   *ierr,
                INT   *nzlu)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    /*----------------------------------------------------------------------
     SPARSKIT ROUTINE ILUK -- ILU WITH LEVEL OF FILL-IN OF K (ILU(k))
     ----------------------------------------------------------------------
     
     on entry:
     ==========
     n       = integer. The row dimension of the matrix A. The matrix
     
     a,ja,ia = matrix stored in Compressed Sparse Row format.
     
     lfil    = integer. The fill-in parameter. Each element whose
     leve-of-fill exceeds lfil during the ILU process is dropped.
     lfil must be .ge. 0
     
     iwk     = integer. The minimum length of arrays alu, jlu, and levs.
     
     On return:
     ===========
     
     alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
     the L and U factors together. The diagonal (stored in
     alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
     contains the i-th row of L (excluding the diagonal entry=1)
     followed by the i-th row of U.
     
     jlu     = integer array of length n containing the pointers to
     the beginning of each row of U in the matrix alu,jlu.
     
     levs    = integer (work) array of size iwk -- which contains the
     levels of each element in alu, jlu.
     
     ierr    = integer. Error message with the following meaning.
     ierr  = 0    --> successful return.
     ierr .gt. 0  --> zero pivot encountered at step number ierr.
     ierr  = -1   --> Error. input matrix may be wrong.
     (The elimination process has generated a
     row in L or U whose length is .gt.  n.)
     ierr  = -2   --> The matrix L overflows the array al.
     ierr  = -3   --> The matrix U overflows the array alu.
     ierr  = -4   --> Illegal value for lfil.
     ierr  = -5   --> zero row encountered in A or U.
     
     work arrays:
     =============
     jw      = integer work array of length 3*n.
     w       = real work array of length n
     
     ----------------------------------------------------------------------
     w, ju (1:n) store the working array [1:ii-1 = L-part, ii:n = U-part]
     jw(n+1:2n)  stores the nonzero indicator.
     
     Notes:
     ------
     All the diagonal elements of the input matrix must be nonzero.
     ---------------------------------------------------------------------- */
    
    // locals
    INT ju0, k, j1, j2, j, ii, i, lenl, lenu, jj, jrow, jpos, n2, jlev, NE;
    REAL t, s, fact;
    SHORT cinindex=0;
    REAL *w;
    INT *ju, *jw, *levs;
    
    if (lfil  <  0) goto F998;
    
    w = (REAL *)fasp_mem_calloc(n, sizeof(REAL));
    ju = (INT *)fasp_mem_calloc(n, sizeof(INT));
    jw = (INT *)fasp_mem_calloc(3*n, sizeof(INT));
    levs = (INT *)fasp_mem_calloc(iwk, sizeof(INT));
    
    --jw;
    --w;
    --ju;
    --jlu;
    --alu;
    --ia;
    --ja;
    --a;
    --levs;
    
    /*-----------------------------------------------------------------------
     shift index for C routines
     -----------------------------------------------------------------------*/
    if (ia[1]  ==  0) cinindex=1 ;
    if (cinindex)
    {
        NE = n + 1; //modify by chunsheng 2012, Sep, 1;
        for (i=1; i<=NE; ++i)  ++ia[i];
        NE = ia[n+1] - 1;
        for (i=1; i<=NE; ++i)  ++ja[i];
    }
    
    /*-----------------------------------------------------------------------
     initialize ju0 (points to next element to be added to alu,jlu)
     and pointer array.
     ------------------------------------------------------------------------*/
    n2 = n + n;
    ju0 = n + 2;
    jlu[1] = ju0;
    
    // initialize nonzero indicator array + levs array --
    for(j = 1; j<=2*n; ++j) jw[j] = 0;
    
    /*-----------------------------------------------------------------------
     beginning of main loop.
     ------------------------------------------------------------------------*/
    for(ii = 1; ii <= n; ++ii) 	{  //500
        j1 = ia[ii];
        j2 = ia[ii + 1] - 1;
        
        //	 unpack L-part and U-part of row of A in arrays w
        lenu = 1;
        lenl = 0;
        jw[ii] = ii;
        w[ii] = 0.0;
        jw[n + ii] = ii;
        
        //
        for(j = j1; j <= j2; ++j) 	{ //170
            k = ja[j];
            t = a[j];
            if (t  ==  0.0) continue;  //goto g170;
            if (k  <  ii) 	{
                ++lenl;
                jw[lenl] = k;
                w[lenl] = t;
                jw[n2 + lenl] = 0;
                jw[n + k] = lenl;
            } else if (k  ==  ii) {
                w[ii] = t;
                jw[n2 + ii] = 0;
            } else 	{
                ++lenu;
                jpos = ii + lenu - 1;
                jw[jpos] = k;
                w[jpos] = t;
                jw[n2 + jpos] = 0;
                jw[n + k] = jpos;
            }
            
        }  //170
        
        jj = 0;
        //	 eliminate previous rows
        
    F150:
        ++jj;
        if (jj  >  lenl) goto F160;
        
        /*----------------------------------------------------------------------
         in order to do the elimination in the correct order we must select
         the smallest column index among jw(k), k=jj+1, ..., lenl.
         -----------------------------------------------------------------------*/
        
        jrow = jw[jj];
        k = jj;
        
        //	 determine smallest column index
        for(j = jj + 1; j <= lenl; ++j) 	{ //151
            if (jw[j]  <  jrow) {
                jrow = jw[j];
                k = j;
            }
        } //151
        
        if (k  !=  jj) {
            //     exchange in jw
            j = jw[jj];
            jw[jj] = jw[k];
            jw[k] = j;
            //     exchange in jw(n+  (pointers/ nonzero indicator).
            jw[n + jrow] = jj;
            jw[n + j] = k;
            //     exchange in jw(n2+  (levels)
            j = jw[n2 + jj];
            jw[n2 + jj] = jw[n2 + k];
            jw[n2 + k] = j;
            //     exchange in w
            s = w[jj];
            w[jj] = w[k];
            w[k] = s;
        }
        
        //	 zero out element in row by resetting jw(n+jrow) to zero.
        jw[n + jrow] = 0;
        
        //	 get the multiplier for row to be eliminated (jrow) + its level
        fact = w[jj]*alu[jrow];
        jlev = jw[n2 + jj];
        if (jlev  >  lfil) goto F150;
        
        //	 combine current row and row jrow
        for(k = ju[jrow]; k <= jlu[jrow + 1] - 1; ++k ) { // 203
            s = fact*alu[k];
            j = jlu[k];
            jpos = jw[n + j];
            if (j  >=  ii) {
                //	 dealing with upper part.
                if (jpos  ==  0) {
                    //	 this is a fill-in element
                    ++lenu;
                    if (lenu  >  n) goto F995;
                    i = ii + lenu - 1;
                    jw[i] = j;
                    jw[n + j] = i;
                    w[i] = -s;
                    jw[n2 + i] = jlev + levs[k] + 1;
                } else 	{
                    //	 this is not a fill-in element
                    w[jpos] = w[jpos] - s;
                    jw[n2 + jpos] = MIN(jw[n2 + jpos], jlev + levs[k] + 1);
                }
            } else 	{
                //	 dealing with lower part.
                if (jpos  ==  0) 	{
                    //	 this is a fill-in element
                    ++lenl;
                    if (lenl  >  n) goto F995;
                    jw[lenl] = j;
                    jw[n + j] = lenl;
                    w[lenl] = -s;
                    jw[n2 + lenl] = jlev + levs[k] + 1;
                } else {
                    //	 this is not a fill-in element
                    w[jpos] = w[jpos] - s;
                    jw[n2 + jpos] = MIN(jw[n2 + jpos], jlev + levs[k] + 1);
                }
            }
            
        } //203
        w[jj] = fact;
        jw[jj] = jrow;
        goto F150;
        
    F160:
        //  reset double-pointer to zero (U-part)
        for(k = 1; k <= lenu; ++k)  jw[n + jw[ii + k - 1]] = 0;
        
        //	 update l-matrix
        for(k = 1; k <= lenl; ++k ) {   //204
            if (ju0  >  iwk) goto F996;
            if (jw[n2 + k]  <=  lfil)  {
                alu[ju0] = w[k];
                jlu[ju0] = jw[k];
                ++ju0;
            }
        } //204
        
        //	 save pointer to beginning of row ii of U
        ju[ii] = ju0;
        
        //	 update u-matrix
        for(k = ii + 1; k <= ii + lenu - 1; ++k ) {  //302
            if (ju0  >  iwk) goto F997;
            
            if (jw[n2 + k]  <=  lfil) {
                jlu[ju0] = jw[k];
                alu[ju0] = w[k];
                levs[ju0] = jw[n2 + k];
                ++ju0;
            }
            
        } //302
        
        if (w[ii]  ==  0.0) goto F999;
        //
        alu[ii] = 1.0/w[ii];
        
        //	 update pointer to beginning of next row of U.
        jlu[ii + 1] = ju0;
        /*----------------------------------------------------------------------
         end main loop
         -----------------------------------------------------------------------*/
    } //500
    
    *nzlu = ju[n] - 1;
    
    if (cinindex)  {
        for ( i = 1; i <= *nzlu; ++i ) 	--jlu[i];
    }
    
    *ierr = 0;
    
F100:
    ++jw;
    ++w;
    ++ju;
    ++jlu;
    ++alu;
    ++ia;
    ++ja;
    ++a;
    ++levs;
    
    fasp_mem_free(w);     w    = NULL;
    fasp_mem_free(ju);    ju   = NULL;
    fasp_mem_free(jw);    jw   = NULL;
    fasp_mem_free(levs);  levs = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
    
    // incomprehensible error. Matrix must be wrong.
F995:
    printf("### ERROR: Incomprehensible error. [%s]\n", __FUNCTION__);
    *ierr = -1;
    goto F100;
    
    // insufficient storage in L.
F996:
    printf("### ERROR: Insufficient storage in L. [%s]\n", __FUNCTION__);
    *ierr = -2;
    goto F100;
    
    // insufficient storage in U.
F997:
    printf("### ERROR: Insufficient storage in U. [%s]\n", __FUNCTION__);
    *ierr = -3;
    goto F100;
    
    // illegal lfil entered.
F998:
    printf("### ERROR: Illegal lfil entered. [%s]\n", __FUNCTION__);
    *ierr = -4;
    return;
    
    // zero row encountered in A or U.
F999:
    printf("### ERROR: Zero row encountered in A or U. [%s]\n", __FUNCTION__);
    *ierr = -5;
    goto F100;
    /*----------------end-of-iluk--------------------------------------------
     ---------------------------------------------------------------------- */
}

/**
 * \fn void fasp_ilut (INT n, REAL *a, INT *ja, INT *ia, INT lfil, REAL droptol,
 *                     REAL *alu, INT *jlu, INT iwk, INT *ierr, INT *nz)
 *
 * \brief Get incomplete LU factorization with dual truncations of a CSR matrix A
 *
 * \param n   row number of A
 * \param a   nonzero entries of A
 * \param ja  integer array of column for A
 * \param ia  integer array of row pointers for A
 * \param lfil  integer. The fill-in parameter. Each row of L and each row
 *              of U will have a maximum of lfil elements (excluding the diagonal
 *              element). lfil must be .ge. 0.
 * \param droptol  real*8. Sets the threshold for dropping small terms in the
 *                 factorization. See below for details on dropping strategy.
 * \param alu  matrix stored in Modified Sparse Row (MSR) format containing
 *             the L and U factors together. The diagonal (stored in
 *             alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
 *             contains the i-th row of L (excluding the diagonal entry=1)
 *             followed by the i-th row of U.
 * \param jlu  integer array of length n containing the pointers to
 *             the beginning of each row of U in the matrix alu,jlu.
 * \param iwk  integer. The lengths of arrays alu and jlu. If the arrays
 *             are not big enough to store the ILU factorizations, ilut
 *             will stop with an error message.
 * \param ierr  integer pointer. Return error message with the following meaning.
 *                0  --> successful return.
 *               >0  --> zero pivot encountered at step number ierr.
 *               -1  --> Error. input matrix may be wrong.
 *                       (The elimination process has generated a
 *                       row in L or U whose length is .gt.  n.)
 *               -2  --> The matrix L overflows the array al.
 *               -3  --> The matrix U overflows the array alu.
 *               -4  --> Illegal value for lfil.
 *               -5  --> zero row encountered.
 * \param nz  integer pointer. Return number of nonzero entries for alu and jlu
 *
 * \note  All the diagonal elements of the input matrix must be nonzero.
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
void fasp_ilut (INT    n,
                REAL  *a,
                INT   *ja,
                INT   *ia,
                INT    lfil,
                REAL   droptol,
                REAL  *alu,
                INT   *jlu,
                INT    iwk,
                INT   *ierr,
                INT   *nz)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    /*--------------------------------------------------------------------*
     *** ILUT preconditioner ***                                      *
     incomplete LU factorization with dual truncation mechanism       *
     ----------------------------------------------------------------------*
     Author: Yousef Saad *May, 5, 1990, Latest revision, August 1996  *
     ----------------------------------------------------------------------*
     PARAMETERS
     -----------
     
     on entry:
     ==========
     n       = integer. The row dimension of the matrix A. The matrix
     
     a,ja,ia = matrix stored in Compressed Sparse Row format.
     
     lfil    = integer. The fill-in parameter. Each row of L and each row
     of U will have a maximum of lfil elements (excluding the diagonal
     element). lfil must be .ge. 0.
     
     droptol = real*8. Sets the threshold for dropping small terms in the
     factorization. See below for details on dropping strategy.
     
     iwk     = integer. The lengths of arrays alu and jlu. If the arrays
     are not big enough to store the ILU factorizations, ilut
     will stop with an error message.
     
     On return:
     ===========
     
     alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
     the L and U factors together. The diagonal (stored in
     alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
     contains the i-th row of L (excluding the diagonal entry=1)
     followed by the i-th row of U.
     
     ju      = integer array of length n containing the pointers to
     the beginning of each row of U in the matrix alu,jlu.
     
     ierr    = integer. Error message with the following meaning.
     ierr  = 0    --> successful return.
     ierr .gt. 0  --> zero pivot encountered at step number ierr.
     ierr  = -1   --> Error. input matrix may be wrong.
     (The elimination process has generated a
     row in L or U whose length is .gt.  n.)
     ierr  = -2   --> The matrix L overflows the array al.
     ierr  = -3   --> The matrix U overflows the array alu.
     ierr  = -4   --> Illegal value for lfil.
     ierr  = -5   --> zero row encountered.
     
     work arrays:
     =============
     jw      = integer work array of length 2*n.
     w       = real work array of length n+1.
     
     ----------------------------------------------------------------------
     w, ju (1:n) store the working array [1:ii-1 = L-part, ii:n = u]
     jw(n+1:2n)  stores nonzero indicators
     
     Notes:
     ------
     The diagonal elements of the input matrix must be nonzero (at least
     'structurally').
     
     ----------------------------------------------------------------------*
     ---- Dual drop strategy works as follows.                             *
     *
     1) Theresholding in L and U as set by droptol. Any element whose *
     magnitude is less than some tolerance (relative to the abs       *
     value of diagonal element in u) is dropped.                      *
     *
     2) Keeping only the largest lfil elements in the i-th row of L   *
     and the largest lfil elements in the i-th row of U (excluding    *
     diagonal elements).                                              *
     *
     Flexibility: one can use droptol=0 to get a strategy based on    *
     keeping the largest elements in each row of L and U. Taking      *
     droptol .ne. 0 but lfil=n will give the usual threshold strategy *
     (however, fill-in is then unpredictible).                        *
     ---------------------------------------------------------------------- */
    
    // locals
    INT ju0, k, j1, j2, j, ii, i, lenl, lenu, jj, jrow, jpos, NE, len;
    REAL t, s, fact, tmp;
    SHORT cinindex=0;
    REAL *w, *tnorm;
    INT  *ju, *jw;
    
    if (lfil  <  0) goto F998;
    
    ju = (INT *)fasp_mem_calloc(n, sizeof(INT));
    jw = (INT *)fasp_mem_calloc(2*n, sizeof(INT));
    w = (REAL *)fasp_mem_calloc(n+1, sizeof(REAL));
    tnorm = (REAL *)fasp_mem_calloc(n, sizeof(REAL));
    
    --jw;
    --ju;
    --w;
    --tnorm;
    --jlu;
    --alu;
    --ia;
    --ja;
    --a;
    
    if (ia[1]  ==  0) cinindex=1 ;
    
    if (cinindex)
    {
        NE = n + 1; //modify by chunsheng 2012, Sep, 1;
        for (i=1; i<=NE; ++i)  ++ia[i];
        NE = ia[n+1] - 1;
        for (i=1; i<=NE; ++i)  ++ja[i];
    }
    
    /*-----------------------------------------------------------------------
     initialize ju0 (points to next element to be added to alu,jlu)
     and pointer array.
     -----------------------------------------------------------------------*/
    ju0 = n + 2;
    jlu[1] = ju0;
    
    // initialize nonzero indicator array.
    for (j = 1; j<=n; ++j)  jw[n + j] = 0;
    
    /*-----------------------------------------------------------------------
     beginning of main loop.
     -----------------------------------------------------------------------*/
    for (ii = 1; ii <= n; ++ii ) {
        j1 = ia[ii];
        j2 = ia[ii + 1] - 1;
        tmp = 0.0;
        for ( k = j1; k<= j2; ++k) 	tmp = tmp + ABS(a[k]);
        tmp = tmp/(REAL)(j2 - j1 + 1);
        tnorm[ii] = tmp*droptol;;
    }
    
    for (ii = 1; ii<=n; ++ii) {
        j1 = ia[ii];
        j2 = ia[ii + 1] - 1;
        
        //   unpack L-part and U-part of row of A in arrays w
        lenu = 1;
        lenl = 0;
        jw[ii] = ii;
        w[ii] = 0.0;
        jw[n + ii] = ii;
        
        for(j = j1; j<=j2; ++j) {
            k = ja[j];
            t = a[j];
            if (k  <  ii) {
                ++lenl;
                jw[lenl] = k;
                w[lenl] = t;
                jw[n + k] = lenl;
            } else if (k == ii) {
                w[ii] = t;
            } else 	{
                ++lenu ;
                jpos = ii + lenu - 1;
                jw[jpos] = k;
                w[jpos] = t;
                jw[n + k] = jpos;
            }
        }
        jj = 0;
        len = 0;
        
        //     eliminate previous rows
    F150:
        ++jj;
        if (jj  >  lenl) goto F160;
        
        /*-----------------------------------------------------------------------
         in order to do the elimination in the correct order we must select
         the smallest column index among jw(k), k=jj+1, ..., lenl.
         -----------------------------------------------------------------------*/
        jrow = jw[jj];
        k = jj;
        
        /*
         determine smallest column index
         */
        for(j = jj + 1; j<=lenl; ++j)	{  //151
            if (jw[j]  <  jrow) {
                jrow = jw[j];
                k = j;
            }
        }    //151
        
        if (k  !=  jj) {
            // exchange in jw
            j = jw[jj];
            jw[jj] = jw[k];
            jw[k] = j;
            // exchange in jr
            jw[n + jrow] = jj;
            jw[n + j] = k;
            // exchange in w
            s = w[jj];
            w[jj] = w[k];
            w[k] = s;
        }
        
        // zero out element in row by setting jw(n+jrow) to zero.
        jw[n + jrow] = 0;
        
        // get the multiplier for row to be eliminated (jrow).
        fact = w[jj]*alu[jrow];
        
        if (ABS(fact)  <=  droptol) goto F150;
        
        // combine current row and row jrow
        for ( k = ju[jrow]; k <= jlu[jrow + 1] - 1; ++k) {   //203
            s = fact*alu[k];
            j = jlu[k];
            jpos = jw[n + j];
            if (j  >=  ii)  {
                //     dealing with upper part.
                if (jpos  ==  0)
                {
                    //     this is a fill-in element
                    ++lenu;
                    if (lenu  >  n) goto F995;
                    i = ii + lenu - 1;
                    jw[i] = j;
                    jw[n + j] = i;
                    w[i] = -s;
                } else 	{
                    //    this is not a fill-in element
                    w[jpos] = w[jpos] - s;
                }
            } else {
                //     dealing  with lower part.
                if (jpos  ==  0) {
                    //     this is a fill-in element
                    ++lenl;
                    if (lenl  >  n) goto F995;
                    jw[lenl] = j;
                    jw[n + j] = lenl;
                    w[lenl] = -s;
                } else 	{
                    //    this is not a fill-in element
                    w[jpos] = w[jpos] - s;
                }
            }
        }  //203
        
        /*
         store this pivot element -- (from left to right -- no danger of
         overlap with the working elements in L (pivots).
         */
        ++len;
        w[len] = fact;
        jw[len] = jrow;
        goto F150;
        
    F160:
        // reset double-pointer to zero (U-part)
        for (k = 1; k <= lenu; ++k ) jw[n + jw[ii + k - 1]] = 0;  //308
        
        // update L-matrix
        lenl = len;
        len = MIN(lenl, lfil);
        
        // sort by quick-split
        fasp_qsplit(&w[1], &jw[1], lenl, len);
        
        // store L-part
        for (k = 1; k <= len; ++k ) 	{   //204
            if (ju0  >  iwk) goto F996;
            alu[ju0] = w[k];
            jlu[ju0] = jw[k];
            ++ju0;
        }
        
        // save pointer to beginning of row ii of U
        ju[ii] = ju0;
        
        // update U-matrix -- first apply dropping strategy
        len = 0;
        for (k = 1; k <= lenu - 1; ++k) {
            //		if ( ABS(w[ii + k])  >  droptol*tnorm )
            if ( ABS(w[ii + k])  >  tnorm[ii] ) {
                ++len;
                w[ii + len] = w[ii + k];
                jw[ii + len] = jw[ii + k];
            }
        }
        
        lenu = len + 1;
        len = MIN(lenu, lfil);
        
        fasp_qsplit(&w[ii + 1], &jw[ii + 1], lenu - 1, len);
        
        // copy
        t = ABS(w[ii]);
        if (len + ju0  >  iwk) goto F997;
        for (k = ii + 1; k<=ii + len - 1; ++k)  {  //302
            jlu[ju0] = jw[k];
            alu[ju0] = w[k];
            t = t + ABS(w[k]);
            ++ju0;
        }
        
        // store inverse of diagonal element of u
        // if (w(ii) .eq. 0.0) w(ii) = (0.0001 + droptol)*tnorm
        if (w[ii]  ==  0.0) w[ii] = tnorm[ii];
        
        alu[ii] = 1.0/w[ii];
        
        // update pointer to beginning of next row of U.
        jlu[ii + 1] = ju0;
        /*-----------------------------------------------------------------------
         end main loop
         ----------------------------------------------------------------------- */
    }
    
    *nz = ju[n] - 1;
    
    if (cinindex) {
        for(i = 1; i <= *nz; ++i)  --jlu[i];
    }
    
    *ierr = 0;
    
F100:
    ++jw;
    ++ju;
    ++w;
    ++tnorm;
    ++jlu;
    ++alu;
    ++ia;
    ++ja;
    ++a;
    
    fasp_mem_free(ju);     ju    = NULL;
    fasp_mem_free(jw);     jw    = NULL;
    fasp_mem_free(w);      w     = NULL;
    fasp_mem_free(tnorm);  tnorm = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
    
F995:    // incomprehensible error. Matrix must be wrong.
    printf("### ERROR: Input matrix may be wrong. [%s]\n", __FUNCTION__);
    *ierr = -1;
    goto F100;
    
F996:    // insufficient storage in L.
    printf("### ERROR: Insufficient storage in L. [%s]\n", __FUNCTION__);
    *ierr = -2;
    goto F100;
    
F997:    // insufficient storage in U.
    printf("### ERROR: Insufficient storage in U. [%s]\n", __FUNCTION__);
    *ierr = -3;
    goto F100;
    
F998:    // illegal lfil entered.
    *ierr = -4;
    printf("### ERROR: Illegal lfil entered. [%s]\n", __FUNCTION__);
    return;
    /*----------------end-of-ilut--------------------------------------------
     -----------------------------------------------------------------------*/
}

/**
 * \fn void fasp_ilutp (INT n, REAL *a, INT *ja, INT *ia, INT lfil, REAL droptol,
 *                      REAL permtol, INT mbloc, REAL *alu, INT *jlu, INT *iperm,
 *                      INT iwk, INT *ierr, INT *nz)
 *
 * \brief Get incomplete LU factorization with pivoting dual truncations of
 *        a CSR matrix A
 *
 * \param n   row number of A
 * \param a   nonzero entries of A
 * \param ja  integer array of column for A
 * \param ia  integer array of row pointers for A
 * \param lfil  integer. The fill-in parameter. Each row of L and each row
 *              of U will have a maximum of lfil elements (excluding the diagonal
 *              element). lfil must be .ge. 0.
 * \param droptol  real*8. Sets the threshold for dropping small terms in the
 *                 factorization. See below for details on dropping strategy.
 * \param permtol  tolerance ratio used to  determne whether or not to permute
 *                 two columns.  At step i columns i and j are permuted when
 *                 abs(a(i,j))*permtol .gt. abs(a(i,i))
 *                 [0 --> never permute; good values 0.1 to 0.01]
 * \param mbloc  integer.If desired, permuting can be done only within the diagonal
 *               blocks of size mbloc. Useful for PDE problems with several
 *               degrees of freedom.. If feature not wanted take mbloc=n.
 * \param alu  matrix stored in Modified Sparse Row (MSR) format containing
 *             the L and U factors together. The diagonal (stored in
 *             alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
 *             contains the i-th row of L (excluding the diagonal entry=1)
 *             followed by the i-th row of U.
 * \param jlu  integer array of length n containing the pointers to
 *             the beginning of each row of U in the matrix alu,jlu.
 * \param iperm permutation arrays
 * \param iwk  integer. The lengths of arrays alu and jlu. If the arrays
 *             are not big enough to store the ILU factorizations, ilut
 *             will stop with an error message.
 * \param ierr  integer pointer. Return error message with the following meaning.
 *                0  --> successful return.
 *               >0  --> zero pivot encountered at step number ierr.
 *               -1  --> Error. input matrix may be wrong.
 *                       (The elimination process has generated a
 *                       row in L or U whose length is .gt.  n.)
 *               -2  --> The matrix L overflows the array al.
 *               -3  --> The matrix U overflows the array alu.
 *               -4  --> Illegal value for lfil.
 *               -5  --> zero row encountered.
 * \param nz  integer pointer. Return number of nonzero entries for alu and jlu
 *
 * \note:  All the diagonal elements of the input matrix must be nonzero.
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
void fasp_ilutp (INT    n,
                 REAL  *a,
                 INT   *ja,
                 INT   *ia,
                 INT    lfil,
                 REAL   droptol,
                 REAL   permtol,
                 INT    mbloc,
                 REAL  *alu,
                 INT   *jlu,
                 INT   *iperm,
                 INT    iwk,
                 INT   *ierr,
                 INT   *nz)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    /*----------------------------------------------------------------------*
     *** ILUTP preconditioner -- ILUT with pivoting  ***              *
     incomplete LU factorization with dual truncation mechanism       *
     ----------------------------------------------------------------------*
     author Yousef Saad *Sep 8, 1993 -- Latest revision, August 1996. *
     ----------------------------------------------------------------------*
     on entry:
     ==========
     n       = integer. The dimension of the matrix A.
     
     a,ja,ia = matrix stored in Compressed Sparse Row format.
     ON RETURN THE COLUMNS OF A ARE PERMUTED. SEE BELOW FOR
     DETAILS.
     
     lfil    = integer. The fill-in parameter. Each row of L and each row
     of U will have a maximum of lfil elements (excluding the
     diagonal element). lfil must be .ge. 0.
     ** WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO
     EARLIER VERSIONS.
     
     droptol = real*8. Sets the threshold for dropping small terms in the
     factorization. See below for details on dropping strategy.
     
     lfil    = integer. The fill-in parameter. Each row of L and
     each row of U will have a maximum of lfil elements.
     
     permtol = tolerance ratio used to  determne whether or not to permute
     two columns.  At step i columns i and j are permuted when
     
     abs(a(i,j))*permtol .gt. abs(a(i,i))
     
     [0 --> never permute; good values 0.1 to 0.01]
     
     mbloc   = if desired, permuting can be done only within the diagonal
     blocks of size mbloc. Useful for PDE problems with several
     degrees of freedom.. If feature not wanted take mbloc=n.
     
     iwk     = integer. The lengths of arrays alu and jlu. If the arrays
     are not big enough to store the ILU factorizations, ilut
     will stop with an error message.
     
     On return:
     ===========
     
     alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
     the L and U factors together. The diagonal (stored in
     alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
     contains the i-th row of L (excluding the diagonal entry=1)
     followed by the i-th row of U.
     
     ju      = integer array of length n containing the pointers to
     the beginning of each row of U in the matrix alu,jlu.
     
     iperm   = contains the permutation arrays.
     iperm(1:n) = old numbers of unknowns
     iperm(n+1:2*n) = reverse permutation = new unknowns.
     
     ierr    = integer. Error message with the following meaning.
     ierr  = 0    --> successful return.
     ierr .gt. 0  --> zero pivot encountered at step number ierr.
     ierr  = -1   --> Error. input matrix may be wrong.
     (The elimination process has generated a
     row in L or U whose length is .gt.  n.)
     ierr  = -2   --> The matrix L overflows the array al.
     ierr  = -3   --> The matrix U overflows the array alu.
     ierr  = -4   --> Illegal value for lfil.
     ierr  = -5   --> zero row encountered.
     
     work arrays:
     =============
     jw      = integer work array of length 2*n.
     w       = real work array of length n
     
     IMPORTANR NOTE:
     --------------
     TO AVOID PERMUTING THE SOLUTION VECTORS ARRAYS FOR EACH LU-SOLVE,
     THE MATRIX A IS PERMUTED ON RETURN. [all column indices are
     changed]. SIMILARLY FOR THE U MATRIX.
     To permute the matrix back to its original state use the loop:
     
     do k=ia(1), ia(n+1)-1
     ja(k) = iperm(ja(k))
     enddo
     
     -----------------------------------------------------------------------*/
    
    // local variables
    INT k, i, j, jrow, ju0, ii, j1, j2, jpos, len, imax, lenu, lenl, jj, icut,NE;
    REAL s, tmp, tnorm, xmax, xmax0, fact, t;
    SHORT cinindex=0;
    REAL  *w;
    INT  *ju, *jw;
    
    if (lfil  <  0) goto F998;
    
    ju = (INT *) fasp_mem_calloc(n, sizeof(INT));
    jw = (INT *) fasp_mem_calloc(2*n, sizeof(INT));
    w  = (REAL *)fasp_mem_calloc(n+1, sizeof(REAL));
    
    --ju;
    --jw;
    --iperm;
    --w;
    --jlu;
    --alu;
    --ia;
    --ja;
    --a;
    
    /*-----------------------------------------------------------------------
     shift index for C routines
     -----------------------------------------------------------------------*/
    if (ia[1]  ==  0) cinindex=1 ;
    
    if (cinindex)
    {
        NE = n + 1; //modify by chunsheng 2012, Sep, 1;
        for (i=1; i<=NE; ++i)  ++ia[i];
        NE = ia[n+1] - 1;
        for (i=1; i<=NE; ++i)  ++ja[i];
    }
    
    /*-----------------------------------------------------------------------
     initialize ju0 (points to next element to be added to alu,jlu)
     and pointer array.
     -----------------------------------------------------------------------*/
    ju0 = n + 2;
    jlu[1] = ju0;
    
    
    //	 integer double pointer array.
    for ( j = 1; j <= n; ++j ) { //1
        jw[n + j] = 0;
        iperm[j] = j;
        iperm[n + j] = j;
    } //1
    
    /*-----------------------------------------------------------------------
     beginning of main loop.
     -----------------------------------------------------------------------*/
    for (ii = 1; ii <= n; ++ii ) 	{ //500
        j1 = ia[ii];
        j2 = ia[ii + 1] - 1;
        
        tnorm = 0.0;
        for (k = j1; k <= j2; ++k ) 	tnorm = tnorm + ABS( a[k] ); //501
        if (tnorm  ==  0.0) goto F999;
        tnorm = tnorm/(REAL)(j2 - j1 + 1);
        
        // unpack L-part and U-part of row of A in arrays  w  --
        lenu = 1;
        lenl = 0;
        jw[ii] = ii;
        w[ii] = 0.0;
        jw[n + ii] = ii;
        //
        for (j = j1; j <= j2; ++j )  { // 170
            k = iperm[n + ja[j]];
            t = a[j];
            if (k  <  ii) {
                ++lenl;
                jw[lenl] = k;
                w[lenl] = t;
                jw[n + k] = lenl;
            } else if (k  ==  ii) {
                w[ii] = t;
            } else {
                ++lenu;
                jpos = ii + lenu - 1;
                jw[jpos] = k;
                w[jpos] = t;
                jw[n + k] = jpos;
            }
        }  //170
        
        jj = 0;
        len = 0;
        
        
        // eliminate previous rows
    F150:
        ++jj;
        if (jj  >  lenl) goto F160;
        
        /*-----------------------------------------------------------------------
         in order to do the elimination in the correct order we must select
         the smallest column index among jw(k), k=jj+1, ..., lenl.
         -----------------------------------------------------------------------*/
        jrow = jw[jj];
        k = jj;
        
        // determine smallest column index
        for (j = jj + 1; j <= lenl; ++j) {  //151
            if (jw[j]  <  jrow) {
                jrow = jw[j];
                k = j;
            }
        }
        
        if (k  !=  jj) 	{
            // exchange in jw
            j = jw[jj];
            jw[jj] = jw[k];
            jw[k] = j;
            // exchange in jr
            jw[n + jrow] = jj;
            jw[n + j] = k;
            // exchange in w
            s = w[jj];
            w[jj] = w[k];
            w[k] = s;
        }
        
        // zero out element in row by resetting jw(n+jrow) to zero.
        jw[n + jrow] = 0;
        
        // get the multiplier for row to be eliminated: jrow
        fact = w[jj]*alu[jrow];
        
        // drop term if small
        if (ABS(fact)  <=  droptol) goto F150;
        
        // combine current row and row jrow
        
        for ( k = ju[jrow]; k <= jlu[jrow + 1] - 1; ++k ) {  //203
            s = fact*alu[k];
            // new column number
            j = iperm[n + jlu[k]];
            jpos = jw[n + j];
            if (j  >=  ii) {
                // dealing with upper part.
                if (jpos  ==  0) {
                    //	 this is a fill-in element
                    ++lenu;
                    i = ii + lenu - 1;
                    if (lenu  >  n) goto F995;
                    jw[i] = j;
                    jw[n + j] = i;
                    w[i] = -s;
                } else {
                    //     no fill-in element --
                    w[jpos] = w[jpos] - s;
                }
                
            } else {
                // dealing with lower part.
                if (jpos  ==  0) {
                    //	 this is a fill-in element
                    ++lenl;
                    if (lenl  >  n) goto F995;
                    jw[lenl] = j;
                    jw[n + j] = lenl;
                    w[lenl] = -s;
                } else {
                    //	 this is not a fill-in element
                    w[jpos] = w[jpos] - s;
                }
            }
        }  //203
        
        /*
         store this pivot element -- (from left to right -- no danger of
         overlap with the working elements in L (pivots).
         */
        
        ++len;
        w[len] = fact;
        jw[len] = jrow;
        goto F150;
        
    F160:
        // reset double-pointer to zero (U-part)
        for ( k = 1; k <= lenu; ++k ) jw[n + jw[ii + k - 1]] = 0;  //308
        
        // update L-matrix
        lenl = len;
        len = MIN(lenl, lfil);
        
        // sort by quick-split
        fasp_qsplit(&w[1], &jw[1], lenl, len);
        
        // store L-part -- in original coordinates ..
        for ( k = 1; k <= len; ++k ) {  // 204
            if (ju0  >  iwk) goto F996;
            alu[ju0] = w[k];
            jlu[ju0] = iperm[jw[k]];
            ++ju0;
        }  //204
        
        // save pointer to beginning of row ii of U
        ju[ii] = ju0;
        
        // update U-matrix -- first apply dropping strategy
        len = 0;
        for(k = 1; k <= lenu - 1; ++k ) {
            if ( ABS(w[ii + k])  >  droptol*tnorm) {
                ++len;
                w[ii + len] = w[ii + k];
                jw[ii + len] = jw[ii + k];
            }
        }
        
        lenu = len + 1;
        len = MIN(lenu, lfil);
        fasp_qsplit(&w[ii + 1], &jw[ii + 1], lenu-1, len);
        
        // determine next pivot --
        imax = ii;
        xmax = ABS(w[imax]);
        xmax0 = xmax;
        icut = ii - 1 + mbloc - (ii - 1)%mbloc;
        
        for ( k = ii + 1; k <= ii + len - 1; ++k ) {
            t = ABS(w[k]);
            if ((t  >  xmax) && (t*permtol  >  xmax0) && (jw[k]  <=  icut)) {
                imax = k;
                xmax = t;
            }
        }
        
        // exchange w's
        tmp = w[ii];
        w[ii] = w[imax];
        w[imax] = tmp;
        
        // update iperm and reverse iperm
        j = jw[imax];
        i = iperm[ii];
        iperm[ii] = iperm[j];
        iperm[j] = i;
        
        // reverse iperm
        iperm[n + iperm[ii]] = ii;
        iperm[n + iperm[j]] = j;
        
        //-----------------------------------------------------------------------
        if (len + ju0  >  iwk) goto F997;
        
        
        // copy U-part in original coordinates
        for ( k = ii + 1; k <= ii + len - 1; ++k ) { //302
            jlu[ju0] = iperm[jw[k]];
            alu[ju0] = w[k];
            ++ju0;
        }
        
        // store inverse of diagonal element of u
        if (w[ii]  ==  0.0) w[ii] = (1.0e-4 + droptol)*tnorm;
        alu[ii] = 1.0/w[ii];
        
        // update pointer to beginning of next row of U.
        jlu[ii + 1] = ju0;
        
        /*-----------------------------------------------------------------------
         end main loop
         -----------------------------------------------------------------------*/
    }  //500
    
    // permute all column indices of LU ...
    for ( k = jlu[1]; k <= jlu[n + 1] - 1; ++k ) 	jlu[k] = iperm[n + jlu[k]];
    
    // ...and of A
    for ( k = ia[1]; k <= ia[n + 1] - 1; ++k )	ja[k] = iperm[n + ja[k]];
    
    *nz = ju[n]- 1;
    
    if (cinindex)  {
        for (i = 1; i <= *nz; ++i ) --jlu[i];
    }
    
    *ierr = 0;
    
F100:
    ++jw;
    ++ju;
    ++iperm;
    ++w;
    ++jlu;
    ++alu;
    ++ia;
    ++ja;
    ++a;
    
    fasp_mem_free(ju);  ju = NULL;
    fasp_mem_free(jw);  jw = NULL;
    fasp_mem_free(w);   w  = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
    
F995:    // incomprehensible error. Matrix must be wrong.
    printf("### ERROR: Input matrix may be wrong. [%s]\n", __FUNCTION__);
    *ierr = -1;
    goto F100;
    
F996:    // insufficient storage in L.
    printf("### ERROR: Insufficient storage in L. [%s]\n", __FUNCTION__);
    *ierr = -2;
    goto F100;
    
F997:    // insufficient storage in U.
    printf("### ERROR: Insufficient storage in U. [%s]\n", __FUNCTION__);
    *ierr = -3;
    goto F100;
    
F998:    // illegal lfil entered.
    printf("### ERROR: Illegal lfil entered. [%s]\n", __FUNCTION__);
    *ierr = -4;
    // goto F100;
    return;
    
F999:    // zero row encountered
    printf("### ERROR: Zero row encountered. [%s]\n", __FUNCTION__);
    *ierr = -5;
    goto F100;
    //----------------end-of-ilutp-------------------------------------------
}

/**
 * \fn void fasp_symbfactor (INT n, INT *colind, INT *rwptr, INT levfill,
 *                           INT nzmax, INT *nzlu, INT *ijlu, INT *uptr, INT *ierr)
 *
 * \brief Symbolic factorization of a CSR matrix A in compressed sparse row format,
 *		    with resulting factors stored in a single MSR data structure.
 *
 * \param n   row number of A
 * \param colind  integer array of column for A
 * \param rwptr  integer array of row pointers for A
 * \param levfill  integer. Level of fill-in allowed
 * \param nzmax  integer. The maximum number of nonzero entries in the
 *		           approximate factorization of a.  This is the amount of storage
 *		           allocated for ijlu.
 * \param nzlu  integer pointer. Return number of nonzero entries for alu and jlu
 * \param ijlu  integer array of length nzlu containing pointers to delimit rows
 *              and specify column number for stored elements of the approximate
 *              factors of A.  the L and U factors are stored as one matrix.
 * \param uptr  integer array of length n containing the pointers to upper trig matrix
 * \param ierr  integer pointer. Return error message with the following meaning.
 *                0  --> successful return.
 *                1  --> not enough storage; check mneed.
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
void fasp_symbfactor (INT   n,
                      INT  *colind,
                      INT  *rwptr,
                      INT   levfill,
                      INT   nzmax,
                      INT  *nzlu,
                      INT  *ijlu,
                      INT  *uptr,
                      INT  *ierr)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif
    
    /**
     ========================================================================
     *
     Symbolic factorization of a matrix in compressed sparse row format,  *
     with resulting factors stored in a single MSR data structure.      *
     *
     This routine uses the CSR data structure of A in two integer vectors *
     colind, rwptr to set up the data structure for the ILU(levfill)    *
     factorization of A in the integer vectors ijlu and uptr.  Both L   *
     and U are stored in the same structure, and uptr(i) is the pointer *
     to the beginning of the i-th row of U in ijlu.         *
     *
     ========================================================================
     *
     Method Used                                                        *
     ===========                                                        *
     *
     The implementation assumes that the diagonal entries are     *
     nonzero, and remain nonzero throughout the elimination       *
     process.  The algorithm proceeds row by row.  When computing     *
     the sparsity pattern of the i-th row, the effect of row      *
     operations from previous rows is considered.  Only those     *
     preceding rows j for which (i,j) is nonzero need be considered,  *
     since otherwise we would not have formed a linear combination    *
     of rows i and j.                         *
     *
     The method used has some variations possible.  The definition    *
     of ILU(s) is not well specified enough to get a factorization    *
     that is uniquely defined, even in the sparsity pattern that      *
     results.  For s = 0 or 1, there is not much variation, but for   *
     higher levels of fill the problem is as follows:  Suppose        *
     during the decomposition while computing the nonzero pattern     *
     for row i the following principal submatrix is obtained:     *
     _______________________                     *
     |          |           |                    *
     |          |           |                    *
     |  j,j     |    j,k    |                    *
     |          |           |                    *
     |__________|___________|                    *
     |          |           |                    *
     |          |           |                    *
     |  i,j     |    i,k    |                    *
     |          |           |                    *
     |__________|___________|                    *
     *
     Furthermore, suppose that entry (i,j) resulted from an earlier   *
     fill-in and has level s1, and (j,k) resulted from an earlier     *
     fill-in and has level s2:                        *
     _______________________                     *
     |          |           |                    *
     |          |           |                    *
     | level 0  | level s2  |                    *
     |          |           |                    *
     |__________|___________|                    *
     |          |           |                    *
     |          |           |                    *
     | level s1 |           |                    *
     |          |           |                    *
     |__________|___________|                    *
     *
     When using A(j,j) to annihilate A(i,j), fill-in will be incurred *
     in A(i,k).  How should its level be defined?  It would not be    *
     operated on if A(i,j) or A(j,m) had not been filled in.  The     *
     version used here is to define its level as s1 + s2 + 1.  However,   *
     other reasonable choices would have been min(s1,s2) or max(s1,s2).   *
     Using the sum gives a more conservative strategy in terms of the *
     growth of the number of nonzeros as s increases.         *
     *
     levels(n+2:nzlu    ) stores the levels from previous rows,       *
     that is, the s2's above.  levels(1:n) stores the fill-levels     *
     of the current row (row i), which are the s1's above.        *
     levels(n+1) is not used, so levels is conformant with MSR format.    *
     *
     Vectors used:                            *
     =============                            *
     *
     lastcol(n):                              *
     The integer lastcol(k) is the row index of the last row     *
     to have a nonzero in column k, including the current        *
     row, and fill-in up to this point.  So for the matrix       *
     *
     |--------------------------|              *
     | 11   12           15     |              *
     | 21   22                26|              *
     |      32  33   34         |              *
     | 41       43   44         |              *
     |      52       54  55   56|              *
     |      62                66|              *
     ---------------------------               *
     *
     after step 1, lastcol() = [1  0  0  0  1  0]      *
     after step 2, lastcol() = [2  2  0  0  2  2]      *
     after step 3, lastcol() = [2  3  3  3  2  3]      *
     after step 4, lastcol() = [4  3  4  4  4  3]      *
     after step 5, lastcol() = [4  5  4  5  5  5]      *
     after step 6, lastcol() = [4  6  4  5  5  6]      *
     *
     Note that on step 2, lastcol(5) = 2 because there is a   *
     fillin position (2,5) in the matrix.  lastcol() is used  *
     to determine if a nonzero occurs in column j because        *
     it is a nonzero in the original matrix, or was a fill.      *
     *
     rowll(n):                                *
     The integer vector rowll is used to keep a linked list of   *
     the nonzeros in the current row, allowing fill-in to be     *
     introduced sensibly.  rowll is initialized with the     *
     original nonzeros of the current row, and then sorted       *
     using a shell sort.  A pointer called head              *
     (what ingenuity) is  initialized.  Note that at any     *
     point rowll may contain garbage left over from previous     *
     rows, which the linked list structure skips over.       *
     For row 4 of the matrix above, first rowll is set to        *
     rowll() = [3  1  2  5  -  -], where - indicates any integer.    *
     Then the vector is sorted, which yields             *
     rowll() = [1  2  3  5  -  -].  The vector is then expanded  *
     to linked list form by setting head = 1  and                *
     rowll() = [2  3  5  -  7  -], where 7 indicates termination.    *
     *
     ijlu(nzlu):                              *
     The returned nonzero structure for the LU factors.      *
     This is built up row by row in MSR format, with both L      *
     and U stored in the data structure.  Another vector, uptr(n),   *
     is used to give pointers to the beginning of the upper      *
     triangular part of the LU factors in ijlu.          *
     *
     levels(n+2:nzlu):                            *
     This vector stores the fill level for each entry from       *
     all the previous rows, used to compute if the current entry *
     will exceed the allowed levels of fill.  The value in       *
     levels(m) is added to the level of fill for the element in  *
     the current row that is being reduced, to figure if         *
     a column entry is to be accepted as fill, or rejected.      *
     See the method explanation above.               *
     *
     levels(1:n):                             *
     This vector stores the fill level number for the current    *
     row's entries.  If they were created as fill elements       *
     themselves, this number is added to the corresponding       *
     entry in levels(n+2:nzlu) to see if a particular column     *
     entry will                          *
     be created as new fill or not.  NOTE: in practice, the      *
     value in levels(1:n) is one larger than the "fill" level of *
     the corresponding row entry, except for the diagonal        *
     entry.  That is why the accept/reject test in the code      *
     is "if (levels(j) + levels(m) .le. levfill + 1)".       *
     *
     ========================================================================
     
     on entry:
     ==========
     n       = The order of the matrix A.
     ija     = Integer array. Matrix A stored in modified sparse row format.
     levfill = Integer. Level of fill-in allowed.
     nzmax   = Integer. The maximum number of nonzero entries in the
     approximate factorization of a.  This is the amount of storage
     allocated for ijlu.
     
     on return:
     ===========
     
     nzlu   = The actual number of entries in the approximate factors, plus one.
     ijlu   = Integer array of length nzlu containing pointers to
     delimit rows and specify column number for stored
     elements of the approximate factors of a.  the l
     and u factors are stored as one matrix.
     uptr   = Integer array of length n containing the pointers to upper trig
     matrix
     
     ierr is an error flag:
     ierr  = -i --> near zero pivot in step i
     ierr  = 0  --> all's OK
     ierr  = 1  --> not enough storage; check mneed.
     ierr  = 2  --> illegal parameter
     
     mneed   = contains the actual number of elements in ldu, or the amount
     of additional storage needed for ldu
     
     work arrays:
     =============
     lastcol    = integer array of length n containing last update of the
     corresponding column.
     levels     = integer array of length n containing the level of
     fill-in in current row in its first n entries, and
     level of fill of previous rows of U in remaining part.
     rowll      = integer array of length n containing pointers to implement a
     linked list for the fill-in elements.
     
     external functions:
     ====================
     ifix, float, min0, srtr
     
     ======================================================================== */
    
    INT icolindj, ijlum, i, j, k, m, ibegin, iend, Ujbeg, Ujend,NE;
    INT head, prev, lm, actlev, lowct, k1, k2, levp1, lmk, nzi, rowct;
    SHORT cinindex=0;
    INT *rowll, *lastcol, *levels;
    
    rowll = (INT *)fasp_mem_calloc(n, sizeof(INT));
    lastcol = (INT *)fasp_mem_calloc(n, sizeof(INT));
    levels = (INT *)fasp_mem_calloc(nzmax, sizeof(INT));
    
    //========================================================================
    //       Beginning of Executable Statements
    //========================================================================
    
    /*-----------------------------------------------------------------------
     shift index for C routines
     -----------------------------------------------------------------------*/
    --rowll;
    --lastcol;
    --levels;
    --colind;
    --rwptr;
    --ijlu;
    --uptr;
    
    if (rwptr[1]  ==  0) cinindex=1 ;
    if (cinindex) {
        NE = n + 1;
        for (i=1; i<=NE; ++i)  ++rwptr[i];
        NE = rwptr[n+1] - 1;
        for (i=1; i<=NE; ++i)  ++colind[i];
    }
    
    // --------------------------------------------------------------
    // Because the first row of the factor contains no strictly lower
    // triangular parts (parts of L), uptr(1) = ijlu(1) = n+2:
    // --------------------------------------------------------------
    ijlu[1] = n + 2;
    uptr[1] = n + 2;
    
    // --------------------------------------------------------
    // The storage for the nonzeros of LU must be at least n+1,
    // for a diagonal matrix:
    // --------------------------------------------------------
    *nzlu = n + 1;
    
    //  --------------------------------------------------------------------
    //  Number of allowed levels plus 1; used for the test of accept/reject.
    //  See the notes about the methodology above.
    //  --------------------------------------------------------------------
    levp1 = levfill + 1;
    
    //  -------------------------------------------------------------
    //  Initially, for all columns there were no nonzeros in the rows
    //  above, because there are no rows above the first one.
    //  -------------------------------------------------------------
    for (i = 1; i<=n; ++i) lastcol[i] = 0;
    
    //  -------------------
    //  Proceed row by row:
    //  -------------------
    for (i = 1; i <= n; ++i) { // 100
        
        // ----------------------------------------------------------
        // Because the matrix diagonal entry is nonzero, the level of
        // fill for that diagonal entry is zero:
        // ----------------------------------------------------------
        levels[i] = 0;
        
        // ----------------------------------------------------------
        // ibegin and iend are the beginning of rows i and i+1, resp.
        // ----------------------------------------------------------
        ibegin = rwptr[i];
        iend = rwptr[i + 1];
        
        //  -------------------------------------------------------------
        //  Number of offdiagonal nonzeros in the original matrix's row i
        //  -------------------------------------------------------------
        nzi = iend - ibegin;
        
        //  --------------------------------------------------------
        //  If only the diagonal entry in row i is nonzero, skip the
        //  fancy stuff; nothing need be done:
        //  --------------------------------------------------------
        if (nzi  >  1) {
            //  ----------------------------------------------------------
            //  Decrement iend, so that it can be used as the ending index
            //  in icolind of row i:
            //  ----------------------------------------------------------
            iend = iend - 1;
            
            //  ---------------------------------------------------------
            //  rowct keeps count of the number of nondiagonal entries in
            //  the current row:
            //  ---------------------------------------------------------
            rowct = 0;
            
            //  ------------------------------------------------------------
            //  For nonzeros in the current row from the original matrix A,
            //  set lastcol to be the current row number, and the levels of
            //  the entry to be 1.  Note that this is really the true level
            //  of the element, plus 1.  At the same time, load up the work
            //  array rowll with the column numbers for the original entries
            //  from row i:
            //  ------------------------------------------------------------
#if DEBUG_MODE > 0
            printf("### DEBUG: %s %d row\n", __FUNCTION__, i);
#endif

            for ( j = ibegin; j <= iend; ++j) {
                icolindj = colind[j];
                lastcol[icolindj] = i;
                if (icolindj !=  i) {
                    levels[icolindj] = 1;
                    rowct = rowct + 1;
                    rowll[rowct] = icolindj;
                }
#if DEBUG_MODE > 0
                printf("### DEBUG: %d\n", icolindj);
#endif
            }
            
            //  ---------------------------------------------------------
            //  Sort the entries in rowll, so that the row has its column
            //  entries in increasing order.
            //  ---------------------------------------------------------
            fasp_sortrow(nzi - 1, &rowll[1]);
            
            //check col index
            fasp_check_col_index(i, nzi-1, &rowll[1]);
            //  ---------------------------------------------------------
            //  Now set up rowll as a linked list containing the original
            //  nonzero column numbers, as described in the methods section:
            //  ---------------------------------------------------------
            head = rowll[1];
            k1 = n + 1;
            for (j = nzi - 1; j >= 1;  --j) {
                k2 = rowll[j];
                rowll[k2] = k1;
                k1 = k2;
            }
            
            //  ------------------------------------------------------------
            //  Increment count of nonzeros in the LU factors by the number
            //  of nonzeros in the original matrix's row i.  Further
            //  incrementing will be necessary if any fill-in actually occurs
            //  ------------------------------------------------------------
            *nzlu = *nzlu + nzi - 1;
            
            //   ------------------------------------------------------------
            //   The integer j will be used as a pointer to track through the
            //   linked list rowll:
            //   ------------------------------------------------------------
            j = head;
            
            //   ------------------------------------------------------------
            //   The integer lowct is used to keep count of the number of
            //   nonzeros in the current row's strictly lower triangular part,
            //   for setting uptr pointers to indicate where in ijlu the upperc
            //   triangular part starts.
            //   ------------------------------------------------------------
            lowct = 0;
            
            //   ------------------------------------------------------------
            //   Fill-in could only have resulted from rows preceding row i,
            //   so we only need check those rows with index j < i.
            //   Furthermore, if the current row has a zero in column j,
            //   there is no need to check the preceding rows; there clearly
            //   could not be any fill-in from those rows to this entry.
            //   ------------------------------------------------------------
            while (j  <  i) {  //80
                //  ------------------------------------------------------------
                //  Increment lower triangular part count, since in this case
                //  (j<i) we got another entry in L:
                //  ------------------------------------------------------------
                lowct = lowct + 1;
                
                //   ---------------------------------------------------------
                //   If the fill level is zero, there is no way to get fill in
                //   occuring.
                //   ---------------------------------------------------------
                if (levfill !=  0) {
                    
                    //   -----------------------------------------------------
                    //   Ujbeg is beginning index of strictly upper triangular
                    //   part of U's j-th row, and Ujend is the ending index
                    //   of it, in ijlu().
                    //   -----------------------------------------------------
                    Ujbeg = uptr[j];
                    Ujend = ijlu[j + 1] - 1;
                    
                    //   -----------------------------------------------------
                    //   Need to set pointer to previous entry before working
                    //   segment of rowll, because if fill occurs that will be
                    //   a moving segment.
                    //   -----------------------------------------------------
                    prev = j;
                    
                    //  -----------------------------------------------------
                    //  lm is the next nonzero pointer in linked list rowll:
                    //  -----------------------------------------------------
                    lm = rowll[j];
                    
                    //  -------------------------------------------------------
                    //  lmk is the fill level in this row, caused by
                    //  eliminating column entry j.  That is, level s1 from the
                    //  methodology explanation above.
                    //  -------------------------------------------------------
                    lmk = levels[j];
                    
                    //  -------------------------------------------------------
                    //  Now proceed through the j-th row of U, because in the
                    //  elimination we add a multiple of it to row i to zero
                    //  out entry (i,j).  If a column entry in row j of U is
                    //  zero, there is no need to worry about fill, because it
                    //  cannot cause a fill in the corresponding entry of row i
                    //  -------------------------------------------------------
                    for (m = Ujbeg; m <= Ujend; ++m) { //60
                        //  ----------------------------------------------------
                        //  ijlum is the column number of the current nonzero in
                        //  row j of U:
                        //  ----------------------------------------------------
                        ijlum = ijlu[m];
                        
                        //  ---------------------------------------------------
                        //  actlev is the actual level (plus 1) of column entry
                        //  j in row i, from summing the level contributions
                        //  s1 and s2 as explained in the methods section.
                        //  Note that the next line could reasonably be
                        //  replaced by, e.g., actlev = max(lmk, levels(m)),
                        //  but this would cause greater fill-in:
                        //  ---------------------------------------------------
                        actlev = lmk + levels[m];
                        
                        //  ---------------------------------------------------
                        //  If lastcol of the current column entry in U is not
                        //  equal to the current row number i, then the current
                        //  row has a zero in column j, and the earlier row j
                        //  in U has a nonzero, so possible fill can occur.
                        //  ---------------------------------------------------
                        if (lastcol[ijlum]  !=  i) {
                            
                            //  --------------------------------------------------
                            //  If actlev < levfill + 1, then the new entry has an
                            //  acceptable fill level and needs to be added to the
                            //  data structure.
                            //  --------------------------------------------------
                            if (actlev  <=  levp1) {
                                
                                //  -------------------------------------------
                                //  Since the column entry ijlum in the current
                                //  row i is to be filled, we need to update
                                //  lastcol for that column number.  Also, the
                                //  level number of the current entry needs to be
                                //  set to actlev.  Note that when we finish
                                //  processing this row, the n-vector levels(1:n)
                                //  will be copied over to the corresponding
                                //  trailing part of levels, so that it can be
                                //  used in subsequent rows:
                                //  -------------------------------------------
                                lastcol[ijlum] = i;
                                levels[ijlum] = actlev;
                                
                                //  -------------------------------------------
                                //  Now find location in the linked list rowll
                                //  where the fillin entry should be placed.
                                //  Chase through the linked list until the next
                                //  nonzero column is to the right of the fill
                                //  column number.
                                //  -------------------------------------------
                                while (lm  <=  ijlum) { //50
                                    prev = lm;
                                    lm = rowll[lm];
                                }  //50
                                
                                //  -------------------------------------------
                                //  Insert new entry into the linked list for
                                //  row i, and increase the nonzero count for LU
                                //  -------------------------------------------
                                rowll[prev] = ijlum;
                                rowll[ijlum] = lm;
                                prev = ijlum;
                                *nzlu = *nzlu + 1;
                            }
                            
                            //  -------------------------------------------------
                            //  Else clause is for when lastcol(ijlum) = i.  In
                            //  this case, the current column has a nonzero, but
                            //  it resulted from an earlier fill-in or from an
                            //  original matrix entry.  In this case, need to
                            //  update the level number for this column to be the
                            //  smaller of the two possible fill contributors,
                            //  the current fill number or the computed one from
                            //  updating this entry from a previous row.
                            //  -------------------------------------------------
                        } else 	{
                            levels[ijlum] = MIN(levels[ijlum], actlev);
                        }
                        
                        //  -------------------------------------------------
                        //  Now go and pick up the next column entry from row
                        //  j of U:
                        //  -------------------------------------------------
                        
                    } //60
                    // -------------------------------------------
                    // End if clause for levfill not equal to zero
                    // -------------------------------------------
                }
                
                // ------------------------------------------------------
                // Pick up next nonzero column index from the linked
                // list, and continue processing the i-th row's nonzeros.
                // This ends the first while loop (j < i).
                // ------------------------------------------------------
                j = rowll[j];
            }  //80
            
            //  ---------------------------------------------------------
            //  Check to see if we have exceeded the allowed memory
            //  storage before storing the results of computing row i's
            //  sparsity pattern into the ijlu and uptr data structures.
            //  ---------------------------------------------------------
            if (*nzlu  >  nzmax) {
                printf("### ERROR: More storage needed! [%s]\n", __FUNCTION__);
                *ierr = 1;
                goto F100;
            }
            
            // ---------------------------------------------------------
            // Storage is adequate, so update ijlu data structure.
            // Row i ends at nzlu + 1:
            // ---------------------------------------------------------
            ijlu[i + 1] = *nzlu + 1;
            
            //  ---------------------------------------------------------
            //  ... and the upper triangular part of LU begins at
            //  lowct entries to right of where row i begins.
            //  ---------------------------------------------------------
            uptr[i] = ijlu[i] + lowct;
            
            //  -----------------------------------------------------
            //  Now chase through linked list for row i, recording
            //  information into ijlu.  At same time, put level data
            //  into the levels array for use on later rows:
            //  -----------------------------------------------------
            j = head;
            k1 = ijlu[i];
            for (k = k1; k <= *nzlu; ++k) {
                ijlu[k] = j;
                levels[k] = levels[j];
                j = rowll[j];
            }
            
        } else 	{
            
            // ---------------------------------------------------------
            // This else clause ends the (nzi > 1) if.  If nzi = 1, then
            // the update of ijlu and uptr is trivial:
            // ---------------------------------------------------------
            ijlu[i + 1] = *nzlu + 1;
            uptr[i] = ijlu[i];
        }
        
        // ----------------------------------------------
        // And you thought we would never get through....
        // ----------------------------------------------
    }  //100
    
    if (cinindex) {
        for ( i = 1; i <= *nzlu; ++i ) --ijlu[i];
        for ( i = 1; i <= n; ++i )     --uptr[i];
        NE = rwptr[n + 1] - 1;
        for ( i = 1; i <= NE; ++i )    --colind[i];
        NE = n + 1;
        for ( i = 1; i <= NE; ++i )    --rwptr[i];
    }
    
    *ierr = 0;
    
F100:
    ++rowll;
    ++lastcol;
    ++levels;
    ++colind;
    ++rwptr;
    ++ijlu;
    ++uptr;

    fasp_mem_free(rowll);    rowll   = NULL;
    fasp_mem_free(lastcol);  lastcol = NULL;
    fasp_mem_free(levels);   levels  = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return;
    //======================== End of symbfac ==============================
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void fasp_qsplit (REAL *a, INT *ind, INT n, INT ncut)
 *
 * \brief Get  a quick-sort split of a real array
 *
 * \param a  a real array. on output a(1:n) is permuted such that
 *           its elements satisfy: abs(a(i)) .ge. abs(a(ncut)) for
 *           i .lt. ncut and abs(a(i)) .le. abs(a(ncut)) for i .gt. ncut.
 * \param ind  is an integer array which permuted in the same way as a(*).
 * \param n  size of array a.
 * \param ncut  integer.
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
static void fasp_qsplit (REAL   *a,
                         INT    *ind,
                         INT     n,
                         INT     ncut)
{
    /*-----------------------------------------------------------------------
     does a quick-sort split of a real array.
     on input a(1:n). is a real array
     on output a(1:n) is permuted such that its elements satisfy:
     abs(a(i)) .ge. abs(a(ncut)) for i .lt. ncut and
     abs(a(i)) .le. abs(a(ncut)) for i .gt. ncut
     ind(1:n) is an integer array which permuted in the same way as a(*).
     -----------------------------------------------------------------------*/
    REAL tmp, abskey;
    INT  itmp, first, last, mid, j;
    
    /* Parameter adjustments */
    --ind;
    --a;
    
    first = 1;
    last = n;
    if ((ncut < first) || (ncut > last)) return;
    
    // outer loop -- while mid .ne. ncut do
F161:
    mid = first;
    abskey = ABS(a[mid]);
    for (j = first + 1; j <= last; ++j ) {
        if (ABS(a[j])  >  abskey) {
            ++mid;
            //     interchange
            tmp = a[mid];
            itmp = ind[mid];
            a[mid] = a[j];
            ind[mid] = ind[j];
            a[j] = tmp;
            ind[j] = itmp;
        }
    }
    
    // interchange
    tmp = a[mid];
    a[mid] = a[first];
    a[first] = tmp;
    //
    itmp = ind[mid];
    ind[mid] = ind[first];
    ind[first] = itmp;
    
    // test for while loop
    if (mid  ==  ncut) {
        ++ind;
        ++a;
        return;
    }
    
    if (mid  >  ncut)  {
        last = mid - 1;
    } else   {
        first = mid + 1;
    }
    
    goto F161;
    /*----------------end-of-qsplit------------------------------------------*/
}

/**
 * \fn static void fasp_sortrow (INT num,INT *q)
 *
 * \brief Shell sort with hardwired increments.
 *
 * \param num  size of q
 * \param q  integer array.
 *
 *
 * \author Chunsheng Feng
 * \date   09/06/2016
 */
static void fasp_sortrow (INT   num,
                          INT  *q)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    #endif
    /**
     ========================================================================
     
     Implement shell sort, with hardwired increments.  The algorithm for
     sorting entries in A(0:n-1) is as follows:
     ----------------------------------------------------------------
     inc = initialinc(n)
     while inc >= 1
     for i = inc to n-1
     j = i
     x = A(i)
     while j >= inc and A(j-inc) > x
     A(j) = A(j-inc)
     j    = j-inc
     end while
     A(j) = x
     end for
     inc = nextinc(inc,n)
     end while
     ----------------------------------------------------------------
     
     The increments here are 1, 4, 13, 40, 121, ..., (3**i - 1)/2, ...
     In this case, nextinc(inc,n) = (inc-1)/3.  Usually shellsort
     would have the largest increment the largest integer of the form
     (3**i - 1)/2 that is less than n, but here it is fixed at 121
     because most sparse matrices have 121 or fewer nonzero entries
     per row.  If this routine is expanded for a complete sparse
     factorization routine, or if a large number of levels of fill is
     allowed, then possibly it should be replaced with more efficient
     sorting.
     
     Any set of increments with 1 as the first one will result in a
     true sorting algorithm.
     
     ========================================================================*/
    
    INT key, icn, ih, ii, i, j, jj;
    INT iinc[6] = {0,1, 4, 13, 40, 121};
    //data iinc/1, 4, 13, 40, 121/;
    
    --q;
    if (num  ==  0)
        icn = 0;
    else if (num  <  14)
        icn = 1;
    else if (num  <  41)
        icn = 2;
    else if (num  <  122)
        icn = 3;
    else if (num  <  365)
        icn = 4;
    else
        icn = 5;
    
    for(ii = 1; ii <= icn; ++ii) { // 40
        ih = iinc[icn + 1 - ii];
        for(j = ih + 1; j <= num; ++j) { // 30
            i = j - ih;
            key = q[j];
            for(jj = 1; jj <= j - ih; jj += ih) { // 10
                if (key  >=  q[i]) {
                    goto F20;
                } else {
                    q[i + ih] = q[i];
                    i = i - ih;
                }
            }  // 10
        F20:
            q[i + ih] = key;
        }  // 30
    } // 40
    
    ++q;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    return;
}

/**
 * \fn static void fasp_check_col_index (INT row, INT num, INT *q)
 *
 * \brief Check the same col index in row.
 *
 * \param row  index of row to check
 * \param num  size of q
 * \param q    integer array.
 *
 * \author Chunsheng Feng
 * \date   07/30/2017
 */
static void fasp_check_col_index (INT row,
                                  INT num,
                                  INT  *q)
{
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif
    
    INT ii;
    INT num_1 = num - 1;
    
    for ( ii = 0; ii < num_1; ++ii ) {
        if ( q[ii] == q[ii+1] ) {
            printf("### ERROR: Multiple entries with same col indices!\n");
            printf("### ERROR: row = %d, col = %d, %d!\n", row, q[ii], q[ii+1]);
            fasp_chkerr(ERROR_SOLVER_ILUSETUP, __FUNCTION__);
        }
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
