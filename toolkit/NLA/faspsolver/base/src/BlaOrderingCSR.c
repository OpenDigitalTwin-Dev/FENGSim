/*! \file  BlaOrderingCSR.c
 *
 *  \brief Generating ordering using algebraic information
 *
 *  \note  This file contains Level-1 (Bla) functions.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static void CMK_ordering (const dCSRmat *, INT, INT, INT, INT, INT *, INT *);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_dcsr_CMK_order (const dCSRmat *A, INT *order, INT *oindex)
 *
 * \brief Ordering vertices of matrix graph corresponding to A.
 *
 * \param A       Pointer to matrix
 * \param oindex  Pointer to index of vertices in order
 * \param order   Pointer to vertices with increasing degree
 *
 * \author Zheng Li, Chensong Zhang
 * \date   05/28/2014
 */
void fasp_dcsr_CMK_order (const dCSRmat *A,
                          INT           *order,
                          INT           *oindex)
{
    const INT *ia = A->IA;
    const INT row = A->row;
    
    INT i, loc, s, vt, mindg, innz;
    
    s = 0;
    vt = 0;
    mindg = row+1;
    
    // select node with minimal degree
    for (i=0; i<row; ++i) {
        innz = ia[i+1] - ia[i];
        if (innz > 1) {
            oindex[i] = -innz;
            if (innz < mindg) {
                mindg = innz;
                vt = i;
            }
        }
        else { // order those diagonal rows first
            oindex[i] = s;
            order[s] = i;
            s ++;
        }
    }
    
    loc = s;
    
    // start to order
    CMK_ordering (A, loc, s, vt, mindg, oindex, order);
}

/**
 * \fn void fasp_dcsr_RCMK_order (const dCSRmat *A, INT *order, INT *oindex,
 *                                INT *rorder)
 *
 * \brief  Resverse CMK ordering
 *
 * \param A       Pointer to matrix
 * \param order   Pointer to vertices with increasing degree
 * \param oindex  Pointer to index of vertices in order
 * \param rorder  Pointer to reverse order
 *
 * \author Zheng Li, Chensong Zhang
 * \date   10/10/2014
 */
void fasp_dcsr_RCMK_order (const dCSRmat *A,
                           INT           *order,
                           INT           *oindex,
                           INT           *rorder)
{
    INT i;
    INT row = A->row;
    
    // Form CMK order
    fasp_dcsr_CMK_order(A, order, oindex);
    
    // Reverse CMK order
    for (i=0; i<row; ++i) rorder[i] = order[row-1-i];
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void CMK_ordering (const dCSRmat *A, INT loc, INT s, INT jj,
 *                               INT mindg, INT *oindex, INT *order)
 *
 * \brief CMK ordering by increasing degree of vertices.
 *
 * \param A       Pointer to matrix
 * \param loc     Main order loop variable
 * \param s       Number of ordered vertices
 * \param jj      Vertices with minimal degree
 * \param mindg   Minimal degree
 * \param oindex  Pointer to index of vertices in order
 * \param order   Pointer to vertices with increasing degree
 *
 * \author Zheng Li, Chensong Zhang
 * \date   05/28/2014
 */
static void CMK_ordering (const dCSRmat *A,
                          INT            loc,
                          INT            s,
                          INT            jj,
                          INT            mindg,
                          INT           *oindex,
                          INT           *order)
{
    const INT  row = A->row;
    const INT *ia  = A->IA;
    const INT *ja  = A->JA;
    
    INT       i, j, sp1, k;
    SHORT     flag = 1;
    
    if (s < row) {
        order[s] = jj;
        oindex[jj] = s;
    }
    
    while (loc <= s && s < row) {
        i = order[loc];
        sp1 = s+1;
        // neighbor nodes are priority.
        for (j=ia[i]+1; j<ia[i+1]; ++j) {
            k = ja[j];
            if (oindex[k] < 0){
                s++;
                order[s] = k;
            }
        }
        // ordering neighbor nodes by increasing degree
        if (s > sp1) {
            while (flag) {
                flag = 0;
                for (i=sp1+1; i<=s; ++i) {
                    if (oindex[order[i]] > oindex[order[i-1]]) {
                        j = order[i];
                        order[i] = order[i-1];
                        order[i-1] = j;
                        flag = 1;
                    }
                }
            }
        }
        
        for (i=sp1; i<=s; ++i) oindex[order[i]] = i;
        
        loc ++;
    }
    
    // deal with remainder
    if (s < row) {
        jj = 0;
        i  = 0;
        while (jj == 0) {
            i ++;
            if (i >= row) {
                mindg++;
                i = 0;
            }
            if (oindex[i] < 0 && (ia[i+1]-ia[i] == mindg)) {
                jj = i;
            }
        }
        
        s ++;
        
        CMK_ordering (A, loc, s, jj, mindg, oindex, order);
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
