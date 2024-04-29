/*! \file  BlaSparseCSRL.c
 *
 *  \brief Sparse matrix operations for dCSRLmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c
 *
 *  Reference: 
 *         John Mellor-Crummey and John Garvin
 *         Optimizaing sparse matrix vector product computations using unroll and
 *         jam, Tech Report Rice Univ, Aug 2002.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2011--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn dCSRLmat * fasp_dcsrl_create (const INT num_rows, const INT num_cols, 
 *                                   const INT num_nonzeros)
 *
 * \brief Create a dCSRLmat object
 *
 * \param num_rows      Number of rows
 * \param num_cols      Number of cols
 * \param num_nonzeros  Number of nonzero entries
 *
 * \author Zhiyang Zhou
 * \date   01/07/2011
 */
dCSRLmat * fasp_dcsrl_create (const INT num_rows,
                              const INT num_cols,
                              const INT num_nonzeros)
{
    dCSRLmat *A   = (dCSRLmat *)fasp_mem_calloc(1, sizeof(dCSRLmat));
    
    A -> row      = num_rows;
    A -> col      = num_cols;
    A -> nnz      = num_nonzeros;
    A -> nz_diff  = NULL;
    A -> index    = NULL;
    A -> start    = NULL;
    A -> ja       = NULL;
    A -> val      = NULL;
    
    return A;
}

/**
 * \fn void fasp_dcsrl_free ( dCSRLmat *A )
 *
 * \brief Destroy a dCSRLmat object
 *
 * \param A   Pointer to the dCSRLmat type matrix
 *
 * \author Zhiyang Zhou
 * \date   01/07/2011
 */
void fasp_dcsrl_free (dCSRLmat *A)
{
    if (A) {  
        if (A -> nz_diff) free(A -> nz_diff);
        if (A -> index)   free(A -> index);
        if (A -> start)   free(A -> start);
        if (A -> ja)      free(A -> ja);
        if (A -> val)     free(A -> val);
        free(A);
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
