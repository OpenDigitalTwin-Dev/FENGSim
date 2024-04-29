/*! \file  BlaSparseBLC.c
 *
 *  \brief Sparse matrix block operations
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c and BlaSparseCSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_block.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_dblc_free (dBLCmat *A)
 *
 * \brief Free block CSR sparse matrix data memory space
 *
 * \param A   Pointer to the dBLCmat matrix
 *
 * \author Xiaozhe Hu
 * \date   04/18/2014
 */
void fasp_dblc_free (dBLCmat *A)
{
    INT i;
    INT num_blocks = (A->brow)*(A->bcol);
    
	if (A == NULL) return; // Nothing need to be freed!
    
    for ( i=0; i<num_blocks; i++ ) {
        fasp_dcsr_free(A->blocks[i]);
        A->blocks[i] = NULL;
    }
    
    fasp_mem_free(A->blocks); A->blocks = NULL;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
