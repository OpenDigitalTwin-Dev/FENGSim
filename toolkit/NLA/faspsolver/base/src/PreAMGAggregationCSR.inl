/*! \file  PreAMGAggregationCSR.inl
 *
 *  \brief Utilities for aggregation methods for CSR matrices
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreAMGSetupSA.c and PreAMGSetupUA.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  \warning This file is also used in FASP4BLKOIL!!!
 */

#ifdef _OPENMP
#include <omp.h>
#endif

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void form_tentative_p (ivector *vertices, dCSRmat *tentp, REAL **basis,
 *                                   INT NumAggregates)
 *
 * \brief Form aggregation based on strong coupled neighbors
 *
 * \param vertices           Pointer to the aggregation of vertices
 * \param tentp              Pointer to the prolongation operators
 * \param basis              Pointer to the near kernel
 * \param NumAggregates      Number of aggregations
 *
 * \author Xiaozhe Hu
 * \date   09/29/2009
 *
 * Modified by Xiaozhe Hu on 05/25/2014
 */
static void form_tentative_p (ivector  *vertices,
                              dCSRmat  *tentp,
                              REAL    **basis,
                              INT       NumAggregates)
{
    INT i, j;

    /* Form tentative prolongation */
    tentp->row = vertices->row;
    tentp->col = NumAggregates;
    tentp->nnz = vertices->row;

    tentp->IA  = (INT *)fasp_mem_calloc(tentp->row+1,sizeof(INT));

    // local variables
    INT  *IA = tentp->IA;
    INT  *vval = vertices->val;
    const INT row = tentp->row;

    // first run
    for ( i = 0, j = 0; i < row; i++ ) {
        IA[i] = j;
        if (vval[i] > UNPT) j++;
    }
    IA[row] = j;

    // allocate memory for P
    tentp->nnz = j;
    tentp->JA  = (INT *)fasp_mem_calloc(tentp->nnz, sizeof(INT));
    tentp->val = (REAL *)fasp_mem_calloc(tentp->nnz, sizeof(REAL));

    INT  *JA = tentp->JA;
    REAL *val = tentp->val;

    // second run
    for (i = 0, j = 0; i < row; i ++) {
        IA[i] = j;
        if (vval[i] > UNPT) {
            JA[j] = vval[i];
            val[j] = basis[0][i];
            j ++;
        }
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
