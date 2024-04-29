/*! \file  PreAMGSetupCR.c
 *
 *  \brief Brannick-Falgout compatible relaxation based AMG: SETUP phase
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxMessage.c, AuxTiming.c, AuxVector.c, and PreAMGCoarsenCR.c
 *
 *  \note  Setup A, P, R and levels using the Compatible Relaxation coarsening
 *         for classic AMG interpolation
 *
 *  Reference: 
 *         J. Brannick and R. Falgout
 *         Compatible relaxation and coarsening in AMG
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  TODO: Not working. Need to be fixed. --Chensong
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_amg_setup_cr (AMG_data *mgl, AMG_param *param)
 *
 * \brief Set up phase of Brannick Falgout CR coarsening for classic AMG
 *
 * \param mgl    Pointer to AMG data: AMG_data
 * \param param  Pointer to AMG parameters: AMG_param
 *
 * \return       FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author James Brannick
 * \date   04/21/2010
 *
 * Modified by Chensong Zhang on 05/10/2013: adjust the structure.
 */
SHORT fasp_amg_setup_cr (AMG_data   *mgl,
                         AMG_param  *param)
{
    const SHORT  prtlvl   = param->print_level;
    const SHORT  min_cdof = MAX(param->coarse_dof,50);
    const INT    m        = mgl[0].A.row;
    
    // local variables
    INT     i_0 = 0, i_n;
    SHORT   level = 0, status = FASP_SUCCESS;
    SHORT   max_levels = param->max_levels;
    REAL    setup_start, setup_end;
    
    // The variable vertices stores level info (fine: 0; coarse: 1)
    ivector vertices = fasp_ivec_create(m); 
    
    fasp_gettime(&setup_start);

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n",
           mgl[0].A.row, mgl[0].A.col, mgl[0].A.nnz);
#endif
    
#if DIAGONAL_PREF
    fasp_dcsr_diagpref(&mgl[0].A); // reorder each row to make diagonal appear first
#endif
    
    // Main AMG setup loop
    while ( (mgl[level].A.row > min_cdof) && (level < max_levels-1) ) {
    
        /*-- Coarsen and form the structure of interpolation --*/
        i_n = mgl[level].A.row-1;
    
        fasp_amg_coarsening_cr(i_0,i_n,&mgl[level].A, &vertices, param);
    
        /*-- Form interpolation --*/
        /* 1. SPARSITY -- Form ip and jp */ 
        /* First a symbolic one
           then gather the list */
        /* 2. COEFFICIENTS -- Form P */ 
        // energymin(mgl[level].A, &vertices[level], mgl[level].P, param);
        // fasp_mem_free(vertices[level].val); vertices[level].val = NULL;
    
        /*-- Form coarse level stiffness matrix --*/
        // fasp_dcsr_trans(mgl[level].P, mgl[level].R);
    
        /*-- Form coarse level stiffness matrix --*/    
        //fasp_blas_dcsr_rap(mgl[level].R, mgl[level].A, mgl[level].P, mgl[level+1].A);
    
        ++level;
        
#if DIAGONAL_PREF
        fasp_dcsr_diagpref(&mgl[level].A); // reorder each row to make diagonal appear first
#endif     
    }
    
    // setup total level number and current level
    mgl[0].num_levels = max_levels = level+1;
    mgl[0].w          = fasp_dvec_create(m);    
    
    for ( level = 1; level < max_levels; ++level ) {
        INT mm = mgl[level].A.row;
        mgl[level].num_levels = max_levels;     
        mgl[level].b = fasp_dvec_create(mm);
        mgl[level].x = fasp_dvec_create(mm);
        mgl[level].w = fasp_dvec_create(mm);
    }
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        fasp_amgcomplexity(mgl,prtlvl);
        fasp_cputime("Compatible relaxation setup", setup_end - setup_start);
    }
    
    fasp_ivec_free(&vertices); 
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
