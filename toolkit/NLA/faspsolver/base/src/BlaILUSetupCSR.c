/*! \file  BlaILUSetupCSR.c
 *
 *  \brief Setup incomplete LU decomposition for dCSRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxTiming.c, BlaILU.c, BlaSparseCSR.c, and PreDataInit.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_ilu_dcsr_setup (dCSRmat *A, ILU_data *iludata, ILU_param *iluparam)
 *
 * \brief Get ILU decomposition of a CSR matrix A
 *
 * \param A         Pointer to dCSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Shiquan Zhang Xiaozhe Hu
 * \date   12/27/2009
 *
 * Modified by Chunsheng Feng on 02/12/2017: add iperm array for ILUTp
 */
SHORT fasp_ilu_dcsr_setup (dCSRmat    *A,
                           ILU_data   *iludata,
                           ILU_param  *iluparam)
{
    const INT   type = iluparam->ILU_type, print_level = iluparam->print_level;
    const INT   n = A->col, nnz = A->nnz, mbloc = n;
    const REAL  ILU_droptol = iluparam->ILU_droptol;
    const REAL  permtol = iluparam->ILU_permtol;
    
    // local variable
    INT    lfil = iluparam->ILU_lfil, lfilt = iluparam->ILU_lfil;
    INT    ierr, iwk, nzlu, nwork, *ijlu, *iperm;
    REAL  *luval;
    
    REAL   setup_start, setup_end, setup_duration;
    SHORT  status = FASP_SUCCESS;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m=%d, n=%d, nnz=%d\n", A->row, n, nnz);
#endif
    
    fasp_gettime(&setup_start);
    
    // Expected amount of memory for ILU needed and allocate memory 
    switch (type) {
        case ILUt:
            iwk=100*nnz;     // iwk is the maxim possible nnz for ILU
            lfilt=(int)floor(n*0.5)+1;
            break;
        case ILUtp:
            iwk=100*nnz;     // iwk is the maxim possible nnz for ILU
            lfilt=(int)floor(n*0.5)+1;
            break;
        default: // ILUk
            if (lfil == 0) iwk=nnz+500;
            else iwk=(lfil+5)*nnz;
    } 
    
    nwork = 4*n;
    
#if DEBUG_MODE > 1
    printf("### DEBUG: fill-in = %d, iwk = %d, nwork = %d\n", lfil, iwk, nwork);
#endif
    
    // setup ILU preconditioner
    iludata->A     = A; // save a pointer to the coeff matrix for ILUtp
    iludata->row   = iludata->col   = n;
    iludata->ilevL = iludata->jlevL = NULL;
    iludata->ilevU = iludata->jlevU = NULL;
    iludata->iperm = NULL;
    iludata->type  = type;
    
    fasp_ilu_data_create(iwk, nwork, iludata);
    
#if DEBUG_MODE > 1
    printf("### DEBUG: memory usage after %s: \n", __FUNCTION__);
    fasp_mem_usage();
#endif
    
    // ILU decomposition
    ijlu  = iludata->ijlu;
    luval = iludata->luval;
    
    switch (type) {

        case ILUt:
            fasp_ilut (n, A->val, A->JA, A->IA, lfilt, ILU_droptol, luval, ijlu,
                       iwk, &ierr, &nzlu);
            break;
            
        case ILUtp:
            iperm = iludata->iperm;
            fasp_ilutp (n, A->val, A->JA, A->IA, lfilt, ILU_droptol, permtol,
                        mbloc, luval, ijlu, iperm, iwk, &ierr, &nzlu);
            break;
            
        default: // ILUk
            fasp_iluk (n, A->val, A->JA, A->IA, lfil, luval, ijlu, iwk,
                       &ierr, &nzlu);
            break;

    } 
   if (ierr != -4) 
    fasp_dcsr_shift(A, -1);
    
#if DEBUG_MODE > 1
    printf("### DEBUG: memory usage after ILU setup: \n");
    fasp_mem_usage();
#endif
    
    iludata->nzlu  = nzlu;
    iludata->nwork = nwork;
    
#if DEBUG_MODE > 1
    printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
#endif    
    
    if (ierr!=0) {
        printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if (iwk<nzlu) {
        printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if (print_level>PRINT_NONE) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        
        switch (type) {
            case ILUt:
                printf("ILUt setup costs %f seconds.\n", setup_duration);    
                break;
            case ILUtp:
                printf("ILUtp setup costs %f seconds.\n", setup_duration);    
                break;
            default: // ILUk
                printf("ILUk setup costs %f seconds.\n", setup_duration);    
                break;
        }     
    }
    
FINISHED:     
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
