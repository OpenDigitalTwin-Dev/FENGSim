/*! \file  BlaILUSetupBSR.c
 *
 *  \brief Setup incomplete LU decomposition for dBSRmat matrices
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxTiming.c, BlaSmallMatInv.c, BlaILU.c,
 *         BlaSmallMat.c, BlaSmallMatInv.c, BlaSparseBSR.c, BlaSparseCSR.c,
 *         BlaSpmvCSR.c, and PreDataInit.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
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

static INT numfactor (dBSRmat *, REAL *, INT *, INT *);
static INT numfactor_mulcol (dBSRmat *, REAL *, INT *, INT *, INT, INT *, INT *);
static INT numfactor_levsch (dBSRmat *, REAL *, INT *, INT *, INT, INT *, INT *);
static void generate_S_theta(dCSRmat *, iCSRmat *, REAL);
// static void topologic_sort_ILU (ILU_data *);
// static void mulcol_independ_set (AMG_data *, INT);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_ilu_dbsr_setup (dBSRmat *A, ILU_data *iludata, ILU_param *iluparam)
 *
 * \brief Get ILU decoposition of a BSR matrix A
 *
 * \param A         Pointer to dBSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   11/08/2010
 *
 * \note Works for general nb (Xiaozhe)
 * \note Change the size of work space by Zheng Li 04/26/2015.
 * \note Modified by Chunsheng Feng on 08/11/2017 for iludata->type not inited.
 */
SHORT fasp_ilu_dbsr_setup(dBSRmat *A,
                          ILU_data *iludata,
                          ILU_param *iluparam)
{
    
    const SHORT  prtlvl = iluparam->print_level;
    const INT    n = A->COL, nnz = A->NNZ, nb = A->nb, nb2 = nb*nb;
    
    // local variables
    INT     lfil = iluparam->ILU_lfil;
    INT     ierr, iwk, nzlu, nwork, *ijlu, *uptr;
    SHORT   status = FASP_SUCCESS;
    REAL    setup_start, setup_end, setup_duration;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m = %d, n = %d, nnz = %d\n", A->ROW, n, nnz);
#endif
    
    fasp_gettime(&setup_start);
    
    // Expected amount of memory for ILU needed and allocate memory
    iwk = (lfil+2)*nnz;
    
#if DEBUG_MODE > 0
    if (iluparam->ILU_type == ILUtp) {
        printf("### WARNING: iludata->type = %d not supported!\n",
               iluparam->ILU_type);
    }
#endif

    // setup preconditioner
    iludata->type  = 0; // Must be initialized
    iludata->iperm = NULL;
    iludata->A     = NULL; // No need for BSR matrix
    iludata->row   = iludata->col = n;
    iludata->nb    = nb;
    iludata->ilevL = iludata->jlevL = NULL;
    iludata->ilevU = iludata->jlevU = NULL;
    
    ijlu = (INT*)fasp_mem_calloc(iwk,sizeof(INT));
    uptr = (INT*)fasp_mem_calloc(A->ROW,sizeof(INT));
    
#if DEBUG_MODE > 1
    printf("### DEBUG: symbolic factorization ... \n");
#endif
    
    // ILU decomposition
    // (1) symbolic factoration
    fasp_symbfactor(A->ROW,A->JA,A->IA,lfil,iwk,&nzlu,ijlu,uptr,&ierr);
    
    if ( ierr != 0 ) {
        printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }

    iludata->luval = (REAL*)fasp_mem_calloc(nzlu*nb2,sizeof(REAL));
    
#if DEBUG_MODE > 1
    printf("### DEBUG: numerical factorization ... \n");
#endif
    
    // (2) numerical factoration
    status = numfactor(A, iludata->luval, ijlu, uptr);
    
    if ( status < 0 ) {
        printf("### ERROR: ILU factorization failed! [%s]\n", __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }

    //nwork = 6*nzlu*nb;
    nwork = 20*A->ROW*A->nb;
    iludata->nzlu  = nzlu;
    iludata->nwork = nwork;
    iludata->ijlu  = (INT*)fasp_mem_calloc(nzlu, sizeof(INT));
    
    memcpy(iludata->ijlu,ijlu,nzlu*sizeof(INT));
    iludata->work = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
    // Check: Is the work space too large? --Xiaozhe
    
#if DEBUG_MODE > 1
    printf("### DEBUG: fill-in = %d, nwork = %d\n", lfil, nwork);
    printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
#endif
    
    if ( iwk < nzlu ) {
        printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        printf("BSR ILU(%d)-seq setup costs %f seconds.\n", lfil, setup_duration);
    }
    
FINISHED:
    fasp_mem_free(ijlu);  ijlu = NULL;
    fasp_mem_free(uptr);  uptr = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/**
 * \fn SHORT fasp_ilu_dbsr_setup_step (dBSRmat *A, ILU_data *iludata,
 *                                     ILU_param *iluparam, INT step)
 *
 * \brief Get ILU decoposition of a BSR matrix A
 *
 * \param A         Pointer to dBSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 * \param step      Step in ILU factorization
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Shiquan Zhang, Xiaozhe Hu, Li Zhao
 * \date   11/08/2010
 *
 * \note Works for general nb (Xiaozhe)
 * \note Change the size of work space by Zheng Li 04/26/2015.
 * \note Modified by Chunsheng Feng on 08/11/2017 for iludata->type not inited.
 * \note Modified by Li Zhao on 04/29/2021: ILU factorization divided into two steps:
 *       step == 1: symbolic factoration; if step == 2: numerical factoration.
 */
SHORT fasp_ilu_dbsr_setup_step (dBSRmat    *A,
								ILU_data   *iludata,
								ILU_param  *iluparam,
								INT step)
{
    
    const SHORT  prtlvl = iluparam->print_level;
    const INT    n = A->COL, nnz = A->NNZ, nb = A->nb, nb2 = nb*nb;
    
    // local variables
       INT     lfil = iluparam->ILU_lfil;
    static INT     ierr, iwk, nzlu, nwork, *ijlu, *uptr;
    SHORT   status = FASP_SUCCESS;

    REAL    setup_start, setup_end, setup_duration;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m = %d, n = %d, nnz = %d\n", A->ROW, n, nnz);
#endif
    
    fasp_gettime(&setup_start);

    if (step==1) {
        // Expected amount of memory for ILU needed and allocate memory
        iwk = (lfil+2)*nnz;
        
    #if DEBUG_MODE > 0
        if (iluparam->ILU_type == ILUtp) {
            printf("### WARNING: iludata->type = %d not supported!\n",
                iluparam->ILU_type);
        }
    #endif

        // setup preconditioner
        iludata->type  = 0; // Must be initialized
        iludata->iperm = NULL;
        iludata->A     = NULL; // No need for BSR matrix
        iludata->row   = iludata->col = n;
        iludata->nb    = nb;
        iludata->ilevL = iludata->jlevL = NULL;
        iludata->ilevU = iludata->jlevU = NULL;
        
        ijlu = (INT*)fasp_mem_calloc(iwk,sizeof(INT));

        if (uptr != NULL)   fasp_mem_free(uptr);
        uptr = (INT*)fasp_mem_calloc(A->ROW,sizeof(INT));
        
    #if DEBUG_MODE > 1
        printf("### DEBUG: symbolic factorization ... \n");
    #endif
        
        // ILU decomposition
        // (1) symbolic factoration
        fasp_symbfactor(A->ROW,A->JA,A->IA,lfil,iwk,&nzlu,ijlu,uptr,&ierr);
        
        iludata->luval = (REAL*)fasp_mem_calloc(nzlu*nb2,sizeof(REAL));
        

    #if DEBUG_MODE > 1
        printf("### DEBUG: numerical factorization ... \n");
    #endif
        
        //nwork = 6*nzlu*nb;
        nwork = 5*A->ROW*A->nb;
        iludata->nwork = nwork;
        iludata->nzlu  = nzlu;
        iludata->ijlu  = (INT*)fasp_mem_calloc(nzlu, sizeof(INT));
        
        memcpy(iludata->ijlu,ijlu,nzlu*sizeof(INT));
        fasp_mem_free(ijlu);  ijlu = NULL;

        iludata->work = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
        // Check: Is the work space too large? --Xiaozhe
        
    #if DEBUG_MODE > 1
        printf("### DEBUG: fill-in = %d, nwork = %d\n", lfil, nwork);
        printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
    #endif
        
        if ( ierr != 0 ) {
            printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
            status = ERROR_SOLVER_ILUSETUP;
            goto FINISHED;
        }
        
        if ( iwk < nzlu ) {
            printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
            status = ERROR_SOLVER_ILUSETUP;
            goto FINISHED;
        }
    }
    else if (step==2) {
        // (2) numerical factoration
        numfactor(A, iludata->luval, iludata->ijlu, uptr);

    } else {

FINISHED:
            fasp_mem_free(uptr);  uptr = NULL;
    }

    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        printf("BSR ILU(%d) setup costs %f seconds.\n", lfil, setup_duration);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/**
 * \fn SHORT fasp_ilu_dbsr_setup_omp (dBSRmat *A, ILU_data *iludata, 
 *                                    ILU_param *iluparam)
 *
 * \brief Multi-thread ILU decoposition of a BSR matrix A based on graph coloring
 *
 * \param A         Pointer to dBSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for 1, 2, 3 nb (Zheng)
 * \note Modified by Chunsheng Feng on 09/06/2017 for iludata->type not inited.
 */
SHORT fasp_ilu_dbsr_setup_omp (dBSRmat    *A,
                               ILU_data   *iludata,
                               ILU_param  *iluparam)
{
    
    const SHORT  prtlvl = iluparam->print_level;
    const INT    n = A->COL, nnz = A->NNZ, nb = A->nb, nb2 = nb*nb;
    
    // local variables
    INT     lfil = iluparam->ILU_lfil;
    INT     ierr, iwk, nzlu, nwork, *ijlu, *uptr;
    SHORT   status = FASP_SUCCESS;

    REAL    setup_start, setup_end, setup_duration;
    REAL    symbolic_start, symbolic_end, numfac_start, numfac_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m = %d, n = %d, nnz = %d\n", A->ROW, n, nnz);
#endif
    
    fasp_gettime(&setup_start);
    
    // Expected amount of memory for ILU needed and allocate memory
    iwk = (lfil+2)*nnz;
    
#if DEBUG_MODE > 0
    if (iluparam->ILU_type == ILUtp) {
        printf("### WARNING: iludata->type = %d not supported any more!\n",
               iluparam->ILU_type);
    }
#endif

    // setup preconditioner
    iludata->type  = 0; // Must be initialized
    iludata->iperm = NULL;
    iludata->A     = NULL; // No need for BSR matrix
    iludata->row   = iludata->col = n;
    iludata->nb    = nb;
    
    ijlu = (INT *) fasp_mem_calloc(iwk,   sizeof(INT));
    uptr = (INT *) fasp_mem_calloc(A->ROW,sizeof(INT));
    
#if DEBUG_MODE > 1
    printf("### DEBUG: symbolic factorization ... \n");
#endif
    
    // ILU decomposition
    // (1) symbolic factoration
    fasp_gettime(&symbolic_start);

    fasp_symbfactor(A->ROW,A->JA,A->IA,lfil,iwk,&nzlu,ijlu,uptr,&ierr);

    fasp_gettime(&symbolic_end);

#if prtlvl > PRINT_MIN
    printf("ILU symbolic factorization time = %f\n", symbolic_end-symbolic_start);
#endif

    nwork = 5*A->ROW*A->nb;
    iludata->nzlu  = nzlu;
    iludata->nwork = nwork;
    iludata->ijlu  = (INT*)fasp_mem_calloc(nzlu,sizeof(INT));
    iludata->luval = (REAL*)fasp_mem_calloc(nzlu*nb2,sizeof(REAL));
    iludata->work  = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
    memcpy(iludata->ijlu,ijlu,nzlu*sizeof(INT));
    fasp_darray_set(nzlu*nb2, iludata->luval, 0.0);

#if DEBUG_MODE > 1
    printf("### DEBUG: numerical factorization ... \n");
#endif
    
    // (2) numerical factoration
    fasp_gettime(&numfac_start);

    numfactor_mulcol(A, iludata->luval, ijlu, uptr, iludata->nlevL,
                     iludata->ilevL, iludata->jlevL);
    
    fasp_gettime(&numfac_end);

#if prtlvl > PRINT_MIN
    printf("ILU numerical factorization time = %f\n", numfac_end-numfac_start);
#endif

#if DEBUG_MODE > 1
    printf("### DEBUG: fill-in = %d, nwork = %d\n", lfil, nwork);
    printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
#endif
    
    if ( ierr != 0 ) {
        printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if ( iwk < nzlu ) {
        printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        printf("BSR ILU(%d)-mc setup costs %f seconds.\n", lfil, setup_duration);
    }
    
FINISHED:
    fasp_mem_free(ijlu);  ijlu = NULL;
    fasp_mem_free(uptr);  uptr = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/**
 * \fn SHORT fasp_ilu_dbsr_setup_levsch_omp (dBSRmat *A, ILU_data *iludata, 
 *                                           ILU_param *iluparam)
 *
 * \brief Get ILU decoposition of a BSR matrix A based on level schedule strategy
 *
 * \param A         Pointer to dBSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for nb = 1, 2, 3 (Zheng)
 * \note Modified by Chunsheng Feng on 09/06/2017 for iludata->type not inited
 */
SHORT fasp_ilu_dbsr_setup_levsch_omp (dBSRmat    *A,
                                      ILU_data   *iludata,
                                      ILU_param  *iluparam)
{
    const SHORT  prtlvl = iluparam->print_level;
    const INT    n = A->COL, nnz = A->NNZ, nb = A->nb, nb2 = nb*nb;
    
    // local variables
    INT lfil = iluparam->ILU_lfil;
    INT ierr, iwk, nzlu, nwork, *ijlu, *uptr;
    SHORT   status = FASP_SUCCESS;

    REAL    setup_start, setup_end, setup_duration;
    REAL    symbolic_start, symbolic_end, numfac_start, numfac_end;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m=%d, n=%d, nnz=%d\n", A->ROW, n, nnz);
#endif
    
    fasp_gettime(&setup_start);
    
    // Expected amount of memory for ILU needed and allocate memory
    iwk = (lfil+2)*nnz;
    
#if DEBUG_MODE > 0
    if (iluparam->ILU_type == ILUtp) {
        printf("### WARNING: iludata->type = %d not supported!\n",
               iluparam->ILU_type);
    }
#endif

    // setup preconditioner
    iludata->type  = 0; // Must be initialized
    iludata->iperm = NULL;
    iludata->A     = NULL; // No need for BSR matrix
    iludata->row   = iludata->col=n;
    iludata->nb    = nb;
    
    ijlu = (INT*)fasp_mem_calloc(iwk,sizeof(INT));
    uptr = (INT*)fasp_mem_calloc(A->ROW,sizeof(INT));
    
#if DEBUG_MODE > 1
    printf("### DEBUG: symbolic factorization ... \n");
#endif
    
    fasp_gettime(&symbolic_start);
    
    // ILU decomposition
    // (1) symbolic factoration
    fasp_symbfactor(A->ROW,A->JA,A->IA,lfil,iwk,&nzlu,ijlu,uptr,&ierr);
    
    fasp_gettime(&symbolic_end);

#if prtlvl > PRINT_MIN
    printf("ILU symbolic factorization time = %f\n", symbolic_end-symbolic_start);
#endif

    nwork = 5*A->ROW*A->nb;
    iludata->nzlu  = nzlu;
    iludata->nwork = nwork;
    iludata->ijlu  = (INT*)fasp_mem_calloc(nzlu,sizeof(INT));
    iludata->luval = (REAL*)fasp_mem_calloc(nzlu*nb2,sizeof(REAL));
    iludata->work  = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
    memcpy(iludata->ijlu,ijlu,nzlu*sizeof(INT));
    fasp_darray_set(nzlu*nb2, iludata->luval, 0.0);
    iludata->uptr = NULL; iludata->ic = NULL; iludata->icmap = NULL;
    
    topologic_sort_ILU(iludata);
    
#if DEBUG_MODE > 1
    printf("### DEBUG: numerical factorization ... \n");
#endif
    
    fasp_gettime(&numfac_start);
    
    // (2) numerical factoration
    numfactor_levsch(A, iludata->luval, ijlu, uptr, iludata->nlevL,
                     iludata->ilevL, iludata->jlevL);
    
    fasp_gettime(&numfac_end);
    
#if prtlvl > PRINT_MIN
    printf("ILU numerical factorization time = %f\n", numfac_end-numfac_start);
#endif

#if DEBUG_MODE > 1
    printf("### DEBUG: fill-in = %d, nwork = %d\n", lfil, nwork);
    printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
#endif
    
    if ( ierr != 0 ) {
        printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if ( iwk < nzlu ) {
        printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
        status = ERROR_SOLVER_ILUSETUP;
        goto FINISHED;
    }
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        printf("BSR ILU(%d)-ls setup costs %f seconds.\n", lfil, setup_duration);
    }
    
FINISHED:
    fasp_mem_free(ijlu);  ijlu = NULL;
    fasp_mem_free(uptr);  uptr = NULL;
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

/**
 * \fn SHORT fasp_ilu_dbsr_setup_levsch_step (dBSRmat *A, ILU_data *iludata,
 *                                            ILU_param *iluparam, INT step)
 *
 * \brief Get ILU decoposition of a BSR matrix A based on level schedule strategy
 *
 * \param A         Pointer to dBSRmat matrix
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 * \param step      Step in ILU factorization
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for nb = 1, 2, 3 (Zheng)
 * \note Modified by Chunsheng Feng on 09/06/2017 for iludata->type not inited
 * \note Modified by Li Zhao on 04/29/2021: ILU factorization divided into two steps:
 *       step == 1: symbolic factoration; if step == 2: numerical factoration.
 */
SHORT fasp_ilu_dbsr_setup_levsch_step (dBSRmat    *A,
                                       ILU_data   *iludata,
                                       ILU_param  *iluparam,
									   INT step)
{
    const SHORT  prtlvl = iluparam->print_level;
    const INT    n = A->COL, nnz = A->NNZ, nb = A->nb, nb2 = nb*nb;
    
    // local variables
    INT lfil = iluparam->ILU_lfil;
    static INT ierr, iwk, nzlu, nwork, *ijlu, *uptr;
    SHORT   status = FASP_SUCCESS;

    REAL    setup_start, setup_end, setup_duration;
    REAL    symbolic_start, symbolic_end, numfac_start, numfac_end;

#if DEBUG_MODE > 0 
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: m=%d, n=%d, nnz=%d\n", A->ROW, n, nnz);
    printf("### DEBUG: step=%d(1: symbolic factoration, 2: numerical factoration)\n", step);// zhaoli 2021.03.24
#endif
    
    fasp_gettime(&setup_start);
   if (step==1) { 
        // Expected amount of memory for ILU needed and allocate memory
        iwk = (lfil+2)*nnz;
        
    #if DEBUG_MODE > 0
        if (iluparam->ILU_type == ILUtp) {
            printf("### WARNING: iludata->type = %d not supported!\n",
                iluparam->ILU_type);
        }
    #endif

        // setup preconditioner
        iludata->type  = 0; // Must be initialized
        iludata->iperm = NULL;
        iludata->A     = NULL; // No need for BSR matrix
        iludata->row   = iludata->col=n;
        iludata->nb    = nb;
        
        fasp_mem_free(ijlu); 
        ijlu = (INT*)fasp_mem_calloc(iwk,sizeof(INT));

        fasp_mem_free(uptr); 
        uptr = (INT*)fasp_mem_calloc(A->ROW,sizeof(INT));
        
    #if DEBUG_MODE > 1
        printf("### DEBUG: symbolic factorization ... \n");
    #endif
        
        fasp_gettime(&symbolic_start);
        
        // ILU decomposition
        // (1) symbolic factoration
        fasp_symbfactor(A->ROW,A->JA,A->IA,lfil,iwk,&nzlu,ijlu,uptr,&ierr);
        
        fasp_gettime(&symbolic_end);

    #if prtlvl > PRINT_MIN
        printf("ILU symbolic factorization time = %f\n", symbolic_end-symbolic_start);
    #endif

        nwork = 5*A->ROW*A->nb;
        iludata->nzlu  = nzlu;
        iludata->nwork = nwork;
        iludata->ijlu  = (INT*)fasp_mem_calloc(nzlu,sizeof(INT));
        iludata->luval = (REAL*)fasp_mem_calloc(nzlu*nb2,sizeof(REAL));
        iludata->work  = (REAL*)fasp_mem_calloc(nwork, sizeof(REAL));
        memcpy(iludata->ijlu,ijlu,nzlu*sizeof(INT));
        fasp_mem_free(ijlu);  ijlu = NULL;

        fasp_darray_set(nzlu*nb2, iludata->luval, 0.0);
        iludata->uptr = NULL; iludata->ic = NULL; iludata->icmap = NULL;
        
        topologic_sort_ILU(iludata);
    #if DEBUG_MODE > 1
        printf("### DEBUG: fill-in = %d, nwork = %d\n", lfil, nwork);
        printf("### DEBUG: iwk = %d, nzlu = %d\n", iwk, nzlu);
    #endif
        
        if ( ierr != 0 ) {
            printf("### ERROR: ILU setup failed (ierr=%d)! [%s]\n", ierr, __FUNCTION__);
            status = ERROR_SOLVER_ILUSETUP;
            goto FINISHED;
        }
        
        if ( iwk < nzlu ) {
            printf("### ERROR: ILU needs more RAM %d! [%s]\n", iwk-nzlu, __FUNCTION__);
            status = ERROR_SOLVER_ILUSETUP;
            goto FINISHED;
        }
   } else if (step==2) {

#if DEBUG_MODE > 1
    printf("### DEBUG: numerical factorization ... \n");
#endif
        
        fasp_gettime(&numfac_start);
        
        // (2) numerical factoration
        numfactor_levsch(A, iludata->luval, iludata->ijlu, uptr, iludata->nlevL,
                        iludata->ilevL, iludata->jlevL);
        fasp_gettime(&numfac_end);
        
#if prtlvl > PRINT_MIN
    printf("ILU numerical factorization time = %f\n", numfac_end-numfac_start);
#endif
   } else {

FINISHED:
//    fasp_mem_free(ijlu);  ijlu = NULL;
        fasp_mem_free(uptr);  uptr = NULL;
   }
    
    if ( prtlvl > PRINT_NONE ) {
        fasp_gettime(&setup_end);
        setup_duration = setup_end - setup_start;
        printf("BSR ILU(%d)-ls setup costs %f seconds.\n", lfil, setup_duration);
    }
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    
    return status;
}

/**
 * \fn SHORT fasp_ilu_dbsr_setup_mc_omp (dBSRmat *A, dCSRmat *Ap, 
 *                                       ILU_data *iludata, ILU_param *iluparam)
 *
 * \brief Multi-thread ILU decoposition of a BSR matrix A based on graph coloring
 *
 * \param A         Pointer to dBSRmat matrix
 * \param Ap        Pointer to dCSRmat matrix which provides sparsity pattern
 * \param iludata   Pointer to ILU_data
 * \param iluparam  Pointer to ILU_param
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for 1, 2, 3 nb (Zheng)
 * \note Modified by Chunsheng Feng on 09/06/2017 for iludata->type not inited.
 */
SHORT fasp_ilu_dbsr_setup_mc_omp (dBSRmat    *A,
                                  dCSRmat    *Ap,
                                  ILU_data   *iludata,
                                  ILU_param  *iluparam)
{
    INT status;
    AMG_data *mgl=fasp_amg_data_create(1);
    dCSRmat pp, Ap1;
    dBSRmat A_LU;
    
    if (iluparam->ILU_lfil==0) {  //for ILU0
        mgl[0].A = fasp_dcsr_sympart(Ap);
    }
    else if (iluparam->ILU_lfil==1) {  // for ILU1
        Ap1 = fasp_dcsr_create(Ap->row,Ap->col, Ap->nnz);
        fasp_dcsr_cp(Ap, &Ap1);
        fasp_blas_dcsr_mxm (Ap,&Ap1,&pp);
        mgl[0].A = fasp_dcsr_sympart(&pp);
        fasp_dcsr_free(&Ap1);
        fasp_dcsr_free(&pp);
    }
    
    mgl->num_levels = 20;
    
    mulcol_independ_set(mgl, 1);
    
    A_LU = fasp_dbsr_perm(A, mgl[0].icmap);
    
    // hold color info with nlevl, ilevL and jlevL.
    iludata->nlevL = mgl[0].colors;
    iludata->ilevL = mgl[0].ic;
    iludata->jlevL = mgl[0].icmap;
    iludata->nlevU = 0;
    iludata->ilevU = NULL;
    iludata->jlevU = NULL;
    iludata->A     = NULL; // No need for BSR matrix
    
#if DEBUG_MODE > 0
    if (iluparam->ILU_type == ILUtp) {
        printf("### WARNING: iludata->type = %d not supported!\n",
               iluparam->ILU_type);
    }
#endif

    // setup preconditioner
    iludata->type  = 0; // Must be initialized
    iludata->iperm = NULL;
    
    status = fasp_ilu_dbsr_setup_omp(&A_LU,iludata,iluparam);
    
    fasp_dcsr_free(&mgl[0].A);
    fasp_dbsr_free(&A_LU);
    
    return status;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static INT numfactor (dBSRmat *A, REAL *luval, INT *jlu, INT *uptr)
 * \brief Get numerical ILU decoposition of a BSR matrix A
 *
 * \param A        Pointer to dBSRmat matrix
 * \param luval    Pointer to numerical value of ILU
 * \param jlu      Pointer to the nonzero pattern of ILU
 * \param uptr     Pointer to the diagnal position of ILU
 *
 * \author Shiquan Zhang
 * \date   11/08/2010
 *
 * \note Works for general nb (Xiaozhe)
 */
static INT numfactor (dBSRmat   *A,
                      REAL      *luval,
                      INT       *jlu,
                      INT       *uptr)
{
    INT n=A->ROW,nb=A->nb, nb2=nb*nb, ib, ibstart,ibstart1;
    INT k, indj, inds, indja,jluj, jlus, ijaj;
    REAL  *mult,*mult1;
    INT *colptrs;
    INT status=FASP_SUCCESS;
    
    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
    
    /**
     *     colptrs is used to hold the indices of entries in LU of row k.
     *     It is initialized to zero here, and then reset after each row's
     *     work. The first segment of the loop on indj effectively solves
     *     the transposed upper triangular system
     *            U(1:k-1, 1:k-1)'L(k,1:k-1)' = A(k,1:k-1)'
     *     via sparse saxpy operations, throwing away disallowed fill.
     *     When the loop index indj reaches the k-th column (i.e., the diag
     *     entry), then the innermost sparse saxpy operation effectively is
     *     applying the previous updates to the corresponding part of U via
     *     sparse vec*mat, discarding disallowed fill-in entries, i.e.
     *            U(k,k:n) = A(k,k:n) - U(1:k-1,k:n)*L(k,1:k-1)
     */
    
    //for (k=0;k<n;k++) colptrs[k]=0;
    memset(colptrs, 0, sizeof(INT)*n);
    
    switch (nb) {
            
        case 1:
            
            for (k = 0; k < n; ++k) {
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart=indj*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                }
                
                colptrs[k] =  k;
                
                for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                    ijaj = A->JA[indja];
                    ibstart=colptrs[ijaj]*nb2;
                    ibstart1=indja*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                }
                
                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    
                    jluj = jlu[indj];
                    
                    luval[indj] = luval[indj]*luval[jluj];
                    mult[0] = luval[indj];
                    
                    for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0)
                            luval[colptrs[jlus]] = luval[colptrs[jlus]] - mult[0]*luval[inds];
                    }
                    
                }
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                
                colptrs[k] =  0;
                luval[k] = 1.0/luval[k];
            }
            
            break;
            
        case 3:
            
            for (k = 0; k < n; ++k) {
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart=indj*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                }
                
                colptrs[k] =  k;
                
                for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                    ijaj = A->JA[indja];
                    ibstart=colptrs[ijaj]*nb2;
                    ibstart1=indja*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                }
                
                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];
                    
                    ibstart=indj*nb2;
                    fasp_blas_smat_mul_nc3(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                    
                    for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            fasp_blas_smat_mul_nc3(mult,&(luval[inds*nb2]),mult1);
                            ibstart=colptrs[jlus]*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                        }
                    }
                    
                }
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                
                colptrs[k] =  0;
                
                fasp_smat_inv_nc3(&(luval[k*nb2]));
            }
            
            break;
            
        case -5:
            
            for (k = 0; k < n; ++k) {
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart=indj*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                }
                
                colptrs[k] =  k;
                
                for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                    ijaj = A->JA[indja];
                    ibstart=colptrs[ijaj]*nb2;
                    ibstart1=indja*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                }
                
                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];
                    
                    ibstart=indj*nb2;
                    fasp_blas_smat_mul_nc5(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                    
                    for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            fasp_blas_smat_mul_nc5(mult,&(luval[inds*nb2]),mult1);
                            ibstart=colptrs[jlus]*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                        }
                    }
                    
                }
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                
                colptrs[k] =  0;
                
                // fasp_smat_inv_nc5(&(luval[k*nb2])); // not numerically stable --zcs 04/26/2021
                status = fasp_smat_invp_nc(&(luval[k*nb2]), 5);
            }
            
            break;
            
        case -7:
            
            for (k = 0; k < n; ++k) {
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart=indj*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                }
                
                colptrs[k] =  k;
                
                for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                    ijaj = A->JA[indja];
                    ibstart=colptrs[ijaj]*nb2;
                    ibstart1=indja*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                }
                
                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];
                    
                    ibstart=indj*nb2;
                    fasp_blas_smat_mul_nc7(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                    
                    for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            fasp_blas_smat_mul_nc7(mult,&(luval[inds*nb2]),mult1);
                            ibstart=colptrs[jlus]*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                        }
                    }
                    
                }
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                
                colptrs[k] =  0;
                
                // fasp_smat_inv(&(luval[k*nb2]),nb); // not numerically stable --zcs 04/26/2021
                status = fasp_smat_invp_nc(&(luval[k*nb2]), nb);
            }
            
            break;
            
        default:
            
            for (k=0;k<n;k++) {
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart=indj*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                }
                
                colptrs[k] =  k;
                
                for (indja = A->IA[k]; indja < A->IA[k+1]; indja++) {
                    ijaj = A->JA[indja];
                    ibstart=colptrs[ijaj]*nb2;
                    ibstart1=indja*nb2;
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                }
                
                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];
                    
                    ibstart=indj*nb2;
                    fasp_blas_smat_mul(&(luval[ibstart]),&(luval[jluj*nb2]),mult,nb);
                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                    
                    for (inds = uptr[jluj]; inds < jlu[jluj+1]; inds++) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            fasp_blas_smat_mul(mult,&(luval[inds*nb2]),mult1,nb);
                            ibstart=colptrs[jlus]*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                        }
                    }
                    
                }
                
                for (indj = jlu[k]; indj < jlu[k+1]; ++indj)
                    colptrs[jlu[indj]] = 0;
                
                colptrs[k] =  0;
                
                //fasp_smat_inv(&(luval[k*nb2]),nb); // not numerically stable --zcs 04/26/2021
                status = fasp_smat_invp_nc(&(luval[k * nb2]), nb);
            }
    }
    
    fasp_mem_free(colptrs);  colptrs = NULL;
    fasp_mem_free(mult);     mult    = NULL;
    fasp_mem_free(mult1);    mult1   = NULL;
    
    return status;
}

/**
 * \fn static INT numfactor_mulcol (dBSRmat *A, REAL *luval, INT *jlu,
 *                                  INT *uptr, INT ncolors, INT *ic, INT *icmap)
 * \brief Multi-thread ILU decoposition of a BSR matrix A based on multi-coloring
 *
 * \param A        Pointer to dBSRmat matrix
 * \param luval    Pointer to numerical value of ILU
 * \param jlu      Pointer to the nonzero pattern of ILU
 * \param uptr     Pointer to the diagnal position of ILU
 * \param ncolors  Number of colors of adjacency graph of A
 * \param ic       Pointer to number of vertices in each color
 * \param icmap    Mapping
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for nb = 1, 2, 3 (Zheng)
 */
static INT numfactor_mulcol (dBSRmat   *A,
                             REAL      *luval,
                             INT       *jlu,
                             INT       *uptr,
                             INT        ncolors,
                             INT       *ic,
                             INT       *icmap)
{
    INT status = FASP_SUCCESS;
    
#ifdef _OPENMP
    INT   n = A->ROW, nb = A->nb, nb2 = nb*nb;
    INT   ib, ibstart,ibstart1;
    INT   k, i, indj, inds, indja,jluj, jlus, ijaj, tmp;
    REAL  *mult, *mult1;
    INT   *colptrs;
    
    /**
     *     colptrs is used to hold the indices of entries in LU of row k.
     *     It is initialized to zero here, and then reset after each row's
     *     work. The first segment of the loop on indj effectively solves
     *     the transposed upper triangular system
     *            U(1:k-1, 1:k-1)'L(k,1:k-1)' = A(k,1:k-1)'
     *     via sparse saxpy operations, throwing away disallowed fill.
     *     When the loop index indj reaches the k-th column (i.e., the diag
     *     entry), then the innermost sparse saxpy operation effectively is
     *     applying the previous updates to the corresponding part of U via
     *     sparse vec*mat, discarding disallowed fill-in entries, i.e.
     *            U(k,k:n) = A(k,k:n) - U(1:k-1,k:n)*L(k,1:k-1)
     */
    
    switch (nb) {
            
        case 1:
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,colptrs,tmp)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
#pragma omp for
                    for (k = ic[i]; k < ic[i+1]; ++k) {
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            luval[indj] = luval[indj]*luval[jluj];
                            tmp = luval[indj];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0)
                                    luval[colptrs[jlus]] = luval[colptrs[jlus]] - tmp*luval[inds];
                            }
                            
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        luval[k] = 1.0/luval[k];
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                }
            }
            
            break;
            
        case 2:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (k = ic[i]; k < ic[i+1]; ++k) {
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc2(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc2(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc2(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;
            
        case 3:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (k = ic[i]; k < ic[i+1]; ++k) {
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc3(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc3(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc3(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;
            
        default:
        {
            if (nb > 3) printf("Multi-thread ILU numerical decomposition for %d\
                               components has not been implemented!!!", nb);
            exit(0);
        }
    }
    
#endif
    
    return status;
}

/**
 * \fn static INT numfactor_levsch (dBSRmat *A, REAL *luval, INT *jlu,
 *                                  INT *uptr, INT ncolors, INT *ic, INT *icmap)
 * \brief Multi-thread ILU decoposition of a BSR matrix A based on level schedule strategy
 *
 * \param A        Pointer to dBSRmat matrix
 * \param luval    Pointer to numerical value of ILU
 * \param jlu      Pointer to the nonzero pattern of ILU
 * \param uptr     Pointer to the diagnal position of ILU
 * \param ncolors  Number of colors of adjacency graph of A
 * \param ic       Pointer to number of vertices in each color
 * \param icmap    Mapping
 *
 * \author Zheng Li, Li Zhao
 * \date   12/04/2016, 09/19/2022
 *
 * \note Only works for 1, 2, 3 nb (Zheng)
 * \note works for forall nb = 1,2,... modified by Li Zhao
 */
static INT numfactor_levsch (dBSRmat *A,
                             REAL *luval,
                             INT *jlu,
                             INT *uptr,
                             INT ncolors,
                             INT *ic,
                             INT *icmap)
{
    INT status = FASP_SUCCESS;
    
#ifdef _OPENMP
    INT n = A->ROW, nb = A->nb, nb2 = nb*nb;
    INT ib, ibstart,ibstart1;
    INT k, i, indj, inds, indja, jluj, jlus, ijaj, tmp, ii;
    REAL *mult, *mult1;
    INT  *colptrs;
    
    /**
     *     colptrs is used to hold the indices of entries in LU of row k.
     *     It is initialized to zero here, and then reset after each row's
     *     work. The first segment of the loop on indj effectively solves
     *     the transposed upper triangular system
     *            U(1:k-1, 1:k-1)'L(k,1:k-1)' = A(k,1:k-1)'
     *     via sparse saxpy operations, throwing away disallowed fill.
     *     When the loop index indj reaches the k-th column (i.e., the diag
     *     entry), then the innermost sparse saxpy operation effectively is
     *     applying the previous updates to the corresponding part of U via
     *     sparse vec*mat, discarding disallowed fill-in entries, i.e.
     *            U(k,k:n) = A(k,k:n) - U(1:k-1,k:n)*L(k,1:k-1)
     */
    
    switch (nb) {
            
        case 1:
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,colptrs,tmp)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
#pragma omp for
                    for (k = ic[i]; k < ic[i+1]; ++k) {
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            luval[indj] = luval[indj]*luval[jluj];
                            tmp = luval[indj];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0)
                                    luval[colptrs[jlus]] = luval[colptrs[jlus]] - tmp*luval[inds];
                            }
                            
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        luval[k] = 1.0/luval[k];
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                }
            }
            
            break;
        case 2:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs,ii)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (ii = ic[i]; ii < ic[i+1]; ++ii) {
                        k = icmap[ii];
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc2(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc2(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc2(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;
            
        case 3:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs,ii)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (ii = ic[i]; ii < ic[i+1]; ++ii) {
                        k = icmap[ii];
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc3(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc3(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc3(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;
            
        case 4:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs,ii)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (ii = ic[i]; ii < ic[i+1]; ++ii) {
                        k = icmap[ii];
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc4(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc4(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc4(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;

        case 5:
            
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs,ii)
                {
                    colptrs=(INT*)fasp_mem_calloc(n,sizeof(INT));
                    memset(colptrs, 0, sizeof(INT)*n);
                    mult=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
                    mult1=(REAL*)fasp_mem_calloc(nb2,sizeof(REAL));
#pragma omp for
                    for (ii = ic[i]; ii < ic[i+1]; ++ii) {
                        k = icmap[ii];
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart=indj*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = 0;
                        }
                        colptrs[k] =  k;
                        for (indja = A->IA[k]; indja < A->IA[k+1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart=colptrs[ijaj]*nb2;
                            ibstart1=indja*nb2;
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib] = A->val[ibstart1+ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart=indj*nb2;
                            fasp_blas_smat_mul_nc5(&(luval[ibstart]),&(luval[jluj*nb2]),mult);
                            for (ib=0;ib<nb2;++ib) luval[ibstart+ib]=mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj+1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul_nc5(mult,&(luval[inds*nb2]),mult1);
                                    ibstart=colptrs[jlus]*nb2;
                                    for (ib=0;ib<nb2;++ib) luval[ibstart+ib]-=mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k+1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] =  0;
                        fasp_smat_inv_nc5(&(luval[k*nb2]));
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult    = NULL;
                    fasp_mem_free(mult1);   mult1   = NULL;
                }
            }
            break;

        default:
            for (i = 0; i < ncolors; ++i) {
#pragma omp parallel private(k,indj,ibstart,ib,indja,ijaj,ibstart1,jluj,inds,jlus,mult,mult1,colptrs,ii)
                {
                    colptrs = (INT*)fasp_mem_calloc(n, sizeof(INT));
                    memset(colptrs, 0, sizeof(INT) * n);
                    mult = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));
                    mult1 = (REAL*)fasp_mem_calloc(nb2, sizeof(REAL));
#pragma omp for
                    for (ii = ic[i]; ii < ic[i + 1]; ++ii) {
                        k = icmap[ii];
                        for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                            colptrs[jlu[indj]] = indj;
                            ibstart = indj * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                        }
                        colptrs[k] = k;
                        for (indja = A->IA[k]; indja < A->IA[k + 1]; ++indja) {
                            ijaj = A->JA[indja];
                            ibstart = colptrs[ijaj] * nb2;
                            ibstart1 = indja * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A->val[ibstart1 + ib];
                        }
                        for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                            jluj = jlu[indj];
                            ibstart = indj * nb2;
                            fasp_blas_smat_mul(&(luval[ibstart]), &(luval[jluj * nb2]), mult, nb);
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];
                            for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                                jlus = jlu[inds];
                                if (colptrs[jlus] != 0) {
                                    fasp_blas_smat_mul(mult, &(luval[inds * nb2]), mult1, nb);
                                    ibstart = colptrs[jlus] * nb2;
                                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                                }
                            }
                        }
                        for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;
                        colptrs[k] = 0;
                        fasp_smat_invp_nc(&(luval[k * nb2]), nb);
                    }
                    fasp_mem_free(colptrs); colptrs = NULL;
                    fasp_mem_free(mult);    mult = NULL;
                    fasp_mem_free(mult1);   mult1 = NULL;
                }
            }
            //if (nb > 5) printf("Multi-thread ILU numerical decomposition for %d components has not been implemented!!!\n", nb);
            //exit(0);
            break;
    }
    
#endif
    
    return status;
}

/**
 * \fn static void generate_S_theta (dCSRmat *A, iCSRmat *S, REAL theta)
 *
 * \brief Generate strong sparsity pattern of A
 *
 * \param A      Pointer to input matrix
 * \param S      Pointer to strong sparsity pattern matrix
 * \param theta  Threshold
 *
 * \author Zheng Li, Chunsheng Feng
 * \date   12/04/2016
 */
static void generate_S_theta (dCSRmat *A,
                              iCSRmat *S,
                              REAL     theta)
{
    const INT row=A->row, col=A->col;
    const INT row_plus_one = row+1;
    const INT nnz=A->IA[row]-A->IA[0];
    
    INT index, i, j, begin_row, end_row;
    INT *ia=A->IA, *ja=A->JA;
    REAL *aj=A->val;
    
    // get the diagnal entry of A
    //dvector diag; fasp_dcsr_getdiag(0, A, &diag);
    
    /* generate S */
    REAL row_abs_sum;
    
    // copy the structure of A to S
    S->row=row; S->col=col; S->nnz=nnz; S->val=NULL;
    
    S->IA=(INT*)fasp_mem_calloc(row_plus_one, sizeof(INT));
    
    S->JA=(INT*)fasp_mem_calloc(nnz, sizeof(INT));
    
    fasp_iarray_cp(row_plus_one, ia, S->IA);
    fasp_iarray_cp(nnz, ja, S->JA);
    
    for (i=0;i<row;++i) {
        /* compute scaling factor and row sum */
        row_abs_sum=0;
        
        begin_row=ia[i]; end_row=ia[i+1];
        
        for (j=begin_row;j<end_row;j++) row_abs_sum+=ABS(aj[j]);
        
        row_abs_sum = row_abs_sum*theta;
        
        /* deal with the diagonal element of S */
        //  for (j=begin_row;j<end_row;j++) {
        //     if (ja[j]==i) {S->JA[j]=-1; break;}
        //  }
        
        /* deal with  the element of S */
        for (j=begin_row;j<end_row;j++){
            /* if $\sum_{j=1}^n |a_{ij}|*theta>= |a_{ij}|$ */
            if ( (row_abs_sum >= ABS(aj[j])) && (ja[j] !=i) ) S->JA[j]=-1;
        }
    } // end for i
    
    /* Compress the strength matrix */
    index=0;
    for (i=0;i<row;++i) {
        S->IA[i]=index;
        begin_row=ia[i]; end_row=ia[i+1]-1;
        for (j=begin_row;j<=end_row;j++) {
            if (S->JA[j]>-1) {
                S->JA[index]=S->JA[j];
                index++;
            }
        }
    }
    
    if (index > 0) {
        S->IA[row]=index;
        S->nnz=index;
        S->JA=(INT*)fasp_mem_realloc(S->JA,index*sizeof(INT));
    }
    else {
        S->nnz = 0;
        S->JA = NULL;
    }
}

/**
 * \fn static void multicoloring (AMG_data *mgl, REAL theta, INT *rowmax,
 *                                INT *groups)
 *
 * \brief Coloring vertices of adjacency graph of A
 *
 * \param mgl      Pointer to input matrix
 * \param theta    Threshold in [0,1]
 * \param rowmax   Pointer to number of each color
 * \param groups   Pointer to index array
 *
 * \author Zheng Li, Chunsheng Feng
 * \date   12/04/2016
 */
static void multicoloring (AMG_data *mgl,
                           REAL      theta,
                           INT      *rowmax,
                           INT      *groups)
{
    INT k, i, j, pre, group, iend;
    INT icount;
    INT front, rear;
    INT *IA, *JA;
	INT *cq, *newr;
    
    const INT n = mgl->A.row;
    dCSRmat   A = mgl->A;
    iCSRmat   S;

    S.IA = S.JA = NULL; S.val = NULL;

    theta = MAX(0.0, MIN(1.0, theta));

    if (theta > 0.0 && theta < 1.0) {
        generate_S_theta(&A, &S, theta);
        IA = S.IA;
        JA = S.JA;
    }
    else if (theta == 1.0) {
        
        mgl->ic = (INT*)malloc(sizeof(INT)*2);
        mgl->icmap = (INT *)malloc(sizeof(INT)*(n+1));
        mgl->ic[0] = 0;
        mgl->ic[1] = n;
        for(k=0; k<n; k++)  mgl->icmap[k]= k;
        
        mgl->colors = 1;
        *groups = 1;
        *rowmax = 1;
        
        printf("### WARNING: Theta = %lf! [%s]\n", theta, __FUNCTION__);
        
        return;
    }
    else {
        IA = A.IA;
        JA = A.JA;
    }
    
    cq = (INT *)malloc(sizeof(INT)*(n+1));
    newr = (INT *)malloc(sizeof(INT)*(n+1));
    
#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
    for ( k=0; k<n; k++ ) cq[k]= k;

    group = 0;
    for ( k=0; k<n; k++ ) {
        if ((A.IA[k+1] - A.IA[k]) > group ) group = A.IA[k+1] - A.IA[k];
    }
    *rowmax = group;
    
    mgl->ic = (INT *)malloc(sizeof(INT)*(group+2));
    mgl->icmap = (INT *)malloc(sizeof(INT)*(n+1));
    
    front = n-1;
    rear = n-1;
    
    memset(newr, -1, sizeof(INT)*(n+1));
    memset(mgl->icmap, 0, sizeof(INT)*n);
    
    group=0;
    icount = 0;
    mgl->ic[0] = 0;
    pre=0;
    
    do {
        //front = (front+1)%n;
        front ++;
        if (front == n ) front =0; // front = front < n ? front : 0 ;
        i = cq[front];
        
        if(i <= pre) {
            mgl->ic[group] = icount;
            mgl->icmap[icount] = i;
            group++;
            icount++;
#if 0
            if ((IA[i+1]-IA[i]) > igold)
                iend = MIN(IA[i+1], (IA[i] + igold));
            else
#endif
                iend = IA[i+1];
            
            for (j= IA[i]; j< iend; j++)  newr[JA[j]] = group;
        }
        else if (newr[i] == group) {
            //rear = (rear +1)%n;
            rear ++;
            if (rear == n) rear = 0;
            cq[rear] = i;
        }
        else {
            mgl->icmap[icount] = i;
            icount++;
#if  0
            if ((IA[i+1] - IA[i]) > igold)  iend =MIN(IA[i+1], (IA[i] + igold));
            else
#endif
                iend = IA[i+1];
            for (j = IA[i]; j< iend; j++)  newr[JA[j]] =  group;
        }
        pre=i;
        
    } while(rear != front);
    
    mgl->ic[group] = icount;
    mgl->colors = group;
    *groups = group;
    
    free(cq);
    free(newr);
    
    fasp_mem_free(S.IA); S.IA = NULL;
    fasp_mem_free(S.JA); S.JA = NULL;

    return;
}

/**
 * \fn void topologic_sort_ILU (ILU_data *iludata)
 *
 * \brief Reordering vertices according to level schedule strategy
 *
 * \param iludata  Pointer to iludata
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/04/2016
 */
void topologic_sort_ILU (ILU_data *iludata)
{
    INT i, j, k, l;
    INT nlevL, nlevU;
    
    INT n = iludata->row;
    INT *ijlu = iludata->ijlu;
    
    INT *level = (INT *)fasp_mem_calloc(n, sizeof(INT));
    INT *jlevL = (INT *)fasp_mem_calloc(n, sizeof(INT));
    INT *ilevL = (INT *)fasp_mem_calloc(n+1, sizeof(INT));

    INT *jlevU = (INT *)fasp_mem_calloc(n, sizeof(INT));
    INT *ilevU = (INT *)fasp_mem_calloc(n+1, sizeof(INT));
        
    nlevL = 0;
    ilevL[0] = 0;
    
    // form level for each row of lower triangular matrix.
    for (i=0; i<n; i++) {
        l = 0;
        for(j=ijlu[i]; j<ijlu[i+1]; j++) if (ijlu[j]<=i) l = MAX(l, level[ijlu[j]]);
        level[i] = l+1;
        ilevL[l+1] ++;
        nlevL = MAX(nlevL, l+1);
    }
    
    for (i=1; i<=nlevL; i++) ilevL[i] += ilevL[i-1];
    
    for (i=0; i<n; i++) {
        k = ilevL[level[i]-1];
        jlevL[k] = i;
        ilevL[level[i]-1]++;
    }
    
    for (i=nlevL-1; i>0; i--) ilevL[i] = ilevL[i-1];
    
    // form level for each row of upper triangular matrix.
    nlevU = 0;
    ilevL[0] = 0;
    
    for (i=0; i<n; i++) level[i] = 0;
    
    ilevU[0] = 0;
    
    for (i=n-1; i>=0; i--) {
        l = 0;
        for (j=ijlu[i]; j<ijlu[i+1]; j++) if (ijlu[j]>=i) l = MAX(l, level[ijlu[j]]);
        level[i] = l+1;
        ilevU[l+1] ++;
        nlevU = MAX(nlevU, l+1);
    }
    
    for (i=1; i<=nlevU; i++) ilevU[i] += ilevU[i-1];
    
    for (i=n-1; i>=0; i--) {
        k = ilevU[level[i]-1];
        jlevU[k] = i;
        ilevU[level[i]-1]++;
    }
    
    for (i=nlevU-1; i>0; i--) ilevU[i] = ilevU[i-1];
    
    ilevU[0] = 0;
    
    iludata->nlevL = nlevL+1; iludata->ilevL = ilevL;iludata->jlevL = jlevL;
    iludata->nlevU = nlevU+1; iludata->ilevU = ilevU;iludata->jlevU = jlevU;
    
    fasp_mem_free(level); level = NULL;
}

/**
 * \fn void mulcol_independ_set (AMG_data *mgl, INT gslvl)
 *
 * \brief Multi-coloring vertices of adjacency graph of A
 *
 * \param mgl      Pointer to input matrix
 * \param gslvl    Used to specify levels of AMG using multicolor smoothing
 *
 * \author Zheng Li, Chunsheng Feng
 * \date   12/04/2016
 */
void mulcol_independ_set (AMG_data *mgl,
                          INT       gslvl)
{
    
    INT Colors, rowmax, level, prtlvl = 0;
    
    REAL theta = 0.00;
    
    INT maxlvl = MIN(gslvl, mgl->num_levels-1);
    
#ifdef _OPENMP
#pragma omp parallel for private(level,rowmax,Colors) schedule(static, 1)
#endif
    for ( level=0; level<maxlvl; level++ ) {
        
        multicoloring(&mgl[level], theta, &rowmax, &Colors);
        
        // print
        if ( prtlvl > PRINT_MIN )
            printf("mgl[%3d].A.row = %12d rowmax = %5d rowavg = %7.2lf colors = %5d theta = %le\n",
                   level, mgl[level].A.row, rowmax, (double)mgl[level].A.nnz/mgl[level].A.row,
                   mgl[level].colors, theta);
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
