/*! \file  PreCSR.c
 *
 *  \brief Preconditioners for dCSRmat matrices
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxParam.c, AuxVector.c, BlaILUSetupCSR.c,
 *         BlaSchwarzSetup.c, BlaSparseCSR.c, BlaSpmvCSR.c, KrySPcg.c,
 *         KrySPvgmres.c, PreAMGSetupRS.c, PreAMGSetupSA.c, PreAMGSetupUA.c,
 *         PreDataInit.c, PreMGCycle.c, PreMGCycleFull.c, and PreMGRecurAMLI.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "PreMGUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn precond *fasp_precond_setup (const SHORT precond_type, AMG_param *amgparam,
 *                                  ILU_param *iluparam, dCSRmat *A)
 *
 * \brief Setup preconditioner interface for iterative methods
 *
 * \param precond_type   Preconditioner type
 * \param amgparam       Pointer to AMG parameters
 * \param iluparam       Pointer to ILU parameters
 * \param A              Pointer to the coefficient matrix
 *
 * \return               Pointer to preconditioner
 *
 * \author Feiteng Huang
 * \date   05/18/2009
 */
precond *fasp_precond_setup (const SHORT   precond_type,
                             AMG_param    *amgparam,
                             ILU_param    *iluparam,
                             dCSRmat      *A)
{
    precond           *pc = NULL;
    AMG_data         *mgl = NULL;
    precond_data  *pcdata = NULL;
    ILU_data         *ILU = NULL;
    dvector         *diag = NULL;

    INT           max_levels, nnz, m, n;
    
    switch (precond_type) {
            
    case PREC_AMG: // AMG preconditioner
            
        pc = (precond *)fasp_mem_calloc(1, sizeof(precond));
        max_levels = amgparam->max_levels;
        nnz = A->nnz; m = A->row; n = A->col;
            
        // initialize A, b, x for mgl[0]    
        mgl=fasp_amg_data_create(max_levels);
        mgl[0].A=fasp_dcsr_create(m,n,nnz); fasp_dcsr_cp(A,&mgl[0].A);
        mgl[0].b=fasp_dvec_create(n); mgl[0].x=fasp_dvec_create(n); 
            
        // setup preconditioner  
        switch (amgparam->AMG_type) {
        case SA_AMG: // Smoothed Aggregation AMG
            fasp_amg_setup_sa(mgl, amgparam); break;
        case UA_AMG: // Unsmoothed Aggregation AMG
            fasp_amg_setup_ua(mgl, amgparam); break;
        default: // Classical AMG
            fasp_amg_setup_rs(mgl, amgparam); break;
        }
            
        pcdata = (precond_data *)fasp_mem_calloc(1, sizeof(precond_data));
        fasp_param_amg_to_prec(pcdata, amgparam);
        pcdata->max_levels = mgl[0].num_levels;
        pcdata->mgl_data = mgl;
            
        pc->data = pcdata;
            
        switch (amgparam->cycle_type) {
        case AMLI_CYCLE: // AMLI cycle
            pc->fct = fasp_precond_amli; break;
        case NL_AMLI_CYCLE: // Nonlinear AMLI
            pc->fct = fasp_precond_namli; break;
        default: // V,W-cycles or hybrid cycles
            pc->fct = fasp_precond_amg; break;
        }
            
        break;
            
    case PREC_FMG: // FMG preconditioner
            
        pc = (precond *)fasp_mem_calloc(1, sizeof(precond));
        max_levels = amgparam->max_levels;
        nnz = A->nnz; m = A->row; n = A->col;
            
        // initialize A, b, x for mgl[0]    
        mgl=fasp_amg_data_create(max_levels);
        mgl[0].A=fasp_dcsr_create(m,n,nnz); fasp_dcsr_cp(A,&mgl[0].A);
        mgl[0].b=fasp_dvec_create(n); mgl[0].x=fasp_dvec_create(n); 
            
        // setup preconditioner  
        switch (amgparam->AMG_type) {
        case SA_AMG: // Smoothed Aggregation AMG
            fasp_amg_setup_sa(mgl, amgparam); break;
        case UA_AMG: // Unsmoothed Aggregation AMG
            fasp_amg_setup_ua(mgl, amgparam); break;
        default: // Classical AMG
            fasp_amg_setup_rs(mgl, amgparam); break;
        }
            
        pcdata = (precond_data *)fasp_mem_calloc(1, sizeof(precond_data));
        fasp_param_amg_to_prec(pcdata, amgparam);
        pcdata->max_levels = mgl[0].num_levels;
        pcdata->mgl_data = mgl;
            
        pc->data = pcdata; pc->fct = fasp_precond_famg;
            
        break;
            
    case PREC_ILU: // ILU preconditioner
            
        pc = (precond *)fasp_mem_calloc(1, sizeof(precond));
        ILU = (ILU_data *)fasp_mem_calloc(1, sizeof(ILU_data));
        fasp_ilu_dcsr_setup(A, ILU, iluparam);
        pc->data = ILU;
        pc->fct = fasp_precond_ilu;
            
        break;
            
    case PREC_DIAG: // Diagonal preconditioner
            
        pc = (precond *)fasp_mem_calloc(1, sizeof(precond));
        diag = (dvector *)fasp_mem_calloc(1, sizeof(dvector));
        fasp_dcsr_getdiag(0, A, diag);    
            
        pc->data = diag; 
        pc->fct  = fasp_precond_diag;
            
        break;

    default: // No preconditioner
            
        break;
            
    }
    
    return pc;
}

/**
 * \fn void fasp_precond_diag (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 */
void fasp_precond_diag (REAL *r, 
                        REAL *z, 
                        void *data)
{
    dvector *diag=(dvector *)data;
    REAL *diagptr=diag->val;
    INT i, m=diag->row;    
    
    memcpy(z,r,m*sizeof(REAL));
    for (i=0;i<m;++i) {
        if (ABS(diag->val[i])>SMALLREAL) z[i]/=diagptr[i];
    }    
}

/**
 * \fn void fasp_precond_ilu (REAL *r, REAL *z, void *data)
 *
 * \brief ILU preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang
 * \date   04/06/2010
 */
void fasp_precond_ilu (REAL *r, 
                       REAL *z, 
                       void *data)
{
    ILU_data *iludata=(ILU_data *)data;
    const INT m=iludata->row, mm1=m-1, memneed=2*m;
    REAL *zz, *zr;
    
    if (iludata->nwork<memneed) goto MEMERR; // check this outside this subroutine!!
    
    zz = iludata->work; 
    zr = iludata->work+m;
    fasp_darray_cp(m, r, zr);     
    
    {
        INT i, j, jj, begin_row, end_row, mm2=m-2;
        INT *ijlu=iludata->ijlu;
        REAL *lu=iludata->luval;
    
        // forward sweep: solve unit lower matrix equation L*zz=zr
        zz[0]=zr[0];

        for (i=1;i<=mm1;++i) {
            begin_row=ijlu[i]; end_row=ijlu[i+1]-1;
            for (j=begin_row;j<=end_row;++j) {
                jj=ijlu[j];
                if (jj<i) zr[i]-=lu[j]*zz[jj];
                else break;
            }
            zz[i]=zr[i]; 
        }
    
        // backward sweep: solve upper matrix equation U*z=zz
        z[mm1]=zz[mm1]*lu[mm1];
        for (i=mm2;i>=0;i--) {
            begin_row=ijlu[i]; end_row=ijlu[i+1]-1;
            for (j=end_row;j>=begin_row;j--) {
                jj=ijlu[j];
                if (jj>i) zz[i]-=lu[j]*z[jj];
                else break;
            } 
            z[i]=zz[i]*lu[i];
        }
    }
    
    return;
    
MEMERR:
    printf("### ERROR: Need %d memory, only %d available!\n",
           memneed, iludata->nwork);
    fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
}

/**
 * \fn void fasp_precond_ilu_forward (REAL *r, REAL *z, void *data)
 *
 * \brief ILU preconditioner: only forward sweep
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu, Shiquang Zhang
 * \date   04/06/2010
 */
void fasp_precond_ilu_forward (REAL *r, 
                               REAL *z, 
                               void *data)
{
    ILU_data *iludata=(ILU_data *)data;
    const INT m=iludata->row, mm1=m-1, memneed=2*m;
    REAL *zz, *zr;
    
    if (iludata->nwork<memneed) goto MEMERR; 
    
    zz = iludata->work; 
    zr = iludata->work+m;
    fasp_darray_cp(m, r, zr);     
    
    {
        INT i, j, jj, begin_row, end_row;
        INT *ijlu=iludata->ijlu;
        REAL *lu=iludata->luval;
    
        // forward sweep: solve unit lower matrix equation L*z=r
        zz[0]=zr[0];
        for (i=1;i<=mm1;++i) {
            begin_row=ijlu[i]; end_row=ijlu[i+1]-1;
            for (j=begin_row;j<=end_row;++j) {
                jj=ijlu[j];
                if (jj<i) zr[i]-=lu[j]*zz[jj];
                else break;
            }
            zz[i]=zr[i];
        }
    }
    
    fasp_darray_cp(m, zz, z); 
    
    return;
    
MEMERR:
    printf("### ERROR: Need %d memory, only %d available!",
           memneed, iludata->nwork);
    fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
}

/**
 * \fn void fasp_precond_ilu_backward (REAL *r, REAL *z, void *data)
 *
 * \brief ILU preconditioner: only backward sweep
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu, Shiquan  Zhang
 * \date   04/06/2010
 */
void fasp_precond_ilu_backward (REAL *r, 
                                REAL *z, 
                                void *data)
{
    ILU_data *iludata=(ILU_data *)data;
    const INT m=iludata->row, mm1=m-1, memneed=2*m;
    REAL *zz;
    
    if (iludata->nwork<memneed) goto MEMERR; 
    
    zz = iludata->work; 
    fasp_darray_cp(m, r, zz);     
    
    {
        INT i, j, jj, begin_row, end_row, mm2=m-2;
        INT *ijlu=iludata->ijlu;
        REAL *lu=iludata->luval;
    
        // backward sweep: solve upper matrix equation U*z=zz
        z[mm1]=zz[mm1]*lu[mm1];
        for (i=mm2;i>=0;i--) {
            begin_row=ijlu[i]; end_row=ijlu[i+1]-1;
            for (j=end_row;j>=begin_row;j--) {
                jj=ijlu[j];
                if (jj>i) zz[i]-=lu[j]*z[jj];
                else break;
            } 
            z[i]=zz[i]*lu[i];
        }
    
    }
    
    return;
    
MEMERR:
    printf("### ERROR: Need %d memory, only %d available!",
           memneed, iludata->nwork);
    fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
}

/**
 * \fn void fasp_precond_swz (REAL *r, REAL *z, void *data)
 *
 * \brief get z from r by Schwarz
 *
 * \param r     Pointer to residual
 * \param z     Pointer to preconditioned residual
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   03/22/2010
 *
 * \note Change Schwarz interface by Zheng Li on 11/18/2014
 */
void fasp_precond_swz (REAL *r,
                       REAL *z,
                       void *data)
{
	SWZ_data  * swzdata  = (SWZ_data *)data;
    SWZ_param * swzparam = swzdata->swzparam;
	const INT   swztype  = swzdata->SWZ_type;
    const INT   n        = swzdata->A.row;
	
    dvector x, b;

    fasp_dvec_alloc(n, &x);
    fasp_dvec_alloc(n, &b);
    fasp_darray_cp(n, r, b.val);

    fasp_dvec_set(n, &x, 0);

	switch (swztype) {
		case SCHWARZ_BACKWARD:
			fasp_dcsr_swz_backward(swzdata, swzparam, &x, &b);
			break;
		case SCHWARZ_SYMMETRIC:
			fasp_dcsr_swz_forward(swzdata, swzparam, &x, &b);
			fasp_dcsr_swz_backward(swzdata, swzparam, &x, &b);
			break;
		default:
			fasp_dcsr_swz_forward(swzdata, swzparam, &x, &b);
			break;
	}

    fasp_darray_cp(n, x.val, z);
}

/**
 * \fn void fasp_precond_amg (REAL *r, REAL *z, void *data)
 *
 * \brief AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Chensong Zhang
 * \date   04/06/2010
 */
void fasp_precond_amg (REAL *r, 
                       REAL *z, 
                       void *data)
{
    precond_data *pcdata=(precond_data *)data;
    const INT m=pcdata->mgl_data[0].A.row;
    const INT maxit=pcdata->maxit;
    INT i;
    
    AMG_param amgparam; fasp_param_amg_init(&amgparam);
    fasp_param_prec_to_amg(&amgparam,pcdata);
        
    AMG_data *mgl = pcdata->mgl_data;
    mgl->b.row=m; fasp_darray_cp(m,r,mgl->b.val); // residual is an input 
    mgl->x.row=m; fasp_dvec_set(m,&mgl->x,0.0);
    
    for ( i=maxit; i--; ) fasp_solver_mgcycle(mgl,&amgparam);
    
    fasp_darray_cp(m,mgl->x.val,z);    
}

/**
 * \fn void fasp_precond_famg (REAL *r, REAL *z, void *data)
 *
 * \brief Full AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/27/2011
 */
void fasp_precond_famg (REAL *r, 
                        REAL *z, 
                        void *data)
{
    precond_data *pcdata=(precond_data *)data;
    const INT m=pcdata->mgl_data[0].A.row;
    const INT maxit=pcdata->maxit;
    INT i;
    
    AMG_param amgparam; fasp_param_amg_init(&amgparam);
    fasp_param_prec_to_amg(&amgparam,pcdata);
    
    AMG_data *mgl = pcdata->mgl_data;
    mgl->b.row=m; fasp_darray_cp(m,r,mgl->b.val); // residual is an input 
    mgl->x.row=m; fasp_dvec_set(m,&mgl->x,0.0);
    
    for ( i=maxit; i--; ) fasp_solver_fmgcycle(mgl,&amgparam);
    
    fasp_darray_cp(m,mgl->x.val,z);    
}

/**
 * \fn void fasp_precond_amli(REAL *r, REAL *z, void *data)
 *
 * \brief AMLI AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   01/23/2011
 */
void fasp_precond_amli (REAL *r, 
                        REAL *z, 
                        void *data)
{
    precond_data *pcdata=(precond_data *)data;
    const INT m=pcdata->mgl_data[0].A.row;
    const INT maxit=pcdata->maxit;
    INT i;
    
    AMG_param amgparam; fasp_param_amg_init(&amgparam);
    fasp_param_prec_to_amg(&amgparam,pcdata);
    
    AMG_data *mgl = pcdata->mgl_data;
    mgl->b.row=m; fasp_darray_cp(m,r,mgl->b.val); // residual is an input 
    mgl->x.row=m; fasp_dvec_set(m,&mgl->x,0.0);
    
    for ( i=maxit; i--; ) fasp_solver_amli(mgl,&amgparam,0);
    
    fasp_darray_cp(m,mgl->x.val,z);    
}

/**
 * \fn void fasp_precond_namli (REAL *r, REAL *z, void *data)
 *
 * \brief Nonlinear AMLI AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   04/25/2011
 */
void fasp_precond_namli (REAL *r, 
                         REAL *z,
                         void *data)
{
    precond_data *pcdata=(precond_data *)data;
    const INT m=pcdata->mgl_data[0].A.row;
    const INT maxit=pcdata->maxit;
    const SHORT num_levels = pcdata->max_levels;
    INT i;
    
    AMG_param amgparam; fasp_param_amg_init(&amgparam);
    fasp_param_prec_to_amg(&amgparam,pcdata);
    
    AMG_data *mgl = pcdata->mgl_data;
    mgl->b.row=m; fasp_darray_cp(m,r,mgl->b.val); // residual is an input 
    mgl->x.row=m; fasp_dvec_set(m,&mgl->x,0.0);

    for ( i=maxit; i--; ) fasp_solver_namli(mgl, &amgparam, 0, num_levels);
    fasp_darray_cp(m,mgl->x.val,z);    
}
    
/**
 * \fn void fasp_precond_amg_nk (REAL *r, REAL *z, void *data)
 *
 * \brief AMG with extra near kernel solve as preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   05/26/2014
 */
void fasp_precond_amg_nk (REAL *r,
                          REAL *z,
                          void *data)
{
    precond_data *pcdata=(precond_data *)data;
    const INT m=pcdata->mgl_data[0].A.row;
    const INT maxit=pcdata->maxit;
    INT i;
    
    dCSRmat *A_nk = pcdata->A_nk;
    dCSRmat *P_nk = pcdata->P_nk;
    dCSRmat *R_nk = pcdata->R_nk;
    
    fasp_darray_set(m, z, 0.0);

    // local variables
    dvector r_nk, z_nk;
    fasp_dvec_alloc(A_nk->row, &r_nk);
    fasp_dvec_alloc(A_nk->row, &z_nk);
    
    //----------------------
    // extra kernel solve
    //----------------------
    // r_nk = R_nk*r
    fasp_blas_dcsr_mxv(R_nk, r, r_nk.val);
    
    // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
    fasp_solver_umfpack(A_nk, &r_nk, &z_nk, 0);
#else
    fasp_coarse_itsolver(A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif
    
    // z = z + P_nk*z_nk;
    fasp_blas_dcsr_aAxpy(1.0, P_nk, z_nk.val, z);
    
    //----------------------
    // AMG solve
    //----------------------
    AMG_param amgparam; fasp_param_amg_init(&amgparam);
    fasp_param_prec_to_amg(&amgparam,pcdata);
    
    AMG_data *mgl = pcdata->mgl_data;
    mgl->b.row=m; fasp_darray_cp(m,r,mgl->b.val); // residual is an input
    mgl->x.row=m; //fasp_dvec_set(m,&mgl->x,0.0);
    fasp_darray_cp(m, z, mgl->x.val);
    
    for ( i=maxit; i--; ) fasp_solver_mgcycle(mgl,&amgparam);
    
    fasp_darray_cp(m,mgl->x.val,z);

    //----------------------
    // extra kernel solve
    //----------------------
    // r = r - A*z
    fasp_blas_dcsr_aAxpy(-1.0, &(pcdata->mgl_data[0].A), z, mgl->b.val);
    
    // r_nk = R_nk*r
    fasp_blas_dcsr_mxv(R_nk, mgl->b.val, r_nk.val);
    
    // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
    fasp_solver_umfpack(A_nk, &r_nk, &z_nk, 0);
#else
    fasp_coarse_itsolver(A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif
    
    // z = z + P_nk*z_nk;
    fasp_blas_dcsr_aAxpy(1.0, P_nk, z_nk.val, z);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
