/*! \file  PreMGRecurAMLI.inl
 *
 *  \brief Subroutines for nonlinear AMLI cycles
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreMGRecurAMLI.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--2021 by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

/*---------------------------------*/
/*--     Private Functions       --*/
/*---------------------------------*/

/**
 * \fn static SHORT Kcycle_dcsr_pgcg (dCSRmat *A, dvector *b,
 *                                    dvector *u, precond *pc)
 *
 * \brief A preconditioned GCR method for solving Au=b
 *
 * \param A    Pointer to the coefficient matrix
 * \param b    Pointer to the dvector of right hand side
 * \param u    Pointer to the dvector of DOFs
 * \param pc   Pointer to the structure of precondition (precond)
 *
 * \author Zheng Li, Chensong Zhang
 * \date   11/09/2014
 *
 * \note   Specially designed for unsmoothed aggregation NAMLI cycle
 *         
 * Reference: YVAN NOTAY, "AN AGGREGATION-BASED ALGEBRAIC MULTIGRID METHOD", 2010
 */
static SHORT Kcycle_dcsr_pgcg (dCSRmat   *A,
                               dvector   *b,
                               dvector   *u,
                               precond   *pc)
{
    REAL   absres, relres, normb;
    REAL   alpha1, alpha2, gamma, rho1, rho2, beta1, beta2, beta3, beta4;
    REAL   *work, *r, *x1, *v1, *v2;
    
    INT    m=A->row;
    REAL   *x = u->val;
    
    // allocate temp memory
    work = (REAL *)fasp_mem_calloc(4*m,sizeof(REAL));
    r = work; x1 = r + m; v1 = r + 2*m; v2 = r + 3*m;
    
    normb = fasp_blas_darray_norm2(m, b->val);
    
    fasp_darray_cp(m, b->val, r);
    
    // preconditioning
    if (pc != NULL)
        pc->fct(r, x, pc->data);
    else
        fasp_darray_cp(m, r, x);
    
    // v1 = A*p
    fasp_blas_dcsr_mxv(A, x, v1);
    
    // rho1 = (p,v1)
    rho1 = fasp_blas_darray_dotprod (m, x, v1);
     
    // alpha1 = (p, r)
    alpha1 = fasp_blas_darray_dotprod (m, x, r);
    
    beta1 = alpha1/rho1;
    
    // r = r - beta1 *v1
    fasp_blas_darray_axpy(m, -beta1, v1, r);
    
    // norm(r)
    absres = fasp_blas_darray_norm2(m, r);
    
    // compute relative residual
    relres = absres/normb;
    
    // if relres reaches tol(0.2), pgcg will stop,
    // otherwise, another one pgcg iteration will do.
    if (relres < 0.2) {
        fasp_blas_darray_ax(m, beta1, x);
        goto FINISH;
    }
    
    // preconditioning
    if (pc != NULL)
        pc->fct(r, x1, pc->data);
    else
        fasp_darray_cp(m, r, x1);
    
    // v2 = A*p
    fasp_blas_dcsr_mxv(A, x1, v2);
    
    // gamma = (x1,v1)
    gamma = fasp_blas_darray_dotprod (m, x1, v1);
    
    // alpha2 = (x1,r)
    alpha2  = fasp_blas_darray_dotprod(m, x1, r);
    
    // rho2 = (x1,v2)
    rho2 = fasp_blas_darray_dotprod(m, x1, v2);
    
    beta2 = rho2 - gamma*gamma/rho1;
    if (ABS(beta2) < SMALLREAL) goto FINISH; // if beta2 is zero, stop --ZCS 01/15/2021

    beta3 = (alpha1 - gamma*alpha2/beta2)/rho1;
    beta4 = alpha2/beta2;
    
    fasp_blas_darray_ax(m, beta3, x);
    
    fasp_blas_darray_axpy(m, beta4, x1, x);
    
FINISH:
    fasp_mem_free(work); work = NULL;

    return FASP_SUCCESS;
}

/**
 * \fn static SHORT Kcycle_dcsr_pgcr (dCSRmat *A, dvector *b,
 *                                    dvector *u, precond *pc)
 *
 * \brief A preconditioned GCR method for solving Au=b
 *
 * \param A    Pointer to the coefficient matrix
 * \param b    Pointer to the dvector of right hand side
 * \param u    Pointer to the dvector of DOFs
 * \param pc   Pointer to the structure of precondition (precond)
 *
 * \author Zheng Li, Chensong Zhang
 * \date   11/09/2014
 *
 * \note   Specially designed for unsmoothed aggregation NAMLI cycle
 */
static SHORT Kcycle_dcsr_pgcr (dCSRmat   *A,
                               dvector   *b,
                               dvector   *u,
                               precond   *pc)
{
    REAL   absres = BIGREAL;
    REAL   relres  = BIGREAL, normb  = BIGREAL;
    REAL   alpha, alpha1, alpha2, alpha3, alpha4, beta, gamma, rho1, rho2;
    
    INT    m=A->row;
    REAL   *x = u->val;
    
    // allocate temp memory
    REAL *work, *r, *x1, *v1, *v2;
    work = (REAL *)fasp_mem_calloc(4*m,sizeof(REAL));
    r = work; x1 = r + m; v1 = r + 2*m; v2 = r + 3*m;
    
    normb=fasp_blas_darray_norm2(m, b->val);
    fasp_darray_cp(m, b->val, r);
    
    // preconditioning
    if (pc != NULL)
        pc->fct(r, x, pc->data);
    else
        fasp_darray_cp(m, r, x);
    
    // v1 = A*x
    fasp_blas_dcsr_mxv(A, x, v1);
    // rho1 = (v1,v1)
    rho1 = fasp_blas_darray_dotprod (m, v1, v1);
    // alpha1 = (r, v1)
    alpha1 = fasp_blas_darray_dotprod (m, v1, r);
    alpha  = alpha1/rho1;
    
    // r = r - alpha *v1
    fasp_blas_darray_axpy(m, -alpha, v1, r);
    
    // norm(r)
    absres = fasp_blas_darray_norm2(m, r);
    
    // compute relative residual
    relres = absres/normb;
    
    // if relres reaches tol(0.2), pgcr will stop,
    // otherwise, another one pgcr iteration will do.
    if (relres < 0.2) {
        fasp_blas_darray_ax(m, alpha, x);
        goto FINISH;
    }
    
    // preconditioning
    if (pc != NULL)
        pc->fct(r, x1, pc->data);
    else
        fasp_darray_cp(m, r, x1);
    
    // v2 = A*x1
    fasp_blas_dcsr_mxv(A, x1, v2);
    
    // gamma = (v1,v2)
    gamma = fasp_blas_darray_dotprod(m, v1, v2);

    // beta = (v2,v2)
    beta  = fasp_blas_darray_dotprod(m, v2, v2);

    // alpha2 = (r,v2)
    alpha2 = fasp_blas_darray_dotprod(m, r, v2);
    rho2   = beta - gamma*gamma/rho1;    
    alpha3 = alpha1/rho1 - gamma*alpha2/(rho1*rho2);
    alpha4 = alpha2/rho2;
    
    // x = alpha3*x + alpha4*x1
    fasp_blas_darray_ax(m, alpha3, x);
    fasp_blas_darray_axpy(m, alpha4, x1, x);
    
FINISH:
    fasp_mem_free(work); work = NULL;
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
