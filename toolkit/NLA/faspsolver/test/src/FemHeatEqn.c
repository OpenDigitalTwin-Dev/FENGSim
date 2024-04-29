/*! \file  FemHeatEqn.c
 *
 *  \brief Setup P1 FEM & backward Euler for the heat transfer's equation
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#define DIM 2
#define MAX_QUAD 49    // max num of quadrature points allowed

#include "heat_fem.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "FemBasis.inl"

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/** 
 * \fn static void localb (REAL (*node)[2], REAL *b, INT num_qp, INT nt,
 *                         REAL dt)
 *
 * \brief Form local right-hand side b from triangle node
 *
 * \param (*node)[2]   vertices of the triangle
 * \param *b           local right-hand side
 * \param num_qp       number of quad point
 * \param nt           number of time steps
 * \param dt           size of time step
 *
 * \author Feiteng Huang
 * \date   02/23/2012
 *
 * Modify by Feiteng Huang 04/01/2012: for heat transfer
 */
static void localb (REAL (*node)[2],
                    REAL *b,
                    INT num_qp,
                    INT nt,
                    REAL dt)
{
    const REAL s=2.0*areaT(node[0][0],node[1][0],node[2][0],
                             node[0][1],node[1][1],node[2][1]);
    REAL p[DIM+1],a;
    REAL gauss[MAX_QUAD][3],g;
    INT i, j;
    for (i=0;i<3*nt;++i)
        b[i] = 0;
    
    fasp_gauss2d(num_qp, 2, gauss); // Gauss integration initial
    
    for (i=0;i<num_qp;++i) {
        g = 1-gauss[i][0]-gauss[i][1];
        for (j=0;j<DIM;++j)
            p[j]=node[0][j]*gauss[i][0]+node[1][j]*gauss[i][1]+node[2][j]*g;
        for (j=0;j<nt;++j) {
            p[DIM]=dt*(j+1);
            a=f(p);
            b[0+3*j]+=s*a*gauss[i][2]*gauss[i][0];
            b[1+3*j]+=s*a*gauss[i][2]*gauss[i][1];
            b[2+3*j]+=s*a*gauss[i][2]*g;
        }
    }
}

/**
 * \fn static void assemble_stiffmat (dCSRmat *A, dCSRmat *M, dvector *b, Mesh *mesh,
 *                                    Mesh_aux *mesh_aux, FEM_param *pt, REAL dt)
 *
 * \brief Assemble stiffness matrix and right-hand side
 *
 * \param *A                           pointer to stiffness matrix
 * \param *M                           pointer to mass matrix
 * \param *b                           pointer to right hand side
 * \param *mesh                        pointer to mesh info
 * \param *mesh_aux                    pointer to auxiliary mesh info
 * \param *pt                          pointer to parameter
 * \param dt                           size of time step
 *
 * \note This subroutine follows Ludmil' note on CSR.
 *
 * \author Feiteng Huang
 * \date   02/23/2012
 *
 * Modified by Feiteng Huang on 04/06/2012: restructure assembling
 */
static void assemble_stiffmat (dCSRmat *A, 
                               dCSRmat *M,
                               dvector *b,
                               Mesh *mesh,
                               Mesh_aux *mesh_aux,
                               FEM_param *pt,
                               REAL dt)
{
    // assemble option
    INT mat = 0, rhs = 0;
    if ( strcmp(pt->option,"ab") == 0 ) { mat = 1; rhs = 1; }
    if ( strcmp(pt->option,"a") == 0 ) mat = 1;
    if ( strcmp(pt->option,"b") == 0 ) rhs = 1;
    
    const INT num_node = mesh->node.row;
    const INT num_edge = mesh_aux->edge.row;
    const INT nnz = num_node + 2*num_edge;
    const REAL epsilon=1;
    
    REAL T[3][2],phi1[2],phi2[2],phi_1,phi_2;
    REAL gauss[MAX_QUAD][DIM+1];
    REAL s;
    
    INT i,j,k,it;
    INT k1,n1,n2,i1;
    REAL tmp_a;
    REAL *btmp = (REAL*)fasp_mem_calloc(3*pt->nt, sizeof(REAL));
    INT tmp, edge_c;
    
    fasp_gauss2d(pt->num_qp_mat, 2, gauss); // Gauss integration initial
    
    // alloc mem for A & b
    A->row = A->col = num_node;
    M->row = M->col = num_node;
    M->nnz = A->nnz = nnz;
    A->IA = (INT*)fasp_mem_calloc(A->row+1, sizeof(INT));
    M->IA = (INT*)fasp_mem_calloc(A->row+1, sizeof(INT));
    A->JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    M->JA = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    A->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    M->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    b->row = num_node*pt->nt;
    b->val = (REAL*)fasp_mem_calloc(b->row, sizeof(REAL));
    INT *count = (INT*)fasp_mem_calloc(num_node, sizeof(INT));
    
    //edge to global idx of A->val
    INT *edge2idx_g1 = (INT*)fasp_mem_calloc(num_edge, sizeof(INT));
    INT *edge2idx_g2 = (INT*)fasp_mem_calloc(num_edge, sizeof(INT));
    
    // get IA
    for (i=0;i<num_edge;++i) {
        n1 = mesh_aux->edge.val[i][0];
        n2 = mesh_aux->edge.val[i][1];
        A->IA[n1+1] += 1;
        M->IA[n1+1] += 1;
        A->IA[n2+1] += 1;
        M->IA[n2+1] += 1;
    }

    for (i=0;i<num_node;++i) {
        A->IA[i+1] += A->IA[i] + 1;
        M->IA[i+1] += A->IA[i] + 1;
        A->JA[A->IA[i]] = i;
        M->JA[A->IA[i]] = i;
    }
    
    // get JA
    for (i=0;i<num_edge;++i) {
        n1 = mesh_aux->edge.val[i][0];
        n2 = mesh_aux->edge.val[i][1];
        count[n1]++;
        count[n2]++;
        A->JA[A->IA[n1]+count[n1]] = n2;
        M->JA[A->IA[n1]+count[n1]] = n2;
        A->JA[A->IA[n2]+count[n2]] = n1;
        M->JA[A->IA[n2]+count[n2]] = n1;
        edge2idx_g1[i] = A->IA[n1]+count[n1];
        edge2idx_g2[i] = A->IA[n2]+count[n2];
    }
    fasp_mem_free(count);
    
    // Loop element by element and compute the actual entries storing them in A
    if (rhs && mat) {
        for (k=0;k<mesh->elem.row;++k) {

            for (i=0;i<mesh->elem.col;++i) {
                j=mesh->elem.val[k][i];
                T[i][0]=mesh->node.val[j][0];
                T[i][1]=mesh->node.val[j][1];
            } // end for i
            s=areaT(T[0][0], T[1][0], T[2][0], T[0][1], T[1][1], T[2][1]);

            localb(T,btmp,pt->num_qp_rhs,pt->nt,dt);

            tmp=0;
            for (k1=0;k1<mesh->elem.col;++k1) {
                //mesh->elem.col == mesh_aux->elem2edge.col
                edge_c = mesh_aux->elem2edge.val[k][k1];
                i=mesh->elem.val[k][k1];

                for (it=0;it<pt->nt;it++)
                    b->val[i+it*num_node]+=btmp[it*3+tmp];
                tmp++;

                // for the diag entry
                gradBasisP1(T, s, k1, phi1);
                tmp_a = 2*s*(phi1[0]*phi1[0]+phi1[1]*phi1[1]*epsilon);
                for (i1=0;i1<pt->num_qp_mat;i1++) {
                    A->val[A->IA[i]] += gauss[i1][2]*tmp_a;
                    phi_1 = basisP1(k1, gauss[i1]);
                    M->val[A->IA[i]] += 2*s*gauss[i1][2]*(phi_1*phi_1);
                } // end for i1

                // for the off-diag entry
                i = edge2idx_g1[edge_c];
                j = edge2idx_g2[edge_c];
                n1 = (k1+1)%3;//local node index in the elem
                n2 = (k1+2)%3;
                gradBasisP1(T, s, n1, phi1);
                gradBasisP1(T, s, n2, phi2);
                tmp_a = 2*s*(phi1[0]*phi2[0]+phi1[1]*phi2[1]*epsilon);
                for (i1=0;i1<pt->num_qp_mat;i1++) {
                    A->val[i] += gauss[i1][2]*tmp_a;
                    A->val[j] += gauss[i1][2]*tmp_a;
                    phi_1 = basisP1(n1, gauss[i1]);
                    phi_2 = basisP1(n2, gauss[i1]);
                    M->val[i] += 2*s*gauss[i1][2]*(phi_1*phi_2);
                    M->val[j] += 2*s*gauss[i1][2]*(phi_1*phi_2);
                } // end for i1
            } // end for k1
        } // end for k
    }// end if
    else if (rhs) {
        for (k=0;k<mesh->elem.row;++k) {

            for (i=0;i<mesh->elem.col;++i) {
                j=mesh->elem.val[k][i];
                T[i][0]=mesh->node.val[j][0];
                T[i][1]=mesh->node.val[j][1];
            } // end for i
            s=areaT(T[0][0], T[1][0], T[2][0], T[0][1], T[1][1], T[2][1]);

            localb(T,btmp,pt->num_qp_rhs,pt->nt,dt);

            tmp=0;
            for (k1=0;k1<mesh->elem.col;++k1) {
                //mesh->elem.col == mesh_aux->elem2edge.col
                edge_c = mesh_aux->elem2edge.val[k][k1];
                i=mesh->elem.val[k][k1];

                for (it=0;it<pt->nt;it++)
                    b->val[i+it*num_node]+=btmp[it*3+tmp];
                tmp++;
            } // end for k1
        } // end for k
    }
    else if (mat) {
        for (k=0;k<mesh->elem.row;++k) {

            for (i=0;i<mesh->elem.col;++i) {
                j=mesh->elem.val[k][i];
                T[i][0]=mesh->node.val[j][0];
                T[i][1]=mesh->node.val[j][1];
            } // end for i
            s=areaT(T[0][0], T[1][0], T[2][0], T[0][1], T[1][1], T[2][1]);

            for (k1=0;k1<mesh->elem.col;++k1) {
                edge_c = mesh_aux->elem2edge.val[k][k1];
                i=mesh->elem.val[k][k1];

                // for the diag entry
                gradBasisP1(T, s, k1, phi1);
                tmp_a = 2*s*(phi1[0]*phi1[0]+phi1[1]*phi1[1]*epsilon);
                for (i1=0;i1<pt->num_qp_mat;i1++) {
                    A->val[A->IA[i]] += gauss[i1][2]*tmp_a;
                    phi_1 = basisP1(k1, gauss[i1]);
                    M->val[A->IA[i]] += 2*s*gauss[i1][2]*(phi_1*phi_1);
                } // end for i1

                // for the off-diag entry
                i = edge2idx_g1[edge_c];
                j = edge2idx_g2[edge_c];
                n1 = (k1+1)%3;//local node index in the elem
                n2 = (k1+2)%3;
                gradBasisP1(T, s, n1, phi1);
                gradBasisP1(T, s, n2, phi2);
                tmp_a = 2*s*(phi1[0]*phi2[0]+phi1[1]*phi2[1]*epsilon);
                for (i1=0;i1<pt->num_qp_mat;i1++) {
                    A->val[i] += gauss[i1][2]*tmp_a;
                    A->val[j] += gauss[i1][2]*tmp_a;
                    phi_1 = basisP1(n1, gauss[i1]);
                    phi_2 = basisP1(n2, gauss[i1]);
                    M->val[i] += 2*s*gauss[i1][2]*(phi_1*phi_2);
                    M->val[j] += 2*s*gauss[i1][2]*(phi_1*phi_2);
                } // end for i1
            } // end for k1
        } // end for k
    }
    else {
        printf("### ERROR: Wrong input value! %s\n", __FUNCTION__);
    }

    fasp_mem_free(edge2idx_g1);
    fasp_mem_free(edge2idx_g2);
    fasp_mem_free(btmp);
}

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT setup_heat (dCSRmat *A_heat,
 *                     dCSRmat *Mass,
 *                     dvector *b_heat,
 *                     Mesh *mesh,
 *                     Mesh_aux *mesh_aux,
 *                     FEM_param *pt,
 *                     dvector *uh_heat,
 *                     Bd_apply_info *bdinfo,
 *                     REAL dt)
 *
 * \brief Setup P1 FEM for the heat transfer's equation
 *
 * \param *A                     pointer to stiffness matrix
 * \param *Mass                  pointer to mass matrix
 * \param *b_heat                pointer to right hand side
 * \param *mesh                  pointer to mesh info
 * \param *mesh_aux              pointer to auxiliary mesh info
 * \param *pt                    pointer to parameter
 * \param *uh_heat                             discrete solution
 * \param *bdinfo                               info to apply boundary condition
 * \param dt                     size of time step
 *
 * \return                       FASP_SUCCESS if succeed
 *
 * \author Chensong Zhang and Feiteng Huang
 * \date   08/10/2010
 *
 * Modified by Feiteng Huang on 04/01/2012, output node, elem, uh, and dof for l2 error
 * Modified by Feiteng Huang on 04/09/2012, restructure the code
 */
INT setup_heat (dCSRmat *A_heat,
                dCSRmat *Mass,
                dvector *b_heat,
                Mesh *mesh,
                Mesh_aux *mesh_aux,
                FEM_param *pt,
                dvector *uh_heat,
                Bd_apply_info *bdinfo,
                REAL dt)
{
    // assemble
    dCSRmat Stiff;
    dvector rhs;
    
    assemble_stiffmat (&Stiff, Mass, &rhs, mesh, mesh_aux, pt, dt);
    fasp_blas_dcsr_add(&Stiff, 1, Mass, 1/dt, A_heat); //form matrix for heat transfer
    
    // get information to deal with Dirichlet boundary condition
    ivector dirichlet,nondirichlet,index;
    INT i,j,k,it;
    INT dirichlet_count = 0;
    for (i=0;i<mesh->node_bd.row;++i) {
        if (mesh->node_bd.val[i] == DIRICHLET)
            dirichlet_count++;
    }
    
    dirichlet = fasp_ivec_create(dirichlet_count);
    nondirichlet = fasp_ivec_create(mesh->node.row-dirichlet_count);
    index = fasp_ivec_create(mesh->node.row);
    
    j = k = 0;
    for (i=0;i<mesh->node_bd.row;++i) {
        if (mesh->node_bd.val[i]==DIRICHLET) { //  Dirichlet boundary node
            dirichlet.val[k]=i;
            index.val[i]=k;
            ++k;
        }
        else { // free variable
            nondirichlet.val[j]=i;
            index.val[i]=j;
            ++j;
        }
    }
    
    // set initial boundary value
    dvector uh = fasp_dvec_create(Stiff.row*pt->nt);
    REAL p[DIM+1];
    for (i=0;i<Stiff.row;++i) {
        if(mesh->node_bd.val[i]==DIRICHLET) { // the node is on the boundary
            for (j=0;j<DIM;++j)
                p[j] = mesh->node.val[i][j];
            for (it=0;it<pt->nt;++it) {
                p[DIM] = dt*(it+1);
                uh.val[i+it*Stiff.row]=u(p);
            }
        }
    }
    
    // output info for l2 error
    ivec_output( &(bdinfo->dof), &nondirichlet);
    ivec_output( &(bdinfo->bd), &dirichlet);
    ivec_output( &(bdinfo->idx), &index);
    dvec_output( uh_heat, &uh);
    dvec_output( b_heat, &rhs);
    
    // clean up memory
    fasp_dcsr_free(&Stiff);
    
    return FASP_SUCCESS;
}

/** 
 * \fn REAL get_l2_error_heat (ddenmat *node,
 *                               idenmat *elem,
 *                               dvector *uh,
 *                               INT num_qp,
 *                               REAL t)
 *
 * \brief get l2 error of fem.
 *
 * \param *node                  node info
 * \param *elem                  elem info
 * \param *ptr_uh                discrete solution
 * \param num_qp                 number of quad point
 * \param t                      time
 *
 * \author Feiteng Huang
 * \date   03/30/2012
 */
REAL get_l2_error_heat (ddenmat *node,
                          idenmat *elem,
                          dvector *uh,
                          INT num_qp,
                          REAL t)
{
    REAL l2error = 0.0;
    
    REAL uh_local[3] = {0, 0, 0};
    REAL T[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    REAL gauss[MAX_QUAD][DIM+1];
    REAL s, l2, a, p[DIM+1], uh_p;
    p[DIM] = t;
    
    INT i,j,k;
    
    fasp_gauss2d(num_qp, 2, gauss); // Gauss integration initial
    
    for (k=0;k<elem->row;++k) {

        for (i=0;i<elem->col;++i) {
            j = elem->val[k][i];
            T[i][0] = node->val[j][0];
            T[i][1] = node->val[j][1];
            uh_local[i] = uh->val[j];
        } // end for i
        s = 2.0*areaT(T[0][0], T[1][0], T[2][0], T[0][1], T[1][1], T[2][1]);

        for (i=0;i<num_qp;++i) {
            l2 = 1 - gauss[i][0] - gauss[i][1];
            for (j=0;j<DIM;++j)
                p[j]=T[0][j]*gauss[i][0]+T[1][j]*gauss[i][1]+T[2][j]*l2;
            a=u(p);
            
            uh_p = uh_local[0]*gauss[i][0] + uh_local[1]*gauss[i][1] + uh_local[2]*l2;
            l2error += s*gauss[i][2]*((a - uh_p)*(a - uh_p));
        }
    }
    
    l2error = sqrt(l2error);
    
    return l2error;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
