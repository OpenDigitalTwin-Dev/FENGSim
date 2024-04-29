/*! \file  FemMesh.c
 *
 *  \brief FEM mesh input, output, refine, etc
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "mesh.h"

#define DIM 2

/**
 * \fn int mesh_init (Mesh *mesh, const char *filename)
 *
 * \brief Initialize the mesh info from input file
 *
 * \param mesh          Mesh info
 * \param filename      Input mesh filename
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int mesh_init (Mesh *mesh, const char *filename)
{
    INT num_node, dim_node, num_elem, dim_elem;
    INT i, j;
    int status = FASP_SUCCESS;
    
    FILE *inputFile = fopen(filename, "r");
    if (inputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    // get the node' coordinates
    if ( fscanf(inputFile, "%d %d", &num_node, &dim_node) > 0 ) {
        mesh->node.row = num_node;
        mesh->node.col = dim_node;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh->node.val    = (REAL **)fasp_mem_calloc(num_node, sizeof(REAL*));
    mesh->node.val[0] = (REAL *)fasp_mem_calloc(num_node*dim_node, sizeof(REAL));
    REAL *tmp_node  = mesh->node.val[0];
    
    for (i=0;i<num_node;++i) {
        // re-point to the val
        mesh->node.val[i]=&tmp_node[dim_node*i];
        for (j=0;j<dim_node;++j) {
            status = fscanf(inputFile, "%lf", &mesh->node.val[i][j]);
        }
    }
    
    // get triangular grid
    if ( fscanf(inputFile, "%d %d", &num_elem, &dim_elem) > 0 ) {
        mesh->elem.row = num_elem;
        mesh->elem.col = dim_elem;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh->elem.val    = (INT **)fasp_mem_calloc(num_elem, sizeof(INT*));
    mesh->elem.val[0] = (INT *)fasp_mem_calloc(num_elem*dim_elem, sizeof(INT));
    INT *tmp_elem     = mesh->elem.val[0];
    
    for (i=0;i<num_elem;++i) {
        // re-point to the val
        mesh->elem.val[i] = &tmp_elem[dim_elem*i];
        for (j=0;j<dim_elem;++j) {
            status = fscanf(inputFile, "%d", &mesh->elem.val[i][j]);
            mesh->elem.val[i][j]--;
        }
    }
    
    // get node boundary flag
    mesh->node_bd.row = num_node;
    mesh->node_bd.val = (INT *)fasp_mem_calloc(num_node, sizeof(INT));
    REAL p[DIM];
    for (i=0;i<num_node;++i) {
        for (j=0;j<dim_node;j++)
            p[j] = mesh->node.val[i][j];
        mesh->node_bd.val[i] = bd_flag(p);
    }
    
    fclose(inputFile);
    return status;
}

/**
 * \fn int mesh_init_pro (Mesh *mesh, const char *filename)
 *
 * \brief Initialize the mesh info from input file with bd flag
 *
 * \param mesh          Mesh info
 * \param filename      Input mesh filename
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int mesh_init_pro (Mesh *mesh, const char *filename)
{
    INT num_node, dim_node, num_elem, dim_elem;
    INT i, j;
    int status = FASP_SUCCESS;
    
    FILE *inputFile=fopen(filename, "r");
    if (inputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    // get the node' coordinates
    if ( fscanf(inputFile, "%d %d", &num_node, &dim_node) > 0 ) {
        mesh->node.row = num_node;
        mesh->node.col = dim_node;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh->node.val    = (REAL **)fasp_mem_calloc(num_node, sizeof(REAL*));
    mesh->node.val[0] = (REAL *)fasp_mem_calloc(num_node*dim_node, sizeof(REAL));
    REAL *tmp_node  = mesh->node.val[0];
    
    for (i=0;i<num_node;++i) {
        // re-point to the val
        mesh->node.val[i]=&tmp_node[dim_node*i];
        for (j=0;j<dim_node;++j) {
            status = fscanf(inputFile, "%lf", &mesh->node.val[i][j]);
        }
    }
    
    // get triangular grid
    if ( fscanf(inputFile, "%d %d", &num_elem, &dim_elem) > 0 ) {
        mesh->elem.row = num_elem;
        mesh->elem.col = dim_elem;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh->elem.val    = (INT **)fasp_mem_calloc(num_elem, sizeof(INT*));
    mesh->elem.val[0] = (INT *)fasp_mem_calloc(num_elem*dim_elem, sizeof(INT));
    INT *tmp_elem     = mesh->elem.val[0];
    
    for (i=0;i<num_elem;++i) {
        // re-point to the val
        mesh->elem.val[i]=&tmp_elem[dim_elem*i];
        for (j=0;j<dim_elem;++j) {
            status = fscanf(inputFile, "%d", &mesh->elem.val[i][j]);
            mesh->elem.val[i][j]--;
        }
    }
    
    // get node boundary flag
    if ( fscanf(inputFile, "%d", &num_elem) > 0 ) {
        mesh->node_bd.row = num_node;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    mesh->node_bd.val = (INT *)fasp_mem_calloc(num_node, sizeof(INT));
    for (i=0;i<num_node;++i) {
        status = fscanf(inputFile, "%d", &mesh->node_bd.val[i]);
    }
    
    fclose(inputFile);
    return status;
}

/**
 * \fn int mesh_aux_init (Mesh *mesh, Mesh_aux *mesh_aux, const char *filename)
 *
 * \brief Initialize the auxiliary mesh info from mesh info and input file
 *
 * \param mesh          Mesh info
 * \param mesh_aux      Auxiliary mesh info
 * \param filename      Input mesh filename
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2009
 */
int mesh_aux_init (Mesh *mesh, Mesh_aux *mesh_aux, const char *filename)
{
    INT num_edge, dim_edge, num_elem2edge, dim_elem2edge;
    INT i, j;
    int status = FASP_SUCCESS;
    
    FILE *inputFile=fopen(filename, "r");
    if (inputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    // get the edge info
    if ( fscanf(inputFile, "%d %d", &num_edge, &dim_edge) > 0 ) {
        mesh_aux->edge.row = num_edge;
        mesh_aux->edge.col = dim_edge;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh_aux->edge.val    = (INT **)fasp_mem_calloc(num_edge, sizeof(INT *));
    mesh_aux->edge.val[0] = (INT *)fasp_mem_calloc(num_edge*dim_edge, sizeof(INT));
    INT *tmp_edge         = mesh_aux->edge.val[0];
    
    for (i=0;i<num_edge;++i) {
        // re-point to the val
        mesh_aux->edge.val[i]=&tmp_edge[dim_edge*i];
        for (j=0;j<dim_edge;++j) {
            status = fscanf(inputFile, "%d", &mesh_aux->edge.val[i][j]);
            mesh_aux->edge.val[i][j]--;
        }
    }
    
    // get elem's edge info
    if ( fscanf(inputFile, "%d %d", &num_elem2edge, &dim_elem2edge) > 0 ) {
        mesh_aux->elem2edge.row=num_elem2edge;
        mesh_aux->elem2edge.col=dim_elem2edge;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh_aux->elem2edge.val    = (INT **)fasp_mem_calloc(num_elem2edge, sizeof(INT*));
    mesh_aux->elem2edge.val[0] = (INT *)fasp_mem_calloc(num_elem2edge*dim_elem2edge, sizeof(INT));
    INT *tmp_elem              = mesh_aux->elem2edge.val[0];
    
    for (i=0;i<num_elem2edge;++i) {
        // re-point to the val
        mesh_aux->elem2edge.val[i] = &tmp_elem[dim_elem2edge*i];
        for (j=0;j<dim_elem2edge;++j) {
            status = fscanf(inputFile, "%d", &mesh_aux->elem2edge.val[i][j]);
            mesh_aux->elem2edge.val[i][j]--;
        }
    }
    
    // get edge boundary flag
    mesh_aux->edge_bd.row = num_edge;
    mesh_aux->edge_bd.val = (INT *)fasp_mem_calloc(num_edge, sizeof(INT));
    REAL p[DIM];
    for (j=0;j<DIM;++j) p[j] = 0.0;
    
    INT n, k;
    for (i=0;i<num_edge;++i) {
        for (j=0;j<dim_edge;++j) {
            n = mesh_aux->edge.val[i][j];
            for (k=0;k<DIM;++k) {
                p[k] += mesh->node.val[n][k]/DIM;
            }
        }
        mesh_aux->edge_bd.val[i] = bd_flag(p);
        for (j=0;j<DIM;++j) p[j] = 0.0;
    }
    
    fclose(inputFile);
    return status;
}

/**
 * \fn int mesh_aux_init_pro (Mesh *mesh, Mesh_aux *mesh_aux, const char *filename)
 *
 * \brief Initialize the auxiliary mesh info from mesh info and input file with bd flag
 *
 * \param mesh          Mesh info
 * \param mesh_aux      Auxiliary mesh info
 * \param filename      Input mesh filename
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2009
 */
int mesh_aux_init_pro (Mesh *mesh, Mesh_aux *mesh_aux, const char *filename)
{
    INT num_edge, dim_edge, num_elem2edge, dim_elem2edge;
    INT i, j;
    int status = FASP_SUCCESS;
    
    FILE *inputFile=fopen(filename, "r");
    if (inputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    // get the edge info
    if ( fscanf(inputFile, "%d %d", &num_edge, &dim_edge) > 0 ) {
        mesh_aux->edge.row=num_edge;
        mesh_aux->edge.col=dim_edge;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh_aux->edge.val    = (INT **)fasp_mem_calloc(num_edge, sizeof(INT *));
    mesh_aux->edge.val[0] = (INT *)fasp_mem_calloc(num_edge*dim_edge, sizeof(INT));
    INT *tmp_edge         = mesh_aux->edge.val[0];
    
    for (i=0;i<num_edge;++i) {
        // re-point to the val
        mesh_aux->edge.val[i]=&tmp_edge[dim_edge*i];
        for (j=0;j<dim_edge;++j) {
            status = fscanf(inputFile, "%d", &mesh_aux->edge.val[i][j]);
            mesh_aux->edge.val[i][j]--;
        }
    }
    
    // get elem's edge info
    if ( fscanf(inputFile, "%d %d", &num_elem2edge, &dim_elem2edge) > 0 ) {
        mesh_aux->elem2edge.row=num_elem2edge;
        mesh_aux->elem2edge.col=dim_elem2edge;
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    // alloc mem we need,
    mesh_aux->elem2edge.val    = (INT **)fasp_mem_calloc(num_elem2edge, sizeof(INT*));
    mesh_aux->elem2edge.val[0] = (INT *)fasp_mem_calloc(num_elem2edge*dim_elem2edge, sizeof(INT));
    INT *tmp_elem              = mesh_aux->elem2edge.val[0];
    
    for (i=0;i<num_elem2edge;++i) {
        // re-point to the val
        mesh_aux->elem2edge.val[i]=&tmp_elem[dim_elem2edge*i];
        for (j=0;j<dim_elem2edge;++j) {
            status = fscanf(inputFile, "%d", &mesh_aux->elem2edge.val[i][j]);
            mesh_aux->elem2edge.val[i][j]--;
        }
    }
    
    // get edge boundary flag
    if ( fscanf(inputFile, "%d", &num_edge) > 0 ) {
        mesh_aux->edge_bd.row = num_edge;
        mesh_aux->edge_bd.val = (INT *)fasp_mem_calloc(num_edge, sizeof(INT));
    }
    else {
        printf("### ERROR: Invalid mesh file!\n");
        exit(ERROR_WRONG_FILE);
    }
    
    for (i=0;i<num_edge;++i) {
        status = fscanf(inputFile, "%d", &mesh_aux->edge_bd.val[i]);
    }
    
    fclose(inputFile);
    return status;
}

/**
 * \fn int mesh_aux_build (Mesh *mesh, Mesh_aux *mesh_aux)
 *
 * \brief Generate auxiliary mesh info
 *
 * \param mesh          Mesh info
 * \param mesh_aux      Auxiliary mesh info
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int mesh_aux_build(Mesh *mesh, Mesh_aux *mesh_aux)
{
    INT num_node = mesh->node.row;
    INT dim_node = mesh->node.col;
    INT num_elem = mesh->elem.row;
    INT dim_elem = mesh->elem.col;
    
    INT num_elem2edge = num_elem;
    INT dim_elem2edge = dim_elem;
    mesh_aux->elem2edge.row = num_elem2edge;
    mesh_aux->elem2edge.col = dim_elem2edge;
    INT num_edge = 3*num_node; // pre-define num_edge, actually num_edge < 3*num_node
    INT dim_edge = dim_node;
    mesh_aux->edge.col = dim_edge;
    
    INT i, j, n1, n2, n1t, n2t, count = 0, edge_c = 0;
    REAL p[DIM];
    mesh_aux->edge.val = (INT **)fasp_mem_calloc(num_edge, sizeof(INT*));
    mesh_aux->edge.val[0] = (INT *)fasp_mem_calloc(num_edge*dim_edge, sizeof(INT));
    INT *tmp_edge = mesh_aux->edge.val[0];
    mesh_aux->elem2edge.val = (INT **)fasp_mem_calloc(num_elem2edge, sizeof(INT*));
    mesh_aux->elem2edge.val[0] = (INT *)fasp_mem_calloc(num_elem2edge*dim_elem2edge, sizeof(INT));
    INT *tmp_elem = mesh_aux->elem2edge.val[0];
    mesh_aux->edge_bd.val = (INT *)fasp_mem_calloc(num_edge, sizeof(INT));
    
    for (i=0;i<num_edge;++i) {
        mesh_aux->edge.val[i] = &tmp_edge[dim_edge*i];
    }
    
    for (i=0;i<num_elem2edge;++i) {
        mesh_aux->elem2edge.val[i] = &tmp_elem[dim_elem2edge*i];
    }
    
    INT *edge_aux = NULL;
    if (num_node < 1e3) { // Why having such a constraint? --Chensong
        // no reason, :-), just to avoid the shortage of memory
        // see edge_aux following --Feiteng
        edge_aux = (INT *)fasp_mem_calloc(num_node*num_node, sizeof(INT));
        for (i=0;i<num_elem;++i) {
            for (j=0;j<dim_elem;++j) {
                n1t = mesh->elem.val[i][(j+1)%dim_elem];
                n2t = mesh->elem.val[i][(j+2)%dim_elem];
                n1 = MIN(n1t, n2t);
                n2 = MAX(n1t, n2t);
                edge_c = n1*num_node+n2;
                if (edge_aux[edge_c] == 0) {
                    mesh_aux->edge.val[count][0] = n1;
                    mesh_aux->edge.val[count][1] = n2;
                    mesh_aux->elem2edge.val[i][j] = count;
                    count++;
                    edge_aux[edge_c] = count;
                }
                else if (edge_aux[edge_c] > 0) {
                    mesh_aux->elem2edge.val[i][j] = edge_aux[edge_c]-1;
                    edge_aux[edge_c] *= -1;
                }
                else {
                    printf("### ERROR: Cannot build mesh auxiliary data!\n");
                    exit(ERROR_MISC);
                }
            }// end of loop j
        }// end of loop i
        for (i=0;i<count;++i) {
            n1 = mesh_aux->edge.val[i][0];
            n2 = mesh_aux->edge.val[i][1];
            edge_c = edge_aux[n1*num_node+n2] - 1;
            if (edge_c >= 0) {
                p[0] = (mesh->node.val[n1][0] + mesh->node.val[n2][0])/2;
                p[1] = (mesh->node.val[n1][1] + mesh->node.val[n2][1])/2;
                mesh_aux->edge_bd.val[edge_c] = bd_flag(p);
            }
        }// end of loop i
    }// end of if
    
    else { // if there is too many node
        INT adj_max = 10; // set a max number of adjacent node
        INT k = 0;
        INT *edge_map= (INT *)fasp_mem_calloc(num_node*adj_max, sizeof(INT));
        INT *adj_count = (INT *)fasp_mem_calloc(num_node, sizeof(INT));
        edge_aux = (INT *)fasp_mem_calloc(num_node*adj_max, sizeof(INT));
        for (i=0;i<num_elem;++i) {
            for (j=0;j<dim_elem;++j) {
                n1t = mesh->elem.val[i][(j+1)%dim_elem];
                n2t = mesh->elem.val[i][(j+2)%dim_elem];
                n1 = MIN(n1t, n2t);
                n2 = MAX(n1t, n2t);
                for (k=0;k<adj_count[n1];++k) {
                    if (edge_map[n1*adj_max+k] == n2) {
                        mesh_aux->elem2edge.val[i][j] = edge_aux[n1*adj_max+k];
                        edge_aux[n1*adj_max+k] *= -1;
                        break;
                    }
                }// end of loop k
                if (k == adj_count[n1]) {
                    mesh_aux->edge.val[count][0] = n1;
                    mesh_aux->edge.val[count][1] = n2;
                    mesh_aux->elem2edge.val[i][j] = count;
                    edge_aux[n1*adj_max+k] = count;
                    edge_map[n1*adj_max+k] = n2;
                    adj_count[n1] += 1;
                    count++;
                }// end of if
            }// end of loop j
        }// end of loop i
        
        for (i=0;i<count;++i) {
            n1 = mesh_aux->edge.val[i][0];
            n2 = mesh_aux->edge.val[i][1];
            for (k=0;k<adj_count[n1];++k) {
                if (edge_map[n1*adj_max+k] == n2) {
                    edge_c = edge_aux[n1*adj_max+k];
                    break;
                }
                if (k == adj_count[n1]) {
                    printf("### ERROR: Cannot build mesh auxiliary data!\n");
                    exit(ERROR_MISC);
                }
            }
            if (edge_c >= 0) {
                p[0] = (mesh->node.val[n1][0] + mesh->node.val[n2][0])/2;
                p[1] = (mesh->node.val[n1][1] + mesh->node.val[n2][1])/2;
                mesh_aux->edge_bd.val[edge_c] = bd_flag(p);
            }
        }// end of loop i
        
        fasp_mem_free(adj_count);
        fasp_mem_free(edge_map);
    }
    
    num_edge = count;
    mesh_aux->edge.row = num_edge;
    mesh_aux->edge_bd.row = num_edge;
    mesh_aux->edge.val[0] = (INT *)fasp_mem_realloc(mesh_aux->edge.val[0], sizeof(INT)*num_edge*dim_edge);
    mesh_aux->edge.val = (INT **)fasp_mem_realloc(mesh_aux->edge.val, sizeof(INT*)*num_edge);
    mesh_aux->edge_bd.val = (INT *)fasp_mem_realloc(mesh_aux->edge_bd.val, sizeof(INT)*num_edge);
    tmp_edge = mesh_aux->edge.val[0];
    for (i=0;i<num_edge;++i) {
        mesh_aux->edge.val[i] = &tmp_edge[i*dim_edge];
    }
    
    fasp_mem_free(edge_aux);
    
    return FASP_SUCCESS;
}

/**
 * \fn int mesh_write (Mesh *mesh, const char *filename)
 *
 * \brief Write the mesh information
 *
 * \param mesh          Mesh info
 * \param filename      Output mesh file
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2012
 */
int mesh_write (Mesh *mesh, const char *filename)
{
    INT num_node=mesh->node.row;
    INT dim_node=mesh->node.col;
    INT num_elem=mesh->elem.row;
    INT dim_elem=mesh->elem.col;
    INT i,j;
    
    FILE *outputFile=fopen(filename, "w");
    if (outputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    fprintf(outputFile, "%d %d\n", num_node, dim_node);
    
    for (i=0;i<num_node;++i) {
        for (j=0;j<dim_node;++j) {
            fprintf(outputFile, "%lf ", mesh->node.val[i][j]);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d %d\n", num_elem, dim_elem);
    
    for (i=0;i<num_elem;++i) {
        for (j=0;j<dim_elem;++j) {
            fprintf(outputFile, "%d ", mesh->elem.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fclose(outputFile);
    
    return FASP_SUCCESS;
}

/**
 * \fn int mesh_write_pro (Mesh *mesh, const char *filename)
 *
 * \brief Write the mesh information with bd flag
 *
 * \param mesh          Mesh info
 * \param filename      Output mesh file
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2012
 */
int mesh_write_pro (Mesh *mesh, const char *filename)
{
    INT num_node=mesh->node.row;
    INT dim_node=mesh->node.col;
    INT num_elem=mesh->elem.row;
    INT dim_elem=mesh->elem.col;
    INT i,j;
    
    FILE *outputFile=fopen(filename, "w");
    if (outputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    fprintf(outputFile, "%d %d\n", num_node, dim_node);
    
    for (i=0;i<num_node;++i) {
        for (j=0;j<dim_node;++j) {
            fprintf(outputFile, "%lf ", mesh->node.val[i][j]);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d %d\n", num_elem, dim_elem);
    
    for (i=0;i<num_elem;++i) {
        for (j=0;j<dim_elem;++j) {
            fprintf(outputFile, "%d ", mesh->elem.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d\n", num_node);
    for (i=0;i<num_node;++i) {
        fprintf(outputFile, "%d\n", mesh->node_bd.val[i]);
    }
    
    fclose(outputFile);
    
    return FASP_SUCCESS;
}

/**
 * \fn int mesh_aux_write (Mesh_aux *mesh_aux, const char *filename)
 *
 * \brief Write the auxiliary mesh information
 *
 * \param mesh_aux      Auxiliary mesh info
 * \param filename      Output mesh file
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2012
 */
int mesh_aux_write (Mesh_aux *mesh_aux, const char *filename)
{
    INT num_edge=mesh_aux->edge.row;
    INT dim_edge=mesh_aux->edge.col;
    INT num_elem2edge=mesh_aux->elem2edge.row;
    INT dim_elem2edge=mesh_aux->elem2edge.col;
    INT i,j;
    
    FILE *outputFile=fopen(filename, "w");
    if (outputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    fprintf(outputFile, "%d %d\n", num_edge, dim_edge);
    
    for (i=0;i<num_edge;++i) {
        for (j=0;j<dim_edge;++j) {
            fprintf(outputFile, "%d ", mesh_aux->edge.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d %d\n", num_elem2edge, dim_elem2edge);
    
    for (i=0;i<num_elem2edge;++i) {
        for (j=0;j<dim_elem2edge;++j) {
            fprintf(outputFile, "%d ", mesh_aux->elem2edge.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fclose(outputFile);
    
    return FASP_SUCCESS;
    
}

/**
 * \fn int mesh_aux_write_pro (Mesh_aux *mesh_aux, const char *filename)
 *
 * \brief Write the auxiliary mesh information with bd flag
 *
 * \param mesh_aux      Auxiliary mesh info
 * \param filename      Output mesh file
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2012
 */
int mesh_aux_write_pro (Mesh_aux *mesh_aux, const char *filename)
{
    INT num_edge=mesh_aux->edge.row;
    INT dim_edge=mesh_aux->edge.col;
    INT num_elem2edge=mesh_aux->elem2edge.row;
    INT dim_elem2edge=mesh_aux->elem2edge.col;
    INT i,j;
    
    FILE *outputFile=fopen(filename, "w");
    if (outputFile==NULL) {
        printf("### ERROR: Cannot open %s!\n", filename);
        exit(ERROR_OPEN_FILE);
    }
    
    fprintf(outputFile, "%d %d\n", num_edge, dim_edge);
    
    for (i=0;i<num_edge;++i) {
        for (j=0;j<dim_edge;++j) {
            fprintf(outputFile, "%d ", mesh_aux->edge.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d %d\n", num_elem2edge, dim_elem2edge);
    
    for (i=0;i<num_elem2edge;++i) {
        for (j=0;j<dim_elem2edge;++j) {
            fprintf(outputFile, "%d ", mesh_aux->elem2edge.val[i][j]+1);
        }
        fprintf(outputFile, "\n");
    }
    
    fprintf(outputFile, "%d\n", num_edge);
    
    for (i=0;i<num_edge;++i) {
        fprintf(outputFile, "%d\n", mesh_aux->edge_bd.val[i]);
    }
    
    fclose(outputFile);
    
    return FASP_SUCCESS;
    
}

/**
 * \fn int mesh_free (Mesh *mesh)
 *
 * \brief free memory for mesh info
 *
 * \param mesh          Mesh info
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int mesh_free (Mesh *mesh)
{
    fasp_mem_free(mesh->node.val[0]);
    fasp_mem_free(mesh->node.val);
    fasp_mem_free(mesh->elem.val[0]);
    fasp_mem_free(mesh->elem.val);
    fasp_mem_free(mesh->node_bd.val);
    
    mesh->node.row = 0;
    mesh->node.col = 0;
    mesh->elem.row = 0;
    mesh->elem.col = 0;
    mesh->node_bd.row = 0;
    mesh->node.val = NULL;
    mesh->elem.val = NULL;
    mesh->node_bd.val  = NULL;
    
    return FASP_SUCCESS;
}

/**
 * \fn int mesh_aux_free (Mesh_aux *mesh_aux)
 *
 * \brief free memory for auxiliary mesh info
 *
 * \param mesh_aux       Auxiliary mesh info
 *
 * \return               1 if succeed 0 if fail
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int mesh_aux_free (Mesh_aux *mesh_aux)
{
    fasp_mem_free(mesh_aux->edge.val[0]);
    fasp_mem_free(mesh_aux->edge.val);
    fasp_mem_free(mesh_aux->elem2edge.val[0]);
    fasp_mem_free(mesh_aux->elem2edge.val);
    fasp_mem_free(mesh_aux->edge_bd.val);
    
    mesh_aux->edge.row = 0;
    mesh_aux->edge.col = 0;
    mesh_aux->elem2edge.row = 0;
    mesh_aux->elem2edge.col = 0;
    mesh_aux->edge_bd.row = 0;
    mesh_aux->edge.val = NULL;
    mesh_aux->elem2edge.val = NULL;
    mesh_aux->edge_bd.val = NULL;
    
    return FASP_SUCCESS;
}

/**
 * \fn int mesh_refine (Mesh *mesh, Mesh_aux *mesh_aux)
 *
 * \brief refine mesh use mesh info and auxiliary mesh info
 *
 * \param mesh          Mesh info
 * \param mesh_aux      Auxiliary mesh info
 *
 * \return              FASP_SUCCESS if succeed
 *
 * \author Feiteng Huang
 * \date   04/06/2009
 */
int mesh_refine(Mesh *mesh, Mesh_aux *mesh_aux)
{
    INT num_node = mesh->node.row;
    INT dim_node = mesh->node.col;
    INT num_elem = mesh->elem.row;
    INT dim_elem = mesh->elem.col;
    INT num_edge = mesh_aux->edge.row;
    INT dim_edge = mesh_aux->edge.col;
    
    INT num_newnode = num_node + num_edge;
    INT num_newedge = 3*num_elem + 2*num_edge;
    INT num_newelem = 4*num_elem;
    
    INT i, j, k;
    INT n[3], e[6] = {0, 0, 0, 0, 0, 0};
    
    mesh->node.row = num_newnode;
    mesh->elem.row = num_newelem;
    mesh->node_bd.row = num_newnode;
    mesh_aux->edge.row = num_newedge;
    mesh_aux->elem2edge.row = num_newelem;
    mesh_aux->edge_bd.row = num_newedge;
    
    // realloc node, elem, edge, elem2edge, node_bd, edge_bd
    mesh->node.val[0] = (REAL *)fasp_mem_realloc(mesh->node.val[0], sizeof(REAL)*num_newnode*dim_node);
    mesh->node.val = (REAL **)fasp_mem_realloc(mesh->node.val, sizeof(REAL *)*num_newnode);
    REAL *tmp_node = mesh->node.val[0];
    mesh->elem.val[0] = (INT *)fasp_mem_realloc(mesh->elem.val[0], sizeof(INT)*num_newelem*dim_elem);
    mesh->elem.val = (INT **)fasp_mem_realloc(mesh->elem.val, sizeof(INT *)*num_newelem);
    INT *tmp_elem = mesh->elem.val[0];
    mesh_aux->edge.val[0] = (INT *)fasp_mem_realloc(mesh_aux->edge.val[0], sizeof(INT)*num_newedge*dim_edge);
    mesh_aux->edge.val = (INT **)fasp_mem_realloc(mesh_aux->edge.val, sizeof(INT *)*num_newedge);
    INT *tmp_edge = mesh_aux->edge.val[0];
    mesh_aux->elem2edge.val[0] = (INT *)fasp_mem_realloc(mesh_aux->elem2edge.val[0], sizeof(INT)*num_newelem*dim_elem);
    mesh_aux->elem2edge.val = (INT **)fasp_mem_realloc(mesh_aux->elem2edge.val, sizeof(INT *)*num_newelem);
    INT *tmp_elem2edge = mesh_aux->elem2edge.val[0];
    mesh->node_bd.val = (INT *)fasp_mem_realloc(mesh->node_bd.val, sizeof(INT)*num_newnode);
    mesh_aux->edge_bd.val = (INT *)fasp_mem_realloc(mesh_aux->edge_bd.val, sizeof(INT)*num_newedge);
    
    for (i=0;i<num_newnode;++i) {
        mesh->node.val[i] = &tmp_node[i*dim_node];
    }
    for (i=0;i<num_newedge;++i) {
        mesh_aux->edge.val[i] = &tmp_edge[i*dim_edge];
    }
    for (i=0;i<num_newelem;++i) {
        mesh->elem.val[i] = &tmp_elem[i*dim_elem];
        mesh_aux->elem2edge.val[i] = &tmp_elem2edge[i*dim_elem];
    }
    
    // update mesh & mesh_aux info
    for (i=0;i<num_edge;++i) {
        
        // update node info
        
        // init node value
        for (k=0;k<dim_node;++k) {
            mesh->node.val[i+num_node][k] = 0;
        }
        for (j=0;j<dim_edge;++j) {
            n[0] = mesh_aux->edge.val[i][j];
            for (k=0;k<dim_node;++k) {
                mesh->node.val[i+num_node][k] += mesh->node.val[n[0]][k]/dim_edge;
            }
        }
        
        // update node_bd info
        mesh->node_bd.val[i+num_node] = mesh_aux->edge_bd.val[i];
        
        // update edge & edge_bd on original edge
        n[0] = num_node + i;
        n[1] = mesh_aux->edge.val[i][1];
        
        // update auxiliary mesh info
        mesh_aux->edge.val[i][1] = n[0];
        mesh_aux->edge.val[i+num_edge][0] = n[0];
        mesh_aux->edge.val[i+num_edge][1] = n[1];
        
        mesh_aux->edge_bd.val[i+num_edge] = mesh_aux->edge_bd.val[i];
    }
    
    for (i=0;i<num_elem;++i) {
        
        for (j=0;j<dim_elem;++j) {
            n[j] = mesh_aux->elem2edge.val[i][j] + num_node;
        }
        
        for (j=0;j<dim_elem;++j) {
            // update edge info on original elem
            mesh_aux->edge.val[2*num_edge+3*i+j][0] = n[(j+1)%dim_elem];
            mesh_aux->edge.val[2*num_edge+3*i+j][1] = n[(j+2)%dim_elem];
            mesh_aux->edge_bd.val[2*num_edge+3*i+j] = INTERIORI;
            if (mesh->elem.val[i][(j+1)%dim_elem] == mesh_aux->edge.val[mesh_aux->elem2edge.val[i][j]][0]) {
                e[2*j] = mesh_aux->elem2edge.val[i][j];
                e[2*j+1] = mesh_aux->elem2edge.val[i][j] + num_edge;
            }
            else {
                e[2*j+1] = mesh_aux->elem2edge.val[i][j];
                e[2*j] = mesh_aux->elem2edge.val[i][j] + num_edge;
            }
            
            // update elem & elem2edge info
            mesh->elem.val[num_elem+3*i+2][j] = n[j];
            mesh_aux->elem2edge.val[num_elem+3*i+2][j] = 2*num_edge+3*i+j;
        }
        
        // update elem info
        mesh->elem.val[num_elem+3*i][0] = mesh->elem.val[i][1];
        mesh->elem.val[num_elem+3*i][1] = n[0];
        mesh->elem.val[num_elem+3*i][2] = n[2];
        
        mesh->elem.val[num_elem+3*i+1][0] = mesh->elem.val[i][2];
        mesh->elem.val[num_elem+3*i+1][1] = n[1];
        mesh->elem.val[num_elem+3*i+1][2] = n[0];
        
        mesh->elem.val[i][1] = n[2];
        mesh->elem.val[i][2] = n[1];
        
        // update elem2edge info
        mesh_aux->elem2edge.val[num_elem+3*i][0] = 2*num_edge+3*i+1;
        mesh_aux->elem2edge.val[num_elem+3*i][1] = e[5];
        mesh_aux->elem2edge.val[num_elem+3*i][2] = e[0];
        
        mesh_aux->elem2edge.val[num_elem+3*i+1][0] = 2*num_edge+3*i+2;
        mesh_aux->elem2edge.val[num_elem+3*i+1][1] = e[1];
        mesh_aux->elem2edge.val[num_elem+3*i+1][2] = e[2];
        
        mesh_aux->elem2edge.val[i][0] = 2*num_edge+3*i;
        mesh_aux->elem2edge.val[i][1] = e[3];
        mesh_aux->elem2edge.val[i][2] = e[4];
    }
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
