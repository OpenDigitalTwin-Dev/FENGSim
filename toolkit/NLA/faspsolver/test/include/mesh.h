/*! \file mesh.h
 *  \brief mesh input, output, refine etc.
 */

#ifndef _MESH_H_
#define _MESH_H_

#define DIRICHLET 1
#define INTERIORI 0

#include "fasp.h"
#include "fasp_functs.h"

extern int bd_flag(double *p);

/** 
 * \struct Mesh
 * \brief mesh data with node, elem, and node_bd
 */ 
typedef struct Mesh{
    
    idenmat elem;
    ddenmat node;
    ivector node_bd;

} Mesh;

/** 
 * \struct mesh_aux
 * \brief auxiliary mesh data with edge, elem2edge, edge_bd
 */ 
typedef struct Mesh_aux{
    
    idenmat edge;
    idenmat elem2edge;
    ivector edge_bd;

} Mesh_aux;

int mesh_init (Mesh *mesh, const char *filename);
int mesh_aux_init (Mesh *mesh, Mesh_aux *mesh_aux, const char *filename);
int mesh_aux_build(Mesh *mesh, Mesh_aux *mesh_aux);

int mesh_write (Mesh *mesh, const char *filename);
int mesh_aux_write (Mesh_aux *mesh_aux, const char *filename);

int mesh_free (Mesh *mesh);
int mesh_aux_free (Mesh_aux *mesh_aux);

int mesh_refine(Mesh *mesh, Mesh_aux *mesh_aux);

#endif
