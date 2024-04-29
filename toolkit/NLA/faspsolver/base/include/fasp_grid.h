/*! \file  fasp_grid.h
 *
 *  \brief Header file for FASP grid
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2015--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#ifndef __FASPGRID_HEADER__  /*-- allow multiple inclusions --*/
#define __FASPGRID_HEADER__  /**< indicate fasp_grid.h has been included before */

/**
 * \struct grid2d
 * \brief Two dimensional grid data structure
 *
 * \note The grid2d structure is simply a list of triangles,
 *       edges and vertices.
 *       edge i has 2 vertices e[i],
 *       triangle i has 3 edges s[i], 3 vertices t[i]
 *       vertex i has two coordinates p[i]
 */
typedef struct grid2d {
    
    REAL (*p)[2];  /**< Coordinates of vertices */
    INT (*e)[2];   /**< Vertices of edges */
    INT (*t)[3];   /**< Vertices of triangles */
    INT (*s)[3];   /**< Edges of triangles */
    INT *pdiri;    /**< Boundary flags (0 <=> interior point) */
    INT *ediri;    /**< Boundary flags (0 <=> interior edge) */
    
    INT *pfather;  /**< Father point or edge */
    INT *efather;  /**< Father edge or triangle */
    INT *tfather;  /**< Father triangle */
    
    INT vertices;  /**< Number of grid points */
    INT edges;     /**< Number of edges */
    INT triangles; /**< Number of triangles */
    
} grid2d; /**< 2D grid type for plotting */

typedef grid2d *pgrid2d; /**< Grid in 2d */

typedef const grid2d *pcgrid2d; /**< Grid in 2d */

#endif /* end if for __FASPGRID_HEADER__ */

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
