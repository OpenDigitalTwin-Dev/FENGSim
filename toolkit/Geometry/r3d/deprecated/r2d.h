/**************************************************************
 *
 *    r2d.h
 *    
 *    Devon Powell
 *    8 February 2015
 *
 *    Routines for fast, geometrically robust clipping operations
 *    and analytic area/moment computations over convex
 *    polygons in 2D. Includes a wrapper function for 
 *    physically conservative rasterization (area sampling)
 *    of quadrilaterals to a Cartesian grid.
 *    
 *    Copyright (C) 2014 Stanford University.
 *    See License.txt for more information.
 *
 *************************************************************/

#ifndef _R2D_H_
#define _R2D_H_

#include <stdint.h>

/**
 * \file r2d.h
 * \author Devon Powell
 * \date 8 Feb 2015
 * \brief Interface for r2d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double` (recommended). `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r2d_real;
#else
typedef double r2d_real;
#endif

/**
 * \brief Integer type used for grid indexing and bit flags
 */
typedef int32_t r2d_int;

/**
 * \brief Long integer type used for grid indexing
 */
typedef int64_t r2d_long;

/** \struct r2d_rvec2
 *  \brief Vector struct.
 */
typedef struct {
	r2d_real x, /*!< \f$x\f$-component. */
			 y;	/*!< \f$y\f$-component. */
} r2d_rvec2;

/** \struct r2d_dvec2
 *  \brief Integer vector struct for grid indexing.
 */
typedef struct {
	r2d_int i, /*!< \f$x\f$-index. */
			j; /*!< \f$y\f$-index. */
} r2d_dvec2;

/** \struct r2d_plane
 *  \brief A plane.
 */
typedef struct {
	r2d_rvec2 n; /*!< Unit-length normal vector. */
	r2d_real d; /*!< Signed perpendicular distance to the origin. */
} r2d_plane;

/** \struct r2d_orientation
 *  \brief Perpendicular distances and bit flags for up to 6 faces.
 */
typedef struct {
	r2d_real fdist[4]; /*!< Signed distances to clip planes. */
	unsigned char fflags; /*!< Bit flags corresponding to inside (1) or outside (0) of each clip plane. */
} r2d_orientation;

/** \struct r2d_vertex
 * \brief A doubly-linked vertex.
 */
typedef struct {
	r2d_rvec2 pos; /*!< Vertex position. */
	unsigned char pnbrs[2]; /*!< Neighbor indices. */
	r2d_orientation orient; /*!< Orientation with respect to clip planes. */
} r2d_vertex;

/** \struct r2d_poly
 * \brief A polygon.
 */
typedef struct {
#define R2D_MAX_VERTS 64
	r2d_vertex verts[R2D_MAX_VERTS]; /*!< Vertex buffer. */
	r2d_int nverts; /*!< Number of vertices in the buffer. */
	r2d_int funclip; /*!< Index of any unclipped vertex. */
} r2d_poly;

/** \struct r2d_dest_grid
 *  \brief Destination grid information.
 */
typedef struct {
	r2d_real* moments[6]; /*!< Gridded moments in row-major (C) order.
							   Must be at least `n.x*n.y` in size. */ 
	r2d_int polyorder; /*!< Polynomial order (0 for constant, 1 for linear, 2 for quadratic) to be rasterized. */
	r2d_orientation* orient; /*!< Buffer for gridpoint orientation checks.
							   Must be at least `(n.x+1)*(n.y+1)` in size.
							   Unused if compiled with `-DUSE_TREE`.*/ 
	r2d_long bufsz; /*!< Allocated size of moments and orient buffers. */
	r2d_dvec2 n; /*!< Grid dimensions (cells per coordinate). */
	r2d_rvec2 d; /*!< Grid cell size. */
} r2d_dest_grid;

/**
 * \brief Tabulated number of moments needed for a given polynomial order.
 */
const static r2d_int r2d_num_moments[3] = {1, 3, 6};


/**
 * \brief Rasterize a convex quadrilateral to the destination grid.
 *
 * \param [in] faces
 * The four faces of the quadrilateral to be voxelized.
 *
 * \param [in, out] grid
 * The destination grid buffer that the quadrilateral will be voxelized to.
 * The results of the rasterization are found in `grid.moments`.
 *
 */
void r2d_rasterize_quad(r2d_plane* faces, r2d_dest_grid* grid);

/**
 * \brief Clip a polygon against four clip planes (find its intersection with a quadrilateral). 
 *
 * \param [in, out] poly 
 * The polygon to be clipped. The distances to the clip plane and bit flags in
 * `poly.verts[...].orient` must be set prior to calling this function.
 *
 * \param [in] andcmp
 * Set of bit flags allowing faces to be skipped. Face `f` will be skipped if `andcmp & (1 << f)`
 * evaluates to `true`. 
 *
 */
void r2d_clip_quad(r2d_poly* poly, unsigned char andcmp);


/**
 * \brief Integrate a polynomial density over a polygon using simplicial decomposition
 *
 * \param [in] poly
 * The polygon over which to integrate.
 *
 * \param [in] polyorder
 * Order of the polynomial density field. 0 for constant (1 moment), 1 for linear
 * (3 moments), 2 for quadratic (6 moments).
 *
 * \param [in, out] moments
 * Array to be filled with the integration results, up to the sepcified `polyorder`. 
 * Order of moments is `1`, `x`, `y`, `x^2`, `y^2`, `x*y`.
 *
 */
void r2d_reduce(r2d_poly* poly, r2d_int polyorder, r2d_real* moments);


/**
 * \brief Initialize a polygon as an axis-aligned box. 
 *
 * \param [in, out] poly
 * The polygon to initialize.
 *
 * \param [in] rbounds
 * An array of two vectors, giving the lower and upper corners of the box.
 *
 */
void r2du_init_box(r2d_poly* poly, r2d_rvec2 rbounds[2]);


/**
 * \brief Get faces (unit normals and distances to the origin) 
 * from an ordered-vertex description of a convex polygon. 
 *
 * \param [in] verts
 * List of polygon vertices.
 *
 * \param [in] nverts
 * Number of polygon vertices in the list.
 *
 * \param [out] faces
 * Array of planes defining the faces of the polygon. Must be at least `nverts` in size.
 *
 */
void r2du_faces_from_verts(r2d_rvec2* verts, r2d_int nverts, r2d_plane* faces);

/**
 * \brief Get the signed area of the triangle defined by the input vertices. 
 *
 * \param [in] pa, pb, pc
 * Vertices defining a triangle from which to calculate an area. 
 *
 * \return
 * The signed area of the input triangle.
 *
 */
r2d_real r2du_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc);


#endif // _R2D_H_
