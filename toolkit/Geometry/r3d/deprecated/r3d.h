/**************************************************************
 *
 *    r3d.h
 *    
 *    Devon Powell
 *    8 February 2015
 *
 *    Routines for fast, geometrically robust clipping operations
 *    and analytic volume/moment computations over convex
 *    polyhedra in 3D. Includes a wrapper function for 
 *    physically conservative voxelization (volume sampling)
 *    of tetrahedra to a Cartesian grid.
 *    
 *    Copyright (C) 2014 Stanford University.
 *    See License.txt for more information.
 *
 *************************************************************/

#ifndef _R3D_H_
#define _R3D_H_

#include <stdint.h>

/**
 * \file r3d.h
 * \author Devon Powell
 * \date 8 Feb 2015
 * \brief Interface for r3d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double` (recommended). `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r3d_real;
#else 
typedef double r3d_real;
#endif

/**
 * \brief Integer type used for grid indexing and bit flags
 */
typedef int32_t r3d_int;

/**
 * \brief Long integer type used for grid indexing
 */
typedef int64_t r3d_long;

/** \struct r3d_rvec3
 *  \brief Vector struct.
 */
typedef struct {
	r3d_real x, /*!< \f$x\f$-component. */
			 y, /*!< \f$y\f$-component. */
			 z; /*!< \f$z\f$-component. */
} r3d_rvec3;

/** \struct r3d_dvec3
 *  \brief Integer vector struct for grid indexing.
 */
typedef struct {
	r3d_int i, /*!< \f$x\f$-index. */
			j, /*!< \f$y\f$-index. */
			k; /*!< \f$z\f$-index. */
} r3d_dvec3;

/** \struct r3d_plane
 *  \brief A plane.
 */
typedef struct {
	r3d_rvec3 n; /*!< Unit-length normal vector. */
	r3d_real d; /*!< Signed perpendicular distance to the origin. */
} r3d_plane;

/** \struct r3d_orientation
 *  \brief Perpendicular distances and bit flags for up to 6 faces.
 */
typedef struct {
	r3d_real fdist[4]; /*!< Signed distances to clip planes. */
	unsigned char fflags; /*!< Bit flags corresponding to inside (1) or outside (0) of each clip plane. */
} r3d_orientation;

/** \struct r3d_vertex
 * \brief A doubly-linked vertex.
 */
typedef struct {
	r3d_rvec3 pos; /*!< Vertex position. */
	unsigned char pnbrs[3]; /*!< Neighbor indices. */
	r3d_orientation orient; /*!< Orientation with respect to clip planes. */
} r3d_vertex;

/** \struct r3d_poly
 * \brief A polyhedron.
 */
typedef struct {
#define R3D_MAX_VERTS 64
	r3d_vertex verts[R3D_MAX_VERTS]; /*!< Vertex buffer. */
	r3d_int nverts; /*!< Number of vertices in the buffer. */
	r3d_int funclip; /*!< Index of any unclipped vertex. */
} r3d_poly;

/** \struct r3d_dest_grid
 *  \brief Destination grid information.
 */
typedef struct {
	r3d_real* moments[10]; /*!< Gridded moments in row-major (C) order.
							   Must be at least `n.x*n.y*n.z` in size. */ 
	r3d_int polyorder; /*!< Polynomial order (0 for constant, 1 for linear, 2 for quadratic) to be voxelized. */
	r3d_orientation* orient; /*!< Buffer for gridpoint orientation checks.
							   Must be at least `(n.x+1)*(n.y+1)*(n.z+1)` in size.
							   Unused if compiled with `-DUSE_TREE`.*/ 
	r3d_long bufsz; /*!< Allocated size of moments and orient buffers. */
	r3d_dvec3 n; /*!< Grid dimensions (cells per coordinate). */
	r3d_rvec3 d; /*!< Grid cell size. */
} r3d_dest_grid;

/**
 * \brief Tabulated number of moments needed for a given polynomial order.
 */
const static r3d_int r3d_num_moments[3] = {1, 4, 10};

/**
 * \brief Voxelize a tetrahedron to the destination grid.
 *
 * \param [in] faces
 * The four faces of the tetrahedron to be voxelized.
 *
 * \param [in, out] grid
 * The destination grid buffer that the tetrahedron will be voxelized to.
 * The results of the voxelization are found in `grid.moments`.
 *
 */
void r3d_voxelize_tet(r3d_plane* faces, r3d_dest_grid* grid);


/**
 * \brief Clip a polyhedron against four clip planes (find its intersection with a tetrahedron). 
 *
 * \param [in, out] poly 
 * The polyehdron to be clipped. The distances to the clip plane and bit flags in
 * `poly.verts[...].orient` must be set prior to calling this function.
 *
 * \param [in] andcmp
 * Set of bit flags allowing faces to be skipped. Face `f` will be skipped if `andcmp & (1 << f)`
 * evaluates to `true`. 
 *
 */
void r3d_clip_tet(r3d_poly* poly, unsigned char andcmp);


/**
 * \brief Integrate a polynomial density over a convex polyhedron using simplicial decomposition
 *
 * \param [in] poly
 * The polyhedron over which to integrate.
 *
 * \param [in] polyorder
 * Order of the polynomial density field. 0 for constant (1 moment), 1 for linear
 * (4 moments), 2 for quadratic (10 moments).
 *
 * \param [in, out] moments
 * Array to be filled with the integration results, up to the sepcified `polyorder`. 
 * Order of moments is `1`, `x`, `y`, `z`, `x^2`, `y^2`, `z^2`, `x*y`, `y*z`, `z*x`.
 *
 */
void r3d_reduce(r3d_poly* poly, r3d_int polyorder, r3d_real* moments);

/**
 * \brief Initialize a polyhedron as an axis-aligned cube. 
 *
 * \param [in, out] poly
 * The polyhedron to initialize.
 *
 * \param [in] rbounds
 * An array of two vectors, giving the lower and upper corners of the box.
 *
 */
void r3du_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]);


/**
 * \brief Get four faces (unit normals and distances to the origin) i
 * from a four-vertex description of a tetrahedron. 
 *
 * \param [in] verts
 * Array of four vectors defining the vertices of the tetrahedron.
 *
 * \param [out] faces
 * Array of four planes defining the faces of the tetrahedron.
 *
 */
void r3du_tet_faces_from_verts(r3d_rvec3* verts, r3d_plane* faces);


/**
 * \brief Get the signed volume of the tetrahedron defined by the input vertices. 
 *
 * \param [in] pa, pb, pc, pd
 * Vertices defining a tetrahedron from which to calculate a volume. 
 *
 * \return
 * The signed volume of the input tetrahedron.
 *
 */
r3d_real r3du_orient(r3d_rvec3 pa, r3d_rvec3 pb, r3d_rvec3 pc, r3d_rvec3 pd);

#endif // _R3D_H_
