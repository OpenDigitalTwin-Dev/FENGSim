/*
 *
 *		r2d.h
 *		
 *		Routines for fast, geometrically robust clipping operations
 *		and analytic area/moment computations over polygons in 2D. 
 *		
 *		Devon Powell
 *		31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *
 */

#ifndef _R2D_H_
#define _R2D_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * \file r2d.h
 * \author Devon Powell
 * \date 31 August 2015
 * \brief Interface for r2d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double`. `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r2d_real;
#else 
typedef double r2d_real;
#endif

/**
 * \brief Integer types used for indexing
 */
typedef int32_t r2d_int;
typedef int64_t r2d_long;

/** \struct r2d_rvec2
 *  \brief A 2-vector.
 */
typedef union {
	struct {
		r2d_real x, /*!< \f$x\f$-component. */
				 y; /*!< \f$y\f$-component. */
	};
	r2d_real xy[2]; /*!< Index-based access to components. */
} r2d_rvec2;

/** \struct r2d_dvec2
 *  \brief An integer 2-vector for grid indexing.
 */
typedef union {
	struct {
		r2d_int i, /*!< \f$x\f$-component. */
				j; /*!< \f$y\f$-component. */
	};
	r2d_int ij[2]; /*!< Index-based access to components. */
} r2d_dvec2;

/** \struct r2d_plane
 *  \brief A plane.
 */
typedef struct {
	r2d_rvec2 n; /*!< Unit-length normal vector. */
	r2d_real d; /*!< Signed perpendicular distance to the origin. */
} r2d_plane;

/** \struct r2d_vertex
 * \brief A doubly-linked vertex.
 */
typedef struct {
	r2d_int pnbrs[2]; /*!< Neighbor indices. */
	r2d_rvec2 pos; /*!< Vertex position. */
} r2d_vertex;

/** \struct r2d_poly
 * \brief A polygon. Can be convex, nonconvex, even multiply-connected.
 */
typedef struct {
#define R2D_MAX_VERTS 256 
	r2d_vertex verts[R2D_MAX_VERTS]; /*!< Vertex buffer. */
	r2d_int nverts; /*!< Number of vertices in the buffer. */
} r2d_poly;

/**
 * \brief Clip a polygon against an arbitrary number of clip planes (find its intersection with a set of half-spaces). 
 *
 * \param [in, out] poly 
 * The polygon to be clipped. 
 *
 * \param [in] planes 
 * An array of planes against which to clip this polygon.
 *
 * \param[in] nplanes 
 * The number of planes in the input array. 
 *
 */
void r2d_clip(r2d_poly* poly, r2d_plane* planes, r2d_int nplanes);

/**
 * \brief Splits a list of polygons across a single plane.  
 *
 * \param [in] inpolys 
 * Array input polyhedra to be split 
 *
 * \param [in] npolys 
 * The number of input polygons
 *
 * \param [in] plane 
 * The plane about which to split the input polys 
 *
 * \param[out] out_pos 
 * The output array of fragments on the positive side of the clip plane. Must be at least npolys
 * long. out_pos[i] and out_neg[i] correspond to inpolys[i], where out_neg[i].nverts or
 * out.pos[i].nverts are set to zero if the poly lies entirely in the positive or negative side of
 * the plane, respectively.
 *
 * \param[out] out_neg 
 * The output array of fragments on the negitive side of the clip plane. Must be at least npolys
 * long. 
 *
 */
void r2d_split(r2d_poly* inpolys, r2d_int npolys, r2d_plane plane, r2d_poly* out_pos, r2d_poly* out_neg);

/**
 * \brief Integrate a polynomial density over a polygon using simplicial decomposition.
 * Uses the fast recursive method of Koehl (2012) to carry out the integration.
 *
 * \param [in] poly
 * The polygon over which to integrate.
 *
 * \param [in] polyorder
 * Order of the polynomial density field. 0 for constant (1 moment), 1 for linear
 * (4 moments), 2 for quadratic (10 moments), etc.
 *
 * \param [in, out] moments
 * Array to be filled with the integration results, up to the specified `polyorder`. Must be at
 * least `(polyorder+1)*(polyorder+2)/2` long. A conventient macro,
 * `R2D_NUM_MOMENTS()` is provided to compute the number of moments for a given order.
 * Order of moments is row-major, i.e. `1`, `x`, `y`, `x^2`, `x*y`, `y^2`, `x^3`, `x^2*y`...
 *
 */
#define R2D_NUM_MOMENTS(order) ((order+1)*(order+2)/2)
void r2d_reduce(r2d_poly* poly, r2d_real* moments, r2d_int polyorder);

/**
 * \brief Checks a polygon to see if all vertices have two valid edges, that all vertices are
 * pointed to by two other vertices, and that there are no vertices that point to themselves. 
 *
 * \param [in] poly
 * The polygon to check.
 *
 * \return
 * 1 if the polygon is good, 0 if not. 
 *
 */
r2d_int r2d_is_good(r2d_poly* poly);

/**
 * \brief Calculates a center of a polygon.
 *
 * \param [in] poly
 * The polygon to check.
 *
 * \return
 * coordinates of a polygon center.
 *
 */
r2d_rvec2 r2d_poly_center(r2d_poly* poly);

/**
 * \brief Adjust moments according to the shift of polygon vertices to the origin.
 *
 * \param [in, out] moments
 * The moments of the shifted polygon.
 *
 * \param [in] polyorder
 * Order of the polygon.
 *
 * \param [in] vc
 * Coordinates of the polygon center, which are used to shift the polygon.
 *
 */
void r2d_shift_moments(r2d_real* moments, r2d_int polyorder, r2d_rvec2 vc);

/**
 * \brief Get the signed volume of the triangle defined by the input vertices. 
 *
 * \param [in] pa, pb, pc
 * Vertices defining a triangle from which to calculate an area. 
 *
 * \return
 * The signed volume of the input triangle.
 *
 */
r2d_real r2d_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc);

/**
 * \brief Prints the vertices and connectivity of a polygon. For debugging. 
 *
 * \param [in] poly
 * The polygon to print.
 *
 */
void r2d_print(r2d_poly* poly);

/**
 * \brief Rotate a polygon about the origin. 
 *
 * \param [in, out] poly 
 * The polygon to rotate. 
 *
 * \param [in] theta 
 * The angle, in radians, by which to rotate the polygon.
 *
 */
void r2d_rotate(r2d_poly* poly, r2d_real theta);

/**
 * \brief Translate a polygon. 
 *
 * \param [in, out] poly 
 * The polygon to translate. 
 *
 * \param [in] shift 
 * The vector by which to translate the polygon. 
 *
 */
void r2d_translate(r2d_poly* poly, r2d_rvec2 shift);

/**
 * \brief Scale a polygon.
 *
 * \param [in, out] poly 
 * The polygon to scale. 
 *
 * \param [in] shift 
 * The factor by which to scale the polygon. 
 *
 */
void r2d_scale(r2d_poly* poly, r2d_real scale);

/**
 * \brief Shear a polygon. Each vertex undergoes the transformation
 * `pos.xy[axb] += shear*pos.xy[axs]`.
 *
 * \param [in, out] poly 
 * The polygon to shear. 
 *
 * \param [in] shear 
 * The factor by which to shear the polygon. 
 *
 * \param [in] axb 
 * The axis (0 or 1 corresponding to x or y) along which to shear the polygon.
 *
 * \param [in] axs 
 * The axis (0 or 1 corresponding to x or y) across which to shear the polygon.
 *
 */
void r2d_shear(r2d_poly* poly, r2d_real shear, r2d_int axb, r2d_int axs);

/**
 * \brief Apply a general affine transformation to a polygon. 
 *
 * \param [in, out] poly 
 * The polygon to transform.
 *
 * \param [in] mat 
 * The 3x3 homogeneous transformation matrix by which to transform
 * the vertices of the polygon.
 *
 */
void r2d_affine(r2d_poly* poly, r2d_real mat[3][3]);

/**
 * \brief Initialize a polygon as an axis-aligned box. 
 *
 * \param [out] poly
 * The polygon to initialize.
 *
 * \param [in] rbounds
 * An array of two vectors, giving the lower and upper corners of the box.
 *
 */
void r2d_init_box(r2d_poly* poly, r2d_rvec2* rbounds);

/**
 * \brief Initialize a (simply-connected) general polygon from a list of vertices. 
 * Can use `r2d_is_good` to check that the output is valid.
 *
 * \param [out] poly
 * The polygon to initialize. 
 *
 * \param [in] vertices
 * Array of length `numverts` giving the vertices of the input polygon, in counterclockwise order.
 *
 * \param [in] numverts
 * Number of vertices in the input polygon. 
 *
 */
void r2d_init_poly(r2d_poly* poly, r2d_rvec2* vertices, r2d_int numverts);

/**
 * \brief Get four faces (unit normals and distances to the origin)
 * from a two-vertex description of an axis-aligned box.
 *
 * \param [out] faces
 * Array of four planes defining the faces of the box.
 *
 * \param [in] rbounds
 * Array of two vectors defining the bounds of the axis-aligned box 
 *
 */
void r2d_box_faces_from_verts(r2d_plane* faces, r2d_rvec2* rbounds);

/**
 * \brief Get all faces (unit normals and distances to the origin)
 * from a general boundary description of a polygon.
 *
 * \param [out] faces
 * Array of planes of length `numverts` defining the faces of the polygon.
 *
 * \param [in] vertices
 * Array of length `numverts` giving the vertices of the input polygon, in counterclockwise order. 
 *
 * \param [in] numverts
 * Number of vertices in the input polygon. 
 *
 */
void r2d_poly_faces_from_verts(r2d_plane* faces, r2d_rvec2* vertices, r2d_int numverts);

#ifdef __cplusplus
}
#endif

#endif // _R2D_H_
