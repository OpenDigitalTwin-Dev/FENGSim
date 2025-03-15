/*
 *
 *		rNd.h
 *		
 *		Routines for fast, geometrically robust clipping operations
 *		and analytic volume/moment computations over 
 *		polytopes of arbitrary dimensionality. 
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

#ifndef _RND_H_
#define _RND_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * \file rNd.h
 * \author Devon Powell
 * \date 31 August 2015
 * \brief Interface for rNd
 */

/*
 * \brief Dimensionality of the geometric calculations.
 */
#define RND_DIM 4

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double` (recommended). `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float rNd_real;
#else 
typedef double rNd_real;
#endif

/**
 * \brief Integer type used for indexing
 */
typedef int32_t rNd_int;
typedef int64_t rNd_long;

/** \struct rNd_rvec
 *  \brief A N-vector.
 */
typedef struct {
	rNd_real xyz[RND_DIM]; /*!< Index-based access to components. */
} rNd_rvec;

/** \struct rNd_dvec
 *  \brief An integer N-vector for grid indexing.
 */
typedef struct {
	rNd_int ijk[RND_DIM]; /*!< Index-based access to components. */
} rNd_dvec;

/** \struct rNd_plane
 *  \brief A hyperplane.
 */
typedef struct {
	rNd_rvec n; /*!< Unit-length normal vector. */
	rNd_real d; /*!< Signed perpendicular distance to the origin. */
} rNd_plane;

/** \struct rNd_vertex
 * \brief A doubly-linked vertex.
 */
typedef struct {
	rNd_int pnbrs[RND_DIM]; /*!< Neighboring vertices. */
	rNd_int finds[RND_DIM][RND_DIM]; /*!< 2-face connectivity (needed for `RND_DIM > 3`). */
	rNd_rvec pos; /*!< Position. */
} rNd_vertex;

/** \struct rNd_poly
 * \brief An `RND_DIM`-dimensional polytope.
 */
typedef struct {
#define RND_MAX_VERTS 1024 
	rNd_vertex verts[RND_MAX_VERTS]; /*!< Vertex buffer. */
	rNd_int nverts, nfaces; /*!< Number of vertices and faces. */
} rNd_poly;

/**
 * \brief Clip a polytope against an arbitrary number of hyperplanes (find its intersection with a set of half-spaces). 
 *
 * \param [in, out] poly 
 * The polytope to be clipped. 
 *
 * \param [in] planes 
 * An array of hyperplanes against which to clip this polytope.
 *
 * \param[in] nplanes 
 * The number of hyperplanes in the input array. 
 *
 */
void rNd_clip(rNd_poly* poly, rNd_plane* planes, rNd_int nplanes);

/**
 * \brief Integrate a polynomial density over a polytope. Still under development.
 *
 * \param [in] poly
 * The polytope over which to integrate.
 *
 * \param [in] polyorder
 * Order of the polynomial density field. 0 for constant (1 moment), 1 for linear
 * (`RND_DIM+1` moments), etc.
 *
 * \param [in, out] moments
 * Array to be filled with the integration results, up to the specified `polyorder`. 
 * Order of moments is row-major.
 *
 */
//#define RND_NUM_MOMENTS(order) ((order+1)*(order+2)*(order+3)/6)
void rNd_reduce(rNd_poly* poly, rNd_real* moments, rNd_int polyorder);

/**
 * \brief Checks a polytope to see if it is a valid `RND_DIM`-polyhedral graph, by testing
 * compliance with Balinski's theorem. 
 *
 * \param [in] poly
 * The polytope to check.
 *
 * \return
 * 1 if the polytope is valid, 0 if not. 
 *
 */
rNd_int rNd_is_good(rNd_poly* poly);

/**
 * \brief Get the signed volume of the `RND_DIM`-simplex defined by the input vertices. 
 *
 * \param [in] verts 
 * `RND_DIM+1` vertices defining a simplex from which to calculate a volume. 
 *
 * \return
 * The signed volume of the input simplex.
 *
 */
rNd_real rNd_orient(rNd_rvec verts[RND_DIM+1]);

/**
 * \brief Prints the vertices and connectivity of a polytope. For debugging. 
 *
 * \param [in] poly
 * The polytope to print.
 *
 */
void rNd_print(rNd_poly* poly);

/**
 * \brief Rotate a polytope about one axis. 
 *
 * \param [in, out] poly 
 * The polytope to rotate. 
 *
 * \param [in] theta 
 * The angle, in radians, by which to rotate the polytope.
 *
 * \param [in] ax1 
 * Along with `ax2`, the plane in which to rotate the polytope.
 *
 * \param [in] ax2 
 * Along with `ax1`, the plane in which to rotate the polytope.
 *
 */
void rNd_rotate(rNd_poly* poly, rNd_real theta, rNd_int ax1, rNd_int ax2);

/**
 * \brief Translate a polytope. 
 *
 * \param [in, out] poly 
 * The polytope to translate. 
 *
 * \param [in] shift 
 * The vector by which to translate the polytope. 
 *
 */
void rNd_translate(rNd_poly* poly, rNd_rvec shift);

/**
 * \brief Scale a polytope.
 *
 * \param [in, out] poly 
 * The polytope to scale. 
 *
 * \param [in] scale 
 * The factor by which to scale the polytope along each dimension. 
 *
 */
void rNd_scale(rNd_poly* poly, rNd_rvec scale);

/**
 * \brief Shear a polytope. Each vertex undergoes the transformation
 * `pos.xyz[axb] += shear*pos.xyz[axs]`.
 *
 * \param [in, out] poly 
 * The polytope to shear. 
 *
 * \param [in] shear 
 * The factor by which to shear the polytope. 
 *
 * \param [in] axb 
 * The axis along which to shear the polytope.
 *
 * \param [in] axs 
 * The axis across which to shear the polytope.
 *
 */
void rNd_shear(rNd_poly* poly, rNd_real shear, rNd_int axb, rNd_int axs);

/**
 * \brief Apply a general affine transformation to a polytope. 
 *
 * \param [in, out] poly 
 * The polytope to transform.
 *
 * \param [in] mat 
 * The `(RND_DIM+1)*(RND_DIM+1)` homogeneous transformation matrix.
 *
 */
void rNd_affine(rNd_poly* poly, rNd_real mat[RND_DIM+1][RND_DIM+1]);

/**
 * \brief Initialize a polytope as an `RND_DIM`-simplex. 
 *
 * \param [out] poly
 * The polytope to initialize.
 *
 * \param [in] verts
 * An array of `RND_DIM+1` vectors, giving the vertices of the simplex.
 *
 */
void rNd_init_simplex(rNd_poly* poly, rNd_rvec verts[RND_DIM+1]);

/**
 * \brief Initialize a polytope as an axis-aligned hyperrectangle. 
 *
 * \param [out] poly
 * The polytope to initialize.
 *
 * \param [in] rbounds
 * An array of two vectors, giving the lower and upper corners of the box.
 *
 */
void rNd_init_box(rNd_poly* poly, rNd_rvec rbounds[2]);

/**
 * \brief Get `RND_DIM+1` hyperfaces (unit normals and distances to the origin)
 * from a `RND_DIM+1`-vertex description of a simplex. 
 *
 * \param [out] faces
 * Array of `RND_DIM+1` hyperplanes defining the faces of the simplex.
 *
 * \param [in] verts
 * Array of `RND_DIM+1` vectors defining the vertices of the simplex.
 *
 */
//void rNd_simplex_faces_from_verts(rNd_plane* faces, rNd_rvec3* verts);

/**
 * \brief Get `2*RND_DIM` hyperfaces (unit normals and distances to the origin)
 * from a two-vertex description of an axis-aligned hyperrectangle.
 *
 * \param [out] faces
 * Array of `2*RND_DIM` hyperplanes defining the faces of the box.
 *
 * \param [in] rbounds
 * Array of two vectors defining the bounds of the axis-aligned box 
 *
 */
//void rNd_box_faces_from_verts(rNd_plane* faces, rNd_rvec3* rbounds);

#ifdef __cplusplus
}
#endif

#endif // _RND_H_
