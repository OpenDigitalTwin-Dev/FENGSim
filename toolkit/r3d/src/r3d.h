/*
 *
 *		r3d.h
 *
 *		Routines for fast, geometrically robust clipping operations and analytic
 *		volume/moment computations over polyhedra in 3D.
 *
 *		Devon Powell 31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los
 *		Alamos National Laboratory (LANL) under contract No. DE-AC52-06NA25396
 *		with the U.S. Department of Energy (DOE). All rights in the program are
 *		reserved by the DOE and Los Alamos National Security, LLC. Permission is
 *		granted to the public to copy and use this software without charge,
 *		provided that this Notice and any statement of authorship are reproduced
 *		on all copies.  Neither the U.S. Government nor LANS makes any warranty,
 *		express or implied, or assumes any liability or responsibility for the use
 *		of this software.
 *
 */

#ifndef _R3D_H_
#define _R3D_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "r3d-config.h"  /* Defines R3D_MAX_VERTS. auto-generated at compile time */

/**
 * \file r3d.h \author Devon Powell \date 31 August 2015 \brief Interface for
 * r3d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double`. `float` precision is enabled by compiling with
 * `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r3d_real;
#else
typedef double r3d_real;
#endif

/**
 * \brief Integer types used for indexing
 */
typedef int32_t r3d_int;
typedef int64_t r3d_long;

/** \struct r3d_rvec3 \brief A 3-vector.
 */
typedef union {
  struct
  {
    r3d_real x, /*!< \f$x\f$-component. */
        y,      /*!< \f$y\f$-component. */
        z;      /*!< \f$z\f$-component. */
  };
  r3d_real xyz[3]; /*!< Index-based access to components. */
} r3d_rvec3;

/** \struct r3d_dvec3 \brief An integer 3-vector for grid indexing.
 */
typedef union {
  struct
  {
    r3d_int i, /*!< \f$x\f$-component. */
        j,     /*!< \f$y\f$-component. */
        k;     /*!< \f$z\f$-component. */
  };
  r3d_int ijk[3]; /*!< Index-based access to components. */
} r3d_dvec3;

/** \struct r3d_plane \brief A plane.
 */
typedef struct
{
  r3d_rvec3 n; /*!< Unit-length normal vector. */
  r3d_real d;  /*!< Signed perpendicular distance to the origin. */
} r3d_plane;

/** \struct r3d_vertex \brief A doubly-linked vertex.
 */
typedef struct
{
  r3d_int pnbrs[3]; /*!< Neighbor indices. */
  r3d_rvec3 pos;    /*!< Vertex position. */
} r3d_vertex;

/** \struct r3d_poly \brief A polyhedron. Can be convex, nonconvex, even
 * multiply-connected.
 */
typedef struct
{
  r3d_vertex verts[R3D_MAX_VERTS]; /*!< Vertex buffer. */
  r3d_int nverts;                  /*!< Number of vertices in the buffer. */
} r3d_poly;

/**
 * \brief Clip a polyhedron against an arbitrary number of clip planes (find its
 * intersection with a set of half-spaces).
 *
 * \param [in, out] poly The polyehdron to be clipped.
 *
 * \param [in] planes An array of planes against which to clip this polyhedron.
 *
 * \param[in] nplanes The number of planes in the input array.
 *
 * \return Status code indicating if operation was successful (1 - successful, 0 - unsuccessful)
 */
int r3d_clip(r3d_poly *poly, r3d_plane *planes, r3d_int nplanes);

/**
 * \brief Splits a list of polyhedra across a single plane.
 *
 * \param [in] inpolys Array of input polyhedra to be split
 *
 * \param [in] npolys The number of input polyhedra
 *
 * \param [in] plane The plane about which to split the input polys
 *
 * \param[out] out_pos The output array of fragments on the positive side of the
 * clip plane. Must be at least npolys long. out_pos[i] and out_neg[i]
 * correspond to inpolys[i], where out_neg[i].nverts or out.pos[i].nverts are
 * set to zero if the poly lies entirely in the positive or negative side of the
 * plane, respectively.

 *
 * \param[out] out_neg The output array of fragments on the negitive side of the
 * clip plane. Must be at least npolys long.
 *
 * \return Status code indicating if operation was successful (1 - successful, 0 - unsuccessful)
 */
int r3d_split(r3d_poly *inpolys, r3d_int npolys, r3d_plane plane,
               r3d_poly *out_pos, r3d_poly *out_neg);

/**
 * \brief Integrate a polynomial density over a polyhedron using simplicial
 * decomposition. Uses the fast recursive method of Koehl (2012) to carry out
 * the integration.
 *
 * \param [in] poly The polyhedron over which to integrate.
 *
 * \param [in] polyorder Order of the polynomial density field. 0 for constant
 * (1 moment), 1 for linear (4 moments), 2 for quadratic (10 moments), etc.
 *
 * \param [in, out] moments Array to be filled with the integration results, up
 * to the specified `polyorder`. Must be at least
 * `(polyorder+1)*(polyorder+2)*(polyorder+3)/6` long. A conventient macro,
 * `R3D_NUM_MOMENTS()` is provided to compute the number of moments for a given
 * order. Order of moments is row-major, i.e. `1`, `x`, `y`, `z`, `x^2`, `x*y`,
 * `x*z`, `y^2`, `y*z`, `z^2`, `x^3`, `x^2*y`...
 *
 */
#define R3D_NUM_MOMENTS(order) ((order + 1) * (order + 2) * (order + 3) / 6)
void r3d_reduce(r3d_poly *poly, r3d_real *moments, r3d_int polyorder);

/**
 * \brief Checks a polyhedron to see if all vertices have three valid edges,
 * that all vertices are pointed to by three other vertices, and that there are
 * no sets of two vertices with more than one edge between them. Overall, checks
 * that the graph is 3-vertex-connected (an `O(nverts^2)` operation), which is
 * required by Steinitz' theorem to be a valid polyhedral graph.
 *
 * \param [in] poly The polyhedron to check.
 *
 * \return 1 if the polyhedron is good, 0 if not.
 *
 */
r3d_int r3d_is_good(r3d_poly *poly);

/**
 * \brief Calculates a center of a polyhedron.
 *
 * \param [in] poly
 * The polygon to check.
 *
 * \return
 * coordinates of a polygon center.
 *
 */
r3d_rvec3 r3d_poly_center(r3d_poly* poly);

/**
 * \brief Adjust moments according to the shift of polyhedron vertices to the origin.
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
void r3d_shift_moments(r3d_real* moments, r3d_int polyorder, r3d_rvec3 vc);

/**
 * \brief Get the signed volume of the tetrahedron defined by the input
 * vertices.
 *
 * \param [in] verts Four vertices defining a tetrahedron from which to
 * calculate a volume.
 *
 * \return The signed volume of the input tetrahedron.
 *
 */
r3d_real r3d_orient(r3d_rvec3 *verts);

/**
 * \brief Prints the vertices and connectivity of a polyhedron. For debugging.
 *
 * \param [in] poly The polyhedron to print.
 *
 */
void r3d_print(r3d_poly *poly);

/**
 * \brief Rotate a polyhedron about one axis.
 *
 * \param [in, out] poly The polyehdron to rotate.
 *
 * \param [in] theta The angle, in radians, by which to rotate the polyhedron.
 *
 * \param [in] axis The axis (0, 1, or 2 corresponding to x, y, or z) about
 * which to rotate the polyhedron.
 *
 */
void r3d_rotate(r3d_poly *poly, r3d_real theta, r3d_int axis);

/**
 * \brief Translate a polyhedron.
 *
 * \param [in, out] poly The polyhedron to translate.
 *
 * \param [in] shift The vector by which to translate the polyhedron.
 *
 */
void r3d_translate(r3d_poly *poly, r3d_rvec3 shift);

/**
 * \brief Scale a polyhedron.
 *
 * \param [in, out] poly The polyhedron to scale.
 *
 * \param [in] shift The factor by which to scale the polyhedron.
 *
 */
void r3d_scale(r3d_poly *poly, r3d_real scale);

/**
 * \brief Shear a polyhedron. Each vertex undergoes the transformation
 * `pos.xyz[axb] += shear*pos.xyz[axs]`.
 *
 * \param [in, out] poly The polyhedron to shear.
 *
 * \param [in] shear The factor by which to shear the polyhedron.
 *
 * \param [in] axb The axis (0, 1, or 2 corresponding to x, y, or z) along which
 * to shear the polyhedron.
 *
 * \param [in] axs The axis (0, 1, or 2 corresponding to x, y, or z) across
 * which to shear the polyhedron.
 *
 */
void r3d_shear(r3d_poly *poly, r3d_real shear, r3d_int axb, r3d_int axs);

/**
 * \brief Apply a general affine transformation to a polyhedron.
 *
 * \param [in, out] poly The polyhedron to transform.
 *
 * \param [in] mat The 4x4 homogeneous transformation matrix by which to
 * transform the vertices of the polyhedron.
 *
 */
void r3d_affine(r3d_poly *poly, r3d_real mat[4][4]);

/**
 * \brief Initialize a polyhedron as a tetrahedron.
 *
 * \param [out] poly The polyhedron to initialize.
 *
 * \param [in] verts An array of four vectors, giving the vertices of the
 * tetrahedron.
 *
 */
void r3d_init_tet(r3d_poly *poly, r3d_rvec3 *verts);

/**
 * \brief Initialize a polyhedron as an axis-aligned cube.
 *
 * \param [out] poly The polyhedron to initialize.
 *
 * \param [in] rbounds An array of two vectors, giving the lower and upper
 * corners of the box.
 *
 */
void r3d_init_box(r3d_poly *poly, r3d_rvec3 *rbounds);

/**
 * \brief Initialize a general polyhedron from a full boundary description. Can
 * use `r3d_is_good` to check that the output is valid.
 *
 * \param [out] poly The polyhedron to initialize.
 *
 * \param [in] vertices Array of length `numverts` giving the vertices of the
 * input polyhedron.
 *
 * \param [in] numverts Number of vertices in the input polyhedron.
 *
 * \param [in] faceinds Connectivity array, giving the indices of vertices in
 * the order they appear around each face of the input polyhedron.
 *
 * \param [in] numvertsperface An array of length `numfaces` giving the number
 * of vertices for each face of the input polyhedron.
 *
 * \param [in] numfaces Number of faces in the input polyhedron.
 *
 */
int r3d_init_poly(r3d_poly *poly, r3d_rvec3 *vertices, r3d_int numverts,
                   r3d_int **faceinds, r3d_int *numvertsperface,
                   r3d_int numfaces);

/**
 * \brief Get four faces (unit normals and distances to the origin) from a
 * four-vertex description of a tetrahedron.
 *
 * \param [out] faces Array of four planes defining the faces of the
 * tetrahedron.
 *
 * \param [in] verts Array of four vectors defining the vertices of the
 * tetrahedron.
 *
 */
void r3d_tet_faces_from_verts(r3d_plane *faces, r3d_rvec3 *verts);

/**
 * \brief Get six faces (unit normals and distances to the origin) from a
 * two-vertex description of an axis-aligned box.
 *
 * \param [out] faces Array of six planes defining the faces of the box.
 *
 * \param [in] rbounds Array of two vectors defining the bounds of the
 * axis-aligned box
 *
 */
void r3d_box_faces_from_verts(r3d_plane *faces, r3d_rvec3 *rbounds);

/**
 * \brief Get all faces (unit normals and distances to the origin) from a full
 * boundary description of a polyhedron.
 *
 * \param [out] faces Array of planes of length `numfaces` defining the faces of
 * the polyhedron.
 *
 * \param [in] vertices Array of length `numverts` giving the vertices of the
 * input polyhedron.
 *
 * \param [in] numverts Number of vertices in the input polyhedron.
 *
 * \param [in] faceinds Connectivity array, giving the indices of vertices in
 * the order they appear around each face of the input polyhedron.
 *
 * \param [in] numvertsperface An array of length `numfaces` giving the number
 * of vertices for each face of the input polyhedron.
 *
 * \param [in] numfaces Number of faces in the input polyhedron.
 *
 */
void r3d_poly_faces_from_verts(r3d_plane *faces, r3d_rvec3 *vertices,
                               r3d_int numverts, r3d_int **faceinds,
                               r3d_int *numvertsperface, r3d_int numfaces);

/**
 * create a type for the boundary representation
 */
typedef struct r3d_brep
{
  r3d_rvec3 *vertices;
  r3d_int numvertices;
  r3d_int **faceinds;
  r3d_int *numvertsperface;
  r3d_int numfaces;
} r3d_brep;

/**
 * \brief Convert from R3D's internal vertex-edge format to the boundary
 * representation.
 *
 * \param [in] poly Pointer to a single R3D polyhedron.
 *
 * \param [out] brep Pointer to an array of boundary representations that will
 * hold the different boundary representations for the different components.
 *
 * \param [out] numcomponents Pointer to an integer that will hold the number
 * of determined components.
 *
 */
void r3d_init_brep(r3d_poly *poly, r3d_brep **brep, r3d_int *numcomponents);


/**
 * \brief Prints the boundary representation of an array of polyhedons given
 * in boundary representation form. For debugging.
 *
 * \param [in] brep Pointer to an array of boundary representations.
 *
 * \param [in] numcomponents The number of disconnected components in the
 * boundary representation array.
 *
 */
void r3d_print_brep(r3d_brep **brep, r3d_int numcomponents);

/**
 * \brief Free all the memory associated with the boundary representation array.
 *
 * \param [in] brep  Pointer to an array of boundary representations.
 *
 * \param [in] numcomponents  The number of disconnected components in the
 * boundary representation array.
 * 
 */
void r3d_free_brep(r3d_brep **brep, r3d_int numcomponents);

#ifdef __cplusplus
}
#endif

#endif // _R3D_H_
