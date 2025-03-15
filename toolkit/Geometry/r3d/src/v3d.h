/*
 *
 *		v3d.h
 *
 *		Routines for voxelizing (analytic volume sampling) 
 *		polyhedra to a regular Cartesian grid using r3d. 
 *		
 *		Devon Powell
 *		15 October 2015
 *
 *		Copyright (c) 2015, The Board of Trustees of the Leland Stanford Junior University, 
 *		through SLAC National Accelerator Laboratory (subject to receipt of any required approvals 
 *		from the U.S. Dept. of Energy). All rights reserved. Redistribution and use in source and 
 *		binary forms, with or without modification, are permitted provided that the following 
 *		conditions are met: 
 *		(1) Redistributions of source code must retain the above copyright notice, 
 *		this list of conditions and the following disclaimer. 
 *		(2) Redistributions in binary form must reproduce the above copyright notice, 
 *		this list of conditions and the following disclaimer in the documentation and/or other 
 *		materials provided with the distribution. 
 *		(3) Neither the name of the Leland Stanford Junior University, SLAC National Accelerator 
 *		Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse 
 *		or promote products derived from this software without specific prior written permission. 
 *		
 *		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
 *		OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
 *		MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 *		COPYRIGHT OWNER, THE UNITED STATES GOVERNMENT, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 *		INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 *		LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 *		BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 *		STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
 *		USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *		
 *		You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to 
 *		the features, functionality or performance of the source code ("Enhancements") to anyone; 
 *		however, if you choose to make your Enhancements available either publicly, or directly to 
 *		SLAC National Accelerator Laboratory, without imposing a separate written license agreement 
 *		for such Enhancements, then you hereby grant the following license: a non-exclusive, 
 *		royalty-free perpetual license to install, use, modify, prepare derivative works, 
 *		incorporate into other computer software, distribute, and sublicense such Enhancements or 
 *		derivative works thereof, in binary and source code form.
 *		
 */

#ifndef _V3D_H_
#define _V3D_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "r3d.h"

/**
 * \file v3d.h
 * \author Devon Powell
 * \date 15 October 2015
 * \brief Interface for r3d voxelization routines
 */

/**
 * \brief Voxelize a polyhedron to the destination grid.
 *
 * \param [in] poly 
 * The polyhedron to be voxelized.
 *
 * \param [in] ibox 
 * Minimum and maximum indices of the polyhedron, found with `r3d_get_ibox()`. These indices are
 * from a virtual grid starting at the origin. 
 *
 * \param [in, out] dest_grid 
 * The voxelization buffer. This grid is a row-major grid patch starting at `d*ibox[0]` and ending
 * at `d*ibox[1]. Must have size of at least 
 * `(ibox[1].i-ibox[0].i)*(ibox[1].j-ibox[0].j)*(ibox[1].k-ibox[0].k)*R3D_NUM_MOMENTS(polyorder)`.
 *
 * \param [in] d 
 * The cell spacing of the grid.
 *
 * \param [in] polyorder
 * Order of the polynomial density field to voxelize. 
 * 0 for constant (1 moment), 1 for linear (4 moments), 2 for quadratic (10 moments), etc.
 *
 */
void r3d_voxelize(r3d_poly* poly, r3d_dvec3 ibox[2], r3d_real* dest_grid, r3d_rvec3 d, r3d_int polyorder);

/**
 * \brief Get the minimal box of grid indices for a polyhedron, given a grid cell spacing,
 * also clamping it to a user-specified range while clipping the polyhedron to that range.
 *
 * \param [in] poly 
 * The polyhedron for which to calculate the index box and clip.
 *
 * \param [in, out] ibox 
 * Minimal range of grid indices covered by the polyhedron.
 *
 * \param [in, out] clampbox 
 * Range of grid indices to which to clamp and clip `ibox` and `poly`, respectively. 
 *
 * \param [in] d 
 * The cell spacing of the grid. The origin of the grid is assumed to lie at the origin in space.
 *
 */
void r3d_clamp_ibox(r3d_poly* poly, r3d_dvec3 ibox[2], r3d_dvec3 clampbox[2], r3d_rvec3 d);

/**
 * \brief Get the minimal box of grid indices for a polyhedron, given a grid cell spacing.
 *
 * \param [in] poly 
 * The polyhedron for which to calculate the index box.
 *
 * \param [out] ibox 
 * Minimal range of grid indices covered by the polyhedron.
 *
 * \param [in] d 
 * The cell spacing of the grid. The origin of the grid is assumed to lie at the origin in space.
 *
 */
void r3d_get_ibox(r3d_poly* poly, r3d_dvec3 ibox[2], r3d_rvec3 d);

#ifdef __cplusplus
}
#endif

#endif // _V3D_H_
