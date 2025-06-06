// Copyright (c) 2005-2008 ASCLEPIOS Project, INRIA Sophia-Antipolis (France)
// All rights reserved.
//
// This file is part of the ImageIO Library, and as been adapted for CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/CGAL_ImageIO/include/CGAL/ImageIO/reech4x4.h $
// $Id: reech4x4.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later
//
//
// Author(s)     :  ASCLEPIOS Project (INRIA Sophia-Antipolis), Laurent Rineau

/*************************************************************************
 * reech4x4.h -
 *
 * $Id: reech4x4.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
 *
 * Copyright©INRIA 1999
 *
 * AUTHOR:
 * Gregoire Malandain (greg@sophia.inria.fr)
 *
 * CREATION DATE:
 *
 *
 * ADDITIONS, CHANGES
 *
 *
 *
 *
 */


/* CAUTION
   DO NOT EDIT THIS FILE,
   UNLESS YOU HAVE A VERY GOOD REASON
 */

#ifndef _reech4x4_h_
#define _reech4x4_h_









extern void Reech3DTriLin4x4_u8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech3DTriLin4x4gb_u8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech3DNearest4x4_u8 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );
extern void Reech2DTriLin4x4_u8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech2DTriLin4x4gb_u8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech2DNearest4x4_u8 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );






extern void Reech3DTriLin4x4_s8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech3DTriLin4x4gb_s8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech3DNearest4x4_s8 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );
extern void Reech2DTriLin4x4_s8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech2DTriLin4x4gb_s8 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech2DNearest4x4_s8 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );






extern void Reech3DTriLin4x4_u16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech3DTriLin4x4gb_u16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech3DNearest4x4_u16 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );
extern void Reech2DTriLin4x4_u16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech2DTriLin4x4gb_u16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech2DNearest4x4_u16 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );






extern void Reech3DTriLin4x4_s16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech3DTriLin4x4gb_s16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech3DNearest4x4_s16 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );
extern void Reech2DTriLin4x4_s16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech2DTriLin4x4gb_s16 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech2DNearest4x4_s16 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );






extern void Reech3DTriLin4x4_r32 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech3DTriLin4x4gb_r32 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech3DNearest4x4_r32 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );
extern void Reech2DTriLin4x4_r32 ( void* theBuf, /* buffer to be resampled */
                             int *theDim,  /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat   /* transformation matrix */
                             );
extern void Reech2DTriLin4x4gb_r32 ( void* theBuf, /* buffer to be resampled */
                             int *theDim, /* dimensions of this buffer */
                             void* resBuf, /* result buffer */
                             int *resDim,  /* dimensions of this buffer */
                             double* mat,   /* transformation matrix */
                             float gain,
                             float bias );
extern void Reech2DNearest4x4_r32 ( void* theBuf, /* buffer to be resampled */
                              int *theDim,  /* dimensions of this buffer */
                              void* resBuf, /* result buffer */
                              int *resDim,  /* dimensions of this buffer */
                              double* mat   /* transformation matrix */
                              );







extern void Reech4x4_verbose ( );
extern void Reech4x4_noverbose ( );

#ifdef CGAL_HEADER_ONLY
#include <CGAL/ImageIO/reech4x4_impl.h>
#endif // CGAL_HEADER_ONLY


#endif
