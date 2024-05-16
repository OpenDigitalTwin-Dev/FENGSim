// file: Constants.h
// author: Christian Wieners
// $Header: /public/M++/src/Constants.h,v 1.5 2009-05-19 16:35:00 maurer Exp $

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <cmath>
#include <cstddef>

const double infty = 1e100;
const double Pi = 4 *atan(1.0);
const double Eps = 1e-15;
const double GeometricTolerance = 1e-10;
const double TimeTolerance = 1e-6;
const double PlotTolerance = 1e-10;
//const size_t BufferSize = 128000;
//const size_t BufferSize = 64000;
const size_t BufferSize = 512000;
const double VeryLarge = 1e30;
const int MaxBroadcastSize = 1000000;
const int MaxPointTypes = 10;
const char Settings[] = "conf/m++conf";
const int MaxQuadraturePoints = 27;
const int MaxNodalPoints = 54;
const int MaxShapeFunctions = 54;

const double MU0 = 4 * Pi * 1e-7;
const double EPSILON0 = 8.854187817 * 1e-12;

#endif
