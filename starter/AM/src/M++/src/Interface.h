// file: Interface.h
// author: Christian Wieners
// $Header: /public/M++/src/Interface.h,v 1.4 2009-04-01 12:32:21 wieners Exp $

#ifndef _INTERFACE_H_
#define _INTERFACE_H_

#include "Point.h"

class Vector;
class Matrix;
class Jacobi;
void DirichletConsistent (Vector&);
void Collect (Vector&);
void Accumulate (Vector&);
void Average (Vector&);
void MakeAdditive (Vector&);
void Accumulate (Matrix&);
void Accumulate (Jacobi&);

void Consistent2Additive (Vector&);
void SetQuasiperiodic (const Point&); 

#endif
