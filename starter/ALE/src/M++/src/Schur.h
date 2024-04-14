// file:   Schur.h
// author: Daniel Maurer
// $Header: /public/M++/src/Schur.h,v 1.9 2009-09-16 13:36:45 maurer Exp $

#ifndef _SCHUR_H_
#define _SCHUR_H_

#include "Small.h"
#include "Preconditioner.h"

#include <cmath>
#include <algorithm>
#include <valarray>

class DDProblem;
void MultiplySubtract (const DDProblem&, 
		       const valarray<Scalar>&, valarray<Scalar>&);

Preconditioner* GetSchurPC();

#endif

