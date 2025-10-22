// file:    Euler.h
// author:  Christian Wieners, Antje Sydow
// $Header: /public/M++/src/Euler.h,v 1.9 2007-11-16 20:27:02 mueller Exp $

#ifndef _EULER_H_
#define _EULER_H_

#include "Newton.h"
#include "TimeSeries.h"

class Euler {
    NonlinearSolver& N;
    int verbose;
    int extrapolate;
    int ReferenceCalculation;
    int max_estep;    
 public:
    Euler (const char*, NonlinearSolver&);
    void operator () (Mesh&, TAssemble&, TimeSeries&, Vector&);
    void operator () (Mesh&, TAssemble&, Vector&);
};

#endif
