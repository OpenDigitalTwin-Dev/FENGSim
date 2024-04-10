// file:    Newmark.h
// author:  Christian Wieners, Antje Sydow, Martin Sauter
// $Header: /public/M++/src/Newmark.h,v 1.5 2007-09-21 12:12:39 neuss Exp $

#ifndef _NEWMARK_H_
#define _NEWMARK_H_

#include "Newton.h"
#include "TimeSeries.h"

class Newmark {
    Newton& N;
    int verbose;
    int extrapolate;
    double beta;
    double gamma;
    int ReferenceCalculation;
    int quasistatic;
 public:
    Newmark (const char* conf, Newton& n);
    void operator () (Mesh& M, TAssemble& A, TimeSeries& TS, Vector& u);
    void operator () (Mesh& M, TAssemble& A, TimeSeries& TS, Vector& u, Vector& v);
};
    
#endif
