// file:   m++.h
// author: Christian Wieners
// $Header: /public/M++/src/m++.h,v 1.5 2007-05-11 13:27:22 sydow Exp $

#ifndef _MPLUSPLUS_H_
#define _MPLUSPLUS_H_

#include "Time.h"
#include "IO.h"
#include "Plot.h"
#include "Algebra.h"
#include "Element.h"
#include "FaceElement.h"
#include "Newton.h"
#include "ESolver.h"
#include "TimeSeries.h"
#include "Euler.h"
#include "Newmark.h"
#include "Small.h"

class DPO {
public:
    DPO (int* Argv, char ***Argc) {
		PPM = new ParallelProgrammingModel(Argv,Argc);
		logging = new Logging;
		ReadConfig(Settings,"DebugLevel",DebugLevel);
		ReadConfig(Settings,"TimeLevel",TimeLevel);
    }
    ~DPO () {
		delete logging;
		delete PPM;
    }
};

#endif
