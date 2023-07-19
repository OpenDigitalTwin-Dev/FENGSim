// file: TimeSeries.h
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/TimeSeries.h,v 1.5 2008-03-05 10:09:35 mueller Exp $

#ifndef _TIMESERIES_H_
#define _TIMESERIES_H_

#include "Compiler.h"
#include "IO.h"

class TSeries {
 public:
    virtual ~TSeries() {};
    virtual string Name () = 0;
    virtual double FirstTStep () = 0;
    virtual double LastTStep () const = 0;
    virtual double NextTStep (double, int) = 0;
    virtual bool SpecialTimeStep (double) {return true; }
    virtual double Min () const { return 0; }
    virtual double Max () const { return infty; }
};

TSeries* GetTimeSeries ();
TSeries* GetTimeSeries (const vector<double>&);

class TimeSeries {
    TSeries* TS;
 public:
    TimeSeries () : TS(GetTimeSeries()) {}
    TimeSeries (const vector<double>& ts_vec) : TS(GetTimeSeries(ts_vec)) {}
    ~TimeSeries () { delete TS; }
    string Name ()                   { return TS->Name(); }
    double FirstTStep ()             { return TS->FirstTStep(); }
    double SecondTStep ()            { return TS->NextTStep(FirstTStep(),-1);}
    double NextTStep (double t)      { return TS->NextTStep(t,-1); }
    double NextTStep (double t,int s){ return TS->NextTStep(t,s); }
    double LastTStep () const        { return TS->LastTStep(); }
    bool SpecialTimeStep (double t)  { return TS->SpecialTimeStep(t); }
    double Min () const { return TS->Min(); }
    double Max () const { return TS->Max(); }
};

#endif
