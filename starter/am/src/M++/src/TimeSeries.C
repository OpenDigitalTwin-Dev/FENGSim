// file: TimeSeries.C
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/TimeSeries.C,v 1.20 2009-01-07 16:55:46 mueller Exp $

#include "TimeSeries.h"
#include "Parallel.h"

#include <list>

// ----------------------------------------------------
//    UniformTimeSeries
// ----------------------------------------------------
class UniformTimeSeries : public TSeries {
protected:
    double t0;
    double T;
    double t;
    double dt;
    double dt_min;
    double dt_max;
public:
    UniformTimeSeries () {
	t0 = 0;         ReadConfig(Settings,"t0",t0);
	T  = 1;         ReadConfig(Settings,"T ",T);
	dt = T;         ReadConfig(Settings,"dt",dt); 
	dt_min = dt;    ReadConfig(Settings,"dt_min",dt_min); 
    }
    UniformTimeSeries (const vector<double>& ts_vec) {
	t0 = 0;
	t = t0;
	T = ts_vec[1];
	dt = ts_vec[0];
	dt_min = ts_vec[2];
	dt_max = ts_vec[3];
//	dt = dt_max;
    }
    string Name () { return "UniformTimeSeries"; };
    double FirstTStep () { return t = t0; }
    double LastTStep () const { return T; }
    double NextTStep (double t1, int steps = 0) { 
	while (t1 >= t-Eps) t += dt; 
	if (t + TimeTolerance > LastTStep())
	    return LastTStep();
	return t;
    }
    double Min () const { return dt_min; }
    double Max () const { return dt_max; }
};

// ----------------------------------------------------
//    TimeSeriesOfFile
// ----------------------------------------------------
class TimeSeriesOfFile : virtual public TSeries {
 protected:
    list<double> ts;
    list<double>::const_iterator i;
 public:
    TimeSeriesOfFile () {
	string TimePath = "./";     ReadConfig(Settings,"TimePath",TimePath); 
	string TimeName = "timeseries";
 	ReadConfig(Settings,"TimeSeriesFile",TimeName);
        double T  = 1;         ReadConfig(Settings,"T ",T);            
	if (PPM->master()) {
	    TimeName = TimeName + ".time";	
	    if (!FileExists(TimeName)) 
		TimeName = TimePath + "conf/time/" + TimeName;
	    if (!FileExists(TimeName)) 
		Exit("No TimeSeriesFile " + TimeName);
	    M_ifstream file(TimeName.c_str());
	    const int len = 128;
	    char L[len];
	    file.getline(L,len);
	    double t;
	    while (sscanf(L,"%lf",&t) == 1) {
		ts.push_back(t);
		file.getline(L,len);
	    }
            ts.push_back(T);
	    PPM->BroadcastInt(ts.size());
	    for (list<double>::const_iterator d=ts.begin(); d!=ts.end(); ++d)
		PPM->BroadcastDouble(*d);
	} else {
	    int n = PPM->BroadcastInt();    
	    for (int i=0; i<n; ++i) 
		ts.push_back(PPM->BroadcastDouble());    
	}
	i = ts.begin();
    }
    virtual ~TimeSeriesOfFile () {};
    virtual string Name () { return "TimeSeriesOfFile"; };
    virtual double FirstTStep () { 
	i = ts.begin();
	return *i; 
    }
    virtual double LastTStep () const { return *(--ts.end()); }
    virtual double NextTStep (double t, int steps = 0) { 
	list<double>::const_iterator j = i;
	++j;
	if (*j < *i) { return *(++i); }
	if (t + TimeTolerance > LastTStep()) { return LastTStep(); }
	while (t >= *i) ++i;
	return *i;
    }
};

// ----------------------------------------------------
//    VariableTimeSeries
// ----------------------------------------------------
class VariableTimeSeries : virtual public TSeries {
 protected:
    double t0;
    double T;
    double t;
    double t_old;
    double dt;
    double dt_min;
    double dt_max;
    int OptSteps;
 public:
    VariableTimeSeries () {
	t0 = 0;         ReadConfig(Settings,"t0",t0);
	T  = 1;         ReadConfig(Settings,"T ",T);
	dt = T;         ReadConfig(Settings,"dt",dt);        if (dt>T) dt = T;
	t  = t0;
	dt_min = 0;     ReadConfig(Settings,"dt_min",dt_min); 
	dt_max = infty; ReadConfig(Settings,"dt_max",dt_max); if (dt>dt_max) dt = dt_max;
	OptSteps = 3;   ReadConfig(Settings,"OptSteps",OptSteps); 
    }
    VariableTimeSeries (const vector<double>& ts_vec) {
	t0 = 0;
	t = t0;
	dt = ts_vec[0];
	T = ts_vec[1];
	dt_min = ts_vec[2];
	dt_max = ts_vec[3];
//	dt = dt_max;
	OptSteps = 3;   ReadConfig(Settings,"OptSteps",OptSteps); 
    }
    virtual ~VariableTimeSeries() {};
    virtual string Name () { return "VariableTimeSeries"; };
    virtual double FirstTStep () { return t = t0; }
    virtual double LastTStep () const { return T; }
    virtual double NextTStep (double t1, int steps) { 
	if (t1 == t0) {
	    t_old = t0;
	    return dt;
	}
	if (steps == -1) steps = OptSteps; 
	
	dt  = t1 - t_old;
	if (steps > 0) dt *= OptSteps / double(steps);
	if (steps == 0) dt *= 2;
	if (dt < dt_min) dt = dt_min;
	if (dt > dt_max) dt = dt_max;
	
	t_old = t1;
	t     = t1 + dt;
	
	if (t + TimeTolerance > LastTStep()) return LastTStep();
	return t; 
    }
    double Min () const { return dt_min; }
    double Max () const { return dt_max; }
};

// ----------------------------------------------------
//    GeneralTimeSeries
// ----------------------------------------------------
class GeneralTimeSeries : public VariableTimeSeries,
			  public TimeSeriesOfFile {
private:
    double dt_from_last_successful_non_list_step;
    bool last_step_was_list_step;
public:
    GeneralTimeSeries () : last_step_was_list_step(false) {}
    string Name () { return "GeneralTimeSeries"; }
    double FirstTStep ()     { return VariableTimeSeries::FirstTStep(); }
    double LastTStep () const{ return VariableTimeSeries::LastTStep(); }
    double NextTStep (double t1, int steps) { 
	double list_step = TimeSeriesOfFile::NextTStep(t1);
	if (t1 == t0) {
	    t_old = t0;
	    dt = max(dt,dt_min);
	    dt = min(dt,dt_max);
	    if (dt + TimeTolerance > list_step) return list_step;
	    else return dt;
	}
	if (steps == -1) steps = OptSteps; 
	
	dt  = t1 - t_old;
	dt *= (1.5 - steps/(2.0*OptSteps));
	if (dt < dt_min) dt = dt_min;
	if (dt > dt_max) dt = dt_max;

        if (last_step_was_list_step) {
            dt = max(dt,dt_from_last_successful_non_list_step);
        }
	t_old = t1;
	t     = t1 + dt;
	
	if (t + TimeTolerance > list_step){
	    if (t_old < list_step) {
                last_step_was_list_step = true;
                return list_step;
            }
	}
	if (t + TimeTolerance > T) {
	    return T;
	}
	else {
            dt_from_last_successful_non_list_step = dt;
            last_step_was_list_step = false;
	    return t; 
	}
    }
    bool SpecialTimeStep (double t1) {
	list<double>::const_iterator j = i;
	if (t1 == *j) return true; 
	return false;
    }
};

// ====================================================
TSeries* GetTimeSeries () {
    string   name = "uniform";      ReadConfig(Settings,"TimeSeries",name);
    if (name == "uniform")     return new UniformTimeSeries();
    else if (name == "file")        return new TimeSeriesOfFile();
    else if (name == "variable")    return new VariableTimeSeries();
    else if (name == "general")     return new GeneralTimeSeries();
    Exit(name + " TimeSeries not implemented");
}
// ====================================================
TSeries* GetTimeSeries(const vector<double>& ts_vec) {
    string   name = "uniform";   ReadConfig(Settings,"TimeSeries",name);
    if (name == "abaqus")        return new VariableTimeSeries(ts_vec);
    if (name == "uniformabaqus") return new UniformTimeSeries(ts_vec);
    if (name == "uniform")       return new UniformTimeSeries();
    if (name == "file")          return new TimeSeriesOfFile();
    if (name == "variable")      return new VariableTimeSeries();
    Exit(name + "TimeSeries not implemented");
}

