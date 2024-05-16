// file:    Euler.C
// author:  Christian Wieners, Antje Sydow
// $Header: /public/M++/src/Euler.C,v 1.5 2007-11-16 20:27:02 mueller Exp $

#include "Euler.h"

Euler::Euler (const char* conf, NonlinearSolver& n) : 
    N(n), verbose(1), extrapolate(1), max_estep(100000),
	ReferenceCalculation(0) { 
    ReadConfig(conf,"EulerVerbose",verbose);
    ReadConfig(conf,"EulerExtrapolate",extrapolate);
    ReadConfig(conf,"ReferenceCalculation",ReferenceCalculation);
    ReadConfig(conf,"EulerSteps",max_estep);    
    if (ReferenceCalculation) NoDate();
}

void Euler::operator () (Mesh& M, TAssemble& A, TimeSeries& TS, Vector& u) {
    Date Start;
    double t = TS.FirstTStep();
    double t_old = t;
    double t_new = TS.SecondTStep();
    double T = TS.LastTStep();
    double dt = t_new - t;
    Vector u_old(u);
    Vector u_new(u);
    A.Initialize(t,u);
    int initstep = A.Step();

    for (int estep=0; estep<max_estep;) {
        if (t > T) { break; }
	Vout(0) << "\n" << "Euler step " << A.Step() + 1 << " from " 
		<< t << " to " << t_new << " (" << dt << ") "<< Date();
        if (extrapolate && (dt*(t-t_old))>0) {
            u_new -= (dt/(t-t_old)) * u_old;
            u_new += (dt/(t-t_old)) * u;
        }
        A.InitTimeStep(t_old,t,t_new,u_old,u,u_new);
        N(A,u_new);

        if (N.converged()) {
            ++estep;            
            A.FinishTimeStep(t_new,u_new,TS.SpecialTimeStep(t_new));
            u_old = u;
            u = u_new;
            if (t_new == T) {
                A.Finalize(t_new,u_new);
                Vout(0) << "Program successful: end of time series reached"
                        << endl;
                break;
            }
            if (A.BlowUp(u_new)) {
                Vout(0) << "Program successful: blowup limit reached"
                        << endl;
                break;
            }
            t_old = t;
            t = t_new;
            t_new = TS.NextTStep(t,N.Iter());
            dt = t_new - t;
        } 
        else {
            dt *= 0.5;
            if (dt < TS.Min()) {
                vout(0) << "Program stopped: dt < dt_min ("
                        << TS.Min() << ")!" << endl;
                break;
            }
            t_new -= dt;
            u_new = u;
            vout(0) << "Euler: repeat time step \n";
        }
    }
    if (ReferenceCalculation) {	
        Vout(0) << "Euler: " << A.Step()-initstep << " steps" << endl;
    }
    else {
        Vout(0) << "Euler: " << A.Step()-initstep << " steps in " 
                << Date()-Start << endl;
    }
}

void Euler::operator () (Mesh& M, TAssemble& A, Vector& u) {
    TimeSeries TS;
    (*this)(M,A,TS,u);
}

