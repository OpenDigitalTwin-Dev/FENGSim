#include "Newmark.h"

Newmark::Newmark (const char* conf, Newton& n) 
	: N(n), verbose(1), extrapolate(1), beta(0.25), gamma(0.5), 
	ReferenceCalculation(0), quasistatic(0) { 
    ReadConfig(conf,"NewmarkVerbose",verbose);
    ReadConfig(conf,"NewmarkExtrapolate",extrapolate);
    ReadConfig(conf,"NewmarkBeta",beta);
    ReadConfig(conf,"NewmarkGamma",gamma);
    ReadConfig(conf,"ReferenceCalculation",ReferenceCalculation);
    ReadConfig(Settings,"quasistatic",quasistatic);
}

void Newmark::operator () (Mesh& M, TAssemble& A, TimeSeries& TS, Vector& u) {
    Vector v(u);
    v = 0;
    (*this)(M,A,TS,u,v);
}

void Newmark::operator () (Mesh& M, TAssemble& A, TimeSeries& TS, 
		      Vector& u, Vector& v) {
    Date Start;
    A.SetNewmarkParameters(gamma,beta);
    Vector a(v);
    a = 0;
    Vector u_old(u);
    Vector v_old(v);
    Vector a_old(a);
    double t_old = TS.FirstTStep();
    double t     = TS.SecondTStep();
    double T     = TS.LastTStep();
    double dt    = t - t_old;
	
    A.Initialize(t,u,v,a);
    int initstep = A.Step();
	
    if (initstep == 0) {
        if (!quasistatic) {
            vout(0) << "\n" << "Newmark calculating initial values \n";
		
            A.InitTimeStep(t_old,t_old,t_old,u_old,u_old,u,v,a);
/*            N(A,a);

            u_old  = u;
            v_old  = A.V_old();
            a_old  = A.A_old();
            v     += dt * A.A_old();
            u     += (0.5*dt*dt) * A.A_old();
*/		
            vout(2) << "in Newmark: u " << u << endl;
            vout(2)	<< "in Newmark: v " << v << endl;
            vout(2)	<< "in Newmark: a " << a << endl;
        } 
    }
    while (t <= T) {
        if (ReferenceCalculation) {
            Vout(0) << "\n" << "Newmark step " << A.Step() + 1 << " from " 
                    << t_old << " to " << t << " (" << dt << ") "<< endl;
        }
        else { 
            Vout(0) << "\n" << "Newmark step " << A.Step() + 1 << " from " 
                    << t_old << " to " << t << " (" << dt << ") "<< Date();
        }	    

        A.InitTimeStep(t_old,t_old,t,u_old,u_old,u,v_old,a_old);
        vout(3) << "in Newmark: Ass.u_old = " << A.U_old() << endl;
        vout(3) << "in Newmark: Ass.v_old = " << A.V_old() << endl;
        vout(3) << "in Newmark: Ass.a_old = " << A.A_old() << endl;
        N(A,u);
	    
        if (N.converged()) {
            // Newmark Update;
	    Vout(6) << "Newton converged"<<endl;
	    v=u; 
            v-=u_old; 
            v*=gamma/(dt*beta); 
            v+=(1.0-gamma/beta)*v_old;
            v+=(dt*(1.0-gamma/(2*beta)))*a_old;
            a=u; 
            a-=u_old; 
            a*=(1.0/(dt*dt*beta));
            a-=(1.0/(dt*beta))*v_old;
            a-=(1.0/(2*beta)-1)*a_old;

            vout(2) << "in Newmark: u " << u << endl;
            vout(2)	<< "in Newmark: v " << v << endl;
            vout(2)	<< "in Newmark: a " << a << endl;
		
            A.FinishTimeStep(t,u,v,TS.SpecialTimeStep(t));
            if (t == T){
                A.Finalize(t,u,v,a);
                Vout(0) << "Program successful: end of time series reached"
                        << endl;
                break;
            }
            t_old = t;
            u_old = u;
            v_old = v;
            a_old = a;
            t = TS.NextTStep(t_old,N.Iter());
            dt = t - t_old;
		
            if (extrapolate){
                u += dt * v;
                u += (0.5*dt*dt) * a;
                v += dt * a;
            }
        } 
        else { 
            dt *= 0.5; 
            if (dt < TS.Min()) {
                vout(0) << "Program stopped: dt < dt_min ("
                        << TS.Min() <<")!" << endl;
                break;
            }
            t -= dt;
            u = u_old;
            v = v_old;
            a = a_old;	
		
            if (extrapolate){
                u += dt * v;
                u += (0.5*dt*dt) * a;
                v += dt * a;
            }
            vout(0) << "Newmark: repeat time step \n";
        }
    }
    if (ReferenceCalculation) {	
        Vout(0) << "Newmark: " << A.Step()-initstep << " steps" << endl;
    }
    else { 
        Vout(0) << "Newmark: " << A.Step()-initstep << " steps in " 
                << Date()-Start << endl;
    }
}
