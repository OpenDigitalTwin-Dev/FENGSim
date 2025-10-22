// file: ESolver.C
// author: Christian Wieners
// $Header: /public/M++/src/ESolver.C,v 1.12 2009-12-08 10:31:10 wieners Exp $

#include "ESolver.h"

void RandomVector (Vector& u, Operator& B, Operator& IA, Operator& P) {
    Vector v(u);
    v << rand;
    v *= (1.0 / RAND_MAX);
    v.ClearDirichletValues();
    Vector b(u);
    b = B * v;
    double s = sqrt(real(b * v));
    b /= s;
    v = IA * b;
    u = P * v;
}

class EigenSolver {
    protected:
    double eps;
    int maxstep;
    int verbose;
    Scalar Bilinear (Operator& A, const Vector& u, const Vector& v) const {
	Vector Au = A * u;    
	return Au * v;
    }
    void GramSchmidt (vector<Vector>& u, Operator& B, int R=0) const {
	R = (R==0) ? u.size() : R;
	for (int i=0; i<R; ++i) {
	    double n1 = sqrt(real(Bilinear(B,u[i],u[i])));
	    double n2 = n1;
	    for (int m=0; m<2; ++m) {
		for (int j=0;j<i;++j) {
		    Scalar s = Bilinear(B,u[j],u[i]);
		    u[i] -= s * u[j];
		}
		n2 =  sqrt(real(Bilinear(B,u[i],u[i])));
		if (n2 > 0.125 * n1) break;
	    }
	    u[i] *= (1/n2);
	    if(n2 < n1 * 1e-4) Exit("GramSchmidt failed");
	}
    }
    void GramSchmidt2 (vector<Vector>& u, vector<Vector>& bu, int R=0) const {
	R = (R==0) ? u.size() : R;
	for (int i=0; i<R; ++i) {
	    double n1 = sqrt(real(bu[i]*u[i]));
	    double n2 = n1;
	    for (int m=0; m<2; ++m) {
		for (int j=0;j<i;++j) {
		    Scalar s = bu[j]*u[i];
		    u[i] -= s * u[j];
                    bu[i] -= s * bu[j];
		}
		n2 = sqrt(real(bu[i]*u[i]));
		if (n2 > 0.125 * n1) break;
	    }
	    u[i] *= (1/n2);
            bu[i] *= (1/n2);
	    if(n2 < n1 * 1e-4) Exit("GramSchmidt failed");
	}
    }
    void RitzStep (vector<Vector>& u, DoubleVector& lambda,
		   Operator& A, Operator& B) {
	int R = u.size();
	HermitianMatrix a(R);
	HermitianMatrix b(R);
	HermitianMatrix e(R);
        vector<Vector> Au(R,u[0]), Bu(R,u[0]);
        for (int s=0; s<R; ++s) Bu[s] = B * u[s];
	GramSchmidt2(u,Bu,R);
        for (int s=0; s<R; ++s) Au[s] = A * u[s];
	for (int s=0; s<R; ++s)
	    for (int r=0; r<=s; ++r) a[s][r] = Au[r] * u[s];
	dout(3) << "a \n" << a;
	EVcomplexO(a,lambda,e);
	dout(4) << "e \n" << e;
	vector<Vector> w(u);
	dout(8) << "w " << w[0];
	dout(8) << "e " << e[0][0];
	for (int r=0; r<R; r++) {
	    u[r] = 0;
	    for (int s=0; s<R; ++s)
		u[r] += eval(e[s][r]) * w[s];
	}
    }
    void RitzDefect (Vector& u, Vector& r, double lambda, 
		     Operator& A, Operator& B) {
	Vector Bu(u);
	r = A * u;
	Bu = B * u;
	r -= lambda * Bu;
    }
public:
    EigenSolver () : eps(1e-5), maxstep(30), verbose(1) {
	ReadConfig(Settings,"EEpsilon",eps);
	ReadConfig(Settings,"EMaxStep",maxstep);
	ReadConfig(Settings,"EVerbose",verbose);
    }
    virtual ~EigenSolver () {}
    virtual void Init (vector<Vector>& U, Operator& B, Operator& IA) const {
	int R = U.size();
	for (int r=0; r<R; ++r) 
	    RandomVector(U[r],B,IA);
    }
    virtual void Init (vector<Vector>& U, Operator& B, 
	       Operator& IA, Operator& P) const {
	int R = U.size();
	for (int r=0; r<R; ++r) RandomVector(U[r],B,IA,P);
    }
    virtual void operator () (vector<Vector>&, DoubleVector&, 
			      Operator&, Operator&, Operator&, int t = 0) {
	Exit("Not implemented");
    }
    virtual void operator () (vector<Vector>&, DoubleVector&, 
			      Operator&, Operator&, Operator&, 
			      Operator&, int t = 0) {
	Exit("Not implemented");
    }
    virtual void operator () (vector<Vector>&, DoubleVector&, 
			      Operator&, Operator&, Operator&, 
			      Operator&, Operator&, int t = 0) {
	Exit("Not implemented");
    }
};

class RitzGalerkin : public EigenSolver {
 public:
    RitzGalerkin () {}
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA) {
	int R = lambda.size();
	Vector v(u[0]),Au(u[0]),Bu(u[0]);
	int k=0;
	int r_eps = 0;
	while (k < maxstep) {
	    RitzStep(u,lambda,A,B);
	    if (r_eps == R) break;
	    Vout(1) << "RitzGalerkin(" << k << ") = " << lambda;
	    ++k;
	    r_eps = 0;
	    for (int r=0; r<R; ++r) {
		RitzDefect(u[r],Au,lambda[r],A,B);
		double defect = norm(Au);
		Vout(3) << " def[" << r << "] = " << defect << endl;
		if (defect < eps) ++r_eps;
		else u[r] -= IA * Au;
	    }
	}
	Vout(0) << "RitzGalerkin(" << k << ") = " << lambda;
    }
};

class LOBPCG2 : public EigenSolver {
 public:
    LOBPCG2 () {}
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, int temp) {
	int R = lambda.size();
	Vector v(u[0]),Au(u[0]),Bu(u[0]);
	vector<Vector> XH(2*R,u[0]);
	int k=0;
	RitzStep(u,lambda,A,B);
	while (k<maxstep) {
	    Vout(1) << "LOBPCG2(" << k << ") = " << lambda;
	    ++k;
	    int r_eps = 0;
	    for (int r=0; r<R; ++r) {
		RitzDefect(u[r],Au,lambda[r],A,B);
		double defect = norm(Au);
		Vout(3) << " def[" << r << "] = " << defect << endl;
		if (defect < eps) {
		    XH[r] = u[r];
		    RandomVector(XH[r+R],B,IA);
		    ++r_eps;
		} else {
		    XH[r] = u[r];
		    XH[r+R] = IA * Au;
		}
	    }
	    DoubleVector Lambda(2*R);
	    RitzStep(XH,Lambda,A,B);
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	    for (int r=0; r<R; ++r) u[r] = XH[r];
	    if (r_eps == R) break;
	}
	Vout(0) << "LOBPCG2(" << k << ") = " << lambda;
    }
};

class LOBPCG3 : public EigenSolver {
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B, Operator& IA, 
			 int R, int rR) {
	HermitianMatrix a(R),b(R),e(R);
        vector<Vector> Au(R,u[0]), Bu(R,u[0]);
        for (int s=0; s<R; ++s) Bu[s] = B * u[s];
        for (int s=0; s<R; ++s) Au[s] = A * u[s];
	for (int s=0; s<R; ++s)
	    for (int r=0; r<=s; ++r) {
		a[s][r] = Au[r] * u[s];
		b[s][r] = Bu[r] * u[s];
	    }
	dout(3) << "a \n" << a << "b \n" << b;
	EVcomplex(a,b,lambda,e);
	dout(4) << "e \n" << e;
	vector<Vector> w(u);
	dout(8) << "w " << w[0];
	dout(8) << "e " << e[0][0];
        for (int r=0; r<rR; r++) {
	    u[r+2*rR] = 0;
	    for (int s=rR; s<R; ++s)
		u[r+2*rR] += eval(e[s][r]) * w[s];
        }
        for (int r=0; r<rR; r++) {
	    u[r] = u[r+2*rR];
	    for (int s=0; s<rR; ++s)
		u[r] += eval(e[s][r]) * w[s];
        }
    }
 public:
    LOBPCG3 () {}
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, int fev = 0) {
	int R = lambda.size();
	int dim = 2*R;
	Vector res(u[0]);
	vector<Vector> XH(3*R,u[0]);
	RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) XH[r]=u[r];
	int k=0;
	while (k<maxstep) {
	    Vout(1) << "LOBPCG3(" << k << ") = " << lambda;
	    ++k;
	    int r_eps = 0;
	    for (int r=0; r<R; ++r) {
		RitzDefect(XH[r],res,lambda[r],A,B);
		double defect = norm(res);
		Vout(3) << " def[" << r << "] = " << defect << endl;
		if (defect < eps) {
		    RandomVector(XH[r+R],B,IA);
		    RandomVector(XH[r+2*R],B,IA);
		    ++r_eps;
		} else XH[r+R] = IA * res;
	    }
            DoubleVector Lambda(dim);
	    RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R);
	    dim = 3*R;
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	    if (r_eps == R) break;
	}
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG3(" << k << ") = " << lambda;
    }
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA,
                      Operator& P, int fev = 0) {
	int R = lambda.size();
	int dim = 2*R;
	Vector res(u[0]);
	vector<Vector> XH(3*R,u[0]);
	RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) XH[r]=u[r];
	int k=0;
	while (k<maxstep) {
	    Vout(1) << "LOBPCG3P(" << k << ") = " << lambda;
	    ++k;
	    int r_eps = 0;
	    for (int r=0; r<R; ++r) {
		RitzDefect(XH[r],res,lambda[r],A,B);
		double defect = norm(res);
		Vout(3) << " def[" << r << "] = " << defect << endl;
		if (defect < eps) {
		    RandomVector(XH[r+R],B,IA,P);
		    RandomVector(XH[r+2*R],B,IA,P);
		    ++r_eps;
		} else {
                    Vector tmp(res);
                    tmp = IA * res;
                    XH[r+R] = P * tmp;
                }
	    }
            DoubleVector Lambda(dim);
	    RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R);
	    dim = 3*R;
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	    if (r_eps == R) break;
	}
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG3(" << k << ") = " << lambda;
    }
};

class LOBPCG4 : public EigenSolver {
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B, Operator& IA, 
			 int R, int rR,
                         int convf[],HermitianMatrix& a,HermitianMatrix& b) {
	HermitianMatrix e(R);
        vector<Vector> Au(R,u[0]), Bu(R,u[0]);
        for (int s=0; s<R; ++s) {
           Au[s] = A * u[s];
           Bu[s] = B * u[s];
        }
	for (int s=0; s<R; ++s)
	    for (int r=0; r<=s; ++r)
               if ((convf[r%rR]!=0)||(convf[s%rR]!=0)) {
		a[s][r] = Au[r] * u[s];
		b[s][r] = Bu[r] * u[s];
	    }
	dout(3) << "a \n" << a << "b \n" << b;
	EVcomplex(a,b,lambda,e);
	dout(4) << "e \n" << e;
	vector<Vector> w(u);
	dout(8) << "w " << w[0];
	dout(8) << "e " << e[0][0];
        for (int r=0; r<rR; r++)
           if (convf[r]==2) {
	    u[r+2*rR] = 0;
	    for (int s=rR; s<R; ++s)
		u[r+2*rR] += eval(e[s][r]) * w[s];
            u[r] = u[r+2*rR];
	    for (int s=0; s<rR; ++s)
		u[r] += eval(e[s][r]) * w[s];
        }
    }
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B, Operator& IA, 
			 Operator& P,int R, int rR,int convf[],
                         HermitianMatrix& a,HermitianMatrix& b,int p=0) {
	HermitianMatrix e(R);
        Vector tmp(u[0]);
        vector<Vector> Au(R,tmp), Bu(R,tmp);
        for (int s=0; s<R; ++s) {
           Au[s] = A * u[s];
           Bu[s] = B * u[s];
        }
	for (int s=0; s<R; ++s)
	    for (int r=0; r<=s; ++r)
               if ((convf[r%rR]!=0)||(convf[s%rR]!=0)) {
		a[s][r] = Au[r] * u[s];
		b[s][r] = Bu[r] * u[s];
	    }
	dout(3) << "a \n" << a << "b \n" << b;
	EVcomplex(a,b,lambda,e);
	dout(4) << "e \n" << e;
	vector<Vector> w(u);
	dout(8) << "w " << w[0];
	dout(8) << "e " << e[0][0];
        for (int r=0; r<rR; r++)
           if (convf[r]==2) {
	    tmp = 0;
	    for (int s=rR; s<R; ++s)
		tmp += eval(e[s][r]) * w[s];
            u[r+2*rR] = p ? P * tmp : tmp;
	    for (int s=0; s<rR; ++s)
		tmp += eval(e[s][r]) * w[s];
            u[r] =  p ? P * tmp : tmp;
        };
    }
 public:
    LOBPCG4 () {}
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]);
	vector<Vector> XH(3*R,u[0]);
        int convf[R];
        HermitianMatrix da(3*R),db(3*R),dfa(2*R),dfb(2*R);

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) { 
           XH[r]=u[r];
           convf[r]=2;
        };
	int k=0;
	while (k++<maxstep) {
            int r_eps = 0;
	    Vout(1) << "LOBPCG4(" << k << ") = " << lambda;
	    for (int r=0; r<R; ++r) {
                if (convf[r]==2) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
		   double defect = norm(res);
                   if (r == fev-1) {
		       Vout(2) << " def[" << r << "] = " << defect << endl; }
                   else 
		       Vout(3) << " def[" << r << "] = " << defect << endl;
                   if (defect < eps) {
                      RandomVector(XH[r+R],B,IA);
                      RandomVector(XH[r+2*R],B,IA);
                      convf[r]=1;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                   };
                } else convf[r] = (convf[r]==3) ? 1 : 0;
		if (convf[r]==2) XH[r+R] = IA * res;
		else
                     if (r < fev) {
                        ++r_eps;
	                if (r_eps >= fev) goto ESend;
                     }
	    }
            DoubleVector Lambda(dim);
            if (dim==2*R) {
               RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf,dfa,dfb);
               dim = 3*R;
	       for (int r=0; r<R; ++r) convf[r] = (convf[r]==1) ? 3 : 2;
            } else RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf,da,db);
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG4(" << k << ") = " << lambda;
    }
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, 
		      Operator& P, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]);
	vector<Vector> XH(3*R,u[0]);
        int convf[R];
        HermitianMatrix da(3*R),db(3*R),dfa(2*R),dfb(2*R);

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) { 
           XH[r]=u[r];
           convf[r]=2;
        };
	int k=0;
	while (k++<maxstep) {
            int r_eps = 0;
	    Vout(1) << "LOBPCG4P(" << k << ") = " << lambda;
	    for (int r=0; r<R; ++r) {
                if (convf[r]==2) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
		   double defect = norm(res);
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if (defect < eps) {
                      RandomVector(XH[r+R],B,IA,P);
                      RandomVector(XH[r+2*R],B,IA,P);
                      convf[r]=1;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                   };
                } else convf[r] = (convf[r]==3) ? 1 : 0;
		if (convf[r]==2) {
                   Vector tmp(res);
                   tmp = IA * res;
                   XH[r+R] = P * tmp;
		} else
                     if (r < fev) {
                        ++r_eps;
	                if (r_eps >= fev) goto ESend;
                     }
	    }
            DoubleVector Lambda(dim);
            if (dim==2*R) {
               RitzStepLOBPCG(XH,Lambda,A,B,IA,P,dim,R,convf,dfa,dfb,0);
               dim = 3*R;
	       for (int r=0; r<R; ++r) convf[r] = (convf[r]==1) ? 3 : 2;
            } else RitzStepLOBPCG(XH,Lambda,A,B,IA,P,dim,R,convf,da,db,0);
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG4P(" << k << ") = " << lambda;
    }
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, 
		      Operator& P, Operator& IB, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]),tmp(res);
	vector<Vector> XH(3*R,u[0]);
        int convf[R];
        HermitianMatrix da(3*R),db(3*R),dfa(2*R),dfb(2*R);

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) { 
           XH[r]=u[r];
           convf[r]=2;
        };
	int k=0;
	while (k++<maxstep) {
            int r_eps = 0;
	    Vout(1) << "LOBPCG4P(" << k << ") = " << lambda;
	    for (int r=0; r<R; ++r) {
                if (convf[r]==2) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
                   tmp = IB * res;
		   double defect = sqrt(real(tmp * res));
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if (defect < eps) {
                      RandomVector(XH[r+R],B,IA,P);
                      RandomVector(XH[r+2*R],B,IA,P);
                      convf[r]=1;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                   };
                } else convf[r] = (convf[r]==3) ? 1 : 0;
		if (convf[r]==2) {
                   tmp = IA * res;
                   XH[r+R] = P * tmp;
                   if (verbose>3) {
                      double s = real(tmp * res);
                      tmp -= XH[r+R];
                      res = B * tmp;
                      s = sqrt(real(tmp*res)/s);
                      Vout(4) << " ProjPrec " << s << endl;
                   }
		} else
                     if (r < fev) {
                        ++r_eps;
	                if (r_eps >= fev) goto ESend;
                     }
	    }
            DoubleVector Lambda(dim);
            if (dim==2*R) {
               RitzStepLOBPCG(XH,Lambda,A,B,IA,P,dim,R,convf,dfa,dfb,0);
               dim = 3*R;
	       for (int r=0; r<R; ++r) convf[r] = (convf[r]==1) ? 3 : 2;
            } else RitzStepLOBPCG(XH,Lambda,A,B,IA,P,dim,R,convf,da,db,0);
	    for (int r=0; r<R; ++r) lambda[r] = Lambda[r];
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG4P(" << k << ") = " << lambda;
    }
};

class LOBPCG5 : public EigenSolver {
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B, Operator& IA, 
			 int R, int rR,int convf[]) {
	HermitianMatrix e(R),a(R),b(R);
        Vector tmp(u[0]);
        vector<Vector> Au(R,tmp), Bu(R,tmp);
        for (int s=0; s<R; ++s) {
           Au[s] = A * u[s];
           Bu[s] = B * u[s];
        }
	for (int s=0; s<R; ++s)
	    for (int r=0; r<=s; ++r) {
		a[s][r] = Au[r] * u[s];
		b[s][r] = Bu[r] * u[s];
	    }
	dout(3) << "a \n" << a << "b \n" << b;
	EVcomplex(a,b,lambda,e);
	dout(4) << "e \n" << e;
	vector<Vector> w(u);
	dout(8) << "w " << w[0];
	dout(8) << "e " << e[0][0];
        for (int r=0; r<rR; r++)
           if (convf[r]==1) {
	    tmp = 0;
	    for (int s=rR; s<R; ++s)
		tmp += eval(e[s][r]) * w[s];
            u[r+2*rR] = tmp;
	    for (int s=0; s<rR; ++s)
		tmp += eval(e[s][r]) * w[s];
            u[r] = tmp;
           } else {
            u[r] = 0;
	    for (int s=0; s<R; ++s)
		u[r] += eval(e[s][r]) * w[s];
           }
    }
 public:
    LOBPCG5 () {}
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA,
                      Operator& IB, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]),tmp(res);
	vector<Vector> XH(3*R,u[0]);
        int convf[R];

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) { 
           XH[r]=u[r];
           convf[r]=1;
        };
	int k=0;
        int r_eps = 0;
	while (k++<maxstep) {
	    Vout(1) << "LOBPCG5(" << k << ") = " << lambda;
	    for (int r=0; r<R; ++r)
                if (convf[r]==1) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
                   tmp = IB * res;
		   double defect = sqrt(real(tmp * res));
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if (defect < eps) {
                      convf[r]=0;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                      if (r < fev) {
                        ++r_eps;
	                if (r_eps >= fev) goto ESend;
                      }
                      RandomVector(XH[r+R],B,IA);
                      RandomVector(XH[r+2*R],B,IA);
                   } else
                       XH[r+R] = IA * res;
                }
            DoubleVector Lambda(dim);
            if (dim==2*R) {
               RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf);
               dim = 3*R;
            } else RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf);
	    for (int r=0; r<R; ++r)
               lambda[r] = Lambda[r];
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG5(" << k << ") = " << lambda;
    }
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, 
		      Operator& P, Operator& IB, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]),tmp(res);
	vector<Vector> XH(3*R,u[0]);
        int convf[R];

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) { 
           XH[r]=u[r];
           convf[r]=1;
        };
	int k=0;
        int r_eps = 0;
	while (k++<maxstep) {
	    Vout(1) << "LOBPCG5P(" << k << ") = " << lambda;
	    for (int r=0; r<R; ++r)
                if (convf[r]==1) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
                   tmp = IB * res;
		   double defect = sqrt(real(tmp * res));
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if (defect < eps) {
                      convf[r]=0;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                      if (r < fev) {
                        ++r_eps;
	                if (r_eps >= fev) goto ESend;
                      }
                      RandomVector(XH[r+R],B,IA,P);
                      RandomVector(XH[r+2*R],B,IA,P);
                   } else {
                       tmp = IA * res;
                       XH[r+R] = P * tmp;
                       if (verbose>3) {
                          double s = real(tmp * res);
                          tmp -= XH[r+R];
                          res = B * tmp;
                          s = sqrt(real(tmp*res)/s);
                          Vout(4) << " ProjPrec " << s << endl;
                       }
		     }
                }
            DoubleVector Lambda(dim);
            if (dim==2*R) {
               RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf);
               dim = 3*R;
            } else RitzStepLOBPCG(XH,Lambda,A,B,IA,dim,R,convf);
	    for (int r=0; r<R; ++r)
               lambda[r] = Lambda[r];
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG5P(" << k << ") = " << lambda;
    }
};

class LOBPCG6 : public EigenSolver {
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B,
			 int R, int rR, int conv) {
        int fVmap[R];
        int totV = R - (R/rR-1)*conv;
        Vector tmp(u[0]);
        vector<Vector> w(u);
        for (int s=0; s<rR; ++s) {
           w[s] = A * u[s];
           fVmap[s] = s;
        }
        int extfV = rR;
        for (int s=rR; s<R; ++s)
           if ((s-rR)%rR >= conv) {
              w[extfV] = A * u[s];
              fVmap[s] = extfV;
              extfV++;
           } else fVmap[s] = -1;
	HermitianMatrix e(totV),a(totV),b(totV);
	for (int s=0; s<R; ++s)
           if (fVmap[s]>=0)
	      for (int r=0; r<=s; ++r)
                 if (fVmap[r]>=0) a[fVmap[s]][fVmap[r]] = w[fVmap[r]] * u[s];
        for (int s=0; s<R; ++s)
           if (fVmap[s]>=0) w[fVmap[s]] = B * u[s];
	for (int s=0; s<R; ++s)
           if (fVmap[s]>=0)
	      for (int r=0; r<=s; ++r)
                 if (fVmap[r]>=0) b[fVmap[s]][fVmap[r]] = w[fVmap[r]] * u[s];
	dout(3) << "a \n" << a << "b \n" << b;
        DoubleVector Lambda(totV);
	EVcomplex(a,b,Lambda,e);
        for (int r=0; r<rR; r++) lambda[r] = Lambda[r];
	dout(4) << "e \n" << e;
	dout(8) << "e " << e[0][0];
        w=u;
        for (int r=0; r<rR; r++)
           if (r >= conv) {
	    tmp = 0;
	    for (int s=rR; s<R; ++s)
               if (fVmap[s]>=0) tmp += eval(e[fVmap[s]][r]) * w[s];
            u[r+2*rR] = tmp;
	    for (int s=0; s<rR; ++s)
	       tmp += eval(e[s][r]) * w[s];
            u[r] = tmp;
           } else {
            u[r] = 0;
	    for (int s=0; s<R; ++s)
		if (fVmap[s]>=0) u[r] += eval(e[fVmap[s]][r]) * w[s];
           }
    };
 public:
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, 
		      Operator& P, Operator& IB, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2*R;
	Vector res(u[0]),tmp(res);
	vector<Vector> XH(3*R,tmp);

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) XH[r]=u[r];
	int k = 0;
        int conv = 0;
	while (k++<maxstep) {
	    Vout(1) << "LOBPCG6P(" << k << ") = " << lambda;
	    for (int r=conv; r<R; ++r) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
                   tmp = IB * res;
		   double defect = sqrt(real(tmp * res));
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if ((defect < eps)&&(r==conv)) {
                      conv++;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                      if (conv == fev) goto ESend;
                   } else {
                       tmp = IA * res;
                       XH[r+R] = P * tmp;
                       if (verbose>3) {
                          double s = real(tmp * res);
                          tmp -= XH[r+R];
                          res = B * tmp;
                          s = sqrt(real(tmp*res)/s);
                          Vout(4) << " ProjPrec " << s << endl;
                       }
		     }
            }
            if (dim==2*R) {
               RitzStepLOBPCG(XH,lambda,A,B,dim,R,conv);
               dim = 3*R;
            } else RitzStepLOBPCG(XH,lambda,A,B,dim,R,conv);
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
	Vout(0) << "LOBPCG6P(" << k << ") = " << lambda;
    }
};

class LOBPCG7 : public EigenSolver {
    void RitzStepLOBPCG (vector<Vector>& u, DoubleVector& lambda,
			 Operator& A, Operator& B,
			 int dim, int R, int conv) {
        int bs = R-conv;
        int size = dim*bs;
        Vector tmp(u[0]);
        vector<Vector> w(size,tmp);
        for (int j=1; j<dim; ++j)
           for (int s=0; s<bs; ++s) u[s+conv+j*R] /= norm(u[s+conv+j*R]);
        for (int j=0; j<dim; ++j)
           for (int s=0; s<bs; ++s) w[s+bs*j] = A * u[s+conv+j*R];
	HermitianMatrix e(size),a(size),b(size);
        for (int j=0; j<dim; ++j)
           for (int s=0; s<bs; ++s)
              for (int i=0; i<dim; ++i)
                 for (int r=0; r<bs; ++r)
                    if (r+bs*i <= s+bs*j) a[s+bs*j][r+bs*i] = w[r+bs*i] * u[s+conv+j*R];
        for (int j=0; j<dim; ++j)
           for (int s=0; s<bs; ++s) w[s+bs*j] = B * u[s+conv+j*R];
        for (int j=0; j<dim; ++j)
           for (int s=0; s<bs; ++s)
              for (int i=0; i<dim; ++i)
                 for (int r=0; r<bs; ++r)
                    if (r+bs*i <= s+bs*j) b[s+bs*j][r+bs*i] = w[r+bs*i] * u[s+conv+j*R];
	dout(3) << "a \n" << a << "b \n" << b;
        DoubleVector Lambda(size);
	EVcomplex(a,b,Lambda,e);
	dout(4) << "e \n" << e;
	dout(8) << "e " << e[0][0];
        for (int j=0; j<dim; ++j)
           for (int s=0; s<bs; ++s) w[s+bs*j] = u[s+conv+j*R];
        for (int r=0; r<bs; r++) {
	    tmp = 0;
            lambda[r+conv] = Lambda[r];
            for (int j=1; j<dim; ++j)
	       for (int s=0; s<bs; ++s)
                  tmp += eval(e[s+bs*j][r]) * w[s+bs*j];
            u[r+conv+2*R] = tmp;
	    for (int s=0; s<bs; ++s)
	       tmp += eval(e[s][r]) * w[s];
            u[r+conv] = tmp;
        }
    }
 public:
    void operator () (vector<Vector>& u, DoubleVector& lambda,
		      Operator& A, Operator& B, Operator& IA, 
		      Operator& P, Operator& IB, int fev = 0) {
	int R = lambda.size();
        if (fev == 0) fev = R;
	int dim = 2;
	Vector res(u[0]),tmp(res);
	vector<Vector> XH(3*R,tmp);

        RitzStep(u,lambda,A,B);
        for (int r=0; r<R; ++r) XH[r]=u[r];
	int k = 0;
        int conv = 0;
	while (k++ < maxstep) {
	    Vout(1) << "LOBPCG7P(" << k << ") = " << lambda;
	    for (int r=conv; r<R; ++r) {
		   RitzDefect(XH[r],res,lambda[r],A,B);
                   tmp = IB * res;
		   double defect = sqrt(real(tmp * res));
                   if (r == fev-1) { Vout(2) << " def[" 
					   << r << "] = " << defect << endl; }
                   else Vout(3) << " def[" << r << "] = " << defect << endl;
                   if ((defect < eps)&&(r==conv)) {
                      conv++;
                      Vout(2) << " the eigenpair " << r 
			      << " has converged" << endl;
                      if (conv == fev) goto ESend;
                      XH[r+R] = B * XH[r];
                      for (int s=conv; s<R; ++s) XH[s] -= (XH[r+R]*XH[s]) * XH[r];
                      for (int s=conv; s<R; ++s) XH[s+2*R] -= (XH[r+R]*XH[s+2*R]) * XH[r];
                   } else {
                       tmp = IA * res;
                       XH[r+R] = P * tmp;
                       if (verbose>3) {
                          double s = real(tmp * res);
                          tmp -= XH[r+R];
                          res = B * tmp;
                          s = sqrt(real(tmp*res)/s);
                          Vout(4) << " ProjPrec " << s << endl;
                       }
                       for (int s=0; s<conv; ++s) XH[r+R] -= (XH[s+R]*XH[r+R]) * XH[s];
		   }
            }
            if (dim==2) {
               RitzStepLOBPCG(XH,lambda,A,B,dim,R,conv);
               dim = 3;
            } else RitzStepLOBPCG(XH,lambda,A,B,dim,R,conv);
	}
ESend:
	for (int r=0; r<R; ++r) u[r] = XH[r];
        for (int r=1; r<R; ++r)
           if (lambda[r]<lambda[r-1]) {
              swap(lambda[r],lambda[r-1]);
              swap(u[r],u[r-1]);
              r-=(r>1)?2:0;
           }
	Vout(0) << "LOBPCG7P(" << k << ") = " << lambda;
    }
};


EigenSolver* GetESolver (const string& name) {
    if (name == "RitzGalerkin" ) return new RitzGalerkin;
    if (name == "LOBPCG2" ) return new LOBPCG2;
    if (name == "LOBPCG3" ) return new LOBPCG3;
    if (name == "LOBPCG4" ) return new LOBPCG4;
    if (name == "LOBPCG5" ) return new LOBPCG5;
    if (name == "LOBPCG6" ) return new LOBPCG6;
    if (name == "LOBPCG7" ) return new LOBPCG7;
    Exit("no esolver " + name);
}

void ESolver::Init (vector<Vector>& u, Operator& B, 
		    Operator& S, Operator& P) const {
    ES->Init(u,B,S,P); 
}

void ESolver::operator () (vector<Vector>& u, DoubleVector& lambda, 
			   Operator& A, Operator& B, 
			   Operator& S, Operator& P, int t) const {
    (*ES)(u,lambda,A,B,S,P,t);
}

void ESolver::operator () (vector<Vector>& u, DoubleVector& lambda, 
			   Operator& A, Operator& B, Operator& S,
			   Operator& P, Operator& IB, int t) const {
    (*ES)(u,lambda,A,B,S,P,IB,t);
}

void ESolver::Init (vector<Vector>& u, Operator& B, Operator& S) const {
    ES->Init(u,B,S); 
}

void ESolver::operator () (vector<Vector>& u, DoubleVector& lambda, 
			   Operator& A, Operator& B, Operator& S, 
			   int t) const {
    (*ES)(u,lambda,A,B,S,t);
}

ESolver::~ESolver () { delete ES; }
