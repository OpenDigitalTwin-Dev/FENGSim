// file: LinearSolver.h
// author: Christian Wieners
// $Header: /public/M++/src/LinearSolver.h,v 1.16 2009-09-21 17:34:28 wieners Exp $

#ifndef _LINEARSOLVER_H_
#define _LINEARSOLVER_H_

#include "Preconditioner.h"

class Iteration {
 protected:
    int verbose;
    int iter;
    int max_iter;
    int LS_iter;
    double Eps;
    double Red;
    double d,d_0;
    Date Start;
 public:
    Iteration (const string& prefix = "Linear") 
	: verbose(0), max_iter(10), LS_iter(3), Eps(1e-10), Red(1e-5) {
	ReadConfig(Settings,prefix + "Verbose",verbose);
	ReadConfig(Settings,prefix + "Steps",max_iter);
	ReadConfig(Settings,prefix + "LineSearchSteps",LS_iter);
	ReadConfig(Settings,prefix + "Epsilon",Eps);
	ReadConfig(Settings,prefix + "Reduction",Red);
    }    
    virtual ~Iteration () {}
    bool converged () const { return (iter<max_iter); }
    double rate () const { 
	if (abs(d) > VeryLarge) Exit("no convergence in iteration");
	if (iter) if (d_0 != 0) return pow(d/d_0,1.0/iter); 
	return 0;
    }
    int Verbose () const { return verbose; }
    int Iter () const { return iter; }
    int Steps () const { return max_iter; }
    int LSSteps () const { return LS_iter; }
    double Epsilon () const { return Eps; }
    double Reduction () const { return Red; }
    void Reduction (double r) { Red = r; }
    double defect () const { return d; }
    virtual string Name () const = 0;
    const Iteration& IterationEnd () const { return *this; }
    friend ostream& operator << (ostream& s, const Iteration& S) {
	s << S.Name() << ": d(" << S.iter << ")= " << S.d 
	  << " rate " << S.rate();
	if (TimeLevel<0) s << endl;
	else s << " time " << Date() - S.Start << endl;
	logging->flush();
	return s;
    }
};

class IterativeSolver : public Operator, public Iteration {
 public:
    IterativeSolver (const string& prefix = "Linear") : Iteration (prefix) {}
    virtual void Solve (Vector&, const Operator&, const Operator&, Vector&)=0;
    friend ostream& operator << (ostream& s, const IterativeSolver& S) {
	return s << S.Name(); 
    }
};
IterativeSolver* GetIterativeSolver (const string&, const string&);
IterativeSolver* GetIterativeSolver();

class Solver : public Operator {
 protected:
    Preconditioner* PC; 
    IterativeSolver* S; 
    const Matrix* A;
 public:
    Solver ();
    Solver (Preconditioner*);
    Solver (Preconditioner*, const string&);
    Solver (Preconditioner*, const string&, const string&);
    Solver (const MatrixGraphs&, const Assemble&);
    Solver (const MatrixGraphs&, const Assemble&, const Transfer&);
    Solver (const MatrixGraphs&, const Assemble&, const string&);
    Solver (IterativeSolver* s, Preconditioner* pc) : S(s), PC(pc) {}        
    virtual Operator& operator () (const Matrix& _A, int reassemblePC = 1) {
	A = &_A;
	if (reassemblePC) {
            if (PC) PC->Destruct();
	    if (PC) PC->Construct(*A);
	}
	return *this;
    }
    virtual ~Solver () { 
	if (PC) PC->Destruct(); 
	delete S;
    }
    virtual void multiply_plus (Vector& u, const Vector& b) const { 
	Vector r = b;
	S->Solve(u,*A,*PC,r); 
    }
    friend ostream& operator << (ostream& s, const Solver& S) {
	return s << *S.S << " with " << *S.PC; 
    }
};
inline constAB<Operator,Vector> 
operator * (const Solver& S, const Vector& v) {
    return constAB<Operator,Vector>(S,v); 
}
Solver* GetSolver ();
Solver* GetSolver (const string&, const string&);
Solver* GetSolver (Preconditioner*, const string&, const string&);
Solver* GetSolver (const MatrixGraphs&, const Assemble&);
Solver* GetSolver (const MatrixGraphs&, const Assemble&,
		   const string&, const string&);

#endif
