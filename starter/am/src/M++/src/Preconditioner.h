// file: Preconditioner.h
// author: Christian Wieners
// $Header: /public/M++/src/Preconditioner.h,v 1.6 2009-09-21 17:34:28 wieners Exp $

#ifndef _PRECONDITIONER_H_
#define _PRECONDITIONER_H_

#include "Time.h"
#include "IO.h"
#include "Algebra.h"
#include "Assemble.h"
#include "Transfer.h"
#include "Sparse.h"

class Preconditioner : public Operator {
 protected:
    int verbose;
 public:
    virtual void Construct (const Matrix&) = 0;
    virtual void Destruct () = 0;
    Preconditioner() : verbose(0) {
	ReadConfig(Settings,"PreconditionerVerbose",verbose);
    }
    virtual void multiply_transpose (Vector& u, const Vector& b) const { 
	Vector r(b);
        Accumulate(r);        
	multiply(u,r); 
        MakeAdditive(u);     
        Accumulate(u);                
    }
    virtual ~Preconditioner() {}
    virtual string Name () const = 0;
    friend ostream& operator << (ostream& s, const Preconditioner& PC) {
	return s << PC.Name(); 
    }
};
inline constAB<Operator,Vector> 
    operator * (const Preconditioner& PC, const Vector& v) {
    return constAB<Operator,Vector>(PC,v); 
}
inline constAB<Vector,Operator> 
    operator * (const Vector& v, const Preconditioner& PC) {
    return constAB<Vector,Operator>(v,PC); 
}

class Transfer;
class Solver;

class MultigridPreconditioner : public Preconditioner {    
 public:
    const MatrixGraphs& G;
    int levels;
    vector<Transfer*> T;
    vector<Preconditioner*> PC;
    Solver* LS;
    virtual void Construct (const Matrix&) = 0;
    virtual void Destruct () = 0;
    virtual string Name () const = 0;
 public:
    MultigridPreconditioner (const MatrixGraphs& g) : 
	G(g), levels(g.Level()), T(levels), PC(levels+1) {}
    virtual ~MultigridPreconditioner() {}
};

class DDPreconditioner : public Preconditioner {
 public:
    int GaussSeidel;
    int backward;
    int forward;
    Matrix* A;
    SparseMatrix* Sp;
 public:
    virtual void Construct (const Matrix&) = 0;
    virtual void Destruct () = 0;
    virtual string Name () const = 0;
    DDPreconditioner () : GaussSeidel(1), backward(0), forward(1), A(0), Sp(0) {
        ReadConfig(Settings,"GaussSeidel",GaussSeidel);
        ReadConfig(Settings,"backward",backward);
        ReadConfig(Settings,"forward",forward);
    }
    virtual ~DDPreconditioner () {}
};


Preconditioner* GetPC ();
Preconditioner* GetPC (const string&);
MultigridPreconditioner* GetMultigrid (const MatrixGraphs&, const Assemble&);
MultigridPreconditioner* GetMultigrid (const MatrixGraphs&, 
				       const Assemble&, const Transfer&);
Preconditioner* GetPC (const MatrixGraphs&, const Assemble&, const string&);
Preconditioner* GetPC (const MatrixGraphs&, const Assemble&);
Preconditioner* GetPC (const MatrixGraphs&, const Assemble&, const Transfer&);

#endif
