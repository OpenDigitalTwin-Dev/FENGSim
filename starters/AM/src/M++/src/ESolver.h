// file: ESolver.h
// author: Christian Wieners
// $Header: /public/M++/src/ESolver.h,v 1.4 2009-04-07 11:20:41 wieners Exp $

#ifndef _ESOLVER_H_
#define _ESOLVER_H_

#include "Algebra.h"
#include "Hermitian.h"

class EigenSolver;
EigenSolver* GetESolver (const string&);

class ESolver {
    string name;
    EigenSolver *ES;
 public:
    ESolver () : name("LOBPCG2") {
	ReadConfig(Settings,"ESolver",name);
	ES = GetESolver(name);
    }
    ~ESolver ();
    void Init (vector<Vector>&, Operator&, Operator&) const;
    void Init (vector<Vector>&, Operator&, Operator&, Operator&) const;
    void operator () (vector<Vector>&, DoubleVector&, 
		      Operator&, Operator&, Operator&, int t=0) const;
    void operator () (vector<Vector>&, DoubleVector&, 
		      Operator&, Operator&, Operator&, Operator&, 
		      int t=0) const;
    void operator () (vector<Vector>&, DoubleVector&, 
		      Operator&, Operator&, Operator&, Operator&, 
		      Operator&, int t=0) const;
};

#endif
