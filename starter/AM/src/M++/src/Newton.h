// file:   Newton.h
// author: Christian Wieners
// $Header: /public/M++/src/Newton.h,v 1.8 2007-10-05 10:54:59 wieners Exp $

#ifndef _NEWTON_H_
#define _NEWTON_H_

#include "LinearSolver.h"
#include "Assemble.h"

class NonlinearSolver : public Iteration {
    string name;
public:
    NonlinearSolver (const char* n) : Iteration(n), name(n) {}
    virtual void operator () (const Assemble&, Vector&) = 0;
    string Name () const { return name; }
};

class Newton : public NonlinearSolver {
    Solver& S;
    int suppressLS;
    int JacobiUpdate;
    // JacobiUpdate: Determines after how many steps the Jacobian is updated
    // 1   : in every step == classical Newton (default)
    // n>1 : in every n-th Newton step
public:
    Newton (Solver& s);
    void operator () (const Assemble& A, Vector& u);
};

#endif
