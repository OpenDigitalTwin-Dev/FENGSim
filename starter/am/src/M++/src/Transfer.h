// file: Transfer.h
// author: Christian Wieners
// $Header: /public/M++/src/Transfer.h,v 1.6 2009-09-21 17:34:27 wieners Exp $

#ifndef _TRANSFER_H_
#define _TRANSFER_H_

#include "IO.h"
#include "Algebra.h"

extern Point kkk;

class Transfer : public Operator {
 protected:
    int verbose;
    vector<bool> dirichlet;
    vector< vector<int> > I;
 public:
    Transfer () : verbose(0) {
	ReadConfig(Settings,"TransferVerbose",verbose);
    }
    virtual ~Transfer () {}  
    virtual void Construct (const matrixgraph&, const matrixgraph&) = 0;
    virtual void loop (Vector&, const Vector&, int, int) const {}
    virtual void multiply (Vector&, const Vector&) const = 0;
    virtual void loop_transpose (Vector&, const Vector&, int, int) const {}
    virtual void multiply_transpose (Vector&, const Vector&) const = 0;
    virtual void Project (const Vector& f, Vector& c) const = 0;
    virtual Transfer* GetTransferPointer () const {}
    const vector<vector<int> >& Get_I() const { return I; }
};
inline constAB<Operator,Vector> 
    operator * (const Transfer& T, const Vector& v) {
    return constAB<Operator,Vector>(T,v); 
}
inline constAB<Vector,Operator> 
    operator * (const Vector& v, const Transfer& T) {
    return constAB<Vector,Operator>(v,T); 
}

Transfer* GetTransfer (const string&);
Transfer* GetTransfer ();

#endif
