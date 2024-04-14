// file:   Small.h
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/Small.h,v 1.13 2009-10-28 15:18:56 maurer Exp $

#ifndef _SMALL_H_
#define _SMALL_H_

#include "LinearSolver.h"
#include "Algebra.h"

#include <cmath>
#include <algorithm>
#include <valarray>

// ----------------------------------------------------
//    SmallMatrix
// ----------------------------------------------------
class SparseMatrix;
class SmallMatrix;

constAB<SmallMatrix,SmallMatrix> operator * (const SmallMatrix& A, const SmallMatrix& B);

class SmallMatrix {
    Scalar* a;
    int n;
    int m;
public:
    SmallMatrix (int N) : n(N), m(N) { 
        a = new Scalar [n*m]; 
        for (int i=0; i<n*m; ++i) a[i] = 0;
        if (n*m==0) {n = 0; m = 0;}
    }
    SmallMatrix (int N, int M) : n(N), m(M) { 
        a = new Scalar [n*m]; 
        for (int i=0; i<n*m; ++i) a[i] = 0;  
        if (n*m==0) {n = 0; m = 0;}
    }
    SmallMatrix (Scalar, int, int);
    SmallMatrix (const SmallMatrix&);
    SmallMatrix (const SparseMatrix&, const vector<int>&);
    SmallMatrix (const SparseMatrix&);

    SmallMatrix (int N, int M, const SmallMatrix& A);
    
    SmallMatrix (const SmallMatrix& A, const SmallMatrix& B);
    
    void Destruct() {delete[] a; a=0; }

    virtual ~SmallMatrix () {Destruct();}

    SmallMatrix& operator = (const SmallMatrix& y) { 
        if (&y != this) memcpy(a,y.ref(),n*m*sizeof(double)); 
	return *this; 
    }

    SmallMatrix& operator = (Scalar x);
    
    SmallMatrix& operator -= (const SmallMatrix& A);
    
    SmallMatrix& operator += (const SmallMatrix& A);

    SmallMatrix& operator *= (const SmallMatrix& A);

    SmallMatrix& operator *= (double b);

    SmallMatrix& operator -= (const constAB<SmallMatrix,SmallMatrix>& AB);
    
    Scalar* operator [] (int i) { return a+m*i; }
    
    double norm () const;
    
    const Scalar* operator [] (int i) const { return a+m*i; } 
    Scalar *Set (int i, int j) const { return a+m*i+j; } 
    int rows() const { return n; }
    int cols() const { return m; }

    void invert();

    void Accumulate();
    
    int nz() const;
    
    void copy (Scalar* nzval, int* rowind, int* colptr) const;
    
    void resize(int N, int M) {
        delete[] a;
        n = N;
        m = M;
        a = new Scalar[n*m];
        for (int i=0; i<n*m; ++i) a[i] = 0;
//         if (n*m==0) {n = 0; m = 0;}
    }
    Scalar* ref() {return a;}
    Scalar* ref() const {return a;}
};

ostream& operator << (ostream& s, const SmallMatrix& M);

inline SmallMatrix operator + (const SmallMatrix& x, const SmallMatrix& y) {
    SmallMatrix z = x; return z += y;
}

inline SmallMatrix operator - (const SmallMatrix& x, const SmallMatrix& y) {
    SmallMatrix z = x; return z -= y;
}

inline SmallMatrix operator * (double b, const SmallMatrix& x) {
    SmallMatrix z = x; return z *= b;
}

class SmallMatrixTrans: public SmallMatrix {
    int* IPIV;
    bool inverted;
    bool lu;
    bool symmetric;
public:
    SmallMatrixTrans(int n, int m, bool symm = false): SmallMatrix(m,n), lu(false), inverted(false), IPIV(0), symmetric(symm) {}

    void Destruct() {
        if (IPIV) delete[] IPIV; IPIV = 0;
    }

    ~SmallMatrixTrans() {Destruct();}
    Scalar& operator() (int i, int j) {return (*this)[j][i];}
    const Scalar& operator() (int i, int j) const {return (*this)[j][i];}
    int rows() const {return SmallMatrix::cols();}
    int cols() const {return SmallMatrix::rows();}
    void makeLU();
    void invert();
    void Solve(Scalar* v, int size = 1, bool trans = false);
    void SetIPIV(Buffer& );
    void SendIPIV(Buffer& );
    size_t GetSizeSendIPIV();
};


class SmallVector;
class DDProblem;

constAB<SmallMatrix,SmallVector> operator * (const SmallMatrix& S, const SmallVector& v);
constAB<DDProblem,SmallVector> operator * (const DDProblem& S, const SmallVector& v);
                                                    
class constBlockVectors;

// ----------------------------------------------------
//    SmallOperator
// ----------------------------------------------------

class SmallOperator;

// ----------------------------------------------------
//    SmallVector
// ----------------------------------------------------

class SmallVector : public valarray<Scalar> {
public:
    SmallVector (Scalar x, int N) : valarray<Scalar>(x,N) {} 
    SmallVector (int N) : valarray<Scalar>(N) {} 
    SmallVector (const constBlockVectors&);
    SmallVector (const SmallMatrix& A, const SmallVector& u);
    
    SmallVector (const SmallVector& u, const SmallMatrix& A);
    
    SmallVector& operator = (const constAB<SmallMatrix,SmallVector>& Au);
    SmallVector& operator = (const constAB<constAB<SmallOperator,Vector>,Vector> & Au);

    SmallVector& operator -= (const constAB<SmallMatrix,SmallVector>& Au);
    SmallVector& operator -= (const constAB<DDProblem,SmallVector>& Au);
    
    SmallVector& operator = (const SmallVector& v);
    
    SmallVector& operator += (const SmallVector& v);
    
    SmallVector& operator -= (const SmallVector& v);
    
    SmallVector& operator *= (const SmallMatrix& A); 
    
    SmallVector& operator *= (const Scalar& a);
    SmallVector& operator *= (const int& a);

    SmallVector& operator = (const Scalar& a);

    double norm () const;
    
    void Accumulate();

    Scalar* ref() {return &(*this)[0];}
    
};

ostream& operator << (ostream& s, const SmallVector& v);

Scalar operator * (const SmallVector& u, const SmallVector& v);

SmallVector operator + (const SmallVector& u, const SmallVector& v);

SmallVector operator * (const SmallVector& u, const SmallMatrix& v);

SmallVector operator - (const SmallVector& u, const SmallVector& v);

SmallVector operator * (Scalar d, const SmallVector& v);

// ----------------------------------------------------
//    SmallFunction
// ----------------------------------------------------
class SmallFunction {
    double h;
 public:
    SmallFunction (double numh = 1e-8) : h(numh) {}
    virtual SmallVector operator () (const SmallVector&) const = 0;
    SmallVector eval (const SmallVector& x) const { return (*this)(x); }
        // operator computes the function value 
    virtual void Jacobian (const SmallVector& x, SmallMatrix& J)  const;
};

// ----------------------------------------------------
//    SmallNewtonMethod
// ----------------------------------------------------

class SmallNewtonMethod : public Iteration {
 public:
    SmallNewtonMethod() : Iteration("SmallNewton") { }
    string Name () const { return "SmallNewtonMethod"; }
    void operator () (SmallVector& x, SmallFunction& f) const;
};

class SmallSolver {
public:
    SmallMatrix operator () (const SmallMatrix &m);
    SmallVector operator () (const SmallMatrix &a, const SmallVector &b);
    virtual void Solve (Scalar *) {}
    virtual ~SmallSolver () {}
};

SmallSolver* GetSmallSolver (SmallMatrix&);
SmallSolver* GetSmallSolver (SmallMatrix&, const string&);

class SmallOperator {
public:
    virtual void multiply (const Vector& u, const Vector& v, SmallVector& c) const {}
    virtual ~SmallOperator () {}
};

#endif
