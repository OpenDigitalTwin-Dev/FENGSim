// file: Algebra.h
// author: Christian Wieners
// $Header: /public/M++/src/Algebra.h,v 1.16 2009-10-26 15:23:10 maurer Exp $

#ifndef _ALGEBRA_H_
#define _ALGEBRA_H_

#include "Tensor.h"
#include "IO.h"
#include "MatrixGraph.h"
#include "Interface.h"

class BasicVector {
    Scalar* a;
    int n;
 public:
    BasicVector (int N);
    BasicVector (Scalar b, int N);
    BasicVector (const BasicVector& u);
    int size () const { return n; }
    Scalar operator [] (int i) const { return a[i]; }
    Scalar& operator [] (int i) { return a[i]; }
    const Scalar* operator () () const { return a; }
    Scalar* operator () () { return a; }
    BasicVector& operator = (const BasicVector& u);
    BasicVector& operator = (Scalar b);
    BasicVector& Basis (int k);
    BasicVector& operator *= (Scalar b);
    BasicVector& operator /= (Scalar b);
    BasicVector& operator += (const BasicVector& u);
    BasicVector& operator -= (const BasicVector& u);
    void Multiply (Scalar b, const BasicVector& u);
    void MultiplyPlus (Scalar b, const BasicVector& u);
    void MultiplyMinus (Scalar b, const BasicVector& u);
    ~BasicVector ();
    friend ostream& operator << (ostream& s, const BasicVector& u);
    friend Scalar operator * (const BasicVector& u, const BasicVector& v);
    double Max () const;
    double Min () const;
    double norm () const;
    void CleanZero ();
    BasicVector& ref () { return *this; }
    const BasicVector& ref () const { return *this; }
};

class Operator {
 public:
    virtual void apply_plus (Vector&, Vector&) { 
	Exit("apply_plus not implemented"); }
    virtual void apply (Vector&, Vector&);
    virtual void apply_minus (Vector&, Vector&);
    virtual void apply_transpose_plus (Vector&, Vector&) { 
	Exit("apply_transpose_plus not implemented"); }
    virtual void apply_transpose (Vector& u, Vector& v);
    virtual void apply_transpose_minus (Vector& u, Vector& v);
    virtual void multiply_plus (Vector&, const Vector&) const {
	Exit("multiply_plus not implemented"); }
    virtual void multiply (Vector& u, const Vector& v) const;
    virtual void multiply_minus (Vector& u, const Vector& v) const;
    virtual void multiply_transpose_plus (Vector&, const Vector&) const { 
	Exit("multiply_transpose_plus not implemented"); }
    virtual void multiply_transpose (Vector& u, const Vector& v) const;
    virtual void multiply_transpose_minus (Vector& u, const Vector& v) const;
    virtual ~Operator () {};
};


class DirichletFlags {
    bool* f;
    int n;
    bool Delete;
 public:
    void ClearDirichletFlags ();
    DirichletFlags (int N);
    DirichletFlags (const DirichletFlags& dir);
    ~DirichletFlags ();
    const bool* D () const { return f; }
    bool* D () { return f; }
};

class Vector : public matrixgraph, public BasicVector, public DirichletFlags {
 public:
    Vector (const matrixgraph& g);
    Vector (const Vector& u);
    Vector (Scalar b, const Vector& u);
    Vector (const constAB<Operator,Vector>& Ov);
    int size () const { return BasicVector::size(); }
    Vector& operator = (const Vector& u);
    Vector& operator = (Scalar b);
    Vector& operator *= (Scalar b);
    Vector& operator *= (int b);
    Vector& operator /= (Scalar b);
    Vector& operator += (const Vector& u);
    Vector& operator -= (const Vector& u);
    Vector& operator = (const constAB<Scalar,Vector>& au);
    Vector& operator += (const constAB<Scalar,Vector>& au);
    Vector& operator -= (const constAB<Scalar,Vector>& au);
    Vector& operator = (const constAB<int,Vector>& au);
    Vector& operator = (const constAB<Vector,Operator>& vO);
    Vector& operator += (const constAB<Vector,Operator>& vO);
    Vector& operator -= (const constAB<Vector,Operator>& vO);
    Vector& operator = (const constAB<Operator,Vector>& Ov);
    Vector& operator += (const constAB<Operator,Vector>& Ov);
    Vector& operator -= (const constAB<Operator,Vector>& Ov);
    
    const Scalar* operator () () const { return BasicVector::operator()(); }
    Scalar* operator () () { return BasicVector::operator()(); }
    const Scalar* operator () (int i) const { return (*this)() + Index(i); }
    Scalar* operator () (int i) { return (*this)() + Index(i); }
    const Scalar* operator () (const row& r) const { return (*this)(r.Id()); }
    Scalar* operator () (const row& r) { return (*this)(r.Id()); }
    const Scalar* operator () (const Point& z) const { 
	return (*this)(find_row(z)); }
    Scalar* operator () (const Point& z) { return (*this)(find_row(z)); }
    Scalar operator () (int i, int k) const { return (*this)(i)[k]; }
    Scalar& operator () (int i, int k) { return (*this)(i)[k]; }
    Scalar operator () (const row& r, int k) const { return (*this)(r)[k]; }
    Scalar& operator () (const row& r, int k) { return (*this)(r)[k]; }
    Scalar operator () (const Point& z, int k) const { return (*this)(z)[k];}
    Scalar& operator () (const Point& z, int k) { return (*this)(z)[k]; }
    const bool* D () const { return DirichletFlags::D(); }
    bool* D () { return DirichletFlags::D(); }
    const bool* D (int i) const { return DirichletFlags::D() + Index(i); }
    bool* D (int i) { return DirichletFlags::D() + Index(i); }
    const bool* D (const row& r) const { return D(r.Id()); }
    bool* D (const row& r) { return D(r.Id()); }
    const bool* D (const Point& z) const { return D(find_row(z)); }
    bool* D (const Point& z) { return D(find_row(z)); }
    bool D (int i, int k) const { return D(i)[k]; }
    bool& D (int i, int k) { return D(i)[k]; }
    bool D (const row& r, int k) const { return D(r)[k]; }
    bool& D (const row& r, int k) { return D(r)[k]; }
    bool D (const Point& z, int k) const { return D(z)[k];}
    bool& D (const Point& z, int k) { return D(z)[k]; }
    void ClearDirichletValues ();
    void print () const;
    #ifdef NDOUBLE
    Vector& operator *= (double b);
    Vector& operator = (const constAB<double,Vector>& au);
    Vector& operator += (const constAB<double,Vector>& au);
    Vector& operator -= (const constAB<double,Vector>& au);
    #endif
};


Vector& operator + (const Vector& u, const Vector& v);
Vector& operator - (const Vector& u, const Vector& v);
Scalar operator * (const Vector& u, const Vector& v);
ostream& operator << (ostream& s, const Vector& u);

double norm (const Vector& u);
#ifndef DCOMPLEX
double norm (const Vector& u, int i);
#endif

class RowValues {
    Scalar* a[MaxNodalPoints]; 
    void fill (Vector& u, const rows& R);
 public:
    RowValues (Vector& u, const rows& R);
    RowValues (Vector& u, const cell& c);
    Scalar& operator () (int i, int k=0) { return a[i][k]; }
};
class RowBndValues {
    BFParts BF;
    Scalar* a[MaxNodalPoints]; 
    bool* b[MaxNodalPoints]; 
    void fill (Vector& u, const rows& R);
 public:
    RowBndValues (Vector& u, const cell& c);    
    Scalar& operator () (int i, int k=0) { return a[i][k]; }
    bool& D (int i, int k=0) { return b[i][k]; }
    bool onBnd () const { return BF.onBnd(); }
    int bc (int i) { return BF[i]; }
};

template <class F> Vector& operator << (Vector& u, const F& f);

Vector& operator << (Vector& u, int (*F)());

void RandomVector (Vector& u, Operator& B, Operator& IA);

void RandomVectors (vector<Vector>& u, Operator& B, Operator& IA, int i = 0);


class Matrix : public matrixgraph, public BasicVector, public Operator {
    const Vector& u;
 public:
    Matrix (const Vector& U) : matrixgraph(U), BasicVector(U.Size()), u(U) {}
    Matrix& operator = (Scalar b) { BasicVector::operator=(b); return *this; }
    const Vector& GetVector () const { return u; }
    int size () const { return matrixgraph::size(); }
    const Scalar* operator () () const { return BasicVector::operator()(); }
    Scalar* operator () () { return BasicVector::operator()(); }
    const Scalar* operator () (int d) const { return (*this)() + Entry(d); }
    Scalar* operator () (int d) { return (*this)() + Entry(d); }
    const Scalar* operator () (const row& r0, const row& r1) const {
	return (*this)() + GetEntry(r0,r1); }
    Scalar* operator () (const row& r0, const row& r1) {
	return (*this)() + GetEntry(r0,r1); }
    Matrix& operator = (const Matrix& A);
    Matrix& operator += (const constAB<double,Matrix>& aA);
    void multiply_plus (Vector&, const Vector&) const;
    void multiply_transpose_plus (Vector&, const Vector&) const;
    void copy (Scalar*, int*, int*) const;
    void ClearDirichletValues ();
    void Symmetric ();

    void EliminateDirichlet();
};

ostream& operator << (ostream& s, const Matrix& u);

constAB<Operator,Vector> operator * (const Matrix& A,const Vector& v);

class RowEntries {
    Scalar* a[MaxNodalPoints][MaxNodalPoints]; 
    int n[MaxNodalPoints]; 
    void fill (Matrix& A, const rows& R);
 public:
    RowEntries (Matrix& A, const rows& R);
    RowEntries (Matrix& A, const cell& c);
    Scalar& operator () (int i, int j, int k=0, int l=0) { 
	return a[i][j][k*n[j]+l]; 
    }
};

#endif
