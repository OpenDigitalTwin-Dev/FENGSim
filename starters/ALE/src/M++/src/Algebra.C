// file: Algebra.C
// author: Christian Wieners
// $Header: /public/M++/src/Algebra.C,v 1.8 2009-10-26 15:23:10 maurer Exp $

#include "Algebra.h"

BasicVector::BasicVector (int N) : n(N) { a = new Scalar [n]; }

BasicVector::BasicVector (Scalar b, int N) : n(N) {
    a = new Scalar [n];
    for (int i=0; i<n; ++i) a[i] = b;
}

BasicVector::BasicVector (const BasicVector& u) : n(u.n) {
    a = new Scalar [n];
    for (int i=0; i<n; ++i) a[i] = u.a[i];
}

BasicVector::~BasicVector () { delete[] a; }

BasicVector& BasicVector::operator = (const BasicVector& u) {
    if (this == &u) return *this;
    if (n != u.n) Exit("BasicVector size not matching");
    for (int i=0; i<n; ++i) a[i] = u.a[i];
    return *this;
}

BasicVector& BasicVector::operator = (Scalar b) {
    for (int i=0; i<n; ++i) a[i] = b;
    return *this;
}

BasicVector& BasicVector::Basis (int k) {
    for (int i=0; i<n; ++i) a[i] = 0;
    a[k] = 1.0;
    return *this;
}

BasicVector& BasicVector::operator *= (Scalar b) {
    for (int i=0; i<n; ++i) a[i] *= b;
    return *this;
}

BasicVector& BasicVector::operator /= (Scalar b) {
    for (int i=0; i<n; ++i) a[i] /= b;
    return *this;
}

BasicVector& BasicVector::operator += (const BasicVector& u) {
    for (int i=0; i<n; ++i) a[i] += u.a[i];
    return *this;
}

BasicVector& BasicVector::operator -= (const BasicVector& u) {
    for (int i=0; i<n; ++i) a[i] -= u.a[i];
    return *this;
}

void BasicVector::Multiply (Scalar b, const BasicVector& u) {
    for (int i=0; i<n; ++i) a[i] = b * u.a[i]; 
}

void BasicVector::MultiplyPlus (Scalar b, const BasicVector& u) {
    for (int i=0; i<n; ++i) a[i] += b * u.a[i]; 
}

void BasicVector::MultiplyMinus (Scalar b, const BasicVector& u) {
    for (int i=0; i<n; ++i) a[i] -= b * u.a[i]; 
}

ostream& operator << (ostream& s, const BasicVector& u) {
    for (int i=0; i<u.n; ++i) s << " " << u.a[i];
    return s << endl;
}

Scalar operator * (const BasicVector& u, const BasicVector& v) {
    Scalar s = 0;
    for (int i=0; i<u.n; ++i) s += conj(u.a[i]) * v.a[i];
    return PPM->Sum(s);
}

double BasicVector::Max () const {
    double s = -infty;
    for (int i=0; i<n; ++i) s = max(s,real(a[i]));
    return PPM->Max(s);
}

double BasicVector::Min () const {
    double s = infty;
    for (int i=0; i<n; ++i) s = min(s,real(a[i]));
    return PPM->Min(s);
}

double BasicVector::norm () const {
    return abs(sqrt((*this)*(*this)));
}

void BasicVector::CleanZero () {
    for (int i=0; i<n; ++i)
        if (abs(a[i]) < 1e-12) a[i] = 0;
}


void DirichletFlags::ClearDirichletFlags () {
    for (int i=0; i<n; ++i) f[i] = false;
}
DirichletFlags::DirichletFlags (int N) : n(N), Delete(true) { 
    f = new bool [n]; 
    ClearDirichletFlags();
}

DirichletFlags::DirichletFlags (const DirichletFlags& dir) :
	f(dir.f), n(dir.n),  Delete(false) {}

DirichletFlags::~DirichletFlags () { if (Delete) delete[] f; }

Vector::Vector (const matrixgraph& g) : 
	matrixgraph(g), BasicVector(g.size()), DirichletFlags(g.size()) {}
Vector::Vector (const Vector& u) : matrixgraph(u), 
	BasicVector(u), DirichletFlags(u) {}
Vector::Vector (Scalar b, const Vector& u) : matrixgraph(u), 
	BasicVector(b,u.size()), DirichletFlags(u) {}
Vector::Vector (const constAB<Operator,Vector>& Ov) :
	matrixgraph(Ov.second()), BasicVector(Ov.second()),
	DirichletFlags(Ov.second()) {
    Ov.first().multiply(*this,Ov.second()); 
}

Vector& Vector::operator = (const Vector& u) {
    this->BasicVector::ref() = u.BasicVector::ref();
    return *this; 
}

Vector& Vector::operator = (Scalar b) {
    this->BasicVector::ref() = b;
    return *this; 
}

Vector& Vector::operator *= (Scalar b) {
    this->BasicVector::ref() *= b;
    return *this; 
}

Vector& Vector::operator *= (int b) {
    this->BasicVector::ref() *= Scalar(b);
    return *this; 
}

Vector& Vector::operator /= (Scalar b) {
    this->BasicVector::ref() /= b;
    return *this; 
}

Vector& Vector::operator += (const Vector& u) {
    this->BasicVector::ref() += u.BasicVector::ref();
    return *this; 
}

Vector& Vector::operator -= (const Vector& u) {
    this->BasicVector::ref() -= u.BasicVector::ref();
    return *this; 
}

Vector& operator + (const Vector& u, const Vector& v) {
    Vector w = u;
    return w += v;
}

Vector& operator - (const Vector& u, const Vector& v) {
    Vector w = u;
    return w -= v;
}
Vector& Vector::operator = (const constAB<Scalar,Vector>& au) {
    Multiply(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator += (const constAB<Scalar,Vector>& au) {
    MultiplyPlus(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator -= (const constAB<Scalar,Vector>& au) {
    MultiplyMinus(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator = (const constAB<int,Vector>& au) {
    Multiply(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator = (const constAB<Vector,Operator>& vO) { 
    vO.second().multiply_transpose(*this,vO.first()); 
    return *this; 
}

Vector& Vector::operator += (const constAB<Vector,Operator>& vO) { 
    vO.second().multiply_transpose_plus(*this,vO.first()); 
    return *this; 
}

Vector& Vector::operator -= (const constAB<Vector,Operator>& vO) { 
    vO.second().multiply_transpose_minus(*this,vO.first()); 
    return *this; 
}

Vector& Vector::operator = (const constAB<Operator,Vector>& Ov) { 
    Ov.first().multiply(*this,Ov.second()); 
    return *this; 
}

Vector& Vector::operator += (const constAB<Operator,Vector>& Ov) {  
    Ov.first().multiply_plus(*this,Ov.second()); 
    return *this; 
}

Vector& Vector::operator -= (const constAB<Operator,Vector>& Ov) { 
    Ov.first().multiply_minus(*this,Ov.second()); 
    return *this; 
}
/*
void Vector::print () const {
    for (row r=rows(); r!=rows_end(); ++r) {
	mout << r() << " :";
	for (int i=0; i<r.n(); ++i)
	    mout << " " << (*this)(r,i);
	mout << endl;
    }
    }*/

void Vector::print () const {
    vector <row> R(Size());
    for (row r=rows(); r!=rows_end(); ++r) 
        R[r.Id()] = r;
    for (int k=0; k<R.size(); ++k) {
        mout << R[k]() << " :";
        for (int i=0; i<R[k].n(); ++i)
            mout << " " << D(R[k],i);
        mout << " :";
        for (int i=0; i<R[k].n(); ++i)
            mout << " " << (*this)(R[k],i);
        mout << endl;
    }
    return;
    for (row r=rows(); r!=rows_end(); ++r) {
        mout << r() << " :";
        for (int i=0; i<r.n(); ++i)
            mout << " " << D(r,i);
        mout << " :";
        for (int i=0; i<r.n(); ++i)
            mout << " " << (*this)(r,i);
        mout << endl;
    }
}


Scalar operator * (const Vector& u, const Vector& v) {
    return u.BasicVector::ref() * v.BasicVector::ref();
}

ostream& operator << (ostream& s, const Vector& u) {
    return s << u.BasicVector::ref(); }

void Vector::ClearDirichletValues () { 
    Scalar* a = (*this)();
    bool* b = D();
    for (int i=0; i<size(); ++i, ++a, ++b) 
        if (*b) *a = 0; 
}

#ifdef NDOUBLE
Vector& Vector::operator *= (double b) {
    this->BasicVector::ref() *= Scalar(b);
    return *this; 
}

Vector& Vector::operator = (const constAB<double,Vector>& au) {
    Multiply(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator += (const constAB<double,Vector>& au) {
    MultiplyPlus(au.first(),au.second()); 
    return *this; 
}

Vector& Vector::operator -= (const constAB<double,Vector>& au) {
    MultiplyMinus(au.first(),au.second()); 
    return *this; 
}
#endif

double norm (const Vector& u) { return u.norm(); }

#ifndef DCOMPLEX
double norm (const Vector& u, int i) {
    double sum = 0.0;
    for (row r = u.rows(); r!=u.rows_end(); ++r)
	if (i < r.n()) sum += u(r,i)*u(r,i);
    return sqrt(sum);
}
#endif

void Operator::apply (Vector& u, Vector& v) { 
    u = 0; apply_plus(u,v); }
void Operator::apply_minus (Vector& u, Vector& v) { 
    Vector tmp(u);
    apply(tmp,v); 
    u -= tmp;
}
void Operator::apply_transpose (Vector& u, Vector& v) { 
    u = 0; apply_transpose_plus(u,v); }
void Operator::apply_transpose_minus (Vector& u, Vector& v) { 
    Vector tmp(u);
    apply_transpose(tmp,v); 
    u -= tmp;
}
void Operator::multiply (Vector& u, const Vector& v) const { 
    u = 0; multiply_plus(u,v); }
void Operator::multiply_minus (Vector& u, const Vector& v) const { 
    Vector tmp(u);
    multiply(tmp,v); 
    u -= tmp;
}
void Operator::multiply_transpose (Vector& u, const Vector& v) const { 
    u = 0; multiply_transpose_plus(u,v); }

void Operator::multiply_transpose_minus (Vector& u, const Vector& v) const { 
    Vector tmp(u);
    multiply_transpose(tmp,v); 
    u -= tmp;
}

void RowValues::fill (Vector& u, const rows& R) {
    for (int i=0; i<R.size(); ++i) a[i] = u(R[i]);
}

RowValues::RowValues (Vector& u, const rows& R) { fill(u,R); }

RowValues::RowValues (Vector& u, const cell& c) { rows R(u,c); fill(u,R); }

void RowBndValues::fill (Vector& u, const rows& R) {
    for (int i=0; i<R.size(); ++i) {
        a[i] = u(R[i]);
        b[i] = u.D(R[i]);
    }
}

RowBndValues::RowBndValues (Vector& u, const cell& c) : BF(u.GetMesh(),c) { 
    rows R(u,c); fill(u,R); }

template <class F> Vector& operator << (Vector& u, const F& f) {
  //for (row r = u.rows(); r!=u.rows_end(); ++r) u(r,0) = eval(f(r()));
    return u;
}

Vector& operator << (Vector& u, int (*F)()) {
    for (int i=0; i<u.size(); ++i) u[i] = Scalar(F());
    return u;
}

void RandomVector (Vector& u, Operator& B, Operator& IA) {
    Vector v(u);
    v << rand;
    v.ClearDirichletValues();
    v *= (1.0 / RAND_MAX);
    Vector b(u);
    b = B * v;
    double s = sqrt(real(b * v));
    b /= s;
    u = IA * b;
}

void RandomVectors (vector<Vector>& u, Operator& B, Operator& IA, int i) {
    for (; i<u.size(); ++i) RandomVector(u[i],B,IA);
}

Matrix& Matrix::operator = (const Matrix& A) {
    for (int i=0; i<Size(); ++i)
        (*this)[i] = A[i]; 
    return *this; 
}

Matrix& Matrix::operator += (const constAB<double,Matrix>& aA) {
    double s = aA.first();
    for (int i=0; i<Size(); ++i)
        (*this)[i] += s * aA.second()[i]; 
    return *this; 
}

void Matrix::multiply_plus (Vector& b, const Vector& u) const {
    const Scalar* a = (*this)();
    for (int i=0; i<nR(); ++i) {
        int d = Diag(i);
        int n = Dof(i);
	
        for (int k=0; k<n; ++k) 
            for (int l=0; l<n; ++l, ++a) 
                b(i,k) += *a * u(i,l);

	for (++d; d<Diag(i+1); ++d) {
            int j = Column(d);
            int m = Dof(j);
            for (int k=0; k<n; ++k) 
                for (int l=0; l<m; ++l, ++a) 
                    b(i,k) += *a * u(j,l);
            for (int k=0; k<m; ++k) 
                for (int l=0; l<n; ++l, ++a) 
                    b(j,k) += *a * u(i,l);
        }
	
    }
    Collect(b);
}

void Matrix::multiply_transpose_plus (Vector& b, const Vector& u) const {
    const Scalar* a = (*this)();
    for (int i=0; i<nR(); ++i) {
        int d = Diag(i);
        int n = Dof(i);
        for (int k=0; k<n; ++k) 
            for (int l=0; l<n; ++l, ++a) 
                b(i,l) += *a * u(i,k);
        for (++d; d<Diag(i+1); ++d) {
            int j = Column(d);
            int m = Dof(j);
            for (int k=0; k<n; ++k) 
                for (int l=0; l<m; ++l, ++a) 
                    b(j,l) += *a * u(i,k);
            for (int k=0; k<m; ++k) 
                for (int l=0; l<n; ++l, ++a) 
                    b(i,l) += *a * u(j,k);
        }
    }
    Collect(b);
}

void Matrix::copy (Scalar* a, int* ind, int* col) const { 
    const Scalar* A = (*this)();
    for (int i=0; i<Index(nR()); ++i) ind[i] = 0;
    for (int i=0; i<nR(); ++i)
        for (int k=0; k<Dof(i); ++k) {
            ind[Index(i)+k] += Dof(i);
            for (int d=Diag(i)+1; d<Diag(i+1); ++d) {
                int j = Column(d);
                int nc = Dof(j);
                for (int l=0; l<nc; ++l) {
                    ++ind[Index(i)+k];
                    ++ind[Index(j)+l];
                }
            }
        }
    for (int i=1; i<Index(nR()); ++i) ind[i] += ind[i-1];
    for (int i=Index(nR()); i>0; --i) ind[i] = ind[i-1];
    ind[0] = 0;
    for (int i=0; i<nR(); ++i) {
        int nc = Dof(i);
        for (int k=0; k<nc; ++k) {
            int d = Diag(i);
            a[ind[Index(i)+k]] = A[Entry(d)+k*nc+k];
            col[ind[Index(i)+k]++] = Index(i)+k;
            for (int l=0; l<nc; ++l) {
                if (l == k) continue;
                a[ind[Index(i)+k]] = A[Entry(d)+k*nc+l];
                col[ind[Index(i)+k]++] = Index(i)+l;
            }
        }
    }
    for (int i=0; i<nR(); ++i) {
        int nr = Dof(i);
        for (int k=0; k<nr; ++k) {
            int d = Diag(i);
            for (++d; d<Diag(i+1); ++d) {
                int j = Column(d);
                int nc = Dof(j);
                for (int l=0; l<nc; ++l) {
                    a[ind[Index(i)+k]] = A[Entry(d)+k*nc+l];
                    col[ind[Index(i)+k]++] = Index(j)+l;
                    a[ind[Index(j)+l]] = A[Entry(d)+nc*nr+l*nr+k]; 
                    col[ind[Index(j)+l]++] = Index(i)+k;
                }
            }
        }
    }
    for (int i=Index(nR()); i>0; --i) ind[i] = ind[i-1];
    ind[0] = 0;
}

void Matrix::ClearDirichletValues () { 
    const bool* Dirichlet = u.D();
    Scalar* A = (*this)();
    int ind = 0;
    for (int i=0; i<nR(); ++i) {
        int nr = Dof(i);
        for (int k=0; k<nr; ++k) {
	    int d = Diag(i);
            if (Dirichlet[ind++]) {
                int e = Entry(d); 
                A[e+k*nr+k] = 1.0;
                for (int l=0; l<nr; ++l) 
                    if (l != k)
                        A[e+k*nr+l] = 0.0;
                for (++d; d<Diag(i+1); ++d) {
                    int j = Column(d);
                    int nc = Dof(j);
                    e = Entry(d); 
                    for (int l=0; l<nc; ++l) 
                        A[e+k*nc+l] = 0.0;
                    e += nr * nc;
                    for (int l=0; l<nc; ++l)
                        if (Dirichlet[Index(j)+l])
                            A[e+l*nr+k] = 0.0;
                }
            }
            else {
                for (++d; d<Diag(i+1); ++d) {
                    int j = Column(d);
                    int nc = Dof(j);
                    int e = Entry(d) + nr * nc;
                    for (int l=0; l<nc; ++l) 
                        if (Dirichlet[Index(j)+l])
                            A[e+l*nr+k] = 0.0;
                }
            }
        }
    }
}

void Matrix::EliminateDirichlet () { 
    const bool* Dirichlet = u.D();
    Scalar* A = (*this)();
    int ind = 0;

    for (int i=0; i<nR(); ++i) {
        int nr = Dof(i);
        for (int k=0; k<nr; ++k) {
	    int d = Diag(i);
            if (Dirichlet[ind++]) {
                int e = Entry(d); 
                A[e+k*nr+k] = 1.0;
                for (int l=0; l<nr; ++l) 
                    if (l != k)
                        A[e+k*nr+l] = 0.0;
                for (++d; d<Diag(i+1); ++d) {
                    int j = Column(d);
                    int nc = Dof(j);
                    e = Entry(d); 
                    for (int l=0; l<nc; ++l) {
                        A[e+k*nc+l] = 0.0;
                    }
                    e += nr * nc;
                    for (int l=0; l<nc; ++l)
                        if (Dirichlet[Index(j)+l]) {
                            A[e+l*nr+k] = 0.0;
                        } else A[e+l*nr+k] = 0.0;
                }
            }
            else {
                for (++d; d<Diag(i+1); ++d) {
                    int j = Column(d);
                    int nc = Dof(j);
                    int e = Entry(d);
                    for (int l=0; l<nc; ++l) {
                        if (Dirichlet[Index(j)+l])
                        A[e+k*nc+l] = 0.0;
                    }
                    e += nr * nc;
                    for (int l=0; l<nc; ++l) 
                        if (Dirichlet[Index(j)+l])
                            A[e+l*nr+k] = 0.0;
                }
            }
        }
    }
}


void Matrix::Symmetric () { 
    Scalar* A = (*this)();
    for (int i=0; i<nR(); ++i) {
        int n = Dof(i);
        int k=Diag(i);
	int e = Entry(k);
	for (int i0=0; i0<n; ++i0) 
	    for (int i1=0; i1<n; ++i1) 
		A[e+i1*n+i0] = A[e+i0*n+i1];
        for (int k=Diag(i)+1; k<Diag(i+1); ++k) {
            int e = Entry(k);
            int j = Column(k);
            int m = Dof(j);
            for (int i0=0; i0<n; ++i0) {
                for (int i1=0; i1<m; ++i1) {
                    A[e+n*m+i1*m+i0] = A[e+i0*n+i1];
                }
            }
        }
    }
}

ostream& operator << (ostream& s, const Matrix& u) {
	return s << u.BasicVector::ref(); }

constAB<Operator,Vector> operator * (const Matrix& A,const Vector& v) {
    return constAB<Operator,Vector>(A,v); }


void RowEntries::fill (Matrix& A, const rows& R) {
    for (int i=0; i<R.size(); ++i) {
        n[i] = R[i].n();
        for (int j=0; j<R.size(); ++j) a[i][j] = A(R[i],R[j]);
    }
}

RowEntries::RowEntries (Matrix& A, const rows& R) { fill(A,R); }
RowEntries::RowEntries (Matrix& A, const cell& c) { rows R(A,c); fill(A,R); }

