// file: Sparse.h
// author: Christian Wieners
// $Header: /public/M++/src/Sparse.h,v 1.16 2009-10-26 15:23:10 maurer Exp $

#ifndef _SPARSE_H_
#define _SPARSE_H_

#include "Tensor.h" 
#include "Algebra.h" 

#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>

class BasicSparseMatrix {
    Scalar* a;
    int* d;
    int* col;
    int n;
public:
    BasicSparseMatrix (int _n, int m) : n(_n) { 
	if (n == 0) return;
	a = new Scalar [m];
	d = new int [n+1];
	col = new int [m];
	d[n] = m;
    }
    void set_zero () {
	int m = d[n];
	for (int i=0;i<m;++i) {
	    a[i] = 0.0;
	    col[i] = 0;
	}
	for (int i=0;i<n+1;++i)
	    d[i] = 0;
	d[n] = m;
    }
    void resize (int _n, int m) {
	if (n != 0) {
	    delete[] a;
	    delete[] d;
	    delete[] col;
	}
	n = _n;
	a = new Scalar [m];
	d = new int [n+1];
	col = new int [m];
	d[n] = m;
    }
    ~BasicSparseMatrix () {
	if (n == 0) return; 
	delete[] a;
	delete[] d;
	delete[] col;
    }
    const Scalar* nzval () const { return a; }
    const int* rowind () const { return d; }
    const int* colptr () const { return col; }
    Scalar* nzval () { return a; }
    int* rowind () { return d; }
    int* colptr () { return col; }
    Scalar nzval (int i) const { return a[i]; }
    int rowind (int i) const { return d[i]; }
    int colptr (int i) const { return col[i]; }
    Scalar& nzval (int i) { return a[i]; }
    int& rowind (int i) { return d[i]; }
    int& colptr (int i) { return col[i]; }
    int size () const { return n; }
    int Size () const { return d[n]; }
    void Diagonal (Scalar* D) const {
	for (int i=0; i<size(); ++i) 
	    D[i] = a[d[i]];
    }
    void SetDiagonal (const Scalar* D) {
	for (int i=0; i<size(); ++i) 
	    a[d[i]] = D[i];
    }
    void CheckDiagonal () {
	for (int i=0; i<size(); ++i) 
	    if (a[d[i]] == Scalar(0.0))
		a[d[i]] = 1;
    }
    void multiply_plus (Scalar* b, const Scalar* u) const {
	for (int i=0; i<n; ++i) 
	    for (int k=d[i]; k<d[i+1]; ++k)
		b[i] += a[k] * u[col[k]];
    }
    int find (int i, int j) const {
	for (int e=d[i]; e<d[i+1]; ++e) 
	    if (j == col[e]) 
		return e;
	return -1;
    }
    void Decompose () {
	for (int i=0; i<n; ++i) {
	    Scalar D = (a[d[i]] = 1.0 / a[d[i]]);
	    for (int e=d[i]+1; e<d[i+1]; ++e) {
		int j = col[e];
		if (j < i) continue;
		int f = find(j,i);
		Scalar s = - (a[f] *= D);
		for (int f=d[j]+1; f<d[j+1]; ++f) {
		    int k = col[f];
		    if (k < i) continue;
		    int g = find(i,k);
		    if (g!=-1)
			a[f] += s * a[g];
		}
	    }
	}
    }
    void Solve (Scalar* u, const Scalar* b) const {
	u[0] = b[0];
	for (int i=1; i<n; ++i) {
	    u[i] = b[i];
	    for (int e=d[i]+1; e<d[i+1]; ++e) {
		int j = col[e];
		if (j>i) continue;
		u[i] -= a[e] * u[j];
	    }
	}
	u[n-1] *= a[d[n-1]];
	for (int i=n-2; i>= 0; --i) {
	    for (int e=d[i]+1; e<d[i+1]; ++e) {
		int j = col[e];
		if (j<i) continue;
		u[i] -= a[e] * u[j];
	    }
	    u[i] *= a[d[i]];
	}
    }
    void DecomposeIC () {
	for (int k=0; k<n; ++k) {
	    Scalar s = a[d[k]];
	    for (int j=0; j<k; ++j) {
		int kj = find(k,j);
		if (kj == -1) continue;
		s -= a[kj] * conj(a[kj]);
	    }
	    s = (a[d[k]] = 1.0 / sqrt(real(s)));
	    for (int i=k+1; i<n; ++i) {	    
		int ik = find(i,k);
		if (ik == -1) continue;
		for (int j=0; j<k; ++j) {
		    int ij = find(i,j);
		    if (ij == -1) continue;
		    int kj = find(k,j);
		    if (kj == -1) continue;
		    a[ik] -= a[ij]*conj(a[kj]);
		}
		a[ik] *= s;
	    }
	}
    }
    void SolveIC (Scalar* u, const Scalar* b) const {
	u[0] = b[0] * a[d[0]];
	for (int i=1; i<n; ++i) {
	    u[i] = b[i];
	    for (int e=d[i]+1; e<d[i+1]; ++e) {
		int j = col[e];
		if (j>i) continue;
		u[i] -= a[e] * u[j];
	    }
	    u[i] *= a[d[i]];
	}
	for (int i=n-1; i>0; --i) {
	    u[i] *= a[d[i]];
	    for (int e=d[i]+1; e<d[i+1]; ++e) {
		int j = col[e];
		if (i<j) continue;
		u[j] -= conj(a[e]) * u[i];
	    }
	}
	u[0] *= a[d[0]];
    }
    void GaussSeidel (Scalar* u, const Scalar* b) const {
	for (int i=0; i<n; ++i) {
	    Scalar r = b[i], diag;
	    for (int k=d[i]; k<d[i+1]; ++k) {
		int j = col[k];
		if (j < i)
		    r -= a[k] * u[j];
                else
                   if (j==i) diag = a[k];
	    }
	    u[i] = r / diag;
	}
    }
    void BackwardGaussSeidel (Scalar* u, const Scalar* b) const {
	for (int i=n-1; i>=0; --i) {
	    Scalar r = b[i], diag;
	    for (int k=d[i+1]-1; k>=d[i]; --k) {
		int j = col[k];
		if (j > i)
		    r -= conj(a[k]) * u[j];
                else
                   if (j==i) diag = a[k];
	    }
	    u[i] = r / diag;
	}
    }
    void GaussSeidel (Scalar* u, const Scalar* b, const Scalar* D) const {
	for (int i=0; i<n; ++i) {
	    Scalar r = b[i];
	    for (int k=d[i]+1; k<d[i+1]; ++k) {
		int j = col[k];
		if (j < i)
		    r -= a[k] * u[j];
	    }
	    u[i] = r * D[i];
	}
    }
    void BackwardGaussSeidel(Scalar* u,const Scalar* b,const Scalar* D) const {
	for (int i=n-1; i>=0; --i) {
	    Scalar r = b[i];
	    for (int k=d[i]+1; k<d[i+1]; ++k) {
		int j = col[k];
		if (j > i)
		    r -= a[k] * u[j];
	    }
	    u[i] = r * D[i];
	}
    }
    friend ostream& operator << (ostream& s, const BasicSparseMatrix& M) {
	for (int i=0; i<M.size(); ++i) {
	    int k=M.rowind(i);
	    for (int k=M.rowind(i); k<M.rowind(i+1); ++k)
		s << " " << "(" << M.colptr(k) << "," << M.nzval(k) << ")";
	    s << endl;
	}
	return s; 
    }
};

//class Matrix;

class SmallMatrix;
class SparseMatrix : public BasicSparseMatrix, public Operator {
    int size_ (const SparseMatrix& M, const vector<bool>& mask) const {
	int n = 0;
	for (int i=0; i<M.size(); ++i)
	    if (mask[i]) ++n;
	return n;
    }
    int Size_ (const SparseMatrix& M, const vector<bool>& mask) const {
	int n = 0;
	for (int i=0; i<M.size(); ++i)
	    if (mask[i]) 
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k)
		    if (mask[M.colptr(k)]) ++n;
	return n;
    }
    int Size_ (const SparseMatrix& M, const vector<int>& rindex) const {
	int n = 0;
	for (int i=0; i<M.size(); ++i)
	    if (rindex[i] != -1) 
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k)
		    if (rindex[M.colptr(k)] != -1) ++n;
	return n;
    }
    int int_in_vector(int x, const vector<int>& y) const {
	if (x == -1) return -1;
	for (int i = 0; i < y.size(); ++i)
	    if (y[i] == x) return i;
	return -1;
    }
    int _Size (const SparseMatrix& M, const vector<int>& Indices) const { 
	int n = 0;
	for (int i = 0; i < Indices.size(); ++i)
	    for (int k = M.rowind(Indices[i]); k < M.rowind(Indices[i]+1); ++k)
		if (int_in_vector(M.colptr(k),Indices) != -1)
		    n++;
	return n;
    }
    typedef map<int,Scalar> SparseVecT;
    typedef SparseVecT::iterator SparVecIt;
    typedef SparseVecT::const_iterator conSparVecIt;
    typedef vector<SparseVecT> SparseMatT;
    typedef vector<int> DofDofId;
    DofDofId ParDofId, ParDofRes;
    
    void convert_sm (SparseMatT&) const;
    void print_sm (const SparseMatT&) const;
    void print_sm_dense (const SparseMatT&) const;
    void print_id_dense (const DofDofId&) const;
    void build_ident (const Matrix&);
    void apply_PC_mat (SparseMatT&, const SparseMatT&) const;
 public:
    void convert_sm_back (const SparseMatT&);
    SparseMatrix (int n, int m) : BasicSparseMatrix(n,m) {}
    SparseMatrix (const Matrix& M) : 
	BasicSparseMatrix(M.size(),M.Size()) {
	  M.copy(nzval(),rowind(),colptr());
    }
    SparseMatrix (const SparseMatrix&); 
    SparseMatrix (const SparseMatrix& M, const vector<bool>& mask) 
	: BasicSparseMatrix(size_(M,mask),Size_(M,mask)) {
	vector<int> index(M.size());
	int* d = rowind();
	int n = 0;
	d[0] = 0;
	for (int i=0; i<M.size(); ++i)
	    if (mask[i]) {
		index[i] = n; 
		++n;
		d[n] = d[n-1];
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k) {
		    int j = M.colptr(k);
		    if (mask[j]) ++(d[n]);
		}
	    }
	int m = 0;
	for (int i=0; i<M.size(); ++i)
	    if (mask[i]) 
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k) {
		    int j = M.colptr(k);
		    if (!mask[j]) continue;
		    nzval(m) = M.nzval(k);
		    colptr(m) = index[j];
		    ++m;
		}
    }
    SparseMatrix (const SparseMatrix& M, const vector<int>& Indices)
      : BasicSparseMatrix(Indices.size(),_Size(M,Indices)) {
	int* d = rowind();
	int m = 0;
	d[0] = 0;
	for (int i = 0; i < Indices.size(); ++i) {
	    d[i+1] = d[i];
	    for (int k=M.rowind(Indices[i]); k<M.rowind(Indices[i]+1); ++k) {
	        int akt = int_in_vector(M.colptr(k),Indices);
		if (akt != -1) {
		    colptr(m) = akt;
		    nzval(m) = M.nzval(k);
		    ++d[i+1];
		    m++;
		}
	    }
	}
    }
    SparseMatrix (const SparseMatrix& M, 
		  const vector<int>& index, 
		  const vector<int>& rindex) 
	: BasicSparseMatrix(index.size(),Size_(M,rindex)) {
	int* d = rowind();
	int n = 0;
	d[0] = 0;
	for (int i=0; i<M.size(); ++i)
	    if (rindex[i] != -1) {
		++n;
		d[n] = d[n-1];
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k) {
		    int j = M.colptr(k);
		    if (rindex[j] != -1) ++(d[n]);
		}
	    }
	int m = 0;
	for (int i=0; i<M.size(); ++i)
	    if (rindex[i] != -1) 
		for (int k=M.rowind(i); k<M.rowind(i+1); ++k) {
		    int j = M.colptr(k);
		    if (rindex[j] == -1) continue;
		    nzval(m) = M.nzval(k);
		    colptr(m) = rindex[j];
		    ++m;
		}
    }
    SparseMatrix (const SmallMatrix&);
    void multiply_plus (Vector& b, const Vector& u) const {
	BasicSparseMatrix::multiply_plus (b(),u());
	Collect(b);
    }
    void pc_mat_convert (const Matrix&);
    void ShrinkIdentify (Vector&, const Vector&) const;
    void ExpandIdentify (Vector&) const;
};
inline constAB<Operator,Vector> 
operator * (const SparseMatrix& S, const Vector& v) {
    return constAB<Operator,Vector>(S,v); 
}

class SparseSolver {
public:
    virtual void Solve (Scalar *) {};
    virtual void Solve (Scalar*, int) {}
    virtual void Solve (Scalar *, const Scalar*) {};
    virtual ~SparseSolver () {}
    virtual void test_output() {};
};
SparseSolver* GetSparseSolver (BasicSparseMatrix&);
SparseSolver* GetSparseSolver (BasicSparseMatrix&, const string&);

#endif
