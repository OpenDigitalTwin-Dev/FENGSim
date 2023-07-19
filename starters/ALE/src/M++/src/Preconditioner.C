// file: Preconditioner.C
// author: Christian Wieners
//         Wolfgang Mueller (SuperLU_local)
// $Header: /public/M++/src/Preconditioner.C,v 1.64 2009-10-16 13:35:17 wieners Exp $

#include "Sparse.h"
#include "Small.h"
#include "Schur.h"
#include "Preconditioner.h"
#include "Interface.h"
#include "Transfer.h"
#include "LinearSolver.h"
#include "DD.h"
#include "MatrixGraph.h"

#include <set>

class Jacobi : public Preconditioner {
 protected:
    Vector* D;
    double theta;
    bool shift_special;
public:
  Jacobi () : D(0), theta(1), shift_special(false) {
      ReadConfig(Settings,"PreconditionerDamp",theta);
      int spec = 0;
      ReadConfig(Settings,"SHIFTspecial",spec);
      if (spec == 0) shift_special = false;
      else shift_special = true;
    }
    void Construct (const Matrix& A) {
	D = new Vector(A);
	const Scalar* a = A();
	for (int i=0; i<A.nR(); ++i) {
	    int d = A.Diag(i);
	    int n = A.Dof(i);
	    for (int k=0; k<n; ++k) 
		for (int l=0; l<n; ++l, ++a) 
		    if (k==l) (*D)(i,k) = *a;
	    for (++d; d<A.Diag(i+1); ++d) {
		int j = A.Column(d);
		int m = A.Dof(j);
		a += 2*m*n;
	    }
	}
	Accumulate(*D);
	vout(10) << "diag " << *D;
	for (int i=0; i<D->size(); ++i) {
	    if ( (*D)[i] == 0.0)
		(*D)[i] = 1.0;
	    else
	        (*D)[i] = 1.0 / (*D)[i];
	}
    }
    void Destruct () { if (D) delete D; D = 0; }
    virtual ~Jacobi () { Destruct(); }
    virtual void multiply (Vector& u, const Vector& b) const {
        for (int i=0; i<D->size(); ++i) {
	  u[i] = (*D)[i] * b[i];
	}
        u *= theta;
	Accumulate(u);
    }
    virtual string Name () const { return "Jacobi"; }
    friend ostream& operator << (ostream& s, const Jacobi& Jac) {
	return s << *(Jac.D); }
};

class JacobiMixed : public Preconditioner {
 protected:
    Vector* D;
    double theta;
    bool shift_special;
public:
  JacobiMixed () : D(0), theta(1), shift_special(false) {
      ReadConfig(Settings,"PreconditionerDamp",theta);
      int spec = 0;
      ReadConfig(Settings,"SHIFTspecial",spec);
      if (spec == 0) shift_special = false;
      else shift_special = true;
    }
    void Construct (const Matrix& A) {
	D = new Vector(A);
	const Scalar* a = A();
	for (int i=0; i<A.nR(); ++i) {
	    int d = A.Diag(i);
	    int n = A.Dof(i);
	    for (int k=0; k<n; ++k) 
		for (int l=0; l<n; ++l, ++a) 
		    if (k==l) (*D)(i,k) = *a;
	    for (++d; d<A.Diag(i+1); ++d) {
		int j = A.Column(d);
		int m = A.Dof(j);
		a += 2*m*n;
	    }
	}
	Accumulate(*D);
	vout(10) << "diag " << *D;
	for (int i=0; i<D->size(); ++i) {
	    if ( (*D)[i] == 0.0)
		(*D)[i] = 1.0;
	    else
	        (*D)[i] = 1.0 / (*D)[i];
	}
	for (row r = (*D).rows(); r != (*D).rows_end(); r++)
	    (*D)(r,(*D).dim()) = 1.0;
    }
    void Destruct () { if (D) delete D; D = 0; }
    virtual ~JacobiMixed () { Destruct(); }
    virtual void multiply (Vector& u, const Vector& b) const {
        for (int i=0; i<D->size(); ++i) {
	  u[i] = (*D)[i] * b[i];
	}
        u *= theta;
	Accumulate(u);
    }
    virtual string Name () const { return "JacobiMixed"; }
    friend ostream& operator << (ostream& s, const JacobiMixed& Jac) {
	return s << *(Jac.D); }
};


class PointBlockJacobi : public Preconditioner {
 protected:
    Matrix* D;
    double theta;
 public:
    PointBlockJacobi () : D(0), theta(1) { 
	ReadConfig(Settings,"PreconditionerDamp",theta); 
    }
    void invert (int n, Scalar* a) const { 
	int* ipv = new int [n];
	Scalar* Mat = new Scalar [n*n];
	Scalar* rhs = new Scalar [n];
	for (int i=0; i<n; ++i) ipv[i] = i;
	for (int i=0; i<n*n; ++i) Mat[i] = a[i];
	for (int i=0; i<n; ++i) {
	    int k = i;
	    double piv = abs(Mat[i*n+i]);
	    for (int j=i+1; j<n; ++j) {
		double sum = abs(Mat[j*n+i]);
		if (sum > piv) {
		    k = j;
		    piv = sum;
		}
	    }
	    if (k != i) {
		swap(ipv[i],ipv[k]);
		for (int j=0; j<n; ++j) swap(Mat[k*n+j],Mat[i*n+j]);
	    }
	    Scalar dinv = Mat[i*n+i];
	    if (abs(dinv)<1e-12) Exit("Error in SmallMatrix Inversion");
	    dinv = Mat[i*n+i] = 1.0/dinv;
	    for (int j=i+1; j<n; ++j) {
		Scalar piv = (Mat[j*n+i] *= dinv);
		for (int k=i+1; k<n; ++k)
		    Mat[j*n+k] -= Mat[i*n+k] * piv;
	    }
	}
	for (int k=0; k<n; ++k) {
	    for (int i=0; i<n; ++i)
		rhs[i] = 0;
	    rhs[k] = 1.0;
	    for (int i=0; i<n; ++i) {
		Scalar sum = rhs[ipv[i]];
		for (int j=0; j<i; ++j)
		    sum -= Mat[i*n+j] * a[j*n+k];
		a[i*n+k] = sum;	// Lii = 1 
	    }
	    for (int i=n-1; i>=0; i--) {
		Scalar sum = a[i*n+k];
		for (int j=i+1; j<n; ++j)
		    sum -= Mat[i*n+j] * a[j*n+k];
		a[i*n+k] = sum * Mat[i*n+i];	// Uii = Inv(Mii) 
	    }
	}	
	delete[] rhs;
	delete[] Mat;
	delete[] ipv;
    }
    void apply (int n, Scalar* u, const Scalar* a, const Scalar* b) const { 
	for (int i=0; i<n; ++i) {
	    Scalar s = 0;
	    for (int k=0; k<n; ++k)
		s += a[i*n+k] * b[k];
	    u[i] = s;
	}
    }
    void Construct (const Matrix& A) {
	D = new Matrix(A);
	Accumulate(*D);
	Scalar* d = (*D)();
	const Vector& v = A.GetVector();
	for (int i=0; i<A.nR(); ++i) {
	    int e = A.Entry(A.Diag(i));
	    int n = A.Dof(i);
	    for (int k=0; k<n; ++k) 
		if (d[e+n*k*k] == 0.0) d[e+n*k*k] = 1;
	}
	for (int i=0; i<A.nR(); ++i) {
	    int e = A.Entry(A.Diag(i));
	    int n = A.Dof(i);
	    for (int k=0; k<n; ++k) 
		if (d[e+n*k*k] == 0.0) d[e+n*k*k] = 1;
	    invert(n,d+e);
//	    for (int k=0; k<n; ++k) 
//		invert(1,d+e+k*n+k);
	}
    }
    void Destruct () { if (D) delete D; D = 0; }
    virtual ~PointBlockJacobi () { Destruct(); }
    virtual void multiply (Vector& u, const Vector& b) const {
	const Scalar* a = (*D)();
	for (int i=0; i<u.nR(); ++i) {
	    int e = u.Entry(u.Diag(i));
	    int n = u.Dof(i);
	    apply(n,u(i),a+e,b(i));
//	    for (int k=0; k<n; ++k) 
//		apply(1,u(i)+k,a+e+k*n+k,b(i)+k);
//		u(i,k) = a[e+k*n+k] * b(i,k);
	}
	u *= theta;
	Accumulate(u);
    }
    virtual string Name () const { return "PointBlockJacobi"; }
    friend ostream& operator << (ostream& s, const PointBlockJacobi& Jac) {
	return s << *(Jac.D); }
};

class PointBlockGaussSeidel : public PointBlockJacobi {
 public:
    PointBlockGaussSeidel () {}
    virtual ~PointBlockGaussSeidel () { Destruct(); }
    virtual void multiply (Vector& u, const Vector& b) const {
	Vector r(b);
	const Scalar* a = (*D)();
        for (int i=0; i<u.nR(); ++i) {
            int d = u.Diag(i);
            int n = u.Dof(i);
            int e = u.Entry(d);
            for (int k=0;k<n;++k) {
                int d = u.Diag(i);
                Scalar R = b(i,k);
                for (int l=0; l < n; ++l)
                    if (k > l) R -= a[e+k*n+l] * u(i,l);
                for (d++; d < u.Diag(i+1); ++d) {
                    int e = u.Entry(d);
                    int j = u.Column(d);
                    int m = u.Dof(j);
                    for (int l=0; l < m; ++l)
                        R -= a[e+k*n+l] * u(j,l);
                }
                r(i,k) = R;
            }
	    apply(n,u(i),a+e,r(i));
	}
	Accumulate(u);
    }
    virtual string Name () const { return "PointGaussSeidel"; }
    friend ostream& operator << (ostream& s, const PointBlockGaussSeidel& P) {
	return s << *(P.D); }
};

class DampedJacobi : public Jacobi {
    double damp;
 public:
    DampedJacobi () {
        damp = 1.0;
	ReadConfig(Settings,"PreconditionerDamp",damp);        
    }
    virtual ~DampedJacobi () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	for (int i=0; i<D->size(); ++i) u[i] = damp * (*D)[i] * b[i];
	Accumulate(u);
    }
    string Name () const { return "DampedJacobi"; }
};

class NoPreconditioner : public Preconditioner {
 public:
    NoPreconditioner ()  {}
    virtual ~NoPreconditioner () {}
    void Construct (const Matrix& A) {}
    void Destruct () {}
    void multiply (Vector& u, const Vector& b) const {
        u = b;
        Accumulate(u);
    }
    string Name () const { return "NoPreconditioner"; }
};

class GaussSeidel : public Preconditioner {
    Matrix* A;
    SparseMatrix* Sp;
public:
    GaussSeidel () : Sp(0), A(0) {}
    void Destruct () { 
	if (Sp) delete Sp; Sp = 0; 
	if (A) delete A; A = 0; 
    }
    void Construct (const Matrix& _A) {
	Destruct();
	A = new Matrix(_A);
	Accumulate(*A);
	Sp = new SparseMatrix(*A);
    }
    virtual ~GaussSeidel () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	Vector r(b);
	Sp->GaussSeidel(u(),r());
        Accumulate(u);     
    }
    void multiply_transpose (Vector& u, const Vector& b) const { 
	Vector r(b);
        Accumulate(r);        
	Sp->BackwardGaussSeidel(u(),r());
        MakeAdditive(u);     
        Accumulate(u);                
    }
    virtual string Name () const { return "GaussSeidel"; }
    friend ostream& operator << (ostream& s, const GaussSeidel& GS) {
	return s << *(GS.Sp); }
};

class FlyingCow : public Preconditioner {
protected:
    Vector* D;
    const Matrix* A;    
public:
    FlyingCow () : D(0), A(0) {}
    void Construct (const Matrix& _A) {
        A = new Matrix(_A);
	D = new Vector(*A);
	const Scalar* a = (*A)();
	for (int i=0; i<A->nR(); ++i) {
	    int d = A->Diag(i);
	    int n = A->Dof(i);
	    for (int k=0; k<n; ++k) 
		for (int l=0; l<n; ++l, ++a) 
		    if (k==l) (*D)(i,k) = *a;
	    for (++d; d<A->Diag(i+1); ++d) {
		int j = A->Column(d);
		int m = A->Dof(j);
		a += 2*m*n;
	    }
	}
	Accumulate(*D);        
	dout(10) << "diag " << *D;
	for (int i=0; i<D->size(); ++i) (*D)[i] = 1.0 / (*D)[i];
    }
    void Destruct () {
        if (D) delete D; D = 0;
        if (A) delete A; A = 0;
    }
    virtual ~FlyingCow () { Destruct(); }
    void forward (Vector& u, const Vector& b) const {
        for (int i=0; i < A->nR(); ++i) {
            int d = A->Diag(i);
            int n = A->Dof(i);
            int e = A->Entry(d);
            for (int k=0;k<n;++k) {
                int d = A->Diag(i);
                Scalar R = b(i,k);
                for (int l=0; l < n; ++l)
                    if (k > l) R -= (*A)[e+k*n+l] * u(i,l);
                for (d++; d < A->Diag(i+1); ++d) {
                    int e = A->Entry(d);
                    int j = A->Column(d);
                    int m = A->Dof(j);
                    for (int l=0; l < m; ++l)
                        R -= (*A)[e+k*n+l] * u(j,l);
                }
                u(i,k) = R*(*D)(i,k);
            }
        }
        Accumulate(u);        
    }
    void backward (Vector& u, const Vector& b) const {
        for (int i=A->nR()-1; i > -1; --i) {
            int d = A->Diag(i);
            int n = A->Dof(i);
            int e = A->Entry(d);
            for (int k=n-1;k>-1;--k) {
                Scalar R = b(i,k);
                for (int l=n-1; l > -1; --l)
                    if (k < l) R -= (*A)[e+k*n+l] * u(i,l);
                for (int p = i+1; p < A->nR(); ++p) {
                    for (int q = A->Diag(p)+1; q < A->Diag(p+1); ++q) {
                        int j = A->Column(q);
                        if (j != i) continue;
                        int e = A->Entry(q);
                        int m = A->Dof(j);
                        for (int l=m-1; l > -1; --l)
                            R -= (*A)[e+m*n+k*n+l] * u(p,l);
                    }
                }
                u(i,k) = R*(*D)(i,k);
            }
        }
        Accumulate(u);                
    }
    void SymmetricGaussSeidel (Vector& u, Vector r) const {
        backward(u,r);
        r -= (*A) * u;
        Vector c(u);
        forward(c,r);
        u += c;
    }
    virtual void multiply (Vector& u, const Vector& b) const {
        Vector r(b);
        SymmetricGaussSeidel(u,r);
    }
    virtual string Name () const { return "FlyingCow"; }
    friend ostream& operator << (ostream& s, const FlyingCow& Jac) {
	return s << *(Jac.D); }
};

//#ifdef UMFSOLVER
class UMF : public Preconditioner {
    SparseMatrix* S;
    SparseSolver* Sol;
 public:
    UMF () : S(0), Sol(0) {}
    void Construct (const Matrix& _A) {
	Date Start; 
	Matrix A(_A);
	if (!A.identify() ) Accumulate(A);
	S = new SparseMatrix(A);
	S->CheckDiagonal();
        if (A.identify() ) S->pc_mat_convert(A);
	Sol = GetSparseSolver(*S,"UMFSolver");
	Vout(2) << "   decompose UMF " << Date() - Start << endl;
    }
    void Destruct () {
	if (S) delete S;     S = 0;
	if (Sol) delete Sol; Sol = 0;
    }
    virtual ~UMF () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	Date Start; 
	if (PPM->size() > 1) Average(u);
        else
	    if (u.identify() ) S->ShrinkIdentify(u,b); 
	Sol->Solve(u(),b());
	if (PPM->size() > 1) Accumulate(u);
        else
	    if (u.identify() ) S->ExpandIdentify(u); 
	Vout(2) << "   solve UMF " << Date() - Start << endl;
    }
    string Name () const { return "UMF"; }
    friend ostream& operator << (ostream& s, const UMF& _UMF) {
	return s << "SuperLU"; }
};
//#endif

class SuperLU : public Preconditioner {
    SparseMatrix* S;
    SparseSolver* Sol;
 public:
    SuperLU () : S(0), Sol(0) {}
    void Construct (const Matrix& _A) {
	Date Start; 
	Matrix A(_A);
	if (!A.identify() ) Accumulate(A);
	S = new SparseMatrix(A);
	S->CheckDiagonal();
        if (A.identify() ) S->pc_mat_convert(A);
	Sol = GetSparseSolver(*S,"SuperLU");
	Vout(2) << "   decompose SuperLU " << Date() - Start << endl;
    }
    void Destruct () {
	if (S) delete S;     S = 0;
	if (Sol) delete Sol; Sol = 0;
    }
    virtual ~SuperLU () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
      u = b;
      Sol->Solve(u());
      return;

	Date Start; 
	u = b;
	if (PPM->size() > 1) Average(u);
        else
	  if (u.identify() ) S->ShrinkIdentify(u,b); 
	Sol->Solve(u());
	if (PPM->size() > 1) Accumulate(u);
        else
           if (u.identify() ) S->ExpandIdentify(u); 
	Vout(2) << "   solve SuperLU " << Date() - Start << endl;
    }
    string Name () const { return "SuperLU"; }
    friend ostream& operator << (ostream& s, const SuperLU& SLU) {
	return s << "SuperLU"; }
};

class SuperLU_local : public Preconditioner {
    SparseMatrix* S;
    SparseSolver* Sol;
    typedef hash_map<Point,int,Hash>::iterator Iterator;
    typedef hash_map<Point,int,Hash>::const_iterator ConstIterator;
    hash_map<Point,set<short>,Hash>* IProc;
    Vector* rhs;
    bool shift_special;
public:
    SuperLU_local () : S(0), Sol(0), rhs(0), IProc(0) {
	int spec = 0;
	ReadConfig(Settings,"SHIFTspecial",spec);
	if (spec == 0) shift_special = false;
	else shift_special = true;
        bool Overlap = false;
        ReadConfig(Settings,"Overlap_Distribution",Overlap);
        if (!Overlap) Exit("SuperLU_local failed: Set Overlap_Distribution = 1");
    }
private:
    void CommunicateMatrix (Matrix& A) {
	Date Start_matrix;
	ExchangeBuffer E;
	Scalar* a = A();
	if (!PPM->master()) {
	    for (row r = A.rows(); r != A.rows_end(); ++r) {
		int id = r.Id();
		int d = A.Entry(A.Diag(id));
		int n = A.Dof(id);
		E.Send(0) << r();
		for (short k=0; k<n; ++k)
		    for (short l=0; l<n; ++l)
			E.Send(0) << a[d+k*n+l];
		E.Send(0) << int(r.size());
		for (entry e = r.entries(); e != r.entries_end(); ++e) {
		    E.Send(0) << e();
		    int m = 2 * n * A.Dof(e.Id());
		    int dd = e.GetEntry();
		    for (int j=0; j<m; ++j)
			E.Send(0) << a[dd+j];
		}
	    }
	}
	E.Communicate();
	for (short q=0; q<PPM->size(); ++q) {
	    Scalar tmp;
	    while (E.Receive(q).size() < E.ReceiveSize(q)) {
		Point x;
		E.Receive(q) >> x;
		row r = A.find_row(x);
		int id = r.Id();
		int d = A.Entry(A.Diag(id));
		int n = A.Dof(id);
		for (int j=0; j<n*n; ++j) {
		    E.Receive(q) >> tmp;
		    a[d+j] += tmp;
		}
		int s;
		E.Receive(q) >> s;
		for (int k=0; k<s; ++k) {
		    Point y;
		    E.Receive(q) >> y;
		    entry e = r.find_entry(y);
		    int dd = e.GetEntry();
		    int m = 2 * n * A.Dof(e.Id());
		    for (int j=0; j<m; ++j) {
			E.Receive(q) >> tmp;
			a[dd+j] += tmp;
		    }
		}
	    }
	}
	tout(3) << "   SuperLU_local: Communicate Matrix " 
		<< Date() - Start_matrix << endl;
    }
    void CreateIProc (const Vector& u) {
	Date Start;
	IProc = new hash_map<Point,set<short>,Hash>;
	ExchangeBuffer E;
	for (row r = u.rows(); r != u.rows_end(); ++r) {
	    E.Send(0) << r() << r.n();
	    for (int k=0;k<r.n();++k)
		E.Send(0) << u(r,k);
	}
	E.Communicate();
	for (short q=0; q<PPM->size(); ++q)  
	    while (E.Receive(q).size() < E.ReceiveSize(q)) {
		Point P;
		int dofs;
		E.Receive(q) >> P >> dofs;
		hash_map<Point,set<short>,Hash>::iterator IProciter = IProc->find(P);
		if (IProciter == IProc->end()) {
		    set<short> tmp;
		    tmp.insert(q);
		    (*IProc)[P] = tmp;
		}
		else {
		    (IProciter->second).insert(q);
		}
		for (int k=0; k<dofs; ++k) {
		    Scalar e;
		    E.Receive(q) >> e;
		}
	    }
	tout(3) << "   SuperLU_local: Create IProc " << Date() - Start << endl;
    }
    void CollectResidual (const Vector& b) const {
	Date Start;
	ExchangeBuffer E;
	for (row r = b.rows(); r != b.rows_end(); ++r) {
	    E.Send(0) << r() << r.n();
	    for (int k=0;k<r.n();++k)
		E.Send(0) << b(r,k);
	}
	
	E.Communicate();

	(*rhs) = 0.0;
	
	for (short q=0; q<PPM->size(); ++q)  
	    while (E.Receive(q).size() < E.ReceiveSize(q)) {
		Point P;
		int dofs;
		E.Receive(q) >> P >> dofs;
		row r = rhs->find_row(P);
		for (int k=0; k<dofs; ++k) {
		    Scalar e;
		    E.Receive(q) >> e;
		    (*rhs)(r,k) += e;
		}
	    }
	tout(4) << "   SuperLU_local: Collect Residual " << Date() - Start << endl;
    }
    void DistributeSolution (Vector& u) const {
	Date Start;
	u = 0.0;
	ExchangeBuffer E;
	if (PPM->master()) {
	    /*
	    for (row r = rhs->rows(); r != rhs->rows_end(); ++r) {
		procset p = u.find_procset(r());
		if (p==u.procsets_end()) continue;
		for (short s=0;s<p.size();++s) {
		    E.Send(p[s]) << r() << r.n();
		    for (int dof = 0; dof < r.n(); ++dof)
			E.Send(p[s]) << (*rhs)(r,dof);
		}
	    }
	    */
	    for (row r = rhs->rows(); r != rhs->rows_end(); ++r) {
		hash_map<Point,set<short>,Hash>::const_iterator iprii = IProc->find(r());
		set<short> pset = iprii->second;
		for (set<short>::const_iterator s = pset.begin(); s != pset.end(); ++s) {
		    E.Send(*s) << r() << r.n();
		    for (int dof = 0; dof < r.n(); ++dof)
			E.Send(*s) << (*rhs)(r,dof);
		}
	    }
	}
	E.Communicate();
	for (short q=0; q<PPM->size(); ++q)  
	    while (E.Receive(q).size() < E.ReceiveSize(q)) {
		Point P;
		int dofs;
		E.Receive(q) >> P >> dofs;
		row r = rhs->find_row(P);
		for (int dof = 0; dof < dofs; ++dof) {
		    Scalar e;
		    E.Receive(q) >> e;
		    u(r,dof) = e;
		}
	    }
	tout(4) << "   SuperLU_local: Distribute Solution "
		<< Date() - Start << endl;
    }
public:
    void Construct (const Matrix& _A) {
	Date Start; 
	Matrix A(_A);
	Vector u(_A);
	rhs = new Vector(u);
	CommunicateMatrix(A);
	if (PPM->master()) {
	    Date Start_convert;
	    S = new SparseMatrix(A);
            if (A.identify() ) S->pc_mat_convert(A);
	    tout(3) << "   SuperLU_local: Construct Sparse Matrix " 
		    << Date() - Start_convert << endl;
	    S->CheckDiagonal();
	    Date Start_create;
	    Sol = GetSparseSolver(*S,"SuperLU");
	    tout(2) << "   SuperLU_local: Create Solver " 
		    << Date() - Start_create << endl;
	}
	CreateIProc(u);
	tout(2) << "   SuperLU_local: Construct " << Date() - Start << endl;
    }
    void Destruct () {
	if (S) delete S;     S = 0;
	if (Sol) delete Sol; Sol = 0;
	if (rhs) delete rhs; rhs = 0;
	if (IProc) delete IProc; IProc = 0;
    }
    virtual ~SuperLU_local () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	Date Start; 
	CollectResidual(b);
	if (PPM->master()) {
	    Date Start_solve;
            if (u.identify() ) {
               Vector tmp(*rhs);
               S->ShrinkIdentify(*rhs,tmp);
            }
	    Sol->Solve((*rhs)());
            if (u.identify() ) S->ExpandIdentify(*rhs);
	    tout(4) << "   SuperLU_local: solve " << Date() - Start_solve << endl;
	}
	DistributeSolution(u);
	tout(3) << "   SuperLU_local: multiply " << Date() - Start << endl;
    }
    string Name () const { return "SuperLU_local"; }
    friend ostream& operator << (ostream& s, const SuperLU_local& SLU) {
	return s << "SuperLU_local"; }
};

#ifdef ILU_MULTILEVEL
#include "../include/ilu_ml.h"

class ILUML : public Preconditioner {
    bool ILUspecial;
protected:
    SparseMatrix *S;
    int ILUType;
    double droptol;
    ilu_ml *ilu;
public:
    ILUML () : Preconditioner(), S(0), ILUType(1000), droptol(2.0), ilu(0), ILUspecial(false) {
	ReadConfig(Settings,"ILUType",ILUType);
	ReadConfig(Settings,"droptol",droptol);
        int iluspec = 0;
	ReadConfig(Settings,"SHIFTspecial",iluspec);
        if (iluspec == 0) ILUspecial = false;
        else ILUspecial = true;
    } 
    virtual void Construct (const Matrix& _A) {
	Date Start;
	Matrix A(_A);
	Accumulate(A);
        if (ILUspecial) {
            Scalar* d = A();
            for (int i=0; i<A.nR(); ++i) {
                int e = A.Entry(A.Diag(i));
                int n = A.Dof(i);
                if (d[e] == 0.0) d[e] = 1;
            }
        }
	SparseMatrix Sp(A);
	S = new SparseMatrix(Sp);
	ilu = GetILUMLPreconditioner(S->size(),S->nzval(),S->colptr(),
				     S->rowind(),ILUType,droptol);
	tout(3) << "   decompose ILUML " << Date() - Start << endl;
    }
    void Destruct () { 
//	if (ilu) delete ilu; 
	if (S) delete S; S = 0;
    }
    virtual ~ILUML () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	u = b;
	ilu->apply(b(),u());
	Accumulate(u);
    }
    virtual string Name () const { return "ILUML"; }
    friend ostream& operator << (ostream& s, const ILUML& ILU) {
	return s << "ILUML"; }
};

#endif 


class Multigrid : public MultigridPreconditioner {    
    const Assemble& assemble;
    vector<Matrix*> AA;
    vector<Vector*> vv;
    vector<const Matrix*> A;
    vector<int> pre;
    vector<int> post;
    vector<int> cycle;
    double theta;
    const Transfer* trans;
 public:
    Multigrid (const MatrixGraphs& g, const Assemble& ass) 
	: MultigridPreconditioner(g), assemble(ass), trans(0),
	  pre(levels+1,1), post(levels+1,1), cycle(g.Level()+1,1),
	  A(levels+1), AA(levels), vv(levels),
	  theta(1) {
	int v = 1;
	if (ReadConfig(Settings,"presmoothing",v)) 
	    for (int level=0; level<=levels; ++level) pre[level] = v;
	if (ReadConfig(Settings,"postsmoothing",v)) 
	    for (int level=0; level<=levels; ++level) post[level] = v;
	string cname = "V"; ReadConfig(Settings,"cycle",cname);
	if (cname == "W") 
	    for (int level=2; level<=levels; ++level) cycle[level] = 2;
	else if (cname == "0") 
	    for (int level=0; level<=levels; ++level) cycle[level] = 0;
	else if (cname == "VW") 
	    for (int level=2; level<=levels; ++level) cycle[level] = 3;
	ReadConfig(Settings,"MultigridVerbose",verbose);
	string smoother= "Jacobi";
	ReadConfig(Settings,"Smoother",smoother);
	for (int k=0; k<levels; ++k) {
	    T[k] = 0;
	    PC[k+1] = GetPC(smoother);
	}
	ReadConfig(Settings,"SmootherDamp",theta);
	string solver = "gmres";   
	ReadConfig(Settings,"BaseSolver",solver);
	string base_pc = "Jacobi"; 
	ReadConfig(Settings,"BasePreconditioner",base_pc);
	LS = GetSolver(GetPC(base_pc),solver,"BaseSolver");
    }
    Multigrid (const MatrixGraphs& g, const Assemble& ass, const Transfer& Tr) 
	: MultigridPreconditioner(g), assemble(ass), trans(&Tr),
	  pre(levels+1,1), post(levels+1,1), cycle(g.Level()+1,1),
	  A(levels+1), AA(levels), vv(levels),
	  theta(1) {
	int v = 1;
	if (ReadConfig(Settings,"presmoothing",v)) 
	    for (int level=0; level<=levels; ++level) pre[level] = v;
	if (ReadConfig(Settings,"postsmoothing",v)) 
	    for (int level=0; level<=levels; ++level) post[level] = v;
	string cname = "V"; ReadConfig(Settings,"cycle",cname);
	if (cname == "W") 
	    for (int level=2; level<=levels; ++level) cycle[level] = 2;
	else if (cname == "0") 
	    for (int level=0; level<=levels; ++level) cycle[level] = 0;
	else if (cname == "VW") 
	    for (int level=2; level<=levels; ++level) cycle[level] = 3;
	ReadConfig(Settings,"MultigridVerbose",verbose);
	string smoother= "Jacobi";
	ReadConfig(Settings,"Smoother",smoother);
	for (int k=0; k<levels; ++k) {
	    T[k] = 0;
	    PC[k+1] = GetPC(smoother);
	}
	ReadConfig(Settings,"SmootherDamp",theta);
	string solver = "gmres";   
	ReadConfig(Settings,"BaseSolver",solver);
	string base_pc = "Jacobi"; 
	ReadConfig(Settings,"BasePreconditioner",base_pc);
	LS = GetSolver(GetPC(base_pc),solver,"BaseSolver");
    }
    void Construct (const Matrix& a) {
	A[levels] = &a;
	for (int k=0; k<levels; ++k) {
	    if (T[k]) continue;
	    if (trans) 
		T[k] = trans->GetTransferPointer();
	    else
		T[k] = GetTransfer();
	    T[k]->Construct(G[k+1],G[k]);
	}
	for (int k=levels; k>0; --k) {
	    PC[k]->Construct(*A[k]);
	    vv[k-1] = new Vector(G[k-1]);
	    T[k-1]->Project(A[k]->GetVector(),*vv[k-1]);
	    AA[k-1] = new Matrix(*vv[k-1]);
	    assemble.Jacobi(*vv[k-1],*AA[k-1]);
	    A[k-1] = AA[k-1];
	}
	(*LS)(*A[0]);
    }
    void Destruct () {
	for (int k=levels; k>0; --k) {
	    PC[k]->Destruct();
	    delete AA[k-1];
	    delete vv[k-1];
	}
    }
    void Cycle (int level, Vector& u, Vector& r) const { 
	double norm_r = norm(r);
        vout(1) << "Cycle " << level << " d " << norm_r << endl;
	if (level==0) {
	    u = *LS * r;
	    return;
	}
	Vector w(u);
	Vector d(*vv[level-1]);
	Vector c(d);
	c = 0;
	for (int i=0; i<pre[level]; ++i) {
	    w = *PC[level] * r; 
	    if (theta != 1) w *= theta;
	    r -= *A[level] * w;
	    u += w;
	}
	vout(5) << "Cycle " << level << " r " << norm(r) << endl;
        vout(6) << endl << r;
	d = r * *T[level-1];
	vout(4) << "Cycle " << level << " d " << norm(d) << endl;
	vout(6) << endl << d;        
	for (int i=0; i<cycle[level]; ++i) Cycle(level-1,c,d);
	vout(4) << "Cycle " << level << " c " << norm(c) << endl;
        vout(6) << endl << c;
	w = *T[level-1] * c;
	vout(5) << "Cycle " << level << " w " << norm(w) << endl;
        vout(6) << endl << w;
	u += w;
	r -= *A[level] * w;
	vout(3) << "Cycle " << level << " r " << norm(r) << endl;
	vout(6) << endl << r;        
	for (int i=0; i<post[level]; ++i) {
	    w = r * *PC[level]; 
	    if (theta != 1) w *= theta;
	    r -= *A[level] * w;
	    u += w;
	}
	vout(11) << "u " << level << " u " << u(Infty,0) << endl; 
    }
    void multiply (Vector& u, const Vector& b) const {
	Vector r = b;
	u = 0;
	Cycle(levels,u,r);
    }
    virtual void multiply_transpose (Vector& u, Vector& v) const { 
	multiply(u,v); 
    }
    string Name () const { return "Multigrid"; }
};


class SSOR : public Preconditioner {
    vector<SmallMatrix> D;
    int N;
    double omega;
    const Matrix *A;
 public:
    SSOR () : omega(1) { ReadConfig(Settings,"omega",omega); }
    void Construct (const Matrix& _A) {
        A = &_A;
        N = _A.nR();
	for (int i=0; i<N; ++i) {
	    int d = _A.Diag(i);
	    int n = _A.Dof(i);
	    D.push_back(SmallMatrix(n));
	    for (int k=0; k<n; ++k) 
	        for (int l=0; l<n; ++l)
		    D[i][k][l] = _A(d)[k*n+l];
	}
	//Accumulate(D);
	//dout(10) << "diag " << *D;
	for (int i=0; i<N; ++i)
	  D[i].invert();
    }
    void Destruct () { D.clear(); }
    virtual ~SSOR () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
      Vector _b(b);
      for (int i=0; i<N; ++i) {
	int d = A->Diag(i);
	int n = A->Dof(i);
	for (++d; d<A->Diag(i+1); ++d) {
	  int j = A->Column(d);
	  int m = A->Dof(j);
	  for(int k=0; k<n; ++k)
	    for(int l=0; l<m; ++l)
	      _b(i,k) -= (*A)(d)[k*m+l] * u(j,l);
	}
	for(int k=0; k<n; ++k) {
	  u(i,k) = 0.0;
	  for(int l=0; l<n; ++l)
	    u(i,k) += D[i][k][l] * _b(i,l);
	}
	for(int k=0; k<n; ++k)
	  u(i,k) *= omega;
      }
      Collect(_b);
      for (int i=N-1; i>=0; --i) {
	int d = A->Diag(i);
	int n = A->Dof(i);
	for(int k=0; k<n; ++k) {
	  u(i,k) = 0.0;
	  for(int l=0; l<n; ++l)
	    u(i,k) += D[i][k][l] * _b(i,l);
	}
	for(int k=0; k<n; ++k)
	  u(i,k) *= omega;
	for (++d; d<A->Diag(i+1); ++d) {
	  int j = A->Column(d);
	  int m = A->Dof(j);
	  for(int k=0; k<m; ++k)
	    for(int l=0; l<n; ++l)
	      _b(j,k) -= (*A)(d)[n*m+k*n+l] * u(i,l);
	}
      }
      Accumulate(u);
    }
    string Name () const { return "SSOR"; }
};

class SGS : public Preconditioner {
    vector<SmallMatrix> D;
    int N;
    const Matrix *AA;
 public:
    SGS () {}
    void Construct (const Matrix& _A) {
	AA = &_A;
        Matrix* A = new Matrix(_A);
	Accumulate(*A);
        N = _A.nR();
	for (int i=0; i<N; ++i) {
	    int d = _A.Diag(i);
	    int n = _A.Dof(i);
	    D.push_back(SmallMatrix(n));
	    for (int k=0; k<n; ++k) {
	        for (int l=0; l<n; ++l)
		    D[i][k][l] = (*A)(d)[k*n+l];
		if (D[i][k][k] == Scalar(0)) D[i][k][k] = 1.0; // !?
	    }
	}
	delete A;
	for (int i=0; i<N; ++i)
	  D[i].invert();
    }
    void Destruct () { D.clear(); }
    virtual ~SGS () { Destruct(); }
    void multiply (Vector& u, const Vector& b) const {
	Vector _b(b);
	for (int i=0; i<N; ++i) {
	    int d = AA->Diag(i);
	    int n = AA->Dof(i);
	    for (++d; d<AA->Diag(i+1); ++d) {
		int j = AA->Column(d);
		int m = AA->Dof(j);
		for(int k=0; k<n; ++k)
		    for(int l=0; l<m; ++l)
			_b(i,k) -= (*AA)(d)[k*m+l] * u(j,l);
	    }
	    for(int k=0; k<n; ++k) {
		u(i,k) = 0.0;
		for(int l=0; l<n; ++l)
		    u(i,k) += D[i][k][l] * _b(i,l);
	    }
	}
	Accumulate(u);
	for (int i=N-1; i>=0; --i) {
	    int d = AA->Diag(i);
	    int n = AA->Dof(i);
	    for(int k=0; k<n; ++k) {
		u(i,k) = 0.0;
		for(int l=0; l<n; ++l)
		    u(i,k) += D[i][k][l] * _b(i,l);
	    }
	    for (++d; d<AA->Diag(i+1); ++d) {
		int j = AA->Column(d);
		int m = AA->Dof(j);
		for(int k=0; k<m; ++k)
		    for(int l=0; l<n; ++l)
			_b(j,k) -= (*AA)(d)[n*m+k*n+l] * u(i,l);
	    }
	}
	Accumulate(u);
    }
    string Name () const { return "SGS"; }
};

string GetPCName () {
    string pc = "Jacobi"; ReadConfig(Settings,"Preconditioner",pc);
    return pc;
}

Preconditioner* GetPC (const string& name) {
    if (name == "Jacobi")                 return new Jacobi();
    if (name == "JacobiMixed")            return new JacobiMixed();
    if (name == "PointBlockJacobi")       return new PointBlockJacobi();
    if (name == "DampedJacobi")           return new DampedJacobi();    
    if (name == "NoPreconditioner")       return new NoPreconditioner();
    if (name == "GaussSeidel")            return new GaussSeidel();
    if (name == "PointBlockGaussSeidel")  return new PointBlockGaussSeidel();
    if (name == "FlyingCow")              return new FlyingCow();
    if (name == "SSOR")                   return new SSOR(); 
    if (name == "SGS")                    return new SGS(); 
    if (name == "VankaPN")                return new VankaPressureNode();
    if (name == "SuperLU")                return new SuperLU();
    if (name == "SuperLU_local")          return new SuperLU_local();
    if (name == "UMF")                    return new UMF();
    if (name == "Schur")                  return GetSchurPC();
#ifdef ILU_MULTILEVEL
    if (name == "ILUML")                  return new ILUML(); 
#endif 
//    if (name == "Multigrid")            return new ILUML(); 
#ifndef DCOMPLEX

    if (name == "AlgebraicDataDecomposition") return new ADD();
    if (name == "DD")                         return new ADD();
    if (name == "OS")			      return new OS();

#endif 
    Exit("no preconditioner " + name + " implemented");
}
Preconditioner* GetPC () { return GetPC(GetPCName()); }

Preconditioner* GetPC (const MatrixGraphs& G, const Assemble& A,
		       const string& name) {
    if (name == "Multigrid") return new Multigrid(G,A); 
    return GetPC(name);
}
Preconditioner* GetPC (const MatrixGraphs& G, const Assemble& A,
		       const Transfer& Tr, const string& name) {
    if (name == "Multigrid") return new Multigrid(G,A,Tr); 
    return GetPC(name);
}
Preconditioner* GetPC (const MatrixGraphs& G, const Assemble& A) {
    return GetPC(G,A,GetPCName()); 
}
Preconditioner* GetPC (const MatrixGraphs& G, 
		       const Assemble& A, const Transfer& Tr) {
    return GetPC(G,A,Tr,GetPCName()); 
}
MultigridPreconditioner* GetMultigrid (const MatrixGraphs& G, 
				       const Assemble& A) {
    return new Multigrid(G,A); 
}
MultigridPreconditioner* GetMultigrid (const MatrixGraphs& G, 
				       const Assemble& A,
				       const Transfer& Tr) {
    return new Multigrid(G,A,Tr); 
}

DDPreconditioner* GetDDPC (const string& name) {
#ifndef DCOMPLEX
    if (name == "OS")	return new OS();
#endif
    Exit("no preconditioner " + name + " implemented");
}

