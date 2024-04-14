// file: Interface.C
// author: Christian Wieners
// $Header: /public/M++/src/Interface.C,v 1.14 2009-06-09 14:50:17 wieners Exp $

#include "Interface.h"
#include "Algebra.h"

Point K = zero;

void SetQuasiperiodic (const Point& k) { K = k; } 

void DirichletConsistent (Vector& u) {
    ExchangeBuffer E;
    for (procset p=u.procsets(); p!=u.procsets_end(); ++p) {
	int i = u.Id(p());
	for (int j=0; j<p.size(); ++j) {
	    int q = p[j]; 
	    if (q == PPM->proc()) continue;
	    E.Send(q) << p();
	    for (int k=0; k<u.Dof(i); ++k)
		if (u.D(i,k))
		    E.Send(q) << u(i,k);
		else
		    E.Send(q) << Scalar(infty);
	}
    }
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q)  
	while (E.Receive(q).size() < E.ReceiveSize(q)) {
	    Point z;
	    dpout(100) << "s " << E.Receive(q).size() 
		       << " S " << E.Receive(q).Size() 
		       << " q " << q 
		       << endl;
	    E.Receive(q) >> z;
	    int i = u.Idx(z);
	    assert(i!=-1);
	    for (int k=0; k<u.Dof(i); ++k) {
		Scalar a;
		E.Receive(q) >> a;
		if (a != infty) {
		    u.D(i,k) = true;
		    u(i,k) = a;
		}
	    }
	}
}
void CommunicateVector (ExchangeBuffer& E, Vector& u) {
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q)  
	while (E.Receive(q).size() < E.ReceiveSize(q)) {
	    Point z;
	    dpout(100) << "s " << E.Receive(q).size() 
		       << " S " << E.Receive(q).Size() 
		       << " q " << q 
		       << endl;
	    E.Receive(q) >> z;
	    int i = u.Id(z);
	    for (int k=0; k<u.Dof(i); ++k) {
		Scalar a;
		E.Receive(q) >> a;
		dpout(200) << z << " from " << q 
		     << " recv " << a 
		     << " with i = " << i 
		     << " and k = " << k << endl;
		u(i,k) += a;
	    }
	}
}
bool master (const Vector& u, const Point& x) {
    procset p = u.find_procset(x);
    if (p == u.procsets_end()) return PPM->proc();   
    return p.master();
}
void CollectIdentify (Vector& u) {
    ExchangeBuffer& E = u.CollectIdentifyBuffer();
    for (identifyset is = u.identifysets(); is!=u.identifysets_end(); ++is) {
	if (is.master()) continue;
	const Point& z = is();
	int j = u.Id(z);
	E.Send(PPM->proc()) << is[0];
	for (int k=0; k<u.Dof(j); ++k) {
	    E.Send(PPM->proc()) << u(j,k);
	    u(j,k) = 0;
	}
    }
    CommunicateVector(E,u);
}
void AccumulateIdentify (Vector& u) {
    ExchangeBuffer& E = u.AccumulateIdentifyBuffer();
    for (identifyset is = u.identifysets(); is!=u.identifysets_end(); ++is) {
	const Point& z = is();
	int j = u.Id(z);
        for (int i=0; i<is.size(); ++i) {
             Point y = is[i];
	     E.Send(PPM->proc()) << y;
	     for (int k=0; k<u.Dof(j); ++k)
		 E.Send(PPM->proc()) << u(j,k);
        }
    }
    CommunicateVector(E,u);
}
void CollectParallel (Vector& u) {
    ExchangeBuffer& E = u.CollectParallelBuffer();
    for (procset p=u.procsets(); p!=u.procsets_end(); ++p) {
	int i = u.Id(p());
	int q = p.master(); 
	if (q == PPM->proc()) continue;
	E.Send(q) << p();
	for (int k=0; k<u.Dof(i); ++k) {
	    E.Send(q) << u(i,k);
	    dpout(99) << "send " << u(i,k) << endl;
	    u(i,k) = 0;
	}
    }
    CommunicateVector(E,u);
}
void Collect (Vector& u) {
    if (u.identify()) CollectIdentify(u);
    if (u.parallel()) CollectParallel(u);
}
void AccumulateParallel (Vector& u) {
    ExchangeBuffer& E = u.AccumulateParallelBuffer();
    for (procset p=u.procsets(); p!=u.procsets_end(); ++p) {
	int i = u.Id(p());
	for (int j=0; j<p.size(); ++j) {
	    int q = p[j]; 
	    if (q == PPM->proc()) continue;
	    E.Send(q) << p();
	    for (int k=0; k<u.Dof(i); ++k)
		E.Send(q) << u(i,k);
	}
    }
    CommunicateVector(E,u);
}
void Average (Vector& u) {
    ExchangeBuffer E;
    for (procset p=u.procsets(); p!=u.procsets_end(); ++p) {
	int i = u.Id(p());
	if (p.master() != PPM->proc()) continue;
	double s = 1.0 / p.size();
	for (int k=0; k<u.Dof(i); ++k)
	    u(i,k) *= s;
	for (int j=0; j<p.size(); ++j) {
	    int q = p[j]; 
	    if (q == PPM->proc()) continue;
	    E.Send(q) << p();
	    for (int k=0; k<u.Dof(i); ++k)
		E.Send(q) << u(i,k);
	}
    }
    CommunicateVector(E,u);
}
void MakeAdditive (Vector& u) {
    for (row r=u.rows(); r!=u.rows_end(); ++r) {
	procset p = u.find_procset(r());
	if (p == u.procsets_end()) continue;
	if (p.master() == PPM->proc()) continue;
	for (int k=0; k<r.n(); ++k)
	    u(r,k) = 0;
    }
    for (identifyset is = u.identifysets(); is!=u.identifysets_end(); ++is) {
	row r = u.find_row(is());
	if (is.master()) continue;
	for (int k=0; k<r.n(); ++k)
	    u(r,k) = 0;
    }
}
void Accumulate (Vector& u) {
    if (u.identify()) AccumulateIdentify(u);
    if (u.parallel()) AccumulateParallel(u);
    if (!u.identify()) return;
    if (K == zero) return;
    for (identifyset is = u.identifysets(); is!=u.identifysets_end(); ++is) {
	const Point& z = is();
	int j = u.Id(z);
	double zK = 0;
	for (int i=0; i<3; ++i) 
	    if (z[i] == 1)
		zK += K[i];
	Scalar s = exp((zK)*iUnit);
	for (int k=0; k<u.Dof(j); ++k)
	    u(j,k) *= s;
    }
}
void CommunicateMatrix (ExchangeBuffer& E, Matrix& A) {
    E.Communicate();
    Scalar* a = A();
    for (short q=0; q<PPM->size(); ++q)  
	while (E.Receive(q).size() < E.ReceiveSize(q)) {
	    Point x;
	    E.Receive(q) >> x;
	    row r = A.find_row(x);
	    if (r == A.rows_end()) Exit("no row in Communicate Matrix");
	    int id = r.Id();
	    int d = A.Entry(A.Diag(id));
	    int n = A.Dof(id);
	    for (int j=0; j<n*n; ++j) {
		Scalar b;
		E.Receive(q) >> b;
		a[d+j] += b;
	    }
	    int s;
	    E.Receive(q) >> s;
	    for (int k=0; k<s; ++k) {
		Point y;
		int m;
		E.Receive(q) >> y >> m;
		int dd = r.GetEntryX(y);
		if (dd == -1) {
		    Scalar tmp;
		    for (int j=0; j<m; ++j)
			E.Receive(q) >> tmp;
		} else {
		    Scalar tmp;
		    for (int j=0; j<m; ++j) {
			E.Receive(q) >> tmp;
			a[dd+j] += tmp;
		    }
		}
	    }
	}
}
void AccumulateIdentify (Matrix& A) {
    ExchangeBuffer E;
    Scalar* a = A();
    int q = PPM->proc();
    for (identifyset is = A.identifysets(); is!=A.identifysets_end(); ++is) {
	const Point& z = is();
	row r = A.find_row(z);
	int id = r.Id();
	int d = A.Entry(A.Diag(id));
	int n = A.Dof(id);
        for (int i=0; i<is.size(); ++i) {
             Point y = is[i];
	     E.Send(q) << y;
	    for (int j=0; j<n*n; ++j)
		E.Send(q) << a[d+j];
	    E.Send(q) << int(r.size());
	    for (entry e=r.entries(); e!=r.entries_end(); ++e) {
		E.Send(q) << e();
		int m = 2 * n * A.Dof(e.Id());
		E.Send(q) << m;
		int dd = e.GetEntry();
		for (int j=0; j<m; ++j)
		    E.Send(q) << a[dd+j];
	    }
	}
    }
    CommunicateMatrix(E,A);
}
void AccumulateParallel (Matrix& A) {
    ExchangeBuffer E;
    Scalar* a = A();
    for (procset p=A.procsets(); p!=A.procsets_end(); ++p) {
	const Point& x = p();
	row r = A.find_row(x);
	int id = r.Id();
	int d = A.Entry(A.Diag(id));
	int n = A.Dof(id);
	for (int i=0; i<p.size(); ++i) {
	    int q = p[i]; 
	    if (q == PPM->proc()) continue;
	    E.Send(q) << x;
	    for (int j=0; j<n*n; ++j)
		E.Send(q) << a[d+j];
	    E.Send(q) << int(r.size());
	    for (entry e=r.entries(); e!=r.entries_end(); ++e) {
		E.Send(q) << e();
		int m = 2 * n * A.Dof(e.Id());
		E.Send(q) << m;
		int dd = e.GetEntry();
		for (int j=0; j<m; ++j)
		    E.Send(q) << a[dd+j];
	    }
	}
    }
    CommunicateMatrix(E,A);
}
void Accumulate (Matrix& A) {
    if (A.identify()) AccumulateIdentify(A);
    if (A.parallel()) AccumulateParallel(A);
}
void Consistent2Additive (Vector& u) {
    const Mesh& M = u.GetMesh();
    for (procset p=u.procsets(); p!=u.procsets_end(); ++p) {
	int i = u.Id(p());
	identifyset is = M.find_identifyset(p());
	if (is != u.identifysets_end()) continue;
	if (p.master() != PPM->proc()) 
	    for (int k=0; k<u.Dof(i); ++k)
		u(i,k) = 0.0;
    }
    Vector v(u);
    u = 0.0;
    for (cell c = M.cells(); c!=M.cells_end(); ++c) {
	rows R(u,c);
	RowValues u_c(u,R);
	RowValues v_c(v,R);
	for (int i=0; i<R.size(); ++i) {
	    for (int k=0; k<R[i].n(); ++k)
		u_c(i,k) = v_c(i,k);
	}
    }
}
