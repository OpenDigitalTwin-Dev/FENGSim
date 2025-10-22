// file: DD.C
// author: Martin Sauter
// purpose: to narrow Preconditioner.C

#include "DD.h"

ADD::ADD () : K(0), DDType("Cell"), S_TO_S(1), CoarseCorr(0), GaussSeidel(0),
	      M(0), Sol(0), UMax(0), NMax(1), CoarseType("trivial"), POU(1),
	      C(0), C_short(0), Sol_short(0), A(0), 
	      voriter(1), rueckiter(0), pqs(1),
	      dist(1), Sp(0) {
    dout(3) << " DomainDecomposition: constructor" << endl;
    ReadConfig(Settings,"DDType",DDType);
    ReadConfig(Settings,"S_TO_S",S_TO_S);
    ReadConfig(Settings,"CoarseCorr",CoarseCorr);
    ReadConfig(Settings,"GaussSeidel",GaussSeidel);
    ReadConfig(Settings,"DDU",UMax);
    ReadConfig(Settings,"DDN",NMax);
    ReadConfig(Settings,"CoarseType",CoarseType);
    ReadConfig(Settings,"POU",POU);
    ReadConfig(Settings,"postiter",rueckiter);
    ReadConfig(Settings,"preiter",voriter);
    ReadConfig(Settings,"pqs",pqs);
    ReadConfig(Settings,"dist",dist);
    if ((CoarseType != "Indices") && (CoarseCorr < 0)) {
	CoarseCorr = -CoarseCorr;
	mout << "CoarseType must be 'Indices' for CoarseCorr < 0!"
	     << " --> CoarseCorr = " << CoarseCorr << endl;
    }
    if ((CoarseType == "Indices") && (CoarseCorr > 0)) {
	CoarseCorr = -CoarseCorr;
	mout << "CoarseCorr may be negative (short-mode) "
	     << " for CoarseType = Indices! --> CoarseCorr =  " 
	     << CoarseCorr << endl;
    }
}

void ADD::Construct_Coarse_trivial (const SparseMatrix& S) {
    M = 1;
    C = new SmallMatrix(M);
    c.resize(M);
    for (int m=0; m<M; ++m) {
	c[m] = new SmallVector(1,S.size());
    }
}
void ADD::Construct_Coarse_lin_x(const SparseMatrix& S) {
    M = 1;
    C = new SmallMatrix(M);
    c.resize(M);
	
    Vector lin_x((*A).GetVector());
    lin_x = 0;
	
    for (int m=0; m<M; ++m) {
	c[m] = new SmallVector(0,S.size());
	for (row r= (*A).rows(); r!=(*A).rows_end(); ++r)
	    lin_x(r,0) = r()[0];
	    
	for (int i = 0; i < S.size(); ++i)
	    (*c[m])[i] = lin_x[i];
    }
}
void ADD::Construct_Coarse_lin_y(const SparseMatrix& S) {
    M = 1;
	
    C = new SmallMatrix(M);
    c.resize(M);
	
    Vector lin_x((*A).GetVector());
    lin_x = 0;
	
    for (int m=0; m<M; ++m) {
	c[m] = new SmallVector(0,S.size());
	for (row r= (*A).rows(); r!=(*A).rows_end(); ++r)
	    lin_x(r,1) = r()[1];
	    
	for (int i = 0; i < S.size(); ++i)
	    (*c[m])[i] = lin_x[i];
    }
}
void ADD::Construct_Coarse_Indices (const SparseMatrix& S) {
    M = K;
	
    C = new SmallMatrix(M);
    c.resize(M);
	
    for (int m=0; m<M; ++m)
	c[m] = new SmallVector(0,S.size());
	
    for (int m=0; m<M; ++m)
	for (int i=0; i< Indices[m].size(); ++i)
	    (*(c[m]))[Indices[m][i]] = 1;
}
void ADD::Construct_Coarse_Indices_short (const SparseMatrix& S) {
    M = K;
	
    C_short = new SmallMatrix(M);
    c_short.resize(M);
	
    for (int m=0; m<M; ++m)
	c_short[m] = new SmallVector(1,Indices[m].size());
}
void ADD::Clear_Dirichlet(const int n, const bool *d) {
    for (int i=0; i<n; ++i)
	if (d[i])
	    for (int m=0; m<M; ++m)
		(*(c[m]))[i] = 0;
}
    
void ADD::Clear_Dirichlet_short(const int n, const bool *d) {
    for (int m=0; m<M; ++m)
	for (int i=0; i<Indices[m].size(); ++i)
	    if (d[Indices[m][i]])
		(*(c_short[m]))[i] = 0;
}
void ADD::Coarse_POU(const int n) {
    for (int i=0; i<n; ++i) {
	Scalar rho = 0.;
	for (int m=0; m<M; ++m)
	    rho += (*(c[m]))[i];
	if (abs(rho) > 0)
	    for (int m=0; m<M; ++m)
		(*(c[m]))[i] /= rho;
    }
}
void ADD::Coarse_POU_short(const int n) {
    SmallVector rho(0,n);
    for (int m=0; m<M; ++m)
	for (int i=0; i<Indices[m].size(); ++i)
	    rho[Indices[m][i]] += (*(c_short[m]))[i];
	
    for (int m=0; m<M; ++m)
	for (int i=0; i<Indices[m].size(); ++i)
	    if (abs(rho[Indices[m][i]]) > 0)
		(*(c_short[m]))[i] /= rho[Indices[m][i]];
}
    
void ADD::Construct_Coarse (const SparseMatrix &S, const bool* d) {
    if (CoarseType == "trivial") Construct_Coarse_trivial(S);
    if (CoarseType == "Indices") Construct_Coarse_Indices(S);
    if (CoarseType == "lin_x") Construct_Coarse_lin_x(S);
    if (CoarseType == "lin_y") Construct_Coarse_lin_y(S);
	
    Clear_Dirichlet(S.size(),d);
	
    if (POU) Coarse_POU(S.size());
	
    SmallMatrix SSmall(S);
    SparseMatrix SpC(*C);
    Sol = GetSparseSolver(SpC);
}
void ADD::Construct_Coarse_short (const SparseMatrix &S, const bool* d) {
    Date Start_Construct;
    if (CoarseType != "Indices") {
	Construct_Coarse (S,d);
	return;
    }
    Construct_Coarse_Indices_short(S);
    tout(3) << "  /-constructing vectors for Coarse-Problem " 
	    << Date() - Start_Construct << endl;
	
    Date Start_Dirichlet;
    Clear_Dirichlet_short(S.size(),d);
    tout(3) << "  |-clear Dirichlet " << Date() - Start_Dirichlet << endl;

    if (POU) {
	Date Start_POU;
	Coarse_POU_short(S.size());
	tout(3) << "  |-make POU " << Date() - Start_POU << endl;
    }
	
    Date Start_fast2;
    for (int m=0; m<M; ++m) {
	set<int> support;
	    
	BasicVector Sc(0,S.size());
	for (int i=0; i<Indices[m].size(); ++i) {
	    for (int d = S.rowind(Indices[m][i]); 
		 d < S.rowind(Indices[m][i]+1); ++d) {
		int col = S.colptr(d);
		Sc[col] += S.nzval(d) * (*c_short[m])[i];
		for (list<int>::const_iterator j = in_ind[col].begin(); 
		     j != in_ind[col].end(); ++j)
		    support.insert(*j);
	    }
	}
	    
	for (set<int>::const_iterator n=support.begin(); 
	     n != support.end(); ++n) {
	    if (*n < m) continue;
	    Scalar erg = 0.0;
	    for (int i=0; i<Indices[*n].size(); ++i)
		erg += (*(c_short[*n]))[i] * Sc[Indices[*n][i]];
	    (*C_short)[m][*n] = (*C_short)[*n][m] = erg;
	}
    }
	
    tout(3) << "  |-constructing coarse-matrix with short variables "
	    << "with support " << Date() - Start_fast2 << endl;
	
    Date Start_Sparse;
    SparseMatrix SpC(*C_short);
    tout(3) << "  |-constructing coarse-matrix in Sparse-Format " 
	    << Date() - Start_Sparse << endl;
	
    mout << "  |--- SpC.Size()/SpC.size(): " 
	 << SpC.Size()/SpC.size() << endl;
	
    Date Start_Sol;
    Sol_short = GetSparseSolver(SpC);
    tout(3) << "  V-constructing solver for coarse-matrix " 
	    << Date() - Start_Sol << endl;
	
}
void ADD::CoarseCorrection (Vector& u, Vector& r) const  {
    BasicVector cr(M);
    for (int m=0; m<M; ++m) {
	Scalar s = 0;
	for (int i=0; i<r.size(); ++i)
	    s += r[i] * (*(c[m]))[i];
	cr[m] = s;
    }
	
    Sol->Solve(cr());
	
    Vector cor(0,u);
	
    for (int m = 0; m < M; ++m)
	for (int i = 0; i < cor.size(); ++i)
	    cor[i] += cr[m] * (*(c[m]))[i];
	
    u += cor;
	
    r -= *A * cor;
}
void ADD::CoarseCorrection_short (Vector& u, Vector& r) const {
    if (CoarseType != "Indices") {
	CoarseCorrection(u,r);
	return;
    }
	
    BasicVector cr(M);
    for (int m=0; m<M; ++m) {
	Scalar s = 0;
	for (int i=0; i<Indices[m].size(); ++i)
	    s += r[Indices[m][i]] * (*(c_short[m]))[i];
	cr[m] = s;
    }
	
    Sol_short->Solve(cr());
	
    Vector cor(0,u);
	
    for (int m = 0; m < M; ++m)
	for (int i=0; i<Indices[m].size(); ++i)
	    cor[Indices[m][i]] += cr[m] * (*(c_short[m]))[i];
	
    u += cor;
	
    r -= *A * cor;
}
void ADD::Construct_LR (const Matrix& A) {
    K = 1;
    Indices.resize(K);
	
    for (int k=0; k<K; ++k) {
	int N = A.size();
	Indices[k].resize(N);
	for (int n=0; n<N; ++n)
	    Indices[k][n] = n;
    }
}
void ADD::Construct_Jacobi(const Matrix& A) {
    K = A.nR();
    Indices.resize(K);
    for (int k=0; k<K; ++k) {
	int i = A.Index(k);
	int N = A.Dof(k);
	Indices[k].resize(N);
	for (int n=0; n<N; ++n)
	    Indices[k][n] = i+n;
    }
    return;

    K = A.size();
    Indices.resize(K);
	
    for (int k=0; k<K; ++k) {
	int N = 1;
	Indices[k].resize(N);
	Indices[k][0] = k;
    }
}
void ADD::Construct_Row (const Matrix& A) {
    SparseMatrix S(A);
	
    K = S.size();
    Indices.resize(K);
	
    for (int k=0; k<K; ++k) {
	int N = S.rowind(k+1) - S.rowind(k);
	int N_min = 0;
	Indices[k].resize(N);
	for (int n=S.rowind(k); n<S.rowind(k+1); ++n)
	    if ((abs(S.nzval(n)) > Eps) || (n == 0)) 
		Indices[k][N_min++] = S.colptr(n);
	Indices[k].resize(N_min);
    }
}
void ADD::Construct_RowOld (const Matrix& A) {
    SparseMatrix S(A);
	
    K = S.size();
    Indices.resize(K);
	
    for (int k=0; k<K; ++k) {
	int N = S.rowind(k+1) - S.rowind(k);
	Indices[k].resize(N);
	for (int n=S.rowind(k); n<S.rowind(k+1); ++n)
	    Indices[k][n-S.rowind(k)] = S.colptr(n);
    }
}
void ADD::Construct_DefU (const Matrix& A) {
    SparseMatrix S(A);
	
    int U = UMax;
    K = S.size();
    Indices.resize(K);
    SMat.resize(K);
    SSol.resize(K);
	
    for (int k=0; k<K; ++k) {
	int N = S.rowind(k+1) - S.rowind(k);
	if (N > U) N = U+1;
	Indices[k].resize(N);
	for (int n=S.rowind(k); n<S.rowind(k)+N; ++n)
	    Indices[k][n-S.rowind(k)] = S.colptr(n);
    }
}
void ADD::Construct_Near (const Matrix& A) {
	
    bool littlemat = false;
    int si = A.size();
    int N = NMax;
    int U = UMax;
    if (N > si) N = si;
    if (U >= N) U = 0;
	
    K = int((A.size()-N)/(N-U)) + 1;
    if (((K-1)*(N-U) + N) != si)
	littlemat = true;
	
    Indices.resize(K+littlemat);

    for (int k = 0; k<K; ++k) {
	Indices[k].resize(N);
	for (int n = k*(N-U); n < k*(N-U)+N; ++n)
	    Indices[k][n-k*(N-U)] = n;
    }
	
    if (littlemat) {
	Indices[K].resize(si-K*(N-U));
	for (int n = K*(N-U); n < si; ++n)
	    Indices[K][n-K*(N-U)] = n;
    }
	
    K += littlemat;
}
void ADD::Construct_Cell_with_global (const Matrix& A) {
    K = A.GetMesh().Cells::size()+1;
	
    Indices.resize(K);
    Indices[0].resize(K);
    K = 0;
    for (cell c=A.cells(); c!=A.cells_end(); ++c, ++K) {
	rows r(A,c);
	int N = 0;
	for (int i=0; i<r.size(); ++i)
	    N += r[i].n();
	Indices[K].resize(N);
	N = 0;
	Indices[0][K] = A.Index(r[0].Id());
	    
	for (int i=0; i<r.size(); ++i) {
	    int index = A.Index(r[i].Id());
	    for (int j=0; j<r[i].n(); ++j)
		Indices[K][N++] = index+j;
	}
    }
    return;
}
    
void ADD::Construct_Cell (const Matrix& A) {
    K = A.GetMesh().Cells::size();
	
    Indices.resize(K);
    K = 0;
    for (cell c=A.cells(); c!=A.cells_end(); ++c, ++K) {
	rows r(A,c);
	int N = 0;
	for (int i=0; i<r.size(); ++i)
	    N += r[i].n();
	Indices[K].resize(N);
	N = 0;
	for (int i=0; i<r.size(); ++i) {
	    int index = A.Index(r[i].Id());
	    for (int j=0; j<r[i].n(); ++j) {
		Indices[K][N++] = index+j;
	    }
	}
	continue;
	mout << K << " cell: ";
	for (int n=0; n<Indices[K].size(); ++n)
	    mout << Indices[K][n] << " ";
	mout << endl;
    }
    return;
    for (int i=0; i<A.size(); ++i) {
	mout << i << ":";
	for (int k=0; k<K; ++k)
	    for (int n=0; n<Indices[k].size(); ++n)
		if (Indices[k][n] == i)
		    mout << k << ",";
	mout << endl;
    }
	
    mout << "\n\n\n";
    for (int i = 0; i < A.size(); ++i) {
	mout << i << ":";
	for (list<int>::const_iterator j = in_ind[i].begin(); 
	     j != in_ind[i].end(); ++j)
	    mout << (*j) <<",";
	mout << endl;
    }
    return;
}
void ADD::create_in_ind (int N) {
    in_ind.resize(N);
    for (int k=0; k<K; ++k) 
	for (int i=0; i<Indices[k].size(); ++i)
	    in_ind[Indices[k][i]].push_back(k);
}

const Matrix* __A;

void ADD::Construct (const Matrix& _A) {
    __A = &_A;
    A = new Matrix(_A);
    Accumulate(*A);
    Date Start_Construct;
    if (DDType == "LR") Construct_LR(*A);
    if (DDType == "Row") Construct_Row(*A);
    if (DDType == "RowOld") Construct_RowOld(*A);
    if (DDType == "Jacobi") Construct_Jacobi(*A);
    if (DDType == "DefU") Construct_DefU(*A);
    if (DDType == "Near") Construct_Near(*A);
    if (DDType == "Cell") Construct_Cell(*A);
    if (DDType == "Cell_global") Construct_Cell_with_global(*A);
//    if (DDType == "Cell_large") Construct_Cell_large(*A);

    for (int k=0; k<K; ++k) 
	sort(Indices[k].begin(),Indices[k].end());

    create_in_ind(A->size());
	
    tout(3) << " constructing " << K 
	    << " vectors for DD " << Date() - Start_Construct << endl;
	
    Date Start_SMat;
	
    SMat.resize(K);
    SSol.resize(K);
    for (int k=0; k<K; ++k) {
	SMat[k] = 0;
	SSol[k] = 0;
    }
	
    SparseMatrix S(*A);
	
    Sp = new SparseMatrix(*A);

    for (int n=0; n<Sp->size(); ++n)
	for (int d = (*Sp).rowind(n) + 1; 
	     d < (*Sp).rowind(n+1); ++d) {
	    int col = (*Sp).colptr(d);
	    if (A->GetVector().D()[col])
		Sp->nzval(d) = 0;
	}
	
    if (S_TO_S == 2) {
	vector<bool> mask(S.size(),false);
	for (int k=0; k<K; ++k) {
	    for (int i = 0; i < Indices[k].size(); ++i)
		mask[Indices[k][i]] = true;
		
	    SMat[k] = new SparseMatrix(S,mask);
		
	    for (int i = 0; i < Indices[k].size(); ++i)
		mask[Indices[k][i]] = false;
	}
    }
    else if (S_TO_S == 1) {
	for (int k=0; k<K; ++k) {
	    SMat[k] = new SparseMatrix(*Sp,Indices[k]);
	}
    }
    else {
	for (int k=0; k<K; ++k) {
	    /* Error for Elasticity */
	    SmallMatrix B(S,Indices[k]);
	    SMat[k] = new SparseMatrix(B);
	}
    }
    tout(3) << "constructing SMat " << Date() - Start_SMat << endl;
	

    size_t Nmax = 0;
    for (int k=0; k<K; ++k) 
	Nmax = max(Nmax,Indices[k].size());
    Date Start2;

    for (int k = 0; k<K; ++k)
	SSol[k] = GetSparseSolver(*SMat[k]);
//	SSol[k] = GetSparseSolver(*SMat[k],"UMFSolver");
    tout(1) << "constructing Solver for SMat " << Date() - Start2 
	    << " for K=" << K
	    << " and N=" << Nmax
	    << endl;
	
    if (CoarseCorr > 0) {
	Date Start3;
	Construct_Coarse(S,Dirichlet());
	tout(3) << "constructing coarse " << Date() - Start3 << endl;
    }
	
    if (CoarseCorr < 0) {
	Date Start5;
	Construct_Coarse_short(S,Dirichlet());
	tout(3) << "constructing coarse with short variables " 
		<< Date() - Start5 << endl;
    }
}
    
void ADD::Destruct () {
    for (int k=0; k<K; ++k) {
	if (SMat[k]) delete SMat[k];
	if (SSol[k]) delete SSol[k];
	SMat[k] = 0;
	SSol[k] = 0;
    }
    SMat.clear();
    SSol.clear();
    if (C) delete C; C = 0;
    if (Sol) delete Sol; Sol = 0;
    if (C_short) delete C_short; C_short = 0;

    if (CoarseCorr > 0) {
        for (int k = 0; k < K; ++K) {
            if (c_short[k]) delete c_short[k];
            c_short[k] = 0;
        }
        c_short.clear();
    }

    if (CoarseCorr < 0) {
        for (int k = 0; k < K; ++K) {
            if (c[k]) delete c[k]; 
            c[k] = 0;
        }
        c.clear();
    }

    if (Sol_short) delete Sol_short; Sol_short = 0;
    if (A) delete A; A = 0;

    if (Sp) delete Sp; Sp = 0;

}

ADD::~ADD () { Destruct(); }
    
void ADD::multiply (Vector& u, const Vector& b) const {
	
    Vector r = b;
    u = 0;
	
    if (CoarseCorr == 1) CoarseCorrection(u,r);
    if (CoarseCorr == -1) CoarseCorrection_short(u,r);

    for (int vor = 0; vor < voriter; ++vor)
	for (int k=0; k<K; ++k) {
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n) 
		s[n] = r[Indices[k][n]];
	    if (GaussSeidel) {
		for (int n=0; n<N; ++n)
		    for (int d = (*Sp).rowind(Indices[k][n]); 
			 d < (*Sp).rowind(Indices[k][n]+1); ++d) {
			int col = (*Sp).colptr(d);
			s[n] -= (*Sp).nzval(d) * u[col];
		    }
	    }
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		u[Indices[k][n]] += s[n];
	}
	
    for (int rueck=0; rueck < rueckiter; ++rueck)
	for (int k=K-1; k>=0; --k) {
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n)
		s[n] = r[Indices[k][n]];
	    if (GaussSeidel) {
		for (int n=0; n<N; ++n)
		    for (int d = (*Sp).rowind(Indices[k][n]); 
			 d < (*Sp).rowind(Indices[k][n]+1); ++d) {
			int col = (*Sp).colptr(d);
			s[n] -= Sp->nzval(d) * u[col];
		    }
	    }
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		u[Indices[k][n]] += s[n];
	}
	
    if (CoarseCorr == 2) CoarseCorrection(u,r);
    if (CoarseCorr == -2) CoarseCorrection_short(u,r);
	
    Accumulate(u);
}

void ADD::multiply_transpose (Vector& u, const Vector& b) const {
    Vector r = b;
    Accumulate(r);        
    u = 0;
	
    if (CoarseCorr == 2) CoarseCorrection(u,r);
    if (CoarseCorr == -2) CoarseCorrection_short(u,r);

    for (int vor = 0; vor < voriter; ++vor)
	for (int k=K-1; k>=0; --k) {
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n)
		s[n] = r[Indices[k][n]];
	    if (GaussSeidel) {
		for (int n=0; n<N; ++n)
		    for (int d = (*Sp).rowind(Indices[k][n]); 
			 d < (*Sp).rowind(Indices[k][n]+1); ++d) {
			int col = (*Sp).colptr(d);
			s[n] -= Sp->nzval(d) * u[col];
		    }
	    }
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		u[Indices[k][n]] += s[n];
	}
    for (int rueck=0; rueck < rueckiter; ++rueck)
	for (int k=0; k<K; ++k) {
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n) 
		s[n] = r[Indices[k][n]];
	    if (GaussSeidel) {
		for (int n=0; n<N; ++n)
		    for (int d = (*Sp).rowind(Indices[k][n]); 
			 d < (*Sp).rowind(Indices[k][n]+1); ++d) {
			int col = (*Sp).colptr(d);
			s[n] -= (*Sp).nzval(d) * u[col];
		    }
	    }
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		u[Indices[k][n]] += s[n];
	}
	
    if (CoarseCorr == 1) CoarseCorrection(u,r);
    if (CoarseCorr == -1) CoarseCorrection_short(u,r);
	
    MakeAdditive(u);     
    Accumulate(u);
}

string ADD::Name () const { return "AlgebraicDataDecomposition"; }
ostream& operator << (ostream& s, const ADD& ADD) {
    return s << "AlgebraicDataDecomposition"; }


// Overlapping Schwarz (OS) --- Overlapping Schwarz (OS) --- Overlapping Schwarz (OS) --- Overlapping Schwarz (OS)
// class OS::SubProblem

void OS::SubProblem::Construct(const SparseMatrix& S) {
    sort(index.begin(),index.end());
 
    if (Sparse_or_Small) {
        SMat = new SparseMatrix(S,index);
        SSol = GetSparseSolver(*SMat);
    }
    else {
        SmallMat = new SmallMatrix(S,index);
        SmallSol = GetSmallSolver(*SmallMat);
    }
}

void OS::SubProblem::Destruct () {
    if (SMat) delete SMat; SMat = 0;
    if (SSol) delete SSol; SSol = 0;
    if (SmallMat) delete SmallMat; SmallMat = 0;
    if (SmallSol) delete SmallSol; SmallSol = 0;
    index.clear();
}

void OS::SubProblem::resize(int N) {
    index.resize(N);
}

void OS::SubProblem::Solve (Scalar* x) const { 

    if (Sparse_or_Small) SSol->Solve(x); 
    else SmallSol->Solve(x);

}

// class OS::Decomposition

OS::Decomposition::Decomposition () : K(0), sp(0), UMax(0), NMax(1), pqs(1), dist(1), maxsize_Small_Solver(0) {
    ReadConfig(Settings,"DDType",DDType);
    ReadConfig(Settings,"DDU",UMax);
    ReadConfig(Settings,"DDN",NMax);
    ReadConfig(Settings,"pqs",pqs);
    ReadConfig(Settings,"dist",dist);
    ReadConfig(Settings,"maxsize_Small_Solver",maxsize_Small_Solver);
}

void OS::Decomposition::Construct_Jacobi(const Matrix& A) {
    K = A.nR();
    sp.resize(K);
    for (int k = 0; k < K; ++k) {
        int i = A.Index(k);
        int N = A.Dof(k);
        sp[k] = new SubProblem(N);
        for (int n = 0; n < N; ++n)
            (*sp[k])[n] = i+n;
    }
    return;
}

void OS::Decomposition::Construct_Cell (const Matrix& A) {
    K = A.GetMesh().Cells::size();
    sp.resize(K);
    K = 0;
    for (cell c=A.cells(); c!=A.cells_end(); ++c, ++K) {
        rows r(A,c);
        int N = 0;
        for (int i=0; i<r.size(); ++i)
	    N += r[i].n();
        sp[K] = new SubProblem(N);
        N = 0;
        for (int i=0; i<r.size(); ++i) {
	    int index = A.Index(r[i].Id());
	    for (int j=0; j<r[i].n(); ++j)
	        (*sp[K])[N++] = index+j;
        }
    }
    return;
}

void OS::Decomposition::Construct_Cell_large (const Matrix& A) {
    int dim = A.dim();
    double frame[3][2];

    vertex v = A.GetMesh().vertices();
    for (int k=0; k<dim; ++k)
        frame[k][0] = frame[k][1] = v[k];
    for (++v; v!=A.GetMesh().vertices_end(); ++v) 
        for (int k=0; k<dim; ++k) {
	    frame[k][0] = min(frame[k][0],v[k]);
	    frame[k][1] = max(frame[k][1],v[k]);
        }

    int P=1,Q=1;
    int S = 1;
    P = Q = pqs;
    if (dim == 3) S = pqs;
    K = P*Q*S;
    vector<Point* > Pnt(K);
    Pnt.resize(K);
    sp.resize(K);
    vector< set<int> > index;
    index.resize(K);
    mout << "DIM: " << dim << endl;
    for (int s = 0; s < S; ++s)
        for (int q = 0; q < Q; ++q)
            for (int p = 0; p < P; ++p)
                if (dim == 3) 
                    Pnt[p+P*q+P*Q*s] = new Point(frame[0][0] + (frame[0][1] - frame[0][0])/(2.0*P) + p * (frame[0][1] - frame[0][0])/P,
                                                 frame[1][0] + (frame[1][1] - frame[1][0])/(2.0*Q) + q * (frame[1][1] - frame[1][0])/Q,
                                                 frame[2][0] + (frame[2][1] - frame[2][0])/(2.0*S) + s * (frame[2][1] - frame[2][0])/S);
                else if (dim == 2) 
                    Pnt[p+P*q+P*Q*s] = new Point(frame[0][0] + (frame[0][1] - frame[0][0])/(2.0*P) + p * (frame[0][1] - frame[0][0])/P,
                                                 frame[1][0] + (frame[1][1] - frame[1][0])/(2.0*Q) + q * (frame[1][1] - frame[1][0])/Q,
                                                 0);

    for (cell c=A.cells(); c!=A.cells_end(); ++c) {
        rows r(A,c);
        for (int k=0; k<K; ++k) {
            Point d = c()-(*(Pnt[k]));
            if (abs(d[0]) < dist*(frame[0][1] - frame[0][0])/(2.0*P))
                if (abs(d[1]) < dist*(frame[1][1] - frame[1][0])/(2.0*Q))
                    if ((dim == 2) || (abs(d[2]) < dist*(frame[2][1] - frame[2][0])/(2.0*S))) {
                        for (int rs = 0; rs < r.size(); ++rs) {
                            int ind = A.Index(r[rs].Id());
                            for (int n = 0; n < r[rs].n(); ++n)
                                index[k].insert(ind+n);
                        }
                   }
        }
    }

            int null_ent = 0;

    for (int k=0;k<K;++k) {
        int sz = index[k].size();
        if (sz != 0) {
            sp[k-null_ent]  = new SubProblem(sz);
            int N = 0;
            for (set<int>::const_iterator z=index[k].begin();
            z != index[k].end();++z) { 
                (*sp[k-null_ent])[N++] = *z;
            }
        }
        else null_ent++;
    }

    K -= null_ent;
    if (null_ent) sp.resize(K);
    mout << "null_ent: " << null_ent << endl;

    for (int s = 0; s < S; ++s)
        for (int q = 0; q < Q; ++q)
            for (int p = 0; p < P; ++p)
                if (dim == 3) 
                    delete Pnt[p+P*q+P*Q*s];
                else if (dim == 2) 
                    delete Pnt[p+P*q+P*Q*s];

    Pnt.clear();
}

void OS::Decomposition::Construct_Row (const Matrix& A) {
    SparseMatrix S(A);

    K = S.size();
    sp.resize(K);

    for (int k=0; k<K; ++k) {
        int N = S.rowind(k+1) - S.rowind(k);
        int N_min = 0;
        sp[k] = new SubProblem(N);
        for (int n = S.rowind(k); n < S.rowind(k+1); ++n)
	    (*sp[k])[N_min++] = S.colptr(n);
    }
    return;
}
   
void OS::Decomposition::Construct (Matrix& A) {
    Date Start_Construct;

    if (DDType == "Jacobi") Construct_Jacobi(A);
    if (DDType == "Row") Construct_Row(A);
    if (DDType == "Cell") Construct_Cell(A);
    if (DDType == "Cell_large") Construct_Cell_large(A);

    SparseMatrix S(A);
    int max_size = 0;
    int min_size = 1000000;
    for (int k=0; k<K; ++k) { 
        int sz = (*sp[k]).size();
        if (sz <= maxsize_Small_Solver) (*sp[k]).set_Sparse_or_Small(false);

        (*sp[k]).Construct(S);
        max_size = max(max_size,sz);
        min_size = min(min_size,sz);
    }

    mout << " constructing " << K 
         << " vectors with Solver for DD" 
         << Date() - Start_Construct << endl;
    mout << "maximum size of subdomain: " << max_size << endl;
    mout << "minimum size of subdomain: " << min_size << endl;
}

void OS::Decomposition::Destruct() {
    for (int k = 0; k < K; ++k) {
        delete sp[k];
        sp[k] = 0;
    }
    sp.clear();
}
// END OF OS::Decomposition

OS::OS () {
    dout(3) << " Overlapping Schwarz: constructor" << endl;
}

void OS::Construct (const Matrix& _A) {
    A = new Matrix(_A);
    Accumulate(*A);
    decomp.Construct(*A);
    Sp = new SparseMatrix(*A);
}

void OS::Destruct () {
    if (A) delete A; A = 0;
    if (Sp) delete Sp; Sp = 0;
    decomp.Destruct();
}

void OS::multiply (Vector& u, const Vector& b) const {
    int K = decomp.size();

    Vector r = b;
    u = 0;
    double Theta = 1.0;
    for (int fw = 0; fw < forward; ++fw)
        for (int k=0; k<K; ++k) {
            int N = (decomp.subprobsize(k));
            BasicVector s(N);
            for (int n=0; n<N; ++n) 
                s[n] = r[decomp.ind(k,n)];

            if (GaussSeidel) {
                for (int i=0; i<decomp.subprobsize(k); ++i)
                    for (int d = (*Sp).rowind(decomp.ind(k,i)); 
                             d < (*Sp).rowind(decomp.ind(k,i)+1); ++d) {
                        int col = (*Sp).colptr(d);
                        s[i] -= Theta*(*Sp).nzval(d) * u[col];
                    }
            }

            decomp.Solve(k,s());
            for (int n=0; n<N; ++n)
                u[decomp.ind(k,n)] += Theta*s[n];
        }

    for (int bw = 0; bw < backward; ++bw)
        for (int k=K-2; k>=0; --k) {
            int N = (decomp.subprobsize(k));
            BasicVector s(N);
            for (int n=0; n<N; ++n)
                s[n] = r[decomp.ind(k,n)];

            if (GaussSeidel) {
                for (int i=0; i<decomp.subprobsize(k); ++i)
                    for (int d = (*Sp).rowind(decomp.ind(k,i)); 
                             d < (*Sp).rowind(decomp.ind(k,i)+1); ++d) {
                        int col = (*Sp).colptr(d);
                        s[i] -= (*Sp).nzval(d) * u[col];
                    }
            }

            decomp.Solve(k,s());
            for (int n=0; n<N; ++n)
                u[decomp.ind(k,n)] += s[n];
        }

    Accumulate(u);
}

