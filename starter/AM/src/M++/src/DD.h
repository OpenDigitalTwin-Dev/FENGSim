// file: DD.h
// author: Martin Sauter, Daniel Maurer
// purpose: to narrow Preconditioner.C
//          contains the following DD preconditioners:
//          * ADD
//          * VankaPressureNode
//          * OS
#ifndef _DD_H_
#define _DD_H_


#include "Sparse.h"
#include "Small.h"
#include "Preconditioner.h"
#include "Interface.h"
#include "Transfer.h"
#include "LinearSolver.h"

#include <set>

class VankaPressureNode : public Preconditioner {
protected:
    int K;   // number of Pressure Nodes
    int dim;
    double VankaDamping;
    vector< vector<int> > Indices;
    vector<SparseMatrix*> SMat;
    vector<SparseSolver*> SSol;
    SparseMatrix* Sp;
    const Matrix* L;
    const bool* essential; //Dirichlet () const { return A->GetVector().D(); }
    bool Essential (int i) const { return essential[i]; }
    bool multiplicative;
public:
    VankaPressureNode() : K(0), dim(0), VankaDamping(1.0), Indices(0), 
			  SMat(0), SSol(0), Sp(0), multiplicative(false) {
	ReadConfig(Settings,"VankaDamping",VankaDamping);
	ReadConfig(Settings,"VankaMulitiplicative", multiplicative);
    } 
    void Construct(const Matrix& A) {
//	Vout(5) << "////////////////  VPN.Construct ---> /////////////////////"<<endl;
	essential = A.GetVector().D();
	map<int,list<int> > rowmap;
	dim = A.GetMesh().dim();
	K= 0;
	L = &A;
	for (cell c=A.cells(); c!=A.cells_end(); ++c, ++K) {
	    Vout(9) <<"cell "<<c<<endl;
	    list<int> rowlist;
	    rows r(A,c);
	    for (int i=0; i<r.size();++i) {
		rowlist.push_back(r[i].Id());
	    }
//	    rowlist.sort();
	    Vout(9) <<"rowlist  "<< rowlist<<endl;
	    int N=0;
	    for (int i=0; i<r.size();++i) {
		if (r[i].n() == dim+1) {
		    rowmap[r[i].Id()].push_back(r[i].Id());
		    Vout(8) <<"row r["<<i<<"] = "<<r[i]()
			 <<"  with Id()="<<r[i].Id()<<endl;
		    list<int>::iterator it = rowmap[r[i].Id()].end();
		    rowmap[r[i].Id()].insert(it,rowlist.begin(),rowlist.end());
		    
		    Vout(9) << "rowmap( )"<<endl<<rowmap[r[i].Id()]<<endl;
		}
	    }
	}
	K = rowmap.size();
	Indices.resize(K);
	Vout(8) << "rowmap"<<endl<<rowmap<<endl;
	int N=0;
	for (map<int,list<int> >::iterator it=rowmap.begin(); 
	     it!=rowmap.end(); ++it,++N) {
	    it->second.sort();
	    it->second.unique();
	    Indices[N].resize(it->second.size()*dim+1);
	    int M=0;
	    for (list<int>::iterator m=it->second.begin(); m!=it->second.end(); ++m) {
		int index = A.Index(*m);
	        for (int j=0; j<dim; ++j) 
		    Indices[N][M++] = index + j;
	    }
//	    mout << "UZUZUZUZU" <<it->first<<endl;
	    Indices[N][M] = A.Index(it->first)+dim;
	    sort(Indices[N].begin(),Indices[N].end());
	}
//	mout << "rowmap"<<endl<<rowmap<<endl;
	for (int i=0; i<K; ++i) {
	    Vout(10) <<endl<<"IIIII="<<i<<endl;
	    for (int j=0; j<Indices[i].size(); ++j) 
		Vout(10)<<Indices[i][j]<<" ";
	}
	SMat.resize(K);
	SSol.resize(K);
	for (int k=0; k<K; ++k) {
	    SMat[k] = 0;
	    SSol[k] = 0;
	}
	Sp = new SparseMatrix(A);
//	SparseMatrix SS(A);
	for (int k=0; k<K; ++k) {
	    SMat[k] = new SparseMatrix(*Sp,Indices[k]);
	}
	
	for (int k = 0; k<K; ++k)
	    SSol[k] = GetSparseSolver(*SMat[k]);

//	Vout(5) <<endl<< "////////////////  <--- VPN.Construct /////////////////////"<<endl;
    }
    void Destruct() {
	if (Sp) delete Sp;
	for (int k=0; k<K; ++k) {
	    if (SMat[k]) delete SMat[k];
	    if (SSol[k]) delete SSol[k];
	    SMat[k] = 0;
	    SSol[k] = 0;
	}
    }
    void multiply (Vector& u, const Vector& b) const {
	
	Vector r(b);
	u = 0;
	Vector c(u);
//	c = 0;
	
	for (int k=0; k<K; ++k) {
	    c = 0;
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n)
		s[n] = r[Indices[k][n]];
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		c[Indices[k][n]] += VankaDamping*s[n];
	    // brute force residual update
//	    r -= (*L) * c;
//	    u += c;

//	    for (int i=0; i<Indices[k].size(); ++i,mout<<endl<<endl) {
	    if (multiplicative) {
		for (int i=0; i<Indices[k].size(); ++i) {
		    Vout(3) << "k="<<k<<"   i="<<i<<"   size="<<Indices[k].size()
			    << "   rownumber="<<Indices[k][i]<<endl
			    << "    d_0="<<(*Sp).rowind(Indices[k][i])
			    << "    d_inf="<<(*Sp).rowind(Indices[k][i]+1)<<endl
			    << "d ";
		    
		    for (int d = (*Sp).rowind(Indices[k][i]); 
			 d < (*Sp).rowind(Indices[k][i]+1); ++d) {
//		    mout<<d << " ";
			int col = (*Sp).colptr(d);
			Vout(3)<<col<<" ";
			r[col] -= (*Sp).nzval(d) * s[i];
		    }
		}
		for (int i=0; i<r.size(); ++i)
		    if (Essential(i)) r[i] = 0;
	    }

/*	for (int k=K-1; k>=0; --k) {
	    int N = Indices[k].size();
	    BasicVector s(N);
	    for (int n=0; n<N; ++n)
		s[n] = r[Indices[k][n]];
	    SSol[k]->Solve(s());
	    for (int n=0; n<N; ++n)
		u[Indices[k][n]] += s[n];
	}
*/
	}
//	Accumulate(u);
//	mout << "=================================================================="<<endl
//	     << "=================================================================="<<endl;
    }

    string Name () const { return "Vanka Pressure Node (VankaPN)"; }
};




class ADD : public Preconditioner {
    class SubProblem {
	vector<int> index;
	SparseMatrix* SMat;
	SparseSolver* SSol;
    public:
	SubProblem (int N = 0) : index(N), SMat(0), SSol(0) {}
	SubProblem (const SubProblem& sp) : 
	    index(sp.index), SMat(sp.SMat), SSol(sp.SSol) {
	    Exit("do not use!");
	}
	int& operator [] (int n) { return index[n]; }
	int operator [] (int n) const { return index[n]; }
	int size () const { return index.size(); }
	void Construct (const SparseMatrix& S) {
	    sort(index.begin(),index.end());
	    SMat = new SparseMatrix(S,index);
	    SSol = GetSparseSolver(*SMat);
	}
	void Destruct () {
	    if (SMat) delete SMat; SMat = 0;
            mout << "destructing SMAT in SubProblem" << endl;
	    if (SSol) delete SSol; SSol = 0;
            mout << "destructing SSOL in SubProblem" << endl;
	}
	void Solve (Scalar* x) { SSol->Solve(x); }

        ~SubProblem () { Destruct(); }
    };
    typedef vector<SubProblem> SubProblems;

    class CoarseProblem {
	vector< SmallVector* > c;
	SmallMatrix* C;
	SparseSolver *Sol;
	int POU;
    public:
	CoarseProblem (int M = 0) : c(M), C(0), Sol(0), POU(1) {
	    ReadConfig(Settings,"POU",POU);
	}
	int size () const { return c.size(); }
	Scalar& operator () (int i, int j) { return (*C)[i][j]; }
	Scalar operator () (int i, int j) const { return (*C)[i][j]; }
	void Construct (const SparseMatrix& S,
			const SubProblems& sp) {
	}
	void Destruct () {
	    if (C) delete C; C = 0;
            mout << "destructing C in CoarseProblem" << endl;
	    if (Sol) delete Sol; Sol = 0;
            mout << "destructing SOL in CoarseProblem" << endl;
	}
	void Solve (Scalar* x) { Sol->Solve(x); }

       ~CoarseProblem () { Destruct(); }
    };

    SubProblems SP;


    int K;
    vector< vector<int> > Indices;
    vector< list<int> > in_ind;
    
    vector<SparseMatrix*> SMat;
    vector<SparseSolver*> SSol;
    string DDType;
    int S_TO_S;
    
    int M;
    SmallMatrix* C;
// nicht mehr ben�tigt
    SparseSolver *Sol; // Solver for C
    vector< SmallVector* > c;
    
// nur noch short
    SmallMatrix* C_short;
    SparseSolver *Sol_short; // Solver for C_short
    vector< SmallVector* > c_short;
    
    int POU;
    Matrix* A;
    SparseMatrix* Sp;
    
    int CoarseCorr;
    int GaussSeidel;
    string CoarseType;
    
    int UMax; 
    int NMax; 
    int pqs; 
    int dist;
    
    int rueckiter;
    int voriter;

    const bool* Dirichlet () const { return A->GetVector().D(); }
    bool Dirichlet (int i) const { return A->GetVector().D(i); }
    
public:
    ADD();
    void Construct_Coarse_trivial (const SparseMatrix&);
    void Construct_Coarse_lin_x(const SparseMatrix&);
    void Construct_Coarse_lin_y(const SparseMatrix&);
    void Construct_Coarse_Indices (const SparseMatrix&);
    void Construct_Coarse_Indices_short (const SparseMatrix&);
    void Clear_Dirichlet(const int, const bool *);
    void Clear_Dirichlet_short(const int, const bool *);
    void Coarse_POU(const int);
    void Coarse_POU_short(const int);
    void Construct_Coarse (const SparseMatrix &, const bool* );
    void Construct_Coarse_short (const SparseMatrix &, const bool* );
    void CoarseCorrection (Vector&, Vector&) const;
    void CoarseCorrection_short (Vector&, Vector&) const;
    void Construct_LR (const Matrix&);
    void Construct_Jacobi(const Matrix&);
    void Construct_Row (const Matrix&);
    void Construct_RowOld (const Matrix&);
    void Construct_DefU (const Matrix&);
    void Construct_Near (const Matrix&);
    void Construct_Cell_with_global (const Matrix&);
    void Construct_Cell (const Matrix&);
    void create_in_ind (int);
    void Construct (const Matrix&);
    void Destruct ();
    virtual ~ADD ();
    void multiply (Vector&, const Vector&) const;
    void multiply_transpose (Vector&, const Vector&) const;
    string Name () const;
    friend ostream& operator << (ostream&, const ADD&); 
    //    friend ostream& operator << (ostream& s, const SuperLU& SLU);
};

class OS: public DDPreconditioner {

    class SubProblem {
	vector<int> index;
	SparseMatrix* SMat;
	SparseSolver* SSol;
        SmallMatrix* SmallMat; 
        SmallSolver* SmallSol;
        bool Sparse_or_Small; // true: Sparse; false: Small

    public:
	SubProblem (int N = 0) : index(N), SMat(0), SSol(0), Sparse_or_Small(true), SmallMat(0), SmallSol(0) {}
	SubProblem (const SubProblem& sp) : 
	    index(sp.index), SMat(sp.SMat), SSol(sp.SSol) {
	    Exit("do not use!");
	}

        void set_Sparse_or_Small(bool Sp_or_Sm) { Sparse_or_Small = Sp_or_Sm; }

	int& operator [] (int n) { return index[n]; }
	int operator [] (int n) const { return index[n]; }
	int size () const { return index.size(); }
        void resize(int);

	void Construct (const SparseMatrix& );
	void Destruct ();
	void Solve (Scalar*) const;
        virtual ~SubProblem () { Destruct(); }

        SparseMatrix SparseMat () const {return *SMat;}
        Scalar entry (int i, int j) const {return (*SMat).nzval((*SMat).find(i,j));}
    }; // END OF class SubProblem

    typedef vector<SubProblem*> SubProblems;

    class Decomposition {
        SubProblems sp;
        int K;
        string DDType;
        int UMax;
        int NMax;
        int pqs;
        double dist;
        int maxsize_Small_Solver; // 0: All Matrices in Sparse; 

        void Construct_Jacobi(const Matrix& );
        void Construct_Cell (const Matrix& );
	void Construct_Row (const Matrix& );
        void Construct_Cell_large (const Matrix& );

    public:
        Decomposition ();
        void Construct (Matrix& );

        int size() const {return K;}
        SubProblems subprobs() const {return sp;}
 	SubProblems& subprobs() {return sp;}
        int subprobsize(int k) const {return (*sp[k]).size();}
        SubProblem ind(int k) const {return *sp[k];}
        int ind(int k, int n) const {return (*sp[k])[n];}

        void Solve (int k, Scalar* x) const { (*sp[k]).Solve(x); }

        void Destruct();
        virtual ~Decomposition() {Destruct();}
    }; //END OF class Decomposition

    Decomposition decomp;

    const bool* Dirichlet () const { return A->GetVector().D(); }
    bool Dirichlet (int i) const { return A->GetVector().D(i); }



public:
    OS ();

    void Construct (const Matrix& );

    void Destruct ();

    virtual ~OS () { Destruct(); }

    void multiply (Vector& , const Vector& ) const;
    string Name () const { return "Overlapping Schwarz"; }
    friend ostream& operator << (ostream& s, const OS& OS) {
	return s << "Overlapping Schwarz"; }

};



#endif

