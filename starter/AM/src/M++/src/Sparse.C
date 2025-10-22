// file: Sparse.C
// author: Christian Wieners
// $Header: /public/M++/src/Sparse.C,v 1.14 2009-11-24 09:46:35 wieners Exp $

#include "Sparse.h"
#include "Small.h"

int NonZeros (const SparseMatrix& M) {
    int nz = 0;
    for (int i=0; i<M.Size(); ++i)
	if (M.nzval(i) != 0.0) ++nz;
    return nz;
}

SparseMatrix::SparseMatrix (const SparseMatrix& M) 
    : BasicSparseMatrix(M.size(),NonZeros(M)) {
    const int n = size();

    Scalar* val = this->nzval();
    int* rowind = this->rowind();
    int* ind = this->colptr();

    int nz = 0;
    rowind[0] = 0;
    for (int r=0; r<n; ++r) {
	for (int i=M.rowind(r); i<M.rowind(r+1); ++i) {
	    if (M.nzval(i) == 0.0) continue;
	    ind[nz] = M.colptr(i);
	    val[nz++] = M.nzval(i);
	}
	rowind[r+1] = nz;
    }
}

SparseMatrix::SparseMatrix (const SmallMatrix& A):
    BasicSparseMatrix(A.rows(),A.nz()) {
    A.copy(nzval(),rowind(),colptr());
}

void SparseMatrix::convert_sm(SparseMatT& M) const {
    const int n = size();
    const Scalar* val = this->nzval();
    const int* rowind = this->rowind();
    const int* ind = this->colptr();
    
    M.clear();
    M.resize(n);
    for (int r = 0; r < n; ++r) {
	SparseVecT& row = M[r];
	for (int i = rowind[r]; i < rowind[r+1]; ++i) row[*(ind++)] = *(val++);
    }
}

void SparseMatrix::print_sm(const SparseMatT& M) const {
    for (int i = 0; i < M.size(); ++i) {
	const SparseVecT& r = M[i];
	mout<<"row "<<i<<": ";
	for (conSparVecIt it = r.begin(); it != r.end(); ++it)
	    mout<<'('<<it->first<<';'<<double_of_Scalar(it->second)<<"),";
	mout<<endl;
    }
    mout<<endl;
};

void SparseMatrix::print_sm_dense(const SparseMatT& M) const {
    printf("%3c ","");
    for(int i = 0; i < M.size(); ++i) printf("%5i ",i);
    mout<<endl;
    for (int i = 0; i < M.size(); ++i) {
	const SparseVecT& r = M[i];
	conSparVecIt rend = r.end();
	printf("%3i ",i);
	for (int j = 0; j < M.size(); ++j) {
	    conSparVecIt it = r.find(j);
	    printf("%5.2f ",
		   (it == rend)?0.0:double_of_Scalar(it->second));
	}
	mout<<endl;
    }
    mout<<endl;
};

void SparseMatrix::print_id_dense(const DofDofId& id) const {
    for(int i = 0; i < id.size(); ++i) printf("%3i ",i);
    mout<<endl;
    for(int i = 0; i < id.size(); ++i) printf("%3i ",id[i]);
    mout<<"\n\n";
}

void SparseMatrix::convert_sm_back(const SparseMatT& M) {
    const int n = M.size();
    int rowpos = 0;
    Scalar* val = this->nzval();
    int* rowind = this->rowind();
    int* ind = this->colptr();
    
    for (int r = 0; r < n; ++r) {
	const SparseVecT& row = M[r];
	rowind[r] = rowpos;
	for (conSparVecIt it = row.begin(); it != row.end(); ++it) {
	    *(val++) = it->second;
	    *(ind++) = it->first;
	    rowpos++;
	}
    }
    rowind[n] = rowpos;
}

void SparseMatrix::build_ident(const Matrix& u) {
   int nDof = u.size();
   
   ParDofId.resize(nDof);
   for (int i = 0; i < nDof; ++i) ParDofId[i] = i;
   for (identifyset is = u.identifysets(); is != u.identifysets_end(); ++is) {
       const Point& pmid = is();
       const int mid = u.Id(pmid);
       int i;
       for (i = 0; i < is.size(); ++i)
	   if (pmid > is[i]) break;
       if (i == is.size() ) {
	   for (int i = 0; i < is.size(); ++i) {
	       ParDofId[u.Id(is[i])] = mid;
	   }
	   nDof -= is.size();
       }
   }
   ParDofRes.resize(nDof);
//   mout<<"rest dof="<<nDof<<" of "<<ParDofId.size()<<endl;
//   mout<<"first run\n"<<ParDofId<<endl;
   int pos = 0;
   for (int i = 0; i < ParDofId.size(); ++i)
       if (ParDofId[i] == i) {
	   ParDofId[i] = pos;
	   ParDofRes[pos++] = i;
       }
       else if (ParDofId[i] < pos) ParDofId[i] = ParDofId[ParDofId[i] ];
       else ParDofId[i] = -ParDofId[i];
   for (int i = 0; i < ParDofId.size(); ++i)
       if (ParDofId[i] < 0) ParDofId[i] = ParDofId[-ParDofId[i] ];
}

void SparseMatrix::apply_PC_mat(SparseMatT& D, const SparseMatT& S) const {
    D.clear();
    D.resize(ParDofRes.size());
    for (int r = 0; r < S.size(); ++r) {
	const SparseVecT& srow = S[r];
	SparseVecT& drow = D[ParDofId[r] ];
	for (conSparVecIt it = srow.begin(); it != srow.end(); ++it)
	    drow[ParDofId[it->first] ] += it->second;
    }
}

void SparseMatrix::pc_mat_convert (const Matrix& A) {
    build_ident(A);
    SparseMatT M,Mpc;
    convert_sm(M);
    apply_PC_mat(Mpc,M);
    resize(ParDofRes.size(),Size());
    convert_sm_back(Mpc);
}

void SparseMatrix::ExpandIdentify (Vector& u) const { 
    Vector v(u);
    for (int i=0; i<u.size(); ++i) u[i] = v[ParDofId[i]];
}

void SparseMatrix::ShrinkIdentify (Vector& u, const Vector& b) const { 
    u = 0;
    for (int i=0; i<u.size(); ++i) u[ParDofId[i]] += b[i];
}



#ifdef SUPERLU

/*
* -- SuperLU routine (version 3.0) --
* Univ. of California Berkeley, Xerox Palo Alto Research Center,
* and Lawrence Berkeley National Lab.
*/

#include <cstdlib>
#undef TRUE
#undef FALSE

#ifdef DCOMPLEX 
#define Create_Dense_Matrix zCreate_Dense_Matrix
#define Create_CompCol_Matrix zCreate_CompCol_Matrix
#define Print_CompCol_Matrix zPrint_CompCol_Matrix
#define Print_SuperNode_Matrix zPrint_SuperNode_Matrix
#define gstrs zgstrs
#define gstrf zgstrf
#define SLU_DT SLU_Z

typedef complex<double> doublecomplex;

#include "superlu/slu_zdefs.h"
#else
#define Create_CompCol_Matrix dCreate_CompCol_Matrix
#define Create_Dense_Matrix dCreate_Dense_Matrix
#define Print_CompCol_Matrix dPrint_CompCol_Matrix
#define Print_SuperNode_Matrix dPrint_SuperNode_Matrix
#define gstrs dgstrs
#define gstrf dgstrf
#define SLU_DT SLU_D
#include "superlu/slu_ddefs.h"
#endif

class SuperSolver : public SparseSolver {
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int *etree;
    trans_t  trans;

    SuperMatrix A;
    NCformat *Astore;
    double   *a;
    int      *asub, *xa;
    int      *perm_c; /* column permutation vector */
    int      *perm_r; /* row permutations from partial pivoting */
    SuperMatrix L;      /* factor L */
    SCformat *Lstore;
    SuperMatrix U;      /* factor U */
    NCformat *Ustore;
    
    int      ldx, info, m, n, nnz;
    double   *xact, *rhs;
    mem_usage_t   mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
    GlobalLU_t Glu;
    
    void slu_decompose() {
        int      lwork = 0, i;
        
        /* Set default values for some parameters */
        int      panel_size;     /* panel size */
        int      relax;          /* no of columns in a relaxed snodes */
        int      permc_spec;

        trans = NOTRANS;
        if ( A.Stype == SLU_NR ) {
            NRformat *Astore = (NRformat *) A.Store;
            AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
            Create_CompCol_Matrix(AA, A.ncol, A.nrow, Astore->nnz, 
                                  (Scalar*) Astore->nzval, Astore->colind, Astore->rowptr,
                                  SLU_NC, A.Dtype, A.Mtype);
            trans = TRANS;
        } else {
            if ( A.Stype == SLU_NC ) AA = &A;
        }

        permc_spec = options.ColPerm;
        if ( permc_spec != MY_PERMC && options.Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
        etree = intMalloc(A.ncol);
        
        sp_preorder(&options, AA, perm_c, etree, &AC);
        panel_size = sp_ienv(1);
        relax = sp_ienv(2);

        /* Compute the LU factorization of A. */
#ifdef SUPERLU30
        double drop_tol = 0.;
        gstrf(&options, &AC, drop_tol, relax, panel_size, etree,
                NULL, lwork, perm_c, perm_r, &L, &U, &stat, &info);
#else
	gstrf(&options, &AC, relax, panel_size, etree,
	      NULL, lwork, perm_c, perm_r, &L, &U, &Glu, &stat, &info);


	
	
#endif        
    }

    void slu_solve(SuperMatrix *B) {
        DNformat *Bstore;
        Bstore = (DNformat *) B->Store;
        if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A.nrow) ||
                  B->Stype != SLU_DN || B->Dtype != SLU_DT || 
                  B->Mtype != SLU_GE )
            info = -7;
        if ( info != 0 ) {cerr << "superlu error in slu_solve"; exit(0);}
        gstrs (trans, &L, &U, perm_c, perm_r, B, &stat, &info);
    }
 public:
    SuperSolver (BasicSparseMatrix& M) {

        set_default_options(&options);
        StatInit(&stat);
        n = m = M.size();
        perm_r = intMalloc(m);
        perm_c = intMalloc(n);
        Create_CompCol_Matrix(&A, m, n, M.Size(), 
                               M.nzval(),
                               M.colptr(),
                               M.rowind(), 
                               SLU_NR, SLU_DT, SLU_GE);
        slu_decompose();
    }
    void Solve (Scalar* b, int nrhs = 1) {
      
        if (nrhs == 0) return;
        SuperMatrix B;

        Create_Dense_Matrix(&B, m, nrhs, b, m, SLU_DN, SLU_DT, SLU_GE);

        slu_solve(&B);



	// ********** by Jiping ******************
	double *sol = (double*) ((DNformat*) B.Store)->nzval;
	for (int i=0; i<m; i++)
	    b[i] = sol[i];
	// *************************************** 
	


	

	
        Destroy_SuperMatrix_Store(&B);
    }
    ~SuperSolver () {   
        SUPERLU_FREE (etree);
        Destroy_CompCol_Permuted(&AC);
        if ( A.Stype == SLU_NR ) {
            Destroy_SuperMatrix_Store(AA);
            SUPERLU_FREE(AA);
        }
        StatFree(&stat);
        SUPERLU_FREE (perm_r);
        SUPERLU_FREE (perm_c);
        Destroy_SuperMatrix_Store(&A);
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);

    }

    void test_output() {
        char CompColL[] = "CompColL";
        Print_CompCol_Matrix(CompColL, &L);
        char CompColU[] = "CompColU";
        Print_CompCol_Matrix(CompColL, &U);
        char SuperNodeL[] = "SuperNodeL";
        Print_SuperNode_Matrix(SuperNodeL, &L);
    }
};
#endif

#ifdef UMFSOLVER


#include <stdio.h>
#include "../include/umfpack_include/umfpack.h"
#include <cmath>
#include <algorithm>

class SparseMatrixDummy {
    int col;
    double val;
public:
    int& Col() { return col; }
    double& Val() { return val; }
    int Col() const { return col; }
    double Val() const { return val; }
};

bool SparseMatrixDummyCompare(SparseMatrixDummy A, SparseMatrixDummy B) {
    return (A.Col() <= B.Col()); 
}



class UMFSolver : public SparseSolver {
    int n;   // dimension
    int m;   // nz elements
    int *Ap;
    int *Ai;
    Scalar *Ax;
    void *Symbolic, *Numeric ;
    double *null;
    int verbose;
public:
    UMFSolver(BasicSparseMatrix& M) : verbose(0)  {
	n = M.size();
	m = M.Size();
//	ReadConfig(Settings,"UMFverbose",verbose);
//	Vout(3) << "UMF      n="<<n<<"     m="<<m<<endl;
	null = (double *) NULL ;
	Ap = new int [n+1];
	for (int i=0; i<n+1; ++i) Ap[i] = M.rowind(i);
	for (int i=0; i<n+1; ++i) Vout(5) << Ap[i]<<" ";
	Vout(5)<<endl<<endl;
	Ai = new int [m];
	Ax = new double [m];
	for (int i=0; i<n; ++i) {
	    int k=Ap[i];
	    int d=Ap[i+1]-k;
	    vector<SparseMatrixDummy> D(d);
	    for (int j=0; j<d; ++j) {
		D[j].Col() = M.colptr(k+j);
		D[j].Val() = M.nzval(k+j);
	    }
	    sort(D.begin(),D.end(),SparseMatrixDummyCompare);
	    for (int j=0; j<d; ++j) {
		Ai[k+j] = D[j].Col();
		Ax[k+j] = D[j].Val();
	    }
	}
	int status=0;
	Vout(5) << "check1 "<<status<<endl;
        status = umfpack_di_symbolic (n, n, Ap,Ai,Ax,
				      &Symbolic, null, null) ;
	Vout(5) << "check2 "<<status<<endl;
	status= umfpack_di_numeric (Ap,Ai,Ax,
				   Symbolic, &Numeric, null, null) ;
	Vout(5) << "check3 "<<status<<endl;
	umfpack_di_free_symbolic (&Symbolic) ;
	Vout(5) << "check4 "<<endl;

//	SmallMatrix SmM(M);
    }
    void Solve (Scalar*u, const Scalar* b) {
	int status=0;
	status= umfpack_di_solve (UMFPACK_At, Ap,Ai,Ax,
				  u, b, Numeric, null, null) ;
	Vout(5) << "check5 "<<status<<endl;
    }
    ~UMFSolver () {
	umfpack_di_free_numeric (&Numeric) ;
	delete [] Ap;
	delete [] Ai;
	delete [] Ax;
    }
};

#endif


SparseSolver* GetSparseSolver (BasicSparseMatrix& S) {
#ifdef SUPERLU
    return new SuperSolver(S);
#endif
#ifdef UMFSOLVER
    return new UMFSolver(S);
#endif
    Exit("Sparsesolver not compiled. File: " + string(__FILE__));
}

SparseSolver* GetSparseSolver (BasicSparseMatrix& S, const string& name) {
#ifdef SUPERLU
    if (name=="SuperLU")  {
	return new SuperSolver(S);
    }
#endif
#ifdef UMFSOLVER
    if (name=="UMFSolver")  return new UMFSolver(S);
#endif
    Exit("Sparsesolver " + name+ " not compiled. File: " + string(__FILE__));
}
