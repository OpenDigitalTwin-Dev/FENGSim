// file:   Schur.C
// author: Daniel Maurer
// $Header: /public/M++/src/Schur.C,v 1.17 2009-11-24 09:46:35 wieners Exp $

// used configs:
// invert: invert Small-matrices instead using LU-decomposition
// checkdiagonal: true/false
// Smalltype: use only Small-matrices, no Sparse-matrices
// printout: proc-list etc.

#include "Schur.h"
#include "Small.h"
#include "Preconditioner.h"
#include "Lapdef.h"
#include "Time.h"
#include "SVD.h"

enum MATRIXTYPE { SPARSE = 0, SMALL = 1 };

ostream& operator << (ostream&, const SmallMatrixTrans&);

class DDProblem {
public:
    DDProblem () {};

    virtual void LU(bool, DateTime*, DateTime*) = 0;
    virtual void SolveLU(DDProblem&, DateTime* DT1 = NULL, DateTime* DT2 = NULL ) = 0;
    virtual void SolveLU(SmallVector&, DateTime* DT1 = NULL, DateTime* DT2 = NULL ) = 0;
    virtual void SolveLU(Scalar*, int, DateTime* DT1 = NULL, DateTime* DT2 = NULL, bool trans = 0 ) = 0;

    void Destruct();

    virtual void SendMatrix (Buffer&) = 0;
    virtual size_t GetSizeSendMatrix () = 0;
    virtual void AddMatrix (Buffer&) = 0;
    virtual void AddMatrix_trans (Buffer&) = 0;
    virtual void SetMatrix (Buffer&) = 0;
    virtual void SetMatrix_trans (Buffer&) = 0;

    virtual void SendIPIV (Buffer&) = 0;
    virtual size_t GetSizeSendIPIV() = 0;
    virtual void SetIPIV (Buffer&) = 0;

    virtual const Scalar operator () (int i, int j) const = 0;
    virtual Scalar& operator () (int i, int j) = 0;

    virtual int rows() const = 0;
    virtual int cols() const = 0;

    friend ostream& operator << (ostream& s, const DDProblem& DDP) {
        char buf[128];
        for (int i=0; i<DDP.rows(); ++i, s << endl) 
            for (int j=0; j<DDP.cols(); ++j) {
                sprintf(buf,"%9.5f",double_of_Scalar(DDP(i,j)));	
                s << buf;
//                 s << " " << DDP(i,j);
        }
    return s;
    }

    virtual int gettype() const = 0;

    virtual Scalar* ref() = 0;
    virtual Scalar* ref() const = 0;
    virtual BasicSparseMatrix* refM() = 0;
    virtual BasicSparseMatrix* refM() const = 0;

    virtual DDProblem& operator -= (const constAB<DDProblem,DDProblem>& ) = 0;

//     Scalar normF();

};

void MultiplySubtract(const DDProblem& A, const valarray<Scalar>& a, 
		      valarray<Scalar>& b) {
    for (int j=0; j<A.rows(); ++j) {
        Scalar s = 0;
        for (int k=0; k<a.size(); ++k) 
            s += A(j,k) * a[k];
        b[j] -= s;
    }
}

void MultiplySubtract(const DDProblem& A, const Scalar* a, Scalar* b) {
    for (int j=0; j<A.rows(); ++j) {
        Scalar s = 0;
        for (int k=0; k<A.cols(); ++k)
            s += A(j,k) * (*(a+k));
        (*(b+j)) = (*(b+j)) - s;
    }
}

void MultiplySubtract_trans(const DDProblem& A, const Scalar* a, Scalar* b) {
    for (int j=0; j<A.cols(); ++j) {
        Scalar s = 0;
        for (int k=0; k< A.rows(); ++k)
            s += A(k,j) * (*(a+k));
        (*(b+j)) = (*(b+j)) - s;
    }
}

class SparseDDProblem: public DDProblem {
    int n,m;
    BasicSparseMatrix *M;
    SparseSolver* S;
    bool lu;
    bool svd;
    vector<ProcSet> PS;

public: 
    SparseDDProblem(int _N, int _M, bool SVD): n(_N), m(_M), lu(false), svd(SVD), M(0), S(0) {}

    void Destruct() {
        if (M) delete M; M = 0;
        if (S) delete S; S = 0;
        PS.clear();
    }

    ~SparseDDProblem() {Destruct();}

    void setSparse(const SparseMatrix&, const vector<int>&, const vector<int>& );

    SparseDDProblem(int _N, int _M, const SparseMatrix& A, const vector<int>& Indices, const vector<int>& Indices2, bool SVD): n(_N), m(_M), svd(SVD) {
        setSparse(A, Indices, Indices2);
    }

    void LU(bool, DateTime*, DateTime*);
    void SolveLU(DDProblem&, DateTime*, DateTime*);
    void SolveLU(SmallVector&, DateTime*, DateTime*);
    void SolveLU(Scalar*, int, DateTime*, DateTime*, bool);

    int rows() const {return n;}
    int cols() const {return m;}
    const Scalar operator () (int i, int j) const {
            int fd = (*M).find(i,j);
            if (fd == -1) return 0;
            return (*M).nzval(fd);
    }

    Scalar& operator () (int i, int j) {
        mout << "warning! no assignment for Sparse possible!\n"; exit(0);
    }

    void SendMatrix (Buffer& B) {mout << "no SendMatrix for Sparse implemented\n"; exit(0);}
    size_t GetSizeSendMatrix () {mout << "no GetSizeSendMatrix for Sparse implemented\n"; exit(0);}
    void AddMatrix (Buffer& B)  {mout << "no AddMatrix for Sparse implemented\n"; exit(0);}
    void SetMatrix (Buffer& B)  {mout << "no SetMatrix for Sparse implemented\n"; exit(0);}
    void AddMatrix_trans (Buffer& B)  {mout << "no AddMatrix for Sparse implemented\n"; exit(0);}
    void SetMatrix_trans (Buffer& B)  {mout << "no SetMatrix for Sparse implemented\n"; exit(0);}

    void SendIPIV (Buffer& B)   {mout << "no SendIPIV for Sparse implemented\n"; exit(0);}
    size_t GetSizeSendIPIV()    {mout << "no GetSizeSendIPIV for Sparse implemented\n"; exit(0);}
    void SetIPIV (Buffer& B)    {mout << "no SetIPIV for Sparse implemented\n"; exit(0);}

    Scalar* ref() {mout << "no ref() for Sparse implemented\n"; exit(0);}
    Scalar* ref() const {mout << "no const ref() for Sparse implemented\n"; exit(0);}
    BasicSparseMatrix* refM() {return M;}
    BasicSparseMatrix* refM() const {return M;}

    DDProblem& operator -= (const constAB<DDProblem,DDProblem>& AB) {mout << "Operator -= for Sparse not implemented\n"; exit(0);}

    int gettype() const {return MATRIXTYPE(SPARSE);}
};

class SmallDDProblem: public DDProblem {
    SmallMatrixTrans M;
    bool svd;

    bool symmetric;
    bool cholesky;

public:
    SmallDDProblem(int _N, int _M, bool SVD, bool symm = false, bool ltl = false): M(_N,_M,ltl), svd(SVD), symmetric(symm), cholesky(ltl) {
    }

    int rows() const {return M.rows();}
    int cols() const {return M.cols();}

    const Scalar operator () (int i, int j) const {
        return M(i,j);
    }
    Scalar& operator () (int i, int j) {
        return M(i,j);
    }

    void Destruct() {}

    ~SmallDDProblem() {Destruct();}

    void LU(bool, DateTime*, DateTime*);

    void SolveLU(DDProblem&, DateTime*, DateTime*);
    void SolveLU(SmallVector&, DateTime*, DateTime*);
    void SolveLU(Scalar*, int, DateTime*, DateTime*, bool);

    void SendMatrix (Buffer&);
    size_t GetSizeSendMatrix ();
    void AddMatrix (Buffer&);
    void SetMatrix (Buffer&);
    void AddMatrix_trans (Buffer&);
    void SetMatrix_trans (Buffer&);
    void SendIPIV (Buffer&);
    size_t GetSizeSendIPIV();
    void SetIPIV (Buffer&);

    Scalar* ref() {return M.ref();}
    Scalar* ref() const {return M.ref();}
    BasicSparseMatrix* refM() {mout << "no BasicSparseMatrix for reference in SMALL\n";exit(0);}
    BasicSparseMatrix* refM() const {mout << "no BasicSparseMatrix for reference in SMALL\n";exit(0);}


    int gettype() const {return MATRIXTYPE(SMALL);}

    DDProblem& operator -= (const constAB<DDProblem,DDProblem>&);

};

class vecProcSet: public vector<ProcSet> {
 vector<int> sz;
 public:
    vecProcSet() {};
    vecProcSet(ProcSet P): vector<ProcSet>(1), sz(1) {(*this)[0] = P; sz[0] = -1;}

    int findequalProcSet(const ProcSet& P) {
        for (int i=0; i<(*this).size(); ++i)
            if ((*this)[i].equalProcSet(P)) return i;
        return -1;
    }

    void Add(const ProcSet& P, int s = -1) {
        if (findequalProcSet(P) == -1) {
            int m = size();
            resize(m+1);
            sz.resize(m+1);
            (*this)[m] = P;
            sz[m] = s;
        }
    }

    ProcSet GetPS(int i) const {
        return (*this)[i];
    }

    int GetSize(int i) const {
        return sz[i];
    }

    void SetSize(int i,int s) {
        sz[i] = s;
    }

    int min_PS(const ProcSet& P1) {
        int m = P1[0];
        for (int i=1; i<P1.size(); ++i)
            if (P1[i] < m) m = P1[i];
        return m;
    }

    void swap(int i, int j) {
        ProcSet PS = (*this)[j];
        (*this)[j] = (*this)[i];
        (*this)[i] = PS;
        int s = sz[j];
        sz[j] = sz[i];
        sz[i] = s;
    }

    void Sort() {
        for (int i=0; i<size(); ++i)
            for (int j=i+1; j<size(); ++j)
                if ((*this)[j].size() < (*this)[i].size()) {
                    swap(i,j);
                }
        for (int i=0; i<size(); ++i)
            for (int j=i+1; j<size(); ++j)
                if ((*this)[i].size() == (*this)[j].size()) {
                    ProcSet Phelp1;
                    Phelp1.Add((*this)[i]);
                    ProcSet Phelp2;
                    Phelp2.Add((*this)[j]);
                    while (1) {
                        int minP1 = min_PS(Phelp1);
                        int minP2 = min_PS(Phelp2);
                        if (minP1 == minP2) {
                            Phelp1.erase(minP1);
                            Phelp2.erase(minP2);
                        } else
                        if (minP1 > minP2) {
                            swap(i,j);
                            break;
                        } else break;
                    }
                }
    }

    ~vecProcSet() {(*this).clear();}
};


class DDind {
    vector<int> IND;
    vector<int> block; 

    vecProcSet vPS;

    vector<vector<int> > invIND;
    vector<vector<int> > invIND0;

    class Less {
    public:
        bool operator () (const row& r0, const row& r1) const {
            return (r0() < r1()); 
        }
    };

public:
    DDind(const DDind&);
    void dissect (const Vector&, vector<int>&, vector<vector<int> >&, vector<int>&);
    void Setind(const Vector&);
    DDind(const Vector&);

    void Destruct();
    virtual ~DDind() {Destruct();}

    int size(int ) const;
    int Size() const;

    int find_block(int ) const;

    int blockentry(int ) const;

    int ind (int i) const {return IND[i];}
    int ind (int i) {return IND[i];}

    vector<int>& invind(int i) {return invIND[i];}

    vector<int>& invind0(int i) {return invIND0[i];}

    ProcSet GetProc (int ) const;
    vecProcSet GetProc() const;

    int findequalProcSet(ProcSet PS) {
        return vPS.findequalProcSet(PS);
    }
};

class ddind {
    DDind* IND;
public:
    ddind (DDind& ind) : IND(&ind) {}
    ddind (const ddind& ind) : IND(ind.IND) {}
    void Destruct() {
//         delete[] IND;
    }
    ~ddind() {
        Destruct();
    }
    int ind (int i) const {return IND->ind(i);}
    int ind (int i) {return IND->ind(i);}
    int size(int i) const {return IND->size(i);}
    int Size() const {return IND->Size();}
    int find_block(int i) const {return IND->find_block(i);}
    int blockentry(int i) const {return IND->blockentry(i);}
    const ddind& ref() const {return *this;}
    ProcSet GetProc (int i) const {return IND->GetProc(i);}
    vecProcSet GetProc() const {return IND->GetProc();}
    vector<int>& invind(int i) {return IND->invind(i);}
    vector<int>& invind0(int i) {return IND->invind0(i);}

    int findequalProcSet(ProcSet PS) {
        return IND->findequalProcSet(PS);
    }
};

class DDVector;

class DDMatrix : public ddind {
    int verbose;
    vector<vector<DDProblem*> > DPM;

    vector<vector<DDProblem*> > M;
    vector<vector<DDProblem*> > M0;
    vector<ProcSet> PM;

    void get_startend(const int&, int&, int&) const;

    class Times {
      public:
        DateTime Communication;
        DateTime LU;
        DateTime SolveLU;
        DateTime MatrixMultiplication;
        DateTime Blockdecomposition;
        Times(string prefix = "", string postfix = "") {
            Communication.SetName(prefix + "Communication" + postfix);
            LU.SetName(prefix + "LU" + postfix);
            SolveLU.SetName(prefix + "SolveLU" + postfix);
            MatrixMultiplication.SetName(prefix + "Matrix Multiplication" + postfix);
            Blockdecomposition.SetName(prefix + "Blockdecomposition" + postfix);
        }
        void ResetTime() {
            Communication.ResetTime();
            LU.ResetTime();
            SolveLU.ResetTime();
            MatrixMultiplication.ResetTime();
            Blockdecomposition.ResetTime();
        }
        void SetMax() {
             Communication.SetMax();
             LU.SetMax();
             SolveLU.SetMax();
             MatrixMultiplication.SetMax();
             Blockdecomposition.SetMax();
        }

        friend ostream& operator << (ostream& s, const Times& TS) {
            s << "Time of " << TS.Communication.GetName() << " " 
              << TS.Communication.GetTime() << " sec.\n";
            s << "Time of " << TS.LU.GetName() << " " 
              << TS.LU.GetTime() << " sec.\n";
            s << "Time of " << TS.SolveLU.GetName() << " " 
              << TS.SolveLU.GetTime() << " sec.\n";
            s << "Time of " << TS.MatrixMultiplication.GetName() << " " 
              << TS.MatrixMultiplication.GetTime() << " sec.\n";
            s << "Time of " << TS.Blockdecomposition.GetName() << " " 
              << TS.Blockdecomposition.GetTime() << " sec.\n";
        return s;
        }
    };

    Times times;
    Times times_step;

    bool invert;
    bool checkdiagonal;
    bool ILU;
    bool cholesky;

    int Smalltype;
    int printout;

    bool DDP_svd;
    bool symmetric;

    bool parallel_mm;

    bool dissect0;

    class class_Solve_step {
        int start;
        int end;
        int SendToL;
        bool SendToU;
        struct struct_send_special {
            int vec;
            int SendTo;
        };

        ProcSet akt_procs;

        vector<struct struct_send_special> send_special;

        vector<int> K_op;
        vector<int> K_step;
        vector<int> K_delta;
      public:
        class_Solve_step(int s, int e, int stl) : start(s), end(e), SendToL(stl), SendToU(false) {
            send_special.resize(0);
            send_special.resize(0);
        }

        void Destruct() {
            K_op.clear();
            K_step.clear();
            K_delta.clear();
            send_special.clear();
        }

        ~class_Solve_step() {Destruct();}

        void addto_send_special(int v, int st) {
            int m=send_special.size();
            for (int i=0; i<m; ++i)
                if ((send_special[i].vec == v) && (send_special[i].SendTo == st)) return;
            send_special.resize(m+1);
            send_special[m].vec = v;
            send_special[m].SendTo = st;
        }

        int getstart() {return start;}
        int getend() {return end;}
        int getSendToL() {return SendToL;}
        void setSendToL(int s) {SendToL = s;}
        bool getSendToU() {return SendToU;}

        int get_send_special_vec(int i) {
            return send_special[i].vec;
        }
        int get_send_special_SendTo(int i) {
            return send_special[i].SendTo;
        }

        int send_special_size() {
            return send_special.size();
        }

        void Set_SendToU(bool b) {
            SendToU = b;
        }

        void setend(int i) {end=i;}
        void set_akt_procs(ProcSet P) {
            akt_procs.Add(P);
        }

        ProcSet get_akt_procs() {return akt_procs;}

        void set_K_op_step(vector<ProcSet>& P);
        vector<int> get_K_step() {return K_step;}
        vector<int> get_K_op() {return K_op;}
        vector<int> get_K_delta() {return K_delta;}
    };

    vector<class_Solve_step*> Solve_step;


public:
    DDMatrix (const DDMatrix& DDM) : ddind(DDM), DPM(DDM.DPM), invert(DDM.invert), checkdiagonal(DDM.checkdiagonal), ILU(DDM.ILU), Smalltype(DDM.Smalltype), printout(DDM.printout), DDP_svd(DDM.DDP_svd), parallel_mm(DDM.parallel_mm), verbose(DDM.verbose), dissect0(DDM.dissect0), cholesky(DDM.cholesky) {}
    DDMatrix (const ddind& d, bool inv, bool chdg, bool ilu, int smtype, int prout, bool svd, bool parallel, int verb = 0, bool symm = false, bool diss = false, bool chol = false) : ddind(d), DPM(0), invert(inv), checkdiagonal(chdg), ILU(ilu), Smalltype(smtype), printout(prout), DDP_svd(svd), parallel_mm(parallel), verbose(verb), times(), times_step(""," in this step"), symmetric(symm), dissect0(diss), cholesky(chol) {}

    bool int_in_PS(const int&, const ProcSet&) const;
    void Send_PS(ExchangeBuffer&, const int &, const ProcSet&, const int&) const;
    size_t GetSizeSend_PS(const ProcSet&) const;
    void Receive_PS(ExchangeBuffer&, const int&, ProcSet&, int&) const;
    void Send_Matrix(ExchangeBuffer&, const int, DDProblem&, int) const;
    size_t GetSizeSend_Matrix(DDProblem&, int) const;
    size_t GetSizeSend_Matrix_with_PS(ProcSet&, ProcSet&, DDProblem&, int) const;

    void Receive_Matrix(ExchangeBuffer&, const int, DDProblem&, bool, int) const;

    void Receive_Matrix_trans(ExchangeBuffer&, const int, DDProblem&, bool, int) const;
    void Send_Matrix_to_Buffer(ExchangeBuffer&, const int, ProcSet&, ProcSet&, DDProblem&, int) const;
    void Receive_Matrix_from_Buffer(ExchangeBuffer&, const int, vector<ProcSet>&, vector<vector<DDProblem*> >&, bool, int, int) const;
    void Send_Vector(ExchangeBuffer&, const int, SmallVector&) const;
    void Receive_Vector(ExchangeBuffer&,const int, SmallVector&, bool) const;

    void Set(const Matrix& );

    void Fill_steplist();

    void Subtract(DDProblem&, DDProblem&, DDProblem&, DateTime*, DateTime*);
    void SubtractTransposed(DDProblem&, DDProblem&, DDProblem&, DateTime*, DateTime*);

    void Invert_M0();
    void Solve_M0(Scalar*) const;
    void Dissect_M0(vector<int>);

    void Invert_parallel(int);
    void Invert_parallel_chol(int);
    void Dissect(vector<int>, vector<int>);
    void Dissect_chol(vector<int>, vector<int>);
    void Send(ExchangeBuffer&, int ps, vector<int>);
    void Invert();

    void Solve_L(DDVector&, const int) const;
    void Solve_L_chol(DDVector&, const int) const;
    void Solve_U(DDVector&, const int) const;
    void Solve_U_chol(DDVector&, const int) const;

    void Communicate_in_L(DDVector&, const int) const;
    void Communicate_in_U(DDVector&, const int) const;

    void SolveLU(DDVector&) const;

    bool getcholesky() const {return cholesky;}

    friend ostream& operator << (ostream& s, const DDMatrix& DDM) {
        if (DDM.getcholesky())
        for (int i=0; i<DDM.DPM.size(); ++i)
            for (int j=i; j<DDM.DPM[i].size(); ++j)
                s << "M[" << i << "][" << j << "] on P " << DDM.ddind::GetProc(i) 
                  << "  x " << DDM.ddind::GetProc(j) << " with type = " << (*DDM.DPM[i][j]).gettype() << endl << *DDM.DPM[i][j] << endl;
        if (!DDM.getcholesky())
        for (int i=0; i<DDM.DPM.size(); ++i)
            for (int j=0; j<DDM.DPM[i].size(); ++j)
                s << "M[" << i << "][" << j << "] on P " << DDM.ddind::GetProc(i) 
                  << "  x " << DDM.ddind::GetProc(j) << " with type = " << (*DDM.DPM[i][j]).gettype() << endl << *DDM.DPM[i][j] << endl;
        return s << endl;
    }

    void Destruct() {
        for (int i=0; i<M.size(); ++i)
            for (int j=0; j<M[i].size(); ++j) {
                if (M[i][j]) delete M[i][j];
                M[i][j] = 0;
            }
        for (int i=0; i<DPM.size(); ++i)
            for (int j=0; j<DPM[i].size(); ++j)
                DPM[i][j] = 0;
//                 delete DPM[i][j];
        for (int i=0; i<M.size(); ++i)
            M[i].clear();
        M.clear();
        for (int i=0; i<DPM.size(); ++i)
            DPM[i].clear();
        DPM.clear();
        PM.clear();
        for (int i=0; i<M0.size(); ++i)
            for (int j=0; j<M0[i].size(); ++j) {
                if (M0[i][j]) delete M0[i][j];
                M0[i][j] = 0;
            }
    }

    virtual ~DDMatrix() {Destruct();}

    int size(int i) const {return ddind::size(i);}
    int Size() const {return ddind::Size();}

    const ddind& ref() const {return ddind::ref();}

    vector<ProcSet>& getPM() {return PM;}
};

class DDVector : public ddind {
    vector<SmallVector*> v;

    vector<SmallVector*> v4;

 public:
    DDVector (const ddind& d) : v(d.Size()), ddind(d) {}
    DDVector (const DDMatrix& D): v(D.Size()), ddind(D.ref()) {}
    DDVector (const DDVector& DDU) : v(DDU.v), ddind(DDU) {}
    void Set (const Vector& );

    void Destruct() {
/*        for (int i=0; i<v4.size(); ++i)
            delete v4[i];
        v4.clear();
        for (int i=0; i<v.size(); ++i)
            v[i] = 0;
        v.clear();*/
    }

    ~DDVector() {Destruct();}

    int size(int i) const {return ddind::size(i);}
    int Size() const {return ddind::Size();}

    void Communicate ();

    SmallVector& getv(int i) { return (*v4[i]);}

    void createnewvec(int i, int size) {
        v4[i]=new SmallVector(size);
        for (int j = 0; j< size; ++j) (*v4[i])[j] = 0;
    }

    void Setv(vector<ProcSet>& PM) {
        v4.resize(PM.size());
        for (int i=0; i<v4.size(); ++i) {
            int p = ddind::findequalProcSet(PM[i])+1;
            if (p != 0 || PM[i].size()==1) { 
                v4[i] = v[p];
           }
        }
    }

    void writevector(Vector& ) const;

    friend ostream& operator << (ostream& s, const DDVector& DDu) {
        for (int i=0; i<DDu.v.size(); ++i)
            s << "v[" << i << "]" << endl << *DDu.v[i];
        return s << endl;
    }
};

class Schur: public Preconditioner {
    DDMatrix* DDM;
    DDind* ind;

    bool DDP_svd;
    bool DDM_invert;
    bool DDM_checkdiagonal;
    bool DDM_ILU;
    int DDM_Smalltype;
    int DDM_printout;
    bool parallel_mm;
    bool symmetric;
    bool dissect0;
    bool cholesky;

public:
    Schur (): DDM(0), ind(0), DDP_svd(false), DDM_invert(false), DDM_checkdiagonal(false), DDM_ILU(false), DDM_Smalltype(MATRIXTYPE(SPARSE)), DDM_printout(-1), parallel_mm(false), dissect0(false), cholesky(false) {
        ReadConfig(Settings,"SVD",DDP_svd);
        ReadConfig(Settings,"invert",DDM_invert);
        ReadConfig(Settings,"checkdiagonal",DDM_checkdiagonal);
        ReadConfig(Settings,"ILU",DDM_ILU);
        ReadConfig(Settings,"Smalltype",DDM_Smalltype);
        ReadConfig(Settings,"printout", DDM_printout);
        ReadConfig(Settings,"parallel_mm", parallel_mm);
        ReadConfig(Settings,"symmetric", symmetric);
        ReadConfig(Settings,"dissect0",dissect0);
        ReadConfig(Settings,"cholesky",cholesky);
        if (!symmetric && cholesky) {
            mout << "'cholesky' needs \"symmetric = true\". This was set here. \n"; 
            symmetric = true;
        }
        if (DDM_invert && cholesky) {
            mout << "'invert' and 'cholesky' may not be \"true\" at the same time! \"invert = false\" was set.\n"; 
            DDM_invert = false;
        }
        if (cholesky && parallel_mm) {
//             mout << "an extra parallel version with 'cholesky' is not implemented yet.\n";
        }
    }

private:
    void Construct(const Matrix&);

    void Destruct();
    virtual ~Schur() {Destruct();}

    void multiply (Vector &, const Vector& ) const;

    string Name () const { return "Schur"; }
    friend ostream& operator << (ostream& s, const Schur& Sch) {return s << "Schur"; }
};

////////////////////////////////////////////////////
//  END OF DECLARATIONS, BEGIN OF DEFINITIONS     //
////////////////////////////////////////////////////

////////////////////////////////////////////////////
//              SMALLMATRIXTRANS                  //
////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////
//                         DDind                            //
//////////////////////////////////////////////////////////////

DDind::DDind(const DDind& DI) : IND(DI.IND.size()), block(DI.block.size()) {
    for (int i=0; i<IND.size(); ++i)
        IND[i] = DI.IND[i];
    for (int i=0; i<block.size(); ++i)
        block[i] = DI.block[i];
    for (int i=0; i<vPS.size(); ++i)
        vPS.Add(DI.vPS[i]);
}

class LessDissect {
public:
    LessDissect () {}
    bool operator () (const cell& c0, const cell& c1) const { 
        return (c0() < c1()); 
    }
};

void DDind::dissect (const Vector& U, vector<int>& invIND, vector<vector<int> >& invIND0, vector<int>& IND) {
    int N = U.size();
    invIND0.resize(3);
    for (int i=0; i<3; ++i)
        invIND0[i].resize(invIND.size());
    int j0 = 0;
    int j1 = 0;
    int j2 = 0;

    vector<int> ind_set(N,-1);
    vector<bool> mark(U.nR(),false);

    for (int i=0; i<invIND.size(); ++i)
        ind_set[invIND[i]] = i;

    const Mesh& M = U.GetMesh();

    vector<cell> C(M.Cells::size()); 
    int nC = 0;
    for (cell c=M.cells(); c!=M.cells_end(); ++c) C[nC++] = c;
    sort(C.begin(),C.end(),LessDissect());
    nC /= 2;
    for (int k=0; k<nC; ++k) {
        rows R(U,C[k]);
        for (int j=0; j<R.size(); ++j) {
            int id = R[j].Id();
            mark[id] = true;
        }
    }
    for (int k=nC; k<C.size(); ++k) {
        rows R(U,C[k]);
        for (int j=0; j<R.size(); ++j) {
            int id = R[j].Id();
            int i = U.Index(id);
            int n = R[j].n();
            if (ind_set[i] == -1) continue;
            for (int k=0; k<n; ++k) 
                ind_set[i+k] = -1;
            if (mark[id]) {
                for (int k=0; k<n; ++k) {
                    invIND0[2][j2++] = i + k;
                }
            } else {
                for (int k=0; k<n; ++k) {
                    invIND0[1][j1++] = i + k;
                }
            }
        }
    }
    for (int k=0; k<ind_set.size(); ++k) 
        if (ind_set[k] != -1) {
            invIND0[0][j0++] = k;
        }
    invIND0[0].resize(j0);
    invIND0[1].resize(j1);
    invIND0[2].resize(j2);

    for (int i=0; i<j0; ++i) {
        invIND[i] = invIND0[0][i];
        IND[invIND0[0][i]] = i;
    }
    for (int i=0; i<j1; ++i) {
        invIND[i+j0] = invIND0[1][i];
        IND[invIND0[1][i]] = i+j0;
    }
    for (int i=0; i<j2; ++i) {
        invIND[i+j0+j1] = invIND0[2][i];
        IND[invIND0[2][i]] = i+j0+j1;
    }

    return;
}








void DDind::Setind(const Vector& U) {
    block.resize(2);
    block[0] = 0;
    block[1] = 0;
    invIND.resize(1);

    int size = U.size(); 

    IND.resize(size);
    invIND[0].resize(size);

    int blocksize = 0;
    vector<vector<row> > R(blocksize);
    vector<int> nR(blocksize);

    for (row r=U.rows(); r != U.rows_end(); ++r) {
        Point x = r();
        int s = r.n();
        procset ps = U.find_procset(x);
        if (ps != U.procsets_end()) {
            int i = vPS.findequalProcSet(*ps);
            if (i != -1) {
                nR[i]++;
                R[i].resize(nR[i]);
                R[i][nR[i]-1] = r;
            } else {
                blocksize++;
                R.resize(blocksize);
                nR.resize(blocksize);
                nR[blocksize-1] = 1;
                R[blocksize-1].resize(nR[blocksize-1]);
                R[blocksize-1][nR[blocksize-1]-1] = r;
                vPS.Add(*ps); 
            }
        }
        else {
            int id = r.Id();
            int i = U.Index(id);
            for (int k=0; k<s; ++k) {
                invIND[0][block[1]] = i+k;
                IND[i+k] = block[1]++;
            }
        }
    }

    invIND[0].resize(block[1]);
    invIND.resize(blocksize+1);

    dissect(U, invIND[0], invIND0, IND);

    vector<int> rowsort(blocksize);

    vPS.Sort();

    for (int i=0; i<blocksize; ++i) 
        sort(R[i].begin(),R[i].end(),Less());

    for (int i=0; i<blocksize; ++i) rowsort[vPS.findequalProcSet(*(U.find_procset(R[i][0]())))] = i;


    block.resize(blocksize+2);
    for (int b=0; b<blocksize; ++b) {
        block[b+2] = block[b+1];
        invIND[b+1].resize(size-block[b+1]); 
        int bsort = rowsort[b];
        for (int n=0; n<nR[bsort]; ++n) {
            int id = (R[bsort][n]).Id();
            int i = U.Index(id);
            int s = (R[bsort][n]).n();
            for (int k=0; k<s; ++k) {
                invIND[b+1][block[b+2]-block[b+1]] = i+k;
                IND[i+k] = block[b+2]++;
            }
        }
        invIND[b+1].resize(block[b+2]-block[b+1]);
    }
}

DDind::DDind(const Vector& u) : IND(u.size()), block(0), invIND(0) {
    vPS.resize(0);
    Setind(u);
}

void DDind::Destruct() {
/*    IND.clear(); 
    for (int i=0; i<invIND.size(); ++i) invIND[i].clear();
    for (int i=0; i<invIND0.size(); ++i) invIND0[i].clear();
    invIND.clear();
    invIND0.clear();
    block.clear();
    vPS.clear();*/
}

int DDind::size(int i) const {
    if (i==-1) return block[block.size()-1];
    if (block.size()<i+2) return 0; 
    return block[i+1]-block[i];
}

int DDind::Size() const {
    return block.size()-1;
}

int DDind::find_block(int i) const {
    for (int b=0; b<block.size()-1; ++b)
        if (i < block[b+1]) return b;
    return -1; // error
}

int DDind::blockentry(int i) const {
    return block[i];
}

ProcSet DDind::GetProc (int i) const {
    if (i==0) return PPM->proc();
    return vPS.GetPS(i-1);
}

vecProcSet DDind::GetProc() const {
    return vPS;
}


////////////////////////////////////////////////////
//                  DDProblem                     //
////////////////////////////////////////////////////
////////////////////////////////////////////////////
//              SPARSEDDPROBLEM                   //
////////////////////////////////////////////////////
void SparseDDProblem::setSparse(const SparseMatrix& A, const vector<int>& Indices, const vector<int>& Indices2) {
    int nonzero = 0;
    int size = A.size();

    if (Indices.size() == 0 || Indices2.size() == 0) return;

    vector<int> searchindex;
    searchindex.resize(size);
    for (int i=0; i<searchindex.size(); ++i) searchindex[i] = -1;
    for (int i=0; i<Indices2.size(); ++i) searchindex[Indices2[i]] = i;

    for (int i = 0; i < Indices.size(); ++i)
        for (int k = A.rowind(Indices[i]); k < A.rowind(Indices[i]+1); ++k)
            if (searchindex[A.colptr(k)] != -1) 
                nonzero++;

    M = new BasicSparseMatrix(n,nonzero);

    int* d = (*M).rowind();
    Scalar* nzval = (*M).nzval();
    int* colptr = (*M).colptr();
    int mi = 0;
    d[0] = 0;
    for (int i = 0; i < Indices.size(); ++i) {
        d[i+1] = d[i];
        for (int k=A.rowind(Indices[i]); k<A.rowind(Indices[i]+1); ++k) {
            int akt = searchindex[A.colptr(k)];
            if (akt != -1) {
                colptr[mi] = akt;
                nzval[mi] = A.nzval(k);
                ++d[i+1];
                mi++;
            }
        }
    }
}

void SparseDDProblem::LU(bool invert = false, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (n != m) {mout << "warning! no quadratic matrix!\n"; exit(0);}
    if (!M) return;
//     mout << *M;
//     mout << "-----\n";
    S = GetSparseSolver(*M);
//     S->test_output();
    delete M; M = 0; // try to use less space
    lu = true;
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}
void SparseDDProblem::SolveLU(DDProblem& b, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (!S) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (cols() != b.rows()) {mout << "ERROR in SolveLU(DDProblem) -- dimensions are not correct -- " << cols() << " : " << b.rows() << b.cols() << "\n";exit(0);}
    if (b.gettype() != MATRIXTYPE(SMALL)) {mout << "ERROR in SparseDDProblem::SolveLU(DDProblem) -- matrix has to be SMALL\n";exit(0);} 
    if (n != 0 && m != 0)
        S->Solve(b.ref(),b.cols());
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}
void SparseDDProblem::SolveLU(SmallVector& V, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (!S) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (!lu) {mout << "Warning! Matrix is not LU-decomposed yet\n"; exit(0);}
    S->Solve(V.ref());
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}
void SparseDDProblem::SolveLU(Scalar* b, int size, DateTime* DT1 = NULL, DateTime* DT2 = NULL, bool trans = 0) {
    if (!S) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
//     if (cols() != b.rows()) {mout << "ERROR in SolveLU(DDProblem) -- dimensions are not correct -- " << cols() << " : " << b.rows() << b.cols() << "\n";exit(0);}
//     if (b.gettype() != MATRIXTYPE(SMALL)) {mout << "ERROR in SparseDDProblem::SolveLU(DDProblem) -- matrix has to be SMALL\n";exit(0);} 
    if (n != 0 && m != 0)
        S->Solve(b,size);
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}


////////////////////////////////////////////////////
//             SMALLDDPROBLEM                     //
////////////////////////////////////////////////////

void SmallDDProblem::LU(bool invert = false, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (rows() == 0 || cols() == 0) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (rows() != cols()) {mout << "warning! no quadratic matrix!\n"; exit(0);}
    M.makeLU();
    if (invert) M.invert();
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}

void SmallDDProblem::SolveLU(DDProblem& b, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (rows() == 0 || cols() == 0) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (cols() != b.rows()) {mout << "ERROR in SolveLU(DDProblem) -- dimensions are not correct\n";exit(0);}
    if (b.gettype() != MATRIXTYPE(SMALL)) {mout << "ERROR in SparseDDProblem::SolveLU(DDProblem) -- matrixtype has to be SMALL\n";exit(0);}
    if (rows() != 0 && cols() != 0)
        M.Solve(b.ref(),b.cols());
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}

void SmallDDProblem::SolveLU(SmallVector& V, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (rows() == 0 || cols() == 0) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (rows() != 0 && cols() != 0)
        M.Solve(V.ref());
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}

void SmallDDProblem::SolveLU(Scalar* V, int size, DateTime* DT1 = NULL, DateTime* DT2 = NULL, bool trans = 0) {
    if (rows() == 0 || cols() == 0) return;
    if (DT1) DT1->SetDate();if (DT2) DT2->SetDate();
    if (rows() != 0 && cols() != 0)
        if (!trans)
            M.Solve(V,size);
        else
            M.Solve(V,size,'U');
    if (DT1) DT1->AddTime();if (DT2) DT2->AddTime();
}

void SmallDDProblem::SendMatrix (Buffer& B) {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return;
    B.fill(*(M.ref()),n*m*sizeof(Scalar));
    return;
}

size_t SmallDDProblem::GetSizeSendMatrix () {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return 0;
    return n*m*sizeof(Scalar);
}

void SmallDDProblem::AddMatrix (Buffer& B) {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return;
    size_t size = n*m*sizeof(Scalar);
    Scalar* tmp = new Scalar [n*m];
    B.read(*tmp, size);
    for (int j=0; j<n*m; ++j) M.ref()[j] += tmp[j];
    delete tmp;
}

void SmallDDProblem::AddMatrix_trans (Buffer& B) {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return;
    size_t size = n*m*sizeof(Scalar);
    Scalar* tmp = new Scalar [n*m];
    B.read(*tmp, size);
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) 
            M.ref()[j*n+i] += tmp[i*m+j];
    delete tmp;
}

void SmallDDProblem::SetMatrix (Buffer& B) {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return;
    size_t size = n*m*sizeof(Scalar);
    B.read(*(M.ref()), size);
}

void SmallDDProblem::SetMatrix_trans (Buffer& B) {
    int n = M.rows(); int m = M.cols();
    if (n*m == 0) return;
    size_t size = n*m*sizeof(Scalar);
    Scalar* tmp = new Scalar [n*m];
    B.read(*tmp, size);
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) 
            M.ref()[j*n+i] = tmp[i*m+j];
    delete tmp;
}

void SmallDDProblem::SendIPIV (Buffer& B) {
    M.SendIPIV(B);
} 

size_t SmallDDProblem::GetSizeSendIPIV() {
    return M.GetSizeSendIPIV();
}

void SmallDDProblem::SetIPIV (Buffer& B) {
    M.SetIPIV(B);
}

DDProblem& SmallDDProblem::operator -= (const constAB<DDProblem,DDProblem>& AB) {
    int _n = AB.first().rows();
    int _m = AB.second().cols();
    int _q = AB.first().cols();
    if (_n != 0 && _m != 0 && _q != 0)
    if (MATRIXTYPE(AB.first().gettype()) == SMALL) {
        char transa = 'N';
        char transb = 'N';
        double minone = -1;
        double one = 1;
        int lda = _n;
        Scalar *C = ref();
        Scalar *A = AB.first().ref();
        Scalar *B = AB.second().ref();
        if (cholesky) {
            transa = 'T';
            _n = AB.first().cols();
            _q = AB.first().rows();
            lda = _q;
        }
        GEMM(&transa, &transb, &_n, &_m, &_q, &minone, A, &lda, B, &_q, &one, C, &_n);
        }
    else if (MATRIXTYPE(AB.first().gettype()) == SPARSE) {
        for (int i=0; i<_n; ++i)
            for (int j=0; j<_m; ++j) {
                Scalar s = 0;
                for (int k = (*AB.first().refM()).rowind(i); k < (*AB.first().refM()).rowind(i+1); ++k)
                    s += (*AB.first().refM()).nzval(k) * AB.second()((*AB.first().refM()).colptr(k),j);
                M(i,j) -= s;
            }
    }
    return *this;
}


////////////////////////////////////////////////////
//                  DDVector                      //
////////////////////////////////////////////////////

void DDVector::Set (const Vector& U) {
    for (int i=0; i<ddind::Size(); ++i)
        v[i] = new SmallVector(ddind::size(i));
    for (int n=0; n<U.size(); ++n) {
        int ind_i = ddind::ind(n);
        int v_i = ddind::find_block(ind_i);
        int sub_i = ddind::blockentry(v_i);
        (*v[v_i])[ind_i-sub_i] = U[n];
    }
}

void DDVector::writevector(Vector& u) const {
    for (int n=0; n<u.size(); ++n) {
        int ind_i = ddind::ind(n);
        int v_i = ddind::find_block(ind_i);
        int sub_i = ddind::blockentry(v_i);
        u[n] = (*v[v_i])[ind_i-sub_i];
    }
}

////////////////////////////////////////////////////
//                  DDMatrix                      //
////////////////////////////////////////////////////

void DDMatrix::class_Solve_step::set_K_op_step(vector<ProcSet>& P) {
    int m=0;
    for (int k=start; k < P.size(); ++k) {
        if (P[k].existselementof(akt_procs)) { // definition of K_op
            m++;
            K_op.resize(m);
            K_op[m-1] = k;
        }
    }
    m=0;
    for (int k=start; k < P.size(); ++k) {
        if (P[k].subset(akt_procs)) {
            m++;
            K_step.resize(m);
            K_step[m-1] = k;
        }
    }
    K_delta.resize(K_op.size()-K_step.size());
    for (int i=K_step.size(); i < K_op.size(); ++i)
        K_delta[i-K_step.size()] = K_op[i];
}

void DDMatrix::Set(const Matrix& _A) {
    vector<int> size(ddind::Size());
    DPM.resize(ddind::Size());

    for (int i=0; i<ddind::Size(); ++i) 
        DPM[i].resize(ddind::Size());

    for (int i=0; i<size.size(); ++i)
        size[i] = ddind::size(i);


    Matrix A(_A);

    if (symmetric) {
        A.EliminateDirichlet();
    }

    if (ddind::size(-1) == 0) {
        DPM[0][0] = new SparseDDProblem(0,0,false);
        return;
    }

    SparseMatrix S(A);

    if (checkdiagonal) S.CheckDiagonal();

    Date Start;

    for (int i=0; i<ddind::Size(); ++i)
        for (int j=0; j<ddind::Size(); ++j) {
            if ((MATRIXTYPE(Smalltype) == SPARSE) && (j == 0)) {
                if (size[i] != 0 && size[j] != 0)
                    DPM[i][j] = new SparseDDProblem(size[i],size[j],S, ddind::invind(i),ddind::invind(j),DDP_svd);
            }
            else 
                if (size[i] != 0 && size[j] != 0)
            DPM[i][j] = new SmallDDProblem(size[i], size[j], DDP_svd, symmetric, cholesky);
    }

    if (dissect0) {
        M0.resize(3);
        for (int i=0; i<3; ++i)
            M0[i].resize(3);

        M0[0][0] = new SparseDDProblem(ddind::invind0(0).size(),ddind::invind0(0).size(),S, ddind::invind0(0),ddind::invind0(0),DDP_svd);
        M0[2][0] = new SparseDDProblem(ddind::invind0(2).size(),ddind::invind0(0).size(),S, ddind::invind0(2),ddind::invind0(0),DDP_svd);

        M0[1][1] = new SparseDDProblem(ddind::invind0(1).size(),ddind::invind0(1).size(),S, ddind::invind0(1),ddind::invind0(1),DDP_svd);
        M0[2][1] = new SparseDDProblem(ddind::invind0(2).size(),ddind::invind0(1).size(),S, ddind::invind0(2),ddind::invind0(1),DDP_svd);

        M0[0][2] = new SmallDDProblem(ddind::invind0(0).size(),ddind::invind0(2).size(),DDP_svd, symmetric, cholesky);
        M0[1][2] = new SmallDDProblem(ddind::invind0(1).size(),ddind::invind0(2).size(),DDP_svd, symmetric, cholesky);
        M0[2][2] = new SmallDDProblem(ddind::invind0(2).size(),ddind::invind0(2).size(),DDP_svd, symmetric, cholesky);

    }


    for (int n=0; n<S.size(); ++n) {
        int ind_i = ddind::ind(n);
        int M_i = ddind::find_block(ind_i);
        int sub_i = ddind::blockentry(M_i);
        for (int d = S.rowind(n); d < S.rowind(n+1); ++d) {
            int col = S.colptr(d);

            int ind_j = ddind::ind(col);
            int M_j = ddind::find_block(ind_j);
            int sub_j = ddind::blockentry(M_j);
            if (((MATRIXTYPE(Smalltype) == SMALL) || (M_j !=0)) && (DPM[M_i][M_j]->rows() != 0) && (DPM[M_i][M_j]->cols() != 0))
                (*DPM[M_i][M_j])(ind_i-sub_i,ind_j-sub_j) = S.nzval(d);

            if (dissect0) {
                int j0 = ddind::invind0(0).size();
                int j1 = ddind::invind0(1).size() + j0;
                if ((M_j == 0) && (M_i == 0)) {
                    if (ind_j >= j1) {
                        if (ind_i < j0) (*M0[0][2])(ind_i,ind_j-j1) = S.nzval(d);
                        else if (ind_i < j1) (*M0[1][2])(ind_i-j0,ind_j-j1) = S.nzval(d);
                        else (*M0[2][2])(ind_i-j1,ind_j-j1) = S.nzval(d);
                    }
                }
            }

            if (checkdiagonal &&
                M_i == M_j && 
                ind_i-sub_i == ind_j-sub_j && 
                abs(S.nzval(d)) < 1.e-18)
              if (((MATRIXTYPE(Smalltype) == SMALL) || (M_j != 0)) && (DPM[M_i][M_j]->rows() != 0) && (DPM[M_i][M_j]->cols() != 0))
                (*DPM[M_i][M_j])(ind_i-sub_i,ind_j-sub_j) = 1;

        }
    }

    tout(1) << "Set DDMatrices in " << Date()-Start << endl;
}

void DDMatrix::Send_PS(ExchangeBuffer& E, const int& q, const ProcSet& PS, const int& size) const {
    E.Send(q) << int(PS.size());
    for (int s=0; s<PS.size(); ++s) {
        E.Send(q) << int(PS[s]);
    }
    E.Send(q) << size;
}

size_t DDMatrix::GetSizeSend_PS(const ProcSet& PS) const {
    return sizeof(int) * (PS.size() + 2);
}

void DDMatrix::Receive_PS(ExchangeBuffer& E, const int& q, ProcSet& PS, int& size) const {
    PS.resize(0);
    int tmps, tmpq;
    E.Receive(q) >> tmps;
    for (int s=0; s<tmps; ++s) {
        E.Receive(q) >> tmpq;
        PS.Add(tmpq);
    }
    E.Receive(q) >> size;
}

void DDMatrix::Send_Matrix(ExchangeBuffer& E, const int q, DDProblem& A, int info = 0) const {
    A.SendMatrix(E.Send(q));
    if (info > 0) {
        A.SendIPIV(E.Send(q));
    }
}

size_t DDMatrix::GetSizeSend_Matrix(DDProblem& A, int info = 0) const {
    size_t size = 0;
    size += A.GetSizeSendMatrix();
    if (info > 0) {
        size += A.GetSizeSendIPIV();
    }
    return size;
}

size_t DDMatrix::GetSizeSend_Matrix_with_PS(ProcSet& P1, ProcSet& P2, DDProblem& A, int info = 0) const {
    size_t size = 0;
    size += GetSizeSend_PS(P1);
    size += GetSizeSend_PS(P2);
    size += GetSizeSend_Matrix(A,info);
    return size;
}

void DDMatrix::Receive_Matrix(ExchangeBuffer& E, const int q, DDProblem& A, bool add_or_set = true, int info = 0) const {
    if (add_or_set)
        A.AddMatrix(E.Receive(q));
    else
        A.SetMatrix(E.Receive(q));
    if (info > 0) {
        A.SetIPIV(E.Receive(q));
    }
}

void DDMatrix::Receive_Matrix_trans(ExchangeBuffer& E, const int q, DDProblem& A, bool add_or_set = true, int info = 0) const {
    if (add_or_set)
        A.AddMatrix_trans(E.Receive(q));
    else
        A.SetMatrix_trans(E.Receive(q));
    if (info > 0) {
        A.SetIPIV(E.Receive(q));
    }
}

void DDMatrix::Send_Matrix_to_Buffer(ExchangeBuffer& E, const int q, ProcSet& P1, ProcSet& P2, DDProblem& A, int info) const {
    Send_PS(E,q,P1,A.rows());
    Send_PS(E,q,P2,A.cols());
    Send_Matrix(E,q,A,info);
}


void DDMatrix::Receive_Matrix_from_Buffer(ExchangeBuffer& E, const int q, vector<ProcSet>& P, vector<vector<DDProblem* > > & M, bool add_or_set, int info, int DDP_svd) const {
    int i = -1; int j = -1;
    ProcSet P1;
    ProcSet P2;
    int rows, cols;

    Receive_PS(E,q,P1,rows);
    for (int s=0; s < P.size(); ++s)
        if (P[s].equalProcSet(P1)) {i = s; break;}
    if (i == -1) pout << "error in i\n";

    Receive_PS(E,q,P2, cols);
    for (int s=0; s < P.size(); ++s)
        if (P[s].equalProcSet(P2)) {j = s; break;}
    if (j == -1) pout << "error in j\n";

    if (cholesky && (i > j)) {
        int t = i;
        i = j;
        j = t;
        t = rows;
        rows = cols;
        cols = t;
        if (M[i][j] == NULL) {
            M[i][j] = new SmallDDProblem(rows,cols, DDP_svd, symmetric, cholesky);
        }
        Receive_Matrix_trans(E,q,*M[i][j],add_or_set,info);
    } else {
        if (M[i][j] == NULL) {
            M[i][j] = new SmallDDProblem(rows,cols, DDP_svd, symmetric, cholesky);
        }
        Receive_Matrix(E,q,*M[i][j],add_or_set,info);
    }
}

void DDMatrix::Send_Vector(ExchangeBuffer& E, const int q, SmallVector& v) const {
    Scalar* vref = v.ref();
    E.Send(q).fill(*vref,sizeof(Scalar)*v.size());
}

void DDMatrix::Receive_Vector(ExchangeBuffer& E, const int q, SmallVector & v, bool add_or_set = true) const {
    if (add_or_set) {
        Scalar* tmp = new Scalar[v.size()];
        E.Receive(q).read(*tmp, sizeof(Scalar)*v.size());
        for (int i=0; i<v.size(); ++i)
            v[i] += tmp[i];
        delete[] tmp;
    } else {
        Scalar* vref = v.ref();
        E.Receive(q).read(*vref, sizeof(Scalar)*v.size());
    }
    return;
}

void DDMatrix::Fill_steplist() {
    vecProcSet L_ges;
    vector<bool> L_ges_used;
    vecProcSet L_akt;
    vector<bool> L_akt_used;
    vector<int> P_list;
    vector<ProcSet> akt_procs;

    vector<int> sz;

    ExchangeBuffer Exchange;

    for (int i=0; i<ddind::Size(); ++i) {
        for (int q=0; q<PPM->size(); ++q) {
            Send_PS(Exchange,q,ddind::GetProc(i),ddind::size(i));
        }
    }

    Exchange.Communicate();

    for (short q=0; q<PPM->size(); ++q) {
        ProcSet P;
        int c;
        while (Exchange.Receive(q).size() < Exchange.ReceiveSize(q)) {
            Receive_PS(Exchange,q,P,c);
            L_ges.Add(P,c);
        }
    }

    L_ges.Sort();

    vout(50) << "Total ProcList\n";
    vout(50) << L_ges;
    L_ges_used.resize(L_ges.size());
    for (int i=0; i<L_ges_used.size(); ++i) L_ges_used[i] = false;

    P_list.resize(PPM->size());
    akt_procs.resize(PPM->size());
    for (short q=0; q<PPM->size(); ++q) {
        P_list[q] = q;
        akt_procs[q].Add(q);
    }

    PM.resize(0);
    sz.resize(0);
    int used = 0;
    int s = 0;

    while (used < L_ges.size()) {
        Solve_step.resize(s+1);
        int start, end, sendto;
        if (s == 0) {
            start = 0; end = 0; sendto = -1;
        } else {
            start = (*Solve_step[s-1]).getend();
            end = (*Solve_step[s-1]).getend();
            sendto = -1;
        }
        L_akt.resize(0);
        for (int i=0; i<L_ges.size(); ++i)
            if (!L_ges_used[i]) {
                L_akt.Add(L_ges[i],L_ges.GetSize(i));
            }

        L_akt_used.resize(L_akt.size());
        for (int i=0; i<L_akt.size(); ++i)
            L_akt_used[i] = false;

        int akt_used = 0;
        int akt = 0;
        while (akt_used < L_akt.size()) {
            while (L_akt_used[akt]) akt++;
            ProcSet PM_akt = L_akt[akt];
            int p = PM_akt[0];
            int akt_p = P_list[p];
            if (akt_p == PPM->proc()) {
                end++;
                int m = PM.size();
                PM.resize(m+1);
                PM[m] = PM_akt;
                sz.resize(m+1);
                sz[m] = L_akt.GetSize(akt);
            }

            for (int i=1; i<PM_akt.size(); ++i) {
                int pi = PM_akt[i];
                int akt_pi = P_list[pi];
                if (PPM->proc() == akt_pi && akt_pi != akt_p) {
                    (*Solve_step[s-1]).setSendToL(akt_p);
                }
                akt_procs[akt_p].Add(akt_procs[akt_pi]);

                if (PPM->proc() == P_list[akt_pi] && PPM->proc() != P_list[p])
                    for (int j=0; j<L_ges.size(); ++j) {
                        if (!L_ges_used[j])
                        if (akt_procs[akt_pi].existselementof(L_ges[j])) {
                            int m=PM.size();
                            PM.resize(m+1);
                            PM[m] = L_ges[j];
                            sz.resize(m+1);
                            sz[m] = L_ges.GetSize(j);
                        }
                    }
                int aendern = P_list[akt_pi];
                for (int k=0; k<P_list.size(); ++k) if (P_list[k] == aendern) P_list[k] = P_list[akt_p];
            }
            L_ges_used[L_ges.findequalProcSet(PM_akt)] = true;
            used++;
            for (int j=0; j<L_akt.size(); ++j) {
                if (L_akt_used[j]) continue;
                if (akt_procs[akt_p].existselementof(L_akt[j])) {L_akt_used[j] = true;akt_used++;}
            }
            for (int i=0; i<L_ges.size(); ++i) {
                if (L_ges_used[i]) continue;
                if (L_ges[i].subset(akt_procs[akt_p])) {
                    if (PPM->proc() == akt_p) {
                        int m=PM.size();
                        PM.resize(m+1);
                        PM[m] = L_ges[i];
                        sz.resize(m+1);
                        sz[m] = L_ges.GetSize(i);
                        end++;
                    }
                    L_ges_used[i] = true;
                    used++;
                }
            }
        }
        Solve_step[s] = new class_Solve_step(start,end,sendto);
        (*Solve_step[s]).set_akt_procs(akt_procs[PPM->proc()]);
        s++;
    }

    bool test = false; // Inversion of the order of \pi \in \m \Pi^s
    ReadConfig(Settings,"test",test);
    if (test)
    for (int i=0; i<Solve_step.size(); ++i) {
        int n=(*Solve_step[i]).getstart();
        int m=(*Solve_step[i]).getend()-1;
        while (n<m) {
            ProcSet PS = PM[n];
            PM[n] = PM[m];
            PM[m] = PS;
            int s = sz[n];
            sz[n] = sz[m];
            sz[m] = s;
            ++n;--m;
        }
    }

    if (PPM->proc() == printout) {
        pout << "ProcList on P:" << PPM->proc() << "\n";

        int sz_ges = 0;
        for (int i=0; i<Solve_step.size(); ++i) {
            pout << "akt_procs: " << (*Solve_step[i]).get_akt_procs() << endl;
            int size = 0;
            for (int j=(*Solve_step[i]).getstart(); j < (*Solve_step[i]).getend(); ++j) {
                size += sz[j];
                sz_ges += sz[j];
                pout << PM[j] << "  ---  " << sz[j] << "  (" << size << " ; " << sz_ges << ")" << endl;
            }
            pout << " -----------------------------\n";
            if ((*Solve_step[i]).getstart() == (*Solve_step[i]).getend()) {
                pout << " --- other matrices ---\n";
                for (int j=(*Solve_step[i]).getstart(); j < PM.size(); ++j) {
                    size += sz[j];
                    sz_ges += sz[j];
                    pout << PM[j] << "  ---  " << sz[j] << "  (" << size << " ; " << sz_ges << ")" << endl;
                 }
                 i = Solve_step.size();
            }
        }

        pout << "Start  End  SendTo  AktPS\n";
        for (int i=0; i<Solve_step.size(); ++i)
            pout << (*Solve_step[i]).getstart() << "   " << (*Solve_step[i]).getend() << "   " << (*Solve_step[i]).getSendToL() << "  " << (*Solve_step[i]).get_akt_procs() << endl;
    }

    M.resize(PM.size());
    for (int i=0; i<M.size(); ++i) M[i].resize(PM.size());

    if (cholesky) {
        for (int i=0; i<M.size(); ++i) {
            int p = ddind::findequalProcSet(PM[i])+1;
            if (p != 0 || PM[i].size()==1) {
                if (!Smalltype && p == 0) {
                    for (int j=1; j<M.size(); ++j) {
                    int q = ddind::findequalProcSet(PM[j])+1;
                    if (q != 0 || PM[j].size()==1)
                        M[j][i] = DPM[q][p];
                    }
                }
                for (int j = i; j<M.size(); ++j) {
                    int q = ddind::findequalProcSet(PM[j])+1;
                    if (q != 0 || PM[j].size()==1)
                    if (j >= i)
                        M[i][j] = DPM[p][q];
                    else
                    if (Smalltype || (!Smalltype && q != 0))
                        if (DPM[q][p]) delete DPM[q][p];
                }
            }
        }
    } else {
        for (int i=0; i<M.size(); ++i) {
            int p = ddind::findequalProcSet(PM[i])+1;
            if (p != 0 || PM[i].size()==1)
                for (int j=0; j<M.size(); ++j) {
                    int q = ddind::findequalProcSet(PM[j])+1;
                    if (q != 0 || PM[j].size()==1) { 
                        M[i][j] = DPM[p][q];
                    }
                }
        }
    }
    for (int i=0; i<Solve_step.size(); ++i)
        (*Solve_step[i]).set_K_op_step(PM);
}

bool DDMatrix::int_in_PS(const int& p,const ProcSet& PS) const {
    for (int i=0; i<PS.size(); ++i)
        if (p == PS[i]) return true;
    return false;
}

void DDMatrix::Subtract(DDProblem& A, DDProblem& B, DDProblem& C, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (DT1) DT1->SetDate();
    if (DT2) DT2->SetDate();
    A -= B*C;
    if (DT1) DT1->AddTime();
    if (DT2) DT2->AddTime();
}

void DDMatrix::SubtractTransposed(DDProblem& A, DDProblem& B, DDProblem& C, DateTime* DT1 = NULL, DateTime* DT2 = NULL) {
    if (DT1) DT1->SetDate();
    if (DT2) DT2->SetDate();
    // A -) B^T * C !!!
    A -= B*C;
    if (DT1) DT1->AddTime();
    if (DT2) DT2->AddTime();
}

void DDMatrix::Invert_M0() {
    Date D1;
    if (M0[0][0]) M0[0][0]->LU(invert, &times.LU, &times_step.LU);
    mout << "Time to Solve " << (*M0[0][0]).rows() << " : " << Date()-D1 << endl;
    if (M0[0][2] && M0[0][2]) (*M0[0][0]).SolveLU(*M0[0][2], &times.SolveLU, &times_step.SolveLU);
    Date D2;
    if (M0[1][1]) M0[1][1]->LU(invert, &times.LU, &times_step.LU);
    mout << "Time to Solve " << (*M0[1][1]).rows() << " : " << Date()-D2 << endl;
    if (M0[1][1] && M0[1][2]) (*M0[1][1]).SolveLU(*M0[1][2], &times.SolveLU, &times_step.SolveLU);
    if (M0[2][2] && M0[2][0] && M0[0][2]) Subtract(*M0[2][2],*M0[2][0],*M0[0][2],&times.MatrixMultiplication,&times_step.MatrixMultiplication);
    if (M0[2][2] && M0[2][1] && M0[1][2]) Subtract(*M0[2][2],*M0[2][1],*M0[1][2],&times.MatrixMultiplication,&times_step.MatrixMultiplication);
    Date D3;
    if (M0[2][2]) M0[2][2]->LU(invert, &times.LU, & times_step.LU);
    mout << "Time to Solve " << (*M0[2][2]).rows() << " : " << Date()-D3 << endl;
}

void DDMatrix::Solve_M0(Scalar* v) const {
    int j0 = 0; 
    if (M0[0][0]) j0 = (*M0[0][0]).rows();
    int j1 = 0; 
    if (M0[1][1]) j1 = (*M0[1][1]).rows();
    int j2 = 0; 
    if (M0[2][2]) j2 = (*M0[2][2]).rows();
    Scalar* b0 = v;
    Scalar* b1 = v+j0;
    Scalar* b2 = v+j0+j1;
    if (M0[0][0]) (*M0[0][0]).SolveLU(b0,1);
    if (M0[2][0]) MultiplySubtract(*M0[2][0],b0,b2);
    if (M0[1][1]) (*M0[1][1]).SolveLU(b1,1);
    if (M0[2][1]) MultiplySubtract(*M0[2][1],b1,b2);
    if (M0[2][2]) (*M0[2][2]).SolveLU(b2,1);
    if (M0[1][2]) MultiplySubtract(*M0[1][2],b2,b1);
    if (M0[0][2]) MultiplySubtract(*M0[0][2],b2,b0);
}

void DDMatrix::Invert_parallel_chol(int steps) {
    int k_min = 0;
    int k_max = 0;
    tout(2) << "start with new parallel version cholesky\n";

// get maximal number of steps
    k_min = (*Solve_step[steps]).getstart();
    k_max = (*Solve_step[steps]).getend();
    int k_steps = k_max-k_min;
    k_steps = PPM->Max(k_steps);

// define sendsize; sendtoproc
    vector<size_t> sendsize;
    sendsize.resize(PPM->size());
    vector<int> sendtoproc;
    sendtoproc.resize(M.size()-(k_min));

// get size of extraprocs
    int extraprocs = 0;
    if (k_min < k_max) extraprocs = (*Solve_step[steps]).get_akt_procs().size();

    vector<int> vps;
    vps.resize(extraprocs);
    for (int i=0; i<extraprocs; ++i)
        vps[i] = (*Solve_step[steps]).get_akt_procs()[i];

// send all PMs to these extraprocs
    PPM->Synchronize();
    times.Communication.SetDate();
    times_step.Communication.SetDate();
    ExchangeBuffer E_PS;

// set sendsize
    for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;

    for (int ex=0; ex<extraprocs; ++ex) {
        int ps = vps[ex];
        sendsize[ps] += sizeof(int)*(extraprocs+1); //send vps
        sendsize[ps] += sizeof(int);
        for (int i=k_min; i<M.size(); ++i)
            sendsize[ps] += GetSizeSend_PS(PM[i]);
    }

    if (extraprocs > 0)
    for (int j=k_min+1; j<M.size(); ++j) {
        int pr = ((j-(k_min+1)) % extraprocs);
        int ps = vps[pr];
        sendtoproc[j-(k_min)] = ps;
//         for (int i=k_min; i<M.size(); ++i) 
        for (int i=k_min; i<=j; ++i) 
            if (M[i][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[i],PM[j],*M[i][j],0);
    }

// send all PS
    for (int ex=0; ex<extraprocs; ++ex) {
        int ps = (*Solve_step[steps]).get_akt_procs()[ex];
        E_PS.Send(ps).resize(sendsize[ps]);
        E_PS.Send(ps) << extraprocs;
        for (int i=0; i<extraprocs; ++i)
            E_PS.Send(ps) << vps[i]; 
        E_PS.Send(ps) << int(M.size()-k_min);

        for (int i=k_min; i<M.size(); ++i)
            Send_PS(E_PS,ps,PM[i],0);
    }

// send even all matrices from k_min+1 < M.size() and define which row is sent to what proc
    if (extraprocs > 0)
    for (int j=k_min+1; j<M.size(); ++j) {
        int ps = sendtoproc[j-(k_min)];
//         for (int i=k_min; i<M.size(); ++i) 
        for (int i=k_min; i<=j; ++i) 
            if (M[i][j]) Send_Matrix_to_Buffer(E_PS,ps,PM[i],PM[j],*M[i][j],0);
    }

// receive PMs and matrices
    E_PS.Communicate();

    vector<ProcSet> PM_parallel;
    vector<vector<DDProblem* > > M_parallel;
    vector<int> all_procs;
    int maxsize = 0;

    int sendto_aftersolve = -1;

    for (short q=0; q<PPM->size(); ++q)
        if (E_PS.Receive(q).size() < E_PS.ReceiveSize(q)) {
            sendto_aftersolve = q;
            E_PS.Receive(q) >> maxsize;
            all_procs.resize(maxsize);
            for (int i=0; i<maxsize; ++i)
                E_PS.Receive(q) >> all_procs[i];

            E_PS.Receive(q) >> maxsize;

    // ... receive PMs
            PM_parallel.resize(maxsize);
            int size_PM;
            for (int i=0; i<maxsize; ++i)
                Receive_PS(E_PS,q,PM_parallel[i],size_PM);

    // ...resize of M_parallel to maxsize
            M_parallel.resize(maxsize);
            for (int i = 0; i < maxsize; ++i)
                M_parallel[i].resize(maxsize);

            for (int i=0; i<maxsize; ++i)
                for (int j=0; j<maxsize; ++j) {
                    M_parallel[i][j] = NULL;
                }

    // ...receive matrices
            while (E_PS.Receive(q).size() < E_PS.ReceiveSize(q))
                Receive_Matrix_from_Buffer(E_PS, q, PM_parallel, M_parallel, false, 0, DDP_svd);
        }

    times.Communication.AddTime();
    times_step.Communication.AddTime();

// solve by communication
    for (int k_step=0; k_step<k_steps; ++k_step) {
        int k=k_min+k_step;
        int made_lu = 0;
// make LU(k,k)
        if (k_min+k_step < k_max) {
            M[k][k]->LU(invert, &times.LU, &times_step.LU);
            made_lu = 1;
        }

        PPM->Synchronize();
        times.Communication.SetDate();
        times_step.Communication.SetDate();

// send diagonal element k to every proc
        ExchangeBuffer E_parallel;
    // set sendsize
        for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;

        if (made_lu)
            for (int ex=0; ex<extraprocs; ++ex) {
                int ps = vps[ex];
                sendsize[ps] += sizeof(int);
                sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[k],PM[k],*M[k][k],1);

/*                for (int i=k+1; i<M.size(); ++i)
                    if (M[i][k]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[i],PM[k],*M[i][k],0);*/
            }

    // send diagonal element
        if (made_lu)
            for (int ex=0; ex<extraprocs; ++ex) {
                int ps = vps[ex];
                E_parallel.Send(ps).resize(sendsize[ps]);
                E_parallel.Send(ps) << made_lu;
                Send_Matrix_to_Buffer(E_parallel,ps,PM[k],PM[k],*M[k][k],1); // 1: including IPIV
/*                for (int i=k+1; i<M.size(); ++i) 
                    if (M[i][k]) Send_Matrix_to_Buffer(E_parallel,ps,PM[i],PM[k],*M[i][k],0);*/
            }

    E_parallel.Communicate();

// receive diagonal element
    for (short q=0; q<PPM->size(); ++q)
        if (E_parallel.Receive(q).size() < E_parallel.ReceiveSize(q)) {
            E_parallel.Receive(q) >> made_lu;
            Receive_Matrix_from_Buffer(E_parallel, q, PM_parallel, M_parallel, false, 1, DDP_svd); // receive with IPIV
/*            while (E_parallel.Receive(q).size() < E_parallel.ReceiveSize(q)) {
                Receive_Matrix_from_Buffer(E_parallel, q, PM_parallel, M_parallel, false, 0, DDP_svd);
            }*/
        }

    times.Communication.AddTime();
    times_step.Communication.AddTime();

// solve on every proc: SolveLU(k_step,k_step..end)
        Date Date_SolveMM;
        if (made_lu)
        for (int i=k_step+1; i<maxsize; ++i)
            if (M_parallel[k_step][i]) {
                (*M_parallel[k_step][k_step]).SolveLU(*M_parallel[k_step][i], &times.SolveLU, &times_step.SolveLU);
            }

        ExchangeBuffer E_Solve;
        for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;

        vector<bool> col_on_proc;
        col_on_proc.resize(maxsize);
        for (int i=0; i<maxsize; ++i) col_on_proc[i] = false;

    // set sendsize
        if (made_lu)
            for (int ex=0; ex<all_procs.size(); ++ex) {
                int ps = all_procs[ex];
                for (int i=k_step+1; i<maxsize; ++i)
                if (M_parallel[k_step][i]) {
                    sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[k_step],PM_parallel[i],*M_parallel[k_step][i],0);
                    col_on_proc[i] = true;
                }
            }
    //send row
        if (made_lu)
            for (int ex=0; ex<all_procs.size(); ++ex) {
                int ps = all_procs[ex];
                E_Solve.Send(ps).resize(sendsize[ps]);
                for (int i=k_step+1; i<maxsize; ++i)
                    if (M_parallel[k_step][i])
                        Send_Matrix_to_Buffer(E_Solve,ps,PM_parallel[k_step],PM_parallel[i],*M_parallel[k_step][i],0);
            }

        E_Solve.Communicate();
    //receive row on every proc
        for (short q=0; q<PPM->size(); ++q)
            while (E_Solve.Receive(q).size() < E_Solve.ReceiveSize(q)) {
                Receive_Matrix_from_Buffer(E_Solve,q,PM_parallel,M_parallel,false,0,DDP_svd);
            }


// and Matrix-Multiplication(k_step+1..end,kstep+1..end)
        if (made_lu)
            for (int i=k_step+1; i<maxsize; ++i) if (col_on_proc[i]) {
                for (int j=k_step+1; j<=i; ++j)
                    if (M_parallel[k_step][j]) {
                        if (M_parallel[j][i] == NULL) {
                            M_parallel[j][i] = new SmallDDProblem((*M_parallel[k_step][j]).cols(),(*M_parallel[k_step][i]).cols(), DDP_svd, symmetric, cholesky);
                        }
                        Subtract(*M_parallel[j][i],*M_parallel[k_step][j],*M_parallel[k_step][i],&times.MatrixMultiplication,&times_step.MatrixMultiplication);
    
                    }
            }

        PPM->Synchronize();
        times.Communication.SetDate();
        times_step.Communication.SetDate();
// set sendsize for aftersolve
        ExchangeBuffer E_aftersolve;
        int ps = sendto_aftersolve;
        for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;
        if (extraprocs > 0)
        for (int j=k_step+1; j<maxsize; ++j)
            if (M_parallel[k_step][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[k_step],PM_parallel[j],*M_parallel[k_step][j],0);

        if (k_step+1 < maxsize)
//             for (int i=k_step+1; i<maxsize; ++i)
                if (M_parallel[k_step+1][k_step+1]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[k_step+1],PM_parallel[k_step+1],*M_parallel[k_step+1][k_step+1],0);

        if (k_step == k_steps-1)
            for (int j=k_step+2; j < maxsize; ++j)
                for (int i=k_step+1; i<=j; ++i) 
                    if (M_parallel[i][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[i],PM_parallel[j],*M_parallel[i][j],0);

        if (ps != -1) E_aftersolve.Send(ps).resize(sendsize[ps]);

// send row k_step to sendto_aftersolve
        if (extraprocs > 0)
        for (int j=k_step+1; j<maxsize; ++j)
            if (M_parallel[k_step][j]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[k_step],PM_parallel[j],*M_parallel[k_step][j],0);

// send column k_step+1 to sendto_aftersolve 
        if (k_step+1 < maxsize)
//             for (int i=k_step+1; i<maxsize; ++i)
                if (M_parallel[k_step+1][k_step+1]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[k_step+1],PM_parallel[k_step+1],*M_parallel[k_step+1][k_step+1],0);

        if (k_step == k_steps-1)
            for (int j=k_step+2; j < maxsize; ++j)
                for (int i=k_step+1; i<=j; ++i) 
                    if (M_parallel[i][j]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[i],PM_parallel[j],*M_parallel[i][j],0);

// receive everything on sendto_aftersolve
        E_aftersolve.Communicate();

        for (short q=0; q<PPM->size(); ++q)
            while (E_aftersolve.Receive(q).size() < E_aftersolve.ReceiveSize(q))
                Receive_Matrix_from_Buffer(E_aftersolve, q, PM, M, false, 0, DDP_svd);

        times.Communication.AddTime();
        times_step.Communication.AddTime();
    }

    for (int i=0; i<maxsize; ++i)
        for (int j=0; j<maxsize; ++j)
            if (M_parallel[i][j] != NULL) {
                delete (M_parallel[i][j]); 
                M_parallel[i][j] = NULL;
            }
    for (int i=0; i<maxsize; ++i)
        M_parallel[i].clear();
    PM_parallel.clear();
    M_parallel.clear();


}

void DDMatrix::Invert_parallel(int steps) {
    int k_min = 0;
    int k_max = 0;
    tout(2) << "start with new parallel version \n";

// get maximal number of steps
    k_min = (*Solve_step[steps]).getstart();
    k_max = (*Solve_step[steps]).getend();
    int k_steps = k_max-k_min;
    k_steps = PPM->Max(k_steps);

// define sendsize; sendtoproc
    vector<size_t> sendsize;
    sendsize.resize(PPM->size());
    vector<int> sendtoproc;
    sendtoproc.resize(M.size()-(k_min));

// get size of extraprocs
    int extraprocs = 0;
    if (k_min < k_max) extraprocs = (*Solve_step[steps]).get_akt_procs().size();

    vector<int> vps;
    vps.resize(extraprocs);
    for (int i=0; i<extraprocs; ++i)
        vps[i] = (*Solve_step[steps]).get_akt_procs()[i];


// send all PMs to these extraprocs
    PPM->Synchronize();
    times.Communication.SetDate();
    times_step.Communication.SetDate();
    ExchangeBuffer E_PS;

// set sendsize
    for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;

    for (int ex=0; ex<extraprocs; ++ex) {
        int ps = vps[ex];
        sendsize[ps] += sizeof(int);
        for (int i=k_min; i<M.size(); ++i)
            sendsize[ps] += GetSizeSend_PS(PM[i]);
    }

    if (extraprocs > 0)
    for (int j=k_min+1; j<M.size(); ++j) {
        int pr = ((j-(k_min+1)) % extraprocs);
        int ps = vps[pr];
        sendtoproc[j-(k_min)] = ps;
        for (int i=k_min; i<M.size(); ++i) 
            if (M[i][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[i],PM[j],*M[i][j],0);
    }

// send all PS
    for (int ex=0; ex<extraprocs; ++ex) {
        int ps = (*Solve_step[steps]).get_akt_procs()[ex];
        E_PS.Send(ps).resize(sendsize[ps]);
        E_PS.Send(ps) << int(M.size()-k_min);

        for (int i=k_min; i<M.size(); ++i)
            Send_PS(E_PS,ps,PM[i],0);
    }

// send even all matrices from k_min+1 < M.size() and define which row is sent to what proc
    if (extraprocs > 0)
    for (int j=k_min+1; j<M.size(); ++j) {
        int ps = sendtoproc[j-(k_min)];
        for (int i=k_min; i<M.size(); ++i) 
            if (M[i][j]) Send_Matrix_to_Buffer(E_PS,ps,PM[i],PM[j],*M[i][j],0);
    }

// receive PMs and matrices
    E_PS.Communicate();

    vector<ProcSet> PM_parallel;
    vector<vector<DDProblem* > > M_parallel;
    int maxsize = 0;

    int sendto_aftersolve = -1;

    for (short q=0; q<PPM->size(); ++q)
        if (E_PS.Receive(q).size() < E_PS.ReceiveSize(q)) {
            sendto_aftersolve = q;
            E_PS.Receive(q) >> maxsize;

    // ... receive PMs
            PM_parallel.resize(maxsize);
            int size_PM;
            for (int i=0; i<maxsize; ++i)
                Receive_PS(E_PS,q,PM_parallel[i],size_PM);

    // ...resize of M_parallel to maxsize
            M_parallel.resize(maxsize);
            for (int i = 0; i < maxsize; ++i)
                M_parallel[i].resize(maxsize);

            for (int i=0; i<maxsize; ++i)
                for (int j=0; j<maxsize; ++j) 
                    M_parallel[i][j] = NULL;

    // ...receive matrices
            while (E_PS.Receive(q).size() < E_PS.ReceiveSize(q))
                Receive_Matrix_from_Buffer(E_PS, q, PM_parallel, M_parallel, false, 0, DDP_svd);
        }

    times.Communication.AddTime();
    times_step.Communication.AddTime();

// solve by communication
    for (int k_step=0; k_step<k_steps; ++k_step) {
        int k=k_min+k_step;
        int made_lu = 0;
// make LU(k,k)
        if (k_min+k_step < k_max) {
            M[k][k]->LU(invert, &times.LU, &times_step.LU);
            made_lu = 1;
        }

        PPM->Synchronize();
        times.Communication.SetDate();
        times_step.Communication.SetDate();

// send column k to every proc
        ExchangeBuffer E_parallel;
    // set sendsize
        for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;

        if (made_lu)
            for (int ex=0; ex<extraprocs; ++ex) {
                int ps = vps[ex];
                sendsize[ps] += sizeof(int);
                sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[k],PM[k],*M[k][k],1);

                for (int i=k+1; i<M.size(); ++i)
                    if (M[i][k]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM[i],PM[k],*M[i][k],0);
            }

    // send columns
        if (made_lu)
            for (int ex=0; ex<extraprocs; ++ex) {
                int ps = vps[ex];
                E_parallel.Send(ps).resize(sendsize[ps]);
                E_parallel.Send(ps) << made_lu;
                Send_Matrix_to_Buffer(E_parallel,ps,PM[k],PM[k],*M[k][k],1); // 1: including IPIV
                for (int i=k+1; i<M.size(); ++i) 
                    if (M[i][k]) Send_Matrix_to_Buffer(E_parallel,ps,PM[i],PM[k],*M[i][k],0);
            }

    E_parallel.Communicate();

// receive all matrices
    for (short q=0; q<PPM->size(); ++q)
        if (E_parallel.Receive(q).size() < E_parallel.ReceiveSize(q)) {
            E_parallel.Receive(q) >> made_lu;
            Receive_Matrix_from_Buffer(E_parallel, q, PM_parallel, M_parallel, false, 1, DDP_svd); // receive with IPIV
            while (E_parallel.Receive(q).size() < E_parallel.ReceiveSize(q)) {
                Receive_Matrix_from_Buffer(E_parallel, q, PM_parallel, M_parallel, false, 0, DDP_svd);
            }
        }

    times.Communication.AddTime();
    times_step.Communication.AddTime();

// solve on every proc: SolveLU(k_step,k_step..end)
// and Matrix-Multiplication(k_step+1..end,kstep+1..end)
        Date Date_SolveMM;
        if (made_lu)
        for (int i=k_step+1; i<maxsize; ++i)
            if (M_parallel[k_step][i]) {
                (*M_parallel[k_step][k_step]).SolveLU(*M_parallel[k_step][i], &times.SolveLU, &times_step.SolveLU);

                long int num_mm = 0;
                for (int j=k_step+1; j<maxsize; ++j)
                    if (M_parallel[j][k_step]) {
                        if (M_parallel[j][i] == NULL) {
                            M_parallel[j][i] = new SmallDDProblem((*M_parallel[j][k_step]).rows(),(*M_parallel[k_step][i]).cols(), DDP_svd, symmetric, cholesky);
                        }
                        Subtract(*M_parallel[j][i],*M_parallel[j][k_step],*M_parallel[k_step][i],&times.MatrixMultiplication,&times_step.MatrixMultiplication);

                    }
            }

        PPM->Synchronize();
        times.Communication.SetDate();
        times_step.Communication.SetDate();
// set sendsize for aftersolve
        ExchangeBuffer E_aftersolve;
        int ps = sendto_aftersolve;
        for (int i=0; i<PPM->size(); ++i) sendsize[i] = 0;
        for (int j=k_step+1; j<maxsize; ++j)
            if (M_parallel[k_step][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[k_step],PM_parallel[j],*M_parallel[k_step][j],0);

        if (k_step+1 < maxsize)
            for (int i=k_step+1; i<maxsize; ++i)
                if (M_parallel[i][k_step+1]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[i],PM_parallel[k_step+1],*M_parallel[i][k_step+1],0);

        if (k_step == k_steps-1)
            for (int i=k_step+1; i < maxsize; ++i)
                for (int j=k_step+2; j<maxsize; ++j) 
                    if (M_parallel[i][j]) sendsize[ps] += GetSizeSend_Matrix_with_PS(PM_parallel[i],PM_parallel[j],*M_parallel[i][j],0);

        if (ps != -1) E_aftersolve.Send(ps).resize(sendsize[ps]);

// send row k_step to sendto_aftersolve
        for (int j=k_step+1; j<maxsize; ++j)
            if (M_parallel[k_step][j]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[k_step],PM_parallel[j],*M_parallel[k_step][j],0);

// send column k_step+1 to sendto_aftersolve 
        if (k_step+1 < maxsize)
            for (int i=k_step+1; i<maxsize; ++i)
                if (M_parallel[i][k_step+1]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[i],PM_parallel[k_step+1],*M_parallel[i][k_step+1],0);

        if (k_step == k_steps-1)
            for (int i=k_step+1; i < maxsize; ++i)
                for (int j=k_step+2; j<maxsize; ++j)
                    if (M_parallel[i][j]) Send_Matrix_to_Buffer(E_aftersolve,ps,PM_parallel[i],PM_parallel[j],*M_parallel[i][j],0);

// receive everything on sendto_aftersolve
        E_aftersolve.Communicate();

        for (short q=0; q<PPM->size(); ++q)
            while (E_aftersolve.Receive(q).size() < E_aftersolve.ReceiveSize(q))
                Receive_Matrix_from_Buffer(E_aftersolve, q, PM, M, false, 0, DDP_svd);

        times.Communication.AddTime();
        times_step.Communication.AddTime();
    }

    for (int i=0; i<maxsize; ++i)
        for (int j=0; j<maxsize; ++j)
            if (M_parallel[i][j] != NULL) {
                delete (M_parallel[i][j]); 
                M_parallel[i][j] = NULL;
            }
    for (int i=0; i<maxsize; ++i)
        M_parallel[i].clear();
    PM_parallel.clear();
    M_parallel.clear();
}

void DDMatrix::Dissect(vector<int> K_step, vector<int> K_op) {
    for (int mk=0; mk<K_step.size(); ++mk) {
        int k = K_step[mk];
//         mout << "LU of M[" << k << "]\n" << *M[k][k] << "----------------------------------------\n";
        M[k][k]->LU(invert, &times.LU, &times_step.LU);

        for (int mi=mk+1; mi < K_op.size(); ++mi) {
            int i = K_op[mi];
            (*M[k][k]).SolveLU(*M[k][i], &times.SolveLU, &times_step.SolveLU);

            int max_mj = K_op.size();
            if (cholesky) max_mj = mi+1;
            for (int mj=mk+1; mj < max_mj; ++mj) {
                int j = K_op[mj];
                if ((ILU && M[j][i]) || !ILU) {
                    if (M[j][i] == NULL) 
                        M[j][i] = new SmallDDProblem((*M[j][k]).rows(),(*M[k][i]).cols(), DDP_svd, symmetric, cholesky);
                    Subtract((*M[j][i]),(*M[j][k]),(*M[k][i]),&times.MatrixMultiplication,&times_step.MatrixMultiplication);
                }
            }
        }
    }
}

void DDMatrix::Dissect_chol(vector<int> K_step, vector<int> K_op) {
    for (int mk=0; mk<K_step.size(); ++mk) {
        int k = K_step[mk];
        M[k][k]->LU(invert, &times.LU, &times_step.LU);

        for (int mi=mk+1; mi < K_op.size(); ++mi) {
            int i = K_op[mi];
            (*M[k][k]).SolveLU(*M[k][i], &times.SolveLU, &times_step.SolveLU);

            for (int mj=mk+1; mj <= mi; ++mj) {
                int j = K_op[mj];
                if ((ILU && M[j][i]) || !ILU) {
                    if (M[j][i] == NULL)
                        M[j][i] = new SmallDDProblem((*M[k][j]).cols(),(*M[k][i]).cols(), DDP_svd, symmetric, cholesky);
                    Subtract((*M[j][i]),(*M[k][j]),(*M[k][i]),&times.MatrixMultiplication,&times_step.MatrixMultiplication);
                }
            }
        }
    }
}

void DDMatrix::Dissect_M0(vector<int> K_op) {
    Invert_M0();
    for (int mi=1; mi < K_op.size(); ++mi) {
        int i = K_op[mi];
        for (int sz = 0; sz < (*M[0][i]).cols(); ++sz) {
            Solve_M0((*M[0][i]).ref() + sz*(*M[0][i]).rows());
        }

        int max_mj = K_op.size();
        if (cholesky) max_mj = mi+1;
        for (int mj=1; mj < max_mj; ++mj) {
            int j = K_op[mj];
            if ((ILU && M[j][i]) || !ILU) {
                if (M[j][i] == NULL) 
                    M[j][i] = new SmallDDProblem((*M[j][0]).rows(),(*M[0][i]).cols(), DDP_svd, symmetric, cholesky);
                Subtract((*M[j][i]),(*M[j][0]),(*M[0][i]),&times.MatrixMultiplication,&times_step.MatrixMultiplication);
            }
        }
    }
}

void DDMatrix::Send(ExchangeBuffer& E, int ps, vector<int> K_delta) {
    size_t sendsize = 0;
    if (ps != -1) {
        for (int mi=0; mi<K_delta.size(); ++mi) {
            int mj = 0; if (cholesky) mj = mi;
            for (; mj<K_delta.size(); ++mj) {
                int i=K_delta[mi];int j=K_delta[mj];
                if ((ILU && M[i][j]) || (!ILU)) sendsize += GetSizeSend_Matrix_with_PS(PM[i],PM[j],*M[i][j],0);
            }
        }

        E.Send(ps).resize(sendsize);

        for (int mi=0; mi<K_delta.size(); ++mi) {
            int mj = 0; if (cholesky) mj = mi;
            for (; mj<K_delta.size(); ++mj) {
                int i=K_delta[mi];int j=K_delta[mj];
                if ((ILU && M[i][j]) || (!ILU)) 
                    Send_Matrix_to_Buffer(E,ps,PM[i],PM[j],*M[i][j],0);
            }
        }
    }
}

void DDMatrix::Invert() {
    Fill_steplist();

    times.ResetTime();
    times.Blockdecomposition.SetDate();

    for (int steps=0; steps< Solve_step.size(); ++steps) {
        times_step.ResetTime();
        times_step.Blockdecomposition.SetDate();

        tout(2) << PPM->proc() << "  Step " << steps << endl;

        if (cholesky) {

            if (steps == 0 && (MATRIXTYPE(Smalltype) == SPARSE)) {
                if (dissect0) {
                    Dissect_M0((*Solve_step[steps]).get_K_op());
                    mout << "block LU with dissect0\n";
                }
                else
                    Dissect((*Solve_step[steps]).get_K_step(),(*Solve_step[steps]).get_K_op());
            }
//                 Dissect((*Solve_step[steps]).get_K_step(),(*Solve_step[steps]).get_K_op());
            else
                if (parallel_mm && steps > 0) {
                    Invert_parallel_chol(steps);
                    }
                else
                Dissect_chol((*Solve_step[steps]).get_K_step(),(*Solve_step[steps]).get_K_op());
        }
        else {
                                                ////////////////////////
            if (steps > 0 && parallel_mm)       //  PARALLEL VERSION  //
                Invert_parallel(steps);         ////////////////////////
    
                                                ////////////////////////
            if (steps == 0 || !parallel_mm)     //  PREVIOUS VERSION  //
                if (steps == 0 && dissect0)
                    Dissect_M0((*Solve_step[steps]).get_K_op());
                else
                    Dissect((*Solve_step[steps]).get_K_step(),(*Solve_step[steps]).get_K_op());
        }

        PPM->Synchronize();
        times.Communication.SetDate();      ////////////////////////
        times_step.Communication.SetDate(); //    SEND MATRICES   //
        ExchangeBuffer E_last;              ////////////////////////
        Send(E_last, (*Solve_step[steps]).getSendToL(),(*Solve_step[steps]).get_K_delta());

        E_last.Communicate();               ////////////////////////
                                            //  RECEIVE MATRICES  //
        for (short q=0; q<PPM->size(); ++q) ////////////////////////
            while (E_last.Receive(q).size() < E_last.ReceiveSize(q))
                Receive_Matrix_from_Buffer(E_last, q, PM, M, true, 0, DDP_svd);

        times.Communication.AddTime();
        times_step.Communication.AddTime();
        times_step.Blockdecomposition.AddTime();

        times_step.SetMax();
        tout(3) << times_step;

        tout(2) << "----------------------------------------------------------\n";
    }  // of  for (int steps=0; steps< Solve_step.size(); ++steps) {

    times.Blockdecomposition.AddTime();
    times.SetMax();
    tout(3) << times;
}

void DDMatrix::Solve_L(DDVector& b,const int steps) const {
    for (int k=(*Solve_step[steps]).getstart(); k<(*Solve_step[steps]).getend(); ++k) {
        if (dissect0 && k == 0) {
            Solve_M0(b.getv(k).ref());
        }
        else
            (*M[k][k]).SolveLU(b.getv(k));
        for (int i=k+1; i<M.size(); ++i)
            if (M[i][k]) { 
                b.getv(i) -= (*M[i][k])*b.getv(k);
        }
    }
}

void DDMatrix::Solve_L_chol(DDVector& b,const int steps) const {
    DateTime* dummy = NULL;
    for (int k=(*Solve_step[steps]).getstart(); k<(*Solve_step[steps]).getend(); ++k) {
/*        if (dissect0 && k == 0) {
           Solve_M0(b.getv(k).ref());
           mout << "Solve_M0 in chol has been executed\n";
        }
        else
*/
            (*M[k][k]).SolveLU(b.getv(k).ref(),1,&(*dummy),&(*dummy),false);
        for (int i=k+1; i<M.size(); ++i)
            if (M[k][i]) { 
                MultiplySubtract_trans((*M[k][i]), b.getv(k).ref(),b.getv(i).ref());
        }
    }
}

void DDMatrix::Solve_U(DDVector&b,const int steps) const {
        for (int i=(*Solve_step[steps]).getend()-1; i>=(*Solve_step[steps]).getstart(); --i)
            for (int k=M.size()-1; k>i; --k)
                if (M[i][k])
                    b.getv(i) -= (*M[i][k])*b.getv(k);
}

void DDMatrix::Solve_U_chol(DDVector&b,const int steps) const {
    DateTime* dummy = NULL;
    for (int i=(*Solve_step[steps]).getend()-1; i>=(*Solve_step[steps]).getstart(); --i) {
        for (int k=M.size()-1; k>i; --k)
            if (M[i][k])
                b.getv(i) -= (*M[i][k])*b.getv(k);
        (*M[i][i]).SolveLU(b.getv(i).ref(),1,&(*dummy),&(*dummy),true);
    }
}

void DDMatrix::Communicate_in_L(DDVector&b, const int steps) const {
    ExchangeBuffer E;

    size_t sendsize = 0;
    int ps = (*Solve_step[steps]).getSendToL();
    if (ps != -1) {
        for (int i=(*Solve_step[steps]).getend(); i<M.size(); ++i) {
            sendsize += GetSizeSend_PS(PM[i]);
            sendsize += b.getv(i).size()*sizeof(Scalar);
        }

        E.Send(ps).resize(sendsize);

        for (int i=(*Solve_step[steps]).getend(); i<M.size(); ++i) {
            Send_PS(E,ps,PM[i],b.getv(i).size());
            Send_Vector(E,ps,b.getv(i));
        }
    }

    E.Communicate();

    for (short q=0; q<PPM->size(); ++q) {
        int i;
        ProcSet P1;
        int size;
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            Receive_PS(E,q,P1, size);
            for (int s=1; s < PM.size(); ++s)
                if (P1.equalProcSet(PM[s])) {i = s; break;}

            if (&(b.getv(i)) == NULL) {
                b.createnewvec(i,size);
            }
            Receive_Vector(E,q,b.getv(i));
            (*Solve_step[steps+1]).Set_SendToU(true);
            if (!int_in_PS(q,P1))
                (*Solve_step[steps+1]).addto_send_special(i,q);
        }
    }
}

void DDMatrix::Communicate_in_U(DDVector&b, const int steps) const {
    ExchangeBuffer E;

    vector<size_t> sendsize;
    sendsize.resize(PPM->size());
    for (int i= 0; i<PPM->size(); ++i) sendsize[i] = 0;

    for (int i=0; i<(*Solve_step[steps]).send_special_size(); ++i) {
        int ps = (*Solve_step[steps]).get_send_special_SendTo(i);
        int sw = (*Solve_step[steps]).get_send_special_vec(i);
        sendsize[ps] += GetSizeSend_PS(PM[sw]);
        sendsize[ps] += b.getv(sw).size()*sizeof(Scalar);
    }

    if ((*Solve_step[steps]).getSendToU()) {
        for (int i=(*Solve_step[steps]).getend()-1; i>=(*Solve_step[steps]).getstart(); --i) {
            for (int p=0; p<PM[i].size(); ++p) {
                int ps=PM[i][p];
                if (ps != PPM->proc()) {
                    sendsize[ps] += GetSizeSend_PS(PM[i]);
                    sendsize[ps] += b.getv(i).size()*sizeof(Scalar);
                }
            }
        }

        for (int i=0; i<PPM->size(); ++i) E.Send(i).resize(sendsize[i]);

        for (int i=(*Solve_step[steps]).getend()-1; i>=(*Solve_step[steps]).getstart(); --i) {
            for (int p=0; p<PM[i].size(); ++p) {
                int ps=PM[i][p];
                if (ps != PPM->proc()) {
                    Send_PS(E,ps,PM[i],b.getv(i).size());
                    Send_Vector(E,ps,b.getv(i));
                } 
            }
        }
    }

    for (int i=0; i<(*Solve_step[steps]).send_special_size(); ++i) {
        int ps = (*Solve_step[steps]).get_send_special_SendTo(i);
        int sw = (*Solve_step[steps]).get_send_special_vec(i);
        Send_PS(E,ps,PM[sw],b.getv(sw).size());
        Send_Vector(E,ps,b.getv(sw));
    }
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q) {
        int i;
        ProcSet P1;
        int size;
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            Receive_PS(E,q,P1, size);
            for (int s=1; s < PM.size(); ++s)
                if (P1.equalProcSet(PM[s])) {i = s; break;}

        if (&(b.getv(i)) == NULL) pout << "ERROR" << endl;//b.createnewvec(i,size);
        Receive_Vector(E,q,b.getv(i),false);
        }
    }
}

void DDMatrix::SolveLU(DDVector& b) const {
    for (int steps=0; steps<Solve_step.size(); ++steps) {
        if ((steps == 0 && (MATRIXTYPE(Smalltype) == SPARSE)) || (cholesky == false))
            Solve_L(b,steps);
        else
            Solve_L_chol(b,steps);

        Communicate_in_L(b,steps);
    }

    vout(50) << "------------------------END OF L-------------------------------------------\n";

    for (int steps=Solve_step.size()-1; steps >=0; --steps) {
        if ((steps == 0 && (MATRIXTYPE(Smalltype) == SPARSE)) || (cholesky == false))
            Solve_U(b,steps);
        else
            Solve_U_chol(b,steps);

        Communicate_in_U(b,steps);

    }

    vout(50) << "------------------------END OF U-------------------------------------------\n";
}


////////////////////////////////////////////////////
//                  SCHUR                         //
////////////////////////////////////////////////////

void Schur::Construct(const Matrix& A) {
    Date Start;

//     SmallMatrixTrans t1(4,4);
//     t1(0,0) = 1; t1(0,1) = 1; t1(0,2) = 3; t1(0,3) = 4;
//     t1(1,0) = 2; t1(1,1) = 2; t1(1,2) = 2; t1(1,3) = 2;
//     t1(2,0) = 3; t1(2,1) = 3; t1(2,2) = 3; t1(2,3) = 3;
//     t1(3,0) = 4; t1(3,1) = 4; t1(3,2) = 4; t1(3,3) = 4;
//     SVD TEST1(t1);
// 
//     SmallMatrixTrans t2(4,4);
//     t2(0,0) = 99; t2(0,1) = 96; t2(0,2) = 93; t2(0,3) = 99;
//     t2(1,0) = 102; t2(1,1) = 99; t2(1,2) = 105; t2(1,3) = 102;
//     t2(2,0) = 105; t2(2,1) = 102; t2(2,2) = 108; t2(2,3) = 105;
//     t2(3,0) = 108; t2(3,1) = 105; t2(3,2) = 111; t2(3,3) = 108;
//     SVD TEST2(t2);
// 
//     TEST1.add(TEST2);
//     TEST1.add(TEST2);
// 
//     TEST1.originaltest();
// 
//     mout << "test1.destruct():\n";
//     TEST1.Destruct();
//     mout << "test2.destruct():\n";
//     TEST2.Destruct();

    ind = new DDind(A.GetVector());
    DDM = new DDMatrix(*ind, DDM_invert, DDM_checkdiagonal, DDM_ILU, DDM_Smalltype, DDM_printout, DDP_svd, parallel_mm, verbose, symmetric, dissect0, cholesky);

    DDM->Set(A);
    DDM->Invert();

    vout(1) << "create Decomposition " << Date() - Start << endl;
}

void Schur::Destruct() { 
    if (ind) delete ind; ind = 0;
    if (DDM) delete DDM; DDM = 0;
}

void Schur::multiply (Vector &u, const Vector& b) const {
    Date Start;

    DDVector ddu(*DDM);
    ddu.Set(b);
    ddu.Setv(DDM->getPM());
    DDM->SolveLU(ddu);

    ddu.writevector(u);
    u.ClearDirichletValues();

    vout(1) << "   multiply Schur " << Date() - Start << endl;
}

Preconditioner* GetSchurPC () {
    return new Schur();
}
