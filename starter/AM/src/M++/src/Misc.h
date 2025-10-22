// file: Misc.h
// author: Wolfgang Mueller
// $Header: /public/M++/src/Misc.h,v 1.1 2008-06-17 11:48:17 mueller Exp $

#ifndef _MISC_H_
#define _MISC_H_

#include <valarray>
#include "Hermitian.h"

inline bool common_face(const cell& c, const cell& d) {
    int z = 0;
    for (int i=0;i<c.Faces();++i)
        for (int j=0;j<d.Faces();++j)
            if (c.Face(i)==d.Face(j)) ++z;
    return (z==1);
}


inline void fiedler (const vector<cell>& C, valarray<double>& F) {
    Date start;
    int N = C.size();
    HermitianMatrix G(N,N);   
    for (int i=0;i<N;++i) {
        Scalar sum = 0;
	for (int j=0;j<N;++j) {
	    if (common_face(C[i],C[j])) G[i][j] = -1;
	    else G[i][j] = 0;
            sum -= real(G[i][j]);
	}
        G[i][i] = sum;
    }
    HermitianMatrix B(N,N);
    B.Identity();
    DoubleVector lambda(N);
    lambda = 0;
    HermitianMatrix E(N,N);
    E.Identity();
    EVcomplex(G,B,lambda,E);
    for (int i=0;i<N;++i)
        F[i] = real(E[i][1]);
}

inline int pow(int b, int e) {
    int p = 1;
    while (e>1) {
        p *= b;
        --e;
    }
    return p;
}

class bundle {
    cell c;
    int d;
    double f;
public:
    bundle () {}
    bundle (cell C, int D, double F) : c(C), d(D), f(F) {}
    cell getc () const { return c; }
    int getd () const { return d; }
    double getf () const { return f; }
};

inline bool bundle_compare (const bundle& a, const bundle& b) {
    return (a.getf() <= b.getf()); 
}

inline void rib (int level, vector<int>& D, int begin, int end, vector<cell>& C) {
    int N = end-begin;
    vector<cell> tmp(N);
    for (int i=0;i<N;++i)
        tmp[i] = C[begin+i];
    valarray<double> F(N);
    fiedler(tmp,F);
    vector<bundle> B(N);
    for (int i=0;i<N;++i)
        B[i] = bundle(C[begin+i],D[begin+i],F[i]);
    sort(B.begin(),B.end(),bundle_compare);
    for (int i=0;i<N;++i) {
        C[begin+i] = B[i].getc();
        D[begin+i] = B[i].getd();
        F[i] = B[i].getf();
    }
    if (level == 0) return;
    else {
        int mid = begin+(end-begin)/2;
        rib(level-1,D,begin,mid,C);
        for (int i=mid;i<end;++i)
            D[i] += pow(2,level);
        rib(level-1,D,mid,end,C);
    }
}

#endif
