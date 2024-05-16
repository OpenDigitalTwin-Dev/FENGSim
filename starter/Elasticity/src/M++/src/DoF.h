// file: DoF.h
// author: Christian Wieners
// $Header: /public/M++/src/DoF.h,v 1.12 2009-05-28 16:09:27 wieners Exp $

#ifndef _DOF_H_
#define _DOF_H_

#include "Mesh.h"
#include "Parallel.h"

#include <map>
#include <string>

enum NODETYPE { VERTEX = 0, EDGE = 1, FACE = 2, CELL = 3 };

class SubVectorMask {
    int tp[MaxPointTypes];
    bool* m[MaxPointTypes];
public:
    SubVectorMask () { 
		for (int i=0; i<MaxPointTypes; ++i) { tp[i] = 0; m[i] = 0; }
    }
    SubVectorMask (const int* n, const char* c) {
		int k = 0;
		for (int i=0; i<MaxPointTypes; ++i) {
			if (n[i] > 0) {
				m[i] = new bool [n[i]];
				for (int j=0; j<n[i]; ++j) m[i][j] = (c[k++] == '1');
			}	
			else m[i] = 0;
			tp[i] = n[i];
		}
    }
    SubVectorMask (const SubVectorMask& s) {
		for (int i=0; i<MaxPointTypes; ++i) {
			int n = tp[i] = s.tp[i];
			if (n) {
				m[i] = new bool [n];
				for (int j=0; j<n; ++j) m[i][j] = s.m[i][j];
			}	
			else m[i] = 0;
		}
    }
    SubVectorMask& operator = (const SubVectorMask& s) {
		for (int i=0; i<MaxPointTypes; ++i) {
			int n = tp[i] = s.tp[i];
			if (n) {
				m[i] = new bool [n];
				for (int j=0; j<n; ++j) m[i][j] = s.m[i][j];
			}	
			else m[i] = 0;
		}
		return *this;
    }
    ~SubVectorMask () {
		for (int i=0; i<MaxPointTypes; ++i) 
			if (m[i]) delete[] m[i];
    }
    const bool* operator [] (int i) const { return m[i]; }
    friend ostream& operator << (ostream& s, const SubVectorMask& S) {
		for (int i=0; i<MaxPointTypes; ++i)
	    for (int j=0; j<S.tp[i]; ++j) s << S.m[i][j];
		return s;
    }
};

class DoF {
    int n_infty;
    map<string,SubVectorMask> sub;
    bool bnd;
public:
    DoF (int n = 0, bool b = false) : n_infty(n), bnd(b) {}
    virtual ~DoF() {}
    virtual int NodalPoints (const cell&) const = 0;
    virtual void NodalPoints (const cell&, vector<Point>&) const = 0;
    virtual void NodalDoFs (const cell&, vector<short>&) const = 0;
    virtual int NodalPointsOnFace (const cell&, int) const { return 0; }
    virtual int NodalPointOnFace (const cell&, int, int) const = 0;
    virtual int NodalPointsOnEdge (const cell& c, int i) const { return 0; }
    virtual int NodalPointOnEdge (const cell& c, int i, int k) const {}
    virtual int TypeDoF (int) const = 0;
    void TypeDoFs (int* n) const {
	for (int i=0; i<MaxPointTypes; ++i) n[i] = TypeDoF(i); }
    virtual string Name () const = 0;
    int nsub () const { return sub.size(); }
    int ninf () const { return n_infty; }
    bool Bnd () const { return bnd; }
    const map<string,SubVectorMask>& Sub () const { return sub; }
    const SubVectorMask& GetSubVector (const char* name) const {
		map<string,SubVectorMask>::const_iterator s=sub.find(name);
		if (s == sub.end()) Exit(string("wrong subvector name ")+string(name));
		return s->second;
    }
    void AddSubVector (const char* name, const char* c, const int* n) {
		sub[name] = SubVectorMask(n,c); }
    void AddSubVector (const char* name, const char* c) {
		int n[MaxPointTypes];
		TypeDoFs(n); 
		sub[name] = SubVectorMask(n,c);
    }
    friend ostream& operator << (ostream& s, const DoF& D) {
		s << D.Name() << ": ";
		int n[MaxPointTypes];
		D.TypeDoFs(n); 
		for (int i=0; i<MaxPointTypes; ++i) s << n[i];
		return s << endl << D.Sub();
    }
};

DoF* GetDoF (const string&, int, int, int);
DoF* GetBndDoF (int k=0, int n=0);
inline DoF* GetDoF (const string& s, int n, int m) { return GetDoF(s,n,m,0); }

class dof {
    DoF* D;
public:
    dof (DoF* d) : D(d) {}
    dof (const string& name, int m, 
	 int dim = -1, int n = 0) : D(GetDoF(name,m,dim,n)) {}
//    ~dof () { delete D; }
    int NodalPoints (const cell& c) const { return D->NodalPoints(c); }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
		return D->NodalPoints(c,z); }
    void NodalDoFs (const cell& c, vector<short>& z) const {
		return D->NodalDoFs(c,z); }
    int NodalPointsOnFace (const cell& c, int i) const {
		return D->NodalPointsOnFace(c,i); }
    int NodalPointOnFace (const cell& c, int i, int j) const {
		return D->NodalPointOnFace(c,i,j); }
    int NodalPointsOnEdge (const cell& c, int i) const { 
		return D->NodalPointsOnEdge(c,i); }
    int NodalPointOnEdge (const cell& c, int i, int j) const { 
		return D->NodalPointOnEdge(c,i,j); }
    int TypeDoF (int tp) const { return D->TypeDoF(tp); }
    void TypeDoFs (int* tp) const { D->TypeDoFs(tp); }
    string Name () const { return D->Name(); }
    int nsub () const { return D->nsub(); }
    int n_infty () const { return D->ninf(); }
    const map<string,SubVectorMask>& Sub () const { return D->Sub(); }
    const SubVectorMask& GetSubVector (const char* name) const {
		return D->GetSubVector(name); }
    void AddSubVector (const char* name, const char* c) {
		int n[MaxPointTypes];
		D->TypeDoFs(n); 
		D->AddSubVector(name,c,n);
    }
    DoF* ptr() { return D; }
    const dof& ref() const { return *this; }
    friend ostream& operator << (ostream& s, const dof& d) {
		return s << *(d.D); }
};

#endif
