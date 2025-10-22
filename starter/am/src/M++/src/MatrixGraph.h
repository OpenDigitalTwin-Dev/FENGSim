// file: MatrixGraph.h
// author: Christian Wieners
// $Header: /public/M++/src/MatrixGraph.h,v 1.17 2009-08-20 13:47:17 wieners Exp $

#ifndef _MATRIX_GRAPH_H_
#define _MATRIX_GRAPH_H_

#include "Compiler.h"
#include "Constants.h" 
#include "Mesh.h" 
#include "DoF.h" 
#include "Identify.h" 

class Entry {
    int id;
    int entry;
 public:
    Entry () {}
    void SetId (int i) { id = i; }
    void SetEntry (int e) { entry = e; }
    int GetEntry () const { return entry; }
    int operator () () const { return id; }
    friend ostream& operator << (ostream& s, const Entry& e) { return s<<e.id;}
};
class entry : public hash_map<Point,Entry,Hash>::const_iterator {
    typedef hash_map<Point,Entry,Hash>::const_iterator Iterator;
 public:
    entry () {}
    entry (hash_map<Point,Entry,Hash>::const_iterator e) : Iterator(e) {}
    const Point& operator () () const { return (*this)->first; }
    int Id () const { return (*this)->second(); }
    int GetEntry () const { return (*this)->second.GetEntry(); }
    friend ostream& operator << (ostream& s, const entry& e) {
	return s << e->first << " : " << e->second << endl;
    }
};
class Entries : public hash_map<Point,Entry,Hash> {
public:
    typedef hash_map<Point,Entry,Hash>::iterator EntryIterator;
    entry entries () const { return entry(begin()); }
    entry entries_end () const { return entry(end()); }
    entry find_entry (const Point& z) const { return entry(find(z)); }
    int GetEntry (const Point& z) const { 
	entry e = find_entry(z);
        #ifndef NDEBUG
	if (e == entries_end()) Exit("entry not found");
        #endif
	return e.GetEntry(); 
    }
    int GetEntryX (const Point& z) const { 
	entry e = find_entry(z);
	if (e == entries_end()) return -1;
	return e.GetEntry(); 
    }
    void Insert (const Point& z) { if (find(z) == end()) (*this)[z] = Entry();}
    const Entries& ref() const { return *this; }
};
class Row : public Entries {
    int id;
    int e;
    int m;
public:
    Row () {}
    Row (int n) : m(n) {}
    void SetId (int i) { id = i; }
    void SetEntry (int f) { e = f; }
    int GetEntry () const { return e; }
    int Id () const { return id; }
    int n () const { return m; }
    friend ostream& operator << (ostream& s, const Row& R) {
	return s << " id " << R.id << endl<<R.Entries::ref();
    }
};
class row : public hash_map<Point,Row,Hash>::const_iterator {
    typedef hash_map<Point,Row,Hash>::const_iterator Iterator;
 public:
    row () {}
    row (hash_map<Point,Row,Hash>::const_iterator r) : Iterator(r) {}
    const Point& operator () () const { return (*this)->first; }
    int Id () const { return (*this)->second.Id(); }
    int n () const { return (*this)->second.n(); }
    int size () const { return (*this)->second.size(); }
    entry entries () const { return (*this)->second.entries(); }
    entry entries_end () const { return (*this)->second.entries_end(); }
    entry find_entry (const Point& z) const { 
	return (*this)->second.find_entry(z); }
    int GetEntry () const { return (*this)->second.GetEntry(); }
    int GetEntry (const Point& z) const { 
	return (*this)->second.Entries::GetEntry(z); }
    int GetEntryX (const Point& z) const {
	return (*this)->second.Entries::GetEntryX(z); }
    int GetEntry (const row& r) const { 
	return (*this)->second.Entries::GetEntry(r()); }
    int GetEntryX (const row& r) const {
	return (*this)->second.Entries::GetEntryX(r()); }
    friend bool operator < (const row& r0, const row& r1) {return (r0()<r1());}
    friend ostream& operator << (ostream& s, const row& r) {
	return s << r->first << " : " << r->second << endl; }
};
class Rows : public hash_map<Point,Row,Hash> {
    int d;
    int m;
 protected:
    void AddEntry (const Point& x, const Point& y);
    void AddEntries (const Point& x, const Point& y);
    void Insert (const Point& x, int n);
    typedef hash_map<Point,Row,Hash>::iterator RowIterator;
    class Less {
    public:
	bool operator () (const RowIterator& r0, const RowIterator& r1)  { 
	    return (r0->first < r1->first); 
	}
    };
 public:
    Rows () {};
    void Numbering ();
    row rows () const { return row(begin()); }
    row rows_end () const { return row(end()); }
    row find_row (const Point& z) const { return row(find(z)); }
    void Insert_infty (int n);
    int Id (const Point& z) const;
    int Idx (const Point& z) const;
    int GetEntry (const row& r0, const row& r1) const;
    int GetEntryX (const row& r0, const row& r1) const;
    int GetDoubleEntryX (const row& r0, const row& r1) const;
    int Dof (const Point& z) const;
    int nC () const { return d; }
    int Size () const { return m; }
    const Rows& ref() const { return *this; }
};
class Extension;
class MatrixGraph : public Rows, 
		    public IdentifySets, 
		    public ProcSets, 
		    public dof 
{
 protected:
    const Mesh& M;
    int N;
    int* index;
    int* diag;
    int* column;
    int* matentry;
    int* n;
    map<string,Extension*> Ex;
    ExchangeBuffer APE;
    ExchangeBuffer CPE;
    ExchangeBuffer AIE;
    ExchangeBuffer CIE;
    void AddCell (const cell& c, int depth = 1);
    void SetProcSetsCell (const cell&);
    void SetProcSetsOverlapCell (const cell&);
    void IdentifyCell (const cell&);
    void IdentifyOverlapCell (const cell&);
 public:
    int pSize () const;
    MatrixGraph (const dof& D, const Mesh& m) : dof(D), M(m) {}
    MatrixGraph (const Mesh& m, const dof& D, int depth = 1);
    void Init();
    int size () const { return N; }
    int Size () const { return Rows::Size(); }
    const int* Index () const { return index; }
    const int* Diag () const { return diag; }
    const int* Column () const { return column; }
    const int* Entry () const { return matentry; }
    const int* Dof () const { return n; }
    int Index (int i) const { return index[i]; }
    int Column (int i) const { return column[i]; }
    int Diag (int i) const { return diag[i]; }
    int Entry (int i) const { return matentry[i]; }
    int Dof (int i) const { return n[i]; }
//    void AddExtension (const char* s) { Ex[s]=new Extension(*this,s); }
    ExchangeBuffer& AccumulateParallelBuffer ();
    ExchangeBuffer& CollectParallelBuffer ();
    ExchangeBuffer& AccumulateIdentifyBuffer ();
    ExchangeBuffer& CollectIdentifyBuffer ();
    const Mesh& GetMesh () const { return M; }
    const Extension& GetExtension (const char* s) const;
    const SubVectorMask& Mask (const char* name) const;
    ~MatrixGraph ();
    friend ostream& operator << (ostream& s, const MatrixGraph& G);
};

class matrixgraph : public dof {
    MatrixGraph* G;
 public:
    matrixgraph (MatrixGraph& g) : dof(g), G(&g) {}
    matrixgraph (const matrixgraph& g) : dof(g), G(g.G) {}
    const MatrixGraph* operator * () const { return G; }
    row rows () const { return G->rows(); }
    row rows_end () const { return G->rows_end(); }
    row find_row (const Point& z) const { return G->find_row(z); }
    int Id (const Point& z) const { return G->Id(z); }
    int Idx (const Point& z) const { return G->Idx(z); }
    int size () const { return G->size(); }
    int Size () const { return G->Size(); }
    int pSize () const { return G->pSize(); }
    const int* Index () const { return G->Index(); }
    const int* Diag () const { return G->Diag(); }
    const int* Column () const { return G->Column(); }
    const int* Entry () const { return G->Entry(); }
    const int* Dof () const { return G->Dof(); }
    int Index (int i) const { return G->Index(i); }
    int Column (int i) const { return G->Column(i); }
    int Diag (int i) const { return G->Diag(i); }
    int Entry (int i) const { return G->Entry(i); }
    int Dof (int i) const { return G->Dof(i); }
    int GetEntry (const row& r0, const row& r1) const { 
	return G->GetEntry(r0,r1); }
    int GetEntryX (const row& r0, const row& r1) const { 
	return G->GetEntryX(r0,r1); }
    int GetDoubleEntryX (const row& r0, const row& r1) const { 
	return G->GetDoubleEntryX(r0,r1); }
    int nR() const { return G->Rows::size(); }
    const Mesh& GetMesh () const { return G->GetMesh(); }
    bool onBnd (const Point& x) const { return G->GetMesh().onBnd(x); }
    int Part (const Point& x) const { return G->GetMesh().Part(x); }
    ExchangeBuffer& AccumulateParallelBuffer () { 
	return G->AccumulateParallelBuffer(); }
    ExchangeBuffer& CollectParallelBuffer () {
	return G->CollectParallelBuffer(); }
    ExchangeBuffer& AccumulateIdentifyBuffer () { 
	return G->AccumulateIdentifyBuffer(); }
    ExchangeBuffer& CollectIdentifyBuffer () { 
	return G->CollectIdentifyBuffer(); }
    cell cells () const { return G->GetMesh().cells(); }
    cell cells_end () const { return G->GetMesh().cells_end(); }
    bnd_face bnd_faces () const { return G->GetMesh().bnd_faces(); }
    bnd_face bnd_faces_end () const { return G->GetMesh().bnd_faces_end(); }
    bnd_face find_bnd_face (const Point& z) const { 
	return G->GetMesh().find_bnd_face(z); }
    edge find_edge (const Point& x) const { return G->GetMesh().find_edge(x); }
    identifyset identifysets () const { return G->identifysets(); }
    identifyset identifysets_end () const { return G->identifysets_end(); }
    identifyset find_identifyset (const Point& z) const { 
	return G->find_identifyset(z); }
    bool identify () const { return G->GetMesh().identify(); }
    bool parallel () const { return G->GetMesh().parallel(); }
    int dim () const { return G->GetMesh().dim(); }
    procset procsets () const { return G->procsets(); }
    procset procsets_end () const { return G->procsets_end(); }
    procset find_procset (const Point& z) const { 
	return G->find_procset(z); }
    void NodalPoints (const cell c, vector<Point>& z) const {
	return G->NodalPoints(c,z); }
    const Extension& GetExtension (const char* s) const { 
	return G->GetExtension(s); }
    const SubVectorMask& Mask (const char* name) const { return G->Mask(name);}
    friend ostream& operator << (ostream& s, const matrixgraph& G) {
	return s << *(G.G); }
};

class rows : public vector<row> {
 public:
    rows (const matrixgraph& g, const cell& c);
    rows (const matrixgraph& g, const cell& c, int face);
    int n (int i) const { return (*this)[i].n(); }
};

class MatrixGraphs {
    vector<MatrixGraph*> G;
    int level;
public:
    MatrixGraphs (Meshes& M, const dof& D, int d = 1);
    ~MatrixGraphs ();
    int Level () const { return level; }
    void SetLevel (int l) { level = l; }
    int size () const { return G.size(); }
    matrixgraph fine () const { return matrixgraph(*G[G.size()-1]); }
    matrixgraph operator [] (int i) const { return matrixgraph(*G[i]); }
    friend ostream& operator << (ostream& s, const MatrixGraphs& g);

    void ReSet (Meshes& M, const dof& D, int d = 1);
};

#endif

