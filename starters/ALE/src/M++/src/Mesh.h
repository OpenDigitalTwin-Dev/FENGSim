// file: Mesh.h
// author: Christian Wieners
// $Header: /public/M++/src/Mesh.h,v 1.14 2009-11-24 09:46:35 wieners Exp $

#ifndef _MESH_H_
#define _MESH_H_

#include "IO.h" 
#include "Vertex.h" 
#include "Edge.h" 
#include "Face.h" 
#include "Cell.h" 
#include "ProcSet.h" 
#include "Identify.h" 
#include "Overlap.h" 

class Mesh : public Vertices, public Edges, public Faces, public Cells, 
    public OverlapCells,
    public BoundaryFaces, public ProcSets, public IdentifySets {
    int d;
    bool Parallel;
    bool IdentifyBnd;
 private:
    void RemoveFace (const Point& F, const face& f, const Point& C);
    
    void RemoveFace (const cell& c, const Point& F);
    
 public:
    cell InsertCell (CELLTYPE tp, int flag, const vector<Point>& x);
    
    void InsertOverlapCell (cell c);

    void InsertOverlapCell (Cell* C);

    void RemoveCell (cell c);
    
    void FinishParallel ();
    
    void Finish ();
    
    int dim () const { return d; }
    bool identify () const { return IdentifyBnd; }
    bool parallel () const { return Parallel; }
    bool onBnd (const Point& x) const;
    
    pair<double,double> MeshWidth () const;
    
    ~Mesh ();
};

ostream& operator << (ostream& s, const Mesh& M);

class CoarseGeometry;
class Meshes {
    vector<Mesh *> M;
    int finelevel;
    int plevel;
 public:
    Meshes ();
    Meshes (const char*);
    Meshes (const char*, int, int);
    Meshes (const CoarseGeometry&, map<string,list<int> >&,
	    map<string,int>&, map<string,list<Point> >&,
	    map<string,int>&);
	Meshes (const vector<Point>& coords, const vector<int>& ids);
	Meshes (double* coords, int n, int* ids, int m);
    void Init ();
    void Init (int level, int _plevel);
    int Level () const { return finelevel; }
    int pLevel () const { return plevel; }
    Mesh& fine () { return *M[finelevel]; }
    Mesh& operator [] (int i) { return *M[i]; }
    const Mesh& fine () const { return *M[finelevel]; }
    const Mesh& operator [] (int i) const { return *M[i]; }
    int dim () const { if (M.size() == 0) return -1; return fine().dim(); }
    void Refine ();
    ~Meshes ();
    friend ostream& operator << (ostream& s, const Meshes& M);

    void ReSet(const char*, int, int);
};

class BFParts {
    int bnd[6];
    int n;
    bool onbnd;
 public:
    BFParts (const Mesh& M, const cell& c);
    int operator [] (int i) const { return bnd[i]; }
    const int* operator () () const { 
	if (onbnd) return bnd; 
	return 0;
    }
    bool onBnd () const { return onbnd; }
    int size() const { return n; }
    friend ostream& operator << (ostream& s, const BFParts& BF);   
};

#endif
