// file: Mesh.C
// author: Christian Wieners
// $Header: /public/M++/src/Mesh.C,v 1.19 2009-11-24 09:46:35 wieners Exp $

#include "Mesh.h" 
#include "Meshfile.h" 
//#include "SlimMesh.h" 
#include "Distribution.h" 
#include <map>

void Mesh::RemoveFace (const Point& F, const face& f, const Point& C) {  
    if (find_cell(C) != cells_end()) return; 
    if (C == Infty) ProcSets::Remove(F,PPM->proc());
    Faces::Remove(F);
    BoundaryFaces::Remove(F);
}

void Mesh::RemoveFace (const cell& c, const Point& F) {  
    face f = find_face(F);
    if (find_cell(f.Left()) == c) RemoveFace(F,f,f.Right());
    else if (find_cell(f.Right()) == c) RemoveFace(F,f,f.Left());
}

cell Mesh::InsertCell (CELLTYPE tp, int flag, const vector<Point>& x) {  
    cell c = Cells::Insert(tp,flag,x);
    for (int i=0; i<c.Corners(); ++i)
        Vertices::Insert(c.Corner(i));
    for (int i=0; i<c.Edges(); ++i)
        Edges::Insert(c.EdgeCorner(i,0),c.EdgeCorner(i,1));
    for (int i=0; i<c.Faces(); ++i)
        Faces::Insert(c.Face(i),c());
    return c;
}

void Mesh::InsertOverlapCell (cell c) {  
    OverlapCells::InsertOverlap(c.Type(),c.Subdomain(),*(c->second));
}

void Mesh::InsertOverlapCell (Cell* C) {  
    OverlapCells::InsertOverlap(C->Type(),C->Subdomain(),*C);
}

void Mesh::RemoveCell (cell c) {  
    for (int i=0; i<c.Corners(); ++i)
        if (Vertices::Remove(c.Corner(i)))
            ProcSets::Remove(c.Corner(i),PPM->proc());
    for (int i=0; i<c.Edges(); ++i)
        if (Edges::Remove(c.Edge(i)))
            ProcSets::Remove(c.Edge(i),PPM->proc());
    for (int i=0; i<c.Faces(); ++i) 
        RemoveFace(c,c.Face(i));
    Cells::Remove(c());
}

void Mesh::FinishParallel () {
    Parallel = PPM->Boolean(procsets() != procsets_end());
}

void Mesh::Finish () {
    Parallel = PPM->Boolean(procsets() != procsets_end());
    cell c = cells();
    if (c == cells_end()) d = 0;
    else d = c.dim();
    d = PPM->Max(d);
    for (bnd_face b=bnd_faces(); b != bnd_faces_end(); ++b) {
        face f = find_face(b());
        cell c = find_cell(f.Left());
        if (c == cells_end()) c = find_cell(f.Right());
        if (c == cells_end()) continue;
        for (int i=0; i<c.Faces(); ++i) {
            if (c.Face(i) != b()) continue;
            IdentifySets::Identify(b(),b.Part());
            for (int j=0; j<c.FaceCorners(i); ++j)
                IdentifySets::Identify(c.FaceCorner(i,j),b.Part());
            for (int j=0; j<c.FaceEdges(i); ++j)
                IdentifySets::Identify(c.FaceEdge(i,j),b.Part());
            i = c.Faces();
        }
    }
    IdentifyBnd = PPM->Boolean(identifysets() != identifysets_end());
    if (!Parallel) return;
    ExchangeBuffer E;
    for (identifyset is=identifysets();is!=identifysets_end();++is) {
	procset p = find_procset(is());
	if (p == procsets_end()) continue;
	for (int i=0; i<p.size(); ++i) {
	    if (p[i] == PPM->proc()) continue;
	    E.Send(p[i]) << short(is.size()) << is();
	    for (int j=0; j<is.size(); ++j) 
		E.Send(p[i]) << is[j];
	}
    }
    for (short q=0; q<PPM->size(); ++q) 
	if (E.Send(q).size())
	    E.Send(q) << short(0);
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q) { 
	if (E.Receive(q).Size() == 0) continue;
	short m;
	E.Receive(q) >> m;
	while (m) {
	    Point x;
	    E.Receive(q) >> x;
	    for (int i=0; i<m; ++i) {
		Point y;
		E.Receive(q) >> y;
		IdentifySets::Insert(x,y);
	    }
	    E.Receive(q) >> m;
	}
    }
}

bool Mesh::onBnd (const Point& x) const { 
    return(find_bnd_face(x)!=bnd_faces_end()); 
}

pair<double,double> Mesh::MeshWidth () const {
    double h_min = infty;
    double h_max = 0;
    for (cell c=cells(); c!=cells_end(); ++c) 
        for (int i=0; i<c.Edges(); ++i) {
            double h = dist(c.EdgeCorner(i,0),c.EdgeCorner(i,1));
            if (h < h_min) h_min = h;
            if (h > h_max) h_max = h;
        }
    return pair<double,double>(PPM->Min(h_min),PPM->Max(h_max));
}

Mesh::~Mesh () { while (Cells::size()) RemoveCell(cells()); } 

ostream& operator << (ostream& s, const Mesh& M) {
    return s << "Vertices: " << M.Vertices::size() << endl << M.Vertices::ref()
	     << "Edges: " << M.Edges::size() << endl << M.Edges::ref()
	     << "Faces: " << M.Faces::size() << endl << M.Faces::ref()
	     << "Cells: " << M.Cells::size() << endl << M.Cells::ref()
	     << "OverlapCells: " << M.OverlapCells::size() << endl 
	     << M.OverlapCells::ref()
	     << "BoundaryFaces: " << M.BoundaryFaces::size() 
             << endl << M.BoundaryFaces::ref()
	     << "ProcSets: " << M.ProcSets::size() <<endl<< M.ProcSets::ref()
	     << "IdentifySets: " << M.IdentifySets::size() << endl
             << M.IdentifySets::ref();
}

Meshes::~Meshes () { for (int i=0; i<=finelevel; ++i) delete M[i]; }

ostream& operator << (ostream& s, const Meshes& M) {
	for (int i=0; i<=M.Level(); ++i) 
	    s << "mesh on level " << i << endl << M[i]; 
	return s;
    }

BFParts::BFParts (const Mesh& M, const cell& c) : n(c.Faces()), onbnd(false) {
    for (int i=0; i<n; ++i) {
        bnd[i] = M.BoundaryFaces::Part(c.Face(i));
        if (bnd[i] != -1) onbnd = true;
    }
}

ostream& operator << (ostream& s, const BFParts& BF) {
    for (int i=0; i<BF.size(); ++i) s << BF[i] << " "; 
    return s << endl;
}

const Point& Move (const Point& x) { return x; }

inline bool RefineEdge (const procset& p, const Mesh& M, Mesh& N) {
    edge e = M.find_edge(p());
    if (e != M.edges_end()) {
	Point x = Move(e());
	N.ProcSets::Copy(p,0.5*(e.Left()+x));
	N.ProcSets::Copy(p,0.5*(e.Right()+x));
	return true;
    }
    return false;
}
inline void RefineFace (const procset& p, const Mesh& M, Mesh& N) {
    face f = M.find_face(p());
    if (f == M.faces_end()) return;
    cell c = M.find_cell(f.Left());
    if (c == M.cells_end()) c = M.find_cell(f.Right());
    if (c == M.cells_end()) return;
    int i = c.facecorner(p());
    if (i == -1) return;
    Point P = Move(f->first);
    Point E0 = Move(0.5*(c.FaceCorner(i,0)+c.FaceCorner(i,1)));
    Point E1 = Move(0.5*(c.FaceCorner(i,1)+c.FaceCorner(i,2)));
    if (c.FaceCorners(i) == 4) {
	Point E2 = Move(0.5*(c.FaceCorner(i,2)+c.FaceCorner(i,3)));
	Point E3 = Move(0.5*(c.FaceCorner(i,3)+c.FaceCorner(i,0)));
	N.ProcSets::Copy(p,0.5*(E0+P));
	N.ProcSets::Copy(p,0.5*(E1+P));
	N.ProcSets::Copy(p,0.5*(E2+P));
	N.ProcSets::Copy(p,0.5*(E3+P));
	N.ProcSets::Copy(p,0.25*(c.FaceCorner(i,0)+E0+P+E3));
	N.ProcSets::Copy(p,0.25*(c.FaceCorner(i,1)+E1+P+E0));
	N.ProcSets::Copy(p,0.25*(c.FaceCorner(i,2)+E2+P+E1));
	N.ProcSets::Copy(p,0.25*(c.FaceCorner(i,3)+E3+P+E2));
    }
    else if (c.FaceCorners(i) == 3) {
	Point E2 = Move(0.5*(c.FaceCorner(i,2)+c.FaceCorner(i,0)));
	N.ProcSets::Copy(p,0.5*(E0+E1));
	N.ProcSets::Copy(p,0.5*(E1+E2));
	N.ProcSets::Copy(p,0.5*(E0+E2));
	N.ProcSets::Copy(p,(1/3.0)*(c.FaceCorner(i,0)+E0+E2));
	N.ProcSets::Copy(p,(1/3.0)*(c.FaceCorner(i,1)+E0+E1));
	N.ProcSets::Copy(p,(1/3.0)*(c.FaceCorner(i,2)+E1+E2));
	N.ProcSets::Copy(p,(1/3.0)*(E0+E1+E2));
    }
}

void check_mesh (const Mesh& M) {
    int dim = M.dim();
    double dmax = -infty;
    for (cell c = M.cells(); c!=M.cells_end(); ++c) {
        BFParts BF(M,c);
        if (!BF.onBnd()) continue;
        for (int face=0; face<c.Faces(); ++face) {
            if (BF[face] != 333) continue;
            for (int corner=0; corner<c.FaceCorners(face); ++corner) {
                Point P = c.FaceCorner(face,corner);
                double d = abs(sqrt((P[0]-10)*(P[0]-10)+P[1]*P[1])-1);
                if (d>dmax) dmax = d;
            }
            for (int edge=0; edge<c.FaceEdges(face); ++edge) {
                int k = c.faceedge(face,edge);
                Point P = c[c.LocalEdge(k)];
                double d = abs(sqrt((P[0]-10)*(P[0]-10)+P[1]*P[1])-1);
                if (d>dmax) dmax = d;
            }
        }
    }
    dmax = PPM->Max(dmax);
}

Point project (const Point& P) {
    Point D = P-Point(10,0,P[2]);
    double d = norm(D);
    Point T = D/d+Point(10,0,P[2]);
    return T;
}

Point project2D (const Point& P) {
    Point D = P-Point(10,0);
    double d = norm(D);
    Point T = D/d+Point(10,0);
    return T;
}

void RefineMesh (const Mesh& M, Mesh& N) {  
    int dim = M.dim();
    for (procset p=M.procsets(); p!=M.procsets_end(); ++p) {
	N.ProcSets::Copy(p);
	if (!RefineEdge(p,M,N))
	    RefineFace(p,M,N);
    }
    vector<Point> z;
    vector<Point> x;
    vector<cell> C(8);
    for (cell c=M.cells(); c!=M.cells_end(); ++c) {
	BFParts bnd(M,c);
	const Rules& R = c.Refine(z);
	for (int i=0; i<R.size(); ++i) {
	    R[i](z,x);
	    if (bnd.onBnd()) {
		Cell* t = CreateCell(R[i].type(),c.Subdomain(),x);
		for (short face=0;face<t->Faces();++face) {
		    int id = R[i].face[face];
		    if (id != -1)
			if (bnd[id] == 333) 
			    for (int edge=0;edge<t->FaceEdges(id);++edge) {
				int k = t->faceedge(face,edge);
				Point P = t->Edge(k);
				if (dim==3) x[8+k] = project(P);
				else x[4+k] = project2D(P);
			    }
		}
	    }
	    C[i] = N.InsertCell(R[i].type(),c.Subdomain(),x);
	}
	BFParts b(M,c);
	if (b.onBnd())
	    for (int k=0; k<R.size(); ++k) 
		for (int j=0; j<C[k].Faces(); ++j) {
		    int i = R[k].face[j];
		    if (i != -1)
			if (b[i] != -1)
			    N.BoundaryFaces::Insert(C[k].Face(j),b[i]);
		}
    }
    N.Finish();
    check_mesh(N);
}
void Meshes::Refine () {
    ++finelevel;
    M.resize(finelevel+1);
    M[finelevel] = new Mesh;
    RefineMesh(*M[finelevel-1],*M[finelevel]);
}

void CoarseMeshBoundary (Mesh& M, const CoarseGeometry& CG) {
    vector<Point> z(CG.Coordinates.begin(),CG.Coordinates.end());
    if (CG.Face_Ids.size() == 0)
	for (face f = M.faces(); f!=M.faces_end(); ++f) 
	    if (f.Right() == Infty)
		M.BoundaryFaces::Insert(f(),1);
    for (list<Ids>::const_iterator i = CG.Face_Ids.begin(); 
	 i!=CG.Face_Ids.end(); ++i) {
	int n = i->size();
	Assert(n == i->type());
	Point x = zero;
	for (int k=0; k<n; ++k) x += z[i->id(k)];
	x *= (1.0 / n);
	M.BoundaryFaces::Insert(x,i->flag());
    }
}
void CoarseMeshBoundary (Mesh& M, const CoarseGeometry& CG, 
			 map<string,list<int> >& bc_nodes,
			 map<string,int>& key_to_bc,
			 map<string,list<Point> >& surf,
			 map<string,int>& surf_to_bc) {
    if (!PPM->master()) return;
    // insert bnd_faces and setting them to 0=default
    for (face f = M.faces(); f!=M.faces_end(); ++f) 
	if (f.Right() == Infty)
	    M.BoundaryFaces::Insert(f(),0);
//    mout << " IN CMB (M,CG,bc_nodes)  : "<<endl<<bc_nodes<<endl;
    vector<Point> z(CG.Coordinates.begin(),CG.Coordinates.end());
    // looping over number of different BCs
    for (map<string,int>::iterator k2b_it=key_to_bc.begin();
	 k2b_it != key_to_bc.end(); ++k2b_it) {
	int bc_id = k2b_it->second;
//	map<string,list<int> >::const_iterator it = bc_nodes[k2b_it->first];
	vector<int> node_id(bc_nodes[k2b_it->first].begin(),
			    bc_nodes[k2b_it->first].end());
//	mout << "Looping over BC    " <<k2b_it->first<<endl;
	for (bnd_face b=M.bnd_faces(); b != M.bnd_faces_end(); ++b) {
	    face f = M.find_face(b());
//	    mout <<"face : "<< f <<endl;
	    cell c = M.find_cell(f.Left());
	    if (c == M.cells_end()) c = M.find_cell(f.Right());
	    if (c == M.cells_end()) continue;
	    int cnt=0;
	    for (int i=0; i<c.Faces(); ++i) {
		if (c.Face(i) != b()) continue;
//		mout << "bnd_face : "<<b() <<endl;
//		int cnt=0;  // corner counter
		for (int j=0; j<c.FaceCorners(i); ++j) {
		    Point FACECORNER = c.FaceCorner(i,j);
		    for (int k=0; k<node_id.size(); ++k) {
			if (FACECORNER == z[node_id[k]]) ++cnt;
		    }
		}
	    }
	    if (cnt>=3) { 
//		mout << "in cnt>=3  with  f() = "<<f()
//		     <<"    and    bc_id="<< bc_id<<endl;
//		M.BoundaryFaces::Remove(f());
		M.BoundaryFaces::Insert(f(),bc_id);
	    }
	}
    } 
    for (map<string,list<Point> >::iterator it=surf.begin(); 
	 it!=surf.end(); ++it) {
	int bc_id=surf_to_bc[it->first];
//	vector<Point> FM (it.begin(),it.end())  // FaceMidpoints
	for (list<Point>::iterator m=it->second.begin(); 
	     m!=it->second.end(); ++m) 
	    M.BoundaryFaces::Insert(*m,bc_id);
    }
}
void CoarseMesh (Mesh& M, const CoarseGeometry& CG) {
    vector<Point> z(CG.Coordinates.begin(),CG.Coordinates.end());
    for (list<Ids>::const_iterator i = CG.Cell_Ids.begin(); 
		 i!=CG.Cell_Ids.end(); ++i) {
		vector<Point> x(i->size());
		for (int k=0; k<i->size(); ++k) x[k] = z[i->id(k)];
		M.InsertCell(M.Type(i->type(),x),i->flag(),x);
    }
}

void SlimCoarseMesh (string name, Mesh& M) {
    string GeoPath = "./";     ReadConfig(Settings,"GeoPath",GeoPath); 
    if (!PPM->master()) return;
    string GeoName = GeoPath + "conf/geo/" + name;
	
    cout << "slim " << name << endl;
	
//    SlimCoarseGeometry CG(name);
}

// the standard M++ geometry file is used here
void CoarseMesh (string name, Mesh& M) {
	
    if (name == "3d.slim") {
		SlimCoarseMesh(name,M);
		Exit();
    }
	
    string GeoPath = "./";     ReadConfig(Settings,"GeoPath",GeoPath); 
    if (!PPM->master()) return;
    string GeoName = name + ".geo";
    if (!FileExists(GeoName)) GeoName = GeoPath + "conf/geo/" + GeoName;
    if (!FileExists(GeoName)) Exit("no mesh " + name);
    M_ifstream geofile(GeoName.c_str());
    SimpleCoarseGeometry CG(geofile);
    CoarseMesh (M,CG);
    CoarseMeshBoundary(M,CG);
}

void Meshes::Init () {
    M[0]->Finish();
    if (M[0]->dim() == 2) Point_2d();
    finelevel = 0;       ReadConfig(Settings,"level",finelevel);
    plevel = finelevel;  ReadConfig(Settings,"plevel",plevel);
    if (plevel > finelevel) {
		plevel = finelevel;
		mout << "plevel set to " << plevel << endl;
    }
    dout(10) << *M[0];
    M.resize(1+finelevel);
    for (int i=0; i<plevel; ++i) {
		M[i+1] = new Mesh;
		RefineMesh(*M[i],*M[i+1]);
    }
    string distr = "Stripes";
    ReadConfig(Settings,"Distribution",distr);
    Distribute(*M[plevel],distr);
    if (M[plevel]->dim() == 2) Point_2d();
    for (int i=plevel; i<finelevel; ++i) {
		M[i+1] = new Mesh;
		RefineMesh(*M[i],*M[i+1]);
    }
    dout(20) << M;
}

void Meshes::Init (int level, int _plevel) {
    M[0]->Finish();
    if (M[0]->dim() == 2) Point_2d();
    finelevel = level;
	plevel = _plevel;
    if (plevel > finelevel) {
		plevel = finelevel;
		mout << "plevel set to " << plevel << endl;
    }
    dout(10) << *M[0];
    M.resize(1+finelevel);
    for (int i=0; i<plevel; ++i) {
		M[i+1] = new Mesh;
		RefineMesh(*M[i],*M[i+1]);
    }
    string distr = "Stripes";
    ReadConfig(Settings,"Distribution",distr);
    Distribute(*M[plevel],distr);
    if (M[plevel]->dim() == 2) Point_2d();
    for (int i=plevel; i<finelevel; ++i) {
		M[i+1] = new Mesh;
		RefineMesh(*M[i],*M[i+1]);
    }
    dout(20) << M;
}

Meshes::Meshes () : M(1) {
    M[0] = new Mesh;
    string name = "UnitCube"; ReadConfig(Settings,"Mesh",name);
    CoarseMesh(name,*M[0]);
    Init();
}

Meshes::Meshes (const CoarseGeometry& CG, 
		map<string, list<int> >& bc_nodes,
		map<string,int >& key_to_bc, 
		map<string,list<Point> >& surf,
		map<string,int>& surf_id   ) : M(1) {
    M[0] = new Mesh;
    CoarseMesh(*M[0],CG);
    CoarseMeshBoundary(*M[0],CG,bc_nodes, key_to_bc, surf,surf_id);
    Init();
}

Meshes::Meshes (const char* Name) : M(1) {
    M[0] = new Mesh;
    string name(Name);
    CoarseMesh(name,*M[0]);
    Init();
}

Meshes::Meshes (const char* Name, int level, int plevel) : M(1) {
    M[0] = new Mesh;
    string name(Name);
    CoarseMesh(name,*M[0]);
    Init(level,plevel);
}

void CoarseMesh (const vector<Point>& coords, const vector<int>& ids, Mesh& M) {
    SimpleCoarseGeometry CG(coords, ids);
    CoarseMesh (M,CG);
    CoarseMeshBoundary(M,CG);
}

Meshes::Meshes (const vector<Point>& coords, const vector<int>& ids) : M(1) {
    M[0] = new Mesh;
    CoarseMesh(coords, ids, *M[0]);
    Init(0,0);
}

void CoarseMesh (double* coords, int n, int* ids, int m, Mesh& M) {
    SimpleCoarseGeometry CG(coords,n,ids,m);
    CoarseMesh (M,CG);
    CoarseMeshBoundary(M,CG);
}

Meshes::Meshes (double* coords, int n, int* ids, int m)  : M(1) {
    M[0] = new Mesh;
    CoarseMesh(coords, n, ids, m, *M[0]);
    Init(0,0);
}

void Meshes::ReSet(const char* Name, int level, int plevel) {
    for (int i=0; i<=finelevel; ++i) delete M[i];
    M.resize(1);
    M[0] = new Mesh;
    string name(Name);
    CoarseMesh(name,*M[0]);
    Init(level,plevel);
}
