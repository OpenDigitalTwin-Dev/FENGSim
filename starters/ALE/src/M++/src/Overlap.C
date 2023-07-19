class Overlap {
    Mesh& M;
    vector< list<Cell> > MarkedCells;
 public:
    Overlap (Mesh& m) : M(m), MarkedCells(PPM->size()) {}
    int Marked () { 
	int n = 0;
	for (short q=0; q<PPM->size(); ++q) 
	    n += MarkedCells[q].size(); 
	return n;
    }
    void MarkCell (const Cell* C, int q) { MarkedCells[q].push_back(*C); }
    void MarkLeftRight () {
	int q = PPM->proc();
	for (cell c=M.cells(); c!=M.cells_end(); ++c) 
	    if (q > 0)                   MarkCell(c->second,q-1);
	    else if (q+1 < PPM->size())  MarkCell(c->second,q+1);
    }
    void MarkAll () {
	for (cell c=M.cells(); c!=M.cells_end(); ++c)
	    for (int q=0; q<PPM->size(); ++q)
		if (q != PPM->proc())
		    MarkCell(c->second,q);
    }
    void MarkOneLayer () {
	for (cell c=M.cells(); c!=M.cells_end(); ++c)
	    for (int i=0; i<c.Faces(); ++i) {
		procset p = M.find_procset(c.Face(i));
		if (p != M.procsets_end())
		    for (int j=0; j<p.size(); ++j)
			if (p[j] != PPM->proc())
			    MarkCell(c->second,p[j]);
	    }
    }
    void MarkTwoLayer () {
	DistributeOverlap(M,"OneLayer");
	for (cell c=M.overlap_cells(); c!=M.overlap_end(); ++c) {
	    for (int i=0; i<c.Faces(); ++i) {
		procset p = M.find_procset(c.Face(i));
		if (p != M.procsets_end())
		    for (int j=0; j<p.size(); ++j)
			if (p[j] != PPM->proc())
			    MarkCell(c->second,p[j]);
	    }
	}
    }
    void OverlapProcSets () {
	ExchangeBuffer E;
	for (procset p = M.procsets(); p != M.procsets_end(); ++p) {
	    if (p.size() < 3) continue;
	    for (int i=0; i<p.size(); ++i) { 
		if (p[i] == PPM->proc()) continue;
		E.Send(p[i]) << p() << *p;
	    }
	}
	E.CommunicateSizeBuffer();
	for (short q=0; q<PPM->size(); ++q) { 
	    while (E.Receive(q).size() < E.Receive(q).Size()) {
		Point P;
		ProcSet PS;
		E.Receive(q) >> P >> PS;
		M.InsertProcSet(P,PS);
	    }
	}
    }
    void Communicate () {
	for (short q=0; q<PPM->size(); ++q) { 
	    for (list<Cell>::const_iterator c = MarkedCells[q].begin();
		 c != MarkedCells[q].end(); ++c) {
		for (int i=0; i<c->Corners(); ++i) 
		    M.AddOverlapProcSet(c->Corner(i),q);
		for (int i=0; i<c->Faces(); ++i) 		    
		    M.AddOverlapProcSet(c->Face(i),q);
		for (int i=0; i<c->Edges(); ++i)
		    M.AddOverlapProcSet((*c).Edge(i),q);
		M.AddOverlapProcSet((*c)(),q);
	    }
	}
	ExchangeBuffer E;
	for (short q=0; q<PPM->size(); ++q) { 
	    for (list<Cell>::const_iterator c = MarkedCells[q].begin();
		 c != MarkedCells[q].end(); ++c) {
		E.Send(q) << short(c->Subdomain());
		E.Send(q) << short(c->Corners());
		for (int i=0; i<c->Corners(); ++i) {
		    E.Send(q) << c->Corner(i);
		    E.Send(q) << *M.find_procset(c->Corner(i));
		}
		for (int i=0; i<c->Faces(); ++i) 		    
		    E.Send(q) << *M.find_procset(c->Face(i));
		for (int i=0; i<c->Edges(); ++i)
		    E.Send(q) << *M.find_procset(c->Edge(i));
		E.Send(q) << *M.find_procset((*c)());
	    }
	}
	E.CommunicateSizeBuffer();
	for (short q=0; q<PPM->size(); ++q) { 
	    while (E.Receive(q).size() < E.Receive(q).Size()) {
		short sd,m;
		E.Receive(q) >> sd;
		E.Receive(q) >> m;
		vector<Point> Q(m);
		ProcSet PS;
		for (int j=0; j<m; ++j) {
		    E.Receive(q) >> Q[j];
		    E.Receive(q) >> PS;
		    M.InsertProcSet(Q[j],PS);
		}
		cell c = M.InsertOverlapCell(Q,sd);
		for (int i=0; i<c.Faces(); ++i) {
		    E.Receive(q) >> PS;
		    M.InsertProcSet(c.Face(i),PS);
		}
		for (int i=0; i<c.Edges(); ++i) {
		    E.Receive(q) >> PS;
		    M.InsertProcSet(c.Edge(i),PS);
		}
		E.Receive(q) >> PS;
		M.InsertProcSet(c(),PS);
	    }
	}
	OverlapProcSets();
    }
};

void DistributeOverlap (Mesh& M, const string& overlap) {
    if (PPM->size() == 1) return;
    Date Start;
    Overlap O(M);
    if (overlap == "LeftRight") O.MarkLeftRight();
    if (overlap == "1")         O.MarkLeftRight();
    if (overlap == "All")       O.MarkAll();
    if (overlap == "OneLayer")  O.MarkOneLayer();
    if (overlap == "TwoLayer")  O.MarkTwoLayer();
    dpout(9) << O.Marked() << " cells marked for overlap" << endl;
    O.Communicate();
    mout << "overlaping " << M.nO() << " cells on proc 0 " << endl; 
    dpout(18) << "overlap mesh " << endl << M;
    tout(7) << "overlap " << overlap << " " << Date() - Start << "\n";
}









class MatrixGraphReference {
    const MatrixGraph& G;
 protected:
    int n;
    int N;
    int M;
    const int* index;
    const int* dindex;
    const int* dof;
    const int* diag;
    const int* column;
    const int* entry;
    const int* Dim;
 public: 
    int nR () const { return n; }
    row rows () const { return G.rows(); }
    row rows_end () const { return G.rows_end(); }
    row find_row (const Point& P) const { return G.find_row(P); }
    cell cells () const { return G.GetMesh().cells(); }
    cell cells_end () const { return G.GetMesh().cells_end(); }
    cell find_cell (const Point& P) const { return G.GetMesh().find_cell(P); }
    const DoF& GetDoF() const { return G.GetDoF(); }
    int Type (int i) const { return G.Type(i); }
    int Id (const Point& P) const { return G.Id(P); }
    int Idx (const Point& P) const { return G.Idx(P); }
    template <class C> int Id (const C& c) const { return G.Id(c()); }
    template <class C> int Idx (const C& c) const { return G.Idx(c()); }
    bool parallel () const { return G.parallel(); }
    const SubVectorMask& Mask (const char* name) const { return G.Mask(name); }
    MatrixGraphReference (const MatrixGraph& g) :
	G(g), n(g.nR()), N(g.dim()), M(g.size()), 
	index(g.Index()), dof(g.Dof()),
	diag(g.Diag()), column(g.Column()), entry(g.MEntry()), Dim(g.MDim()),
	dindex(g.DIndex()) {}
    const MatrixGraph& GetMatrixGraph () const { 
	return G; 
    }
    const Mesh& GetMesh () const {return G.GetMesh();}
    int dim () const { return N; }
    int Index (int i) const { return index[i]; }
    int DIndex (int i) const { return dindex[i]; }
    int Diag (int i) const { return diag[i]; }
    const int* Diag () const { return diag; }
    int Dof (int i) const { return dof[i]; }
    int Dof (const Point& P) const { return dof[G.Id(P)]; } 
    const int* Dof () const { return dof; }
    int Column (int i) const { return column[i]; }
    const int* Column () const { return column; }
    int Entry (int i) const { return entry[i]; }
    int Dims (int i) const { return Dim[i]; }
    int size (const Point& P) const { return dof[G.Id(P)]; } 
    int size () const { return G.size(); } 
    int pSize () const { return G.pSize(); } 
    template <class O> void Accumulate (O& o) const { 
	G.GetInterface().Accumulate(o); 
    }
    void DeleteCopies (Vector& x) const { G.GetInterface().DeleteCopies(x); }
    void DistributeDirichlet (Vector& x) const { 
	G.GetInterface().DistributeDirichlet(x); 
    }
    void Collect (Vector& x) const { G.GetInterface().Collect(x); }
};
inline MatrixGraph::MatrixGraph (const MatrixGraphReference& G) :
    mesh(G.GetMesh()), I(G.GetMesh()), 
    DOF(G.GetMatrixGraph().GetDoF()), 
    ps_end(G.GetMesh().procsets_end()) {} 

class Index : public vector<short> {
    short j;
    int Size (const bool* mask, int n) const {
	int s = 0;
	for (int i=0; i<n; ++i) 
	    if (mask[i]) ++s;
	return s;
    }
 public:
    Index () {} 
    Index (int J, const bool* mask, int n) : 
	vector<short>(Size(mask,n)), j(J) {
	int k = 0;
	for (int l=0; l<n; ++l)
	    if (mask[l]) 
		(*this)[k++] = l;
    }
    Index (int J, const bool* mask, int n, int s) : vector<short>(s), j(J) {
	int k = 0;
	for (int l=0; l<n; ++l)
	    if (mask[l]) 
		(*this)[k++] = l;
    }
    short operator () () const { return j; } 
};
const Index index0;
class Rows;
class Indices : public vector<Index> {
    int Size (const bool* mask, int n) const {
	int s = 0;
	for (int i=0; i<n; ++i) 
	    if (mask[i]) ++s;
	return s;
    }
    int Size (const Rows&, const SubVectorMask&) const;
public:
    Indices (const Rows&, const SubVectorMask&);
    const Index& operator () (int j) const {
	for (int i=0; i<size(); ++i)
	    if ((*this)[i]() == j) return (*this)[i];
	return index0;
    }
};
inline ostream& operator << (ostream& s, const Indices& I) {
    for (int i=0; i<I.size(); ++i) {
	s << I[i]() << " :"; 
	for (int j=0; j<I[i].size(); ++j) 
	    s << " " << I[i][j];
	s << endl;
    }
    return s;
} 

class Rows : public vector<row> {
 public:
    Rows () {}
    Rows (const cell& c, const MatrixGraphReference& G) {
	vector<Point> P;
	G.GetDoF().CellNodalPoints(c,P);
	resize(P.size());
	for (int i=0; i<size(); ++i) (*this)[i] = G.find_row(P[i]);
    }
    Rows (const vector<Point>& P, const MatrixGraphReference& G) {
	resize(P.size());
	for (int i=0; i<size(); ++i) (*this)[i] = G.find_row(P[i]);
    }
    Rows (const Rows& r, const Indices& I) : vector<row>(I.size()) {
	for (int i=0; i<I.size(); ++i) (*this)[i] = r[I[i]()];
    }
};
inline ostream& operator << (ostream& s, const Rows& R) {
    for (int i=0; i<R.size(); ++i) 
	s << R[i]->first << " : " << R[i]->second << endl;
    return s;
}
inline int Indices::Size (const Rows& r, const SubVectorMask& Mask) const {
    int s = 0;
    for (int i=0; i<r.size(); ++i) {
	const bool* mask = Mask[r[i].Type()];
	if (mask == 0) continue;
	if (Size(mask,r[i].n())) ++s;
    }
    return s;
}
inline Indices::Indices (const Rows& r, const SubVectorMask& Mask) : 
    vector<Index>(Size(r,Mask)) {
    int j = 0;
	for (int i=0; i<r.size(); ++i) {
	    const bool* mask = Mask[r[i].Type()];
	    if (mask == 0) continue;
	    int n = Size(mask,r[i].n());
	    if (n) (*this)[j++] = Index(i,mask,r[i].n(),n);
	}
}

inline void MatrixEntries (const Rows& R, int* e) { 
    int k = R.size();
    for (int i=0; i<k; ++i) {
	e[i*k+i] = R[i].GetEntry();
	for (int j=0; j<i; ++j) {
	    if (R[i]() > R[j]()) {
		e[i*k+j] = R[i].GetEntry(R[j]);
		e[j*k+i] = e[i*k+j] + R[i].n() * R[j].n();
	    }
	    else {
		e[j*k+i] = R[j].GetEntry(R[i]);
		e[i*k+j] = e[j*k+i] + R[i].n() * R[j].n();
	    }
	}
    }
}
inline void MatrixEntriesX (const Rows& R, int* d, int* e)
{ 
    int k = R.size();
    for (int i=0; i<k; ++i) {
	d[i] = R[i].n();
	e[i*k+i] = R[i].GetEntry();
	for (int j=0; j<i; ++j) {
	    if (R[i]->first > R[j]->first) {
		e[i*k+j] = R[i].GetEntryX(R[j]);
		if (e[i*k+j] == -1) e[j*k+i] = -1;
		else       	    e[j*k+i] = e[i*k+j] + d[i] * d[j];
	    }
	    else { 
		e[j*k+i] = R[j].GetEntryX(R[i]);
		if (e[j*k+i] == -1) e[i*k+j] = -1;
		else 		    e[i*k+j] = e[j*k+i] + d[i] * d[j];
	    }
	}
    }
}
