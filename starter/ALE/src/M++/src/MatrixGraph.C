// file: MatrixGraph.C
// author: Christian Wieners
// $Header: /public/M++/src/MatrixGraph.C,v 1.14 2009-08-20 15:36:20 wieners Exp $

#include "MatrixGraph.h"

void Rows::AddEntry (const Point& x, const Point& y) {
    (*this)[x].Entries::Insert(y); }

void Rows::AddEntries (const Point& x, const Point& y) {
    if (y < x) AddEntry(x,y); 
    else AddEntry(y,x); 
}

void Rows::Insert (const Point& x, int n) { 
    if (find(x) == end()) (*this)[x] = Row(n); 
}

void Rows::Numbering () {
    int nR = size();
    vector<RowIterator> r(nR);
    int i = 0;
    for (RowIterator ri=begin(); ri!=end(); ++ri) r[i++] = ri;
    sort(r.begin(),r.end(),Less());
    for (int j=0; j<nR; ++j) r[j]->second.SetId(j);
    d = 0;
    m = 0;
    for (int i=0; i<nR; ++i) {
        r[i]->second.SetId(i);
        r[i]->second.SetEntry(m);
        ++d;
        m += r[i]->second.n() * r[i]->second.n(); 
        for (Row::EntryIterator e = (r[i]->second).begin(); 
             e != (r[i]->second).end(); ++e) {
            row rr = find_row(e->first());
            int j = rr.Id();
            e->second.SetId(j);
            e->second.SetEntry(m);
            ++d;
            m += 2 * r[i]->second.n() * r[j]->second.n(); 
        }
    }
}

void Rows::Insert_infty (int n) { 
    if (n == 0) return;
    if (find(Infty) == end()) (*this)[Infty] = Row(n); 
    row r_infty = find_row(Infty);
    for (row r=rows(); r!=rows_end(); ++r)
        if (r != r_infty)
            AddEntries (r(),Infty);
}

int Rows::Id (const Point& z) const { 
    row r = find_row(z);
    return r.Id();
}

int Rows::Idx (const Point& z) const { 
    row r = find_row(z);
    if (r == rows_end()) return -1;
    return r.Id();
}

int Rows::GetEntry (const row& r0, const row& r1) const { 
    if (r0 == r1) return r0.GetEntry();
    if (r1 < r0) return r0.GetEntry(r1()); 
    else return r1.GetEntry(r0())+r0.n()*r1.n(); 
}

int Rows::GetEntryX (const row& r0, const row& r1) const { 
    if (r0 == r1) return r0.GetEntry();
    if (r1 < r0) return r0.GetEntryX(r1()); 
    else return r1.GetEntryX(r0())+r0.n()*r1.n(); 
}

int Rows::GetDoubleEntryX (const row& r0, const row& r1) const { 
    if (r0 == r1) return r0.GetEntry();
    if (r1 < r0) return r0.GetEntryX(r1()); 
    else return r1.GetEntryX(r0());
}

int Rows::Dof (const Point& z) const { 
    row r = find_row(z);
    return r.n();
}

void MatrixGraph::AddCell (const cell& c, int depth) {
    vector<Point> z;
    NodalPoints(c,z);
    vector<short> n;
    NodalDoFs(c,n);
    for (int i=0; i<z.size(); ++i) Rows::Insert(z[i],n[i]);
    if (depth)
        for (int i=1; i<z.size(); ++i)
            for (int j=0; j<i; ++j)
                Rows::AddEntries(z[i],z[j]);
}

void MatrixGraph::SetProcSetsCell (const cell& c) {
    for (int i=0; i<c.Faces(); ++i) {
        procset p = M.find_procset(c.Face(i));
        if (p == M.procsets_end()) continue;
        vector<Point> z;
        NodalPoints(c,z);
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
	    ProcSets::Add(z[j],p);
        }
	for (int l=0; l<c.FaceEdges(i); ++l) {
	    procset p = M.find_procset(c.FaceEdge(i,l));
	    if (p == M.procsets_end()) continue;
	    for (int k=0; k<NodalPointsOnEdge(c,c.faceedge(i,l)); ++k) {
		int nk = NodalPointOnEdge(c,c.faceedge(i,l),k);
		ProcSets::Add(z[nk],p);
	    }
	}
    }
}

void MatrixGraph::SetProcSetsOverlapCell (const cell& c) {
    for (int i=0; i<c.Faces(); ++i) {
        procset p = M.find_procset(c.Face(i));
        if (p == M.procsets_end()) continue;
        vector<Point> z;
        NodalPoints(c,z);
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
	    ProcSets::Append(z[j],p);
        }
	for (int l=0; l<c.FaceEdges(i); ++l) {
	    procset p = M.find_procset(c.FaceEdge(i,l));
	    if (p == M.procsets_end()) continue;
	    for (int k=0; k<NodalPointsOnEdge(c,c.faceedge(i,l)); ++k) {
		int nk = NodalPointOnEdge(c,c.faceedge(i,l),k);
		ProcSets::Append(z[nk],p);
	    }
	}
    }
}

void MatrixGraph::IdentifyCell (const cell& c) {
    for (int i=0; i<c.Faces(); ++i) {
        identifyset is = M.find_identifyset(c.Face(i));
        if (is == M.identifysets_end()) continue;
        int mode = M.Part(c.Face(i));
        vector<Point> z;
        NodalPoints(c,z);
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
            dpout(3) << "ident: i " << i 
                     << " k " << k << " j " << j << endl; 
            Identify(z[j],mode);
	    procset p = M.find_procset(z[j]);
	    if (p == M.procsets_end()) continue;
            ProcSets::Add(z[j],p);
        }
        procset p = M.find_procset(is());
        if (p == M.procsets_end()) continue;
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
            ProcSets::Add(z[j],p);
        }
    }
}

void MatrixGraph::IdentifyOverlapCell (const cell& c) {
    for (int i=0; i<c.Faces(); ++i) {
        identifyset is = M.find_identifyset(c.Face(i));
        if (is == M.identifysets_end()) continue;
        int mode = M.Part(c.Face(i));
        vector<Point> z;
        NodalPoints(c,z);
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
            dpout(3) << "ident: i " << i 
                     << " k " << k << " j " << j << endl; 
            Identify(z[j],mode);
	    procset p = M.find_procset(z[j]);
	    if (p == M.procsets_end()) continue;
            ProcSets::Append(z[j],p);
        }
        procset p = M.find_procset(is());
        if (p == M.procsets_end()) continue;
        for (int k=0; k<NodalPointsOnFace(c,i); ++k) {
            int j = NodalPointOnFace(c,i,k);
            ProcSets::Append(z[j],p);
        }
    }
}

int MatrixGraph::pSize () const { 
    int cnt = 0;
    for (row r=rows(); r!=rows_end(); ++r)
        if (master(r()))
            cnt += r.n();
    return PPM->Sum(cnt); 
}

MatrixGraph::MatrixGraph (const Mesh& m, const dof& D, int depth) 
    : dof(D), M(m) {
    for (cell c=M.cells(); c!=M.cells_end(); ++c) 
        AddCell(c,depth);
    for (cell c=M.overlap(); c!=M.overlap_end(); ++c) 
        AddCell(c,depth);
    Init();
}

void MatrixGraph::Init() {
    Insert_infty(n_infty()); 
    dpout(5) << "M ProcSets: " << M.ProcSets::size() << endl 
	     << M.ProcSets::ref();
    dpout(1) << "M IdentifySets: " << M.IdentifySets::size()
	     << endl << M.IdentifySets::ref();
    dpout(5) << "A ProcSets: " << ProcSets::size() << endl 
             << ProcSets::ref();
    if (M.parallel()) {
	for (row r=rows(); r!=rows_end(); ++r) {
	    procset p = M.find_procset(r());
	    if (p != M.procsets_end())
		ProcSets::Copy(p);
	}
        for (cell c=M.cells(); c!=M.cells_end(); ++c) 
            SetProcSetsCell(c);
	for (cell c=M.overlap(); c!=M.overlap_end(); ++c) 
            SetProcSetsOverlapCell(c);
	if (n_infty()) ProcSets::AddInfty();
    }
    dpout(5) << "a ProcSets: " << ProcSets::size() << endl 
             << ProcSets::ref();
    if (M.identify()) {
	for (identifyset is=M.identifysets();is!=M.identifysets_end();++is) {
	    row r = find_row(is());
	    if (r != rows_end()) 
		IdentifySets::Insert(is);
	}
        for (cell c=M.cells(); c!=M.cells_end(); ++c) 
            IdentifyCell(c);
	for (cell c=M.overlap(); c!=M.overlap_end(); ++c) 
            IdentifyOverlapCell(c);
	for (identifyset is=M.identifysets();is!=M.identifysets_end();++is) {
	    row r = find_row(is());
	    if (r != rows_end()) 
		IdentifySets::Insert(is);
	}
	for (identifyset is=identifysets();is!=identifysets_end();++is) {
	    for (int i=0; i<is.size(); ++i) {
		procset p = find_procset(is[i]);
		if (p == procsets_end()) continue;
		for (int j=0; j<is.size(); ++j) 
		    ProcSets::Add(is[j],p);
	    }
	}
	for (identifyset is=identifysets(); is!=identifysets_end(); ++is) {
	    int n = Rows::Dof(is()); 
	    for (int i=0; i<is.size(); ++i) 
		Rows::Insert(is[i],n);
	    procset p = find_procset(is());
	    if (p == procsets_end()) continue;
	    for (int i=0; i<is.size(); ++i) 
		ProcSets::Copy(p,is[i]);
	}
    }
    Numbering();
    int nR = Rows::size();

   
    
    index = new int [nR+1];
    diag = new int [nR+1];
    n = new int [nR];
    column = new int [nC()];
    matentry = new int [nC()];
    vector<row> r(nR);
    for (row ri=rows(); ri!=rows_end(); ++ri) {
        r[ri.Id()] = ri;
        n[ri.Id()] = ri.n();
    } 
    int d = 0;
    index[0] = 0;
    diag[0] = d;
    matentry[d] = 0;
    for (int i=0; i<nR; ++i) {
        index[i+1] = index[i] + n[i];
        column[d] = i;
        matentry[d] = r[i].GetEntry();
        ++d;
        for (entry e=r[i].entries(); e!=r[i].entries_end(); ++e) {
            column[d] = e.Id();
            matentry[d] = e.GetEntry();
            ++d;
        }
        diag[i+1] = d;
    }
    N = index[nR];	
    mout << pSize() << " unknowns " << endl; 
    dout(3) << " for " << dof::Name() << endl;
}


ExchangeBuffer& MatrixGraph::AccumulateParallelBuffer () { APE.Rewind(); return APE; }
ExchangeBuffer& MatrixGraph::CollectParallelBuffer () { CPE.Rewind();return CPE; }
ExchangeBuffer& MatrixGraph::AccumulateIdentifyBuffer () { AIE.Rewind(); return AIE; }
ExchangeBuffer& MatrixGraph::CollectIdentifyBuffer () { CIE.Rewind();return CIE; }
const Extension& MatrixGraph::GetExtension (const char* s) const { 
    map<string,Extension*>::const_iterator m = Ex.find(s);
    if (m == Ex.end()) Exit(string("wrong subvector name ")+string(s));
    return *m->second;
}

const SubVectorMask& MatrixGraph::Mask (const char* name) const {
    return GetSubVector(name);
}

MatrixGraph::~MatrixGraph () {
    delete [] n;
    delete [] column;
    delete [] diag;
    delete [] index;
    delete [] matentry;
//	for (map<string,Extension*>::iterator m= Ex.begin(); m!=Ex.end(); ++m)
//	    delete m->second;
}

ostream& operator << (ostream& s, const MatrixGraph& G) {
    s << "dof" << endl;
    for (int i=0; i<G.Rows::size(); ++i) s << G.Dof(i) << " ";
    s << endl << "index" << endl;
    for (int i=0; i<=G.Rows::size(); ++i) s << G.Index(i) << " ";
    s << endl << "diag" << endl;
    for (int i=0; i<=G.Rows::size(); ++i) s << G.Diag(i) << " ";
    s << endl << "column" << endl;
    for (int i=0; i<G.nC(); ++i) s << G.Column(i) << " ";
    s << endl << "entry" << endl;
    for (int i=0; i<G.nC(); ++i) s << G.Entry(i) << " ";
    return s << endl << "Rows: " << endl << G.Rows::ref()
             << "ProcSets: " << G.ProcSets::size() << endl 
             << G.ProcSets::ref()
             << "IdentifySets: " << endl << G.IdentifySets::size()
             << endl << G.IdentifySets::ref();
}

rows::rows (const matrixgraph& g, const cell& c) {
    vector<Point> z;
    g.NodalPoints(c,z);
    resize(z.size());
    for (int i=0; i<size(); ++i) {
        (*this)[i] = g.find_row(z[i]);
    }
}

rows::rows (const matrixgraph& g, const cell& c, int face) {
    vector<Point> z;
    g.NodalPoints(c,z);
    resize(g.NodalPointsOnFace(c,face));
    for (int i=0; i<size(); ++i) {
        (*this)[i] = g.find_row(z[g.NodalPointOnFace(c,face,i)]);
    }
}

MatrixGraphs::MatrixGraphs (Meshes& M, const dof& D, int d) {
    G.resize(M.Level()-M.pLevel()+1);
    for (level=0; level<G.size(); ++level)
        G[level] = new MatrixGraph (M[level+M.pLevel()],D,d);
    --level;
}

MatrixGraphs::~MatrixGraphs () { for (int l=0; l<G.size(); ++l) delete G[l]; }

void MatrixGraphs::ReSet (Meshes& M, const dof& D, int d) {
    for (int l=0; l<G.size(); ++l) delete G[l];
    G.resize(M.Level()-M.pLevel()+1);
    for (level=0; level<G.size(); ++level)
        G[level] = new MatrixGraph (M[level+M.pLevel()],D,d);
    --level;
}

ostream& operator << (ostream& s, const MatrixGraphs& g) {
    for (int i=0; i<g.G.size(); ++i)
        s << "MatrixGraph on level " << i << endl << g[i];
    return s;
}
