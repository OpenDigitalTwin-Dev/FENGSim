// file: Transfer.C
// author: Christian Wieners
//         Wolfgang Mueller (SerendipityTransfer 2 and 3D)
// $Header: /public/M++/src/Transfer.C,v 1.25 2009-11-26 09:04:23 wieners Exp $

#include "Transfer.h"

const int MaxDoFs = 100;

class SimpleTransfer : public Transfer {
 public:
    virtual ~SimpleTransfer () {}  
    virtual void loop (Vector& f, const Vector& c, int k0, int K) const {
	vout(5) << "SimpleTransfer: loop" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
            double s = 1 / double(n);
	    for (int j=0; j<n; ++j) 
		for (int k=k0; k<fDofi; ++k)
		    f(i,k) += s * c(Ii[j],k);
	}
    }
    virtual void multiply (Vector& f, const Vector& c) const {
	vout(5) << "SimpleTransfer: multiply" << endl;
	f = 0;
	loop(f,c,0,MaxDoFs);
	f.ClearDirichletValues(); 
    }
    virtual void loop_transpose (Vector& c, const Vector& f, int k0, int K) const {
	vout(5) << "SimpleTransfer: loop_transpose" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
            double s = 1 / double(n);
	    for (int j=0; j<n; ++j) 
		for (int k=k0; k<fDofi; ++k) 
		    c(Ii[j],k) += s * f(i,k);
	}
    }
    virtual void multiply_transpose (Vector& c, const Vector& f) const {
	vout(5) << "SimpleTransfer: multiply_transpose" << endl;
	c = 0;
	loop_transpose(c,f,0,MaxDoFs);
	c.ClearDirichletValues(); 
	Collect(c);
    }
    virtual void Project (const Vector& f, Vector& c) const {
	vout(5) << "SimpleTransfer: Project" << endl;
	for (row r=c.rows(); r!=c.rows_end(); ++r) {
	    row r_f = f.find_row(r());
	    if (r_f==f.rows_end()) continue;
//	    if (r_f==f.rows_end()) Exit("Stop");
	    for (int i=0; i<r.n(); ++i) {
		c(r,i) = f(r_f,i);
		c.D(r,i) = f.D(r_f,i);
	    }
	}
    }
};

class LinearTransfer : public SimpleTransfer {
 public:
    LinearTransfer () {}
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {
	I.resize(FG.nR());
	const Mesh& CM = CG.GetMesh();
	for (cell c = CM.cells(); c != CM.cells_end(); ++c) {
	    for (int i=0; i<c.Corners(); ++i) {
		int id = FG.Id(c[i]);
		I[id].resize(1);
		I[id][0] = CG.Id(c[i]);
	    }
	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(c[c.LocalEdge(i)]);
		I[id].resize(2);
		I[id][0] = CG.Id(c.EdgeCorner(i,0));
		I[id][1] = CG.Id(c.EdgeCorner(i,1));
	    }
	    int id = FG.Idx(c[c.LocalCenter()]);
	    if (id != -1) {
		I[id].resize(c.Corners());
		for (int i=0; i<c.Corners(); ++i) 
		    I[id][i] = CG.Id(c[i]);
	    }
	    if (c.plane()) continue;
	    for (int i=0; i<c.Faces(); ++i) {
		int n = c.FaceCorners(i);
		if (n != 4) break;
		int id = FG.Id(c[c.LocalFace(i)]);
		if (id != -1) {
		    I[id].resize(4);
		    for (int j=0; j<4; ++j) 
			I[id][j] = CG.Id(c.FaceCorner(i,j));
		}
	    }
	}
    }
};

class SerendipityTransfer : public SimpleTransfer {
public:
    SerendipityTransfer () {}
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {
	vout(5) << "SerendipityTransfer: Construct" << endl;
	I.resize(FG.nR());
	const Mesh& CM = CG.GetMesh();
	for (cell c = CM.cells(); c != CM.cells_end(); ++c) {

	    for (int i=0; i<c.Corners(); ++i) {
		int id = FG.Id(c[i]);
		I[id].resize(1);
		I[id][0] = CG.Id(c[i]);
	    }
	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(c[c.LocalEdge(i)]);
		I[id].resize(1);
		I[id][0] = CG.Id(c.Edge(i));
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(c[c.LocalFace(i)]);
		I[id].resize(8);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		for (int j=0; j<c.FaceEdges(i); ++j) 
		    I[id][n+j] = CG.Id(c.Edge(c.faceedge(i,j)));
	    }

	    int id = FG.Id(c[c.LocalCenter()]);
	    I[id].resize(20);	    
	    for (int j=0; j<c.Corners(); ++j)
		I[id][j] = CG.Id(c[j]);
	    for (int j=0; j<c.Edges(); ++j)
		I[id][8+j] = CG.Id(c.Edge(j));

	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalCorner(c.edgecorner(i,0))]+c[c.LocalEdge(i)]));
		I[id].resize(3);
		I[id][0] = CG.Id(c.EdgeCorner(i,0));
		I[id][1] = CG.Id(c.EdgeCorner(i,1));
		I[id][2] = CG.Id(c.Edge(i));
	    }

	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalCorner(c.edgecorner(i,1))]+c[c.LocalEdge(i)]));
		I[id].resize(3);
		I[id][0] = CG.Id(c.EdgeCorner(i,1));
		I[id][1] = CG.Id(c.EdgeCorner(i,0));
		I[id][2] = CG.Id(c.Edge(i));
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalEdge(c.faceedge(i,0))]+c[c.LocalFace(i)]));
		I[id].resize(9);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 		
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		bool circuit = c.faceedgecircuit(i);
		if (circuit) {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,2)));
		}
		else {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,3)));
		}
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalEdge(c.faceedge(i,1))]+c[c.LocalFace(i)]));
		I[id].resize(9);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 		
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		bool circuit = c.faceedgecircuit(i);
		if (circuit) {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,3)));
		}
		else {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,2)));
		}
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalEdge(c.faceedge(i,2))]+c[c.LocalFace(i)]));
		I[id].resize(9);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 		
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		bool circuit = c.faceedgecircuit(i);
		if (circuit) {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,0)));
		}
		else {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,1)));
		}
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalEdge(c.faceedge(i,3))]+c[c.LocalFace(i)]));
		I[id].resize(9);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 		
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		bool circuit = c.faceedgecircuit(i);
		if (circuit) {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,0)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,1)));
		}
		else {
		    I[id][n]   = CG.Id(c.Edge(c.faceedge(i,1)));
		    I[id][n+1] = CG.Id(c.Edge(c.faceedge(i,2)));
		    I[id][n+2] = CG.Id(c.Edge(c.faceedge(i,3)));
		    I[id][n+3] = CG.Id(c.Edge(c.faceedge(i,0)));
		}
	    }

	    for (int i=0; i<c.Faces(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalCenter()]+c[c.LocalFace(i)]));
		I[id].resize(21);
		int n = c.FaceCorners(i);
		for (int j=0; j<n; ++j) 		
		    I[id][j] = CG.Id(c.FaceCorner(i,j));
		for (int j=0; j<n; ++j) 		
		    I[id][n+j] = CG.Id(c.Edge(c.faceedge(i,j)));
		int fari = c.FarFace(i);
		for (int j=0; j<n; ++j) 		
		    I[id][2*n+j] = CG.Id(c.FaceCorner(fari,j));
		for (int j=0; j<n; ++j) 		
		    I[id][3*n+j] = CG.Id(c.Edge(c.faceedge(fari,j)));
		for (int j=0; j<n; ++j) 		
		    I[id][4*n+j] = CG.Id(c.Edge(c.MiddleEdge(i,j)));	
	    }
	}
    }
    void loop (Vector& f, const Vector& c, int k0, int K) const {
	vout(5) << "SerendipityTransfer: loop" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
	    double s;
	    if (n == 1) {
		for (int k=k0; k<fDofi; ++k)
		    f(i,k) += c(Ii[0],k);		
	    }
	    else if (n == 8) {
		s = -0.25;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.5;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
	    }
	    else if (n == 20) {
		s = -0.25;
		for (int j=0; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.25;
		for (int j=8; j<20; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
	    }
	    else if (n == 3) {
		for (int k=k0; k<fDofi; ++k) {
		    f(i,k) += 0.375 * c(Ii[0],k);
		    f(i,k) += (-0.125) * c(Ii[1],k);
		    f(i,k) += 0.75 * c(Ii[2],k);
		}
	    }
	    else if (n == 9) {
		s = -0.1875;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.375;
		for (int j=4; j<6; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		for (int k=k0; k<fDofi; ++k) {
		    f(i,k) += 0.75 * c(Ii[6],k);
		    f(i,k) += 0.25 * c(Ii[7],k);
		}
	    }
	    else if (n == 21) {
		s = -0.28125;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.375;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = -0.15625;
		for (int j=8; j<12; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.125;
		for (int j=12; j<16; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.1875;
		for (int j=16; j<20; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
	    }
	}
    }

    void multiply (Vector& f, const Vector& c) const {
	vout(5) << "SerendipityTransfer: multiply" << endl;
	f = 0;
	loop(f,c,0,MaxDoFs);
	f.ClearDirichletValues(); 
    }

    void loop_transpose (Vector& c, const Vector& f, int k0, int K) const {
	vout(5) << "SerendipityTransfer: loop_transpose" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
	    double s;
	    if (n == 1) {
		for (int k=k0; k<fDofi; ++k)
		    c(Ii[0],k) += f(i,k);		
	    }
	    else if (n == 8) {
		s = -0.25;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.5;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
	    }
	    else if (n == 20) {
		s = -0.25;
		for (int j=0; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.25;
		for (int j=8; j<20; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
	    }
	    else if (n == 3) {
		for (int k=k0; k<fDofi; ++k) {
		    c(Ii[0],k) += 0.375 * f(i,k);
		    c(Ii[1],k) += (-0.125) * f(i,k);
		    c(Ii[2],k) += 0.75 * f(i,k);
		}
	    }
	    else if (n == 9) {
		s = -0.1875;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.375;
		for (int j=4; j<6; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		for (int k=k0; k<fDofi; ++k) {
		    c(Ii[6],k) += 0.75 * f(i,k);
		    c(Ii[7],k) += 0.25 * f(i,k);
		}
	    }
	    else if (n == 21) {
		s = -0.28125;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.375;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = -0.15625;
		for (int j=8; j<12; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.125;
		for (int j=12; j<16; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.1875;
		for (int j=16; j<20; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
	    }
	}
    }

    void multiply_transpose (Vector& c, const Vector& f) const {
	vout(5) << "SerendipityTransfer: multiply_transpose" << endl;
	c = 0;
	loop_transpose(c,f,0,MaxDoFs);
	c.ClearDirichletValues(); 
	Collect(c);
    }

};

class Serendipity2DTransfer : public SimpleTransfer {
public:
    Serendipity2DTransfer () {}
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {
	I.resize(FG.nR());
	const Mesh& CM = CG.GetMesh();
	for (cell c = CM.cells(); c != CM.cells_end(); ++c) {
	    for (int i=0; i<c.Corners(); ++i) {
		int id = FG.Id(c[i]);
		I[id].resize(1);
		I[id][0] = CG.Id(c[i]);
	    }
	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(c[c.LocalEdge(i)]);
		I[id].resize(1);
		I[id][0] = CG.Id(c.Edge(i));
	    }
	    int id = FG.Id(c[c.LocalCenter()]);
	    I[id].resize(8);	    
	    for (int j=0; j<c.Corners(); ++j)
		I[id][j] = CG.Id(c[j]);
	    for (int j=0; j<c.Edges(); ++j)
		I[id][4+j] = CG.Id(c.Edge(j));

	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalCorner(c.edgecorner(i,0))]+c[c.LocalEdge(i)]));
		I[id].resize(3);
		I[id][0] = CG.Id(c.EdgeCorner(i,0));
		I[id][1] = CG.Id(c.EdgeCorner(i,1));
		I[id][2] = CG.Id(c.Edge(i));
	    }

	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalCorner(c.edgecorner(i,1))]+c[c.LocalEdge(i)]));
		I[id].resize(3);
		I[id][0] = CG.Id(c.EdgeCorner(i,1));
		I[id][1] = CG.Id(c.EdgeCorner(i,0));
		I[id][2] = CG.Id(c.Edge(i));
	    }

	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Id(0.5*(c[c.LocalEdge(i)]+c[c.LocalCenter()]));
		I[id].resize(9);
		for (int j=0; j<c.Corners(); ++j) 		
		    I[id][j] = CG.Id(c[j]);
		int k1 = (i+1) % 4;
		int k2 = (i+2) % 4;
		int k3 = (i+3) % 4;
		I[id][4]   = CG.Id(c.Edge(k1));
		I[id][4+1] = CG.Id(c.Edge(k3));
		I[id][4+2] = CG.Id(c.Edge(i));
		I[id][4+3] = CG.Id(c.Edge(k2));
	    }
	}
    }
    void loop (Vector& f, const Vector& c, int k0, int K) const {
	vout(5) << "SerendipityTransfer: loop" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
	    double s;
	    if (n == 1) {
		for (int k=k0; k<fDofi; ++k)
		    f(i,k) += c(Ii[0],k);		
	    }
	    else if (n == 8) {
		s = -0.25;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.5;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
	    }
	    else if (n == 3) {
		for (int k=k0; k<fDofi; ++k) {
		    f(i,k) += 0.375 * c(Ii[0],k);
		    f(i,k) += (-0.125) * c(Ii[1],k);
		    f(i,k) += 0.75 * c(Ii[2],k);
		}
	    }
	    else if (n == 9) {
		s = -0.1875;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		s = 0.375;
		for (int j=4; j<6; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			f(i,k) += s * c(Ii[j],k);
		for (int k=k0; k<fDofi; ++k) {
		    f(i,k) += 0.75 * c(Ii[6],k);
		    f(i,k) += 0.25 * c(Ii[7],k);
		}
	    }
	}
    }
    void multiply (Vector& f, const Vector& c) const {
	vout(5) << "Serendipity2DTransfer: multiply" << endl;
	f = 0;
	loop(f,c,0,MaxDoFs);
	f.ClearDirichletValues(); 
    }
    void loop_transpose (Vector& c, const Vector& f, int k0, int K) const {
	vout(5) << "SerendipityTransfer: loop_transpose" << endl;
	for (int i=0; i<I.size(); ++i) {
            const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            int fDofi = min(f.Dof(i),K);
	    double s;
	    if (n == 1) {
		for (int k=k0; k<fDofi; ++k)
		    c(Ii[0],k) += f(i,k);		
	    }
	    else if (n == 8) {
		s = -0.25;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.5;
		for (int j=4; j<8; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
	    }
	    else if (n == 3) {
		for (int k=k0; k<fDofi; ++k) {
		    c(Ii[0],k) += 0.375 * f(i,k);
		    c(Ii[1],k) += (-0.125) * f(i,k);
		    c(Ii[2],k) += 0.75 * f(i,k);
		}
	    }
	    else if (n == 9) {
		s = -0.1875;
		for (int j=0; j<4; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		s = 0.375;
		for (int j=4; j<6; ++j) 
		    for (int k=k0; k<fDofi; ++k)
			c(Ii[j],k) += s * f(i,k);
		for (int k=k0; k<fDofi; ++k) {
		    c(Ii[6],k) += 0.75 * f(i,k);
		    c(Ii[7],k) += 0.25 * f(i,k);
		}
	    }
	}
    }
    void multiply_transpose (Vector& c, const Vector& f) const {
	vout(5) << "Serendipity2DTransfer: multiply_transpose" << endl;
	c = 0;
	loop_transpose(c,f,0,MaxDoFs);
	c.ClearDirichletValues(); 
	Collect(c);
    }
};

class CosseratTransfer : public Transfer {
    Transfer* T1;
    Transfer* T2;
    string disc;
    int dim;
public:
    CosseratTransfer (int d = 3) : T1(0), T2(0), disc("CosseratP1"), dim(d) {
	ReadConfig(Settings,"Discretization",disc);
	if (disc=="CosseratP1") {
	    T1 = GetTransfer("VertexDoF");
	    T2 = T1;
	}
	else if (disc=="CosseratP2P1") {
	    if (dim==3) {
		T1 = GetTransfer("Serendipity");
		T2 = GetTransfer("VertexDoF");
	    } 
	    else {
		T1 = GetTransfer("Serendipity2D");
		T2 = GetTransfer("VertexDoF");
	    }
	}
	else if (disc=="CosseratP2") {
	    if (dim==3) T1 = GetTransfer("Serendipity");
	    else T1 = GetTransfer("Serendipity2D");
	    T2 = T1;
	}
	else {
	    Exit("Discretization not defined in CosseratTransfer!");
	}
    }
    ~CosseratTransfer () {
	if (disc=="CosseratP2P1") {
	    delete T2;
	    delete T1;
	}
	else {
	    T2 = 0;
	    delete T1;
	}
    }
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {    
	vout(5) << "CosseratTransfer: Construct" << endl;
	if (disc=="CosseratP2P1") {
	    T1->Construct(FG,CG);
	    T2->Construct(FG,CG);
	}
	else T1->Construct(FG,CG);
    }
    void multiply (Vector& f, const Vector& c) const {
	vout(5) << "CosseratTransfer: multiply" << endl;
	f = 0;
	T1->loop(f,c,0,dim);
	T2->loop(f,c,dim,3*dim-3);
	f.ClearDirichletValues();
    }
    void multiply_transpose (Vector& c, const Vector& f) const {
	vout(5) << "CosseratTransfer: multiply_transpose" << endl;
	c = 0;
	T1->loop_transpose(c,f,0,dim);
	T2->loop_transpose(c,f,dim,3*dim-3);
	c.ClearDirichletValues(); 
	Collect(c);
    }
    void Project (const Vector& f, Vector& c) const {
	vout(5) << "CosseratTransfer: Project" << endl;
	T1->Project(f,c);
	T2->Project(f,c);
    }
};

class QuadraticTransfer : public SimpleTransfer {
 public:
    QuadraticTransfer() {}
    virtual void Construct (const matrixgraph& FG, const matrixgraph& CG) {
	I.resize(FG.nR());
	const Mesh& CM = CG.GetMesh();
	for (cell c = CM.cells(); c != CM.cells_end(); ++c) {
	    for (int i=0; i<c.Corners(); ++i) {
		int id = FG.Id(c[i]);
		I[id].resize(1);
		I[id][0] = CG.Id(c[i]);
	    }
	    for (int i=0; i<c.Edges(); ++i) {
		int id = FG.Idx(c[c.LocalEdge(i)]);
		I[id].resize(1);
		I[id][0] = CG.Id(c.Edge(i));
	    }
	    for (int i=0; i<c.Edges(); ++i) {
		for (int j=0; j<2; ++j) {
//		    int id = FG.Idx(c[0.5*(c.LocalEdge(i)+c.EdgeCorner(i,j))]);
		    int id = FG.Idx(0.5*(c.Edge(i)+c.EdgeCorner(i,j)));
		    if (id == -1) {
			mout << "EE " << c.Edge(i) << "-"<<c.EdgeCorner(i,j)
			     <<"  |||  "<< c.LocalEdge(i) << "-"<<c[c.LocalEdge(i)]
			     <<"    :   "
			     << c[0.5*(c.LocalEdge(i)+c.EdgeCorner(i,j))]
			     <<endl;			
		    }
		    I[id].resize(2);
		    I[id][0] = CG.Idx(c.Edge(i));
		    I[id][1] = CG.Idx(c.EdgeCorner(i,j));
		}
	    }
	    // 2d cell
	    if (c.plane()) {
		// Triangle
		if (c.Corners()==3) {
		    for (int i=0; i<3; ++i) {
			int id = FG.Idx(0.5*(c.Edge(i)+c.Edge((i+1)%3)));
			I[id].resize(2);
			I[id][0] = CG.Idx(c.Edge(i));
			I[id][1] = CG.Idx(c.Edge((i+1)%3));
		    }
		}
		else {
		    int id = FG.Idx(c[c.LocalCenter()]);
		    if (id != -1) {
			I[id].resize(1);
			I[id][0] = CG.Idx(c.Center());
		    }
		    for (int i=0; i<c.Edges(); ++i) {
			int id = FG.Idx(0.5*(c.Edge(i) + c.Center()));
			if (id == -1) mout << "Mist"<<endl;
			I[id].resize(2);
			I[id][0] = CG.Idx(c.Edge(i));
			I[id][1] = CG.Idx(c.Center());
		    }
		}
	    }
	    // 3d cell - only hexahedra so far
	    else {
		int id = FG.Idx(c[c.LocalCenter()]);
		I[id].resize(1);
		I[id][0] = CG.Idx(c.Center());
		for (int i=0; i<c.Faces(); ++i) {
		    id = FG.Idx(c[c.LocalFace(i)]);
		    I[id].resize(1);
		    I[id][0] = CG.Id(c.Face(i));
		    for (int j=0; j<c.FaceEdges(i); ++j) {
			int id=FG.Idx(0.5*(c.Face(i) + c.FaceEdge(i,j)));
			if (id == -1) mout << "DRECK"<<endl;
			I[id].resize(2);
			I[id][0] = CG.Idx(c.Face(i));
			I[id][1] = CG.Idx(c.FaceEdge(i,j));
		    }
		    id = FG.Idx(0.5* (c.Face(i)+c.Center()));
		    I[id].resize(2);
		    I[id][0] = CG.Idx(c.Face(i));
		    I[id][1] = CG.Idx(c.Center());
		}
	    }
	}
    }
};


class TaylorHoodQuadraticTransfer : public Transfer {
    Transfer* T1;
    Transfer* T2;
public:
    TaylorHoodQuadraticTransfer() 
	: T1(GetTransfer("Quadratic")), T2(GetTransfer("VertexDoF")) { }
    virtual void Construct (const matrixgraph& FG, const matrixgraph& CG) {    
	mout << "in TaylorHoodQuadraticTransfer ..."<<endl;
	T1->Construct(FG,CG);
//	I = T1->Get_I();
	T2->Construct(FG,CG);
//	J = T2->Get_I();

	mout << "T1.size "<<T1->Get_I().size()
	     << "   ;  T2.size "<<T2->Get_I().size()<<endl;
    }
    virtual void multiply (Vector& f, const Vector& c) const {
	f = 0;
	int dim = f.GetMesh().dim();
//	mout << "dimminger " <<dim<<endl;
	for (int i=0; i<(T1->Get_I()).size(); ++i) {
            const vector<int>& Ii = T1->Get_I()[i];
	    int n = Ii.size();
	    if (n == 0) continue;
            double s = 1 / double(n);
	    for (int j=0; j<n; ++j) 
		for (int k=0; k<dim; ++k)
		    f(i,k) += s * c(Ii[j],k);
	}
	for (int i=0; i<T2->Get_I().size(); ++i) {
	    const vector<int>& Ji = T2->Get_I()[i];
	    int m = Ji.size();
	    if (m == 0) continue;
	    double s = 1/double(m);
	    for (int j=0; j<m; ++j) f(i,dim) += s * c(Ji[j],dim);
	    
	}
	f.ClearDirichletValues(); 
    }
    virtual void multiply_transpose (Vector& c, const Vector& f) const {
	c = 0;
	int dim = f.GetMesh().dim();
	for (int i=0; i<T1->Get_I().size(); ++i) {
            const vector<int>& Ii = T1->Get_I()[i];
	    int n = Ii.size();
	    if (n == 0) continue;
//            int fDofi = f.Dof(i);
            double s = 1 / double(n);
	    for (int j=0; j<n; ++j) 
		for (int k=0; k<dim; ++k) 
		    c(Ii[j],k) += s * f(i,k);
	}
	for (int i=0; i<T2->Get_I().size(); ++i) {
	    const vector<int>& Ji = T2->Get_I()[i];
	    int m = Ji.size();
	    if (m == 0) continue;
	    double s = 1/double(m);
	    for (int j=0; j<m; ++j) c(Ji[j],dim) += s*f(i,dim);
	}
	c.ClearDirichletValues(); 
	Collect(c);
    }
    virtual void Project (const Vector& f, Vector& c) const {
	for (row r=c.rows(); r!=c.rows_end(); ++r) {
	    row r_f = f.find_row(r());
	    if (r_f==f.rows_end()) continue;
	    for (int i=0; i<r.n(); ++i) {
		c(r,i) = f(r_f,i);
		c.D(r,i) = f.D(r_f,i);
	    }
	}
    }
};

//#ifdef DCOMPLEX
//extern 
//#endif
Point kkk;

class CurlBrickTransfer : public Transfer {
    Gradient ik;
public:
    CurlBrickTransfer () {
        ik = Gradient(iUnit * kkk);
//	Exit("CurlBrickTransfer in Transfer.C has no extern Point kkk");
    }
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {
        I.resize(FG.nR());
        const Mesh& CM = CG.GetMesh();
        for (cell c = CM.cells(); c != CM.cells_end(); ++c) {
            for (int i=0; i<c.Edges(); ++i)
                for (int j=0; j<2; ++j) {
		    int id = FG.Id(0.5 * (c.Edge(i) + c.EdgeCorner(i,j) ));
		    if (id == -1) {
			mout << "E " << c.Edge(i) << c[c.Edge(i)] << endl;
			
		    }
		    I[id].resize(1);
		    I[id][0] = CG.Id(c.Edge(i));
                }
	    Point d[3];
	    d[0] = 0.5 * (c[1] - c[0]);
	    d[1] = 0.5 * (c[2] - c[1]);
	    d[2] = 0.5 * (c[4] - c[0]);
	    
	    for (int i=0; i<3; ++i)
                for (int j=0; j<2; ++j) {
		    int id = FG.Id(c() + (j?0.5:-0.5)*d[i]);
		    if (id != -1) {
			I[id].resize(4);
			for (int k1 = 0; k1<2; ++k1)
			    for (int k2 = 0; k2<2; ++k2)
				I[id][k2+2*k1] = CG.Id(c() + (k1?1:-1) * 
						       d[(i+1)%3]
						       + (k2?1:-1) * 
						       d[(i+2)%3]);
			
		    }
		}
	    for (int i=0; i<3; ++i)
		for (int j=0; j<2; ++j) {
		    Point fc = c() + (j?1:-1) * d[i];
		    for (int ds = 1; ds<3; ++ds)
			for (int jj = 0; jj<2; ++jj) {
			    int id = FG.Id(fc + 
					   (jj?0.5:-0.5) * d[(i+ds)%3]);
			    if (id != -1) {
				I[id].resize(2);
				for (int k = 0; k<2; ++k)
				    I[id][k] = CG.Id(fc + (k?1:-1) * 
						     d[(i+3-ds)%3]);
			    }
			}
		}
	    
	}
    }
    void multiply (Vector& f, const Vector& c0) const {
	Vector c(c0);
	for (row r=c.rows(); r!=c.rows_end(); ++r)
	    c(r,0) *= exp(ik * r());
        for (int i=0; i<I.size(); ++i) {
	    const vector<int>& Ii = I[i];
            int n = Ii.size();
	    double s = 1.0 / (2*n);
            Scalar sum = 0;
	    for (int j=0; j<n; ++j) sum += s * c(Ii[j],0);
	    f(i,0) = sum;
        }
        f.ClearDirichletValues();
        for (row r=f.rows(); r!=f.rows_end(); ++r)
            f(r,0) *= conj(exp(ik * r()));
    }
    void multiply_transpose (Vector& c, const Vector& f0) const {
	Vector f(f0);
	for (row r=f.rows(); r!=f.rows_end(); ++r)
	    f(r,0) *= conj(exp(ik * r()));
	c = 0;
	for (int i=0; i<I.size(); ++i) {
	    const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    Scalar val = (1.0/(2*n)) * f(i,0);
	    for (int j=0; j<n; ++j) c(Ii[j],0) += val;
	}
	c.ClearDirichletValues();
	Collect(c);
	for (row r=c.rows(); r!=c.rows_end(); ++r)
	    c(r,0) *= exp(ik * r());
    }
    void Project (const Vector& f, Vector& c) const {
	c = 0;
	for (int i=0; i<I.size(); ++i) {
	    const vector<int>& Ii = I[i];
	    int n = Ii.size();
	    if (n != 1) continue;
		for (int j=0; j<n; ++j) c(Ii[j],0) += f(i,0);
	}
	c.ClearDirichletValues();
    }
    friend class TwoLevelCurlBrickTransfer;
};

class WeightTransfer : public Transfer {
protected:
	vector< vector<double> > C;
	Gradient ik;
public:
	void multiply (Vector& f, const Vector& c0) const {
		Vector c(c0);
		for (row r=c.rows(); r!=c.rows_end(); ++r)
			c(r,0) *= exp(ik * r());
		for (int i=0; i<I.size(); ++i) {
			const vector<int>& Ii = I[i];
			const vector<double>& Ci = C[i];
			int n = Ii.size();
			Scalar s = 0;
			for (int j=0; j<n; ++j) s += Ci[j] * c(Ii[j],0);
			f(i,0) = s;
		}
		f.ClearDirichletValues();
		for (row r=f.rows(); r!=f.rows_end(); ++r)
			f(r,0) *= conj(exp(ik * r()));
	}
	void multiply_transpose (Vector& c, const Vector& f0) const {
		Vector f(f0);
		for (row r=f.rows(); r!=f.rows_end(); ++r)
			f(r,0) *= conj(exp(ik * r()));
		c = 0;
		for (int i=0; i<I.size(); ++i) {
			const vector<int>& Ii = I[i];
			const vector<double>& Ci = C[i];
			int n = Ii.size();
			Scalar s = f(i,0);
			for (int j=0; j<n; ++j) c(Ii[j],0) += Ci[j] * s;
		}
		c.ClearDirichletValues(); 
		Collect(c);
		for (row r=c.rows(); r!=c.rows_end(); ++r)
			c(r,0) *= exp(ik * r());		
	}
	void Project (const Vector& f, Vector& c) const {
		c = 0;
		for (int i=0; i<I.size(); ++i) {
			const vector<int>& Ii = I[i];
			int n = Ii.size();
			if (n != 1) continue;
			for (int j=0; j<n; ++j) c(Ii[j],0) += f(i,0);
		}
		c.ClearDirichletValues();
	}			
};

class CurlTetTransfer : public WeightTransfer {
    vector<Point> ver,gflam;

   double flam(int n,const Point& x) const {
      switch(n) {
         case 0: return 1-x[0]-x[1]-x[2];
         case 1: return x[0];
         case 2: return x[1];
         case 3: return x[2];
      }
   }
   Point bas(int n1,int n2,const Point& x) const {
      return flam(n1,x)*gflam[n2]-flam(n2,x)*gflam[n1];
   }
   double int_bas(int n1,int n2,Point x1,Point x2) const {
      if (ver[n1]>ver[n2]) swap(n1,n2);
      if (x1>x2) swap(x1,x2);
      Point med=0.5*(x1+x2);
      return (x2-x1)*(bas(n1,n2,x1)+4*bas(n1,n2,med)+bas(n1,n2,x2))/6;
   }
   void apply_to_cell(int p11, int p12, int p21, int p22, const cell& c, const matrixgraph& FG, const matrixgraph& CG) {
            Point fp1 = 0.5*(c[p11]+c[p12]);
            Point fp2 = 0.5*(c[p21]+c[p22]);
            Point fdof = 0.5*(fp1+fp2);
            Point rfp1 = 0.5*(ver[p11]+ver[p12]);
            Point rfp2 = 0.5*(ver[p21]+ver[p22]);
	    int id = FG.Id(fdof);
	    if (id == -1) {
		    mout << "can't find id " << fdof << endl;
	    }
	    vector<int> ids;
	    vector<double> coef;
	    for(int ec1=0;ec1<4;ec1++)
		    for(int ec2=ec1+1;ec2<4;ec2++) {
                            // coarse edge
                            double w = int_bas(ec1,ec2,rfp1,rfp2);
			    if (abs(w)<1.0e-7) continue;
			    Point cdof = 0.5*(c[ec1]+c[ec2]);
			    int cid = CG.Id(cdof);
			    if (cid == -1) {
				    mout << "can't find cid " << cdof << endl;
			    }
                            double sign = ((((c[ec1]<c[ec2])==(ver[ec1]<ver[ec2]))==((fp1<fp2)==(rfp1<rfp2)))*2-1);
			    ids.push_back(cid);
                            coef.push_back(w*sign);
		    }
	    I[id] = ids;
            C[id] = coef;
   }
public:
    CurlTetTransfer () : ver(4), gflam(4) {
        ik = Gradient(iUnit * kkk);
    }
    void Construct (const matrixgraph& FG, const matrixgraph& CG) {
        ver[0]=Point(0,0,0);
        ver[1]=Point(1,0,0);
        ver[2]=Point(0,1,0);
        ver[3]=Point(0,0,1);
        gflam=ver;
        gflam[0]=Point(-1,-1,-1);
        I.resize(FG.nR());
        C.resize(I.size());
        const Mesh& CM = CG.GetMesh();
        for (cell c = CM.cells(); c != CM.cells_end(); ++c) {
            // DOFs on edges
            for (int i=0; i<c.Edges(); ++i)
                for (int j=0; j<2; ++j) {
		    int id = FG.Id(0.5 * (c.Edge(i) + c.EdgeCorner(i,j) ) );
		    if (id == -1) {
			mout << "E " << c.Edge(i) << c[c.Edge(i)] << endl;
		    }
		    I[id] = vector<int>(1,CG.Id(c.Edge(i)));
		    C[id] = vector<double>(1,0.5);
                }
	    // DOFs on faces
            int fv[3];
            int n1=0;
            for(fv[0]=0; fv[0]<4; fv[0]++)
               for(fv[1]=fv[0]+1; fv[1]<4; fv[1]++)
                  for(fv[2]=fv[1]+1; fv[2]<4; fv[2]++)
                     // face
                     for(int j=0; j<3; j++)
                        // new fine edge
			apply_to_cell(fv[j],fv[(j+1)%3],fv[j],fv[(j+2)%3],c,FG,CG);
	    // internal DOF
	    Point fdof=c();
	    apply_to_cell(1,2,0,3,c,FG,CG);
        }
   }
};

class TwoLevelCurlBrickTransfer : public WeightTransfer {
 public:
 TwoLevelCurlBrickTransfer() {};
 void Construct(const MatrixGraphs& G, int FL, int CL) {
    if (FL<=CL) {
       mout << "error in TwoLevelCurlBrickTransfer constructor\n";
       return;
    }
    CurlBrickTransfer tr;
    tr.Construct(G[FL],G[FL-1]);
    I = tr.I;
    C.resize(I.size() );
    for (int i = 0; i < I.size(); ++i) {
       int n = I[i].size();
       double s = 0.5/n;
       C[i].resize(n);
       for (int j = 0; j < n; ++j) C[i][j] = s;
    };
    for (int k = FL-1; k > CL; --k) {
      tr.Construct(G[k],G[k-1]);
      for (int i = 0; i < I.size(); ++i) {
         vector<int>& rI = I[i];
         vector<double>& rC = C[i];
         map<int, double> nIC;
         map<int, double>::iterator nICit;
         int Len = 0;
         for (int j = 0; j < rI.size(); ++j) {
            vector<int>& rtI = tr.I[ rI[j] ];
            double s = 0.5 * rC[j] / rtI.size();
            for (int l = 0; l < rtI.size(); ++l) {
               nICit = nIC.find( rtI[l] );
               if ( nICit == nIC.end() ) {
                  Len++;
                  nIC[ rtI[l] ] = s;
               }
               else nICit->second += s;
            }
         }
         rI.resize(Len);
         rC.resize(Len);
         int j = 0;
         for (nICit = nIC.begin(); nICit != nIC.end(); ++nICit) {
            rI[j] = nICit->first;
            rC[j] = nICit->second;
            ++j;
         }
      }
    }
 };
};

Transfer* GetTransfer (const string& name) { 
//    if (name == "curl")  return new CurlBrickTransfer(); 
//    if (name == "linear")  return new LinearTransfer(); 
   if (name=="EdgeDoF") return new CurlBrickTransfer();
   else if (name=="TetCurl") return new CurlTetTransfer();
   else if (name=="VertexDoF") return new LinearTransfer();
   else if (name=="Serendipity") return new SerendipityTransfer();
   else if (name=="Serendipity2D") return new Serendipity2DTransfer();
   else if (name=="Quadratic") return new QuadraticTransfer();
   else if (name=="TaylorHoodQuadratic") return new TaylorHoodQuadraticTransfer();
   else if (name=="Cosserat") return new CosseratTransfer();
   else if (name=="Cosserat2D") return new CosseratTransfer(2);
   Exit("no transfer "+name+" implemented");
}
string GetTransferName () {
    string pc = "VertexDoF"; ReadConfig(Settings,"Transfer",pc);
    return pc;
}
Transfer* GetTransfer () { return GetTransfer(GetTransferName()); }
