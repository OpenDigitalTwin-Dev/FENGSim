// file:    Cell.C
// author:  Christian Wieners
// $Header: /public/M++/src/Cell.C,v 1.25 2009-04-03 14:42:18 sauter Exp $

#include "Cell.h" 
#include "Debug.h" 
#include "IO.h" 

// ----------------------------------------------------
//    TRIANGLE
// ----------------------------------------------------
const Point LocalCornersTri[] = 
{ Point(0.0,0.0,0.0), Point(1.0,0.0,0.0), Point(0.0,1.0,0.0) }; 
const Point* LocalEdgesTri;
const Point* LocalFaceNormalsTri;
const Point* LocalCenterTri;
class Triangle;
const Triangle* ReferenceTriangle;
const Rules* Rtri;
class Triangle : public Cell {
    void CheckOrientation (vector<Point>& x) {
		Point v = x[1] - x[0];
		Point w = x[2] - x[0];
		if (v[0]*w[1] - v[1]*w[0] < 0) std::swap(x[1],x[2]);
    }
public:
    Triangle (const vector<Point>& z, int sd) : Cell(z,sd) {
		CheckOrientation(*this); }
    Point operator () () const { return (1.0/3.0)*(z(0)+z(1)+z(2)); }
    CELLTYPE Type () const { return TRIANGLE; }
    CELLTYPE ReferenceType () const { return TRIANGLE; }
    int Corners () const { return 3; }
    const Point& LocalCorner (int i) const { return LocalCornersTri[i]; }
    const Point& LocalEdge (int i) const { return LocalEdgesTri[i]; }
    const Point& LocalFace (int i) const { return LocalEdgesTri[i]; }
    const Point& LocalFaceNormal (int i) const {return LocalFaceNormalsTri[i];}
	const Point& LocalCenter () const { return *LocalCenterTri; }
    Point LocalToGlobal (const Point& z) const { return P1_tri.Eval(z,*this); }
    Transformation GetTransformation (const Point& x) const {
		vector<Point> y(2);
		P1_tri.EvalGradient(y,x,*this);
//	mout <<" y" <<endl << y<<endl;
//	vector<Point> yy(2);
//	yy[0] = z(1) - z(0);
//	yy[1] = z(2) - z(0);
//	mout <<" yy" <<endl << yy<<endl;
		return Transformation(y); 
    }
// this does not seem to work
/*
  Transformation GetTransformation (const Point& x) const {
  vector<Point> y(2);
  y[0] = z(1) - z(0);
  y[1] = z(2) - z(0);
  return Transformation(y); 
  }
*/
    int Edges () const { return 3; }
    Point Edge (int i) const {
		switch (i) {
		case 0: return 0.5 * (z(0) + z(1));
		case 1: return 0.5 * (z(1) + z(2));
		case 2: return 0.5 * (z(2) + z(0));
		}
    }
    const Point& EdgeCorner (int i, int j) const { 
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return z(0);
			case 1: return z(1);
			}
		case 1:
			switch (j) {
			case 0: return z(1);
			case 1: return z(2);
			}
		case 2:
			switch (j) {
			case 0: return z(2);
			case 1: return z(0);
			}
		}
    }
    short edgecorner (int i, int j) const { 
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return 0;
			case 1: return 1;
			}
		case 1:
			switch (j) {
			case 0: return 1;
			case 1: return 2;
			}
		case 2:
			switch (j) {
			case 0: return 2;
			case 1: return 0;
			}
		}
    }
    int Faces () const { return 3; }
    Point Face (int i) const { 
		switch (i) {
		case 0: return 0.5 * (z(0) + z(1));
		case 1: return 0.5 * (z(1) + z(2));
		case 2: return 0.5 * (z(2) + z(0));
		}
    }
    short FaceCorners (int i) const { return 2; }
    const Point& FaceCorner (int i, int j) const {
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return z(0);
			case 1: return z(1);
			}
		case 1:
			switch (j) {
			case 0: return z(1);
			case 1: return z(2);
			}
		case 2:
			switch (j) {
			case 0: return z(2);
			case 1: return z(0);
			}
		}
    }
    short facecorner (int i, int j) const {
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return 0;
			case 1: return 1;
			}
		case 1:
			switch (j) {
			case 0: return 1;
			case 1: return 2;
			}
		case 2:
			switch (j) {
			case 0: return 2;
			case 1: return 0;
			}
		}
    }
    short FaceEdges (int i) const { return 1; }
    short faceedge (int i, int j) const { return i; }
    double LocalFaceArea (int i) const { 
		switch (i) {
		case 0: return 1; 
		case 1: return sqrt(2.0); 
		case 2: return 1; 
		}
    }
    const Rules& Refine (vector<Point>& z) const { 
		z.resize(6);
		int n = 0; 
		for (int i=0; i<Corners(); ++i) z[n++] = Corner(i);
		for (int i=0; i<Edges(); ++i)   z[n++] = Edge(i);
		return *Rtri; 
    }
    int dim () const { return 2; }
    bool plane () const { return true; }
    const Cell* ReferenceCell () const { return ReferenceTriangle; }
};

// ----------------------------------------------------
//    QUADRILATERAL
// ----------------------------------------------------
const Point LocalCornersQuad[] = { Point(0.0,0.0,0.0), Point(1.0,0.0,0.0), 
								   Point(1.0,1.0,0.0), Point(0.0,1.0,0.0) }; 
const Point* LocalEdgesQuad;
const Point* LocalFaceNormalsQuad;
const Point* LocalCenterQuad;
class Quadrilateral;
const Quadrilateral* ReferenceQuadrilateral;
const Rules* Rquad;
class Quadrilateral : public Cell {
    void CheckOrientation (vector<Point>& x) {
		if (x.size() < 8) return;
		int n = 4; 
		for (int i=0; i<Edges(); ++i) x[n++] = Edge(i);
    }
public:
    Quadrilateral (const vector<Point>& z, int sd) : Cell(z,sd) {
		CheckOrientation(*this); }
    Point operator () () const { return 0.25*(z(0)+z(1)+z(2)+z(3)); }
    CELLTYPE Type () const { return QUADRILATERAL; }
    CELLTYPE ReferenceType () const { return QUADRILATERAL; }
    int Corners () const { return 4; }
    const Point& LocalCorner (int i) const { return LocalCornersQuad[i]; }
    const Point& LocalEdge (int i) const { return LocalEdgesQuad[i]; }
    const Point& LocalFace (int i) const { return LocalEdgesQuad[i]; }
    const Point& LocalFaceNormal (int i) const{return LocalFaceNormalsQuad[i];}
    const Point& LocalCenter () const { return *LocalCenterQuad; }
    Point LocalToGlobal (const Point& z) const { return P1_quad.Eval(z,*this);}
    Transformation GetTransformation (const Point& x) const {
		vector<Point> y(2);
		P1_quad.EvalGradient(y,x,*this);
		return Transformation(y); 
    }
    int Edges () const { return 4; }
    Point Edge (int i) const {
		switch (i) {
		case 0: return 0.5 * (z(0) + z(1));
		case 1: return 0.5 * (z(1) + z(2));
		case 2: return 0.5 * (z(2) + z(3));
		case 3: return 0.5 * (z(3) + z(0));
		}
    }
    const Point& EdgeCorner (int i, int j) const { 
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return z(0);
			case 1: return z(1);
			}
		case 1:
			switch (j) {
			case 0: return z(1);
			case 1: return z(2);
			}
		case 2:
			switch (j) {
			case 0: return z(2);
			case 1: return z(3);
			}
		case 3:
			switch (j) {
			case 0: return z(3);
			case 1: return z(0);
			}
		}
    }
    short edgecorner (int i, int j) const { 
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return 0;
			case 1: return 1;
			}
		case 1:
			switch (j) {
			case 0: return 1;
			case 1: return 2;
			}
		case 2:
			switch (j) {
			case 0: return 2;
			case 1: return 3;
			}
		case 3:
			switch (j) {
			case 0: return 3;
	    case 1: return 0;
			}
		}
    }
    int Faces () const { return 4; }
    Point Face (int i) const { 
		switch (i) {
		case 0: return 0.5 * (z(0) + z(1));
		case 1: return 0.5 * (z(1) + z(2));
		case 2: return 0.5 * (z(2) + z(3));
		case 3: return 0.5 * (z(3) + z(0));
		}
    }
    short FaceCorners (int i) const { return 2; }
    const Point& FaceCorner (int i, int j) const {
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return z(0);
			case 1: return z(1);
			}
		case 1:
			switch (j) {
			case 0: return z(1);
			case 1: return z(2);
			}
		case 2:
			switch (j) {
			case 0: return z(2);
			case 1: return z(3);
			}
		case 3:
			switch (j) {
			case 0: return z(3);
			case 1: return z(0);
			}
		}
    }
    short facecorner (int i, int j) const {
		switch (i) {
		case 0: 
			switch (j) {
			case 0: return 0;
			case 1: return 1;
			}
		case 1:
			switch (j) {
			case 0: return 1;
			case 1: return 2;
			}
		case 2:
			switch (j) {
			case 0: return 2;
			case 1: return 3;
			}
		case 3:
			switch (j) {
			case 0: return 3;
			case 1: return 0;
			}
		}
    }
    short FaceEdges (int i) const { return 1; }
    short faceedge (int i, int j) const { return i; }
    double LocalFaceArea (int i) const { return 1; }
    virtual const Rules& Refine (vector<Point>& z) const { 
		z.resize(9);
		int n = 0; 
		for (int i=0; i<Corners(); ++i) z[n++] = Corner(i);
		for (int i=0; i<Edges(); ++i)   z[n++] = Edge(i);
		z[n] = Center();
		return *Rquad; 
    }
    int dim () const { return 2; }
    bool plane () const { return true; }
    const Cell* ReferenceCell () const { return ReferenceQuadrilateral; }
};

// ----------------------------------------------------
//    QUADRILATERAL2
// ----------------------------------------------------
class Quadrilateral2;
const Quadrilateral2* ReferenceQuadrilateral2;
const Rules* R2quad;
class Quadrilateral2 : public Quadrilateral {
public:
    Quadrilateral2 (const vector<Point>& z, int sd) : Quadrilateral(z,sd) {}
    Point LocalToGlobal (const Point& z) const { 
		return P2_quadSerendipity.Eval(z,*this); }
    Transformation GetTransformation (const Point& x) const {
		vector<Point> y(2);
		P2_quadSerendipity.EvalGradient(y,x,*this);
		return Transformation(y); 
    }
    CELLTYPE Type () const { return QUADRILATERAL2; }
    const Rules& Refine (vector<Point>& z) const { 
		z.resize(21);
		int n = 0; 
		for (int i=0; i<Corners(); ++i) z[n++] = Corner(i);
		for (int i=0; i<Edges(); ++i)   z[n++] = LocalToGlobal(LocalEdge(i)); 
		z[n++] = LocalToGlobal(LocalCenter());
		for (int i=0; i<Edges(); ++i) {
			z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,0))
										+LocalEdge(i)));
			z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,1))
										+LocalEdge(i))); 
			z[n++] = LocalToGlobal(0.5*(LocalCenter()+LocalEdge(i))); 
		}
		return *R2quad; 
    }
};

// ----------------------------------------------------
//    TETRAHEDRON
// ----------------------------------------------------
const Point LocalCornersTet[] = 
    { Point(0.0,0.0,0.0), Point(1.0,0.0,0.0), 
      Point(0.0,1.0,0.0), Point(0.0,0.0,1.0) }; 
const Point* LocalEdgesTet;
const Point* LocalFacesTet;
const Point* LocalFaceNormalsTet;
const Point* LocalCenterTet;
class Tetrahedron;
const Tetrahedron* ReferenceTetrahedron;
const Rules* Rtet0;
const Rules* Rtet1;
const Rules* Rtet2;
class Tetrahedron : public Cell {
    void CheckOrientation (vector<Point>& x) {
	Point u = x[1] - x[0];
	Point v = x[2] - x[0];
	Point w = x[3] - x[0];
	if (det(u,v,w) < 0) std::swap(x[1],x[2]);
    }
public:
    Tetrahedron (const vector<Point>& z, int sd) : Cell(z,sd) {
	CheckOrientation(*this); }
    Point operator () () const { return 0.25*(z(0)+z(1)+z(2)+z(3)); }
    CELLTYPE Type () const { return TETRAHEDRON; }
    CELLTYPE ReferenceType () const { return TETRAHEDRON; }
    int Corners () const { return 4; }
    const Point& LocalCorner (int i) const { return LocalCornersTet[i]; }
    const Point& LocalEdge (int i) const { return LocalEdgesTet[i]; }
    const Point& LocalFace (int i) const { return LocalFacesTet[i]; }
    const Point& LocalFaceNormal (int i) const {return LocalFaceNormalsTet[i];}
    const Point& LocalCenter () const { return *LocalCenterTet; }
    Point LocalToGlobal (const Point& z) const { return P1_tet.Eval(z,*this); }

    Transformation GetTransformation (const Point& x) const {
	vector<Point> y(3);
	P1_tet.EvalGradient(y,x,*this);
	return Transformation(y); 
    }

// It is questionable whether this transformation works since in 2d, 
// the equivalent transformation for triangles does not seem to work. 
// However, further debugging is necessary...
/*
    Transformation GetTransformation (const Point& x) const {
	vector<Point> y(3);
	y[0] = z(1) - z(0);
	y[1] = z(2) - z(0);
	y[2] = z(3) - z(0);
	double dett = det(y[0],y[1],y[2]);
	if (dett < 0) { 
	    std::cout<<" ; "<<dett<<" ->SWAP  ;"; std::swap(y[2],y[1]); 
	}

	mout << "T0 " << endl << Transformation(y); 

	P1_tet.EvalGradient(y,x,*this);

	mout << "T1 " << endl << Transformation(y); 


	return Transformation(y); 
    }

*/

    int Edges () const { return 6; }
    Point Edge (int i) const {
	switch (i) {
	case 0: return 0.5 * (z(0) + z(1));
	case 1: return 0.5 * (z(1) + z(2));
	case 2: return 0.5 * (z(0) + z(2));
	case 3: return 0.5 * (z(0) + z(3));
	case 4: return 0.5 * (z(1) + z(3));
	case 5: return 0.5 * (z(2) + z(3));
	}
    }
    const Point& EdgeCorner (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(1);
	    }
	case 1:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(2);
	    }
	case 2:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(2);
	    }
	case 3:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(3);
	    }
	case 4:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(3);
	    }
	case 5:
	    switch (j) {
	    case 0: return z(2);
	    case 1: return z(3);
	    }
	}
    }
    short edgecorner (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 0;
	    case 1: return 1;
	    }
	case 1:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 2;
	    }
	case 2:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 2;
	    }
	case 3:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 3;
	    }
	case 4:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 3;
	    }
	case 5:
	    switch (j) {
	    case 0: return 2;
	    case 1: return 3;
	    }
	}
    }
    int Faces () const { return 4; }
    Point Face (int i) const { 
	switch (i) {
	case 0: return (1.0/3.0) * (z(0) + z(2) + z(1));
	case 1: return (1.0/3.0) * (z(1) + z(2) + z(3));
	case 2: return (1.0/3.0) * (z(0) + z(3) + z(2));
	case 3: return (1.0/3.0) * (z(0) + z(1) + z(3));
	}
    }
    short FaceCorners (int i) const { return 3; }
    const Point& FaceCorner (int i, int j) const {
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(2);
	    case 2: return z(1);
	    }
	case 1:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(2);
	    case 2: return z(3);
	    }
	case 2:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(3);
	    case 2: return z(2);
	    }
	case 3:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(1);
	    case 2: return z(3);
	    }
	}
    }
    short facecorner (int i, int j) const {
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 0;
	    case 1: return 2;
	    case 2: return 1;
	    }
	case 1:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 2;
	    case 2: return 3;
	    }
	case 2:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 3;
	    case 2: return 2;
	    }
	case 3:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 1;
	    case 2: return 3;
	    }
	}
    }
    short FaceEdges (int i) const { return 3; }
    short faceedge (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 2;
	    case 1: return 1;
	    case 2: return 0;
	    }
	case 1:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 5;
	    case 2: return 4;
	    }
	case 2:
	    switch (j) {
	    case 0: return 3;
	    case 1: return 5;
	    case 2: return 2;
	    }
	case 3:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 4;
	    case 2: return 3;
	    }
	}
    }
    double LocalFaceArea (int i) const { 
	switch (i) {
	case 0: return 0.5; 
	case 1: return sqrt(0.75); 
	case 2: return 0.5; 
	case 3: return 0.5; 
	}
    }
    const Rules& Refine (vector<Point>& z) const { 
	z.resize(10);
	int n = 0; 
	for (int i=0; i<Corners(); ++i) z[n++] = Corner(i);
	for (int i=0; i<Edges(); ++i) z[n++] = Edge(i);

	double t0 = dist(z[6],z[8]);
	double t1 = dist(z[5],z[7]);
	double t2 = dist(z[4],z[9]);
	if (t0<=t1&&t0<=t2) return *Rtet0;
	if (t1<=t0&&t1<=t2) return *Rtet1;
	if (t2<=t1&&t2<=t0) return *Rtet2;
    }
    int dim () const { return 3; }
    bool plane () const { return false; }
    const Cell* ReferenceCell () const { return ReferenceTetrahedron; }
};

// ----------------------------------------------------
//    HEXAHEDRON
// ----------------------------------------------------
const Point LocalCornersHex[8] = 
    { Point(0.0,0.0,0.0), Point(1.0,0.0,0.0), 
      Point(1.0,1.0,0.0), Point(0.0,1.0,0.0), 
      Point(0.0,0.0,1.0), Point(1.0,0.0,1.0), 
      Point(1.0,1.0,1.0), Point(0.0,1.0,1.0) }; 
const Point* LocalEdgesHex;
const Point* LocalFacesHex;
const Point* LocalFaceNormalsHex;
const Point* LocalCenterHex;
class Hexahedron;
const Hexahedron* ReferenceHexahedron;
const Rules* Rhex;
class Hexahedron : public Cell {
    void CheckOrientation (vector<Point>& x) {
	Point u = x[1] - x[0];
	Point v = x[3] - x[0];
	Point w = x[4] - x[0];
//	cout <<x<<endl;
	if (det(u,v,w) < 0) {
	    std::swap(x[0],x[4]);
	    std::swap(x[1],x[5]);
	    std::swap(x[2],x[6]);
	    std::swap(x[3],x[7]);
	    u = x[1] - x[0];
	    v = x[3] - x[0];
	    w = x[4] - x[0];
	}
	if (det(u,v,w) < 0) {
	    cout << "Error in Hex-orientation; file: Cell.C\n";
	    exit(1);
	}
	if (x.size() < 20) return;
	int n = 8; 
	for (int i=0; i<Edges(); ++i) x[n++] = Edge(i);
	if (x.size() < 26) return;
	for (int i=0; i<Faces(); ++i) x[n++] = Face(i);
	if (x.size() < 27) return;
	x[n++] = Center();
    }
public:
    Hexahedron (const vector<Point>& z, int sd) : Cell(z,sd) {
	CheckOrientation(*this); }
    Point operator () () const { 
	return 0.125*(z(0)+z(1)+z(2)+z(3)+z(4)+z(5)+z(6)+z(7)); 
    }
    virtual CELLTYPE Type () const { return HEXAHEDRON; }
    CELLTYPE ReferenceType () const { return HEXAHEDRON; }
    int Corners () const { return 8; }
    const Point& LocalCorner (int i) const { return LocalCornersHex[i]; }
    const Point& LocalEdge (int i) const { return LocalEdgesHex[i]; }
    const Point& LocalFace (int i) const { return LocalFacesHex[i]; }
    const Point& LocalFaceNormal (int i) const {return LocalFaceNormalsHex[i];}
    double LocalFaceArea (int i) const { return 1; }
    const Point& LocalCenter () const { return *LocalCenterHex; }
    Point LocalToGlobal (const Point& z) const { return P1_hex.Eval(z,*this); }
    Transformation GetTransformation (const Point& x) const {
	vector<Point> y(3);
	P1_hex.EvalGradient(y,x,*this);
	return Transformation(y); 
    }
    int Edges () const { return 12; }
    Point Edge (int i) const {
	switch (i) {
	case 0: return 0.5 * (z(0) + z(1));
	case 1: return 0.5 * (z(1) + z(2));
	case 2: return 0.5 * (z(2) + z(3));
	case 3: return 0.5 * (z(3) + z(0));
	case 4: return 0.5 * (z(0) + z(4));
	case 5: return 0.5 * (z(1) + z(5));
	case 6: return 0.5 * (z(2) + z(6));
	case 7: return 0.5 * (z(3) + z(7));
	case 8: return 0.5 * (z(4) + z(5));
	case 9: return 0.5 * (z(5) + z(6));
	case 10: return 0.5 * (z(6) + z(7));
	case 11: return 0.5 * (z(7) + z(4));
	}
    }
    const Point& EdgeCorner (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(1);
	    }
	case 1:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(2);
	    }
	case 2:
	    switch (j) {
	    case 0: return z(2);
	    case 1: return z(3);
	    }
	case 3:
	    switch (j) {
	    case 0: return z(3);
	    case 1: return z(0);
	    }
	case 4:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(4);
	    }
	case 5:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(5);
	    }
	case 6:
	    switch (j) {
	    case 0: return z(2);
	    case 1: return z(6);
	    }
	case 7:
	    switch (j) {
	    case 0: return z(3);
	    case 1: return z(7);
	    }
	case 8:
	    switch (j) {
	    case 0: return z(4);
	    case 1: return z(5);
	    }
	case 9:
	    switch (j) {
	    case 0: return z(5);
	    case 1: return z(6);
	    }
	case 10:
	    switch (j) {
	    case 0: return z(6);
	    case 1: return z(7);
	    }
	case 11:
	    switch (j) {
	    case 0: return z(7);
	    case 1: return z(4);
	    }

	}
    }
    short edgecorner (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 0;
	    case 1: return 1;
	    }
	case 1:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 2;
	    }
	case 2:
	    switch (j) {
	    case 0: return 2;
	    case 1: return 3;
	    }
	case 3:
	    switch (j) {
	    case 0: return 3;
	    case 1: return 0;
	    }
	case 4:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 4;
	    }
	case 5:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 5;
	    }
	case 6:
	    switch (j) {
	    case 0: return 2;
	    case 1: return 6;
	    }
	case 7:
	    switch (j) {
	    case 0: return 3;
	    case 1: return 7;
	    }
	case 8:
	    switch (j) {
	    case 0: return 4;
	    case 1: return 5;
	    }
	case 9:
	    switch (j) {
	    case 0: return 5;
	    case 1: return 6;
	    }
	case 10:
	    switch (j) {
	    case 0: return 6;
	    case 1: return 7;
	    }
	case 11:
	    switch (j) {
	    case 0: return 7;
	    case 1: return 4;
	    }
	}
    }
    int Faces () const { return 6; }
    Point Face (int i) const { 
	switch (i) {
	case 0: return 0.25 * (z(0) + z(1) + z(2) + z(3));
	case 1: return 0.25 * (z(0) + z(1) + z(5) + z(4));
	case 2: return 0.25 * (z(1) + z(2) + z(5) + z(6));
	case 3: return 0.25 * (z(2) + z(3) + z(6) + z(7));
	case 4: return 0.25 * (z(0) + z(3) + z(4) + z(7));
	case 5: return 0.25 * (z(4) + z(5) + z(6) + z(7));
	}
    }
    short FarFace (int i) const { 
	switch (i) {
	case 0: return 5;
	case 1: return 3;
	case 2: return 4;
	case 3: return 1;
	case 4: return 2;
	case 5: return 0;
	}
    }
    short MiddleEdge (int i,int j) const { 
	if (i==5) i = 0;
	else if (i==3) i = 1;
	else if (i==4) i = 2;
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 4;
	    case 1: return 5;
	    case 2: return 6;
	    case 3: return 7;
	    }
	case 1:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 3;
	    case 2: return 9;
	    case 3: return 11;
	    }
	case 2:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 2;
	    case 2: return 8;
	    case 3: return 10;
	    }
	}
    }
    short FaceCorners (int i) const { return 4; }
    const Point& FaceCorner (int i, int j) const {
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(3);
	    case 2: return z(2);
	    case 3: return z(1);
	    }
	case 1:
	    switch (j) {
	    case 0: return z(0);
	    case 1: return z(1);
	    case 2: return z(5);
	    case 3: return z(4);
	    }
	case 2:
	    switch (j) {
	    case 0: return z(1);
	    case 1: return z(2);
	    case 2: return z(6);
	    case 3: return z(5);
	    }
	case 3:
	    switch (j) {
	    case 0: return z(2);
	    case 1: return z(3);
	    case 2: return z(7);
	    case 3: return z(6);
	    }
	case 4:
	    switch (j) {
	    case 0: return z(3);
	    case 1: return z(0);
	    case 2: return z(4);
	    case 3: return z(7);
	    }
	case 5:
	    switch (j) {
	    case 0: return z(4);
	    case 1: return z(5);
	    case 2: return z(6);
	    case 3: return z(7);
	    }
	}
    }
    short facecorner (int i, int j) const {
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 0;
	    case 1: return 3;
	    case 2: return 2;
	    case 3: return 1;
	    }
	case 1:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 1;
	    case 2: return 5;
	    case 3: return 4;
	    }
	case 2:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 2;
	    case 2: return 6;
	    case 3: return 5;
	    }
	case 3:
	    switch (j) {
	    case 0: return 2;
	    case 1: return 3;
	    case 2: return 7;
	    case 3: return 6;
	    }
	case 4:
	    switch (j) {
	    case 0: return 3;
	    case 1: return 0;
	    case 2: return 4;
	    case 3: return 7;
	    }
	case 5:
	    switch (j) {
	    case 0: return 4;
	    case 1: return 5;
	    case 2: return 6;
	    case 3: return 7;
	    }
	}
    }
    short FaceEdges (int i) const { return 4; }
    bool faceedgecircuit (int i) const { 
	switch (i) {
	case 0: 
	    return true;
	case 1:
	    return false;
	case 2:
	    return false;
	case 3:
	    return false;
	case 4:
	    return false;
	case 5:
	    return true;
	}
    }
    short faceedge (int i, int j) const { 
	switch (i) {
	case 0: 
	    switch (j) {
	    case 0: return 0;
	    case 1: return 1;
	    case 2: return 2;
	    case 3: return 3;
	    }
	case 1:
	    switch (j) {
	    case 0: return 0;
	    case 1: return 4;
	    case 2: return 5;
	    case 3: return 8;
	    }
	case 2:
	    switch (j) {
	    case 0: return 1;
	    case 1: return 5;
	    case 2: return 6;
	    case 3: return 9;
	    }
	case 3:
	    switch (j) {
	    case 0: return 2;
	    case 1: return 6;
	    case 2: return 7;
	    case 3: return 10;
	    }
	case 4:
	    switch (j) {
	    case 0: return 3;
	    case 1: return 7;
	    case 2: return 4;
	    case 3: return 11;
	    }
	case 5:
	    switch (j) {
	    case 0: return 8;
	    case 1: return 9;
	    case 2: return 10;
	    case 3: return 11;
	    }
	}
    }
    virtual const Rules& Refine (vector<Point>& z) const { 
	z.resize(27);
	int n = 0; 
	for (int i=0; i<Corners(); ++i) z[n++] = Corner(i);
	for (int i=0; i<Edges(); ++i)   z[n++] = Edge(i);
	for (int i=0; i<Faces(); ++i)   z[n++] = Face(i);
	z[n++] = Center();
	return *Rhex; 
    }
    int dim () const { return 3; }
    bool plane () const { return false; }
    const Cell* ReferenceCell () const { return ReferenceHexahedron; }
};

// ----------------------------------------------------
//    HEXAHEDRON20
// ----------------------------------------------------
class Hexahedron20;
const Hexahedron20* ReferenceHexahedron20;
const Rules* R20hex;
class Hexahedron20 : public Hexahedron {
public:
    Hexahedron20 (const vector<Point>& z, int sd) : Hexahedron(z,sd) {}
    Point LocalToGlobal (const Point& z) const { 
	return P2_hexSerendipity.Eval(z,*this); 
    }
    Transformation GetTransformation (const Point& x) const {
	vector<Point> y(3);
	P2_hexSerendipity.EvalGradient(y,x,*this);
	return Transformation(y); 
    }
    virtual CELLTYPE Type () const { return HEXAHEDRON20; }
    virtual const Rules& Refine (vector<Point>& z) const { 
	z.resize(81);
	int n = 0; 
	for (int i=0; i<Corners(); ++i) { z[n++]= Corner(i); }                 
	for (int i=0; i<Edges(); ++i)   { z[n++]= LocalToGlobal(LocalEdge(i));}
	for (int i=0; i<Faces(); ++i)   { z[n++]= LocalToGlobal(LocalFace(i));}
	z[n++] = LocalToGlobal(LocalCenter());                                 
	for (int i=0; i<Edges(); ++i) {                                   
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,0))
					+LocalEdge(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,1))
					+LocalEdge(i)));                       
	}
	for (int i=0; i<Faces(); ++i) {                                   
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,0))
					+LocalFace(i)));
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,1))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,2))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,3))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalCenter()+LocalFace(i)));          
	}
	return *R20hex; 
    }
};

// ----------------------------------------------------
//    HEXAHEDRON27
// ----------------------------------------------------
class Hexahedron27;
const Hexahedron27* ReferenceHexahedron27;
const Rules* R27hex;
class Hexahedron27 : public Hexahedron {
public:
    Hexahedron27 (const vector<Point>& z, int sd) : Hexahedron(z,sd) {}
    Point LocalToGlobal (const Point& z) const { 
	return P2_hex.Eval(z,*this); }
    Transformation GetTransformation (const Point& x) const {
	vector<Point> y(3);
	P2_hex.EvalGradient(y,x,*this);
	return Transformation(y); 
    }
    virtual CELLTYPE Type () const { return HEXAHEDRON27; }
    virtual const Rules& Refine (vector<Point>& z) const { 
	z.resize(125);
	int n = 0; 
	for (int i=0; i<Corners(); ++i) { z[n++]= Corner(i); }            
	for (int i=0; i<Edges(); ++i)   { z[n++]= LocalToGlobal(LocalEdge(i));}
	for (int i=0; i<Faces(); ++i)   { z[n++]= LocalToGlobal(LocalFace(i));}
	z[n++] = LocalToGlobal(LocalCenter());                        
	for (int i=0; i<Edges(); ++i) {                                   
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,0))
					+LocalEdge(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(edgecorner(i,1))
					+LocalEdge(i))); 
	}
	for (int i=0; i<Faces(); ++i) {                                   
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,0))
					+LocalFace(i)));
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,1))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,2))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(faceedge(i,3))
					+LocalFace(i))); 
	    z[n++] = LocalToGlobal(0.5*(LocalCenter()+LocalFace(i))); 
	}
	for (int i=0; i<Faces(); ++i) {
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(facecorner(i,0))+LocalFace(i)));
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(facecorner(i,1))+LocalFace(i)));
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(facecorner(i,2))+LocalFace(i)));
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(facecorner(i,3))+LocalFace(i)));
	}
	for (int i=0; i<Edges(); ++i)
	    z[n++] = LocalToGlobal(0.5*(LocalEdge(i)+LocalCenter()));
	for (int i=0; i<Corners(); ++i) 
	    z[n++] = LocalToGlobal(0.5*(LocalCorner(i)+LocalCenter()));  
	return *R27hex; 
    }
};

int dim (const vector<Point>& z) {
    for (int i=0; i<z.size(); ++i)
	if (z[i][2] != 0) return 3;
    return 2;
}
CELLTYPE Cells::Type (int tp, const vector<Point>& z) {
    if (dim(z) == 2) { 
	if (tp == 3 && z.size() == 3)  return TRIANGLE;
	if (tp == 4 && z.size() == 4)  return QUADRILATERAL;
	if (tp == 4 && z.size() == 8)  return QUADRILATERAL2;
    } else {
	if (tp == 4 && z.size() == 4)  return TETRAHEDRON;
	if (tp == 8 && z.size() == 8)  return HEXAHEDRON;
    	if (tp == 8 && z.size() == 20) return HEXAHEDRON20;
    	if (tp == 8 && z.size() == 27) return HEXAHEDRON27;
    }
    Exit(tp + " not implemented; file: Cell.C");
}
bool CheckQuadType (const Cell& C) {
    for (int i=0; i<C.Edges(); ++i)
	if (C[4+i] != C.Edge(i)) return true;
    return false;
}
Cell* CreateCell (CELLTYPE tp, int flag, const vector<Point>& z) {
    if (tp == TRIANGLE)      return new Triangle(z,flag);
    if (tp == QUADRILATERAL) return new Quadrilateral(z,flag);
    if (tp == QUADRILATERAL2) {
		Cell* c = new Quadrilateral2(z,flag);
		if (CheckQuadType(*c)) return c;
		delete c;
		c = new Quadrilateral(z,flag);
		c->resize(4);
		return c;
    }
    if (tp == TETRAHEDRON)   return new Tetrahedron(z,flag);
    if (tp == HEXAHEDRON)    return new Hexahedron(z,flag);
    if (tp == HEXAHEDRON20)  return new Hexahedron20(z,flag);
    if (tp == HEXAHEDRON27)  return new Hexahedron27(z,flag);
    Exit(tp + " not implemented; file: Cell.C");
}
cell Cells::Insert (CELLTYPE tp, int flag, const vector<Point>& z) {
    return Insert(CreateCell(tp,flag,z));
}

const short Rtri0[] = {0,3,5}; 
const short Rtri1[] = {3,1,4}; 
const short Rtri2[] = {5,3,4}; 
const short Rtri3[] = {5,4,2}; 

const short Rquad0[] = {0,4,8,7}; 
const short Rquad1[] = {1,5,8,4}; 
const short Rquad2[] = {2,6,8,5}; 
const short Rquad3[] = {3,7,8,6}; 

const short R2quad0[] = {0,4,8,7, 9,11,20,19}; 
const short R2quad1[] = {1,5,8,4,12,14,11,10}; 
const short R2quad2[] = {2,6,8,5,15,17,14,13}; 
const short R2quad3[] = {3,7,8,6,18,20,17,16}; 

const short Rtet0a[] = {0,4,6,7}; 
const short Rtet1a[] = {1,5,4,8}; 
const short Rtet2a[] = {2,6,5,9}; 
const short Rtet3a[] = {7,8,9,3}; 
const short Rtet4a[] = {6,4,7,8}; 
const short Rtet5a[] = {6,7,9,8}; 
const short Rtet6a[] = {6,9,5,8}; 
const short Rtet7a[] = {6,5,4,8}; 

const short Rtet0b[] = {0,4,6,7}; 
const short Rtet1b[] = {1,5,4,8}; 
const short Rtet2b[] = {2,6,5,9}; 
const short Rtet3b[] = {7,8,9,3}; 
const short Rtet4b[] = {5,4,6,7}; 
const short Rtet5b[] = {5,6,9,7}; 
const short Rtet6b[] = {5,9,8,7}; 
const short Rtet7b[] = {5,8,4,7}; 

const short Rtet0c[] = {0,4,6,7}; 
const short Rtet1c[] = {1,5,4,8}; 
const short Rtet2c[] = {2,6,5,9}; 
const short Rtet3c[] = {7,8,9,3}; 
const short Rtet4c[] = {4,5,6,9}; 
const short Rtet5c[] = {4,6,7,9}; 
const short Rtet6c[] = {4,7,8,9}; 
const short Rtet7c[] = {4,8,5,9}; 

const short Rhex0[] = { 0, 8,20,11,12,21,26,24}; 
const short Rhex1[] = { 8, 1, 9,20,21,13,22,26}; 
const short Rhex2[] = {11,20,10, 3,24,26,23,15}; 
const short Rhex3[] = {20, 9, 2,10,26,22,14,23}; 
const short Rhex4[] = {12,21,26,24, 4,16,25,19}; 
const short Rhex5[] = {21,13,22,26,16, 5,17,25}; 
const short Rhex6[] = {24,26,23,15,19,25,18, 7}; 
const short Rhex7[] = {26,22,14,23,25,17, 6,18}; 

const short R20hex0[] = { 0, 8,20,11,12,21,26,24,27,51,54,34,35,56,55,71,57,60,75,73};
const short R20hex1[] = { 8, 1, 9,20,21,13,22,26,28,29,52,51,56,37,61,55,58,62,65,60};
const short R20hex2[] = {11,20,10, 3,24,26,23,15,54,53,32,33,71,55,66,41,75,70,68,72};
const short R20hex3[] = {20, 9, 2,10,26,22,14,23,52,30,31,53,55,61,39,66,65,63,67,70};
const short R20hex4[] = {12,21,26,24, 4,16,25,19,57,60,75,73,36,59,80,74,43,76,79,50};
const short R20hex5[] = {21,13,22,26,16, 5,17,25,58,62,65,60,59,38,64,80,44,45,77,76};
const short R20hex6[] = {24,26,23,15,19,25,18, 7,75,70,68,72,74,80,69,42,79,78,48,49};
const short R20hex7[] = {26,22,14,23,25,17, 6,18,65,63,67,70,80,64,40,69,77,46,47,78};

const short R27hex0[] = { 0, 8,20,11,12,21,26,24,27,51,54,34,35,56,55,71,57,60,75,73, 81, 85,105,108, 98,109,117};
const short R27hex1[] = { 8, 1, 9,20,21,13,22,26,28,29,52,51,56,37,61,55,58,62,65,60, 84, 86, 89,106,105,110,118};
const short R27hex2[] = {11,20,10, 3,24,26,23,15,54,53,32,33,71,55,66,41,75,70,68,72, 82,108,107, 94, 97,112,120}; 
const short R27hex3[] = {20, 9, 2,10,26,22,14,23,52,30,31,53,55,61,39,66,65,63,67,70, 83,106, 90, 93,107,111,119};
const short R27hex4[] = {12,21,26,24, 4,16,25,19,57,60,75,73,36,59,80,74,43,76,79,50,109, 88,113,116, 99,101,121}; 
const short R27hex5[] = {21,13,22,26,16, 5,17,25,58,62,65,60,59,38,64,80,44,45,77,76,110, 87, 92,114,113,102,122};
const short R27hex6[] = {24,26,23,15,19,25,18, 7,75,70,68,72,74,80,69,42,79,78,48,49,112,116,115, 95,100,104,124};
const short R27hex7[] = {26,22,14,23,25,17, 6,18,65,63,67,70,80,64,40,69,77,46,47,78,111,114, 91, 96,115,103,123};

class ReferenceCells {
    bool onFace(const Point& z, const Cell& D, int j) {
	for (int k=0; k<D.FaceCorners(j); ++k)
	    if (z == D.FaceCorner(j,k)) return true;
	return false;
    }
    bool onFace(const Cell& C, int i, const Cell& D, int j) {
	for (int k=0; k<C.FaceCorners(i); ++k)
	    for (int l=0; l<D.FaceCorners(j); ++l)
	    if (C.FaceCorner(i,k) == D.FaceCorner(j,l)) return true;
	return false;
    }
    bool onTriangleFace(const Cell& C, int i, const Cell& D, int j) {
	int n = 0;
	for (int k=0; k<C.FaceEdges(i); ++k)
	    for (int l=0; l<D.FaceCorners(j); ++l)
		if (C.FaceEdge(i,k) == D.FaceCorner(j,l)) ++n;
	if (n==3) return true;
	if (n<2) return false;
	for (int k=0; k<C.FaceCorners(i); ++k)
	    for (int l=0; l<D.FaceCorners(j); ++l)
		if (C.FaceCorner(i,k) == D.FaceCorner(j,l)) return true;
	return false;
    }
    void RefineFaces (const Cell& C, const Cell& c, vector<short>& f) {
	f.resize(c.Faces());
	for (int j=0; j<c.Faces(); ++j) f[j] = -1;
	for (int i=0; i<C.Faces(); ++i) {
	    if (C.FaceCorners(i) == 3) {
		for (int j=0; j<c.Faces(); ++j) 
		    if (onTriangleFace(C,i,c,j))
			f[j] = i;
		continue;
	    }
	    Point z = C.Face(i);
	    for (int j=0; j<c.Faces(); ++j) 
		if (onFace(z,c,j))
		    if (onFace(C,i,c,j))
			f[j] = i;
	}
    }
    void RefineEdges (const Cell& C, const vector<Point>& z) {
	for (int i=0; i<C.Edges(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Edge(i) == z[j]) cout << j << ", ";
	cout << endl;
    }
    void RefineCell_old (const Cell& C, const vector<Point>& z) {
	for (int i=0; i<C.Corners(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Corner(i) == z[j]) cout << j << ", ";
	for (int i=0; i<C.Edges(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Edge(i) == z[j]) cout << j << ", ";
	for (int i=0; i<C.Faces(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Face(i) == z[j]) cout << j << ", ";
	for (int j=0; j<z.size(); ++j) 
	    if (C.Center() == z[j]) cout << j << ", ";
	cout << endl;
    }
    void RefineCell (const Cell& C, const vector<Point>& z) {
	for (int i=0; i<C.Corners(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Corner(i) == z[j]) cout << j << " co " << C.Corner(i) << endl;
	for (int i=0; i<C.Edges(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Edge(i) == z[j]) cout << j << " ed " << C.Edge(i) << endl;
	for (int i=0; i<C.Faces(); ++i) 
	    for (int j=0; j<z.size(); ++j) 
		if (C.Face(i) == z[j]) cout << j << " fa " << i << " " << C.Face(i) << endl;
	for (int j=0; j<z.size(); ++j) 
	    if (C.Center() == z[j]) cout << j << " ce " << C.Center() << endl;
	cout << endl;
    }
    const Rules* Refine (int n, Rule* R, const Cell& C) {
	vector<Point> z;
	vector<Point> x;
	C.Refine(z);
//	for (int j=0; j<z.size(); ++j) 
//	    cout << j << " z " << z[j] << endl;
	for (int i=0; i<n; ++i) {
	    R[i](z,x);
	    Cell* c = CreateCell(R[i].type(),-1,x);
//	    RefineCell(*c,z);
//	    RefineCell_old(*c,z);
	    RefineFaces(C,*c,R[i].face);
	    delete c;
	}
	return new Rules(n,R);
    }
    const Point* LocalEdges (const Cell& C) {
	Point* z = new Point [C.Edges()];
	for (int i=0; i<C.Edges(); ++i)
	    z[i] = C.LocalToGlobal(0.5*(C.LocalCorner(C.edgecorner(i,0))
					+C.LocalCorner(C.edgecorner(i,1))));
	return z;
    }
    const Point* LocalFaces (const Cell& C) {
	Point* z = new Point[C.Faces()];
	for (int i=0; i<C.Faces(); ++i) {
	    z[i] = zero;
	    for (int j=0; j<C.FaceCorners(i); ++j) {
		z[i] += C.FaceCorner(i,j);
	    }
	    z[i] *= 1.0 / C.FaceCorners(i);
	}
	return z;
    }
    const Point* LocalFaceNormals (const Cell& C) {
	Point* z = new Point [C.Faces()];
	for (int i=0; i<C.Faces(); ++i) {
	    z[i] = zero;
	    if (C.FaceCorners(i) == 2) {
		const Point& P0 = C.FaceCorner(i,0);
		const Point& P1 = C.FaceCorner(i,1);
		Point N (P1[1] - P0[1], P0[0] - P1[0]);
		z[i] = (1/norm(N)) * N; 
	    }
	    else if (C.FaceCorners(i) > 2) {
		const Point& P0 = C.FaceCorner(i,0);
		const Point& P1 = C.FaceCorner(i,1);
		const Point& P2 = C.FaceCorner(i,2);
		Point X = P1-P0;
		Point Y = P2-P0;
		Point N = curl(X,Y);
		z[i] = (1/norm(N)) * N; 
	    }
	}
	return z;
    }
    const Point* LocalCenter (const Cell& C) {
	Point* z = new Point;
	*z = zero;
	for (int i=0; i<C.Corners(); ++i) *z += C[i];
	*z *= (1.0/C.Corners());
	return z;
    }
public:
    ReferenceCells() {
	// ReferenceTriangle
	ReferenceTriangle 
	    = new Triangle(Points(3,LocalCornersTri),-1);
	LocalEdgesTri       = LocalEdges(*ReferenceTriangle);
	LocalFaceNormalsTri = LocalFaceNormals(*ReferenceTriangle);
	LocalCenterTri      = LocalCenter(*ReferenceTriangle);
	Rule R_tri[] = { Rule(TRIANGLE,3,Rtri0),
			 Rule(TRIANGLE,3,Rtri1),
			 Rule(TRIANGLE,3,Rtri2),
			 Rule(TRIANGLE,3,Rtri3)};
	Rtri = Refine(4,R_tri,*ReferenceTriangle);

	// ReferenceQuadrilateral
	ReferenceQuadrilateral 
	    = new Quadrilateral(Points(4,LocalCornersQuad),-1);
	LocalEdgesQuad       = LocalEdges(*ReferenceQuadrilateral);
	LocalFaceNormalsQuad = LocalFaceNormals(*ReferenceQuadrilateral);
	LocalCenterQuad      = LocalCenter(*ReferenceQuadrilateral);
	Rule R_quad[] = { Rule(QUADRILATERAL,4,Rquad0),
			  Rule(QUADRILATERAL,4,Rquad1),
			  Rule(QUADRILATERAL,4,Rquad2),
			  Rule(QUADRILATERAL,4,Rquad3)};
	Rquad = Refine(4,R_quad,*ReferenceQuadrilateral);

	// ReferenceQuadrilateral2
	Rule R2_quad[] = { Rule(QUADRILATERAL2,8,R2quad0),
			   Rule(QUADRILATERAL2,8,R2quad1),
			   Rule(QUADRILATERAL2,8,R2quad2),
			   Rule(QUADRILATERAL2,8,R2quad3)};
	ReferenceQuadrilateral2
	    = new Quadrilateral2(Points(4,LocalCornersQuad,
					4,LocalEdgesQuad),-1);
	R2quad = Refine(4,R2_quad,*ReferenceQuadrilateral2);

	// ReferenceTetrahedron 
	ReferenceTetrahedron 
	    = new Tetrahedron(Points(4,LocalCornersTet),-1);
	LocalEdgesTet       = LocalEdges(*ReferenceTetrahedron);
	LocalFacesTet       = LocalFaces(*ReferenceTetrahedron);
	LocalFaceNormalsTet = LocalFaceNormals(*ReferenceTetrahedron);
	LocalCenterTet      = LocalCenter(*ReferenceTetrahedron);
	Rule R_tet0[] = { Rule(TETRAHEDRON,4,Rtet0a),
			 Rule(TETRAHEDRON,4,Rtet1a),
			 Rule(TETRAHEDRON,4,Rtet2a),
			 Rule(TETRAHEDRON,4,Rtet3a),
			 Rule(TETRAHEDRON,4,Rtet4a),
			 Rule(TETRAHEDRON,4,Rtet5a),
			 Rule(TETRAHEDRON,4,Rtet6a),
			 Rule(TETRAHEDRON,4,Rtet7a)};
	Rule R_tet1[] = { Rule(TETRAHEDRON,4,Rtet0b),
			 Rule(TETRAHEDRON,4,Rtet1b),
			 Rule(TETRAHEDRON,4,Rtet2b),
			 Rule(TETRAHEDRON,4,Rtet3b),
			 Rule(TETRAHEDRON,4,Rtet4b),
			 Rule(TETRAHEDRON,4,Rtet5b),
			 Rule(TETRAHEDRON,4,Rtet6b),
			 Rule(TETRAHEDRON,4,Rtet7b)};
	Rule R_tet2[] = { Rule(TETRAHEDRON,4,Rtet0c),
			 Rule(TETRAHEDRON,4,Rtet1c),
			 Rule(TETRAHEDRON,4,Rtet2c),
			 Rule(TETRAHEDRON,4,Rtet3c),
			 Rule(TETRAHEDRON,4,Rtet4c),
			 Rule(TETRAHEDRON,4,Rtet5c),
			 Rule(TETRAHEDRON,4,Rtet6c),
			 Rule(TETRAHEDRON,4,Rtet7c)};
	Rtet0 = Refine(8,R_tet0,*ReferenceTetrahedron);
	Rtet1 = Refine(8,R_tet1,*ReferenceTetrahedron);
	Rtet2 = Refine(8,R_tet2,*ReferenceTetrahedron);

	// ReferenceHexahedron
	ReferenceHexahedron 
	    = new Hexahedron(Points(8,LocalCornersHex),-1);
	LocalEdgesHex       = LocalEdges(*ReferenceHexahedron);
	LocalFacesHex       = LocalFaces(*ReferenceHexahedron);
	LocalFaceNormalsHex = LocalFaceNormals(*ReferenceHexahedron);
	LocalCenterHex      = LocalCenter(*ReferenceHexahedron);
	Rule R_hex[] = { Rule(HEXAHEDRON,8,Rhex0),
			 Rule(HEXAHEDRON,8,Rhex1),
			 Rule(HEXAHEDRON,8,Rhex2),
			 Rule(HEXAHEDRON,8,Rhex3),
			 Rule(HEXAHEDRON,8,Rhex4),
			 Rule(HEXAHEDRON,8,Rhex5),
			 Rule(HEXAHEDRON,8,Rhex6),
			 Rule(HEXAHEDRON,8,Rhex7)};
	Rhex = Refine(8,R_hex,*ReferenceHexahedron);

	// ReferenceHexahedron20
	Rule R20_hex[] = { Rule(HEXAHEDRON20,20,R20hex0),
			   Rule(HEXAHEDRON20,20,R20hex1),
			   Rule(HEXAHEDRON20,20,R20hex2),
			   Rule(HEXAHEDRON20,20,R20hex3),
			   Rule(HEXAHEDRON20,20,R20hex4),
			   Rule(HEXAHEDRON20,20,R20hex5),
			   Rule(HEXAHEDRON20,20,R20hex6),
			   Rule(HEXAHEDRON20,20,R20hex7)};
	ReferenceHexahedron20 
	    = new Hexahedron20(Points( 8,LocalCornersHex,
				     12,LocalEdgesHex),-1);
	R20hex = Refine(8,R20_hex,*ReferenceHexahedron20);

	// ReferenceHexahedron27
	Rule R27_hex[] = { Rule(HEXAHEDRON27,27,R27hex0),
			   Rule(HEXAHEDRON27,27,R27hex1),
			   Rule(HEXAHEDRON27,27,R27hex2),
			   Rule(HEXAHEDRON27,27,R27hex3),
			   Rule(HEXAHEDRON27,27,R27hex4),
			   Rule(HEXAHEDRON27,27,R27hex5),
			   Rule(HEXAHEDRON27,27,R27hex6),
			   Rule(HEXAHEDRON27,27,R27hex7)};

	ReferenceHexahedron27 
	    = new Hexahedron27(Points( 8,LocalCornersHex,
				      12,LocalEdgesHex,
				       6,LocalFacesHex,
				       1,LocalCenterHex),-1);
//	R27hex = Refine(8,R27_hex,*ReferenceHexahedron27);
    }
    ~ReferenceCells() {
	delete[] LocalEdgesTri;
	delete[] LocalFaceNormalsTri;
	delete LocalCenterTri;
	delete Rtri;
	delete ReferenceTriangle;
	delete[] LocalEdgesQuad;
	delete[] LocalFaceNormalsQuad;
	delete LocalCenterQuad;     
	delete Rquad;
	delete ReferenceQuadrilateral;
	delete R2quad;
	delete ReferenceQuadrilateral2;
	delete[] LocalEdgesTet;       
	delete[] LocalFacesTet;       
	delete[] LocalFaceNormalsTet;
	delete LocalCenterTet;      
	delete Rtet0;
	delete Rtet1;
	delete Rtet2;
	delete[] LocalEdgesHex;       
	delete[] LocalFacesHex;       
	delete[] LocalFaceNormalsHex;
	delete LocalCenterHex;
	delete ReferenceTetrahedron;
	delete Rhex;
	delete ReferenceHexahedron;
	delete R20hex;
	delete ReferenceHexahedron20;
//	delete R27hex;
	delete ReferenceHexahedron27;
    }
    const Cell* operator () (CELLTYPE tp) const {
	if (tp == TRIANGLE)       return ReferenceTriangle; 
	if (tp == QUADRILATERAL)  return ReferenceQuadrilateral;
	if (tp == QUADRILATERAL2) return ReferenceQuadrilateral;
	if (tp == TETRAHEDRON)    return ReferenceTetrahedron; 
	if (tp == HEXAHEDRON)     return ReferenceHexahedron; 
	if (tp == HEXAHEDRON20)   return ReferenceHexahedron; 
	if (tp == HEXAHEDRON27)   return ReferenceHexahedron; 
    }
};
const ReferenceCells RC;
