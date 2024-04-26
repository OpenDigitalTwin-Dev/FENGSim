// file: Solution.h
// author: Jiping Xin

#include "m++.h"

#include "exampleid.h"




class Segment {
  public:
    Point a1;
    Point a2;
    double length () const {
	double t1 = a1[0] - a2[0];
	double t2 = a1[1] - a2[1];
	return sqrt(t1*t1 + t2*t2);
    }
};

class PathPlanning {
    
    
 
    vector<Segment> am_path_planning;
    double source_v;
    double source_x;
    double source_y;
    double source_z;
    double source_h;
    double ppdistance;
    Point cur_pos;
    int cur_id;
    
 public:
    
    PathPlanning () {}
    void Initial () {
	ifstream is;
	is.open("./AdditiveManufacturing/conf/geo/pathplanning.vtk");
	const int len = 256;
	char L[len];
	double z[3];
	for (int i = 0; i < 6; i++) {
	    is.getline(L,len);
	}
	Point seg[2];
	int n = 0;
	ppdistance = 0;
	while (strncasecmp("CELLS",L,5)) {
	    sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
	    seg[n] = Point(z[0], z[1], z[2]);
	    n++;
	    if (n == 2) {
		Segment S;
		S.a1 = seg[0];
		S.a2 = seg[1];
		am_path_planning.push_back(S);
		n = 0;
		ppdistance += S.length();
	    }
	    is.getline(L,len);
	}
	/*
	for (int i = 0; i < am_path_planning.size(); i++) {
	    cout << am_path_planning[i].a1 << ": " << am_path_planning[i].a2 << endl;
	}
	*/	
	ReadConfig(Settings, "SourceV", source_v);
	ReadConfig(Settings, "SourceX", source_x);
	ReadConfig(Settings, "SourceY", source_y);
	ReadConfig(Settings, "SourceZ", source_z);
	ReadConfig(Settings, "SourceH", source_h);
    }
    Point SetCurrentPosition (double num, double dt) {
	cur_id = num;
        // ==============================================================
	//                   
	//       1.    ------d-------s
	//
	//       2.    --------------s(d)
	//
	// ==============================================================
	double time = num * dt;
	double d = time * source_v;
	double s = 0;
	for (int i = 0; i < am_path_planning.size(); i++) {
	    s += am_path_planning[i].length();
	    if (s >= d) {
		double t = (s - d) / am_path_planning[i].length();
		cur_pos = am_path_planning[i].a1  + (am_path_planning[i].a2 - am_path_planning[i].a1) * (1 - t);
		mout << "  path planning: " << ppdistance << " d: " << d << " s: " << s << " cur_pos: " << cur_pos << endl;
		return cur_pos;
	    }
	}
    }
    int cur_index () {
	return cur_id;
    }
    Point cur_position () {
	return cur_pos;
    }
    double distance () {
	return ppdistance;
    }
    bool stop (double id, double dt) {
	if (id*dt*source_v >= ppdistance)
	    return true;
	return false;
    }
    bool IsActive (cell c) {
	Point p = cur_pos;
	if (p[0]-source_x < c()[0] && c()[0] < p[0]+source_x)
	    if (p[1]-source_y < c()[1] && c()[1] < p[1]+source_y)
		if (p[2]-source_z < c()[2] && c()[2] <= p[2])
		    return true;
	return false;
    }
    void ExportCurrentMesh (Mesh& M) {
	if (!PPM->master()) return;
	string filename = string("./AdditiveManufacturing/conf/geo/thinwall2.geo");  
	ofstream out(filename.c_str()); 
	out<< "POINTS:" << endl;
	int n = 0;
	for (cell c = M.cells(); c != M.cells_end(); c++) {
	    if (IsActive(c)) {
		hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(c());
		it->second->SetSubdomain(1);
		for (int i = 0; i < c.Corners(); i++)
		    out << c.Corner(i) << endl;
		n++;
	    }
	    else if (c()[2] < source_h) {
		hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(c());
		it->second->SetSubdomain(1);
		for (int i = 0; i < c.Corners(); i++)
		    out << c.Corner(i) << endl;
		n++;
	    }
	    else if (c.Subdomain() == 1) {
		for (int i = 0; i < c.Corners(); i++)
		    out << c.Corner(i) << endl;
		n++;
	    }
	}
	out << "CELLS: " << endl;
	for (int i = 0; i < n; i++) out << 4 << " " << 1 << " " << i*4 << " " << i*4 + 1 << " " << i*4 + 2 << " " << i*4 + 3 << " " << endl;
    }
    bool IsSource (bnd_face bf) {
	Point p = cur_pos;
	if (p[2] == bf()[2]) {
	    if (p[0]-source_x < bf()[0] && bf()[0] < p[0]+source_x) {
		if (p[1]-source_y < bf()[1] && bf()[1] < p[1]+source_y) {
		    return true;
		}
	    }
	}
	return false;
    }    
};


class AMBVP {
    int example_id;
  public:
    AMBVP () {
	ReadConfig(Settings, "EXAMPLE", example_id);
    }
    PathPlanning* pp = new PathPlanning;
    void SetDomain (Mesh& M);
    void SetBoundary (Mesh& M);
    // given data
    bool IsDirichlet (int id);
    double Coefficient (Point p) const;
    double Source (Point p) const;
    double Dirichlet (Point p, int id) const;
    double Neumann (Point p, int id) const;
    double Solution (const Point& p) const;
};
