// file: Meshfile.h
// author: Christian Wieners
// $Header: /public/M++/src/Meshfile.h,v 1.3 2007-07-06 10:33:37 sydow Exp $

#ifndef _MESHFILE_H_
#define _MESHFILE_H_

#include <list>

#include "IO.h"
#include "Debug.h"

class Coordinate {
    double z[3];
public:
    Coordinate (const double* y) { memcpy(z,y,3*sizeof(double)); }
    double operator [] (int i) const { return z[i]; }
};
inline ostream& operator << (ostream& s, const Coordinate& z) {
    return s << z[0] << " " << z[1] << " " << z[2];
}
class Ids {
    int N;
    int M;
    int F;
    int ids[27];
public:
    Ids (int n, int m, int f, const int* i) : N(n), M(m), F(f) {
		memcpy(ids,i,m*sizeof(int));
    }
    int type () const { return N; }
    int size () const { return M; }
    int flag () const { return F; }
    int id (int i) const { return ids[i]; }
};
inline ostream& operator << (ostream& s, const Ids& i) {
    s << "type " << i.type()
      << " flag " << i.flag()
      << " id";
    for (int k=0; k<i.size(); ++k) s << " " << i.id(k);
    return s;
}
struct CoarseGeometry {
    list<Coordinate> Coordinates;
    list<Ids> Cell_Ids;
    list<Ids> Face_Ids;
}; 
class SimpleCoarseGeometry : public CoarseGeometry {
    bool Insert (char* L, list<Coordinate>& Coordinates) {
		dout(100) << L << "\n";
		double z[3];
		int d = sscanf(L,"%lf %lf %lf",z,z+1,z+2);
		if (d<2) return false;
		if (d==2) z[2] = 0;
		Coordinates.push_back(Coordinate(z));
		return true;
    }
    bool Insert (char* L, list<Ids>& ids) {
		dout(99) << L << "\n";
		int n,f,i[27];
		int m = sscanf(L,"%d %d %d %d %d %d %d %d %d %d %d" 
					   "%d %d %d %d %d %d %d %d %d %d %d"
					   "%d %d %d %d %d %d %d %d %d %d %d",
					   &n,&f,
					   i,i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9,
					   i+10,i+11,i+12,i+13,i+14,i+15,i+16,i+17,i+18,i+19,
					   i+20,i+21,i+22,i+23,i+24,i+25,i+26);
		if (m < 3) return false;
		//for (int k=0; k<m; ++k) --i[k+1];
		ids.push_back(Ids(n,m-2,f,i));
		return true;
    }
public:
    SimpleCoarseGeometry (istream& is) {
		const int len = 256;
		char L[len];
		is.getline(L,len);
		if (strncasecmp("points",L,6) == 0) is.getline(L,len);
		while (Insert(L,Coordinates)) is.getline(L,len);
		if (strncasecmp("cells",L,5) == 0) is.getline(L,len);
		while (Insert(L,Cell_Ids)) is.getline(L,len);
		if (strncasecmp("faces",L,5) == 0) is.getline(L,len);
		while (Insert(L,Face_Ids)) is.getline(L,len);
    }
	SimpleCoarseGeometry (const vector<Point>& coords, const vector<int>& ids) {
		for (int i=0; i<coords.size(); i++) {
			Coordinates.push_back(Coordinate(coords[i]()));
		}
		int n = ids.size() / 4; 
		for (int i=0; i<n; i++) {
			int id[4];
			for (int j=0; j<4; j++)
				id[j] = ids[4*i+j];
			Cell_Ids.push_back(Ids(4,4,0,id));
		}
    }
  	SimpleCoarseGeometry (double* coords, int n, int* ids, int m) {
		int num_vertices = n/3;
		for (int i=0; i<num_vertices; i++) {
			double t[3];
			for (int j=0; j<3; j++)
				t[j] = coords[3*i+j];
			Coordinates.push_back(Coordinate(t));
		}
		int num_cells = m/4; 
		for (int i=0; i<num_cells; i++) {
			int t[4];
			for (int j=0; j<4; j++)
				t[j] = ids[4*i+j];
			Cell_Ids.push_back(Ids(4,4,0,t));
		}
    }
}; 
inline ostream& operator << (ostream& s, const SimpleCoarseGeometry& M) {
    return s << "POINTS: " << M.Coordinates.size() << endl << M.Coordinates
			 << "CELLS: " << M.Cell_Ids.size() << endl << M.Cell_Ids
			 << "FACES: " << M.Face_Ids.size() << endl << M.Face_Ids;
}

#endif
