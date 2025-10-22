//Copyright (c) 2019 Ultimaker B.V.
//CuraEngine is released under the terms of the AGPLv3 or higher.

#include <gtest/gtest.h>

#include "../../../../../../toolkit/Geometry/cura_engine/src/infill.h"
#include "ReadTestPolygons.h"

//#define TEST_INFILL_SVG_OUTPUT
#ifdef TEST_INFILL_SVG_OUTPUT
#include <cstdlib>
#include "../src/utils/SVG.h"
#endif //TEST_INFILL_SVG_OUTPUT
#include "fstream"

namespace cura
{
    void VtkToPolygons (std::string filename, std::vector<Polygons>& layers, std::vector<double>& heights) {

	std::vector<Point3> pnts;
	heights.clear();
	
	std::ifstream is;
	is.open(filename.c_str());
	
	const int len = 512*2;
	char L[len];
	
	for (int i = 0; i < 5; i++) is.getline(L,len);
	
	double scale = 1000;
	
	while (is.getline(L,len)) {
	    if (strncasecmp("POLYGONS", L, 8) == 0) break;
	    double z[3];
	    sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
	    Point3 p;
	    p.x = z[0] * scale;
	    p.y = z[1] * scale;
	    p.z = z[2] * scale;
	    pnts.push_back(p);

	    if (heights.size() > 0) {
	        if ((z[2] * scale) != heights[heights.size()-1])
		    heights.push_back(z[2] * scale);
	    }
	    else {
		heights.push_back(z[2] * scale);
	    }
	}
	
	while (is.getline(L,len)) {
	    cura::Polygon Poly;
	    cura::Polygons Polys;
	    
	    int m = -1;
	    sscanf(L, "%d[^ ]", &m);
	    std::cout << "polygon: " << m << std::endl;

	    std::string ss1 = "%*d";
	    for (int i = 0; i < m; i++) {
		std::string ss2 = "";
		ss2 = ss1 + " %d[^ ]";
		int v;
		sscanf(L, ss2.c_str(), &v);
		Poly.emplace_back(pnts[v].x, pnts[v].y);
		ss1 += " %*d";
	    }
	    Polys.add(Poly);
	    layers.push_back(cura::Polygons(Polys));
	}
	std::cout << "layers: " << layers.size() << std::endl;
	is.close();
    }
    
    void ExportOutLinesToVtk (Polygons P, double height, std::string filename=" ") {
	int n = 0;
	for (int i = 0; i < P.size(); i++) {
	    n += P[i].size();
	}
	
	std::ofstream out;
	out.open((std::string("./data/vtk/")+filename).c_str());
	//out.open(filename.c_str());
	
	out <<"# vtk DataFile Version 2.0" << std::endl;
	out << "slices example" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	out << "POINTS " << n << " float" << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    for (int j = 0; j < P[i].size(); j++) {
		out << P[i][j].X << " " << P[i][j].Y << " " << height << std::endl;
	    }
	}
	out << "CELLS " << P.size() << " " << P.size() + n + P.size() << std::endl;
	int m = 0;
	for (int i = 0; i < P.size(); i++) {
	    out << P[i].size() + 1;
	    for(int j = 0; j < P[i].size(); j++) { //Find the starting corner in the sliced layer.
		out << " " << m;
		m++;
	    }
	    out << " " << m - P[i].size();
	    out << std::endl;
	}
	out << "CELL_TYPES " << P.size() << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    out << 4 << std::endl;
	}
    }

    void ExportPathLinesToVtk (Polygons P, double height, std::string filename) {
	int n = 0;
	for (int i = 0; i < P.size(); i++) {
	    n += P[i].size();
	}
	
	std::ofstream out;
	out.open((std::string("./data/vtk/")+filename).c_str());
	//out.open(filename.c_str());
	
	out <<"# vtk DataFile Version 2.0" << std::endl;
	out << "slices example" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	out << "POINTS " << n << " float" << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    for (int j = 0; j < P[i].size(); j++) {
		out << P[i][j].X << " " << P[i][j].Y << " " << height << std::endl;
	    }
	}
	out << "CELLS " << P.size() << " " << P.size() + n << std::endl;
	int m = 0;
	for (int i = 0; i < P.size(); i++) {
	    out << P[i].size();
	    for(int j = 0; j < P[i].size(); j++) { //Find the starting corner in the sliced layer.
		out << " " << m;
		m++;
	    }
	    out << std::endl;
	}
	out << "CELL_TYPES " << P.size() << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    out << 4 << std::endl;
	}
    }

    void ExportPathLinesToMbdyn (Polygons P, double height) {
	std::ofstream out;
	out.open("../../mbdyn/robot/_ur3.traj");
	
	for (int i = 0; i < P.size(); i++) {
	    for (int j = 0; j < P[i].size()-1; j++) {
		out << P[i][j].X << " " << P[i][j].Y << " " << height << " "
		    << P[i][j+1].X << " " << P[i][j+1].Y << " " << height << " "
		    << "90 0 0 1"
		    << std::endl;
	    }
	}
    }    
}
