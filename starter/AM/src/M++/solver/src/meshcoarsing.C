#include<iostream>
#include "tetgen.h"
#include "m++.h"

void get_bnd_vertices (Vertices& vs, const Meshes& M) {
    for (bnd_face bf=M.fine().bnd_faces(); bf!=M.fine().bnd_faces_end(); bf++) {
	if (M.fine().find_face(bf()).Left()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Right());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    for (int j=0; j<c.FaceCorners(i); j++) {
			Point v = c.FaceCorner(i,j);
			if (vs.find(v)==vs.end()) {
			    vs.Insert(v);
			}
		    }
		}
	    }
	}
	else if (M.fine().find_face(bf()).Right()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Left());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    for (int j=0; j<c.FaceCorners(i); j++) {
			Point v = c.FaceCorner(i,j);
			if (vs.find(v)==vs.end()) {
			    vs.Insert(v);
			}
		    }
		}
	    }
	}
    }
}

void get_slices_vertices (Vertices& vs, const Meshes& M, double h0, double h1) {
    for (cell c=M.fine().cells(); c!=M.fine().cells_end(); c++) {
	for (int i=0; i<c.Faces(); i++) {
	    double k = std::fmod(c.Face(i)[2]-h0,h1);
	    if (k==0) {
		int l = (c.Face(i)[2]-h0) / h1;
		if (c()[2] < h0+l*h1 && c()[2] > h0) {
		    for (int j=0; j<c.FaceCorners(i); j++) {
			Point v = c.FaceCorner(i,j);
			if (vs.find(v)==vs.end()) {
			    vs.Insert(v);
			}
		    }
		}
	    }
	}
    }
}

void set_vertices_ordering (Vertices& vs) {
    int num = 0;
    for (vertex v=vs.vertices(); v!=vs.vertices_end(); v++) {
	vs.find(v)->second.SetPart(num);
	num++;
    }
}

void export_to_off (Vertices& vs, const Meshes& M, double h0, double h1) {
    std::ofstream out;
    out.open("./solver/conf/geo/test.off");

    out << "OFF" << endl;
    out << vs.size() << " " << M.fine().BoundaryFaces::size() << " 0"<< endl;
    for (vertex v=vs.vertices(); v!=vs.vertices_end(); v++) {
	out << v()[0] << " " << v()[1] << " " << v()[2] << endl;
    }
    for (bnd_face bf=M.fine().bnd_faces(); bf!=M.fine().bnd_faces_end(); bf++) {
	if (M.fine().find_face(bf()).Left()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Right());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    out << "3 ";
		    for (int j=0; j<c.FaceCorners(i); j++) {
			out << vs.find(c.FaceCorner(i,j))->second._part() << " ";
		    }
		    out << endl;
		}
	    }
	}
	else if (M.fine().find_face(bf()).Right()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Left());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    out << "3 ";
		    for (int j=0; j<c.FaceCorners(i); j++) {
			out << vs.find(c.FaceCorner(i,j))->second._part() << " ";
		    }
		    out << endl;
		}
	    }
	}
    }
    for (cell c=M.fine().cells(); c!=M.fine().cells_end(); c++) {
	for (int i=0; i<c.Faces(); i++) {
	    double k = std::fmod(c.Face(i)[2]-h0,h1);
	    if (k==0) {
		int l = (c.Face(i)[2]-h0) / h1;
		if (c()[2] < h0+l*h1 && c()[2] > h0) {
		    out << "3 ";
		    for (int j=0; j<c.FaceCorners(i); j++) {
			out << vs.find(c.FaceCorner(i,j))->second._part() << " ";
		    }
		    out << endl;
		}
	    }
	}
    }
    out.close();
}

void mesh_coarsing () {
    
    Date Start;
    /*    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    int dim = M.dim();
    mout << M.fine().Cells::size() << endl;

    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector x(G.fine());
    x = 0;
    Plot P(M.fine());
    P.vertexdata(x,dim);
    P.vtk_vertexdata("thinmesh",0,0);

    // polygon coarsing 
    double h0 = 2;
    double h1 = 1;
    ReadConfig(Settings, "h0", h0);
    ReadConfig(Settings, "h1", h1);
    Vertices vs;
    get_bnd_vertices (vs, M);
    get_slices_vertices (vs, M, h0, h1);
    set_vertices_ordering (vs);
    export_to_off (vs, M, h0, h1);

    // remeshing with the constraints 
    string flags = "pkYa5";
    ReadConfig(Settings, "tetgenflags", flags);
    tetgenio tin, tout, addin, bgmin;
    tin.load_off("./solver/conf/geo/test");
    tetrahedralize(const_cast<char*>(flags.c_str()), &tin, NULL);
    */

    
    Meshes M2("thinwall");
    Discretization disc2linear(M2.dim());
    MatrixGraphs G2linear(M2, disc2linear);
    Vector x2linear(G2linear.fine());
    x2linear = 0;
    mout << x2linear.size() << endl;

    Discretization disc2cell("cell",1);
    MatrixGraphs G2cell(M2, disc2cell);
    Vector x2cell(G2cell.fine());
    x2cell = PPM->proc();
    mout << x2cell.size() << endl;


    Plot P2(M2.fine(),3,1);
    P2.vertexdata(x2linear,M2.dim());
    P2.vtk_vertexdata("mesh_parallel",0,0);

    P2.celldata(x2cell,1);
    P2.vtk_celldata("mesh_parallel",0,0);
    
    return;
}
