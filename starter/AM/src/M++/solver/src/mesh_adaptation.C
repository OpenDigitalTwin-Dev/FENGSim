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
	    //  fmod Returns the floating-point remainder of numer/denom (rounded towards zero):
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

void vtk2geo ();

void mesh_coarsing () {    
    Date Start;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    int dim = M.dim();
    mout << "fine mesh cells: " << M.fine().Cells::size() << endl;

    // read original mesh from .geo and export to .vtk
    Plot P(M.fine());
    P.vtk_mesh("fine_mesh",0,0);

    // polygon coarsing and export to .off, h0 the base height, h1 the coarse height
    double h0 = 2;
    double h1 = 1;
    ReadConfig(Settings, "h0", h0);
    ReadConfig(Settings, "h1", h1);
    Vertices vs;
    get_bnd_vertices (vs, M);
    get_slices_vertices (vs, M, h0, h1);
    set_vertices_ordering (vs);
    export_to_off (vs, M, h0, h1);

    // remeshing with the constraints and export to .vtk
    string flags = "pkYa1";
    ReadConfig(Settings, "tetgenflags", flags);
    tetgenio tin, tout, addin, bgmin;
    tin.load_off("./solver/conf/geo/test");
    tetrahedralize(const_cast<char*>(flags.c_str()), &tin, NULL);

    // change .vtk to .geo in solver/conf/thinwall2.geo
    vtk2geo();
    
    Meshes M2("thinwall2");
    dim = M2.dim();
    mout << "coarse mesh cells: " << M2.fine().Cells::size() << endl;

    Plot P2(M2.fine());
    P2.vtk_mesh("coarse_mesh",0,0);

    return;
}

void mesh_combine (int i, int j, double h1, double h2, double h3, const Meshes& M, const Meshes& M2) {
    double H1 = h1+i*h2;
    double H2 = h1+i*h2+j*h3;
    mout << H1 << " " << H2 << endl;
    int N1,N2;
    N1 = 0;
    N2 = 0;
    // export to .geo
    std::ofstream out;
    out.open("./solver/conf/geo/thinwall3.geo");
    out << "POINTS" << endl;
    // fine mesh 
    for (cell c=M.fine().cells(); c!=M.fine().cells_end(); c++) {
	if (c()[2]<H2&&c()[2]>H1) {
	    for (int i=0; i<c.Corners(); i++) {
		out << c.Corner(i) << endl;
	    }
	    N1++;
	}
    }
    mout << "N1: " << N1 << endl;
    // coarse mesh
    for (cell c=M2.fine().cells(); c!=M2.fine().cells_end(); c++) {
	if (c()[2]<H1) {
	    for (int i=0; i<c.Corners(); i++) {
		out << c.Corner(i) << endl;
	    }
	    N2++;
	}
    }
    mout << "N2: " << N2 << endl;
    out << "CELLS" << endl;
    for (int i=0; i<N1+N2; i++) {
	out << "4 0 " << i*4+0 << " " << i*4+1 << " " << i*4+2 << " " << i*4+3 << endl;
    }
    out.close();
    // export to .vtk
    Meshes M3("thinwall3");
    mout << "adaptive mesh cells: " << M3.fine().Cells::size() << endl;
    Plot P(M3.fine());
    P.vtk_mesh((string("adaptive_mesh_")+to_string(i*9+j)).c_str(),0,0);
}

void mesh_adaptive () {
    mesh_coarsing ();
    
    Meshes M("thinwall");
    Meshes M2("thinwall2");
    
    for (int i=0; i<6; i++) {
	for (int j=0; j<9; j++) {
	    // 2 the height of base, 2 the coarse height, 0.25 the slice height
	    mesh_combine(i,j,2,2,0.25,M,M2);
	}
    }
}

void mesh_partitioning () {
    Meshes M("thinwall3");
    Discretization disc("cell",1);
    MatrixGraphs G(M,disc);
    Vector x(G.fine());
    x = PPM->proc();

    Plot P(M.fine(),M.dim(),1);
    P.celldata(x,1);
    P.vtk_celldata("mesh_distribute",0,0);
}
