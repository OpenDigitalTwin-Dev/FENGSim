#include<iostream>
#include "tetgen.h"
#include "m++.h"

void meshcoarsing () {
    
    Date Start;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    int dim = M.dim();
    mout << M.fine().Cells::size() << endl;

    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector x(G.fine());
    x = 0;
    int k = 0;
    for (row r=x.rows(); r!=x.rows_end(); r++) {
	x(r,0) = k;
	k++;
    }
    Plot P(M.fine());
    P.vertexdata(x,dim);
    P.vtk_vertexdata("meshcoarsing",0,0);





    Vertices vs;
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

    int num = 0;
    for (vertex v=vs.vertices(); v!=vs.vertices_end(); v++) {
	vs.find(v)->second.SetPart(num);
	num++;
    }

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
    out.close();
    
    /*
    std::ofstream out;
    out.open("./solver/conf/geo/test.poly");
    
    out << "# Part 1 - node list" << endl;
    out << "# node count, 3 dim, no attribute, no boundary marker" << endl;
    out << M.fine().BoundaryFaces::size()*3 << " 3 0 0" << endl;
    out << "# Node index, node coordinates" << endl;
    int n = 0;
    for (bnd_face bf=M.fine().bnd_faces(); bf!=M.fine().bnd_faces_end(); bf++) {
	if (M.fine().find_face(bf()).Left()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Right());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    for (int j=0; j<c.FaceCorners(i); j++) {
			n++;
			out << n
			    << " " << c.FaceCorner(i,j)[0]
			    << " " << c.FaceCorner(i,j)[1]
			    << " " << c.FaceCorner(i,j)[2]
			    << endl;
		    }
		}
	    }
	}
	else if (M.fine().find_face(bf()).Right()==Infty) {
	    cell c = M.fine().find_cell(M.fine().find_face(bf()).Left());
	    for (int i=0; i<c.Faces(); i++) {
		if (c.Face(i)==bf()) {
		    for (int j=0; j<c.FaceCorners(i); j++) {
			n++;
			out << n
			    << " " << c.FaceCorner(i,j)[0]
			    << " " << c.FaceCorner(i,j)[1]
			    << " " << c.FaceCorner(i,j)[2]
			    << endl;
		    }
		}
	    }
	}
    }

    out << "# Part 2 - facet list" << endl;
    out << "# facet count, no boundary marker" << endl;
    out << M.fine().BoundaryFaces::size() << " 0" << endl;
    out << "# facets" << endl;
    for (int i=0; i<M.fine().BoundaryFaces::size(); i++) {
	out << 1 << endl;
	out << "3 " << 1+i*3 << " " << 2+i*3 << " " << 3+i*3 << endl;
    }

    out << "# Part 3 - hole list" << endl;
    out << "0 # no hole" << endl;

    out << "# Part 4 - region list" << endl;
    out << "0 # no region" << endl;
    */













    tetgenio tin, tout, addin, bgmin;
    tin.load_off("./solver/conf/geo/test");
    tetrahedralize("pkYa5", &tin, NULL);

    return;
    //out.save_nodes("barout");
    //out.save_elements("barout");
    //out.save_faces("barout");
    //out.save_poly("barout");
}
