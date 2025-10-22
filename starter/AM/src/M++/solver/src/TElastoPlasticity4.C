#include "TElastoPlasticity.h"
//#include "Rezoning.h"
#include "LogarithmicStrain.h"

// *****************************
// updated lagrange
// *****************************

void TElastoPlasticityAssemble::Update4 (Vector& _x1, Vector& _s1ud, Vector& _ep1ud, Vector& _st1) {
    for (cell c = _x1.GetMesh().cells(); c != _x1.GetMesh().cells_end(); c++) {
	VectorFieldElement E(disc, _x1, c);
	for (int q = 0; q < E.nQ(); q++) {
	    //-----------------------------------------------------------------------------
	    // return mapping
	    // k
	    Tensor s1ud = sym(E.VectorGradient(q, _x1));
	    Tensor ep1ud = GetTensorFromVector(c(), _ep1ud);
	    Tensor st1 = GetTensorFromVector(c(), _st1);
			
	    Tensor4 C;
	    C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One);
			
	    Tensor S = dev(st1);
	    Tensor N;
	    if (norm(S) == 0) N = Zero;
	    else N = 1.0 / norm(S) * S;
			
	    double Y = sqrt(2.0/3.0) * k_0;
	    if (norm(S) > Y) {
		double gamma = Frobenius(N,C*s1ud) / Frobenius(N,C*N);
		//double gamma = Frobenius(N,s1ud);
		ep1ud = gamma * N;
		Tensor st1ud = C * (s1ud - ep1ud);
		st1 += st1ud;
		SetTensorToVector(s1ud, c(), _s1ud);
		SetTensorToVector(ep1ud, c(), _ep1ud);
		SetTensorToVector(st1, c(), _st1);
	    }
	    else {
		Tensor st1ud = C * (s1ud - ep1ud);
		st1 += st1ud ;
		SetTensorToVector(s1ud, c(), _s1ud);
		SetTensorToVector(ep1ud, c(), _ep1ud);
		SetTensorToVector(st1, c(), _st1);
	    }
	}
    }
}

void TElastoPlasticityAssemble::Jacobi4 (Matrix& A, Vector& R) {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, A, c);
	RowEntries A_c(A, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int j = 0; j < E.size(); ++j) {
		    for (int l = 0; l < dim; ++l) {
			for (int q = 0; q < E.nQ(); q++) {
			    A_c(i,i,k,k) += rho0 / E.Area() * E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
			}
		    }
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

// *****************************
// updated lagrange
// *****************************
double TElastoPlasticityAssemble::Residual4 (Vector& b, Vector& x1, Vector& _s1ud, Vector& _ep1ud, Vector& _st1, Vector& Alpha1, Vector& Beta1, double time) {
    b = 0;
    for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, b, c);
	RowValues b_c(b, E);
	// Ma + F = R : F
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    //-----------------------------------------------------------------------------
		    // return mapping
		    // k
		    Tensor s1ud = GetTensorFromVector(c(), _s1ud);
		    Tensor ep1ud = GetTensorFromVector(c(), _ep1ud);
		    Tensor4 C;
		    C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One);
		    Tensor st1ud  = C * (s1ud - ep1ud);
		    // test
		    Tensor st1 = GetTensorFromVector(E.QPoint(q), _st1);
		    Tensor sigma2 = st1ud;
		    //-----------------------------------------------------------------------------
		    b_c(i, k) += -1.0 * Frobenius(sym(E.VectorGradient(q, i, k)), sigma2) * E.QWeight(q);
		}
	    }
	}
	// F'(x)(x'-x) + F(x) = 0  : F(x)
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i,k) += E.VectorValue(q, i, k) * Source(E.QPoint(q), time) * E.QWeight(q);
		}
	    }
	}
	// F'(x)(x'-x) + F(x) = 0  : F(x)
	RowBndValues u_c(b, c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i) {
	    if (!IsDirichlet(u_c.bc(i)) && u_c.bc(i) != -1) {
		VectorFieldFaceElement E(disc, b, c, i);
		for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
		    int k = disc.NodalPointOnFace(c,i,j);
		    for (int l = 0; l < dim; l++) {
			for (int q = 0; q < E.nQ(); q++) {
			    u_c(k,l) += E.VectorValue(q, j, l) * Neumann(E.QPoint(q), time, u_c.bc(i)) * E.QWeight(q);
			}
		    }
		}			
	    }
	}
    }
    b.ClearDirichletValues();
    Collect(b);
}

void TElastoPlasticityAssemble::MeshExport (const Vector& x1) {
    if (coords1!=NULL) {
	delete coords1;
	coords1 = NULL;
    }
    if (ids1!=NULL) {
	delete ids1;
	ids1 = NULL;
    }
    if (fixed1!=NULL) {
	delete fixed1;
	fixed1 = NULL;
    }
    cells1.clear();
    // vertices
    num_vertices = x1.GetMesh().Vertices::size();
    coords1 = new double[num_vertices*3];
    int dim = x1.GetMesh().dim();
    for (row r=x1.rows(); r!=x1.rows_end(); r++) {
	for (int i=0; i<dim; i++) {
	    coords1[r.Id()*3+i] = Point(r()+x1(r))[i];
	}
	if (dim<3) coords1[r.Id()*3+2] = 0;
    }
    // cells
    num_cells = x1.GetMesh().Cells::size();
    ids1 = new int[num_cells*4];
    int i = 0;
    for (cell c=x1.GetMesh().cells(); c!=x1.GetMesh().cells_end(); c++) {
	for (int j=0; j<c.Corners(); j++) {
	    ids1[i*4+j] = x1.Id(c.Corner(j));
	}
	i++;
	Point p1 = c.Corner(0) + Point(x1(c.Corner(0),0),x1(c.Corner(0),1));
	Point p2 = c.Corner(1) + Point(x1(c.Corner(1),0),x1(c.Corner(1),1));
	Point p3 = c.Corner(2) + Point(x1(c.Corner(2),0),x1(c.Corner(2),1));
	Point p4 = c.Corner(3) + Point(x1(c.Corner(3),0),x1(c.Corner(3),1));
	Point p = 0.25 * (p1 + p2 + p3 + p4);
	cells1.push_back(Point(p[0],p[1]));
    }
    // fixed
    fixed1 = new bool[num_vertices];
    for (int i=0; i<num_vertices; i++) fixed1[i] = 0;
    for (bnd_face bf=x1.bnd_faces(); bf!=x1.bnd_faces_end(); bf++) {
	Point left = x1.GetMesh().find_face(bf()).Left();
	cell c = x1.GetMesh().find_cell(left);
	for (int i=0; i<c.Faces(); i++) {
	    if (c.Face(i) == bf()) {
		for (int j=0; j<c.FaceCorners(i); j++) {
		    fixed1[x1.Id(c.FaceCorner(i,j))] = 1;
		}
	    }
	}
    }
    bnds1.clear();
    bndsid1.clear();
    for (bnd_face bf=x1.bnd_faces(); bf!=x1.bnd_faces_end(); bf++) {
	Point left = x1.GetMesh().find_face(bf()).Left();
	cell c = x1.GetMesh().find_cell(left);
	for (int i=0; i<c.Faces(); i++) {
	    if (c.Face(i) == bf()) {
		Point bnd;
		for (int j=0; j<c.FaceCorners(i); j++) {
		    bnd += c.FaceCorner(i,j) + x1(c.FaceCorner(i,j));
		}
		bnd[2] = 0;
		bnd *= 1.0 / c.FaceCorners(i);
		bnds1.push_back(bnd);
		bndsid1.push_back(bf.Part());
	    }
	}
    }
}

void TElastoPlasticityAssemble::MeshBndImport (Meshes& M2) {
    for (int i=0; i<bnds1.size(); i++) {
	hash_map<Point,BoundaryFace,Hash>::iterator it = M2.fine().BoundaryFaces::find(bnds1[i]);
	it->second.SetPart(bndsid1[i]);
    }
}

void TElastoPlasticityAssemble::DispExport (const Vector& x0, const Vector& x1, const Vector& x3) {
    disp0.clear();
    disp1.clear();
    vertices1.clear();
    for (row r = x3.rows(); r != x3.rows_end(); r++) {
	Point d = x3(r);
	d += r();
	vertices1.push_back(Point(d[0],d[1]));
	disp0.push_back(Point(x0(r)[0],x0(r)[1]));
	disp1.push_back(Point(x1(r)[0],x1(r)[1]));
    }
}

void TElastoPlasticityAssemble::DispImport (Vector& x0, Vector& x1) {
    for (int i=0; i<vertices1.size(); i++) {
	x0(vertices1[i],0) = disp0[i][0];
	x0(vertices1[i],1) = disp0[i][1];
	x1(vertices1[i],0) = disp1[i][0];
	x1(vertices1[i],1) = disp1[i][1];
    }
}

void TElastoPlasticityAssemble::PhysicsExport (const Vector& x1, const Vector& s1ud, const Vector& ep1ud, const Vector& st1) {
    cells1.clear();
    strain1ud.clear();
    epsilonp1ud.clear();
    stress1.clear();
    for (cell c = x1.GetMesh().cells(); c != x1.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, x1, c);
	for (int q = 0; q < E.nQ(); q++) {
	    Point p1 = c.Corner(0) + Point(x1(c.Corner(0),0),x1(c.Corner(0),1));
	    Point p2 = c.Corner(1) + Point(x1(c.Corner(1),0),x1(c.Corner(1),1));
	    Point p3 = c.Corner(2) + Point(x1(c.Corner(2),0),x1(c.Corner(2),1));
	    Point p4 = c.Corner(3) + Point(x1(c.Corner(3),0),x1(c.Corner(3),1));
	    Point p = 0.25 * (p1 + p2 + p3 + p4);
	    cells1.push_back(Point(p[0],p[1]));
	    // incremental strain
	    Tensor s = Tensor(s1ud(E.QPoint(q),0),
			      s1ud(E.QPoint(q),1),
			      s1ud(E.QPoint(q),2),
			      s1ud(E.QPoint(q),3),
			      s1ud(E.QPoint(q),4),
			      s1ud(E.QPoint(q),5),
			      s1ud(E.QPoint(q),6),
			      s1ud(E.QPoint(q),7),
			      s1ud(E.QPoint(q),8));
	    strain1ud.push_back(s);
	    // incremental plasticity strain
	    s = Tensor(ep1ud(E.QPoint(q),0),
		       ep1ud(E.QPoint(q),1),
		       ep1ud(E.QPoint(q),2),
		       ep1ud(E.QPoint(q),3),
		       ep1ud(E.QPoint(q),4),
		       ep1ud(E.QPoint(q),5),
		       ep1ud(E.QPoint(q),6),
		       ep1ud(E.QPoint(q),7),
		       ep1ud(E.QPoint(q),8));
	    epsilonp1ud.push_back(s);
	    // stress
	    s = Tensor(st1(E.QPoint(q),0),
		       st1(E.QPoint(q),1),
		       st1(E.QPoint(q),2),
		       st1(E.QPoint(q),3),
		       st1(E.QPoint(q),4),
		       st1(E.QPoint(q),5),
		       st1(E.QPoint(q),6),
		       st1(E.QPoint(q),7),
		       st1(E.QPoint(q),8));
	    stress1.push_back(s);
	}
    }
}

void TElastoPlasticityAssemble::PhysicsImport (Vector& s1ud, Vector& ep1ud, Vector& st1) {
    for (int i=0; i<cells1.size(); i++) {
	Point p(cells1[i][0],cells1[i][1]);
	// incremental strain
	s1ud(p,0) = strain1ud[i][0][0];
	s1ud(p,1) = strain1ud[i][0][1];
	s1ud(p,2) = strain1ud[i][0][2];
	s1ud(p,3) = strain1ud[i][1][0];
	s1ud(p,4) = strain1ud[i][1][1];
	s1ud(p,5) = strain1ud[i][1][2];
	s1ud(p,6) = strain1ud[i][2][0];
	s1ud(p,7) = strain1ud[i][2][1];
	s1ud(p,8) = strain1ud[i][2][2];
	// incremental plastic strain
	ep1ud(p,0) = epsilonp1ud[i][0][0];
	ep1ud(p,1) = epsilonp1ud[i][0][1];
	ep1ud(p,2) = epsilonp1ud[i][0][2];
	ep1ud(p,3) = epsilonp1ud[i][1][0];
	ep1ud(p,4) = epsilonp1ud[i][1][1];
	ep1ud(p,5) = epsilonp1ud[i][1][2];
	ep1ud(p,6) = epsilonp1ud[i][2][0];
	ep1ud(p,7) = epsilonp1ud[i][2][1];
	ep1ud(p,8) = epsilonp1ud[i][2][2];
	// stress
	st1(p,0) = stress1[i][0][0];
	st1(p,1) = stress1[i][0][1];
	st1(p,2) = stress1[i][0][2];
	st1(p,3) = stress1[i][1][0];
	st1(p,4) = stress1[i][1][1];
	st1(p,5) = stress1[i][1][2];
	st1(p,6) = stress1[i][2][0];
	st1(p,7) = stress1[i][2][1];
	st1(p,8) = stress1[i][2][2];
    }	
}

void printvector (Vector& b, int n=2) {
    for (row r=b.rows(); r!=b.rows_end(); r++) {
	mout << r() << ": ";
	for (int i=0; i<n; i++)
	    mout << b(r,i) << " ";
	mout << endl;
    }
    mout << endl;
}

void TElastoPlasticityAssemble::TESolver4 (Meshes& M2, int num, double dT) {
    time_k = dT;
    SetSubDomain(M2.fine());
    if (num==1) SetBoundaryType(M2.fine(), num*dT);
    else MeshBndImport (M2);
	
    int dim = M2.dim();
    Discretization disc(dim);
    MatrixGraphs G(M2, disc);
    Vector gd(G.fine());
    Vector x0(gd);
    Vector x1(gd);
    Vector a2(gd);
    Vector x3(gd);
    Matrix A(gd);
    Vector b(gd);
    A = 0;
    b = 0;
    gd = 0;
    x0 = 0;
    x1 = 0;
    a2 = 0;
    x3 = 0;
	
    int nq = disc.GetQuad((M2.fine()).cells()).size();
    Discretization disc_scalar("cell", 1*nq);
    Discretization disc_tensor("cell", 9*nq);
    MatrixGraphs G_scalar(M2, disc_scalar);
    MatrixGraphs G_tensor(M2, disc_tensor);
    Vector alpha1(G_scalar.fine());
    Vector beta1(G_tensor.fine());
    Vector s1ud(G_tensor.fine());
    Vector ep1ud(G_tensor.fine());
    Vector st1(G_tensor.fine());
    Vector r1(G_scalar.fine());
    alpha1 = 0;
    beta1 = 0;
    s1ud = 0;
    ep1ud = 0;
    st1 = 0;
    r1 = 1;
	
    SetDirichlet(gd, num*dT);
    if (num==1) {
	SetH0(x0);
	SetH1(x1,dT);
	rho0 = 1.0 / M2.fine().Cells::size();
    }
    else {
	DispImport(x0, x1);
	PhysicsImport(s1ud, ep1ud, st1);
    }
    for (row r=gd.rows(); r!=gd.rows_end(); r++) {
	for (int i=0; i<dim; i++) {
	    if (gd.D(r,i)==true) {
		gd(r,i) = (gd(r,i) - 2.0*x1(r,i) + x0(r,i)) / dT / dT;
	    }
	}
    }

    Jacobi4(A, r1);
    Residual4(b, x1, s1ud, ep1ud, st1, alpha1, beta1, num*dT);
	
    Solver S;
    S(A);
    a2 = 0;
    S.multiply_plus(a2, b);
    a2 += gd;

    // **************************************
    // *********  update ********************
	
    for (row r=x3.rows(); r!=x3.rows_end(); r++) {
	for (int i=0; i<dim; i++) {
	    x3(r,i) = dT *dT * a2(r,i) + 2.0 * x1(r,i) - x0(r,i);
	}
    }
    x0 = x1;
    x1 = x3;
	
    Update4(x1, s1ud, ep1ud, st1);

    MeshExport(x1);
    DispExport(x0, x1, x1);
    PhysicsExport(x1, s1ud, ep1ud, st1);



    // **************************************
    // **************************************
	

    if (num%1==0) {
	Plot P(M2.fine());
	char buffer [10];
	sprintf(buffer,"%d",num/1);
	string filename1 = string("telastoplasticity_deform") + buffer;
	string filename2 = string("telastoplasticity_undeform") + buffer;	
	P.vertexdata(x1, dim);
	P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	P.vtk_vertex_vector(filename2.c_str(), 0, 0);
	//string filename3 = string("./data/vtk/smoothing") + buffer + string(".vtk");	
	//Rezoning(coords1,num_vertices,ids1,num_cells,fixed1,filename3);
    }

}

void TElastoPlasticity4Main () {
    Date Start;
	
    double T = 1;
    double nT = 1;
    double dT = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", nT);
    dT = T / nT;
	
    mout << "*************** " << 1 << " ******************" << endl;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    TElastoPlasticityAssemble EPA(M.dim());
    EPA.TESolver4(M, 1, dT);
	
    for (int i=2; i<nT+1; i++) {
	//for (int i=2; i<50; i++) {
	mout << "*************** " << i << " ******************" << endl;
	Meshes M2(EPA.coords1, EPA.num_vertices*3, EPA.ids1, EPA.num_cells*4);
	EPA.TESolver4(M2, i, dT);
    }
	
    tout(1) << Date() - Start << endl;
    return;    
}



