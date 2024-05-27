// file: TElasticity.C
// author: Jiping Xin
// This module is to solve transient linear elasticity problem.

#include "m++.h"
#include "TElasticity.h"

void TElasticityAssemble::SetInitialCondition0 (Vector& x0) {
    x0 = 0;
    for (row r = x0.rows(); r != x0.rows_end(); r++) {
	Point p = h0(r());
	for (int i = 0; i < dim; i++) {
	    x0(r, i) = p[i];
	}
    }
}

void TElasticityAssemble::SetInitialCondition1 (Vector& x1) {
    x1 = 0;
    for (row r = x1.rows(); r != x1.rows_end(); r++) {
	Point p = h1(r(), dt);
	for (int i = 0; i < dim; i++) {
	    x1(r, i) = p[i];
	}
    }
}

void TElasticityAssemble::SetDirichletBC (Vector& u, double t) {
    u = 0;
    u.ClearDirichletFlags();
    for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
	RowBndValues u_c(u,c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i) {
	    if (!IsDirichlet(u_c.bc(i))) continue;
	    VectorFieldElement E(disc,u,c);
	    for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
		int k = disc.NodalPointOnFace(c,i,j);
		g_D(E[k](), t, k, u_c, u_c.bc(i));
	    }
	}
    }
    DirichletConsistent(u);
}

void TElasticityAssemble::Jacobi (Matrix& A) {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc,A,c);
	RowEntries A_c(A,E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int j = 0; j < E.size(); ++j) {
		for (int k = 0; k < dim; ++k) {
		    for (int l = 0; l < dim; ++l) {
			for (int q = 0; q < E.nQ(); q++) {
			    A_c(i, j, k, l) += E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
			    A_c(i, j, k, l) += dt * dt * (2.0 * mu * Frobenius(sym(E.VectorGradient(q, i, k)), sym(E.VectorGradient(q, j, l)))
							  + lambda * E.Divergence(q, i, k) * E.Divergence(q, j, l)
				) * E.QWeight(q);
			}
		    }
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

double TElasticityAssemble::Residual (Vector& b, const Vector& u_d, Vector& x0, Vector& x1, double time) {
    b = 0;
    for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, b, c);
	RowValues b_c(b, E);
	// source 
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i, k) += dt * dt * E.VectorValue(q, i, k) * f(E.QPoint(q), time) * E.QWeight(q);
		}
	    }
	}
	// initial values
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i,k) += 2 * E.VectorValue(q, i, k) * E.VectorValue(q, x1) * E.QWeight(q);
		    b_c(i,k) += -1 * E.VectorValue(q, i, k) * E.VectorValue(q, x0) * E.QWeight(q);
		}
	    }
	}
	// dirichlet boundary condition
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i,k) += -1.0 * E.VectorValue(q, i, k) * E.VectorValue(q, u_d) * E.QWeight(q);
		    b_c(i,k) += -1.0 * dt * dt * (2.0 * mu * Frobenius(sym(E.VectorGradient(q, i, k)),sym(E.VectorGradient(q, u_d)))
						  + lambda * E.Divergence(q, i, k) * E.Divergence(q, u_d)
			) * E.QWeight(q);
		}
	    }
	}
	// neumann boundary condition
	RowBndValues u_c(b,c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i) {
	    if (!IsDirichlet(u_c.bc(i))) {
		if (u_c.bc(i) != -1) {
		    VectorFieldFaceElement E(disc, b, c, i);
		    for (int j = 0; j < disc.NodalPointsOnFace(c, i); ++j) {
			int k = disc.NodalPointOnFace(c, i, j);
			for (int l = 0; l < dim; l++) {
			    for (int q = 0; q < E.nQ(); q++) {
				u_c(k, l) += dt * dt * E.VectorValue(q, j, l) * g_N(E.QPoint(q), time, u_c.bc(i)) * E.QWeight(q);
			    }
			}
		    }
		}
	    }
	}
    }
    b.ClearDirichletValues();
    Collect(b);
}

double TElasticityAssemble::L2Error (const Vector& x0, const Vector& x1, const Vector& x2, double time) {
    double t = 0;
    for (cell c = x2.GetMesh().cells(); c != x2.GetMesh().cells_end(); c++) {
	VectorFieldElement E(disc, x2, c);
	for (int q = 0; q < E.nQ(); q++) {
	    Gradient s = 1.0 / 4.0 * (E.VectorValue(q, x2) + 2 * E.VectorValue(q, x1) + E.VectorValue(q, x0) );
	    t += (u(E.QPoint(q), time) - s) * (u(E.QPoint(q), time) - s) * E.QWeight(q);
	}
    }
    t = PPM->Sum(t);
    return sqrt(t);
}

void TElasticityMain () { 
    Date Start;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    int dim = M.dim();
    
    double T = 1;
    double n = 1;
    double l = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", n);
    ReadConfig(Settings, "TimeLevel", l);
    //n = pow(2, l);
    double dt = T/n;
    
    TElasticityAssemble TEA(dim, dt);
    TEA.SetSubDomain(M.fine());
    TEA.SetBoundaryType(M.fine());
    TEA.time_k = dt;
    
    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector u_d(G.fine());
    Vector x0(u_d);
    Vector x1(u_d);
    Vector x2(u_d);
    Matrix A(u_d);
    Vector b(u_d);
    
    TEA.SetInitialCondition0(x0);
    TEA.SetInitialCondition1(x1);
    
    for (int i = 2; i < n+1; i++) {
        mout << "time step: " << n << " ( " << i << " )" << endl;
	TEA.SetDirichletBC(u_d, i * dt);
	
	Start = Date();
	TEA.Jacobi(A);
	mout << "assemble matrix: " << Date() - Start << endl;
	
	Start = Date();
	TEA.Residual(b, u_d, x0, x1, (i - 1) * dt);
	mout << "assemble vector: " << Date() - Start << endl;
	
	Start = Date();
	Solver S;
	S(A);
	x2 = 0;
	S.multiply_plus(x2, b);
	x2 += u_d;
	tout(1) << Date() - Start << endl;
	
	mout << "L2 error: "<< TEA.L2Error(x0, x1, x2, (i - 1) * dt) << endl;
	
	// plot
	string filename1 = string("telasticity_deform_") + to_string(i);
	string filename2 = string("telasticity_undeform_") + to_string(i);
	Plot P(M.fine());
	P.vertexdata(x2, dim);
	//P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	//P.vtk_vertex_vector(filename2.c_str(), 0, 0);
	P.vtk_vertexdata(filename1.c_str(),100,1);
	if (i == 2) {
	    P.vertexdata(x0, 2);
	    //P.vtk_vertex_vector((string("telasticity_deform_")+to_string(0)).c_str(), 0, 1);
	    //P.vtk_vertex_vector((string("telasticity_undeform")+to_string(0)).c_str(), 0, 0);
	    P.vtk_vertexdata((string("telasticity_deform_")+to_string(0)).c_str(),100,1);
	    P.vertexdata(x1, 2);
	    //P.vtk_vertex_vector((string("telasticity_deform_")+to_string(1)).c_str(), 0, 1);
	    //P.vtk_vertex_vector((string("telasticity_undeform")+to_string(1)).c_str(), 0, 0);
	    P.vtk_vertexdata((string("telasticity_deform_")+to_string(1)).c_str(),100,1);
	}
	
	
	x0 = x1;
	x1 = x2;
    }
    return;
}





