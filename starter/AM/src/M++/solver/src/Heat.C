// file: Heat.C
// author: Jiping Xin

#include "Heat.h"

void HeatAssemble::SetDirichletBC (Vector& u, double t) {
	u = 0;
	// we need to reset dirichlet flags
	// if u has been done by "DirichletConsistent", the vertices on neumann boundary
	// also is set dirichlet
	u.ClearDirichletFlags();
	for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
		RowBndValues u_c(u, c);	    
		if (!u_c.onBnd()) continue;
		for (int i = 0; i < c.Faces(); ++i)	{
			if (!IsDirichlet(u_c.bc(i))) continue;
			ScalarElement E(disc, u, c);
			for (int j = 0; j < disc.NodalPointsOnFace(c, i); ++j) {
				int k = disc.NodalPointOnFace(c, i, j);
				u_c.D(k) = true;
				u_c(k) = g_D(E[k](), u_c.bc(i), t);
			}
		}
	}
	DirichletConsistent(u);
}

void HeatAssemble::SetInitialCondition (Vector& x1) {
	x1 = 0;
	for (row r = x1.rows(); r != x1.rows_end(); r++) {
	    x1(r,0) = v(r());
	}
}

void HeatAssemble::AssembleMatrix (Matrix& A) const {
	A = 0;
	for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc, A, c);
	    RowEntries A_c(A, E);
	    for (int i = 0; i < E.size(); ++i) {
			for (int j = 0; j < E.size(); ++j) {
				for (int q = 0; q < E.nQ(); q++) {
					A_c(i,j) += (
						E.Value(q, i) * E.Value(q, j) + dt * E.Derivative(q, i) * E.Derivative(q, j)
						) * E.QWeight(q);
				}
			}
	    }
	}
	A.ClearDirichletValues();
}

void HeatAssemble::AssembleVector (const Vector& x1, const Vector& g_d, Vector& b, double t) {
	b = 0;
	for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc, b, c);
	    RowValues b_c(b, E);
	    for (int i = 0; i < E.size(); ++i) {
			for (int q = 0; q < E.nQ(); q++) {
				b_c(i) += dt * E.Value(q, i) * f(E.QPoint(q), t) * E.QWeight(q);
			}
	    }
	    for (int i = 0; i < E.size(); ++i) {
			for (int q = 0; q < E.nQ(); q++) {
				b_c(i) += E.Value(q, i) * E.Value(q, x1) * E.QWeight(q);
			}
	    }
	    for (int i = 0; i < E.size(); ++i) {
			for (int q = 0; q < E.nQ(); q++) {
				b_c(i) += (-1.0) * (
					E.Value(q, i) * E.Value(q, g_d) + dt * E.Derivative(q, i) * E.Derivative(q, g_d)
					) * E.QWeight(q);
			}
	    }
	    // we need to put neumann b.c. at the end, 
	    // because "if (!u_c.onBnd()) continue;"
	    RowBndValues u_c(b, c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i) {
			if (!IsDirichlet(u_c.bc(i))) {
				if (u_c.bc(i) != -1) {
					ScalarFaceElement E(disc, b, c, i);
					for (int j = 0; j < disc.NodalPointsOnFace(c, i); ++j) {
						int k = disc.NodalPointOnFace(c, i, j);
						for (int q = 0; q < E.nQ(); q++) {
							u_c(k) += dt * E.Value(q, j) * g_N(E.QPoint(q), u_c.bc(i), t) * E.QWeight(q);
						}
					}
				}
			}
	    }
	}
	b.ClearDirichletValues();
	Collect(b);
}

double HeatAssemble::L2Error (const Vector& x, double t) {
	double s = 0;
	for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	    ScalarElement E(disc, x, c);
	    for (int q = 0; q < E.nQ(); q++) {
			s += (u(E.QPoint(q), t) - E.Value(q, x)) * (u(E.QPoint(q), t) - E.Value(q, x)) * E.QWeight(q);
	    }
	}
	s = PPM->Sum(s);
	return sqrt(s);
}

void SetMaxMin (Vector x, double& max, double& min) {
    for (row r = x.rows(); r != x.rows_end(); r++) {
		if (x(r,0) > max) max = x(r,0);
		if (x(r,0) < min) min = x(r,0);
    }
}

void HeatMain () {
    Date Start;
    
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
	
    double T = 1;
    double n = 1;
    double l = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", n);
    ReadConfig(Settings, "TimeLevel", l);
    n = pow(2, l);
    double dt = T/n;
    
    HeatAssemble HA(dt);
    HA.SetSubDomain(M.fine());
    HA.SetBoundaryType(M.fine());
    
    Discretization disc;
    MatrixGraphs G(M, disc);
    Vector g_d(G.fine());
    Vector b(g_d);
    Vector x1(g_d);
    Vector x2(g_d);
    Matrix A(g_d);
    double max = -infty;
    double min = infty;
    
    HA.SetInitialCondition(x1);
    HA.SetDirichletBC(g_d, 0);
    
    Plot P(M.fine());
    P.vertexdata(x1);
    P.vtk_vertexdata("heat_0");
    
    Start = Date();
    HA.AssembleMatrix(A);
    mout << "assemble matrix: " << Date() - Start << endl;
	
    Solver S;
    for (int i = 1; i < n+1; i++) {	      
        mout << "time step: " << i << endl;
		HA.SetDirichletBC(g_d, i*dt);
		
		Start = Date();
		HA.AssembleVector(x1, g_d, b, i*dt);
		mout << "assemble vector: " << Date() - Start << endl;
		
		Start = Date();
		S(A);
		x2 = 0;
		S.multiply_plus(x2, b);
		x2 += g_d;
		tout(1) << Date() - Start << endl;
		
		P.vertexdata(x2);
		string filename = string("heat_") + to_string(i);
		P.vtk_vertexdata(filename.c_str());
		
		x1 = x2;
		mout << endl;
    }
    
    mout << endl << "L2 error: " << HA.L2Error(x2, T) << endl;
	SetMaxMin(x2, max, min);
	mout << max << " " << min << endl;
    
    return;
}

