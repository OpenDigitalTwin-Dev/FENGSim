// file: Poisson.C
// author: Jiping Xin
#include "Poisson.h"
#include "FASPInterface.h"

void PoissonAssemble::SetDirichletValues (Vector& u) {
    u = 0;
    for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
	RowBndValues u_c(u, c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i)	{
	    if (!IsDirichlet(u_c.bc(i))) continue;
	    ScalarElement E(disc, u, c);
	    for (int j = 0; j < disc.NodalPointsOnFace(c, i); ++j) {
		int k = disc.NodalPointOnFace(c, i, j);
		u_c.D(k) = true;
		u_c(k) = g_D(E[k](), u_c.bc(i));
	    }
	}
    }
    DirichletConsistent(u);
}

void PoissonAssemble::AssembleMatrix (Matrix& A) const {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	ScalarElement E(disc, A, c);
	RowEntries A_c(A, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int j = 0; j < E.size(); ++j) {
		for (int q = 0; q < E.nQ(); q++) {
		    A_c(i,j) += alpha(E.QPoint(q)) * E.Derivative(q, i) * E.Derivative(q, j) * E.QWeight(q);
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

void PoissonAssemble::AssembleVector (const Vector& u_d, Vector& b) {
    b = 0;
    for (cell c = u_d.GetMesh().cells(); c != u_d.GetMesh().cells_end(); ++c) {
	ScalarElement E(disc, u_d, c);
	RowValues b_c(b, E);
	// source 
	for (int i = 0; i < E.size(); ++i) {
	    for (int q = 0; q < E.nQ(); q++) {
		b_c(i) += E.Value(q, i) * f(E.QPoint(q)) * E.QWeight(q);
	    }
	}
	// dirichlet
	for (int i = 0; i < E.size(); ++i) {
	    for (int q = 0; q < E.nQ(); q++) {
		b_c(i) += (-1) * alpha(E.QPoint(q)) * E.Derivative(q, i) * E.Derivative(q, u_d) * E.QWeight(q);
	    }
	}
	// neumann
	RowBndValues u_c(b, c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i) {
	    if (!IsDirichlet(u_c.bc(i)) && u_c.bc(i) != -1) {
		ScalarFaceElement E(disc, u_d, c, i);
		for (int j = 0; j < disc.NodalPointsOnFace(c, i); ++j) {
		    int k = disc.NodalPointOnFace(c, i, j);
		    for (int q = 0; q < E.nQ(); q++) {
			u_c(k) += E.Value(q,j) * g_N(E.QPoint(q), u_c.bc(i)) * E.QWeight(q);
		    }
		}
	    }
	}
    }
    b.ClearDirichletValues();
    Collect(b);
}

double PoissonAssemble::L2Error (const Vector& x) {
    double t = 0;
    for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	ScalarElement E(disc, x, c);
	for (int q = 0; q < E.nQ(); q++) {
	    t += pow(u(E.QPoint(q)) - E.Value(q, x), 2) * E.QWeight(q);
	}
    }
    double s = PPM->Sum(t);
    return sqrt(s);
}

void PoissonAssemble::vtk_derivative (const char* name, const Vector& x) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    ofstream out(filename.c_str()); 
    int n = x.GetMesh().Cells::size();
    out<< "# vtk DataFile Version 2.0" << endl
       << "Unstructured Grid by M++" << endl
       << "ASCII" << endl
       << "DATASET UNSTRUCTURED_GRID" << endl
       << "POINTS "<< n <<" float" << endl;
    for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++)
	out << c()[0] << " " << c()[1] << " " << c()[2] << endl;
    out << "CELLS " << n << " " << 2 * n << endl;
    for (int i = 0; i < n; i++) out << "1 " << i << endl;
    out << "CELL_TYPES " << n << endl;
    for (int i = 0; i < n; i++) out << "1" << endl;
    out << "POINT_DATA" << " " << n << endl;
    out << "VECTORS" << " " << "vectors" << " " << "float" << endl;
    for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	ScalarElement E(disc,x,c);
	Point z;
	for (int i = 0; i < 3; i++) z[i] = (E.Derivative(c(),x))[i];
	out << z[0] << " " << z[1] << " " << z[2] << endl;
    }
}

void PoissonMain () { 
    Date Start;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    
    PoissonAssemble PA;
    PA.SetDomain(M.fine());
    PA.SetBoundary(M.fine());
    
    Discretization disc;
    MatrixGraphs G(M,disc);
    Vector g_d(G.fine());
    Vector b(g_d);
    Vector x(g_d);
    Matrix A(g_d);
    
    PA.SetDirichletValues(g_d);
    
    Start = Date();
    PA.AssembleMatrix(A);
    mout << "assemble matrix: " << Date() - Start << endl;
    
    Start = Date();
    PA.AssembleVector(g_d, b);
    mout << "assemble vector: " << Date() - Start << endl;
    
    x = 0;
    string sol = "CG";
    ReadConfig(Settings,"LinearSolver",sol);
    if (sol=="SuperLU") {
	SparseMatrix _A(A);
	Scalar* _b = b();
	GetSparseSolver(_A,"SuperLU")->Solve(_b,1);
	for (row r=x.rows(); r!=x.rows_end(); r++)
	    x(r,0) = _b[r.Id()];
    }
    else if (sol=="FASP") {
      //FASPInterface(A, b, x);
    }
    else {
	Solver S;
	S(A);
	S.multiply_plus(x, b);
    }
    x += g_d;
    
    tout(1) << Date() - Start << endl;
    
    mout << "L2 error: "<< PA.L2Error(x) << endl;
    
    Plot P(M.fine());
    P.vertexdata(x);
    P.vtk_vertexdata("poisson_linear");
    
    return;
}




