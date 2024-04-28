// file: Heat.C
// author: Jiping Xin

#include "m++.h"
#include "PoissonProblem.h"
#include "ThermoElasticityBVP.h"

class PoissonAssemble : public PoissonProblems {
    Discretization disc;
public:
    PoissonAssemble () {}
    void SetDirichletValues (Vector& u) {
	u = 0;
	for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
	    RowBndValues u_c(u,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i)	{
	        if (!IsDirichlet(u_c.bc(i))) continue;
		ScalarElement E(disc,u,c);
		for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
		    int k = disc.NodalPointOnFace(c,i,j);
		    u_c.D(k) = true;
		    u_c(k) = Dirichlet(E[k](),u_c.bc(i));
		}
	    }
	}
	DirichletConsistent(u);
    }
    void AssembleMatrix (Matrix& A) const {
	A = 0;
	for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc,A,c);
	    RowEntries A_c(A,E);
	    for (int i = 0; i < E.size(); ++i) {
		for (int j = 0; j < E.size(); ++j) {
		    for (int q = 0; q < E.nQ(); q++) {
			A_c(i,j) += Coefficient(E.QPoint(q)) * E.Derivative(q,i) * E.Derivative(q,j) * E.QWeight(q);
		    }
		}
	    }
	}
	A.ClearDirichletValues();
    }
    void AssembleVector (const Vector& u, Vector& b) {
	b = 0;
	for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc,u,c);
	    RowValues b_c(b,E);
	    // source
	    for (int i = 0; i < E.size(); ++i) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i) += E.Value(q,i) * Source(E.QPoint(q)) * E.QWeight(q);
		}
	    }
	    // dirichlet
	    for (int i = 0; i < E.size(); ++i) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i) += (-1.0) * Coefficient(E.QPoint(q)) * E.Derivative(q,i) * E.Derivative(q,u) * E.QWeight(q);
		}
	    }
	    // neumann
	    RowBndValues u_c(b,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i) {
	        if (!IsDirichlet(u_c.bc(i)) && u_c.bc(i) != -1) {
		    ScalarFaceElement E(disc,u,c,i);
		    for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
			int k = disc.NodalPointOnFace(c,i,j);
			for (int q = 0; q < E.nQ(); q++) {
			    u_c(k) += E.Value(q,j) * Neumann(E.QPoint(q),u_c.bc(i)) * E.QWeight(q);
			}
		    }
		}
	    }
	}
	b.ClearDirichletValues();
	Collect(b);
    }
    double L2Error (const Vector& x) {
	double t = 0;
	for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	    ScalarElement E(disc,x,c);
	    for (int q = 0; q < E.nQ(); q++) {
		t += pow(Solution(E.QPoint(q)) - E.Value(q,x),2) * E.QWeight(q);
	    }
	}
	double s = PPM->Sum(t);
	return sqrt(s);
    }
    void vtk_derivative (const char* name, const Vector& x) {
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
};

class ThermoElasticityAssemble_D : public ThermoElasticityBVP_D {
    Discretization disc;
    Discretization disc_t;
    int dim;
public:
    ThermoElasticityAssemble_D (int _dim) : disc(Discretization(_dim)) {
        dim = _dim;
    }
    void DirichletBC (Vector& u, double t=0) {
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
		    DirichletValue(k, u_c, u_c.bc(i), E[k]());
		}
	    }
	}
	DirichletConsistent(u);
    }
    void Jacobi (Matrix& A, double t=0) {
        A = 0;
	for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	    VectorFieldElement E(disc,A,c);
	    RowEntries A_c(A,E);
	    for (int i = 0; i < E.size(); ++i) {
	        for (int j = 0; j < E.size(); ++j) {
		    for (int k = 0; k < dim; ++k) {
		        for (int l = 0; l < dim; ++l) {
			    for (int q = 0; q < E.nQ(); q++) {
			        A_c(i,j,k,l) += (2.0 * mu * Frobenius(sym(E.VectorGradient(q,i,k)),sym(E.VectorGradient(q,j,l)))
						 + lambda * E.Divergence(q,i,k) * E.Divergence(q,j,l)
				    ) * E.QWeight(q);
			    }
			}
		    }
		}
	    }
	}
	A.ClearDirichletValues();
    }
    double Residual (Vector& b, const Vector& u0, Vector& temp, double t) {
	b = 0;
	for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	    VectorFieldElement ED(disc, b, c);
	    ScalarElement ET(disc_t, temp, c);	    
	    RowValues b_c(b, ED);
	    // source
	    for (int i = 0; i < ED.size(); ++i) {
	        for (int k = 0; k < dim; ++k) {
		    for (int q = 0; q < ED.nQ(); q++) {
		        b_c(i,k) += ED.VectorValue(q, i, k) * SourceValue(c.Subdomain(), ED.QPoint(q), t) * ED.QWeight(q);
			double T = ET.Value(q, temp);
			Tensor Temp((2.0*mu+3.0*lambda)*T,0,0,0,(2.0*mu+3.0*lambda)*T,0,0,0,(2.0*mu+3.0*lambda)*T);
			b_c(i,k) += Frobenius(sym(ED.VectorGradient(q,i,k)),Temp)* ED.QWeight(q);
		    }
		}
	    }
	    // dirichlet b.c.
	    for (int i = 0; i < ED.size(); ++i) {
	        for (int k = 0; k < dim; ++k) {
		    for (int q = 0; q < ED.nQ(); q++) {
		        b_c(i,k) += -(2.0 * mu * Frobenius(sym(ED.VectorGradient(q,i,k)),sym(ED.VectorGradient(q,u0)))
				      + lambda * ED.Divergence(q,i,k) * ED.Divergence(q,u0)
			    ) * ED.QWeight(q);
		    }
		}
	    }
	    // neumann b.c.
	    RowBndValues u_c(b, c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i) {
	        if (!IsDirichlet(u_c.bc(i)) && u_c.bc(i) != -1) {
		    VectorFieldFaceElement FED(disc, b, c, i);
		    for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
		        int k = disc.NodalPointOnFace(c,i,j);
			for (int l = 0; l < dim; l++) {
			    for (int q = 0; q < FED.nQ(); q++) {
			        u_c(k,l) += FED.VectorValue(q,j,l) * NeumannValue(u_c.bc(i), FED.QPoint(q), t) * FED.QWeight(q);
			    }
			}
		    }
		}
	    }
	}
	b.ClearDirichletValues();
	Collect(b);
    }
};

void AMSolver (Meshes& M, PathPlanning* pp) {     
    Date Start;
    // mesh
    //string name = "UnitCube";
    //ReadConfig(Settings, "Mesh", name);
    //Meshes M(name.c_str());
    PoissonAssemble L;
    L.pp = pp;
    L.SetDomain(M.fine());
    L.SetBoundary(M.fine());

    // discretization
    Discretization disc;
    MatrixGraphs G(M,disc);
    Vector u(G.fine());
    Vector b(u);
    Vector x(u);
    Matrix A(u);
    // assemble matrix and vector
    L.SetDirichletValues(u);
    Start = Date();
    L.AssembleMatrix(A);
    mout << "assemble matrix: " << Date() - Start << endl;
    Start = Date();
    L.AssembleVector(u,b);
    mout << "assemble vector: " << Date() - Start << endl;
    // solve linear equations
    Solver S;
    S(A);
    x = 0;
    S.multiply_plus(x,b);
    x += u;
    tout(1) << Date() - Start << endl;
    //mout << "L2 error: "<< L.L2Error(x) << endl;
    // plot
    
    Plot P(M.fine());
    P.vertexdata(x);
    //string filename = string("am_temp_") + to_string(pp->cur_index());
    //P.vtk_vertexdata(filename.c_str());

    return;
    
    // elasticity
    ThermoElasticityGeoID GeoID;
    GeoID.SetSubDomain(M.fine());
    GeoID.SetBoundaryType(M.fine());
    int dim = M.dim();
    ThermoElasticityAssemble_D AD(dim);
    Discretization disc_d(dim);
    MatrixGraphs G_d(M,disc_d);
    Vector u_d(G_d.fine());
    Matrix A_d(u_d);
    Vector b_d(u_d);
    Vector x_d(u_d);
    AD.DirichletBC(u_d,0);
    AD.Jacobi(A_d);
    AD.Residual(b_d, u_d, x, 0);	
    S(A_d);
    x_d = 0;
    S.multiply_plus(x_d,b_d);
    x_d += u_d;

    P.vertexdata(x_d,dim);
    //filename = string("am_mesh_") + to_string(pp->cur_index());
    //P.vtk_vertexdata(filename.c_str(),0,1);

    return;
}

void AMMain () {
    Date Start;
    // mesh
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());

    PathPlanning* PP = new PathPlanning;

    // time interval
    double T = 1;
    double n = 1;
    double l = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", n);
    ReadConfig(Settings, "TimeLevel", l);
    //n = pow(2, l);
    double dt = T/n;    
        
    for (int i = 0; i < n+1; i++) {
        mout << "*******************" << endl;
        mout << "time step: " << i << endl;
	mout << "*******************" << endl;
	if (PP->stop(i, dt)) break;
	
	PP->GetPosition(i, dt);
	PP->ExportMesh(M.fine());
	ReadConfig(Settings, "Mesh2", name);
	Meshes M2(name.c_str());

	AMSolver(M2, PP); 
	
	ofstream out("./data/vtk/am_current_pos_" + to_string(i) + ".vtk");
	out << PP->GetPosition(i, dt) << endl;
	out.close();
    }
    return;
}

