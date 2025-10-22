// file: ThermoElasticity.C
// author: Jiping Xin

#include "m++.h"
#include "ThermalElastoPlasticityProblems.h"
#include "TElastoPlasticity.h"

class ThermalElastoPlasticityAssemble_T : public ThermalElastoPlasticityProblems_T {
    Discretization disc;
    double dt;
public:
    ThermalElastoPlasticityAssemble_T () {}
    void set_dt (double _dt) {
	dt = _dt;
    }
    void SetDirichletBC (Vector& u, double time) {
        u = 0;
	// we need to reset dirichlet flags
	// if u has been done by "DirichletConsistent", the vertices on neumann boundary
	// also is set dirichlet
	u.ClearDirichletFlags();
	for (cell c=u.GetMesh().cells(); c!=u.GetMesh().cells_end(); ++c) {	  
	    RowBndValues u_c(u,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i=0; i<c.Faces(); ++i) {
		if (!IsDirichlet(u_c.bc(i))) continue;
		ScalarElement E(disc,u,c);
		for (int j=0; j<disc.NodalPointsOnFace(c,i); ++j) {
		    // j is the id for nodal point on face
		    // k is the id for nodal point on cell
		    int k = disc.NodalPointOnFace(c,i,j);
		    g_D(k,u_c,u_c.bc(i),E[k](),time);
		}
	    }
	}
	DirichletConsistent(u);
    }
    void SetInitialCondition (Vector& x0) {
	x0 = 0;
	for (row r=x0.rows(); r!=x0.rows_end(); r++) {
	    x0(r,0) = u0(r());
	}
    }
    void AssembleMatrix (Matrix& A, double time) {
        A = 0;
	for (cell c=A.GetMesh().cells(); c!=A.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc,A,c);
	    RowEntries A_c(A,E);
	    for (int i=0; i<E.size(); ++i) {
		for (int j=0; j<E.size(); ++j) {
		    for (int q=0; q<E.nQ(); q++) {
		        A_c(i,j) += (E.Value(q,i) * E.Value(q,j) + a(c.Subdomain(),E.QPoint(q),time) * dt * E.Derivative(q,i) * E.Derivative(q,j)) * E.QWeight(q);
		    }
		}
	    }
	}
	A.ClearDirichletValues();
    }
    void AssembleVector (Vector& b, const Vector& x0, const Vector& g_d, double time) {
	b = 0;
	for (cell c=b.GetMesh().cells(); c!=b.GetMesh().cells_end(); ++c) {
	    ScalarElement E(disc,b,c);
	    RowValues b_c(b,E);
	    // source
	    for (int i=0; i<E.size(); ++i) {
		for (int q=0; q<E.nQ(); q++) {
		    b_c(i) += dt * E.Value(q,i) * f(c.Subdomain(),E.QPoint(q),time) * E.QWeight(q);
		}
	    }
	    // initial
	    for (int i=0; i<E.size(); ++i) {
		for (int q=0; q<E.nQ(); q++) {
		    b_c(i) += E.Value(q,i) * E.Value(q,x0) * E.QWeight(q);
		}
	    }
	    // dirichlet
	    for (int i=0; i<E.size(); ++i) {
		for (int q=0; q<E.nQ(); q++) {
		    b_c(i) += (-1.0) * (E.Value(q,i) * E.Value(q,g_d) + a(c.Subdomain(),E.QPoint(q),time) * dt * E.Derivative(q,i) * E.Derivative(q,g_d)) * E.QWeight(q);
		}
	    }
	    // neumann
	    // we need to put neumann b.c. at the end, 
	    // because "if (!u_c.onBnd()) continue;"
	    RowBndValues u_c(b,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i=0; i<c.Faces(); ++i) {
		if (!IsDirichlet(u_c.bc(i))) {
		    if (u_c.bc(i)!=-1) {
			ScalarFaceElement E(disc,b,c,i);
			for (int j=0; j<disc.NodalPointsOnFace(c,i); ++j) {
			    int k = disc.NodalPointOnFace(c,i,j);
			    for (int q=0; q<E.nQ(); q++) {
				u_c(k) += dt * E.Value(q,j) * g_N(u_c.bc(i),E.QPoint(q),time) * E.QWeight(q);
			    }
			}
		    }
		}
	    }
	}
	b.ClearDirichletValues();
	Collect(b);
    }
    double L2Error (const Vector& x, double time) {
	double s = 0;
	for (cell c=x.GetMesh().cells(); c!=x.GetMesh().cells_end(); c++) {
	    ScalarElement E(disc,x,c);
	    for (int q=0; q<E.nQ(); q++) {
		s += (u(E.QPoint(q),time) - E.Value(q,x)) * (u(E.QPoint(q),time) - E.Value(q,x)) * E.QWeight(q);
	    }
	}
	s = PPM->Sum(s);
	return sqrt(s);
    }
};

class ThermalElastoPlasticityAssemble : public TElastoPlasticityAssemble {
    Discretization disc_t;
public:
    ThermalElastoPlasticityAssemble (int _dim) : TElastoPlasticityAssemble(_dim) {}
    double Residual3 (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1, double time, const Vector& x1_t) {
	b = 0;
	for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	    VectorFieldElement E(disc, b, c);
	    ScalarElement E_t(disc_t, x1_t, c);
	    RowValues b_c(b, E);
	    // F'(x)(x'-x) + F(x) = 0  : F(x)
	    for (int i = 0; i < E.size(); ++i) {
		for (int k = 0; k < dim; ++k) {
		    for (int q = 0; q < E.nQ(); q++) {
			//-----------------------------------------------------------------------------
			// return mapping
			// k
			Tensor epsilon2 = sym(E.VectorGradient(q, x1));
			Tensor epsilonp1 = GetTensorFromVector(c(), EpsilonP1, q);
			double alpha1 = Alpha1(c(), q);
			Tensor beta1 = GetTensorFromVector(c(), Beta1, q);
			// k + 1
			Tensor epsilonp2;
			double alpha2;
			Tensor beta2;
			Tensor4 C2;
			Tensor sigma2;
			//Update(epsilon2, epsilonp1, alpha1, beta1, epsilonp2, alpha2, beta2, C2, sigma2);
			Tensor4 C;
			C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One);
			sigma2 = C * (epsilon2 - epsilonp1);
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
			b_c(i,k) += Frobenius(sym(E.VectorGradient(q, i, k)), One) * (3.0 * Lambda + 2.0 * Mu) * E_t.Value(q,x1_t) * E.QWeight(q);
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
};

void ThermalElastoPlasticityMain () { 
    Date Start;
    // mesh
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    mout << "cells: " << M.fine().Cells::size() << endl;
    int dim = M.dim();
    ThermalElastoPlasticityProblems TEP;
    TEP.SetSubDomain(M.fine());
    TEP.SetBoundaryType(M.fine());

    // time interval
    double T = 1;
    double n = 1;
    double l = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", n);
    ReadConfig(Settings, "TimeLevel", l);
    //n = pow(2,l);
    double dt = T/n;

    // ************************************************************
    // heat equation
    // ************************************************************
    
    // heat equation
    ThermalElastoPlasticityAssemble_T AT;
    AT.set_dt(dt);

    // linear algebra heat equation
    Discretization disc_t;
    MatrixGraphs G_t(M,disc_t);
    Vector g_t(G_t.fine());
    Vector b_t(g_t);
    Vector x0_t(g_t);
    Vector x1_t(g_t);
    Matrix A_t(g_t);
    
    // initial and boundary conditions for heat equation
    AT.SetInitialCondition(x0_t);

    Plot P(M.fine());

    // ************************************************************
    // elastoplasticity equation
    // ************************************************************
    
    ThermalElastoPlasticityAssemble TEPA(M.dim());
    
    Discretization disc_d(dim);
    MatrixGraphs G_d(M, disc_d);
    Vector g_d(G_d.fine());
    Vector x0_d(g_d);
    Vector x1_d(g_d);
    Vector x2_d(g_d);
    Vector x3_d(g_d);
    Matrix A_d(g_d);
    Vector b_d(g_d);
    A_d = 0;
    b_d = 0;
    g_d = 0;
    x0_d = 0; // strange x is not zero ??
    x1_d = 0; // strange x is not zero ??
    x2_d = 0;
    x3_d = 0;
    
    int nq = disc_d.GetQuad((M.fine()).cells()).size();
    Discretization disc_scalar("cell", 1*nq);
    Discretization disc_tensor("cell", 9*nq);
    MatrixGraphs G_scalar(M, disc_scalar);
    MatrixGraphs G_tensor(M, disc_tensor);
    Vector epsilonp1(G_tensor.fine());
    Vector alpha1(G_scalar.fine());
    Vector beta1(G_tensor.fine());
    epsilonp1 = 0;
    alpha1 = 0;
    beta1 = 0;    
    
    TEPA.time_k = dt;    
    TEPA.SetH0(x0_d);
    TEPA.SetH1(x1_d,dt);

    for (int i=1; i<n+1; i++) {
        mout << "time step: " << i << endl;
	
	// ************************************************************
	// heat equation
	// ************************************************************

	AT.SetDirichletBC(g_t,i*dt);

	// assemble matrix
	Start = Date();
	AT.AssembleMatrix(A_t,i*dt);
	mout << "assemble matrix: " << Date() - Start << endl;

	// assemble vector
	Start = Date();
	AT.AssembleVector(b_t,x0_t,g_t,i*dt);
	mout << "assemble vector: " << Date() - Start << endl;

        // solve linear equations
	Start = Date();
	Solver S;
	S(A_t);
	x1_t = 0;
	S.multiply_plus(x1_t,b_t);
	x1_t += g_t;
	tout(1) << Date() - Start << endl;
	
	x0_t = x1_t;
	
        // plot
	P.vertexdata(x1_t);
	char buffer [10];
	sprintf(buffer,"%d",i);
	string filename = string("heat_") + buffer;
	P.vtk_vertexdata(filename.c_str());

	
	// ************************************************************
	// elastoplasticity equation
	// ************************************************************

	
	mout << endl;
	mout << "time step: " << i << endl;
	g_d = 0;
	TEPA.SetDirichlet(g_d, i*dt);
	for (row r=g_d.rows(); r!=g_d.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		if (g_d.D(r,i)==true) {
		    g_d(r,i) = (g_d(r,i) - 2*x1_d(r,i) + x0_d(r,i)) / dt / dt;
		}
	    }
	}
	// assemble matrix
	TEPA.Jacobi2(A_d);
	// assemble vector
	TEPA.Residual3(b_d, g_d, x0_d, x1_d, epsilonp1, alpha1, beta1, i*dt, x1_t);
	
	Solver S_d;
	S_d(A_d);
	x2_d = 0;
	S_d.multiply_plus(x2_d, b_d);
	x2_d += g_d;
	
	for (row r=x3_d.rows(); r!=x3_d.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		x3_d(r,i) = dt*dt*x2_d(r,i) + 2*x1_d(r,i) - x0_d(r,i);
	    }
	}
	x0_d = x1_d;
	x1_d = x3_d;
	
	TEPA.Update(x1_d, epsilonp1, alpha1, beta1);
	
	string filename1 = string("telastoplasticity_deform_") + buffer;
	string filename2 = string("telastoplasticity_undeform_") + buffer;	
	P.vertexdata(x3_d, dim);
	P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	P.vtk_vertex_vector(filename2.c_str(), 0, 0);
    }
    mout << "L2 error: "<< AT.L2Error(x1_t,T) << endl;
    mout << "l2 error: " << TEPA.L2Error(x3_d) << endl;

    return;
}

