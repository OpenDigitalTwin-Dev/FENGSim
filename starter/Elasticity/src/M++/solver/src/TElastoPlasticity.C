#include "TElastoPlasticity.h"
#include "LogarithmicStrain.h"

TElastoPlasticityAssemble::TElastoPlasticityAssemble (int _dim) : disc(Discretization(_dim)) {
    dim = _dim;
    Mu = mu;
    Lambda = lambda;
    for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	    I4.set(i, j, Tensor(i,j));
	}
    }
}

void TElastoPlasticityAssemble::SetH0 (Vector& x0) {
    x0 = 0;
    for (row r = x0.rows(); r != x0.rows_end(); r++) {
	Point p = h0(r());
	for (int i = 0; i < dim; i++) {
	    x0(r, i) = p[i];
	}
    }
}

void TElastoPlasticityAssemble::SetH1 (Vector& x1, double dt) {
    x1 = 0;
    for (row r = x1.rows(); r != x1.rows_end(); r++) {
	Point p = h1(r(), dt);
	for (int i = 0; i < dim; i++) {
	    x1(r, i) = p[i];
	}
    }
}

void TElastoPlasticityAssemble::SetDirichlet (Vector& u, double t) {
    u.ClearDirichletFlags();
    for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
	RowBndValues u_c(u, c);	    
	if (!u_c.onBnd()) continue;
	for (int i = 0; i < c.Faces(); ++i) {
	    if (!IsDirichlet(u_c.bc(i))) continue;
	    VectorFieldElement E(disc, u, c);
	    for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
		int k = disc.NodalPointOnFace(c, i, j);
		Dirichlet(E[k](), t, k, u_c, u_c.bc(i));
	    }
	}
    }
    DirichletConsistent(u);
}

void TElastoPlasticityAssemble::Update (const Tensor epsilon2, const Tensor epsilonp1, const double alpha1, const Tensor beta1,
					Tensor& epsilonp2, double& alpha2, Tensor& beta2, Tensor4& C2, Tensor& sigma2) {    
    Tensor4 J1;
    Tensor4 J2;
    for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	    J1.set(i, j, Tensor(i,j));
	}
	J2.set(i, i, One);
    }
    Tensor4 Dev = J1 - 1.0 / 3.0 * J2;
    Tensor T = 2.0 * Mu * (epsilon2 - epsilonp1) + Lambda * trace(epsilon2 - epsilonp1) * One;
    Tensor S = dev(T);
    
    Tensor N;
    if (norm(S) == 0) N = Zero;
    else N = 1.0 / norm(S) * S;
    
    double d;
    if (norm(S) == 0) d = 0;
    else d = 1.0 / norm(S);
    
    Tensor4 C;
    C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One);
    
    double Y = Mu * h_0 * alpha1 + sqrt(2.0/3.0) * k_0;
    double dGamma = 1.0 / 2.0 / Mu * (norm(S) - Y ) / (1.0 + 0.5 * h_0);
    
    if (norm(S) > Y) {
	epsilonp2 = epsilonp1 + dGamma * N;
	alpha2 = alpha1 + dGamma;
	sigma2 = C * (epsilon2 - epsilonp2);
	C2 = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One)
	    - (2.0 * Mu / (1.0 + 0.5 * h_0)) * DyadicProduct(N, N)
	    - 4.0 * Mu * Mu * dGamma * d * (Dev - DyadicProduct(N, N))
	    ;
    }
    else {
	epsilonp2 = epsilonp1;
	alpha2 = alpha1;
	sigma2 = T;
	C2 = C;
    }
}

void TElastoPlasticityAssemble::Update (Vector& x, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1) {
    for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	VectorFieldElement E(disc, x, c);
	for (int q = 0; q < E.nQ(); q++) {
	    //-----------------------------------------------------------------------------
	    // return mapping
	    // k
	    Tensor epsilon2 = sym(E.VectorGradient(q, x));
	    Tensor epsilonp1 = GetTensorFromVector(c(), EpsilonP1, q);
	    double alpha1 = Alpha1(c(), q);
	    Tensor beta1 = GetTensorFromVector(c(), Beta1, q);
	    // k + 1
	    Tensor epsilonp2;
	    double alpha2;
	    Tensor beta2;
	    Tensor4 C2;
	    Tensor sigma2;
	    Update(epsilon2, epsilonp1, alpha1, beta1,
		   epsilonp2, alpha2, beta2, C2, sigma2);
	    //-----------------------------------------------------------------------------
	    Alpha1(c(), q) = alpha2;
	    SetTensorToVector(beta2, c(), Beta1, q);
	    SetTensorToVector(epsilonp2, c(), EpsilonP1, q);
	}
    }
}

double TElastoPlasticityAssemble::L2Error (const Vector& x) {
    double t = 0;
    for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	VectorFieldElement E(disc,x,c);
	for (int q = 0; q < E.nQ(); q++) {
	    Point p = Solution(E.QPoint(q),0.5);
	    t += (E.VectorValue(q,x) - p) * (E.VectorValue(q,x) - p) * E.QWeight(q);
	}
    }
    t = PPM->Sum(t);
    return sqrt(t);
}

void TElastoPlasticityAssemble::Jacobi (Matrix& A) {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, A, c);
	RowEntries A_c(A, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int j = 0; j < E.size(); ++j) {
		    for (int l = 0; l < dim; ++l) {
			for (int q = 0; q < E.nQ(); q++) {
			    A_c(i,j,k,l) += E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
			}
		    }
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

double TElastoPlasticityAssemble::Residual (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1, double time) {
    b = 0;
    for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, b, c);
	RowValues b_c(b, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    b_c(i, k) += -1.0 * E.VectorValue(q, g_D) * E.VectorValue(q, i, k) * E.QWeight(q);
		}
	    }
	}
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

void TElastoPlasticityMain () {
    Date Start;  
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    TElastoPlasticityAssemble EPA(M.dim());
    EPA.SetSubDomain(M.fine());
    EPA.SetBoundaryType(M.fine());
    
    int dim = M.dim();
    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector g_d(G.fine());
    Vector x0(g_d);
    Vector x1(g_d);
    Vector x2(g_d);
    Vector x3(g_d);
    Matrix A(g_d);
    Vector b(g_d);
    A = 0;
    b = 0;
    g_d = 0;
    x0 = 0; // strange x is not zero ??
    x1 = 0; // strange x is not zero ??
    x2 = 0;
    x3 = 0;
    
    int nq = disc.GetQuad((M.fine()).cells()).size();
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
    
    Plot P(M.fine());
    double T = 1;
    double nT = 1;
    double dT = 0.1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", nT);
    dT = T / nT;
    EPA.time_k = dT;
    
    EPA.SetH0(x0);
    EPA.SetH1(x1,dT);
    for (int i = 0; i < nT + 1; i++) {
        mout << endl;
	mout << "time step: " << i << endl;
	g_d = 0;
	EPA.SetDirichlet(g_d, i*dT);
	for (row r=g_d.rows(); r!=g_d.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		if (g_d.D(r,i)==true) {
		    g_d(r,i) = (g_d(r,i) - 2*x1(r,i) + x0(r,i)) / dT / dT;
		}
	    }
	}
	EPA.Jacobi(A);
	EPA.Residual(b, g_d, x0, x1, epsilonp1, alpha1, beta1, i * dT);
	
	Solver S;
	S(A);
	x2 = 0;
	S.multiply_plus(x2, b);
	x2 += g_d;
	
	for (row r=x3.rows(); r!=x3.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		x3(r,i) = dT*dT*x2(r,i) + 2*x1(r,i) - x0(r,i);
	    }
	}
	x0 = x1;
	x1 = x3;
	
	EPA.Update(x1,epsilonp1,alpha1,beta1);
	
	char buffer [10];
	sprintf(buffer,"%d",i);
	string filename1 = string("telastoplasticity_deform") + buffer;
	string filename2 = string("telastoplasticity_undeform") + buffer;	
	P.vertexdata(x3, dim);
	P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	P.vtk_vertex_vector(filename2.c_str(), 0, 0);
    }
    
    mout << "l2 error: " << EPA.L2Error(x3) << endl;
	
    tout(1) << Date() - Start << endl;
    return;    
}

// *****************************
// mass lumping
// *****************************
void TElastoPlasticityAssemble::Jacobi2 (Matrix& A) {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, A, c);
	RowEntries A_c(A, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int j = 0; j < E.size(); ++j) {
		    for (int l = 0; l < dim; ++l) {
			for (int q = 0; q < E.nQ(); q++) {
			    A_c(i,i,k,k) += E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
			}
		    }
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

// *****************************
// mass lumping
// *****************************
double TElastoPlasticityAssemble::Residual2 (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1, double time) {
    b = 0;
    for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, b, c);
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

void TElastoPlasticity2Main () {
    Date Start;  
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    TElastoPlasticityAssemble EPA(M.dim());
    EPA.SetSubDomain(M.fine());
    EPA.SetBoundaryType(M.fine());
    
    int dim = M.dim();
    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector g_d(G.fine());
    Vector x0(g_d);
    Vector x1(g_d);
    Vector x2(g_d);
    Vector x3(g_d);
    Matrix A(g_d);
    Vector b(g_d);
    A = 0;
    b = 0;
    g_d = 0;
    x0 = 0; // strange x is not zero ??
    x1 = 0; // strange x is not zero ??
    x2 = 0;
    x3 = 0;
    
    int nq = disc.GetQuad((M.fine()).cells()).size();
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
    
    Plot P(M.fine());
    double T = 1;
    double nT = 1;
    double dT = 0.1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", nT);
    dT = T / nT;
    EPA.time_k = dT;
    
    EPA.SetH0(x0);
    EPA.SetH1(x1,dT);
    for (int i = 0; i < nT + 1; i++) {
        mout << endl;
	mout << "time step: " << i << endl;
	g_d = 0;
	EPA.SetDirichlet(g_d, i*dT);
	for (row r=g_d.rows(); r!=g_d.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		if (g_d.D(r,i)==true) {
		    g_d(r,i) = (g_d(r,i) - 2*x1(r,i) + x0(r,i)) / dT / dT;
		}
	    }
	}
	EPA.Jacobi2(A);
	EPA.Residual2(b, g_d, x0, x1, epsilonp1, alpha1, beta1, i * dT);
	
	Solver S;
	S(A);
	x2 = 0;
	S.multiply_plus(x2, b);
	x2 += g_d;

	for (row r=x3.rows(); r!=x3.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		x3(r,i) = dT*dT*x2(r,i) + 2*x1(r,i) - x0(r,i);
	    }
	}
	x0 = x1;
	x1 = x3;
	
	EPA.Update(x1,epsilonp1,alpha1,beta1);
	char buffer [10];
	sprintf(buffer,"%d",i);
	string filename1 = string("telastoplasticity_deform_") + buffer;
	string filename2 = string("telastoplasticity_undeform_") + buffer;
	P.vertexdata(x3, dim);
	//P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	//P.vtk_vertex_vector(filename2.c_str(), 0, 0);
	P.vtk_vertexdata(filename1.c_str(), 100, 1);
    }

    //P.vtk_vertexdata("fengsim_deform",100,1);
    //P.vtk_vertexdata("fengsim_undeform",100,0);
    
    mout << "l2 error: " << EPA.L2Error(x3) << endl;
    
    tout(1) << Date() - Start << endl;
    return;    
}

// *****************************
// total lagrange
// *****************************

void TElastoPlasticityAssemble::Jacobi3 (Matrix& A) {
    A = 0;
    for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, A, c);
	RowEntries A_c(A, E);
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int j = 0; j < E.size(); ++j) {
		    for (int l = 0; l < dim; ++l) {
			for (int q = 0; q < E.nQ(); q++) {
			    A_c(i,i,k,k) += E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
			}
		    }
		}
	    }
	}
    }
    A.ClearDirichletValues();
}

// *****************************
// total lagrange
// *****************************
double TElastoPlasticityAssemble::Residual3 (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1, double time) {
    b = 0;
    for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	VectorFieldElement E(disc, b, c);
	RowValues b_c(b, E);
	// F'(x)(x'-x) + F(x) = 0  : F(x)
	for (int i = 0; i < E.size(); ++i) {
	    for (int k = 0; k < dim; ++k) {
		for (int q = 0; q < E.nQ(); q++) {
		    //-----------------------------------------------------------------------------
		    // return mapping
		    // k
		    Tensor epsilon2 = 0.5 * (E.VectorGradient(q, x1) + transpose(E.VectorGradient(q, x1)) +
					     E.VectorGradient(q, x1) * transpose(E.VectorGradient(q, x1)));
		    Tensor epsilonp1 = GetTensorFromVector(c(), EpsilonP1, q);
		    double alpha1 = Alpha1(c(), q);
		    Tensor beta1 = GetTensorFromVector(c(), Beta1, q);
		    // k + 1
		    Tensor epsilonp2;
		    double alpha2;
		    Tensor beta2;
		    Tensor4 C2;
		    Tensor sigma2;
		    Update(epsilon2, epsilonp1, alpha1, beta1,
			   epsilonp2, alpha2, beta2, C2, sigma2);
		    Tensor e = 0.5 * (E.VectorGradient(q, i, k) + transpose(E.VectorGradient(q, i, k))
				      + E.VectorGradient(q, i, k) * transpose(E.VectorGradient(q, x1))
				      + E.VectorGradient(q, x1) * transpose(E.VectorGradient(q, i, k)));
		    //-----------------------------------------------------------------------------
		    b_c(i, k) += -1.0 * Frobenius(e,sigma2) * E.QWeight(q);
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


void TElastoPlasticity3Main () {
    Date Start;  
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    TElastoPlasticityAssemble EPA(M.dim());
    EPA.SetSubDomain(M.fine());
    EPA.SetBoundaryType(M.fine());
    
    int dim = M.dim();
    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector g_d(G.fine());
    Vector x0(g_d);
    Vector x1(g_d);
    Vector x2(g_d);
    Vector x3(g_d);
    Matrix A(g_d);
    Vector b(g_d);
    A = 0;
    b = 0;
    g_d = 0;
    x0 = 0; // strange x is not zero ??
    x1 = 0; // strange x is not zero ??
    x2 = 0;
    x3 = 0;
    
    int nq = disc.GetQuad((M.fine()).cells()).size();
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
	
    Plot P(M.fine());
    double T = 1;
    double nT = 1;
    double dT = 0.1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", nT);
    dT = T / nT;
    EPA.time_k = dT;
    
    EPA.SetH0(x0);
    EPA.SetH1(x1,dT);
    for (int i = 0; i < nT + 1; i++) {
        mout << endl;
	mout << "time step: " << i << endl;
	g_d = 0;
	EPA.SetDirichlet(g_d, i*dT);
	for (row r=g_d.rows(); r!=g_d.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		if (g_d.D(r,i)==true) {
		    g_d(r,i) = (g_d(r,i) - 2*x1(r,i) + x0(r,i)) / dT / dT;
		}
	    }
	}
	EPA.Jacobi3(A);
	EPA.Residual3(b, g_d, x0, x1, epsilonp1, alpha1, beta1, i * dT);

	Solver S;
	S(A);
	x2 = 0;
	S.multiply_plus(x2, b);
	x2 += g_d;
	
	for (row r=x3.rows(); r!=x3.rows_end(); r++) {
	    for (int i=0; i<dim; i++) {
		x3(r,i) = dT*dT*x2(r,i) + 2*x1(r,i) - x0(r,i);
	    }
	}
	x0 = x1;
	x1 = x3;
	
	if (i%100==0) {
	    char buffer [10];
	    sprintf(buffer,"%d",i/100);
	    string filename1 = string("telastoplasticity_deform") + buffer;
	    string filename2 = string("telastoplasticity_undeform") + buffer;	
	    P.vertexdata(x3, dim);
	    P.vtk_vertex_vector(filename1.c_str(), 0, 1);
	    P.vtk_vertex_vector(filename2.c_str(), 0, 0);
	}
    }
    
    mout << "l2 error: " << EPA.L2Error(x3) << endl;
    
    tout(1) << Date() - Start << endl;
    return;    
}

