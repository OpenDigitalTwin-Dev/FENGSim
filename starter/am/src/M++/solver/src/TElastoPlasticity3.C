#include "TElastoPlasticity.h"
#include "LogarithmicStrain.h"


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

