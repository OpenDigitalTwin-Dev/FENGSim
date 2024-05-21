#include "TElastoPlasticity.h"
//#include "Rezoning.h"
#include "LogarithmicStrain.h"

void TElastoPlasticityAssemble::Update6 (const Vector& _RDisp, Vector& _LStrain, Vector& _CStress) {
	for (cell c = _RDisp.GetMesh().cells(); c != _RDisp.GetMesh().cells_end(); c++) {
		VectorFieldElement E(disc, _RDisp, c);
		for (int q = 0; q < E.nQ(); q++) {
			Tensor RDG = Invert(One - E.VectorGradient(q, _RDisp));
			Tensor B;
			Tensor LS = GetTensorFromVector(c(), _LStrain);
			ExpStrainEigen3(LS, B);
			B = RDG*B*transpose(RDG);
			Tensor S;
			LogStrainEigen3(B, S);
			Tensor4 C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One, One);
			Tensor CS = C * S;
			Tensor tau = dev(CS);
			double Y = sqrt(2.0/3.0) * k_0;
			if (norm(tau) > Y) {
				double dGamma = 1.0 / 2.0 / Mu * (norm(tau) - Y ) / (1.0 + 0.5 * h_0);
				Tensor SS = S - dGamma * tau / norm(tau);
				Tensor CSS = C * SS;
				SetTensorToVector(CSS, c(), _CStress);
				SetTensorToVector(SS, c(), _LStrain);
			}
			else {
				SetTensorToVector(CS, c(), _CStress);
				SetTensorToVector(S, c(), _LStrain);
			}
		}
	}
}

void TElastoPlasticityAssemble::Jacobi6 (Matrix& A) {
	A = 0;
	for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
		VectorFieldElement E(disc, A, c);
		RowEntries A_c(A, E);
		for (int i = 0; i < E.size(); ++i) {
			for (int k = 0; k < dim; ++k) {
				for (int j = 0; j < E.size(); ++j) {
					for (int l = 0; l < dim; ++l) {
						for (int q = 0; q < E.nQ(); q++) {
							A_c(i,i,k,k) += rho0 / E.Area() *
								E.VectorValue(q, i, k) * E.VectorValue(q, j, l) * E.QWeight(q);
						}
					}
				}
			}
		}
	}
	A.ClearDirichletValues();
}

double TElastoPlasticityAssemble::Residual6 (Vector& b, Vector& CS, double time) {
	b = 0;
	for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
		VectorFieldElement E(disc, b, c);
		RowValues b_c(b, E);
		for (int i = 0; i < E.size(); ++i) {
			for (int k = 0; k < dim; ++k) {
				for (int q = 0; q < E.nQ(); q++) {
					//-----------------------------------------------------------------------------
					// return mapping
					// k
					Tensor sigma2 = GetTensorFromVector(E.QPoint(q), CS);
					//-----------------------------------------------------------------------------
					b_c(i, k) += -1.0 * Frobenius(sym(E.VectorGradient(q, i, k)), sigma2) * E.QWeight(q);
				}
			}
		}
		// F'(x)(x'-x) + F(x) = 0  : F(x)
		for (int i = 0; i < E.size(); ++i) {
			for (int k = 0; k < dim; ++k) {
				for (int q = 0; q < E.nQ(); q++) {
					b_c(i, k) += E.VectorValue(q, i, k) * Source(E.QPoint(q), time) * E.QWeight(q);
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
							u_c(k, l) += E.VectorValue(q, j, l) * Neumann(E.QPoint(q), time, u_c.bc(i)) * E.QWeight(q);
						}
					}
				}			
			}
		}
	}
	b.ClearDirichletValues();
	Collect(b);
}

void TElastoPlasticityAssemble::TESolver6 (Meshes& M2, int num, double dT) {  
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
	Vector x2(gd);
	Vector x3(gd);
	Matrix A(gd);
	Vector b(gd);
	A = 0;
	b = 0;
	gd = 0;
	x0 = 0;
	x1 = 0;
	x2 = 0;
	x3 = 0;
	Vector TDisp(gd);
	Vector RDisp(gd);
	TDisp = 0;
	RDisp = 0;
	
	int nq = disc.GetQuad((M2.fine()).cells()).size();
	Discretization disc_scalar("cell", 1*nq);
	Discretization disc_vector("cell", 3*nq);
	Discretization disc_tensor("cell", 9*nq);
	MatrixGraphs G_scalar(M2, disc_scalar);
	MatrixGraphs G_vector(M2, disc_vector);
	MatrixGraphs G_tensor(M2, disc_tensor);
	Vector LStrain(G_tensor.fine());
	Vector CStress(G_tensor.fine());
	LStrain = 0;
	CStress = 0;
	
	// ***********************************************************
	// *****  update configuration *******************************	
	SetDirichlet(gd, num*dT);
	if (num==1) {
		SetH0(x0);
		SetH1(x1, dT);
		rho0 = 1.0 / M2.fine().Cells::size();
		TDisp = -1.0 * x0;
		RDisp = -1.0 * x0;
		Update6(RDisp, LStrain, CStress);
	}
	else {
		DispImport(x0, x1);
		PhysicsImport5(LStrain);		
		PhysicsImport6(TDisp, RDisp);
		Update6(RDisp, LStrain, CStress);
	}
	for (row r=gd.rows(); r!=gd.rows_end(); r++) {
		for (int i=0; i<dim; i++) {
			if (gd.D(r,i)==true) {
				gd(r,i) = (gd(r,i) - 2*x1(r,i) + x0(r,i)) / dT / dT;
			}
		}
	}
	// ***********************************************************
	// **********  solver ******** *******************************
	
	Jacobi6(A);
	Residual6(b, CStress, num*dT);
	
	Solver S;
	S(A);
	x2 = 0;
	S.multiply_plus(x2, b);
	x2 += gd;
	
	for (row r=x3.rows(); r!=x3.rows_end(); r++) {
		for (int i=0; i<dim; i++) {
			x3(r,i) = dT*dT*x2(r,i) + 2*x1(r,i) - x0(r,i);
		}
	}
	
	// ***********************************************************
	// *****  update configuration *******************************	
	
	// hypla don't use dynamic explicit, very smart for me !!!
	if (!ngeom) {
		x0 = x1;
		x1 = x3;
	}
	else {
		x0 = -1.0*x3;
		x1 = 0;
	}

	// **************************************
	// *********  plot  *********************	
	int step = 50;
	if (num%step==0) {
		Plot P(M2.fine());
		char buffer [10];
		sprintf(buffer,"%d",num/step);
		string filename1 = string("telastoplasticity_deform") + buffer;
		string filename2 = string("telastoplasticity_undeform") + buffer;
		if (!ngeom) {
			P.vertexdata(TDisp, dim);
		}
		else {
			P.vertexdata(TDisp, dim);
		}
		P.vtk_vertex_vector(filename1.c_str(), 0, 1);
		P.vtk_vertex_vector(filename2.c_str(), 0, 0);
	}
	// *********  plot  *********************
	
	for (row r=x3.rows(); r!=x3.rows_end(); r++) {
		if (!ngeom) {
			TDisp(r,0) = x3(r,0);
			TDisp(r,1) = x3(r,1);
		}
		else {
			TDisp(r,0) += x3(r,0);
			TDisp(r,1) += x3(r,1);
		}
		RDisp(r,0) = x3(r,0);
		RDisp(r,1) = x3(r,1);
	}
	
	if (!ngeom) x3=0;
	
	MeshExport(x3);
	DispExport(x0, x1, x3);
	PhysicsExport5(LStrain);
	PhysicsExport6(TDisp, RDisp);
}

void TElastoPlasticity6Main () {
	Date Start;	
	double T = 1;
    double nT = 1;
    double dT = 1;
    ReadConfig(Settings, "Time", T);
    ReadConfig(Settings, "TimeSteps", nT);
    dT = T / nT;

	int ng = 0;
	ReadConfig(Settings, "NGeom", ng);
	
	mout << "*************** " << 1 << " ******************" << endl;
	string name = "UnitCube";
	ReadConfig(Settings, "Mesh", name);
	Meshes M(name.c_str());
	TElastoPlasticityAssemble EPA(M.dim());
	EPA.SetNGeom(ng);
	EPA.TESolver6(M, 1, dT);
	
	for (int i=2; i<nT+1; i++) {
		mout << "*************** " << i << " ******************" << endl;
		Meshes M2(EPA.coords1, EPA.num_vertices*3, EPA.ids1, EPA.num_cells*4);
		EPA.TESolver6(M2, i, dT);
	}
    tout(1) << Date() - Start << endl;
    return;    
}
