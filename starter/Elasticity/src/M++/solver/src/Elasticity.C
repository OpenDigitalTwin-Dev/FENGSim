// file: Elasticity.C
// author: Jiping Xin

#include "m++.h"
#include "Elasticity.h"

void ElasticityAssemble::SetDirichletBC (Vector& u) {
	u = 0;
	for (cell c = u.GetMesh().cells(); c != u.GetMesh().cells_end(); ++c) {	  
	    RowBndValues u_c(u,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i) {
	        if (!IsDirichlet(u_c.bc(i))) continue;
			VectorFieldElement E(disc,u,c);
			for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
				int k = disc.NodalPointOnFace(c,i,j);
				Point p = g_D(k, u_c, u_c.bc(i), E[k]());
				for (int l = 0; l < dim; l++) {
					u_c.D(k,l) = true;
					u_c(k,l) = p[l];
				}
			}
	    }
	}
	DirichletConsistent(u);
}

void ElasticityAssemble::Jacobi (Matrix& A) {
	A = 0;
	for (cell c = A.GetMesh().cells(); c != A.GetMesh().cells_end(); ++c) {
	    VectorFieldElement E(disc,A,c);
	    RowEntries A_c(A,E);
	    for (int i = 0; i < E.size(); ++i) {
	        for (int j = 0; j < E.size(); ++j) {
				for (int k = 0; k < dim; ++k) {
					for (int l = 0; l < dim; ++l) {
						for (int q = 0; q < E.nQ(); q++) {
							A_c(i,j,k,l) += (2 * mu * Frobenius(sym(E.VectorGradient(q,i,k)), sym(E.VectorGradient(q,j,l)))
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

double ElasticityAssemble::Residual (Vector& b, const Vector& u0) {
	b = 0;
	for (cell c = b.GetMesh().cells(); c != b.GetMesh().cells_end(); ++c) {
	    VectorFieldElement E(disc,b,c);
	    RowValues b_c(b,E);
	    // source 
	    for (int i = 0; i < E.size(); ++i) {
	        for (int k = 0; k < dim; ++k) {
				for (int q = 0; q < E.nQ(); q++) {
					b_c(i,k) += E.VectorValue(q,i,k) * f(c.Subdomain(), E.QPoint(q)) * E.QWeight(q);
				}
			}
	    }
	    // dirichlet b.c.
	    for (int i = 0; i < E.size(); ++i) {
	        for (int k = 0; k < dim; ++k) {
				for (int q = 0; q < E.nQ(); q++) {
					b_c(i,k) += -(2 * mu * Frobenius(sym(E.VectorGradient(q,i,k)), sym(E.VectorGradient(q,u0)))
								  + lambda * E.Divergence(q,i,k) * E.Divergence(q, u0)
						) * E.QWeight(q);
				}
			}
	    }
	    // neumann b.c.
	    RowBndValues u_c(b,c);	    
	    if (!u_c.onBnd()) continue;
	    for (int i = 0; i < c.Faces(); ++i) {
	        if (!IsDirichlet(u_c.bc(i)) && u_c.bc(i) != -1) { 
				VectorFieldFaceElement E(disc,b,c,i);
				for (int j = 0; j < disc.NodalPointsOnFace(c,i); ++j) {
					int k = disc.NodalPointOnFace(c,i,j);
					for (int l = 0; l < dim; l++) {
						for (int q = 0; q < E.nQ(); q++) {
							u_c(k,l) += E.VectorValue(q,j,l) * g_N(u_c.bc(i), E.QPoint(q)) * E.QWeight(q);
						}
					}
				}
			}
	    }
	}
	b.ClearDirichletValues();
	Collect(b);
}

double ElasticityAssemble::L2Error (const Vector& x) {
	double t = 0;
	for (cell c = x.GetMesh().cells(); c != x.GetMesh().cells_end(); c++) {
	    VectorFieldElement E(disc,x,c);
	    for (int q = 0; q < E.nQ(); q++) {
	        Point p = u(E.QPoint(q));
			t += (E.VectorValue(q,x) - p) * (E.VectorValue(q,x) - p) * E.QWeight(q);
	    }
	}
	t = PPM->Sum(t);
	return sqrt(t);
}

void ElasticityMain () { 
    Date Start;
    string name = "UnitCube";
    ReadConfig(Settings, "Mesh", name);
    Meshes M(name.c_str());
    int dim = M.dim();
	mout << dim << endl;
    mout << M.fine().Cells::size() << endl;
	
    ElasticityAssemble EA(dim);
    EA.SetSubDomain(M.fine());
    EA.SetBoundaryType(M.fine());

    Discretization disc(dim);
    MatrixGraphs G(M, disc);
    Vector u_d(G.fine());
    Matrix A(u_d);
    Vector b(u_d);
    Vector x(u_d);

    EA.SetDirichletBC(u_d);

    Start = Date();
    EA.Jacobi(A);
    mout << "assemble matrix: " << Date() - Start << endl;
    Start = Date();
    EA.Residual(b,u_d);
    mout << "assemble vector: " << Date() - Start << endl;
	
    Solver S;
    S(A);
    x = 0;
    S.multiply_plus(x,b);
    x += u_d;
    tout(1) << Date() - Start << endl;
    mout << "L2 error: "<< EA.L2Error(x) << endl;
        
    Plot P(M.fine(),1,6);
    P.vertexdata(x,dim);
    //P.vtk_vertex_vector("fengsim_deform",0,1);
    //P.vtk_vertex_vector("fengsim_undeform",0,0);
    //P.vtk_vertexdata("elasticity_3_deform",dim-1,1);
    //P.vtk_vertexdata("elasticity_3_undeform",dim-1,0);
    P.vtk_vertexdata("fengsim_deform",100,1);
    //P.vtk_vertexdata("fengsim_undeform",100,0);

    Discretization disc2("cell",6);
    MatrixGraphs G2(M, disc2);
    Vector x_strain(G2.fine());
    Vector x_stress(G2.fine());
    x_strain = 0;
    x_stress = 0;
    for (cell c=x.cells(); c!=x.cells_end(); c++) {
		VectorFieldElement E(disc,x,c);
		Tensor T = sym(E.VectorGradient(Point(0.25,0.25,0.25),x));
		// 0 3 4
		// 3 1 5
		// 4 5 2
		x_strain(c(),0) = T[0][0];
		x_strain(c(),1) = T[1][1];
		x_strain(c(),2) = T[2][2];
		x_strain(c(),3) = T[0][1];
		x_strain(c(),4) = T[0][2];
		x_strain(c(),5) = T[1][2];
		Tensor TT = 2.0 * EA.mu * T + EA.lambda * trace(T) * One;
		// 0 3 4
		// 3 1 5
		// 4 5 2
		x_stress(c(),0) = TT[0][0];
		x_stress(c(),1) = TT[1][1];
		x_stress(c(),2) = TT[2][2];
		x_stress(c(),3) = TT[0][1];
		x_stress(c(),4) = TT[0][2];
		x_stress(c(),5) = TT[1][2];
    }
    
    P.celldata(x_strain,6);
    P.vtk_celltensor("fengsim_deform_strain",0,1);
    P.celldata(x_stress,6);
    P.vtk_celltensor("fengsim_deform_stress",0,1);
    
    return;
}



