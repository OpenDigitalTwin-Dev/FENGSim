#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues> 

double* LGStrain2 (double* tensor, int fun) {
	Eigen::MatrixXd m(3,3);
	m(0,0) = tensor[0];
	m(0,1) = tensor[1];
	m(0,2) = tensor[2];
	m(1,0) = tensor[3];
	m(1,1) = tensor[4];
	m(1,2) = tensor[5];
	m(2,0) = tensor[6];
	m(2,1) = tensor[7];
	m(2,2) = tensor[8];
	
	Eigen::EigenSolver<Eigen::MatrixXd> solver;
	solver.compute(m);
	double* logstrain = new double[9];
	for (int j=0; j<9; j++) logstrain[j] = 0;
	double s[3][9];
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			for (int k=0; k<3; k++) {
				s[i][j*3+k] = solver.eigenvectors().col(i)(j).real() * solver.eigenvectors().col(i)(k).real();
			}
		}
	}
	int n = 3;
	/*if (solver.eigenvalues()(0).real() != solver.eigenvalues()(1).real())
		n = 2;
	else
	n = 1;*/

	
	  
	for (int i=0; i<n; i++) {
		double t = 1;
		if (fun==0) {
		    t = 0.5*log(abs(solver.eigenvalues()(i).real()));
		}
		else if (fun==1) {
			t = exp(2.0*solver.eigenvalues()(i).real());
		}
		else if (fun==2) {
			t = abs(solver.eigenvalues()(i).real()) - 1.0;
		}
		else if (fun==3) {
			t = exp(0.5*solver.eigenvalues()(i).real());
		}
		else if (fun==4) {
			t = exp(solver.eigenvalues()(i).real());
		}
		for (int j=0; j<9; j++) {
			logstrain[j] += s[i][j] * t;
		}
	}
	return logstrain;
}

double* ExpStrain2 (double* tensor) {
	Eigen::MatrixXd m(3,3);
	m(0,0) = tensor[0];
	m(0,1) = tensor[1];
	m(0,2) = tensor[2];
	m(1,0) = tensor[3];
	m(1,1) = tensor[4];
	m(1,2) = tensor[5];
	m(2,0) = tensor[6];
	m(2,1) = tensor[7];
	m(2,2) = tensor[8];

	Eigen::EigenSolver<Eigen::MatrixXd> solver;
	solver.compute(m);


	
	double* logstrain = new double[9];
	for (int j=0; j<9; j++) logstrain[j] = 0;
	for (int i=0; i<3; i++) {
		double t = exp(2.0*solver.eigenvalues()(i).real());
		double s[9];
		for (int j=0; j<3; j++) {
			for (int k=0; k<3; k++) {
				s[j*3+k] = solver.eigenvectors().col(i)(j).real() * solver.eigenvectors().col(i)(k).real();
			}
		}
		for (int j=0; j<9; j++) {
			logstrain[j] += s[j] * t;
		}
	}
	return logstrain;
}

double* NStrain (double* tensor) {
	Eigen::MatrixXd m(3,3);
	m(0,0) = tensor[0];
	m(0,1) = tensor[1];
	m(0,2) = tensor[2];
	m(1,0) = tensor[3];
	m(1,1) = tensor[4];
	m(1,2) = tensor[5];
	m(2,0) = tensor[6];
	m(2,1) = tensor[7];
	m(2,2) = tensor[8];
	
	Eigen::EigenSolver<Eigen::MatrixXd> solver;
	solver.compute(m);
	double* logstrain = new double[9];
	for (int j=0; j<9; j++) logstrain[j] = 0;
	double s[3][9];
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			for (int k=0; k<3; k++) {
				s[i][j*3+k] = solver.eigenvectors().col(i)(j).real() * solver.eigenvectors().col(i)(k).real();
			}
		}
	}
	int n = 3;
	/*if (solver.eigenvalues()(0).real() != solver.eigenvalues()(1).real())
		n = 2;
	else
	n = 1;*/
	for (int i=0; i<n; i++) {
		double t = solver.eigenvalues()(i).real()-1;
		for (int j=0; j<9; j++) {
			logstrain[j] += s[i][j] * t;
		}
	}
	return logstrain;
}
	
	


