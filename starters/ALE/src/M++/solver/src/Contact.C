#include "Contact.h"
#include "m++.h"
/*#include <CGAL/Simple_cartesian.h>
#include <CGAL/Segment_2.h>
#include <CGAL/intersections.h>
#include <iostream>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;

bool SegmentIntersection (double* x) {
	Point_2 p1(x[0],x[1]), q1(x[2],x[3]);
    Segment_2 s1(p1,q1);
	Point_2 p2(x[4],x[5]), q2(x[6],x[7]);
    Segment_2 s2(p2,q2);

	const auto result = CGAL::intersection(s1, s2);
	if (result) {
		if (const Segment_2* s = boost::get<Segment_2>(&*result)) {
			return true;
		}
		else {
			const Point_2* p = boost::get<Point_2>(&*result);
			if (p->x()==p1[0]&&p->y()==p1[1])
				return false;
			if (p->x()==p2[0]&&p->y()==p2[1])
				return false;
			return true;
		}
	}
	return false;
}
*/

SmallVector multiply2 (const SmallMatrix& A, const SmallVector& x) {
    SmallVector r(A.rows());
    for (int i=0; i<A.rows(); i++) {
        for (int j=0; j<A.cols(); j++) {
	    r[i] += A[i][j]*x[j];
	}
    }
    return r;
}

void SmallCG2 (SmallVector& x, SmallVector& b, SmallMatrix& A, 
	       double epsilon, int kmax) {
    int n = b.size();
    SmallVector r(n);
    r = b - multiply2(A,x);
    double rho0 = r.norm()*r.norm();
    double rho1 = rho0;
    double rho2 = rho0;
    int k = 1;
    SmallVector p(n);
    SmallVector w(n);
    double beta,alpha;
    while (sqrt(rho1)>epsilon*b.norm()&&k<kmax) {
        if (k==1) p = r;
	else {
	    beta = rho1/rho0;
	    p = r + beta*p;
	}
	w = multiply2(A,p);
	alpha = rho1/(p*w);
	x = x + alpha * p;
	r = r - alpha * w;
	rho2 = r.norm()*r.norm();
	rho0 = rho1;
	rho1 = rho2;
	k++;
    }
    mout << "  cg iter. steps: " << k << endl;
}

void ContactMain () {
    double EI = 1e5;
    double omega = 3e5; // ppt value 3e5
    double q = 1e3; // ppt value 1e6
    double delta = 1e-3;
    ReadConfig(Settings, "omega", omega);
    ReadConfig(Settings, "q", q);
    ReadConfig(Settings, "delta", delta);
    mout << "omega = " << omega << endl;
    mout << "q = " << q << endl;
    mout << "delta = " << delta << endl;

    SmallMatrix A(3);
    SmallVector b(3);
    SmallVector x(3);
    A[0][0] = 4.0*EI + omega;
    A[0][1] = 6.0*EI + omega;
    A[0][2] = 8.0*EI + omega;
    A[1][0] = 6.0*EI + omega;
    A[1][1] = 12.0*EI + omega;
    A[1][2] = 18.0*EI + omega;
    A[2][0] = 8.0*EI + omega;
    A[2][1] = 18.0*EI + omega;
    A[2][2] = 144.0/5.0*EI + omega;
    b[0] = 1.0/3.0*q + omega*delta;
    b[1] = 1.0/4.0*q + omega*delta;
    b[2] = 1.0/5.0*q + omega*delta;
    SmallCG2 (x, b, A, 1e-15, 1000);
    mout << x[0] << endl;
    mout << x[1] << endl;
    mout << x[2] << endl;
    mout << "penetr: " << x[0]+x[1]+x[2]-delta << endl;
    
    ofstream out("./data/vtk/contact.vtk");
    int n = 100;
    for (int i=0; i<n+1; i++) {
	double x0 = 1.0/n*i;
	double x1 = 1-(x[0]*x0*x0 + x[1]*x0*x0*x0 + x[2]*x0*x0*x0*x0);
	out << x0 << " " << x1 << " " << 0 << endl;
    }
    out.close();
    
  /*	std::cout << "contact test" << std::endl;
	double x[8];
	x[0] = 0;
	x[1] = 0;
	x[2] = 1;
	x[3] = 1;
	x[4] = 1;
	x[5] = 0;
	x[6] = 0;
	x[7] = 1;
	std::cout << "intersection a point: " << SegmentIntersection(x) << std::endl;
	x[0] = 0;
	x[1] = 0;
	x[2] = 0;
	x[3] = 1;
	x[4] = 0;
	x[5] = -1;
	x[6] = 0;
	x[7] = 2;
	std::cout << "intersection a segment: " << SegmentIntersection(x) << std::endl;
	x[0] = 0;
	x[1] = 0;
	x[2] = 1;
	x[3] = 0;
	x[4] = 0;
	x[5] = 0;
	x[6] = 0;
	x[7] = 1;
	std::cout << "intersection at vertex: " << SegmentIntersection(x) << std::endl;
	x[0] = 0;
	x[1] = 0;
	x[2] = 0;
	x[3] = 1;
	x[4] = 1;
	x[5] = 0;
	x[6] = 1;
	x[7] = 1;
	std::cout << "no intersection: " << SegmentIntersection(x) << std::endl;*/
}
