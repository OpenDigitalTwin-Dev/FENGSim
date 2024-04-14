// file:   Newton.C
// author: Christian Wieners
// $Header: /public/M++/src/Newton.C,v 1.3 2007-10-09 14:25:45 mueller Exp $

#include "Newton.h"

Newton::Newton (Solver& s) 
	: NonlinearSolver("Newton"), S(s), suppressLS(0),JacobiUpdate(1) {
    ReadConfig(Settings,"NewtonSuppressFirstLineSearch",suppressLS);
    ReadConfig(Settings,"NewtonJacobiUpdate",JacobiUpdate);
}

void Newton::operator () (const Assemble& A, Vector& u) {
    Start = Date();
    double E, E_0;

    A.Dirichlet(u);
    Vector r(u);
    Matrix J(u);
    d = d_0 = A.Residual(u,r);
    E = E_0 = A.Energy(u);

    double eps = Eps + Red * d;
    double d_previous = d;
    int LS_cnt = 0;

    int JU_cnt = 0; // counts iteration steps without Jacobian update
    Vector c(r);

    for (iter=0; iter<max_iter; ++iter) {
        vout(3) << " Newton: r(" << iter << ")= " << r << endl;
        if (d < eps) { break; }
        vout(1) << " Newton: d(" << iter << ")= " << d << endl;

        // Determines whether Jacobi Matrix is updated
        if ( ( (iter-JU_cnt) >= JacobiUpdate) || (iter==0) ) {
            A.Jacobi(u,J);
            JU_cnt = iter;
            c = S(J) * r;
        }
        else c = S(J,0) * r;

        vout(3) << " Newton: c(" << iter << ")= " << c << endl;
        u -= c;
        vout(5) << " Newton: u-c " << u << endl;
        d = A.Residual(u,r);
        E = A.Energy(u);
        if (d > d_previous) {
            for (int l=1; l<=LS_iter; ++l) {
                if (iter == 0)
                    if (suppressLS) {
                        vout(2) << "  line search suppressed \n";
                        break;
                    }
                vout(1) << "  line search " << l 
                        << ": d("<< iter <<")= " << d << endl;
                c *= 0.5;
                u += c;
                d = A.Residual(u,r);
                E = A.Energy(u);
                if (d < d_previous) break;
            }
        }
        if (d > d_previous) {
            vout(5) << "  Newton: line search unsuccessful." << endl;
            ++LS_cnt;
            if (LS_cnt == 3) {
                vout(1) << "  Newton: too many line searches unsuccessful."
                        << endl;
                iter = max_iter;
            }
        }
        d_previous = d;
    }
    Vout(0) << " " << IterationEnd();
}
