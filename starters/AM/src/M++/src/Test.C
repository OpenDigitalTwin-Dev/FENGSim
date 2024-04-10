// file:    Test.C
// author:  Christian Wieners
// $Header: /public/M++/src/Test.C,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#include "Quadrature.h"
#include "IO.h"

void TestQuadrature (const char* name) {
    const Quadrature& Q = GetQuadrature(name);
    mout << name << endl << Q;
}
void TestQuadrature() {
    TestQuadrature("Qtet11");
    TestQuadrature("Qint3");
    TestQuadrature("Qtri16");
}
void Test() {
    TestQuadrature();
}
