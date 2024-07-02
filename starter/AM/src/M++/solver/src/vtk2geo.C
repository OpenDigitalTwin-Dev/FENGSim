#include "m++.h"

void vtk2geo () {
    ifstream is("./tetgen-tmpfile.1.vtk");
    ofstream out("./solver/conf/geo/thinwall2.geo"); 
    const int len = 256;
    char L[len];
    for (int i=0; i<5; i++) {
	is.getline(L,len);
    }
    mout << L << endl;
    int n = 0;
    sscanf(L,"%*s %d %*s", &n);
    mout << n << endl;

    out << "POINTS" << endl;
    for (int i=0; i<n; i++) {
	is.getline(L,len);
	double z[3];
	int d = sscanf(L,"%lf %lf %lf",z,z+1,z+2);
	out << z[0] << " "  << z[1] << " "  << z[2] << endl;
    }
    
    is.getline(L,len);
    is.getline(L,len);
    mout << L << endl;
    sscanf(L,"%*s %d %*d", &n);
    mout << n << endl;

    out << "CELLS" << endl;
    for (int i=0; i<n; i++) {
	is.getline(L,len);
	int z[4];
	int d = sscanf(L,"%*d %d %d %d %d",z,z+1,z+2,z+3);
	out << "4 0 " << z[0] << " " << z[1] << " " << z[2] << " " << z[3] << endl;
    }
}
