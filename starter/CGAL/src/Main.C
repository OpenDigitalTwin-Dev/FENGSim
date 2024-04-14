// file:   Main.C
// author: Jiping Xin

#include "CGALInterface.h"

int main (int argv, char** argc) {

    CGAL_IncrementalTriangulation_2();
    CGAL_DelaunayTriangulation_2();
    CGAL_ConstrainedDelaunayTriangulation_2();
    CGAL_MeshGeneration_2();

    CGAL_Triangulation_3();
    CGAL_DelaunayTriangulation_3();
    CGAL_MeshGeneration_3 ();
    
    return 0;
}
