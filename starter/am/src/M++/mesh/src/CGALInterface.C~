#include <stdio.h>
#include <vector>
#include <fstream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>

// geom properties
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
// triangulation
typedef CGAL::Triangulation_2<K> IncrementalTri;
typedef CGAL::Delaunay_triangulation_2<K> DelaunayTri;
typedef CGAL::Constrained_Delaunay_triangulation_2<K> ConstrainedDelaunayTri;


template<typename T>
void ImportPointCloud (std::vector<T>& v) {
    std::ifstream is;
    is.open(std::string("/home/jiping/OpenDT/M++/Machining/cgal/point_cloud.cin").c_str());
    const int len = 256;
    char L[len];
    while (is.getline(L, len)) {
	double z[2];
	sscanf(L, "%lf %lf", z, z + 1);
	v.push_back(T(z[0], z[1]));
    }
    is.close();
}

template<typename T>
void ImportConstraints (T& DT) {
    typedef typename T::Point Point;
    std::ifstream is;
    is.open(std::string("/home/jiping/OpenDT/M++/Machining/cgal/constraints.cin").c_str());

    const int len = 256;
    char L[len];
    while (is.getline(L,len)) {
	double z[4];
	sscanf(L, "%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
	DT.insert_constraint(Point(z[0],z[1]), Point(z[2],z[3]));
    }
}

void IncrementalTriangulation () {
    std::vector<IncrementalTri::Point> v;
    ImportPointCloud<IncrementalTri::Point>(v);
    IncrementalTri T;
    T.insert(v.begin(),v.end());
    //ExportTriangulation<IncrementalTri>(T);
}

void DelaunayTriangulation() {
    std::vector<DelaunayTri::Point> v;
    ImportPointCloud<DelaunayTri::Point>(v);
    DelaunayTri T;
    T.insert(v.begin(),v.end());
    //ExportTriangulation<DelaunayTri>(T);    
}

void ConstrainedDelaunayTriangulation() {
    std::vector<ConstrainedDelaunayTri::Point> v;
    ImportPointCloud<ConstrainedDelaunayTri::Point>(v);
    ConstrainedDelaunayTri T;
    T.insert(v.begin(),v.end());
    ImportConstraints<ConstrainedDelaunayTri>(T);
    //ExportTriangulation<ConstrainedDelaunayTri>(T);
}

#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>

typedef CGAL::Delaunay_mesh_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;

void CGALMeshGeneration() {
    std::vector<CDT::Point> v;
    ImportPointCloud<CDT::Point>(v);
    CDT T;
    T.insert(v.begin(),v.end());
    ImportConstraints<CDT>(T);
    
    Mesher mesher(T);
    mesher.set_criteria(Criteria(0.125,0.25));
    mesher.refine_mesh();
    CGAL::lloyd_optimize_mesh_2(T,CGAL::parameters::max_iteration_number = 10);
    //ExportTriangulation<CDT>(T);
}

#include <CGAL/algorithm.h>
#include <CGAL/Alpha_shape_2.h>

typedef K::Point_2                                           Point;
typedef K::FT                                                FT;
typedef CGAL::Alpha_shape_vertex_base_2<K>                   asVb;
typedef CGAL::Alpha_shape_face_base_2<K>                     asFb;
typedef CGAL::Triangulation_data_structure_2<asVb,asFb>      asTds;
typedef CGAL::Delaunay_triangulation_2<K,asTds>              asTriangulation_2;
typedef CGAL::Alpha_shape_2<asTriangulation_2>               Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;
typedef Alpha_shape_2::Alpha_shape_vertices_iterator         Alpha_shape_vertices_iterator;

bool FindNext (Point& P3, const Point& P1, const Point& P2, const Alpha_shape_2& A) {
    for (Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(); it != A.alpha_shape_edges_end(); ++it) {
	if (it->first->vertex(it->first->cw(it->second))->point() == P2) {
	    if (it->first->vertex(it->first->ccw(it->second))->point() != P1) {
		P3 = it->first->vertex(it->first->ccw(it->second))->point();
		return true;
	    }
	}
	if (it->first->vertex(it->first->ccw(it->second))->point() == P2) {
	    if (it->first->vertex(it->first->cw(it->second))->point() != P1) {
		P3 = it->first->vertex(it->first->cw(it->second))->point();
		return true;
	    }
	}
    }
    return false;
}

#include <CGAL/Polygon_2_algorithms.h>
bool check_inside(Point pt, Point *pgn_begin, Point *pgn_end, K traits) {
    switch(CGAL::bounded_side_2(pgn_begin, pgn_end, pt, traits)) {
    case CGAL::ON_BOUNDED_SIDE :
	//std::cout << " is inside the polygon.\n";
	return true;
	break;
    case CGAL::ON_BOUNDARY:
	//std::cout << " is on the polygon boundary.\n";
	return true;
	break;
    case CGAL::ON_UNBOUNDED_SIDE:
	//std::cout << " is outside the polygon.\n";
	return false;
	break;
    }
}


void CGALMeshGeneration(double a, double b, Point *pgn_begin, Point *pgn_end, std::vector<Point> pp) {
    std::cout << "  generate a new mesh: (" << a << ", " << b << ")" << std::endl;
    //std::vector<CDT::Point> v;
    //ImportPointCloud<CDT::Point>(v);
    CDT T;
    //T.insert(v.begin(),v.end());

    //ImportConstraints<CDT>(T);

    Mesher mesher(T);

    mesher.set_criteria(Criteria(a, b));
    mesher.refine_mesh();

    CGAL::lloyd_optimize_mesh_2(T,CGAL::parameters::max_iteration_number = 10);

    //ExportTriangulation2<CDT>(T, pgn_begin, pgn_end);
}


