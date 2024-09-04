#include "CGALInterface.h"

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
void CGAL_ImportPointCloud_2 (std::vector<T>& v) {
    std::ifstream is;
    is.open(std::string("./../data/input/point_cloud_2.cin").c_str());
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
void CGAL_ImportConstraints_2 (T& DT) {
    typedef typename T::Point Point;
    std::ifstream is;
    is.open(std::string("./../data/input/constraints_2.cin").c_str());
    const int len = 256;
    char L[len];
    while (is.getline(L,len)) {
	double z[4];
	sscanf(L, "%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
	DT.insert_constraint(Point(z[0],z[1]), Point(z[2],z[3]));
    }
}

template<typename T>
int CGAL_ExportTriangulation_2 (T& Tri, std::string filename) {
    typedef typename T::Face_iterator Face_iterator;
    std::ofstream out(filename.c_str());
    int n = Tri.number_of_faces();
    out<< "# vtk DataFile Version 2.0" << std::endl
       << "Unstructured Grid by M++" << std::endl
       << "ASCII" << std::endl
       << "DATASET UNSTRUCTURED_GRID" << std::endl
       << "POINTS "<< n*3 <<" float" << std::endl;
    for (Face_iterator it = Tri.faces_begin(); it != Tri.faces_end(); ++it) {
	out << it->vertex(0)->point() << " 0" << std::endl;
        out << it->vertex(1)->point() << " 0" << std::endl;
        out << it->vertex(2)->point() << " 0" << std::endl;
    }
    out << "CELLS " << n << " " << 4 * n << std::endl;
    for (int i = 0; i < n; i ++) {
        out << 3 << " " << i*3 << " " << i*3 + 1  << " " << i*3 + 2 << std::endl;  
    }
    out << "CELL_TYPES " << n << std::endl;
    for (int i = 0; i < n; i++) out << "5" << std::endl;
    out.close();
    return 0;
}

void CGAL_IncrementalTriangulation_2 () {
    std::vector<IncrementalTri::Point> v;
    CGAL_ImportPointCloud_2<IncrementalTri::Point>(v);
    IncrementalTri T;
    T.insert(v.begin(),v.end());
    CGAL_ExportTriangulation_2<IncrementalTri>(T, "./../data/output/cgal_2_1.vtk");
}

void CGAL_DelaunayTriangulation_2() {
    std::vector<DelaunayTri::Point> v;
    CGAL_ImportPointCloud_2<DelaunayTri::Point>(v);
    DelaunayTri T;
    T.insert(v.begin(),v.end());
    CGAL_ExportTriangulation_2<DelaunayTri>(T, "./../data/output/cgal_2_2.vtk");    
}

void CGAL_ConstrainedDelaunayTriangulation_2() {
    std::vector<ConstrainedDelaunayTri::Point> v;
    CGAL_ImportPointCloud_2<ConstrainedDelaunayTri::Point>(v);
    ConstrainedDelaunayTri T;
    T.insert(v.begin(),v.end());
    CGAL_ImportConstraints_2<ConstrainedDelaunayTri>(T);
    CGAL_ExportTriangulation_2<ConstrainedDelaunayTri>(T, "./../data/output/cgal_2_3.vtk");
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

void CGAL_MeshGeneration_2() {
    std::vector<CDT::Point> v;
    CGAL_ImportPointCloud_2<CDT::Point>(v);
    CDT T;
    T.insert(v.begin(),v.end());
    CGAL_ImportConstraints_2<CDT>(T);
    
    Mesher mesher(T);
    mesher.set_criteria(Criteria(0.125,0.05));
    mesher.refine_mesh();
    CGAL::lloyd_optimize_mesh_2(T,CGAL::parameters::max_iteration_number = 10);
    CGAL_ExportTriangulation_2<CDT>(T, "./../data/output/cgal_2_4.vtk");

	// pay attention
	// the boundary should also be constraints.

	
	// fracture
	/*
	for (int i=0; i<10; i++) {
		std::cout << 0.1+0.03+0.03*cos(2*3.1415926/10*i) << " " << 0.5+0.03*sin(2*3.1415926/10*i)
				  << " "
				  << 0.1+0.03+0.03*cos(2*3.1415926/10*(i+1)) << " " << 0.5+0.03*sin(2*3.1415926/10*(i+1))
				  << std::endl; 
	}
	for (int i=0; i<11; i++) {
		std::cout << 0.1+0.03 << " " << 0.5
				  << " "
				  << 0.1+0.03+0.03*cos(2*3.1415926/10*i) << " " << 0.5+0.03*sin(2*3.1415926/10*i)
				  << std::endl; 
				  }*/
}

#include <CGAL/Triangulation_3.h>

typedef CGAL::Triangulation_3<K>      Triangulation;
typedef Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
typedef Triangulation::Finite_edges_iterator Finite_edges_iterator;
typedef Triangulation::Finite_facets_iterator Finite_facets_iterator;
typedef Triangulation::Finite_cells_iterator Finite_cells_iterator;
typedef Triangulation::Simplex        Simplex;
typedef Triangulation::Locate_type    Locate_type;
typedef Triangulation::Point          Point;

template<typename T>
void CGAL_ImportPointCloud_3 (std::list<T>& v) {
    std::ifstream is;
    is.open(std::string("./../data/input/point_cloud_3.cin").c_str());
    const int len = 256;
    char L[len];
    while (is.getline(L, len)) {
	double z[3];
	sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
	v.push_back(T(z[0], z[1], z[2]));
    }
    is.close();
}

template<typename T>
void CGAL_ExportTriangulation_3 (T& Tri, std::string filename) {
    typedef typename T::Finite_cells_iterator Finite_cells_iterator;
    std::ofstream out(filename.c_str());
    int n = Tri.number_of_finite_cells();
    out<< "# vtk DataFile Version 2.0" << std::endl
       << "Unstructured Grid by M++" << std::endl
       << "ASCII" << std::endl
       << "DATASET UNSTRUCTURED_GRID" << std::endl
       << "POINTS "<< n*4 <<" float" << std::endl;
    for (Finite_cells_iterator cit = Tri.finite_cells_begin(); cit != Tri.finite_cells_end(); cit++) {
	for (int i=0; i<4; i++) {
	    out << CGAL::to_double(cit->vertex(i)->point().x()) << " "
		<< CGAL::to_double(cit->vertex(i)->point().y()) << " "
		<< CGAL::to_double(cit->vertex(i)->point().z()) << std::endl;
	}
    }
    out << "CELLS " << n << " " << 5 * n << std::endl;
    for (int i = 0; i < n; i ++) {
        out << 4 << " " << i*4 << " " << i*4 + 1  << " " << i*4 + 2  << " " << i*4 + 3 << std::endl;  
    }
    out << "CELL_TYPES " << n << std::endl;
    for (int i = 0; i < n; i++) out << "10" << std::endl;
    out.close();
}

void CGAL_Triangulation_3() {
    std::list<Point> L;
    CGAL_ImportPointCloud_3<Point>(L);
    Triangulation T(L.begin(), L.end());
    CGAL_ExportTriangulation_3<Triangulation>(T, "./../data/output/cgal_3_1.vtk");
}

#include <CGAL/Delaunay_triangulation_3.h>
typedef CGAL::Delaunay_triangulation_3<K> DelaunayTriangulation;

void CGAL_DelaunayTriangulation_3() {
    std::list<Point> L;
    CGAL_ImportPointCloud_3<Point>(L);
    DelaunayTriangulation T(L.begin(), L.end());
    CGAL_ExportTriangulation_3<DelaunayTriangulation>(T, "./../data/output/cgal_3_2.vtk");
}

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>
#include <CGAL/IO/File_medit.h>

typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;
typedef CGAL::Sequential_tag Concurrency_tag;
typedef CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default,Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

using namespace CGAL::parameters;

template <class C3T3>
void CGAL_ExportTriangulation_3_1( const C3T3& c3t3, std::string filename) {
    std::ofstream out(filename.c_str());
    typedef typename C3T3::Triangulation Tr;
    typedef typename C3T3::Facets_in_complex_iterator Facet_iterator;
    typedef typename C3T3::Cells_in_complex_iterator Cell_iterator;
    
    typedef typename Tr::Finite_vertices_iterator Finite_vertices_iterator;
    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Weighted_point Weighted_point;
    
    const Tr& tr = c3t3.triangulation();
    
    out << "# vtk DataFile Version 2.0" << std::endl
	<< "Unstructured Grid by M++" << std::endl
	<< "ASCII" << std::endl
	<< "DATASET UNSTRUCTURED_GRID" << std::endl
	<< "POINTS "<< tr.number_of_vertices() <<" float" << std::endl;
    
    boost::unordered_map<Vertex_handle, int> V;
    int inum = 1;
    for( Finite_vertices_iterator vit = tr.finite_vertices_begin(); vit != tr.finite_vertices_end(); ++vit) {
	V[vit] = inum++;
	Weighted_point p = vit->point();
	out << CGAL::to_double(p.x()) << " "
	    << CGAL::to_double(p.y()) << " "
	    << CGAL::to_double(p.z()) << std::endl;
    }
    
    out << "CELLS " << c3t3.number_of_cells_in_complex() << " " << 5 * c3t3.number_of_cells_in_complex() << std::endl;
    for( Cell_iterator cit = c3t3.cells_in_complex_begin(); cit != c3t3.cells_in_complex_end(); ++cit ) {
	out << 4 << " ";
	for (int i=0; i<4; i++)
	    out << V[cit->vertex(i)]-1 << " ";
	out << std::endl;
    }
    
    out << "CELL_TYPES " << c3t3.number_of_cells_in_complex() << std::endl;
    for (int i = 0; i < c3t3.number_of_cells_in_complex(); i++) out << "10" << std::endl;
    out.close();
} 

void CGAL_MeshGeneration_3 () {
    Polyhedron polyhedron;
    std::ifstream input("./../data/input/point_cloud_3.off");
    input >> polyhedron;
    input.close();
    Mesh_domain domain(polyhedron);
    Mesh_criteria criteria(facet_angle=25,
			   facet_size=0.15,
			   facet_distance=0.001,
    			   cell_radius_edge_ratio=3);
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());
    CGAL_ExportTriangulation_3_1(c3t3, "./../data/output/cgal_3_3.vtk");
    Mesh_criteria new_criteria(cell_radius_edge_ratio=3, cell_size=0.03);
    //CGAL::refine_mesh_3(c3t3, domain, new_criteria, manifold());
    //CGAL_ExportTriangulation_3_1(c3t3, "./../data/output/cgal_3_4.vtk");
}
