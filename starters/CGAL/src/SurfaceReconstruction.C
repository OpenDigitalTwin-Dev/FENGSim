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

template<typename T>
int ExportTriangulation (T& Tri) {
    typedef typename T::Vertex_iterator Vertex_iterator;
    typedef typename T::Edge_iterator Edge_iterator;
    typedef typename T::Face_iterator Face_iterator;
    // **********************************************************
    std::cout << "  vertices: " << Tri.number_of_vertices() << std::endl;
    std::ofstream out("/home/jiping/OpenDT/M++/Machining/cgal/tri_vertices.cin");
    out.precision(15);
    for (Vertex_iterator it = Tri.vertices_begin(); it != Tri.vertices_end(); ++it) {
        out << it->point() << std::endl;
    }
    out.close();
    // **********************************************************
    /*std::cout << "    faces: " << Tri.number_of_faces() << std::endl;
      out.open("/home/jiping/OpenDT/M++/ElastoPlasticity/cgal/tri_faces.cin");
    for (Face_iterator it = Tri.faces_begin(); it != Tri.faces_end(); ++it) {
	out << it->vertex(0)->point() << std::endl;
        out << it->vertex(1)->point() << std::endl;
        out << it->vertex(2)->point() << std::endl;
    }
    out.close();*/
    // **********************************************************
    std::cout << "  edges: " << std::endl;
    out.open("/home/jiping/OpenDT/M++/Machining/cgal/tri_edges.cin");
    for (Edge_iterator it = Tri.edges_begin(); it != Tri.edges_end(); ++it) {
        out << it->first->vertex(it->first->cw(it->second))->point() << std::endl;
	out << it->first->vertex(it->first->ccw(it->second))->point() << std::endl << std::endl;
    }
    out.close();
    return 0;
}

void IncrementalTriangulation () {
    std::vector<IncrementalTri::Point> v;
    ImportPointCloud<IncrementalTri::Point>(v);
    IncrementalTri T;
    T.insert(v.begin(),v.end());
    ExportTriangulation<IncrementalTri>(T);
}

void DelaunayTriangulation() {
    std::vector<DelaunayTri::Point> v;
    ImportPointCloud<DelaunayTri::Point>(v);
    DelaunayTri T;
    T.insert(v.begin(),v.end());
    ExportTriangulation<DelaunayTri>(T);    
}

void ConstrainedDelaunayTriangulation() {
    std::vector<ConstrainedDelaunayTri::Point> v;
    ImportPointCloud<ConstrainedDelaunayTri::Point>(v);
    ConstrainedDelaunayTri T;
    T.insert(v.begin(),v.end());
    ImportConstraints<ConstrainedDelaunayTri>(T);
    ExportTriangulation<ConstrainedDelaunayTri>(T);
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
    ExportTriangulation<CDT>(T);
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

template<typename T>
int ExportTriangulation2 (T& Tri, Point *pgn_begin, Point *pgn_end) {
    typedef typename T::Face_iterator Face_iterator;
    std::ofstream out;
    out.open("/home/jiping/OpenDT/M++/Machining/cgal/mesh.cin");
    out.precision(4);
    int n = 0;
    for (Face_iterator it = Tri.faces_begin(); it != Tri.faces_end(); ++it) {
	double x = 0;
	double y = 0;
	for (int i = 0; i < 3; i++) {
	    x += it->vertex(i)->point().x();
	    y += it->vertex(i)->point().y();
	}
	x /= 3.0;
	y /= 3.0;
	if (check_inside(Point(x, y), pgn_begin, pgn_end, K())) {
	    out << it->vertex(0)->point() << std::endl;
	    out << it->vertex(1)->point() << std::endl << std::endl;
	    out << it->vertex(1)->point() << std::endl;
	    out << it->vertex(2)->point() << std::endl << std::endl;
	    out << it->vertex(2)->point() << std::endl;
	    out << it->vertex(0)->point() << std::endl << std::endl;
	    n++;
	}
    }
    out.close();
    std::cout << "    triangles num: " << n << std::endl;

    out.open("/home/jiping/OpenDT/M++/Machining/conf/geo/machining.geo");
    out << "POINTS:" << std::endl;
    for (Face_iterator it = Tri.faces_begin(); it != Tri.faces_end(); ++it) {
	double x = 0;
	double y = 0;
	for (int i = 0; i < 3; i++) {
	    x += it->vertex(i)->point().x();
	    y += it->vertex(i)->point().y();



	    


	    
	    
	}
	x /= 3.0;
	y /= 3.0;
	if (check_inside(Point(x, y), pgn_begin, pgn_end, K())) {
	    out << it->vertex(0)->point() << " " << 0 << std::endl;
	    out << it->vertex(1)->point() << " " << 0 << std::endl;
	    out << it->vertex(2)->point() << " " << 0 << std::endl;


/*
	    for (int i = 0; i < 3; i++) {
		if (abs(it->vertex(i)->point().y()) < 1e-10) {
		    if (abs(it->vertex(i)->point().x()) < 1e-10) {
			std::cout << it->vertex(0)->point().x() << " " << it->vertex(0)->point().y() << std::endl;
			std::cout << it->vertex(1)->point().x() << " " << it->vertex(1)->point().y() << std::endl;
			std::cout << it->vertex(2)->point().x() << " " << it->vertex(2)->point().y() << std::endl;
			std::cout << std::endl;
		    }
		}
	    }
*/

	    
	}
    }
    out << "CELLS:" << std::endl;
    for (int i = 0; i < n; i ++) {
        out << 3 << " " << 0 << " " << i*3 << " " << i*3 + 1  << " " << i*3 + 2 << std::endl;  
    }
    out.close();
    
    return 0;
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

    ExportTriangulation2<CDT>(T, pgn_begin, pgn_end);
}

void AlphaShape2 (std::vector<double> pc, double a, int mesher, double para1, double para2) {  








    // 1. alpha shape
    

    std::vector<Alpha_shape_2::Point> v;


//ImportPointCloud<Alpha_shape_2::Point>(v);




    for (int i = 0; i < pc.size() /2 ; i++) {
	Alpha_shape_2::Point P(pc[i*2], pc[i*2+1]);
	v.push_back(P);
    }


    Alpha_shape_2 A(v.begin(), v.end(), a, Alpha_shape_2::GENERAL);



    // 2. new mesh 

    CDT T;
    
    



    

    
    
    // alpha shape 
    //std::ofstream out;
    //out.open("/home/jiping/OpenDT/M++/Machining/cgal/alpha_shape.cin");
    //out.precision(15);
    int m = 0;
    for (Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(); it != A.alpha_shape_edges_end(); ++it) {
        //out << it->first->vertex(it->first->cw(it->second))->point() << std::endl;
	//out << it->first->vertex(it->first->ccw(it->second))->point() << std::endl << std::endl;


	double z[4];
	z[0] = it->first->vertex(it->first->cw(it->second))->point().x();
	z[1] = it->first->vertex(it->first->cw(it->second))->point().y();
	z[2] = it->first->vertex(it->first->ccw(it->second))->point().x();
	z[3] = it->first->vertex(it->first->ccw(it->second))->point().y();
	T.insert_constraint(CDT::Point(z[0],z[1]), CDT::Point(z[2],z[3]));

	
	m++;
    }
    //out.close();

    /*
    out.open("/home/jiping/OpenDT/M++/Machining/cgal/constraints.cin");
    for (Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(); it != A.alpha_shape_edges_end(); ++it) {
        out << it->first->vertex(it->first->cw(it->second))->point() << " ";
	out << it->first->vertex(it->first->ccw(it->second))->point() << std::endl;
    }
    out.close();




    ExportTriangulation(A);
    */







    
    
    // mesh based on alpha shape
    Point poly[m];
    std::vector<Point> pp;
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin();
    poly[0] = it->first->vertex(it->first->cw(it->second))->point();
    poly[1] = it->first->vertex(it->first->ccw(it->second))->point();
    pp.push_back(poly[0]);
    pp.push_back(poly[1]);
    Point P;
    int l = 0;
    //std::cout << poly[0] << " " << poly[1] << std::endl;
    while (FindNext(P, poly[l], poly[l+1], A) && l < m-2) {
	poly[l+2] = P;
	pp.push_back(P);
	//std::cout << poly[l+1] << " " << poly[l+2] << std::endl;
	l++;
    }


    
    std::vector<CDT::Point> vv;
    
    for (int i = 0; i < pp.size(); i++) {
	vv.push_back(CDT::Point(pp[i].x(), pp[i].y()));
    }
    
    T.insert(vv.begin(),vv.end());

    


    
 
    
    if (mesher == 1) {
	// a new qualified mesh from alpha shape
        //out.open("/home/jiping/M++/Machining/cgal/point_cloud.cin");
	//out.precision(10);
	//for (int i = 0; i < m; i++) out << poly[i] << std::endl;
	//out.close();
	//CGALMeshGeneration(para1, para2, poly, poly+m, pp);





	 Mesher mesher(T);
	 mesher.set_criteria(Criteria(para1, para2));
	 mesher.refine_mesh();
	 CGAL::lloyd_optimize_mesh_2(T,CGAL::parameters::max_iteration_number = 10);

	 ExportTriangulation2<CDT>(T, poly, poly+m);


	
    }
    else if (mesher == 2)
        ExportTriangulation2(A, poly, poly+m);
    
    
}




//#include <CGAL/Alpha_shape_3.h>
//#include <CGAL/Alpha_shape_cell_base_3.h>
//#include <CGAL/Alpha_shape_vertex_base_3.h>
//#include <CGAL/Delaunay_triangulation_3.h>
//#include <cassert>
//#include <fstream>
//#include <list>
//typedef CGAL::Alpha_shape_vertex_base_3<K>          Vb3;
//typedef CGAL::Alpha_shape_cell_base_3<K>            Fb3;
//typedef CGAL::Triangulation_data_structure_3<Vb3,Fb3>  Tds3;
//typedef CGAL::Delaunay_triangulation_3<K,Tds3>       Triangulation_3;
//typedef CGAL::Alpha_shape_3<Triangulation_3>         Alpha_shape_3;
//typedef K::Point_3                                  Point3;
//typedef Alpha_shape_3::Alpha_iterator                Alpha_iterator3;
/*
int main()
{
  std::list<Point> lp;
  //read input
  std::ifstream is("./data/bunny_1000");
  int n;
  is >> n;
  std::cout << "Reading " << n << " points " << std::endl;
  Point p;
  for( ; n>0 ; n--)
  {
    is >> p;
    lp.push_back(p);
  }
  // compute alpha shape
  Alpha_shape_3 as(lp.begin(),lp.end());
  std::cout << "Alpha shape computed in REGULARIZED mode by default" << std::endl;
  // find optimal alpha value
  Alpha_iterator opt = as.find_optimal_alpha(1);
  std::cout << "Optimal alpha value to get one connected component is " << *opt << std::endl;
  as.set_alpha(*opt);
  assert(as.number_of_solid_components() == 1);
  return 0;
}
*/







