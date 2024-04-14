#include "test.h"

#include "portage/intersect/intersect_polys_r2d.h"
#include "portage/intersect/intersect_r2d.h"

void test_intersection () {
	std::cout << "------------------------------------" << std::endl;
	std::cout << "calculation in report" << std::endl;
	std::cout << "------------------------------------" << std::endl;
	std::vector<Wonton::Point<2>> source_poly;
	Wonton::Point<2> a1(0,0);
	Wonton::Point<2> a2(1,0);
	Wonton::Point<2> a3(0,1);
	source_poly.push_back(a1);
	source_poly.push_back(a2);
	source_poly.push_back(a3);
	std::vector<Wonton::Point<2>> target_poly;
	//Wonton::Point<2> b1(0,0);
	//Wonton::Point<2> b2(1,0);
	//Wonton::Point<2> b3(1,1);
	Wonton::Point<2> b1(0.5,0);
	Wonton::Point<2> b2(1.5,0);
	Wonton::Point<2> b3(0.5,1);
	target_poly.push_back(b1);
	target_poly.push_back(b2);
	target_poly.push_back(b3);

	Portage::NumericTolerances_t num_tols = Portage::DEFAULT_NUMERIC_TOLERANCES<2>;
	std::vector<double> results = Portage::intersect_polys_r2d(source_poly,target_poly,num_tols,true,Wonton::CoordSysType::Cartesian);

	for (int i=0; i<results.size(); i++)
		std::cout << results[i] << std::endl;

	std::cout << "------------------------------------" << std::endl;
	std::cout << "mesh intersection" << std::endl;
	std::cout << "------------------------------------" << std::endl;
	std::shared_ptr<Wonton::Simple_Mesh> sourcemesh = std::make_shared<Wonton::Simple_Mesh>(0, 0, 1, 1, 2, 2);
	std::shared_ptr<Wonton::Simple_Mesh> targetmesh = std::make_shared<Wonton::Simple_Mesh>(0, 0, 1, 1, 3, 3);
	const Wonton::Simple_Mesh_Wrapper sm(*sourcemesh);
	const Wonton::Simple_Mesh_Wrapper tm(*targetmesh);
	
	std::shared_ptr<Wonton::Simple_State> sourcestate = std::make_shared<Wonton::Simple_State>(sourcemesh);
	const Wonton::Simple_State_Wrapper ss(*sourcestate);

	Portage::IntersectR2D<Portage::Entity_kind::CELL,
						  Wonton::Simple_Mesh_Wrapper,
						  Wonton::Simple_State_Wrapper,
						  Wonton::Simple_Mesh_Wrapper>
		isect{sm, ss, tm, num_tols};

	// srccells is the cells from source mesh which are overlapped with target cell
	// in isect(tarcell, srccells) tarcell and srccells should be given.
	int tarcell = 4;
	std::vector<int> srccells({0});
	std::vector<Portage::Weights_t> srcwts = isect(tarcell, srccells);
	std::cout << "tarcell: " << tarcell << std::endl;
	std::vector<Wonton::Point<2>> cell_coord;
	tm.cell_get_coordinates(tarcell, &cell_coord);
	for (int i=0; i<cell_coord.size(); i++)
		std::cout << cell_coord[i] << std::endl;
	for (int i=0; i<srccells.size(); i++)
		std::cout << "srccells: " << srccells[i] << std::endl;

	std::cout << srcwts.size() << std::endl;
	int srcent = srcwts[0].entityID;
	std::vector<double> moments = srcwts[0].weights;
	int const num_moments = moments.size();
	for (int j = 0; j < num_moments; j++)
		std::cout << "i, j, m: " << srcent << ", " << j << ", " << moments[j] << std::endl;		
}
