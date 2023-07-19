#include <iostream>
#include <portage/search/search_kdtree.h>
#include <portage/intersect/intersect_r2d.h>

// wonton includes
#include "wonton/support/wonton.h"
#include "wonton/support/Point.h"
#include "wonton/mesh/simple/simple_mesh.h"
#include "wonton/mesh/simple/simple_mesh_wrapper.h"
#include "wonton/state/simple/simple_state.h"
#include "wonton/state/simple/simple_state_wrapper.h"
#include "wonton/mesh/jali/jali_mesh_wrapper.h"

// portage includes
#include "portage/interpolate/interpolate_1st_order.h"
#include "portage/interpolate/interpolate_2nd_order.h"
#include "simple_intersect_for_tests.h"
#include "portage/support/portage.h"
#include "portage/driver/coredriver.h"

#include "JaliStateVector.h"
#include "JaliState.h"

#include <fstream>


#include "portage/search/search_simple.h"

#include "wonton/support/wonton.h"
#include "wonton/mesh/jali/jali_mesh_wrapper.h"
#include "wonton/state/jali/jali_state_wrapper.h"

#include "Mesh.hh"
#include "MeshFactory.hh"
#include "JaliStateVector.h"
#include "JaliState.h"

void test_search () {
	
	std::cout << "simple mesh" << std::endl;
	std::cout << "--------------------------------" << std::endl;
    std::cout << "simple search" << std::endl;
	
	Wonton::Simple_Mesh sm{0, 0, 1, 1, 20, 20};
	Wonton::Simple_Mesh tm{0, 0, 1, 1, 30, 30};
	const Wonton::Simple_Mesh_Wrapper source_mesh_wrapper(sm);
	const Wonton::Simple_Mesh_Wrapper target_mesh_wrapper(tm);
	std::cout << sm.num_entities(Wonton::Entity_kind::NODE, Wonton::Entity_type::ALL) << std::endl;
	std::cout << tm.num_entities(Wonton::Entity_kind::NODE, Wonton::Entity_type::ALL) << std::endl;
	
	Portage::SearchSimple<Wonton::Simple_Mesh_Wrapper,
						  Wonton::Simple_Mesh_Wrapper>
		search(source_mesh_wrapper, target_mesh_wrapper);
	
	for (int tc = 0; tc < 4; ++tc) {
		std::vector<int> candidates;
		search(tc, &candidates);
		for (int i=0; i<candidates.size(); i++)
			std::cout << candidates[i] << " ";
		std::cout << std::endl;
	}
	
	std::cout << std::endl;

	// *******************************************
	// *******************************************
	
	std::cout << "kd tree search" << std::endl;
	
	Portage::SearchKDTree<2, Portage::Entity_kind::CELL,
						  Wonton::Simple_Mesh_Wrapper,
						  Wonton::Simple_Mesh_Wrapper>
		search_kd(source_mesh_wrapper, target_mesh_wrapper);
	
	std::vector<Wonton::Point<2>> cell_coord;
	std::cout << "cell num: " << target_mesh_wrapper.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL) << std::endl;
	int cell_id = 100;
	std::cout << "target cell: " << std::endl;
    target_mesh_wrapper.cell_get_coordinates(cell_id, &cell_coord);
	for (int i=0; i<cell_coord.size(); i++)
		std::cout << cell_coord[i] << std::endl;
	std::cout << std::endl;
	std::vector<int> candidates = search_kd(cell_id);
	std::cout << "source cell: " << std::endl;
	for (int i=0; i<candidates.size(); i++) {
	    source_mesh_wrapper.cell_get_coordinates(candidates[i], &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			std::cout << cell_coord[j] << std::endl;
		std::cout << std::endl;
	}
	
	//export_overlap(source_mesh_wrapper, candidates, target_mesh_wrapper, cell_id);
	
	
	// *******************************************
	// *******************************************

	std::cout << "jali mesh" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::shared_ptr<Jali::Mesh> jali_source_mesh;
	std::shared_ptr<Jali::Mesh> jali_target_mesh;
	Jali::MeshFactory mesh_factory(MPI_COMM_WORLD);
	bool mstkOK = Jali::framework_available(Jali::MSTK);
	std::cout << "mstk framework is available: " << mstkOK << std::endl;
	mesh_factory.included_entities(Jali::Entity_kind::ALL_KIND);
    jali_source_mesh = mesh_factory("source.exo");
	jali_target_mesh = mesh_factory("target.exo");
	const Wonton::Jali_Mesh_Wrapper jali_sourceMeshWrapper(*jali_source_mesh);
	const Wonton::Jali_Mesh_Wrapper jali_targetMeshWrapper(*jali_target_mesh);

	Portage::SearchKDTree<2, Portage::Entity_kind::CELL,
						  Wonton::Jali_Mesh_Wrapper,
						  Wonton::Jali_Mesh_Wrapper>
		jali_search_kd(jali_sourceMeshWrapper, jali_targetMeshWrapper);
	
	cell_id = 50;
	candidates = jali_search_kd(cell_id);	
	//export_overlap(jali_sourceMeshWrapper, candidates, jali_targetMeshWrapper, cell_id);
	
}
