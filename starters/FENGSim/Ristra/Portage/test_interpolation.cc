#include "test.h"
#include "wonton/state/simple/simple_state_wrapper.h"

// ***************************************
// ***************************************
//
// this is an example from test_interp_2nd_order.cc
//
// ***************************************
// ***************************************

void test_interpolation() {
	std::shared_ptr<Wonton::Simple_Mesh> source_mesh =
		std::make_shared<Wonton::Simple_Mesh>(0.0, 0.0, 1.0, 1.0, 2, 2);
	std::shared_ptr<Wonton::Simple_Mesh> target_mesh =
		std::make_shared<Wonton::Simple_Mesh>(0.0, 0.0, 1.0, 1.0, 3, 3);
	
	// Create mesh wrappers
	
	Wonton::Simple_Mesh_Wrapper sourceMeshWrapper(*source_mesh);
	Wonton::Simple_Mesh_Wrapper targetMeshWrapper(*target_mesh);
	
	// count cells
	
	const int ncells_source =
		sourceMeshWrapper.num_owned_cells();
	const int ncells_target =
		targetMeshWrapper.num_owned_cells();
	
	// Create a state object
	
	Wonton::Simple_State source_state(source_mesh);
	
	// Define a state vector with constant value and add it to the source state
	
	std::vector<double> data(ncells_source);
	for (int i=0; i<data.size(); i++)
		data[i] = i+1;
	source_state.add("cellvars", Wonton::Entity_kind::CELL, &(data[0]));
	
	// Create state wrapper

	Wonton::Simple_State_Wrapper sourceStateWrapper(source_state);
	
	// Gather the cell coordinates as Portage Points for source and target meshes
	// for intersection. The outer vector is the cells, the inner vector is the
	// points of the vertices of that cell.
	
	std::vector<std::vector<Wonton::Point<2>>>
		source_cell_coords(ncells_source);
	std::vector<std::vector<Wonton::Point<2>>>
		target_cell_coords(ncells_target);
	
	// Actually get the Wonton::Points

	for (int c = 0; c < ncells_source; ++c)
		sourceMeshWrapper.cell_get_coordinates(c, &(source_cell_coords[c]));
	for (int c = 0; c < ncells_target; ++c)
		targetMeshWrapper.cell_get_coordinates(c, &(target_cell_coords[c]));

	// Interpolate from source to target mesh using the independent calculation
	// in simple_intersect_for_tests.h
	
	std::vector<double> outvals(ncells_target);
	std::vector<std::vector<Portage::Weights_t>>
		sources_and_weights(ncells_target);
	
	// Loop over target cells
	
	for (int c = 0; c < ncells_target; ++c) {
		
		std::vector<int> xcells;
		std::vector<std::vector<double>> xwts;
		
		// Compute the moments
		// xcells is the source cell indices that intersect
		// xwts is the moments vector for each cell that intersects

		BOX_INTERSECT::intersection_moments<2>(target_cell_coords[c],
											   source_cell_coords,
											   &xcells, &xwts);

		// Pack the results into a vector of true Portage::Weights_t
		
		int const num_intersect_cells = xcells.size();
		std::vector<Portage::Weights_t> wtsvec(num_intersect_cells);
		for (int i = 0; i < num_intersect_cells; ++i) {
			wtsvec[i].entityID = xcells[i];
			wtsvec[i].weights = xwts[i];
		}
		
		// Put the weights in final form
		
		sources_and_weights[c] = wtsvec;
	}
	
	// Now do it the Portage way
	
	// use default tolerances
	Portage::NumericTolerances_t num_tols = Portage::DEFAULT_NUMERIC_TOLERANCES<2>;

	// Create Interpolation object

	Portage::Interpolate_1stOrder<2, Wonton::Entity_kind::CELL,
								  Wonton::Simple_Mesh_Wrapper,
								  Wonton::Simple_Mesh_Wrapper,
								  Wonton::Simple_State_Wrapper,
								  Wonton::Simple_State_Wrapper,
								  double>
		interpolator(sourceMeshWrapper, targetMeshWrapper, sourceStateWrapper,
					 num_tols);
	
	interpolator.set_interpolation_variable("cellvars");

	int id = 1;
	std::cout << "target cell: " << std::endl;
	for (int i=0; i<target_cell_coords[id].size(); i++) {
		std::cout << target_cell_coords[id][i] << std::endl;
	}
	std::cout << "interpolation for " << id << ": " << std::endl;
	std::cout << interpolator(id, sources_and_weights[id]) << std::endl;
	std::cout << "source cells: " << std::endl;
	for (int i=0; i<sources_and_weights[id].size(); i++) {
		int id2 = sources_and_weights[id][i].entityID;
		for (int j=0; j<source_cell_coords[id2].size(); j++) {
			std::cout << source_cell_coords[id2][j] << std::endl;
		}
		std::cout << std::endl;
	}
		
	Wonton::transform(targetMeshWrapper.begin(Wonton::Entity_kind::CELL),
					  targetMeshWrapper.end(Wonton::Entity_kind::CELL),
					  sources_and_weights.begin(),
					  outvals.begin(), interpolator);
	
	
}
