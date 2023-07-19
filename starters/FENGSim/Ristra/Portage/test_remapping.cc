#include <iostream>
#include <memory>

#include "mpi.h"

#include "wonton/support/wonton.h"
#include "wonton/mesh/jali/jali_mesh_wrapper.h"
#include "wonton/state/jali/jali_state_wrapper.h"

#include "portage/driver/mmdriver.h"
#include "portage/intersect/intersect_rNd.h"
#include "portage/interpolate/interpolate_1st_order.h"

#include "Mesh.hh"
#include "MeshFactory.hh"

#include "test.h"

double TOL = 1e-6;

// ***************************************
// ***************************************
//
// this is an example from test_driver.cc
//
// ***************************************
// ***************************************

double compute_constant_field(JaliGeometry::Point centroid) {
	return 1.0;
}

double compute_linear_field(JaliGeometry::Point centroid) {
	return centroid[0]+centroid[1];
}
double compute_linear_field_3d(JaliGeometry::Point centroid) {
	return 100*(centroid[0]+centroid[1]+centroid[2]);
}
double compute_quadratic_field(JaliGeometry::Point centroid) {
	return 3*3*centroid[0]*centroid[0]+40*40*centroid[1]*centroid[1];
}
double compute_quadratic_field_3d(JaliGeometry::Point centroid) {
	return 3*3*centroid[0]*centroid[0] + 40*40*centroid[1]*centroid[1] +
		500*500*centroid[2]*centroid[2];
}

void test_remapping_1() {
	// step 1 mesh
	if (!Jali::framework_available(Jali::MSTK)) return;
	Jali::MeshFactory mesh_factory(MPI_COMM_WORLD);
	mesh_factory.framework(Jali::MSTK);	
	std::shared_ptr<Jali::Mesh> sourceMesh = mesh_factory(0.0, 0.0, 1.0, 1.0, 2, 2);
	std::cout << "source cells num: " <<sourceMesh->num_cells<Jali::Entity_type::ALL>() << std::endl;
	std::shared_ptr<Jali::Mesh> targetMesh = mesh_factory(0.0, 0.0, 1.0, 1.0, 3, 3);
	std::cout << "target cells num: " << targetMesh->num_cells<Jali::Entity_type::ALL>() << std::endl;
	Wonton::Jali_Mesh_Wrapper sourceMeshWrapper(*sourceMesh);
	Wonton::Jali_Mesh_Wrapper targetMeshWrapper(*targetMesh);

	// step 2 state
	std::shared_ptr<Jali::State> sourceState = Jali::State::create(sourceMesh);
	std::shared_ptr<Jali::State> targetState = Jali::State::create(targetMesh);	
	Wonton::Jali_State_Wrapper sourceStateWrapper(*sourceState);
	Wonton::Jali_State_Wrapper targetStateWrapper(*targetState);
	
	const int nsrccells = sourceMeshWrapper.num_owned_cells() +
		sourceMeshWrapper.num_ghost_cells();
	std::vector<double> sourceData(nsrccells);
	for (int c = 0; c < nsrccells; ++c) {
		JaliGeometry::Point cen = sourceMesh->cell_centroid(c);
		//sourceData[c] = compute_constant_field(cen);
		sourceData[c] = compute_linear_field(cen);
		std::cout << sourceData[c] << " ";
	}
	std::cout << std::endl;
	sourceState->add("celldata", sourceMesh, Jali::Entity_kind::CELL,
					 Jali::Entity_type::ALL, &(sourceData[0]));
	
	const int ntarcells = targetMeshWrapper.num_owned_cells();
	std::vector<double> targetData(ntarcells, 0.0);  
	targetState->add("celldata", targetMesh, Jali::Entity_kind::CELL,
					 Jali::Entity_type::ALL, &(targetData[0]));

	// step 3 remapping
	std::vector<std::string> remap_fields;
	remap_fields.emplace_back("celldata");
	
	Portage::MMDriver<Portage::SearchKDTree,
					  Portage::IntersectRnD,
					  Portage::Interpolate_1stOrder, 2,
					  Wonton::Jali_Mesh_Wrapper, Wonton::Jali_State_Wrapper,
					  Wonton::Jali_Mesh_Wrapper, Wonton::Jali_State_Wrapper>
		d(sourceMeshWrapper, sourceStateWrapper, targetMeshWrapper,
		  targetStateWrapper);
	d.set_remap_var_names(remap_fields);
	d.run();

	// step 4 error estimate
	Jali::UniStateVector<double, Jali::Mesh> cellvecout;
	bool found = targetState->get<double, Jali::Mesh>("celldata", targetMesh,
													  Jali::Entity_kind::CELL,
													  Jali::Entity_type::ALL,
													  &cellvecout);
	std::cout << "if found interpolation value: " << found << std::endl;
	double source_integral = 0.0;
	for (int c = 0; c < nsrccells; ++c) {
		double cellvol = sourceMesh->cell_volume(c);
		source_integral += sourceData[c]*cellvol;
	}
	
	double field_err2 = 0., target_integral = 0.;
	for (int c = 0; c < ntarcells; ++c) {
		JaliGeometry::Point ccen = targetMesh->cell_centroid(c);
		//double error = compute_constant_field(ccen) - cellvecout[c];
		double error = compute_linear_field(ccen) - cellvecout[c];
		field_err2 += error*error;
		
		double cellvol = targetMesh->cell_volume(c);
		target_integral += cellvecout[c]*cellvol;

		std::cout << cellvecout[c] << " ";
	}
	std::cout << std::endl;
	
	std::cout << "error: " << field_err2
			  << " source integral: " << source_integral
			  << " target integral: " << target_integral << std::endl;
}

void test_remapping_2() {
	// ************************************
	// step 1 mesh
	// ************************************
	if (!Jali::framework_available(Jali::MSTK)) return;
	Jali::MeshFactory mesh_factory(MPI_COMM_WORLD);
	mesh_factory.framework(Jali::MSTK);	
	mesh_factory.included_entities(Jali::Entity_kind::ALL_KIND);
    std::shared_ptr<Jali::Mesh> sourceMesh = mesh_factory("source.exo");
	std::cout << "source cells num: " <<sourceMesh->num_cells<Jali::Entity_type::ALL>() << std::endl;
	std::shared_ptr<Jali::Mesh> targetMesh = mesh_factory("target.exo");
	std::cout << "target cells num: " << targetMesh->num_cells<Jali::Entity_type::ALL>() << std::endl;
	Wonton::Jali_Mesh_Wrapper sourceMeshWrapper(*sourceMesh);
	Wonton::Jali_Mesh_Wrapper targetMeshWrapper(*targetMesh);

	// ************************************
	// step 2 state
	// ************************************
	std::shared_ptr<Jali::State> sourceState = Jali::State::create(sourceMesh);
	std::shared_ptr<Jali::State> targetState = Jali::State::create(targetMesh);	
	Wonton::Jali_State_Wrapper sourceStateWrapper(*sourceState);
	Wonton::Jali_State_Wrapper targetStateWrapper(*targetState);
	
	const int nsrccells = sourceMeshWrapper.num_owned_cells() +
		sourceMeshWrapper.num_ghost_cells();
	std::vector<double> sourceData(nsrccells);
	for (int c = 0; c < nsrccells; ++c) {
		JaliGeometry::Point cen = sourceMesh->cell_centroid(c);
		//sourceData[c] = compute_constant_field(cen);
		sourceData[c] = compute_linear_field(cen);
		std::cout << sourceData[c] << " ";
	}
	std::cout << std::endl;
	sourceState->add("celldata", sourceMesh, Jali::Entity_kind::CELL,
					 Jali::Entity_type::ALL, &(sourceData[0]));
	
	const int ntarcells = targetMeshWrapper.num_owned_cells();
	std::vector<double> targetData(ntarcells, 0.0);  
	targetState->add("celldata", targetMesh, Jali::Entity_kind::CELL,
					 Jali::Entity_type::ALL, &(targetData[0]));

	// ************************************
	// step 3 remapping
	// ************************************
	std::vector<std::string> remap_fields;
	remap_fields.emplace_back("celldata");
	
	Portage::MMDriver<Portage::SearchKDTree,
					  Portage::IntersectRnD,
					  Portage::Interpolate_1stOrder, 2,
					  Wonton::Jali_Mesh_Wrapper, Wonton::Jali_State_Wrapper,
					  Wonton::Jali_Mesh_Wrapper, Wonton::Jali_State_Wrapper>
		d(sourceMeshWrapper, sourceStateWrapper, targetMeshWrapper,
		  targetStateWrapper);
	d.set_remap_var_names(remap_fields);
	d.run();

	// ************************************
	// step 4 error estimate
	// ************************************
	Jali::UniStateVector<double, Jali::Mesh> cellvecout;
	bool found = targetState->get<double, Jali::Mesh>("celldata", targetMesh,
													  Jali::Entity_kind::CELL,
													  Jali::Entity_type::ALL,
													  &cellvecout);
	std::cout << "if found interpolation value: " << found << std::endl;
	double source_integral = 0.0;
	for (int c = 0; c < nsrccells; ++c) {
		double cellvol = sourceMesh->cell_volume(c);
		source_integral += sourceData[c]*cellvol;
	}
	
	double field_err2 = 0., target_integral = 0.;
	for (int c = 0; c < ntarcells; ++c) {
		JaliGeometry::Point ccen = targetMesh->cell_centroid(c);
		//double error = compute_constant_field(ccen) - cellvecout[c];
		double error = compute_linear_field(ccen) - cellvecout[c];
		field_err2 += error*error;
		
		double cellvol = targetMesh->cell_volume(c);
		target_integral += cellvecout[c]*cellvol;
		
		std::cout << cellvecout[c] << " ";
	}
	std::cout << std::endl;
	
	std::cout << "error: " << field_err2
			  << " source integral: " << source_integral
			  << " target integral: " << target_integral << std::endl;

	// ************************************
	// export data to vtk
	// ************************************
	export_vtk(cellvecout, targetMeshWrapper);
}

void test_remapping() {
	test_remapping_1();
	test_remapping_2();
}
