//Copyright (c) 2019 Ultimaker B.V.
//CuraEngine is released under the terms of the AGPLv3 or higher.

#include "SlicePhaseTest.h"

// *************************************************
// the original codes are in the path
// "FENGSim/toolkit/cura_engine/tests/integration/SlicePhaseTest.cpp"
// *************************************************

//using namespace cura;
void SlicePhaseTestMain (int argc, char** argv) {
    std::ifstream is;
    is.open(std::string("./solver/conf/slicing.conf").c_str());
    const int len = 512;
    char L[len];
    is.getline(L,len);
    is.getline(L,len);
    std::string stlfile = L;
    is.getline(L,len);
    std::string vtkfile = L;
    is.getline(L,len);
    std::string vtkfile_pathplanning = L;
    is.getline(L,len);
    std::string clifile_meshing = L;
    is.getline(L,len);
    double layer_height_0 = 0.2;
    sscanf(L,"%lf", &layer_height_0);
    is.getline(L,len);
    double layer_height = 0.25;
    sscanf(L,"%lf", &layer_height);
    
    std::cout << stlfile << std::endl;
    std::cout << vtkfile << std::endl;
    std::cout << vtkfile_pathplanning << std::endl;
    std::cout << clifile_meshing << std::endl;
    std::cout << layer_height_0 << std::endl;
    std::cout << layer_height << std::endl;

    // configuration
    cura::Application::getInstance().current_slice = new cura::Slice(1);
    //And a few settings that we want to default.
    cura::Scene& scene = cura::Application::getInstance().current_slice->scene;    
    scene.settings.add("slicing_tolerance", "middle");
    scene.settings.add("layer_height_0", std::to_string(layer_height_0));
    scene.settings.add("layer_height", std::to_string(layer_height));
    scene.settings.add("magic_mesh_surface_mode", "normal");
    scene.settings.add("meshfix_extensive_stitching", "false");
    scene.settings.add("meshfix_keep_open_polygons", "false");
    scene.settings.add("minimum_polygon_circumference", "1");
    scene.settings.add("meshfix_maximum_resolution", "0.00001");
    scene.settings.add("meshfix_maximum_deviation", "0.00001");
    scene.settings.add("xy_offset", "0");
    scene.settings.add("xy_offset_layer_0", "0");
    
    // import stl mesh
    cura::MeshGroup& mesh_group = scene.mesh_groups.back();
    const cura::FMatrix3x3 transformation;

    cura::loadMeshIntoMeshGroup(&mesh_group, stlfile.c_str(), transformation, scene.settings);

    cura::Mesh& cube_mesh = mesh_group.meshes[0];

    // generate slices
    const cura::coord_t layer_thickness = scene.settings.get<cura::coord_t>("layer_height");
    const cura::coord_t initial_layer_thickness = scene.settings.get<cura::coord_t>("layer_height_0");

    constexpr bool variable_layer_height = false;
    constexpr std::vector<cura::AdaptiveLayer>* variable_layer_height_values = nullptr;
    const size_t num_layers = (cube_mesh.getAABB().max.z - initial_layer_thickness) / layer_thickness + 1;

    cura::Slicer slicer(&cube_mesh, layer_thickness, num_layers, variable_layer_height, variable_layer_height_values);
    
    std::cout << "The number of layers in the output must equal the requested number of layers." << std::endl 
	      << "  " << slicer.layers.size() << " " << num_layers << std::endl;

    /*****************************************************************************/
    /*                                                                           */
    /*   export slices for visualization                                         */
    /*                                                                           */
    /*****************************************************************************/

    Export2VTK(vtkfile, slicer, initial_layer_thickness, layer_thickness);

    /*****************************************************************************/
    /*                                                                           */
    /* export slices for path planning                                           */
    /*                                                                           */
    /* the difference between Export2VTK and Export2VTK4PathPlanning is that     */
    /* initial layer and need to pay attention the initial layer hieight is in   */
    /* fact the 0.5 of the buttom                                                */
    /*                                                                           */
    /*****************************************************************************/

    Export2VTK4PathPlanning(vtkfile_pathplanning, slicer, initial_layer_thickness, layer_thickness);

    /*****************************************************************************/
    /*                                                                           */
    /* export slices for meshing                                                 */
    /*                                                                           */
    /* different format                                                          */
    /*                                                                           */
    /*****************************************************************************/

    Export2Cli4Mesh(clifile_meshing, slicer, initial_layer_thickness, layer_thickness, cube_mesh.getAABB().min.z);

    


}







