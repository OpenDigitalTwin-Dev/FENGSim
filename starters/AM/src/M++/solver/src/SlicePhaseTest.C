//Copyright (c) 2019 Ultimaker B.V.
//CuraEngine is released under the terms of the AGPLv3 or higher.

#include <gtest/gtest.h>

#include "../../../../../../tools/cura_engine/src/Application.h"
#include "../../../../../../tools/cura_engine/src/Slice.h"
#include "../../../../../../tools/cura_engine/src/slicer.h"
#include "../../../../../../tools/cura_engine/src/utils/floatpoint.h"
#include "../../../../../../tools/cura_engine/src/utils/polygon.h"
//#include "../../../../../../tools/cura_engine/src/utils/polygonUtils.h"

#include "fstream"

//using namespace cura;
void SlicePhaseTestMain (int argc, char** argv) {
    std::ifstream is;
    is.open(std::string("./solver/conf/cura.conf").c_str());
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
    scene.settings.add("meshfix_maximum_resolution", "0.04");
    scene.settings.add("meshfix_maximum_deviation", "0.02");
    scene.settings.add("xy_offset", "0");
    scene.settings.add("xy_offset_layer_0", "0");
    
    // import stl mesh
    cura::MeshGroup& mesh_group = scene.mesh_groups.back();
    const cura::FMatrix3x3 transformation;
    cura::loadMeshIntoMeshGroup(&mesh_group, stlfile.c_str(), transformation, scene.settings);
    //cura::loadMeshIntoMeshGroup(&mesh_group, "Cura/conf/geo/sphere.stl", transformation, scene.settings);
    //cura::loadMeshIntoMeshGroup(&mesh_group, "Cura/conf/geo/cube.stl", transformation, scene.settings);
    //cura::loadMeshIntoMeshGroup(&mesh_group, "Cura/conf/geo/cylinder1000.stl", transformation, scene.settings);
    cura::Mesh& cube_mesh = mesh_group.meshes[0];

    // generate slices
    const cura::coord_t layer_thickness = scene.settings.get<cura::coord_t>("layer_height");
    const cura::coord_t initial_layer_thickness = scene.settings.get<cura::coord_t>("layer_height_0");
    constexpr bool variable_layer_height = false;
    constexpr std::vector<cura::AdaptiveLayer>* variable_layer_height_values = nullptr;
    const size_t num_layers = (cube_mesh.getAABB().max.z - initial_layer_thickness) / layer_thickness + 1;
    cura::Slicer slicer(&cube_mesh, layer_thickness, num_layers, variable_layer_height, variable_layer_height_values);
    std::cout << "The number of layers in the output must equal the requested number of layers." << std::endl 
	      << slicer.layers.size() << " " << num_layers << std::endl;


    double scale = 1000;

    // *************************************************
    //
    // export slices for visualization
    //
    // *************************************************
    int n = 0;
    for(int i = 0; i < slicer.layers.size(); i++) {
	const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    n += sliced_polygon.size();
	}
    }
    
    std::cout << n << std::endl;
    std::cout << cube_mesh.getAABB().min.z << " " << cube_mesh.getAABB().max.z << std::endl;
    std::cout << cube_mesh.getAABB().min.x << " " << cube_mesh.getAABB().max.x << std::endl;
    std::cout << cube_mesh.getAABB().min.y << " " << cube_mesh.getAABB().max.y << std::endl;

    
    std::ofstream out;
    out.open(vtkfile.c_str());
    //out.open("/home/jiping/M++/data/vtk/slices.vtk");
    out <<"# vtk DataFile Version 2.0" << std::endl;
    out << "slices example" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET POLYDATA" << std::endl;
    out << "POINTS " << n << " float" << std::endl;
    for(int i = 0; i < slicer.layers.size(); i++) {
        const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    for(int k = 0; k < sliced_polygon.size(); k++) {
	        out << sliced_polygon[k].X / scale << " " << sliced_polygon[k].Y / scale  << " " << (initial_layer_thickness + i * layer_thickness) / scale << std::endl;
	    }
	}
    }
    out << "POLYGONS " << slicer.layers.size() << " " << slicer.layers.size() + n << std::endl;
    int m = 0;
    for(int i = 0; i < slicer.layers.size(); i++) {
        const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    out << sliced_polygon.size();
	    for(int k = 0; k < sliced_polygon.size(); k++) {
	        out << " " << m;
		m++;
	    }
	    out << std::endl;
	}
    }
    out.close();

    // *************************************************
    //
    // export slices for path planning
    //
    // *************************************************


    n = 0;
    for(int i = 1; i < slicer.layers.size(); i++) {
	const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    n += sliced_polygon.size();
	}
    }
    
    std::cout << n << std::endl;
    std::cout << cube_mesh.getAABB().min.z << " " << cube_mesh.getAABB().max.z << std::endl;
    std::cout << cube_mesh.getAABB().min.x << " " << cube_mesh.getAABB().max.x << std::endl;
    std::cout << cube_mesh.getAABB().min.y << " " << cube_mesh.getAABB().max.y << std::endl;

    
    out.open(vtkfile_pathplanning.c_str());
    //out.open("/home/jiping/M++/data/vtk/slices.vtk");
    out <<"# vtk DataFile Version 2.0" << std::endl;
    out << "slices example" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET POLYDATA" << std::endl;
    out << "POINTS " << n << " float" << std::endl;
    for(int i = 1; i < slicer.layers.size(); i++) {
        const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    for(int k = 0; k < sliced_polygon.size(); k++) {
	        out << sliced_polygon[k].X / scale << " " << sliced_polygon[k].Y / scale  << " " << (initial_layer_thickness + i * layer_thickness) / scale << std::endl;
	    }
	}
    }
    out << "POLYGONS " << slicer.layers.size() << " " << slicer.layers.size() + n << std::endl;
    m = 0;
    for(int i = 1; i < slicer.layers.size(); i++) {
        const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    out << sliced_polygon.size();
	    for(int k = 0; k < sliced_polygon.size(); k++) {
	        out << " " << m;
		m++;
	    }
	    out << std::endl;
	}
    }
    out.close();



    
    // *************************************************
    //
    // export slices for meshing
    //
    // *************************************************
    out.open(clifile_meshing.c_str());
    out << "$$HEADERSTART" << std::endl;
    out << "$$ASCII" << std::endl;
    out << "$$UNITS/1" << std::endl;
    out << "$$DATE/230718" << std::endl;
    out << "$$LAYERS/" << slicer.layers.size() + 1<< std::endl;
    out << "$$HEADEREND" << std::endl;
    out << "$$GEOMETRYSTART" << std::endl;



    const cura::SlicerLayer& layer = slicer.layers[0];
    out << "$$LAYER/" << cube_mesh.getAABB().min.z << std::endl;
    for (int j = 0; j < layer.polygons.size(); j++) {
      cura::Polygon sliced_polygon = layer.polygons[j];
      out << "$$POLYLINE/0,1," << sliced_polygon.size() + 1;
      for(int k = 0; k < sliced_polygon.size(); k++) {
	  out << "," << (sliced_polygon[k].X) / scale << "," << (sliced_polygon[k].Y) / scale;
      }
      out << "," << (sliced_polygon[0].X) / scale << "," << (sliced_polygon[0].Y) / scale
	  << std::endl;
    }

    const cura::SlicerLayer& layer1 = slicer.layers[1];
    out << "$$LAYER/" << initial_layer_thickness / scale << std::endl;
    for (int j = 0; j < layer1.polygons.size(); j++) {
      cura::Polygon sliced_polygon = layer1.polygons[j];
      out << "$$POLYLINE/0,1," << sliced_polygon.size() + 1;
      for(int k = 0; k < sliced_polygon.size(); k++) {
	  out << "," << (sliced_polygon[k].X) / scale << "," << (sliced_polygon[k].Y) / scale;
      }
      out << "," << (sliced_polygon[0].X) / scale << "," << (sliced_polygon[0].Y) / scale
	  << std::endl;
    }

    
    for(int i = 1; i < slicer.layers.size(); i++) {
        const cura::SlicerLayer& layer2 = slicer.layers[i];
	out << "$$LAYER/" << (initial_layer_thickness + i * layer_thickness) / scale << std::endl;
	for (int j = 0; j < layer2.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer2.polygons[j];
	    out << "$$POLYLINE/0,1," << sliced_polygon.size() + 1;
	    for(int k = 0; k < sliced_polygon.size(); k++) {
	        out << "," << (sliced_polygon[k].X) / scale << "," << (sliced_polygon[k].Y) / scale;
	    }
	    out << "," << (sliced_polygon[0].X) / scale << "," << (sliced_polygon[0].Y) / scale
		<< std::endl;
	}
    }
    out << "$$GEOMETRYEND" << std::endl;
    
    out.close();



    


}












/*
namespace cura
{

class AdaptiveLayer;


    // Integration test on the slicing phase of CuraEngine. This tests if the
    // slicing algorithm correctly splits a 3D model up into 2D layers.
    
class SlicePhaseTest : public testing::Test
{
    void SetUp()
    {
        //Set up a scene so that we may request settings.
        Application::getInstance().current_slice = new Slice(1);

        //And a few settings that we want to default.
        Scene& scene = Application::getInstance().current_slice->scene;
        scene.settings.add("slicing_tolerance", "middle");
        scene.settings.add("layer_height_0", "0.2");
        scene.settings.add("layer_height", "0.1");
        scene.settings.add("magic_mesh_surface_mode", "normal");
        scene.settings.add("meshfix_extensive_stitching", "false");
        scene.settings.add("meshfix_keep_open_polygons", "false");
        scene.settings.add("minimum_polygon_circumference", "1");
        scene.settings.add("meshfix_maximum_resolution", "0.04");
        scene.settings.add("meshfix_maximum_deviation", "0.02");
        scene.settings.add("xy_offset", "0");
        scene.settings.add("xy_offset_layer_0", "0");
    }
};

TEST_F(SlicePhaseTest, Cube)
{
    Scene& scene = Application::getInstance().current_slice->scene;
    MeshGroup& mesh_group = scene.mesh_groups.back();

    const FMatrix3x3 transformation;
    //Path to cube.stl is relative to CMAKE_CURRENT_SOURCE_DIR/tests.
    ASSERT_TRUE(loadMeshIntoMeshGroup(&mesh_group, "integration/resources/cube.stl", transformation, scene.settings));
    EXPECT_EQ(mesh_group.meshes.size(), 1);
    Mesh& cube_mesh = mesh_group.meshes[0];

    const coord_t layer_thickness = scene.settings.get<coord_t>("layer_height");
    const coord_t initial_layer_thickness = scene.settings.get<coord_t>("layer_height_0");
    constexpr bool variable_layer_height = false;
    constexpr std::vector<AdaptiveLayer>* variable_layer_height_values = nullptr;
    const size_t num_layers = (cube_mesh.getAABB().max.z - initial_layer_thickness) / layer_thickness + 1;
    Slicer slicer(&cube_mesh, layer_thickness, num_layers, variable_layer_height, variable_layer_height_values);

    ASSERT_EQ(slicer.layers.size(), num_layers) << "The number of layers in the output must equal the requested number of layers.";

    //Since a cube has the same slice at all heights, every layer must be the same square.
    Polygon square;
    square.emplace_back(0, 0);
    square.emplace_back(10000, 0); //10mm cube.
    square.emplace_back(10000, 10000);
    square.emplace_back(0, 10000);

    for(size_t layer_nr = 0; layer_nr < num_layers; layer_nr++)
    {
        const SlicerLayer& layer = slicer.layers[layer_nr];
        EXPECT_EQ(layer.polygons.size(), 1);
        if(layer.polygons.size() == 1)
        {
            Polygon sliced_polygon = layer.polygons[0];
            EXPECT_EQ(sliced_polygon.size(), square.size());
            if(sliced_polygon.size() == square.size())
            {
                int start_corner = -1;
                for(size_t corner_idx = 0; corner_idx < square.size(); corner_idx++) //Find the starting corner in the sliced layer.
                {
                    if(square[corner_idx] == sliced_polygon[0])
                    {
                        start_corner = corner_idx;
                        break;
                    }
                }
                EXPECT_NE(start_corner, -1) << "The first vertex of the sliced polygon must be one of the vertices of the ground truth square.";

                if(start_corner != -1)
                {
                    for(size_t corner_idx = 0; corner_idx < square.size(); corner_idx++) //Check if every subsequent corner is correct.
                    {
                        EXPECT_EQ(square[(corner_idx + start_corner) % square.size()], sliced_polygon[corner_idx]);
                    }
                }
            }
        }
    }
}

TEST_F(SlicePhaseTest, Cylinder1000)
{
    Scene& scene = Application::getInstance().current_slice->scene;
    MeshGroup& mesh_group = scene.mesh_groups.back();

    const FMatrix3x3 transformation;
    //Path to cylinder1000.stl is relative to CMAKE_CURRENT_SOURCE_DIR/tests.
    ASSERT_TRUE(loadMeshIntoMeshGroup(&mesh_group, "integration/resources/cylinder1000.stl", transformation, scene.settings));
    EXPECT_EQ(mesh_group.meshes.size(), 1);
    Mesh& cylinder_mesh = mesh_group.meshes[0];

    const coord_t layer_thickness = scene.settings.get<coord_t>("layer_height");
    const coord_t initial_layer_thickness = scene.settings.get<coord_t>("layer_height_0");
    constexpr bool variable_layer_height = false;
    constexpr std::vector<AdaptiveLayer>* variable_layer_height_values = nullptr;
    const size_t num_layers = (cylinder_mesh.getAABB().max.z - initial_layer_thickness) / layer_thickness + 1;
    Slicer slicer(&cylinder_mesh, layer_thickness, num_layers, variable_layer_height, variable_layer_height_values);

    ASSERT_EQ(slicer.layers.size(), num_layers) << "The number of layers in the output must equal the requested number of layers.";

    //Since a cylinder has the same slice at all heights, every layer must be the same circle.
    constexpr size_t num_vertices = 1000; //Create a circle with this number of vertices (first vertex is in the +X direction).
    constexpr coord_t radius = 10000; //10mm radius.
    Polygon circle;
    circle.reserve(num_vertices);
    for(size_t i = 0; i < 1000; i++)
    {
        const coord_t x = std::cos(M_PI * 2 / num_vertices * i) * radius;
        const coord_t y = std::sin(M_PI * 2 / num_vertices * i) * radius;
        circle.emplace_back(x, y);
    }
    Polygons circles;
    circles.add(circle);

    for(size_t layer_nr = 0; layer_nr < num_layers; layer_nr++)
    {
        const SlicerLayer& layer = slicer.layers[layer_nr];
        EXPECT_EQ(layer.polygons.size(), 1);
        if(layer.polygons.size() == 1)
        {
            Polygon sliced_polygon = layer.polygons[0];
            //Due to the reduction in resolution, the final slice will not have the same vertices as the input.
            //Let's say that are allowed to be up to 1/500th of the surface area off.
            EXPECT_LE(PolygonUtils::relativeHammingDistance(layer.polygons, circles), 0.002);
        }
    }
}

} //namespace cura

*/
