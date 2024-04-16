#include "SlicePhaseTest.h"

void Export2VTK (std::string vtkfile, cura::Slicer slicer, const cura::coord_t initial_layer_thickness, const cura::coord_t layer_thickness) {
    double scale = 1000;

    int n = 0;
    for(int i = 0; i < slicer.layers.size(); i++) {
	const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    n += sliced_polygon.size();
	}
    }
    
    //std::cout << n << std::endl;
    //std::cout << cube_mesh.getAABB().min.z << " " << cube_mesh.getAABB().max.z << std::endl;
    //std::cout << cube_mesh.getAABB().min.x << " " << cube_mesh.getAABB().max.x << std::endl;
    //std::cout << cube_mesh.getAABB().min.y << " " << cube_mesh.getAABB().max.y << std::endl;
    
    std::ofstream out;
    out.open(vtkfile.c_str()); // the file is in the build path of FENGSim
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
		out << sliced_polygon[k].X / scale << " "
		    << sliced_polygon[k].Y / scale << " "
		    << (initial_layer_thickness + i * layer_thickness) / scale << std::endl;
	    }
	}
    }	
    int m = 0;
    for(int i = 0; i < slicer.layers.size(); i++) {
	const cura::SlicerLayer& layer = slicer.layers[i];
	for (int j = 0; j < layer.polygons.size(); j++) {
	    cura::Polygon sliced_polygon = layer.polygons[j];
	    for(int k = 0; k < sliced_polygon.size(); k++) {
		m++;
	    }
	}
    }
    out << "POLYGONS " << slicer.layers.size() << " " << slicer.layers.size() + n  << std::endl;
    m = 0;
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
}
