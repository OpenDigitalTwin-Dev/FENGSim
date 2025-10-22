#include "../../../../../../toolkit/Geometry/cura_engine/src/Application.h"
#include "../../../../../../toolkit/Geometry/cura_engine/src/Slice.h"
#include "../../../../../../toolkit/Geometry/cura_engine/src/slicer.h"
#include "../../../../../../toolkit/Geometry/cura_engine/src/utils/floatpoint.h"
#include "../../../../../../toolkit/Geometry/cura_engine/src/utils/polygon.h"

#include "fstream"

void Export2VTK (std::string vtkfile, cura::Slicer slicer, const cura::coord_t initial_layer_thickness, const cura::coord_t layer_thickness);
void Export2VTK4PathPlanning (std::string vtkfile_pathplanning, cura::Slicer slicer, const cura::coord_t initial_layer_thickness, const cura::coord_t layer_thickness);
void Export2Cli4Mesh (std::string clifile_meshing, cura::Slicer slicer, const cura::coord_t initial_layer_thickness, const cura::coord_t layer_thickness, double buttom);

