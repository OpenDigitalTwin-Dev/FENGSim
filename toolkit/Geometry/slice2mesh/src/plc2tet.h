#ifndef PLC2TET_H
#define PLC2TET_H

#include <cinolib/meshes/tetmesh.h>
#include <cinolib/meshes/trimesh.h>
#include <cinolib/sliced_object.h>
#include "common.h"

void plc2tet(const Trimesh<>   & plc,
             const SlicedObj<> & obj,
             const std::string & flags,
                   Tetmesh<>   & m);

#endif // PLC2TET_H
