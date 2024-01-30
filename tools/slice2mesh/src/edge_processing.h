#ifndef EDGE_PROCESSING_H
#define EDGE_PROCESSING_H

#include <cinolib/sliced_object.h>
#include "common.h"


void edge_wise_intersections(const SlicedObj<>     & obj,
                                   SLICE2MESH_data & data);

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void process_edge_pair(std::vector<V_data> & points,
                       E_data              & e_below,
                       E_data              & e_above);


#endif // EDGE_PROCESSING_H
