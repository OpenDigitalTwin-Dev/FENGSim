#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <set>
#include <cinolib/geometry/vec3.h>

using namespace cinolib;

// Face classification for the PLC
enum
{
    SRF_FACE_VERT = 0x00000001, // External faces generated extruding the slice edges along the build direction
    SRF_FACE_DOWN = 0x00000002, // External faces generated triangulating the slices (pointing downwards)
    SRF_FACE_UP   = 0x00000004, // External faces generated triangulating the slices (pointing upwards)
    INTERNAL_FACE = 0x00000008, // Internal Faces
  //TINY_FEATURE  = 0x00000010, // Features so tiny that are better to remove prior tetmeshing the PLC...
};

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

typedef struct
{
    // Segment IDs
    //
    int slice_id   = -1;
    int seg_id     = -1;

    // Height (z is supposed to be the bulding direction)
    //
    double z_coord = 0.0;

    // IDs of the edge vertices
    //
    int endpoints[2] = { -1 , -1 };

    // IDs of the vertices to mesh in between the edge extrema. These can be:
    //  - a vertex of a polygon in the slice above lifted to the current slice
    //  - the intersection point (xy-wise) with a segment in the slice below
    //
    std::set<int>    bot_splits;         // use a set because it has unique IDs
    std::vector<int> ordered_bot_splits; // ordered from endpoints[0] to endpoints[1] (not included)

    // IDs of the vertices to mesh in between the edge extrema, after lifting the
    // egde to the next slice. These can be:
    //  - a vertex of a polygon in the slice above
    //  - the intersection point (xy-wise) with a segment in the slice above
    //
    std::set<int>    top_splits;         // use a set because it has unique IDs
    std::vector<int> ordered_top_splits; // ordered from endpoints[0] to endpoints[1] (not included)
}
E_data;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

typedef struct
{
    vec3d pos;

    // ID of the corresponding vertex on the next slice. This can be:
    // - a vertex of a polygon in the slice above
    // - a newly generated vertex having same x,y as pos and z of the slice above
    //
    int lifted_image = -1;
}
V_data;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

typedef struct
{
    std::vector<V_data>              v_list;
    std::vector<std::vector<E_data>> e_list;
}
SLICE2MESH_data;


#endif // COMMON_TYPES_H
