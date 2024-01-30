#include "plc2tet.h"
#include <cinolib/tetgen_wrap.h>
#include <cinolib/interval.h>

void plc2tet(const Trimesh<>   & plc,
             const SlicedObj<> & obj,
             const std::string & flags,
                   Tetmesh<>   & m)
{
    tetgen_wrap(plc, flags.c_str(), m);

    //uint ns = obj.num_slices()-1;
    //std::vector<float> min_z(ns), max_z(ns);
    //for(uint sid=0; sid<ns; ++sid)
    //{
    //    min_z.at(sid) = obj.slice_z(sid  );
    //    max_z.at(sid) = obj.slice_z(sid+1);
    //}

    //for(uint pid=0; pid<m.num_polys(); ++pid)
    //{
    //    float z = m.poly_centroid(pid).z();
    //    uint sid=0;
    //    while(sid<ns && !is_into_interval(z, min_z.at(sid), max_z.at(sid))) ++sid;
    //    assert(sid<ns);
    //    m.poly_data(pid).label = sid;
    //}

    //m.poly_color_wrt_label(true);
}
