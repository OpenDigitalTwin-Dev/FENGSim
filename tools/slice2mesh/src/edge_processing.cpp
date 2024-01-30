#include "edge_processing.h"
#include <cinolib/intersection.h>

enum
{
    _A_ = 0,
    _B_ = 1,
    _C_ = 2,
    _D_ = 3,
};

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void lift_unlifted_edge_extrema(const SlicedObj<> & obj, SLICE2MESH_data & data)
{
    for(uint sid=0; sid<obj.num_slices()-1;         ++sid) // for each slice
    for(uint eid=0; eid<data.e_list.at(sid).size(); ++eid) // for each segment
    for(int  ext=0; ext<2;                          ++ext) // for each segment extrema
    {
        E_data & e   = data.e_list.at(sid).at(eid);
        int      vid = e.endpoints[ext];
        if(data.v_list.at(vid).lifted_image == -1)
        {
            data.v_list.at(vid).lifted_image = data.v_list.size(); // fresh_id
            V_data vd;
            vd.pos      = data.v_list.at(vid).pos;
            vd.pos.z()  = obj.slice_z(sid+1);
            data.v_list.push_back(vd);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

std::vector<int> order_split_points(const SLICE2MESH_data & data, const int vid_beg, const int vid_end, const std::set<int> & splits)
{
    std::vector<int> ordered_splits;
    if (splits.empty()) return ordered_splits;

    vec3d A = data.v_list.at(vid_beg).pos;
    vec3d B = data.v_list.at(vid_end).pos;
    vec3d u = B-A;
    assert(u.length() > 0);

    std::vector<std::pair<double,int>> dot_list;
    for(int vid : splits)
    {
        vec3d  P   = data.v_list.at(vid).pos;
        vec3d  v   = P - A;
        double dot = u.dot(v);
        if (dot < 0)
        {
            //cout << endl;
            //cout << vid_beg << "\t" << vid_end << endl;
            //cout << A << endl;
            //cout << B << endl;
            //cout << P << endl;
            //cout << "AB " << (A-B).length() << endl;
            //cout << "PA " << (P-A).length() << endl;
            //cout << "PB " << (P-B).length() << endl;
            //cout << "uv"  << u.dot(v) << endl;
            //cout << endl;
            //assert(dot < 0)
        }
        else
        if (u.dot(v) >= u.dot(u))
        {
            //cout << endl;
            //cout << vid_beg << "\t" << vid_end << endl;
            //cout << A << endl;
            //cout << B << endl;
            //cout << P << endl;
            //cout << "AB " << (A-B).length() << endl;
            //cout << "PA " << (P-A).length() << endl;
            //cout << "PB " << (P-B).length() << endl;
            //cout << "uv - uu " << u.dot(v) << "\t" << u.dot(u) << endl;
            //cout << endl;
            //assert(u.dot(v) < u.dot(u));
        }
        else
        dot_list.push_back(std::make_pair(dot, vid));
    }

    std::sort(dot_list.begin(), dot_list.end());

    for(std::pair<double,int> p : dot_list)
    {
        ordered_splits.push_back(p.second);
    }

    return ordered_splits;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void order_split_points(const SlicedObj<> & obj, SLICE2MESH_data & data)
{
    for(uint sid=0; sid<obj.num_slices()-1;         ++sid) // for each slice
    for(uint eid=0; eid<data.e_list.at(sid).size(); ++eid) // for each segment
    {
        E_data & e = data.e_list.at(sid).at(eid);
        int vid_A  = e.endpoints[0];
        int vid_B  = e.endpoints[1];
        int vid_C  = data.v_list.at(vid_A).lifted_image;
        int vid_D  = data.v_list.at(vid_B).lifted_image;

        assert(vid_A != -1); assert(vid_B != -1);
        assert(vid_C != -1); assert(vid_D != -1);

        e.ordered_bot_splits = order_split_points(data, vid_A, vid_B, e.bot_splits);
        e.ordered_top_splits = order_split_points(data, vid_C, vid_D, e.top_splits);
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void edge_wise_intersections(const SlicedObj<> & obj, SLICE2MESH_data & data)
{
    for(uint sid=0; sid<obj.num_slices()-1;           ++sid) // for each slice
    for(uint ei =0; ei <data.e_list.at(sid  ).size(); ++ei ) // for each segment below
    for(uint ej =0; ej <data.e_list.at(sid+1).size(); ++ej ) // for each segment above
    {
        process_edge_pair(data.v_list, data.e_list.at(sid).at(ei), data.e_list.at(sid+1).at(ej));
    }

    lift_unlifted_edge_extrema(obj, data);
    order_split_points(obj, data);
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool lift_bot_left(const bool is[4], const int vids[4], std::vector<V_data> & points);
bool lift_bot_right(const bool is[4], const int vids[4], std::vector<V_data> & points);

bool set_top_split (const bool is[4], const int vids[4], std::vector<V_data> & points, E_data & e_below);
bool set_bot_split (const bool is[4], const int vids[4], std::vector<V_data> & points, E_data & e_above);

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void process_edge_pair(std::vector<V_data> & points,
                       E_data              & e_below,
                       E_data              & e_above)
{
    int vids[4] =
    {
        e_below.endpoints[0],
        e_below.endpoints[1],
        e_above.endpoints[0],
        e_above.endpoints[1]
    };

    vec2d A(points[vids[_A_]].pos.x(), points[vids[_A_]].pos.y());
    vec2d B(points[vids[_B_]].pos.x(), points[vids[_B_]].pos.y());
    vec2d C(points[vids[_C_]].pos.x(), points[vids[_C_]].pos.y());
    vec2d D(points[vids[_D_]].pos.x(), points[vids[_D_]].pos.y());

    std::vector<vec2d> res;
    segment2D_intersection(A,B,C,D,res);
    //segment_intersection_2D(A, B, C, D, res); // discard z-coord

    for(vec2d P : res)
    {
        bool is[4] = // check for similarity with segment endpoints
        {
            ((A-P).length() < 1e-10),
            ((B-P).length() < 1e-10),
            ((C-P).length() < 1e-10),
            ((D-P).length() < 1e-10)
        };

        if (is[_A_]) assert(!is[_B_]);
        if (is[_B_]) assert(!is[_A_]);
        if (is[_C_]) assert(!is[_D_]);
        if (is[_D_]) assert(!is[_C_]);

        // debug
        //if (!is[_A_] && !is[_B_] && !is[_C_] && !is[_D_])
        //{
        //    vec2d u = P - A;
        //    vec2d v = B - A;
        //    if (u.dot(v) < 0 || u.dot(v) > v.dot(v))
        //    {
        //        cout.precision(17);
        //        cout << A << "\t" << B << endl;
        //        cout << C << "\t" << D << endl;
        //        cout << P << endl;
        //        cout << "P-A dot B-A   " << u.dot(v) << endl;
        //        cout << "B-A dot B-A   " << u.dot(u) << endl;
        //        cout << "d(P,A)   " << (P-A).length() << endl;
        //        cout << "d(P,B)   " << (P-B).length() << endl;
        //        cout << "is " << is[0] << " " << is[1] << " " << is[2] << " " << is[3] << endl;
        //    }
        //    assert(u.dot(v) >= 0);
        //    assert(u.dot(v) <= v.dot(v));
        //}

        bool b0 = lift_bot_left (is, vids, points);
        bool b1 = lift_bot_right(is, vids, points);
        bool b2 = set_top_split (is, vids, points, e_below);
        bool b3 = set_bot_split (is, vids, points, e_above);

        if (b0 || b1 || b2 || b3) // if any of the routines above asks for a new vertex, add it
        {
            V_data vd;
            vd.pos.x()  = P.x();
            vd.pos.y()  = P.y();
            vd.pos.z()  = e_above.z_coord;
            points.push_back(vd);
        }
    }
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool lift_bot_left(const bool is[4], const int vids[4], std::vector<V_data> & points)
{
    if (points[vids[_A_]].lifted_image != -1) return false;

    if (is[_A_] && is[_C_]) points[vids[_A_]].lifted_image = vids[_C_]; else
    if (is[_A_] && is[_D_]) points[vids[_A_]].lifted_image = vids[_D_]; else
    if (is[_A_])
    {
        points[vids[_A_]].lifted_image = points.size(); // fresh_vid
        return true;
    }
    return false;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool lift_bot_right(const bool is[4], const int vids[4], std::vector<V_data> & points)
{
    if (points[vids[_B_]].lifted_image != -1) return false;

    if (is[_B_] && is[_C_]) points[vids[_B_]].lifted_image = vids[_C_]; else
    if (is[_B_] && is[_D_]) points[vids[_B_]].lifted_image = vids[_D_]; else
    if (is[_B_])
    {
        points[vids[_B_]].lifted_image = points.size(); // fresh_vid
        return true;
    }
    return false;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool set_top_split(const bool is[4], const int vids[4], std::vector<V_data> & points, E_data & e_below)
{
    if (is[_C_] && !is[_A_] && !is[_B_])
    {
        e_below.top_splits.insert(vids[_C_]);
    }
    else if (is[_D_] && !is[_A_] && !is[_B_])
    {
        e_below.top_splits.insert(vids[_D_]);
    }
    else if (!is[_A_] && !is[_B_] && !is[_C_] && !is[_D_])
    {
        e_below.top_splits.insert(points.size());  // fresh_vid
        return true;
    }
    return false;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool set_bot_split(const bool is[4], const int vids[4], std::vector<V_data> & points, E_data & e_above)
{
    if ( is[_A_] && !is[_C_] && !is[_D_])
    {
        assert(points[vids[_A_]].lifted_image != -1);
        e_above.bot_splits.insert(points[vids[_A_]].lifted_image);
    }
    else if ( is[_B_] && !is[_C_] && !is[_D_])
    {
        assert(points[vids[_B_]].lifted_image != -1);
        e_above.bot_splits.insert(points[vids[_B_]].lifted_image);
    }
    else if (!is[_A_] && !is[_B_] && !is[_C_] && !is[_D_])
    {
        e_above.bot_splits.insert(points.size());  // fresh_vid
        return true;
    }   
    return false;
}
