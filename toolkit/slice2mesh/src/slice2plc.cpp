#include "slice2plc.h"
#include "edge_processing.h"
#include "trianglulate.h"
#include <cinolib/profiler.h>

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void initialize(const SlicedObj<> & obj, SLICE2MESH_data & data)
{
    data.v_list.clear();
    data.e_list.clear();
    data.e_list.resize(obj.num_slices());

    for(uint sid=0; sid<obj.num_slices(); ++sid)
    {
        std::vector<vec3d> v;
        std::vector<uint>  e;
        obj.slice_segments(sid,v,e);

        uint base_addr = data.v_list.size();

        for(vec3d p : v)
        {
            V_data vd;
            vd.pos = p;
            data.v_list.push_back(vd);
        }

        for(uint eid=0; eid<e.size()/2; ++eid)
        {
            E_data ed;
            ed.endpoints[0] = base_addr + e.at(2*eid  );
            ed.endpoints[1] = base_addr + e.at(2*eid+1);
            ed.slice_id     = sid;
            ed.seg_id       = eid;
            ed.z_coord      = obj.slice_z(sid);
            data.e_list.at(sid).push_back(ed);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void mesh_vertical(const SlicedObj<> & obj,
                   const SLICE2MESH_data     & data,
                         std::vector<uint>   & tris,
                         std::vector<int>    & labels)
{
    for(uint sid=0; sid<obj.num_slices()-1;         ++sid)
    for(uint eid=0; eid<data.e_list.at(sid).size(); ++eid) // for each segment
    {
        const E_data & e = data.e_list.at(sid).at(eid);
        int vids[4] =
        {
            e.endpoints[0],
            e.endpoints[1],
            data.v_list[e.endpoints[0]].lifted_image,
            data.v_list[e.endpoints[1]].lifted_image
        };
        assert(vids[0] != -1); assert(vids[1] != -1);
        assert(vids[2] != -1); assert(vids[3] != -1);

        int n_new_tris = triangulate_quad(vids, e.ordered_bot_splits, e.ordered_top_splits, tris);

        for(int i=0; i<n_new_tris; ++i) labels.push_back(SRF_FACE_VERT);
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void mesh_horizontal(const SlicedObj<> & obj,
                     const SLICE2MESH_data     & data,
                           std::vector<uint>   & tris,
                           std::vector<int>    & labels)
{
    for(uint sid=0; sid<obj.num_slices(); ++sid) // for each slice
    {
        std::vector<uint> segs;
        std::set<uint>    unique_slice_verts;

        for(uint eid=0; eid<data.e_list.at(sid).size(); ++eid) // for each segment
        {
            const E_data & e = data.e_list.at(sid).at(eid);
            int  vid_A = e.endpoints[0]; assert(vid_A != -1);
            int  vid_B = e.endpoints[1]; assert(vid_B != -1);

            segs.push_back(vid_A);
            unique_slice_verts.insert(vid_A);
            for(int vid : e.ordered_bot_splits)
            {
                segs.push_back(vid);
                segs.push_back(vid);
                unique_slice_verts.insert(vid);
            }
            segs.push_back(vid_B);
            unique_slice_verts.insert(vid_B);
        }

        if(sid > 0)
        {
            for(uint eid=0; eid<data.e_list.at(sid-1).size(); ++eid) // for each segment
            {
                const E_data & e = data.e_list.at(sid-1).at(eid);
                int  vid_A = data.v_list[e.endpoints[0]].lifted_image; assert(vid_A != -1);
                int  vid_B = data.v_list[e.endpoints[1]].lifted_image; assert(vid_B != -1);

                segs.push_back(vid_A);
                unique_slice_verts.insert(vid_A);
                for(int vid : e.ordered_top_splits)
                {
                    segs.push_back(vid);
                    segs.push_back(vid);
                    unique_slice_verts.insert(vid);
                }
                segs.push_back(vid_B);
                unique_slice_verts.insert(vid_B);
            }
        }

        std::vector<double> coords_in;
        std::vector<uint>   verts;
        std::map<uint,uint> v_map;
        uint fresh_id = 0;
        for(int vid : unique_slice_verts)
        {
            verts.push_back(vid);
            v_map[vid] = fresh_id;
            ++fresh_id;

            coords_in.push_back(data.v_list.at(vid).pos.x());
            coords_in.push_back(data.v_list.at(vid).pos.y());
        }

        std::vector<uint> segs_in;
        for(int vid : segs)
        {
            segs_in.push_back(v_map.at(vid));
        }

        if(coords_in.empty()) continue;

        //for(auto c : coords_in) std::cout << "coord: " << c << std::endl;
        //for(auto c : segs_in)   std::cout << "seg: " << c << std::endl;
        std::vector<double> holes_in, coords_out;
        std::vector<uint> tris_out;
        triangle_wrap(coords_in, segs_in, holes_in, obj.slice_z(sid), "Q", coords_out, tris_out);

//        static int count = 0;
//        Trimesh<> m_tmp;
//        triangle_wrap(coords_in, segs_in, holes_in, obj.slice(sid).z_coord, "Q", m_tmp);
//        std::string fname("./debug");
//        fname += std::to_string(count++) + ".off";
//        m_tmp.save(fname.c_str());

        //if (coords_in.size()/2 != coords_out.size()/3)
        //{
        //    std::cout << "coords in: " << coords_in.size()/2 << "\tcoords_out: " << coords_out.size()/3 << std::endl;
        //    //Trimesh m(coords4_out, tris_out);
        //    //m.save("/Users/cino/Desktop/test.obj");
        //    //assert(coords_in.size()/2 == coords_out.size()/3);
        //}
        for(uint tid=0; tid<tris_out.size()/3; ++tid)
        {
            int v0 = tris_out.at(3*tid+0);
            int v1 = tris_out.at(3*tid+1);
            int v2 = tris_out.at(3*tid+2);

            int nverts = coords_in.size()/2;
            if (v0 >= nverts || v1 >= nverts || v2 >= nverts)
            {
                std::cout << "triangle " << tid << "(" << v0 << "," << v1 << "," << v2 << ") as it contains a newly generated vertex! (N_IN_VERTS " << nverts << ")" << std::endl;
                continue;
            }

            int vid_A = verts.at(v0);
            int vid_B = verts.at(v1);
            int vid_C = verts.at(v2);

            vec3d p = (data.v_list.at(vid_A).pos +
                       data.v_list.at(vid_B).pos +
                       data.v_list.at(vid_C).pos) / 3.0;

            bool belongs_to_curr = obj.slice_contains(sid, vec2d(p));
            bool belongs_to_prev = (sid > 0 && obj.slice_contains(sid-1,vec2d(p)));

            if (belongs_to_curr || belongs_to_prev)
            {
                tris.push_back(vid_A);
                tris.push_back(vid_B);
                tris.push_back(vid_C);

                if ( belongs_to_curr && !belongs_to_prev)
                {
                    labels.push_back(SRF_FACE_DOWN);
                    std::swap(tris.at(tris.size()-1),tris.at(tris.size()-2)); // flip triangle orientation
                }
                else
                if (!belongs_to_curr &&  belongs_to_prev) labels.push_back(SRF_FACE_UP);   else
                if ( belongs_to_curr &&  belongs_to_prev && sid < obj.num_slices()-1) labels.push_back(INTERNAL_FACE);
                else
                {
                    labels.push_back(SRF_FACE_UP);
                }
            }
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void slice2plc(const SlicedObj<> & obj, Trimesh<> & plc)
{
    assert(obj.num_slices() >= 2);

    SLICE2MESH_data data;

    Profiler profiler;

    profiler.push("Slice2Mesh Initialization");
    initialize(obj, data);
    edge_wise_intersections(obj, data);
    profiler.pop();

    std::vector<vec3d> verts;
    for(V_data p : data.v_list) verts.push_back(p.pos);

    std::vector<uint> tris;
    std::vector<int>  labels;
    profiler.push("Horizontal meshing");
    mesh_horizontal(obj, data, tris, labels);
    profiler.pop();
    profiler.push("Vertical meshing");
    mesh_vertical(obj, data, tris, labels);
    profiler.pop();

    plc = Trimesh<>(verts, tris);
    for(uint pid=0; pid<plc.num_polys(); ++pid)
    {
        plc.poly_data(pid).label = labels.at(pid);
    }

    plc.poly_color_wrt_label();
}
