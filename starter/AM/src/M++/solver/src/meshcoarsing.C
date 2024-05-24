#include<iostream>
#include "tetgen.h"

void meshcoarsing () {
    tetgenio in, out, addin, bgmin;
    in.load_stl("./solver/conf/geo/test.stl");
    tetgenbehavior b;
    b.parse_commandline("-pk");
    tetrahedralize(&b, &in, &out);

    return;
    out.save_nodes("barout");
    out.save_elements("barout");
    out.save_faces("barout");
    out.save_poly("barout");
}
