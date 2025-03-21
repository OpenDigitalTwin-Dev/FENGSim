/********************************************************************************
*  This file is part of CinoLib                                                 *
*  Copyright(C) 2016: Marco Livesu                                              *
*                                                                               *
*  The MIT License                                                              *
*                                                                               *
*  Permission is hereby granted, free of charge, to any person obtaining a      *
*  copy of this software and associated documentation files (the "Software"),   *
*  to deal in the Software without restriction, including without limitation    *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,     *
*  and/or sell copies of the Software, and to permit persons to whom the        *
*  Software is furnished to do so, subject to the following conditions:         *
*                                                                               *
*  The above copyright notice and this permission notice shall be included in   *
*  all copies or substantial portions of the Software.                          *
*                                                                               *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS *
*  IN THE SOFTWARE.                                                             *
*                                                                               *
*  Author(s):                                                                   *
*                                                                               *
*     Marco Livesu (marco.livesu@gmail.com)                                     *
*     http://pers.ge.imati.cnr.it/livesu/                                       *
*                                                                               *
*     Italian National Research Council (CNR)                                   *
*     Institute for Applied Mathematics and Information Technologies (IMATI)    *
*     Via de Marini, 6                                                          *
*     16149 Genoa,                                                              *
*     Italy                                                                     *
*********************************************************************************/
#include <cinolib/grid_mesh.h>
#include <cinolib/serialize_2D.h>
#include <vector>

namespace cinolib
{

template<class M, class V, class E, class P>
CINO_INLINE
void grid_mesh(const uint                quads_per_row,
               const uint                quads_per_col,
                     Quadmesh<M,V,E,P> & m)
{
    std::vector<vec3d> points;
    std::vector<uint>  polys;
    for(uint r=0; r<=quads_per_row; ++r)
    for(uint c=0; c<=quads_per_col; ++c)
    {
        points.push_back(vec3d(c,r,0));

        if (r<quads_per_row && c<quads_per_col)
        {
            polys.push_back(serialize_2D_index(r  , c,   quads_per_col+1));
            polys.push_back(serialize_2D_index(r  , c+1, quads_per_col+1));
            polys.push_back(serialize_2D_index(r+1, c+1, quads_per_col+1));
            polys.push_back(serialize_2D_index(r+1, c,   quads_per_col+1));
        }
    }
    m = Quadmesh<M,V,E,P>(points, polys);
}

}
