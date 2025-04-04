// Copyright (c) 1997  ETH Zurich (Switzerland).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Polyhedron/include/CGAL/Polyhedron_traits_3.h $
// $Id: Polyhedron_traits_3.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Lutz Kettner  <kettner@mpi-sb.mpg.de>)

#ifndef CGAL_POLYHEDRON_TRAITS_3_H
#define CGAL_POLYHEDRON_TRAITS_3_H 1

#include <CGAL/license/Polyhedron.h>


#include <CGAL/basic.h>

namespace CGAL {

template < class Kernel_ >
class Polyhedron_traits_3 {
public:
    typedef Kernel_                   Kernel;
    typedef typename Kernel::Point_3  Point_3;
    typedef typename Kernel::Plane_3  Plane_3;

    typedef typename Kernel::Construct_opposite_plane_3
                                      Construct_opposite_plane_3;
private:
    Kernel m_kernel;

public:
    Polyhedron_traits_3() {}
    Polyhedron_traits_3( const Kernel& kernel) : m_kernel(kernel) {}

    Construct_opposite_plane_3 construct_opposite_plane_3_object() const {
        return m_kernel.construct_opposite_plane_3_object();
    }
};

} //namespace CGAL

#endif // CGAL_POLYHEDRON_TRAITS_3_H //
