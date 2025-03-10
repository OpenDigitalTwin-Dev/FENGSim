// Copyright (c) 1997-2001
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Optimisation_basic/include/CGAL/Optimisation/Construct_point_2.h $
// $Id: Construct_point_2.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sven Schoenherr <sven@inf.ethz.ch>

#ifndef CGAL_OPTIMISATION_CONSTRUCT_POINT_2_H
#define CGAL_OPTIMISATION_CONSTRUCT_POINT_2_H

#include <CGAL/Point_2.h>
#include <vector>
#include <functional>
#include <iterator>

namespace CGAL {

// Class declaration
// =================
template < class K >
class _Construct_point_2;

// Class interface
// ===============
template < class K_ >
class _Construct_point_2 {
  public:
    // self
    typedef  K_                         K;
    typedef  _Construct_point_2<K>      Self;

    // types
    typedef  typename K::Point_2        Point;

    // creation
    _Construct_point_2( ) { }

    // operations
    template < class InputIterator >
    Point
    operator() ( int, InputIterator first, InputIterator last) const
    {
        InputIterator i(first);
        typename K::RT x = *(i++);
        typename K::RT y = *(i++);
        typedef typename K::Construct_point_2 Construct_point_2;
        Construct_point_2 construct_point_2 = K().construct_point_2_object();
        if (i==last) {
            return construct_point_2(x,y);
        } else {
            typename K::RT h = *(i++);
            return construct_point_2(x,y,h);
        }
    }
};

} //namespace CGAL

#endif // CGAL_OPTIMISATION_CONSTRUCT_POINT_2_H

// ===== EOF ==================================================================
