// Copyright (c) 1997-2000  Max-Planck-Institute Saarbruecken (Germany).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Nef_2/include/CGAL/Nef_2/Object_index.h $
// $Id: Object_index.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Michael Seel <seel@mpi-sb.mpg.de>

#ifndef CGAL_NEF_2_OBJECT_INDEX_H
#define CGAL_NEF_2_OBJECT_INDEX_H

#include <CGAL/license/Nef_2.h>


#include <CGAL/basic.h>
#include <CGAL/Unique_hash_map.h>
#include <string>
#include <sstream>

namespace CGAL {

template <typename I>
class Object_index {
  char _prefix;
  CGAL::Unique_hash_map<I,int> _index;
public:
  Object_index() : _prefix('\0'), _index(-1) {}
  Object_index(I first, I beyond, char c=' ') : _prefix(c), _index(-1)
  { for(int i=0 ; first!=beyond; ++i,++first) _index[first]=i; }
  int operator[](const I& it) const { return _index[it]; }
  int& operator[](const I& it) { return _index[it]; }

  void index(I first, I beyond, char c=' ')
  { _prefix=c;
    for(int i=0 ; first!=beyond; ++i,++first) _index[first]=i;
  }
  std::string operator()(const I& it, bool verbose=true) const
  { if (verbose && _index[it]==-1) return "nil";
    if (verbose && _index[it]==-2) return "end";
    std::ostringstream os;
    if (verbose) os << _prefix;
    os << _index[it];
    return os.str();
  }
};

} //namespace CGAL

#endif //CGAL_NEF_2_OBJECT_INDEX_H
