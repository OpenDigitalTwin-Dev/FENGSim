// Copyright (c) 2017  GeometryFactory (France).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Surface_mesh/include/CGAL/boost/graph/properties_Surface_mesh_time_stamp.h $
// $Id: properties_Surface_mesh_time_stamp.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Andreas Fabri

#ifndef CGAL_PROPERTIES_SURFACE_MESH_TIME_STAMP_H
#define CGAL_PROPERTIES_SURFACE_MESH_TIME_STAMP_H

#ifndef DOXYGEN_RUNNING

#include <CGAL/Surface_mesh.h>
#include <set>

namespace boost {


template <typename P>
struct property_map<CGAL::Surface_mesh<P>, CGAL::vertex_time_stamp_t>
{

  typedef typename boost::graph_traits<CGAL::Surface_mesh<P> >::vertex_descriptor vertex_descriptor;

  typedef typename CGAL::Surface_mesh<P>::template Property_map<vertex_descriptor, std::size_t> type;
  typedef type const_type;
};


template <typename P>
struct property_map<CGAL::Surface_mesh<P>, CGAL::halfedge_time_stamp_t>
{

  typedef typename boost::graph_traits<CGAL::Surface_mesh<P> >::halfedge_descriptor halfedge_descriptor;

  typedef typename CGAL::Surface_mesh<P>::template Property_map<halfedge_descriptor, std::size_t> type;
  typedef type const_type;
};


template <typename P>
struct property_map<CGAL::Surface_mesh<P>, CGAL::face_time_stamp_t>
{

  typedef typename boost::graph_traits<CGAL::Surface_mesh<P> >::face_descriptor face_descriptor;

  typedef typename CGAL::Surface_mesh<P>::template Property_map<face_descriptor, std::size_t> type;
  typedef type const_type;
};

} // namespace boost

namespace CGAL {


#define CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(Tag) \
  typename boost::lazy_disable_if<                      \
     boost::is_const<P>,                                \
     Get_pmap_of_surface_mesh<P, Tag >                  \
   >::type

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::vertex_time_stamp_t)
inline get(CGAL::vertex_time_stamp_t, Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::vertex_descriptor vertex_descriptor;
  return smesh. template add_property_map<vertex_descriptor,std::size_t>("v:time_stamp").first;
}

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::vertex_time_stamp_t)
inline get(CGAL::vertex_time_stamp_t, const Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::vertex_descriptor vertex_descriptor;
  return smesh. template property_map<vertex_descriptor,std::size_t>("v:time_stamp").first;
}

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::halfedge_time_stamp_t)
inline get(CGAL::halfedge_time_stamp_t, Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::halfedge_descriptor halfedge_descriptor;
  return smesh. template add_property_map<halfedge_descriptor,std::size_t>("h:time_stamp").first;
}

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::halfedge_time_stamp_t)
inline get(CGAL::halfedge_time_stamp_t, const Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::halfedge_descriptor halfedge_descriptor;
  return smesh. template property_map<halfedge_descriptor,std::size_t>("h:time_stamp").first;
}

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::face_time_stamp_t)
inline get(CGAL::face_time_stamp_t, Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::face_descriptor face_descriptor;
  return smesh. template add_property_map<face_descriptor,std::size_t>("v:time_stamp").first;
}

template <typename P>
CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE(CGAL::face_time_stamp_t)
inline get(CGAL::face_time_stamp_t, const Surface_mesh<P> & smesh)
{
  typedef typename boost::graph_traits<Surface_mesh<P> >::face_descriptor face_descriptor;
  return smesh. template property_map<face_descriptor,std::size_t>("v:time_stamp").first;
}
} // namespace CGAL

#undef CGAL_PROPERTY_SURFACE_MESH_RETURN_TYPE

#endif // DOXYGEN_RUNNING

#endif //CGAL_PROPERTIES_SURFACE_MESH_TIME_STAMP_H
