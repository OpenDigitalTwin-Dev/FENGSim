// Copyright (c) 1997
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Stream_support/include/CGAL/IO/generic_copy_OFF.h $
// $Id: generic_copy_OFF.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Lutz Kettner  <kettner@mpi-sb.mpg.de>

#ifndef CGAL_IO_GENERIC_COPY_OFF_H
#define CGAL_IO_GENERIC_COPY_OFF_H 1

#include <CGAL/basic.h>
#include <cstddef>
#include <CGAL/IO/File_header_OFF.h>
#include <CGAL/IO/File_scanner_OFF.h>
#include <iostream>

namespace CGAL {

template <class Writer>
void
generic_copy_OFF( File_scanner_OFF& scanner,
                  std::ostream& out,
                  Writer& writer) {
    std::istream& in = scanner.in();
    // scans a polyhedral surface in OFF from `in' and writes it
    // to `out' in the format provided by `writer'.
    if ( ! in) {
        if ( scanner.verbose()) {
            std::cerr << " " << std::endl;
            std::cerr << "generic_copy_OFF(): "
                         "input error: file format is not in OFF."
                      << std::endl;
        }
        return;
    }

    // Print header. Number of halfedges is only trusted if it is
    // a polyhedral surface.
    writer.write_header( out,
                         scanner.size_of_vertices(),
                         scanner.polyhedral_surface() ?
                             scanner.size_of_halfedges() : 0,
                         scanner.size_of_facets());

    // read in all vertices
    double  x,  y,  z;  // Point coordinates.
    std::size_t  i;
    for ( i = 0; i < scanner.size_of_vertices(); i++) {
        scanner.scan_vertex( x, y, z);
        writer.write_vertex( x, y, z);
        scanner.skip_to_next_vertex( i);
    }

    // read in all facets
    writer.write_facet_header();
    for ( i = 0; i < scanner.size_of_facets(); i++) {
        if ( ! in)
            return;
        std::size_t no;
        scanner.scan_facet( no, i);
        writer.write_facet_begin( no);
        for ( std::size_t j = 0; j < no; j++) {
          std::size_t index;
            scanner.scan_facet_vertex_index( index, i);
            writer.write_facet_vertex_index( index);
        }
        writer.write_facet_end();
        scanner.skip_to_next_facet( i);
    }
    writer.write_footer();
}

template <class Writer>
void
generic_copy_OFF( std::istream& in, std::ostream& out, Writer& writer,
                  bool verbose = false) {
    // scans a polyhedral surface in OFF from `in' and writes it
    // to `out' in the format provided by `writer'.
    File_scanner_OFF scanner( in, verbose);
    generic_copy_OFF( scanner, out, writer);
}

} //namespace CGAL
#endif // CGAL_IO_GENERIC_COPY_OFF_H //
// EOF //
