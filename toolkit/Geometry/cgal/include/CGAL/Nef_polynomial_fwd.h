// Copyright (c) 1999,2003,2004,2005 Max-Planck-Institute Saarbruecken (Germany).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Nef_2/include/CGAL/Nef_polynomial_fwd.h $
// $Id: Nef_polynomial_fwd.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Andreas Fabri, Stefan Schirra, Sylvain Pion

#ifndef CGAL_NEF_POLYNOMIAL_FWD_H
#define CGAL_NEF_POLYNOMIAL_FWD_H

#include <CGAL/enum.h>

// Forward declarations of functions over Polynomial and Nef_polynomial

namespace CGAL {

namespace Nef {
template <typename> class Polynomial;

template <typename ET> double to_double(const Polynomial<ET> &);

//template <typename ET>
//std::pair<double,double> to_interval(const Polynomial<ET> &);

template <typename ET>
Sign sign(const Polynomial<ET> &);

template <typename ET>
Polynomial<ET> abs(const Polynomial<ET> &);

template <typename ET>
Polynomial<ET> gcd(const Polynomial<ET> &, const Polynomial<ET> &);

}
// Nef_polynomial

template <typename> class Nef_polynomial;


} //namespace CGAL

#endif // CGAL_NEF_POLYNOMIAL_FWD_H
