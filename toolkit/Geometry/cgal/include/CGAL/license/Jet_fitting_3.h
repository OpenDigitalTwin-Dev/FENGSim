// Copyright (c) 2016  GeometryFactory SARL (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Installation/include/CGAL/license/Jet_fitting_3.h $
// $Id: Jet_fitting_3.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s) : Andreas Fabri
//
// Warning: this file is generated, see include/CGAL/licence/README.md

#ifndef CGAL_LICENSE_JET_FITTING_3_H
#define CGAL_LICENSE_JET_FITTING_3_H

#include <CGAL/config.h>
#include <CGAL/license.h>

#ifdef CGAL_JET_FITTING_3_COMMERCIAL_LICENSE

#  if CGAL_JET_FITTING_3_COMMERCIAL_LICENSE < CGAL_RELEASE_DATE

#    if defined(CGAL_LICENSE_WARNING)

       CGAL_pragma_warning("Your commercial license for CGAL does not cover "
                           "this release of the Estimation of Local Differential Properties of Point-Sampled Surfaces package.")
#    endif

#    ifdef CGAL_LICENSE_ERROR
#      error "Your commercial license for CGAL does not cover this release \
              of the Estimation of Local Differential Properties of Point-Sampled Surfaces package. \
              You get this error, as you defined CGAL_LICENSE_ERROR."
#    endif // CGAL_LICENSE_ERROR

#  endif // CGAL_JET_FITTING_3_COMMERCIAL_LICENSE < CGAL_RELEASE_DATE

#else // no CGAL_JET_FITTING_3_COMMERCIAL_LICENSE

#  if defined(CGAL_LICENSE_WARNING)
     CGAL_pragma_warning("\nThe macro CGAL_JET_FITTING_3_COMMERCIAL_LICENSE is not defined."
                          "\nYou use the CGAL Estimation of Local Differential Properties of Point-Sampled Surfaces package under "
                          "the terms of the GPLv3+.")
#  endif // CGAL_LICENSE_WARNING

#  ifdef CGAL_LICENSE_ERROR
#    error "The macro CGAL_JET_FITTING_3_COMMERCIAL_LICENSE is not defined.\
            You use the CGAL Estimation of Local Differential Properties of Point-Sampled Surfaces package under the terms of \
            the GPLv3+. You get this error, as you defined CGAL_LICENSE_ERROR."
#  endif // CGAL_LICENSE_ERROR

#endif // no CGAL_JET_FITTING_3_COMMERCIAL_LICENSE

#endif // CGAL_LICENSE_JET_FITTING_3_H
