/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
 Copyright (C) 2024 Camm Maguire

This file is part of GNU Common Lisp, herein referred to as GCL

GCL is free software; you can redistribute it and/or modify it under
the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GCL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
License for more details.

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

*/

/*
	include.h
*/

#ifndef INCLUDE_H

#define INCLUDE_H

/* whether to use prototypes or not */
#ifdef __STDC__
#define P__(x) x
#else
#define P__(x)
#endif

#include "options.h"
#include "gclincl.h"

#ifdef __GNUC__
#ifndef alloca
#define alloca __builtin_alloca
#endif
#endif

#ifdef IN_NUM_CO
#ifdef HAVE_ISNORMAL
#define ISNORMAL(a) isnormal(a)
#else
#ifdef HAVE_IEEEFP
#include <ieeefp.h>
#define ISNORMAL(a) (fpclass(a)>=FP_NZERO)
#else
#define ISNORMAL(a) ((sizeof (a) == sizeof (float)) ? \
       gcl_isnormal_float(a) : \
       gcl_isnormal_double(a))
#endif
#endif
#endif

#ifdef NEED_ISFINITE
#ifdef HAVE_ISFINITE
#define ISFINITE(a) isfinite(a)
#else
#ifdef HAVE_FINITE
#include <ieeefp.h>
#define ISFINITE(a) finite(a)
#else
#error "No isfinite found"
#endif
#endif
#endif

#include "config.h"

#ifdef IN_NUM_CO
#ifdef HAVE_VALUES_H
#include <values.h>
#endif
#ifdef HAVE_FLOAT_H
#include <float.h>
#endif
#endif

#ifdef UNIX
#include <ctype.h>
#define	isalphanum(x)	isalnum(x)
#endif

#if defined(GMP) || defined(NEED_MP_H)
#include "../h/mp.h"
#endif

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif
#ifdef USE_READLINE
#include <readline/readline.h>
#endif

#include "../h/sfun_argd.h"
#include "../h/compbas.h"
#include "../h/enum.h"
#include "../h/pageinfo.h"
#include "../h/lu.h"
#include "../h/globals.h"
#include "../h/fixnum.h"
#include "../h/type.h"
#include "../h/object.h"
#include "../h/vs.h"
#include "../h/bds.h"
#include "../h/frame.h"
#include "../h/lex.h"
#include "../h/eval.h"

#include "../h/compprotos.h"
#include "../h/protoize.h"
#include "../h/compprotos.h"
#include "../h/notcomp.h"
#include "../h/funlink.h"
#include "../h/att_ext.h"
#ifndef INICOMP
#include "../h/new_decl.h"
#endif
#include "compbas2.h"
#include "compat.h"

#include "../h/rgbc.h"
#include "../o/regexp.h"

#include "../h/immnum.h"
#endif
