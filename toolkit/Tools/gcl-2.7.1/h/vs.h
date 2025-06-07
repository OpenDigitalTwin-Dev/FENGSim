/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa

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
	vs.h

	value stack
*/

EXTER object *vs_org,*vs_limit,*vs_base,*vs_top;

#define	vs_push(x_)	*vs_top++ = (x_)

#define	vs_pop		*--vs_top
#define	vs_popp		--vs_top
#define	vs_head		vs_top[-1]

#define	vs_mark		object *old_vs_top=vs_top
#define	vs_reset	vs_top=old_vs_top

#define	vs_check	if (vs_top>=vs_limit)  vs_overflow()

#define	vs_check_push(x_)  (vs_top >= vs_limit ?  (object)vs_overflow() : (*vs_top++=(x_)))

#define	check_arg(n_)  if (vs_top-vs_base!=(n_))  check_arg_failed(n_)

#define CHECK_ARG_RANGE(n_,m_) if (VFUN_NARGS<n_ || VFUN_NARGS>m_) check_arg_range(n_,m_)

#define	MMcheck_arg(n_)  do {\
			 if (vs_top-vs_base<(n_))  too_few_arguments();  \
			 else if (vs_top-vs_base>(n_)) too_many_arguments();} while (0)

#define vs_reserve(x_)	if(vs_base+(x_) >= vs_limit) vs_overflow();

