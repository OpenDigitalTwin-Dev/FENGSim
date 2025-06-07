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

	frame.c

	frame and non-local jump
*/

#include "include.h"

void
unwind(frame_ptr fr, object tag)
{
        signals_allowed = 0;
	nlj_fr = fr;
	nlj_tag = tag;
	nlj_active = TRUE;
	while (frs_top != fr
	       && frs_top->frs_class == FRS_CATCH
	       && frs_top >= frs_org
		/*
		&& frs_top->frs_class != FRS_PROTECT
		&& frs_top->frs_class != FRS_CATCHALL
		*/
	      ) {
		--frs_top;
	}
	if (frs_top<frs_org) {
	  frs_top=frs_org;
	  FEerror("Cannot unwind frame", 0);
	}
	lex_env = frs_top->frs_lex;
	ihs_top = frs_top->frs_ihs;
	bds_unwind(frs_top->frs_bds_top);
	in_signal_handler = frs_top->frs_in_signal_handler;
	signals_allowed=sig_normal;
	longjmp((void *)frs_top->frs_jmpbuf, 0);
	/* never reached */
}

frame_ptr frs_sch (object frame_id)
{
	frame_ptr top;

	for (top = frs_top;  top >= frs_org;  top--)
		if (top->frs_val == frame_id && top->frs_class == FRS_CATCH)
			return(top);
	return(NULL);
}

frame_ptr frs_sch_catch(object frame_id)
{
  frame_ptr top;
  
  for(top = frs_top;  top >= frs_org  ;top--)
    if ((top->frs_val == frame_id && top->frs_class == FRS_CATCH)
	|| top->frs_class == FRS_CATCHALL
	)
      return(top);
  return(NULL);
}



