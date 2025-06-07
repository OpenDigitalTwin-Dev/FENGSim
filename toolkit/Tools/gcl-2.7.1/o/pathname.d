/* -*-C-*- */
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
	pathname.d
	IMPLEMENTATION-DEPENTENT

	This file contains those functions that interpret namestrings.
*/

#include <string.h>
#include "include.h"


DEFUN("INIT-PATHNAME",object,fSinit_pathname,SI,7,7,NONE,OO,OO,OO,OO,
      (object host,object device,object directory,object name,object type,object version,object namestring),"") {

  object x=alloc_object(t_pathname);

  x->pn.pn_host=host;
  x->pn.pn_device=device;
  x->pn.pn_directory=directory;
  x->pn.pn_name=name;
  x->pn.pn_type=type;
  x->pn.pn_version=version;
  x->pn.pn_namestring=namestring;

  RETURN1(x);

}


void
gcl_init_pathname(void) {
}

void
gcl_init_pathname_function() {
}
