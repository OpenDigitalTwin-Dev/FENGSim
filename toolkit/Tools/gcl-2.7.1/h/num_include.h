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
	number routine include file
*/

#define WSIZ 32
#define MASK	0x7fffffff

#ifdef MV


#endif

object Vrandom_state;

#ifndef PI
#define PI			3.141592653589793
#endif

#define LOG_WORD_SIZE           (8*SIZEOF_LONG)
#define MOST_POSITIVE_FIX	((long)((((unsigned long)1)<<(LOG_WORD_SIZE-1))-1))
#define MOST_NEGATIVE_FIX	( - MOST_POSITIVE_FIX - 1 )
