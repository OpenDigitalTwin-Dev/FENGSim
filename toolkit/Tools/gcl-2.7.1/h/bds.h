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
	bds.h

	bind stack
*/

typedef struct bds_bd {
	object	bds_sym;	/*  symbol  */
	object	bds_val;	/*  previous value of the symbol  */
} *bds_ptr;

EXTER bds_ptr bds_org,bds_limit,bds_top;

#ifdef KCLOVM

/* for multiprocessing */
EXTER struct bds_bd save_bind_stack[BDSSIZE + BDSGETA + BDSGETA];
EXTER bds_ptr bds_save_org;
EXTER bds_ptr bds_save_limit;
EXTER bds_ptr bds_save_top;

#endif

#define	bds_check  if (bds_top >= bds_limit)  bds_overflow()

/* do this so that an interrupt in the middle will leave the VALID
   part of the bds stack ie (<= bds_top) in a valid state, so
   that a throw out will be ok */
#define	bds_bind(sym, val)  \
	({object _sym=(sym),_val=(val);\
          if (++bds_top>=bds_limit) bds_overflow(); \
          bds_top->bds_sym=_sym;  \
          bds_top->bds_val=_sym->s.s_dbind;  \
          _sym->s.s_dbind=_val;})

#define	bds_unwind1 ((bds_top->bds_sym)->s.s_dbind = bds_top->bds_val, --bds_top)


