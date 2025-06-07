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
	sequence.d

	sequence routines
*/

#include "include.h"
#include "page.h"

/*
	I know the following name is not good.
*/
object
alloc_simple_vector(fixnum l) {

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_simple_vector);
  x->sv.sv_hasfillp = FALSE;
  x->sv.sv_adjustable = FALSE;
  x->sv.sv_dim = l;
  x->sv.sv_self = NULL;
  set_array_elttype(x,aet_object);
  x->sv.sv_rank = 1;

  return(x);

}

object
alloc_vector(fixnum l,enum aelttype aet) {

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_vector);
  x->v.v_hasfillp = TRUE;
  x->v.v_adjustable = TRUE;
  x->v.v_displaced = Cnil;
  x->v.v_dim = l;
  x->v.v_fillp = l;
  x->v.v_self = NULL;
  set_array_elttype(x,(short)aet);
  x->v.v_rank = 1;

  return(x);

}

object
alloc_simple_bitvector(fixnum l) {

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_simple_bitvector);
  x->sbv.sbv_hasfillp = FALSE;
  x->sbv.sbv_adjustable = FALSE;
  x->sbv.sbv_dim = l;
  x->sbv.sbv_offset = 0;
  x->sbv.sbv_self = NULL;
  set_array_elttype(x,aet_bit);
  x->sbv.sbv_rank = 1;

  return(x);

}

object
alloc_bitvector(fixnum l) {

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_bitvector);
  x->bv.bv_hasfillp = TRUE;
  x->bv.bv_adjustable = TRUE;
  x->bv.bv_displaced = Cnil;
  x->bv.bv_dim = l;
  x->bv.bv_fillp = l;
  x->bv.bv_offset = 0;
  x->bv.bv_self = NULL;
  set_array_elttype(x,aet_bit);
  x->bv.bv_rank = 1;

  return(x);

}


@(defun subseq (sequence start &optional end &aux x)
	int s, e;
	int i, j;
@
	s = fixnnint(start);
	if (end == Cnil)
		e = -1;
	else
		e = fixnnint(end);
	switch (type_of(sequence)) {
	case t_symbol:
		if (sequence == Cnil) {
			if (s > 0)
				goto ILLEGAL_START_END;
			if (e > 0)
				goto ILLEGAL_START_END;
			@(return Cnil)
		}
		FEwrong_type_argument(sLsequence, sequence);

	case t_cons:
		if (e >= 0)
			if ((e -= s) < 0)
				goto ILLEGAL_START_END;
		while (s-- > 0) {
			if (!consp(sequence))
				goto ILLEGAL_START_END;
			sequence = sequence->c.c_cdr;
		}
		if (e < 0)
			@(return `copy_list(sequence)`)
		x=n_cons_from_x(e,sequence);
		@(return x)

	case t_simple_vector:/*FIXME simple copies to simple*/
	case t_vector:
		if (s > VLEN(sequence))
			goto ILLEGAL_START_END;
		if (e < 0)
		  e = VLEN(sequence);
		else if (e < s || e > VLEN(sequence))
			goto ILLEGAL_START_END;
		x = sequence->v.v_elttype==aet_object ?
		  alloc_simple_vector(e-s) :
		  alloc_vector(e - s, sequence->v.v_elttype);
		array_allocself(x, FALSE,OBJNULL);
		switch (sequence->v.v_elttype) {
		case aet_object:
		  /*FIXME: memcpy size*/
			for (i = s, j = 0;  i < e;  i++, j++)
				x->v.v_self[j] = sequence->v.v_self[i];
			break;

		case aet_lf:
			for (i = s, j = 0;  i < e;  i++, j++)
			  ((double *)x->a.a_self)[j] =
			    ((double *)sequence->a.a_self)[i];
			break;

		case aet_sf:
			for (i = s, j = 0;  i < e;  i++, j++)
			  ((float *)x->a.a_self)[j] =
			    ((float *)sequence->a.a_self)[i];
			break;

		case aet_nnfix:
		case aet_fix:
			for (i = s, j = 0;  i < e;  i++, j++)
			  ((fixnum *)x->a.a_self)[j] =
			    ((fixnum *)sequence->a.a_self)[i];
			break;

		case aet_int:
		case aet_nnint:
		case aet_uint:
			for (i = s, j = 0;  i < e;  i++, j++)
				UINT_GCL(x, j) = UINT_GCL(sequence, i);
			break;

		case aet_short:
		case aet_nnshort:
		case aet_ushort:
			for (i = s, j = 0;  i < e;  i++, j++)
				USHORT_GCL(x, j) = USHORT_GCL(sequence, i);
			break;
		case aet_char:
		case aet_nnchar:
		case aet_uchar:
			for (i = s, j = 0;  i < e;  i++, j++)	
			  x->st.st_self[j] = sequence->st.st_self[i];
			break;
	
		}
		@(return x)


	case t_simple_string:
	case t_string:
		if (s > VLEN(sequence))
			goto ILLEGAL_START_END;
		if (e < 0)
		  e = VLEN(sequence);
		else if (e < s || e > VLEN(sequence))
			goto ILLEGAL_START_END;
	       {BEGIN_NO_INTERRUPT;	
		x = alloc_simple_string(e - s);
		x->st.st_self = alloc_relblock(e - s);
		END_NO_INTERRUPT;}
		for (i = s, j = 0;  i < e;  i++, j++)
			x->st.st_self[j] = sequence->st.st_self[i];
		@(return x)

	case t_simple_bitvector:
	case t_bitvector:
		if (s > VLEN(sequence))
			goto ILLEGAL_START_END;
		if (e < 0)
		  e = VLEN(sequence);
		else if (e < s || e > VLEN(sequence))
			goto ILLEGAL_START_END;
		{BEGIN_NO_INTERRUPT;
		x = alloc_simple_bitvector(e - s);
		x->bv.bv_self = alloc_relblock(ceil((e-s),BV_ALLOC)*sizeof(*x->bv.bv_self));
		s += sequence->bv.bv_offset;
		e += sequence->bv.bv_offset;
		for (i = s, j = 0;  i < e;  i++, j++)
		  if (BITREF(sequence,i))
		    SET_BITREF(x,j);
		  else
		    CLEAR_BITREF(x,j);
		END_NO_INTERRUPT;}
		@(return x)

	default:
		FEwrong_type_argument(sLsequence, vs_base[0]);
	}

ILLEGAL_START_END:
	FEerror("~S and ~S are illegal as :START and :END~%\
for the sequence ~S.", 3, start, end, sequence);
@)

LFD(Lcopy_seq)()
{
	check_arg(1);
	vs_push(small_fixnum(0));
	Lsubseq();
}

int
length(x)
object x;
{
	int i;

	switch (type_of(x)) {
	case t_symbol:
		if (x == Cnil)
			return(0);
		FEwrong_type_argument(sLsequence, x);
		return(0);
	case t_cons:

#define cendp(obj) ((!consp(obj)))
		for (i = 0;  !cendp(x);  i++, x = x->c.c_cdr)
			;
		if (x==Cnil) return(i);
		FEwrong_type_argument(sLlist,x);
		return(0);


	case t_simple_vector:
	case t_simple_string:
	case t_simple_bitvector:
	case t_vector:
	case t_string:
	case t_bitvector:
	  return(VLEN(x));

	default:
		FEwrong_type_argument(sLsequence, x);
		return(0);
	}
}

LFD(Llength)()
{
	check_arg(1);
	vs_base[0] = make_fixnum(length(vs_base[0]));
}

LFD(Lreverse)()
{
	check_arg(1);
	vs_base[0] = reverse(vs_base[0]);
}

object
reverse(seq)
object seq;
{
	object x, y, *v;
	int i, j, k;

	switch (type_of(seq)) {
	case t_symbol:
		if (seq == Cnil)
			return(Cnil);
		FEwrong_type_argument(sLsequence, seq);

	case t_cons:
		v = vs_top;
		vs_push(Cnil);
		for (x = seq;  !endp(x);  x = x->c.c_cdr)
			*v = make_cons(x->c.c_car, *v);
		return(vs_pop);

	case t_simple_vector:
	case t_vector:
		x = seq;
		k = VLEN(x);
		y = x->v.v_elttype==aet_object ?
		  alloc_simple_vector(k) : alloc_vector(k, x->v.v_elttype);
		vs_push(y);
		array_allocself(y, FALSE,OBJNULL);
		switch (x->v.v_elttype) {
		case aet_object:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
				y->v.v_self[j] = x->v.v_self[i];
			break;

		case aet_lf:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
			  ((double *)y->a.a_self)[j] = ((double *)x->a.a_self)[i];
			break;

		case aet_sf:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
			  ((float *)y->a.a_self)[j] = ((float *)x->a.a_self)[i];
			break;

		case aet_fix:
		case aet_nnfix:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
			  ((fixnum *)y->a.a_self)[j] = ((fixnum *)x->a.a_self)[i];
			break;

		case aet_int:
		case aet_nnint:
		case aet_uint:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
				UINT_GCL(y, j) = UINT_GCL(x, i);
			break;

		case aet_short:
		case aet_nnshort:
		case aet_ushort:
			for (j = k - 1, i = 0;  j >=0;  --j, i++)
				USHORT_GCL(y, j) = USHORT_GCL(x, i);
			break;
		case aet_char:
		case aet_nnchar:
		case aet_uchar:
		    goto TYPE_STRING;
		}
		return(vs_pop);

	case t_simple_string:
	case t_string:
		x = seq;
		y = alloc_simple_string(VLEN(x));
		TYPE_STRING:
		{BEGIN_NO_INTERRUPT;
		vs_push(y);
		y->st.st_self
		  = alloc_relblock(VLEN(x));
		for (j = VLEN(x) - 1, i = 0;  j >=0;  --j, i++)
			y->st.st_self[j] = x->st.st_self[i];
		END_NO_INTERRUPT;}
		return(vs_pop);

	case t_simple_bitvector:
	case t_bitvector:
		x = seq;
		{BEGIN_NO_INTERRUPT;	
		  y = alloc_simple_bitvector(VLEN(x));
		vs_push(y);
		y->bv.bv_self=alloc_relblock(ceil(VLEN(x),BV_ALLOC)*sizeof(*y->bv.bv_self));
		for (j = VLEN(x) - 1, i = x->bv.bv_offset;
		     j >=0;
		     --j, i++)
		  if (BITREF(x,i))
		    SET_BITREF(y,j);
		  else
		    CLEAR_BITREF(y,j);
		END_NO_INTERRUPT;}	
		return(vs_pop);

	default:
		FEwrong_type_argument(sLsequence, seq);
		return(Cnil);
	}
}

LFD(Lnreverse)()
{
	check_arg(1);
	vs_base[0] = nreverse(vs_base[0]);
}

object /*FIXME boot*/
nreverse(seq)
object seq;
{
	object x, y, z;
	int i, j, k;

	switch (type_of(seq)) {
	case t_symbol:
		if (seq == Cnil)
			return(Cnil);
		FEwrong_type_argument(sLsequence, seq);

	case t_cons:
		for (x = Cnil, y = seq;  !endp(y->c.c_cdr);) {
			z = y;
			y = y->c.c_cdr;
			z->c.c_cdr = x;
			x = z;
		}
		y->c.c_cdr = x;
		return(y);

	case t_simple_vector:
	case t_vector:
		x = seq;
		k = VLEN(x);
		switch (x->v.v_elttype) {
		case aet_object:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				y = x->v.v_self[i];
				x->v.v_self[i] = x->v.v_self[j];
				x->v.v_self[j] = y;
			}
			return(seq);

		case aet_lf:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				longfloat y;
				y = ((double *)x->a.a_self)[i];
				((double *)x->a.a_self)[i] = ((double *)x->a.a_self)[j];
				((double *)x->a.a_self)[j] = y;
			}
			return(seq);

		case aet_sf:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				shortfloat y;
				y = ((float *)x->a.a_self)[i];
				((float *)x->a.a_self)[i] = ((float *)x->a.a_self)[j];
				((float *)x->a.a_self)[j] = y;
			}
			return(seq);

		case aet_fix:
		case aet_nnfix:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				fixnum y;
				y = ((fixnum *)x->a.a_self)[i];
				((fixnum *)x->a.a_self)[i] = ((fixnum *)x->a.a_self)[j];
				((fixnum *)x->a.a_self)[j] = y;
			}
			return(seq);

		case aet_int:
		case aet_nnint:
		case aet_uint:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				unsigned int y;
				y = UINT_GCL(x, i);
				UINT_GCL(x, i) = UINT_GCL(x, j);
				UINT_GCL(x, j) = y;
			}
			return(seq);

		case aet_short:
		case aet_nnshort:
		case aet_ushort:
			for (i = 0, j = k - 1;  i < j;  i++, --j) {
				unsigned short y;
				y = USHORT_GCL(x, i);
				USHORT_GCL(x, i) = USHORT_GCL(x, j);
				USHORT_GCL(x, j) = y;
			}
			return(seq);
		case aet_char:
		case aet_nnchar:
		case aet_uchar:
		    goto TYPE_STRING;
		}

	case t_simple_string:
	case t_string:
		x = seq;
	TYPE_STRING:	
		for (i = 0, j = VLEN(x) - 1;  i < j;  i++, --j) {
			k = x->st.st_self[i];
			x->st.st_self[i] = x->st.st_self[j];
			x->st.st_self[j] = k;
		}
		return(seq);

	case t_simple_bitvector:
	case t_bitvector:
		x = seq;
		for (i = x->bv.bv_offset,
		       j = VLEN(x) + x->bv.bv_offset - 1;
		     i < j;
		     i++, --j) {
		  k = BITREF(x,i);
		  if (BITREF(x,j))
		    SET_BITREF(x,i);
		  else
		    CLEAR_BITREF(x,i);
		  if (k)
		    SET_BITREF(x,j);
		  else
		    CLEAR_BITREF(x,j);
		}
		return(seq);

	default:
		FEwrong_type_argument(sLsequence, seq);
		return(Cnil);
	}
}


void
gcl_init_sequence_function()
{
	make_function("SUBSEQ", Lsubseq);
	make_function("COPY-SEQ", Lcopy_seq);
	make_function("LENGTH", Llength);
	make_function("REVERSE", Lreverse);
	make_function("NREVERSE", Lnreverse);
}
