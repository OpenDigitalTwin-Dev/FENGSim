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
	string.d

	string routines
*/

#include <string.h>
#include "include.h"

object sKstart1;
object sKend1;
object sKstart2;
object sKend2;
object sKinitial_element;
object sKelement_type;

object
alloc_string(fixnum l) {/*FIXME consolidate these through fSmake_vector*/

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_string);
  x->st.st_hasfillp = TRUE;
  x->st.st_adjustable = TRUE;
  set_array_elttype(x,aet_ch);
  x->st.st_rank = 1;
  x->st.st_displaced=Cnil;
  x->st.st_dim = l;
  x->st.st_fillp = l;
  x->st.st_self = NULL;

  return(x);

}

object
alloc_simple_string(fixnum l) {

  object x;

  if (l<0 || l>=ARRAY_DIMENSION_LIMIT)
    TYPE_ERROR(make_fixnum(l),list(3,sLinteger,make_fixnum(0),MMcons(make_fixnum(ARRAY_DIMENSION_LIMIT),Cnil)));

  x = alloc_object(t_simple_string);
  x->sst.sst_hasfillp = FALSE;
  x->sst.sst_adjustable = FALSE;
  set_array_elttype(x,aet_ch);
  x->sst.sst_rank = 1;
  x->sst.sst_dim = l;
  x->sst.sst_self = NULL;

  return(x);

}

/*
	Make_simple_string(s) makes a simple string from C string s.
*/
object
make_simple_string(s)
const char *s;
{
	int l, i;
	char *p;
	object x;
	vs_mark;
	{BEGIN_NO_INTERRUPT;
	for (l = 0;  s[l] != '\0';  l++);
	x = alloc_simple_string(l);
	vs_push(x);
	p = alloc_relblock(l);
	for (i = 0;  i < l;  i++)
		p[i] = s[i];
	x->st.st_self = p;
	vs_reset;
	END_NO_INTERRUPT;}
	return(x);
}

object
make_string(s)
char *s;
{
	int l, i;
	char *p;
	object x;
	vs_mark;
	{BEGIN_NO_INTERRUPT;	
	for (l = 0;  s[l] != '\0';  l++)
		;
	x = alloc_string(l);
	vs_push(x);
	p = alloc_relblock(l);
	for (i = 0;  i < l;  i++)
		p[i] = s[i];
	x->st.st_self = p;
	vs_reset;
	END_NO_INTERRUPT;}	
	return(x);
}

/*
	This correponds to string= (just the string equality).
*/
bool
string_eq(x, y)
object x, y;
{
	int i, j;


	if (!stringp(x) || !stringp(y))
	  error("string expected");
	i = VLEN(x);
	j = VLEN(y);
	if (i != j)
		return(FALSE);
	for (i = 0;  i < j;  i++)
		if (x->st.st_self[i] != y->st.st_self[i])
			return(FALSE);
	return(TRUE);
}

/*
	This corresponds to string-equal
	(string equality ignoring the case).
*/
bool
string_equal(x, y)
object x, y;
{
	int i, j;
	char *p, *q;

/*
	if (type_of(x) != t_string || type_of(y) != t_string)
		error("string expected");
*/
	i = VLEN(x);
	j = VLEN(y);
	if (i != j)
		return(FALSE);
	p = x->st.st_self;
	q = y->st.st_self;
	for (i = 0;  i < j;  i++)
		if ((isLower(p[i]) ? p[i] - ('a' - 'A') : p[i])
		 != (isLower(q[i]) ? q[i] - ('a' - 'A') : q[i]))
			return(FALSE);
	return(TRUE);
}

/*
	Copy_simple_string(x) copies string x to a simple string.
*/
object
copy_simple_string(x)
object x;
{
	object y;
	int i;
	vs_mark;

	vs_push(x);

	{BEGIN_NO_INTERRUPT;
	y = alloc_object(t_simple_string);
	y->st.st_hasfillp = FALSE;
	y->st.st_adjustable = FALSE;
	y->st.st_dim = VLEN(x);
	set_array_elttype(y,aet_ch);
	y->st.st_rank = 1;
	y->st.st_self = NULL;
	vs_push(y);
	y->st.st_self = alloc_relblock(VLEN(x));
	for (i = 0;  i < VLEN(x);  i++)
		y->st.st_self[i] = x->st.st_self[i];
	vs_reset;
	END_NO_INTERRUPT;}
	return(y);
}

object
copy_string(x)
object x;
{
	object y;
	int i;
	vs_mark;

	vs_push(x);
/*
	if (type_of(x) != t_string)
		error("string expected");
*/
	{BEGIN_NO_INTERRUPT;	
	y = alloc_object(t_string);
	y->st.st_hasfillp = TRUE;
	y->st.st_adjustable = TRUE;
	y->st.st_dim = VLEN(x);
	VSET_MAX_FILLP(y);
	set_array_elttype(y,aet_ch);
	/* y->st.st_eltsize = elt_size(aet_ch); */
	y->st.st_rank = 1;
	SET_ADISP(y,Cnil);
	y->st.st_self = NULL;
	vs_push(y);
	y->st.st_self = alloc_relblock(VLEN(x));
	for (i = 0;  i < VLEN(x);  i++)
		y->st.st_self[i] = x->st.st_self[i];
	vs_reset;
	END_NO_INTERRUPT;	}
	return(y);
}

object
coerce_to_string(x)
object x;
{
	object y;
	vs_mark;

	switch (type_of(x)) {
	case t_symbol:
	  return(x->s.s_name);

	case t_fixnum:
		x = coerce_to_character(x);
		vs_push(x);

	case t_character:
		{BEGIN_NO_INTERRUPT;	
		y = alloc_simple_string(1);
		vs_push(y);
		y->st.st_self = alloc_relblock(1);
		y->st.st_self[0] = char_code(x);
		vs_reset;
		END_NO_INTERRUPT;}	
		return(y);

	case t_simple_string:
	case t_string:
		return(x);
	default:
		break;
	}
	vs_push(x);
	x=wrong_type_argument(sLstring,x);
	vs_popp;
	return(Cnil);
}


void
get_string_start_end(str, start, end, ps, pe)
object str, start, end;
int *ps, *pe;
{
	if (start == Cnil)
		*ps = 0;
	else if (type_of(start) != t_fixnum)
		goto E;
	else {
		*ps = fix(start);
		if (*ps < 0)
			goto E;
	}
	if (end == Cnil) {
	  *pe = VLEN(str);
		if (*pe < *ps)
			goto E;
	} else if (type_of(end) != t_fixnum)
		goto E;
	else {
		*pe = fix(end);
		if (*pe < *ps || *pe > VLEN(str))
			goto E;
	}
	return;

E:
	FEerror("~S and ~S are illegal as :START and :END~%\
for the str ~S.", 3, start, end, str);
}

@(defun string_eq (string1 string2
		   &key start1 end1 start2 end2)
	int s1=0, e1=0, s2=0, e2=0;
@
	string1 = coerce_to_string(string1);
	string2 = coerce_to_string(string2);
	get_string_start_end(string1, start1, end1, &s1, &e1);
	get_string_start_end(string2, start2, end2, &s2, &e2);
	if (e1 - s1 != e2 - s2)
		@(return Cnil)
	while (s1 < e1)
		if (string1->st.st_self[s1++] !=
		    string2->st.st_self[s2++])
			@(return Cnil)
	@(return Ct)
@)

@(defun string_equal (string1 string2
		      &key start1 end1 start2 end2)
	int s1=0, e1=0, s2=0, e2=0;
	int i1, i2;
@
	string1 = coerce_to_string(string1);
	string2 = coerce_to_string(string2);
	get_string_start_end(string1, start1, end1, &s1, &e1);
	get_string_start_end(string2, start2, end2, &s2, &e2);
	if (e1 - s1 != e2 - s2)
		@(return Cnil)
	while (s1 < e1) {
		i1 = string1->st.st_self[s1++];
		i2 = string2->st.st_self[s2++];
		if (isLower(i1))
			i1 -= 'a' - 'A';
		if (isLower(i2))
			i2 -= 'a' - 'A';
		if (i1 != i2)
			@(return Cnil)
	}
	@(return Ct)
@)


int string_sign, string_boundary;

@(static defun string_cmp (string1 string2
		    &key start1 end1 start2 end2)
	int s1=0, e1=0, s2=0, e2=0;
	int i1, i2;
	int s;
@
	string1 = coerce_to_string(string1);
	string2 = coerce_to_string(string2);
	get_string_start_end(string1, start1, end1, &s1, &e1);
	get_string_start_end(string2, start2, end2, &s2, &e2);
	while (s1 < e1) {
		if (s2 == e2)
			@(return `string_sign>0 ? Cnil : make_fixnum(s1)`)
		i1 = string1->ust.ust_self[s1];
		i2 = string2->ust.ust_self[s2];
		if (string_sign == 0) {
			if (i1 != i2)
				@(return `make_fixnum(s1)`)
		} else {
			s = string_sign*(i2-i1);
			if (s > 0)
				@(return `make_fixnum(s1)`)
			if (s < 0)
				@(return Cnil)
		}
		s1++;
		s2++;
	}
	if (s2 == e2)
		@(return `string_boundary==0 ? make_fixnum(s1) : Cnil`)
	@(return `string_sign>=0 ? make_fixnum(s1) : Cnil`)
@)

LFD(Lstring_l)()  { string_sign =  1;  string_boundary = 1;  FFN(Lstring_cmp)(); }
LFD(Lstring_g)()  { string_sign = -1;  string_boundary = 1;  FFN(Lstring_cmp)(); }
LFD(Lstring_le)() { string_sign =  1;  string_boundary = 0;  FFN(Lstring_cmp)(); }
LFD(Lstring_ge)() { string_sign = -1;  string_boundary = 0;  FFN(Lstring_cmp)(); }
LFD(Lstring_neq)() { string_sign = 0;  string_boundary = 1;  FFN(Lstring_cmp)(); }

@(static defun string_compare (string1 string2
			&key start1 end1 start2 end2)
	int s1=0, e1=0, s2=0, e2=0;
	int i1, i2;
	int s;
@
	string1 = coerce_to_string(string1);
	string2 = coerce_to_string(string2);
	get_string_start_end(string1, start1, end1, &s1, &e1);
	get_string_start_end(string2, start2, end2, &s2, &e2);
	while (s1 < e1) {
		if (s2 == e2)
			@(return `string_sign>0 ? Cnil : make_fixnum(s1)`)
		i1 = string1->ust.ust_self[s1];
		if (isLower(i1))
			i1 -= 'a' - 'A';
		i2 = string2->ust.ust_self[s2];
		if (isLower(i2))
			i2 -= 'a' - 'A';
		if (string_sign == 0) {
			if (i1 != i2)
				@(return `make_fixnum(s1)`)
		} else {
			s = string_sign*(i2-i1);
			if (s > 0)
				@(return `make_fixnum(s1)`)
			if (s < 0)
				@(return Cnil)
		}
		s1++;
		s2++;
	}
	if (s2 == e2)
		@(return `string_boundary==0 ? make_fixnum(s1) : Cnil`)
	@(return `string_sign>=0 ? make_fixnum(s1) : Cnil`)
@)

LFD(Lstring_lessp)()      { string_sign =  1; string_boundary = 1;  FFN(Lstring_compare)(); }
LFD(Lstring_greaterp)()   { string_sign = -1; string_boundary = 1;  FFN(Lstring_compare)(); }
LFD(Lstring_not_greaterp)(){ string_sign =  1; string_boundary = 0;  FFN(Lstring_compare)(); }
LFD(Lstring_not_lessp)()   { string_sign = -1; string_boundary = 0;  FFN(Lstring_compare)(); }
LFD(Lstring_not_equal)()   { string_sign =  0; string_boundary = 1;  FFN(Lstring_compare)(); }

/* element_type is currently ignored -- character == base-char == standard-char */
@(defun make_string (size
		     &key (initial_element `code_char(' ')` ) element_type 
		     &aux x)
	int i;
@
	while (type_of(size) != t_fixnum || fix(size) < 0)
		size
		= wrong_type_argument(TSnon_negative_integer, size);
		/*  bignum not allowed, this is PRACTICAL!!  */
	while (type_of(initial_element) != t_character ||
	       char_bits(initial_element) != 0 ||
	       char_font(initial_element) != 0)
		initial_element
		= wrong_type_argument(sLcharacter, initial_element);
	{BEGIN_NO_INTERRUPT;	
	x = alloc_simple_string(fix(size));
	x->st.st_self = alloc_relblock(fix(size));
	for (i = 0;  i < fix(size);  i++)
		x->st.st_self[i] = char_code(initial_element);
	END_NO_INTERRUPT;	}
	@(return x)
@)

static bool
member_char(c, char_bag)
int c;
object char_bag;
{

	int i, f;

	switch (type_of(char_bag)) {
	case t_symbol:
	case t_cons:
		while (!endp(char_bag)) {
			if (type_of(char_bag->c.c_car) == t_character
 			    &&	c == char_code(char_bag->c.c_car))
				return(TRUE);
			char_bag = char_bag->c.c_cdr;
		}
		return(FALSE);

	case t_simple_vector:
	case t_vector:
	  for (i = 0, f = VLEN(char_bag);  i < f;  i++) {
			if (type_of(char_bag->v.v_self[i]) == t_character
			  && c == char_code(char_bag->v.v_self[i]))
				return(TRUE);
		}
		return(FALSE);

	case t_simple_string:
	case t_string:
	  for (i = 0, f = VLEN(char_bag);  i < f;  i++) {
			if (c == char_bag->st.st_self[i])
				return(TRUE);
		}
		return(FALSE);

	case t_simple_bitvector:
	case t_bitvector:
		return(FALSE);

	default:
		FEerror("~S is not a sequence.", 1, char_bag);
		return(FALSE);
	}
}

/*static void Lstring_trim0();*/

@(static defun string_trim0 (char_bag strng &aux res)
	int i, j, k;
@
	strng = coerce_to_string(strng);
	i = 0;
        j = VLEN(strng) - 1;
	if (left_trim)
		for (;  i <= j;  i++)
			if (!member_char(strng->st.st_self[i], char_bag))
				break;
	if (right_trim)
		for (;  j >= i;  --j)
			if (!member_char(strng->st.st_self[j], char_bag))
				break;
	k = j - i + 1;
	{BEGIN_NO_INTERRUPT;	
	res = alloc_simple_string(k);
	res->st.st_self = alloc_relblock(k);
	for (j = 0;  j < k;  j++)
		res->st.st_self[j] = strng->st.st_self[i + j];
	END_NO_INTERRUPT;	}
	@(return res)
@)

LFD(Lstring_trim)() { left_trim = right_trim = TRUE; FFN(Lstring_trim0)(); }
LFD(Lstring_left_trim)() { left_trim = TRUE; right_trim = FALSE; FFN(Lstring_trim0)(); }
LFD(Lstring_right_trim)() { left_trim = FALSE; right_trim = TRUE; FFN(Lstring_trim0)();}

static int char_upcase(c, bp)
int c, *bp;
{
	if (isLower(c))
		return(c - ('a' - 'A'));
	else
		return(c);
}

static int char_downcase(c, bp)
int c, *bp;
{
	if (isUpper(c))
		return(c + ('a' - 'A'));
	else
		return(c);
}

static int char_capitalize(c, bp)
int c, *bp;
{
	if (isLower(c)) {
		if (*bp)
			c -= 'a' - 'A';
		*bp = FALSE;
	} else if (isUpper(c)) {
		if (!*bp)
			c += 'a' - 'A';
		*bp = FALSE;
	} else if (!isDigit(c))
		*bp = TRUE;
	else
		*bp = FALSE;
	return(c);
}

@(static defun string_case (strng &key start end &aux conv)
	int s=0, e=0, i;
	bool b;
@
	strng = coerce_to_string(strng);
	get_string_start_end(strng, start, end, &s, &e);
	conv = copy_simple_string(strng);
	b = TRUE;
	for (i = s;  i < e;  i++)
		conv->st.st_self[i] =
		(*casefun)(conv->st.st_self[i], &b);
	@(return conv)
@)

LFD(Lstring_upcase)()     { casefun =     char_upcase;  FFN(Lstring_case)(); }
LFD(Lstring_downcase)()   { casefun =   char_downcase;  FFN(Lstring_case)(); }
LFD(Lstring_capitalize)() { casefun = char_capitalize;  FFN(Lstring_case)(); }


@(static defun nstring_case (strng &key start end)
	int s=0, e=0, i;
	bool b;
@
	check_type_string(&strng);
	get_string_start_end(strng, start, end, &s, &e);
	b = TRUE;
	for (i = s;  i < e;  i++)
		strng->st.st_self[i] =
		(*casefun)(strng->st.st_self[i], &b);
	@(return strng)
@)

LFD(Lnstring_upcase)()     { casefun =     char_upcase;  FFN(Lnstring_case)(); }
LFD(Lnstring_downcase)()   { casefun =   char_downcase;  FFN(Lnstring_case)(); }
LFD(Lnstring_capitalize)() { casefun = char_capitalize;  FFN(Lnstring_case)(); }


@(defun string (x)
@
	@(return `coerce_to_string(x)`)
@)

DEFUN("STRING-CONCATENATE",object,fLstring_concatenate,SI,0,63,NONE,OO,OO,OO,OO,(object first,...),"") {

  fixnum i,l,m,narg=INIT_NARGS(0);
  object x,ll=Cnil,z;
  va_list ap;

  va_start(ap,first);
  vs_base=vs_top;
  for (l=i=0;(z=NEXT_ARG(narg,ap,ll,first,OBJNULL))!=OBJNULL;i++) {
    vs_push(coerce_to_string(z));
    l += VLEN(vs_head);
  }
  va_end(ap);

  {
    object *p;
    BEGIN_NO_INTERRUPT;	
    x=alloc_simple_string(l);
    (x)->st.st_self = alloc_relblock(l);
    for (l=0,p=vs_base;p<vs_top && (m=VLEN(*p))>=0;p++,l+=m)
      memcpy(x->st.st_self+l,(*p)->st.st_self,m);
    END_NO_INTERRUPT;

  }	

  RETURN1(x);

}
  
/* static void */
/* FFN(siLstring_concatenate)() */
/* { */
/* 	int narg, i, l, m; */
/* 	object *v; */

/* 	narg = vs_top - vs_base; */
/* 	for (i = 0, l = 0;  i < narg;  i++) { */
/* 		vs_base[i] = coerce_to_string(vs_base[i]); */
/* 		l += vs_base[i]->st.st_fillp; */
/* 	} */
/* 	v = vs_top; */
/* 	{BEGIN_NO_INTERRUPT;	 */
/* 	vs_push(alloc_simple_string(l)); */
/* 	(*v)->st.st_self = alloc_relblock(l); */
/* 	for (i = 0, l = 0;  i < narg;  i++) */
/* 		for (m = 0;  m < vs_base[i]->st.st_fillp;  m++) */
/* 			(*v)->st.st_self[l++] */
/* 			= vs_base[i]->st.st_self[m]; */
/* 	vs_base[0] = *v; */
/* 	vs_top = vs_base + 1; */
/* 	END_NO_INTERRUPT;}	 */
/* } */

void
gcl_init_string_function()
{
	sKstart1 = make_keyword("START1");
	sKend1 = make_keyword("END1");
	sKstart2 = make_keyword("START2");
	sKend2 = make_keyword("END2");
	sKinitial_element = make_keyword("INITIAL-ELEMENT");
	sKelement_type = make_keyword("ELEMENT-TYPE");
	sKstart = make_keyword("START");
	sKend = make_keyword("END");

	make_function("STRING=", Lstring_eq);
	make_function("STRING-EQUAL", Lstring_equal);
	make_function("STRING<", Lstring_l);
	make_function("STRING>", Lstring_g);
	make_function("STRING<=", Lstring_le);
	make_function("STRING>=", Lstring_ge);
	make_function("STRING/=", Lstring_neq);
	make_function("STRING-LESSP", Lstring_lessp);
	make_function("STRING-GREATERP", Lstring_greaterp);
	make_function("STRING-NOT-LESSP", Lstring_not_lessp);
	make_function("STRING-NOT-GREATERP", Lstring_not_greaterp);
	make_function("STRING-NOT-EQUAL", Lstring_not_equal);
	make_function("MAKE-STRING", Lmake_string);
	make_function("STRING-TRIM", Lstring_trim);
	make_function("STRING-LEFT-TRIM", Lstring_left_trim);
	make_function("STRING-RIGHT-TRIM", Lstring_right_trim);
	make_function("STRING-UPCASE", Lstring_upcase);
	make_function("STRING-DOWNCASE", Lstring_downcase);
	make_function("STRING-CAPITALIZE", Lstring_capitalize);
	make_function("NSTRING-UPCASE", Lnstring_upcase);
	make_function("NSTRING-DOWNCASE", Lnstring_downcase);
	make_function("NSTRING-CAPITALIZE", Lnstring_capitalize);
	make_function("STRING", Lstring);

}
