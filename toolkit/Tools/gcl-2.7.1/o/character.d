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
	character.d

	character routines
*/

#include "include.h"

#define	CHFONTLIM	1	/*  character font limit  */
#define	CHBITSLIM	1	/*  character bits limit  */
#define	CHCODEFLEN	8	/*  character code field length  */
#define	CHFONTFLEN	0	/*  character font field length  */
#define	CHBITSFLEN      0	/*  character bits field length  */

@(defun standard_char_p (c)
	int i;
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	i = char_code(c);
	if ((' ' <= i && i < '\177') || i == '\n')
		@(return Ct)
	@(return Cnil)
@)

@(defun graphic_char_p (c)
	int i;
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	i = char_code(c);
	if (' ' <= i && i < '\177')
		@(return Ct)
	@(return Cnil)
@)

@(defun alpha_char_p (c)
	int i;
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	i = char_code(c);
	if (isalpha(i))
		@(return Ct)
	else
		@(return Cnil)
@)

@(defun upper_case_p (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	if (isUpper(char_code(c)))
		@(return Ct)
	@(return Cnil)
@)

@(defun lower_case_p (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	if (isLower(char_code(c)))
		@(return Ct)
	@(return Cnil)
@)

@(defun both_case_p (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	if (isUpper(char_code(c)) || isLower(char_code(c)))
		@(return Ct)
	else
		@(return Cnil)
@)

/*
	Digitp(i, r) returns the weight of code i
	as a digit of radix r.
	If r > 36 or i is not a digit, -1 is returned.
*/
int
digitp(i, r)
int i, r;
{
	if ('0' <= i && i <= '9' && 1 < r && i < '0' + r)
		return(i - '0');
	if ('A' <= i && 10 < r && r <= 36 && i < 'A' + (r - 10))
		return(i - 'A' + 10);
	if ('a' <= i && 10 < r && r <= 36 && i < 'a' + (r - 10))
		return(i - 'a' + 10);
	return(-1);
}

@(defun digit_char_p (c &optional (r `make_fixnum(10)`))
	int d;
@
	check_type_character(&c);
	check_type_non_negative_integer(&r);
	if (type_of(r) == t_bignum)
		@(return Cnil)
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	d = digitp(char_code(c), fix(r));
	if (d < 0)
		@(return Cnil)
	@(return `make_fixnum(d)`)
@)

@(defun alphanumericp (c)
	int i;
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	i = char_code(c);
	if (isalphanum(i))
		@(return Ct)
	else
		@(return Cnil)
@)

bool
char_eq(x, y)
object x, y;
{
	return(char_code(x) == char_code(y)
	    && char_bits(x) == char_bits(y)
	    && char_font(x) == char_font(y));
}

@(defun char_eq (c &rest)
	int i;
@
	check_type_character(&c);
	for (i = 0;  i < narg;  i++)
		check_type_character(&vs_base[i]);
	for (i = 1;  i < narg;  i++)
		if (!char_eq(vs_base[i-1], vs_base[i]))
			@(return Cnil)
	@(return Ct)
@)

@(defun char_neq (c &rest)
	int i, j;
@
	check_type_character(&c);
	for (i = 0;  i < narg;  i++)
		check_type_character(&vs_base[i]);
	if (narg == 0)
		@(return Ct)
	for (i = 1;  i < narg;  i++)
		for (j = 0;  j < i;  j++)
			if (char_eq(vs_base[j], vs_base[i]))
				@(return Cnil)
	@(return Ct)
@)


static int
char_cmp(x, y)
object x, y;
{
	if (char_font(x) < char_font(y))
		return(-1);
	if (char_font(x) > char_font(y))
		return(1);
	if (char_bits(x) < char_bits(y))
		return(-1);
	if (char_bits(x) > char_bits(y))
		return(1);
	if (char_code(x) < char_code(y))
		return(-1);
	if (char_code(x) > char_code(y))
		return(1);
	return(0);
}

static void
Lchar_cmp(s, t)
int s, t;
{
	int narg, i;

	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0; i < narg; i++)
		check_type_character(&vs_base[i]);
	for (i = 1; i < narg; i++)
		if (s*char_cmp(vs_base[i], vs_base[i-1]) < t) {
			vs_top = vs_base+1;
			vs_base[0] = Cnil;
			return;
		}
	vs_top = vs_base+1;
	vs_base[0] = Ct;
}

LFD(Lchar_l)()  { Lchar_cmp( 1, 1); }
LFD(Lchar_g)()  { Lchar_cmp(-1, 1); }
LFD(Lchar_le)() { Lchar_cmp( 1, 0); }
LFD(Lchar_ge)() { Lchar_cmp(-1, 0); }


bool
char_equal(x, y)
object x, y;
{
	int i, j;

	i = char_code(x);
	j = char_code(y);
	if (isLower(i))
		i -= 'a' - 'A';
	if (isLower(j))
		j -= 'a' - 'A';
	return(i == j);
}

@(defun char_equal (c &rest)
	int i;
@
	check_type_character(&c);
	for (i = 0;  i < narg;  i++)
		check_type_character(&vs_base[i]);
	for (i = 1;  i < narg;  i++)
		if (!char_equal(vs_base[i-1], vs_base[i]))
			@(return Cnil)
	@(return Ct)
@)

@(defun char_not_equal (c &rest)
	int i, j;
@
	check_type_character(&c);
	for (i = 0;  i < narg;  i++)
		check_type_character(&vs_base[i]);
	for (i = 1;  i < narg;  i++)
		for (j = 0;  j < i;  j++)
			if (char_equal(vs_base[j], vs_base[i]))
				@(return Cnil)
	@(return Ct)
@)


static int
char_compare(x, y)
object x, y;
{
	int i, j;

	i = char_code(x);
	j = char_code(y);
	if (isLower(i))
		i -= 'a' - 'A';
	if (isLower(j))
		j -= 'a' - 'A';
	if (i < j)
		return(-1);
	else if (i == j)
		return(0);
	else
		return(1);
}

static void
Lchar_compare(s, t)
int s, t;
{
	int narg, i;

	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0; i < narg; i++)
		check_type_character(&vs_base[i]);
	for (i = 1; i < narg; i++)
		if (s*char_compare(vs_base[i], vs_base[i-1]) < t) {
			vs_top = vs_base+1;
			vs_base[0] = Cnil;
			return;
		}
	vs_top = vs_base+1;
	vs_base[0] = Ct;
}

LFD(Lchar_lessp)()        { Lchar_compare( 1, 1); }
LFD(Lchar_greaterp)()     { Lchar_compare(-1, 1); }
LFD(Lchar_not_greaterp)() { Lchar_compare( 1, 0); }
LFD(Lchar_not_lessp)()    { Lchar_compare(-1, 0); }


object
coerce_to_character(x)
object x;
{
BEGIN:
	switch (type_of(x)) {
	case t_fixnum:
		if (0 <= fix(x) && fix(x) < CHCODELIM)
			return(code_char(fix(x)));
		break;

	case t_character:
		return(x);

	case t_symbol:
	  x=coerce_to_string(x);
	case t_simple_string:
	case t_string:
	  if (VLEN(x) == 1)
	    return(code_char(x->ust.ust_self[0]));
		break;
	default:
		break;
	}
	vs_push(x);
	x = wrong_type_argument(sLcharacter, x);
	vs_popp;
	goto BEGIN;
}

@(defun character (x)
@
	@(return `coerce_to_character(x)`)
@)

@(defun char_code (c)
@
	check_type_character(&c);
	@(return `make_fixnum(char_code(c))`)
@)

@(defun code_char (c &o (b `make_fixnum(0)`) (f `make_fixnum(0)`))
	object x;
@
	check_type_non_negative_integer(&c);
        b=make_fixnum(0);/*FIXME*/
	check_type_non_negative_integer(&b);
        f=make_fixnum(0);/*FIXME*/
	check_type_non_negative_integer(&f);
	if (type_of(c) == t_bignum)
		@(return Cnil)
	if (type_of(b) == t_bignum)
		@(return Cnil)
	if (type_of(f) == t_bignum)
		@(return Cnil)
	if (fix(c)>=CHCODELIM || fix(b)>=CHBITSLIM || fix(f)>=CHFONTLIM)
		@(return Cnil)
	if (fix(b) == 0 && fix(f) == 0)
		@(return `code_char(fix(c))`)
	x = alloc_object(t_character);
	char_code(x) = fix(c);
	char_bits(x) = fix(b);
	char_font(x) = fix(f);
	@(return x)
@)

@(defun char_upcase (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return c)
	if (isLower(char_code(c)))
		@(return `code_char(char_code(c) - ('a' - 'A'))`)
	else
		@(return c)
@)

@(defun char_downcase (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	if (isUpper(char_code(c)))
		@(return `code_char(char_code(c) + ('a' - 'A'))`)
	else
		@(return c)
@)

int
digit_weight(w, r)
int w, r;
{
	if (r < 2 || r > 36 || w < 0 || w >= r)
		return(-1);
	if (w < 10)
		return(w + '0');
	else
		return(w - 10 + 'A');
}

@(defun digit_char (w &optional (r `make_fixnum(10)`))
	object x,f;
	int dw;
@
	check_type_non_negative_integer(&w);
	check_type_non_negative_integer(&r);
        f=make_fixnum(0);/*FIXME*/
	check_type_non_negative_integer(&f);
	if (type_of(w) == t_bignum)
		@(return Cnil)
	if (type_of(r) == t_bignum)
		@(return Cnil)
	if (type_of(f) == t_bignum)
		@(return Cnil)
	dw = digit_weight(fix(w), fix(r));
	if (dw < 0)
		@(return Cnil)
	if (fix(f) >= CHFONTLIM)
		@(return Cnil)
	if (fix(f) == 0)
		@(return `code_char(dw)`)
	x = alloc_object(t_character);
	char_code(x) = dw;
	char_bits(x) = 0;
	char_font(x) = fix(f);
	@(return x)
@)

@(defun char_int (c)
	int i;
@
	check_type_character(&c);
	i = (char_font(c)*CHBITSLIM + char_bits(c))*CHCODELIM
	  + char_code(c);
	@(return `make_fixnum(i)`)
@)

@(defun char_name (c)
@
	check_type_character(&c);
	if (char_bits(c) != 0 || char_font(c) != 0)
		@(return Cnil)
	switch (char_code(c)) {
	case '\r':
		@(return STreturn)

	case ' ':
		@(return STspace)

	case '\177':
		@(return STrubout)
	
	case '\f':
		@(return STpage)

	case '\t':
		@(return STtab)

	case '\b':
		@(return STbackspace)

	case '\n':
		@(return STnewline)
	}
        if (char_code(c)<' ' || char_code(c) >='\177') {
          object x=make_simple_string(" ");
          x->st.st_self[0]=char_code(c);
          @(return x)
        }
	@(return Cnil)
@)

@(defun name_char (s)
@
	s = coerce_to_string(s);
	if (string_equal(s, STreturn))
		@(return `code_char('\r')`)
	if (string_equal(s, STspace))
		@(return `code_char(' ')`)
	if (string_equal(s, STrubout))
		@(return `code_char('\177')`)
	if (string_equal(s, STpage))
		@(return `code_char('\f')`)
	if (string_equal(s, STtab))
		@(return `code_char('\t')`)
	if (string_equal(s, STbackspace))
		@(return `code_char('\b')`)
	if (string_equal(s, STlinefeed) || string_equal(s, STnewline))
		@(return `code_char('\n')`)
        if (VLEN(s)==1) @(return `code_char(s->st.st_self[0])`)
        if (VLEN(s)==2 && s->st.st_self[0]=='^') {
	  int ch=s->st.st_self[1]-'A'+1;
	  @(return `code_char(ch)`)
        }
        if (VLEN(s)==3 && s->st.st_self[0]=='^' && s->st.st_self[1]=='\\' && s->st.st_self[2]=='\\') {
	  int ch=s->st.st_self[1]-'A'+1;
	  @(return `code_char(ch)`)
        }
        if (VLEN(s)==4 && s->st.st_self[0]=='\\') {
	  int ch=(s->st.st_self[1]-'0')*8*8+(s->st.st_self[2]-'0')*8+(s->st.st_self[3]-'0');
	  @(return `code_char(ch)`)
        }
	@(return Cnil)
@)

void
gcl_init_character()
{
	int i;

	for (i = 0;  i < CHCODELIM;  i++) {
	  object x=(object)(character_table+i),y=(object)(character_name_table+i);
	  set_type_of(x,t_character);
	  x->ch.ch_code = i;
	  x->ch.tt=((' ' <= i && i < '\177') || i == '\n');
	  x->ch.ch_font = 0;
	  x->ch.ch_bits = 0;
	  x->ch.ch_name=y;
	  set_type_of(y,t_simple_string);
	  y->sst.sst_hasfillp = FALSE;
	  y->sst.sst_adjustable = FALSE;
	  set_array_elttype(y,aet_ch);
	  y->sst.sst_rank = 1;
	  y->sst.sst_dim = 1;
	  y->sst.sst_self = (void *)&x->ch.ch_code;

	}

 	make_constant("CHAR-CODE-LIMIT", make_fixnum(CHCODELIM));
 	make_si_constant("CHAR-FONT-LIMIT", make_fixnum(CHFONTLIM));
 	make_si_constant("CHAR-BITS-LIMIT", make_fixnum(CHBITSLIM));

	STreturn = make_simple_string("Return");
	enter_mark_origin(&STreturn);
	STspace = make_simple_string("Space");
	enter_mark_origin(&STspace);
	STrubout = make_simple_string("Rubout");
	enter_mark_origin(&STrubout);
	STpage = make_simple_string("Page");
	enter_mark_origin(&STpage);
	STtab = make_simple_string("Tab");
	enter_mark_origin(&STtab);
	STbackspace = make_simple_string("Backspace");
	enter_mark_origin(&STbackspace);
	STlinefeed = make_simple_string("Linefeed");
	enter_mark_origin(&STlinefeed);

	STnewline = make_simple_string("Newline");
	enter_mark_origin(&STnewline);

	make_si_constant("CHAR-CONTROL-BIT", make_fixnum(0));
	make_si_constant("CHAR-META-BIT", make_fixnum(0));
	make_si_constant("CHAR-SUPER-BIT", make_fixnum(0));
	make_si_constant("CHAR-HYPER-BIT", make_fixnum(0));

}

@(defun make_char (c &o (b `make_fixnum(0)`) (f `make_fixnum(0)`))
	object x;
	int code;
@
	check_type_character(&c);
	code = char_code(c);
	check_type_non_negative_integer(&b);
	check_type_non_negative_integer(&f);
	if (type_of(b) == t_bignum)
		@(return Cnil)
	if (type_of(f) == t_bignum)
		@(return Cnil)
	if (fix(b)>=CHBITSLIM || fix(f)>=CHFONTLIM)
		@(return Cnil)
	if (fix(b) == 0 && fix(f) == 0)
		@(return `code_char(code)`)
	x = alloc_object(t_character);
	char_code(x) = code;
	char_bits(x) = fix(b);
	char_font(x) = fix(f);
	@(return x)
@)

@(defun char_bits (c)
@
	check_type_character(&c);
	@(return `small_fixnum(char_bits(c))`)
@)

@(defun char_font (c)
@
	check_type_character(&c);
	@(return `small_fixnum(char_font(c))`)
@)

@(defun char_bit (c n)
@
	check_type_character(&c);
	FEerror("Cannot get char-bit of ~S.", 1, c);
@)

@(defun set_char_bit (c n v)
@
	check_type_character(&c);
	FEerror("Cannot set char-bit of ~S.", 1, c);
@)

@(defun string_char_p (c)
@
	check_type_character(&c);
	if (char_font(c) != 0 || char_bits(c) != 0)
		@(return Cnil)
	@(return Ct)
@)

@(defun int_char (x)
	int i, c, b, f;
@
	check_type_non_negative_integer(&x);
	if (type_of(x) == t_bignum)
		@(return Cnil)
	i = fix(x);
	c = i % CHCODELIM;
	i /= CHCODELIM;
	b = i % CHBITSLIM;
	i /= CHBITSLIM;
	f = i % CHFONTLIM;
	i /= CHFONTLIM;
	if (i > 0)
		@(return Cnil)
	if (b == 0 && f == 0)
		@(return `code_char(c)`)
	x = alloc_object(t_character);
	char_code(x) = c;
	char_bits(x) = b;
	char_font(x) = f;
	@(return x)
@)

void
gcl_init_character_function()
{
	make_function("STANDARD-CHAR-P", Lstandard_char_p);
	make_function("GRAPHIC-CHAR-P", Lgraphic_char_p);
	make_function("ALPHA-CHAR-P", Lalpha_char_p);
	make_function("UPPER-CASE-P", Lupper_case_p);
	make_function("LOWER-CASE-P", Llower_case_p);
	make_function("BOTH-CASE-P", Lboth_case_p);
	make_function("DIGIT-CHAR-P", Ldigit_char_p);
	make_function("ALPHANUMERICP", Lalphanumericp);
	make_function("CHAR=", Lchar_eq);
	make_function("CHAR/=", Lchar_neq);
	make_function("CHAR<", Lchar_l);
	make_function("CHAR>", Lchar_g);
	make_function("CHAR<=", Lchar_le);
	make_function("CHAR>=", Lchar_ge);
	make_function("CHAR-EQUAL", Lchar_equal);
	make_function("CHAR-NOT-EQUAL", Lchar_not_equal);
	make_function("CHAR-LESSP", Lchar_lessp);
	make_function("CHAR-GREATERP", Lchar_greaterp);
	make_function("CHAR-NOT-GREATERP", Lchar_not_greaterp);
	make_function("CHAR-NOT-LESSP", Lchar_not_lessp);
	make_function("CHARACTER", Lcharacter);
	make_function("CHAR-CODE", Lchar_code);
	make_function("CODE-CHAR", Lcode_char);
	make_function("CHAR-UPCASE", Lchar_upcase);
	make_function("CHAR-DOWNCASE", Lchar_downcase);
	make_function("DIGIT-CHAR", Ldigit_char);
	make_function("CHAR-INT", Lchar_int);
	make_function("CHAR-NAME", Lchar_name);
	make_function("NAME-CHAR", Lname_char);
	make_si_function("INT-CHAR", Lint_char);
	make_si_function("MAKE-CHAR", Lmake_char);
	make_si_function("CHAR-BITS", Lchar_bits);
	make_si_function("CHAR-FONT", Lchar_font);
	make_si_function("CHAR-BIT", Lchar_bit);
	make_si_function("SET-CHAR-BIT", Lset_char_bit);
	make_si_function("STRING-CHAR-P", Lstring_char_p);
}
