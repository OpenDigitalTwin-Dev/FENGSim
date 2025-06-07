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
	format.c
*/

#include "include.h"
#include "num_include.h"

static int
fmt_thousand(int,int,bool,bool,int);

static void
fmt_exponent1(int);

static void
fmt_write_numeral(int,int);

static void
fmt_write_ordinal(int,int);

static int
fmt_nonillion(int,int,bool,bool,int);

static void
fmt_roman(int,int,int,int,int);

static void
fmt_integer(object,bool,bool,fixnum,fixnum,fixnum,fixnum,fixnum);

static void
fmt_semicolon(bool,bool);

static void
fmt_up_and_out(bool,bool);

static void
fmt_justification(volatile bool,bool);

static void
fmt_iteration(bool,bool);

static void
fmt_function(bool,bool);

static void
fmt_conditional(bool,bool);

static void
fmt_case(bool,bool);

static void
fmt_indirection(bool,bool);

static void
fmt_asterisk(bool,bool);

static void
fmt_tabulate(bool,bool);

static void
fmt_newline(bool,bool);

static void
fmt_ppnewline(bool,bool);

static void
fmt_ppindent(bool,bool);

static void
fmt_tilde(bool,bool);

static void
fmt_bar(bool,bool);

static void
fmt_ampersand(bool,bool);

static void
fmt_percent(bool,bool);

static void
fmt_dollars_float(bool,bool);

static void
fmt_general_float(bool,bool);

static void
fmt_exponential_float(bool,bool);

static void
fmt_fix_float(bool,bool);

static void
fmt_character(bool,bool);

static void
fmt_proc_character(object,bool,bool);

static void
fmt_plural(bool,bool);

static void
fmt_radix(bool,bool);

static void
fmt_hexadecimal(bool,bool);

static void
fmt_octal(bool,bool);

static void
fmt_binary(bool,bool);

static void
fmt_error(char *);

static void
fmt_ascii(bool, bool);

static void
fmt_S_expression(bool, bool);

static void
fmt_write(bool, bool);

static void
fmt_decimal(bool, bool);


object sSAindent_formatted_outputA;

#define	ctl_string	(fmt_string->st.st_self + ctl_origin)

#define	fmt_old		VOL object old_fmt_stream; \
			VOL int old_ctl_origin; \
			VOL int old_ctl_index; \
			VOL int old_ctl_end; \
			object * VOL old_fmt_base; \
			VOL int old_fmt_index; \
			VOL int old_fmt_end; \
			VOL object old_fmt_iteration_list; \
			jmp_bufp   VOL old_fmt_jmp_bufp; \
			VOL int old_fmt_indents; \
			VOL object old_fmt_string ; \
			VOL object(*old_fmt_advance)(void) ;	\
			VOL void (*old_fmt_lt)(volatile bool,bool) ;	\
                        VOL format_parameter *old_fmt_paramp
#define	fmt_save	old_fmt_stream = fmt_stream; \
			old_ctl_origin = ctl_origin; \
			old_ctl_index = ctl_index; \
			old_ctl_end = ctl_end; \
			old_fmt_base = fmt_base; \
			old_fmt_index = fmt_index; \
			old_fmt_end = fmt_end; \
			old_fmt_iteration_list = fmt_iteration_list; \
			old_fmt_jmp_bufp = fmt_jmp_bufp; \
			old_fmt_indents = fmt_indents; \
			old_fmt_string = fmt_string ; \
			old_fmt_advance=fmt_advance ;	\
			old_fmt_lt=fmt_lt ;	\
                        old_fmt_paramp = fmt_paramp
#define	fmt_restore	fmt_stream = old_fmt_stream; \
			ctl_origin = old_ctl_origin; \
			ctl_index = old_ctl_index; \
			ctl_end = old_ctl_end; \
			fmt_base = old_fmt_base; \
			fmt_index = old_fmt_index; \
			fmt_iteration_list = old_fmt_iteration_list; \
			fmt_end = old_fmt_end; \
			fmt_jmp_bufp = old_fmt_jmp_bufp; \
			fmt_indents = old_fmt_indents; \
			fmt_string = old_fmt_string ; \
			fmt_advance=old_fmt_advance ;	\
			fmt_lt=old_fmt_lt ;	\
                        fmt_paramp = old_fmt_paramp 

#define	fmt_old1	VOL object old_fmt_stream; \
			VOL int old_ctl_origin; \
			VOL int old_ctl_index; \
			VOL int old_ctl_end; \
			jmp_bufp   VOL old_fmt_jmp_bufp; \
			VOL int old_fmt_indents; \
			VOL object old_fmt_string ; \
                        VOL format_parameter *old_fmt_paramp
#define	fmt_save1       old_fmt_stream = fmt_stream; \
			old_ctl_origin = ctl_origin; \
			old_ctl_index = ctl_index; \
			old_ctl_end = ctl_end; \
			old_fmt_jmp_bufp = fmt_jmp_bufp; \
			old_fmt_indents = fmt_indents; \
			old_fmt_string = fmt_string ; \
                        old_fmt_paramp = fmt_paramp
#define	fmt_restore1	fmt_stream = old_fmt_stream; \
			ctl_origin = old_ctl_origin; \
			ctl_index = old_ctl_index; \
			ctl_end = old_ctl_end; \
			fmt_jmp_bufp = old_fmt_jmp_bufp; \
			fmt_indents = old_fmt_indents; \
			fmt_string = old_fmt_string ; \
                        fmt_paramp = old_fmt_paramp 

#define MAX_MINCOL 1024
#define BOUND_MINCOL(a_) ({fixnum _t=a_; _t=_t<0 ? 0 : _t;if (_t>MAX_MINCOL) _t=MAX_MINCOL;_t;})

typedef struct {
	  fixnum fmt_param_type;
	  fixnum fmt_param_value;
	  object fmt_param_object;
	} format_parameter;

format_parameter fmt_param[100];
VOL format_parameter *fmt_paramp;
#define FMT_PARAM (fmt_paramp)

#undef writec_stream
#define writec_stream(a,b) writec_pstream(a,b)
#undef writestr_stream
#define writestr_stream(a,b) writestr_pstream(a,b)

#ifndef WRITEC_NEWLINE
#define  WRITEC_NEWLINE(strm) (writec_stream('\n',strm))
#endif

object fmt_temporary_stream;
object fmt_temporary_string;

int fmt_nparam;
enum fmt_types {
  fmt_null,
  fmt_int,
  fmt_char};

char *fmt_big_numeral[] = {
	"thousand",
	"million",
	"billion",
	"trillion",
	"quadrillion",
	"quintillion",
	"sextillion",
	"septillion",
	"octillion"
};

char *fmt_numeral[] = {
	"zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine",
	"ten", "eleven", "twelve", "thirteen", "fourteen",
	"fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
	"zero", "ten", "twenty", "thirty", "forty",
	"fifty", "sixty", "seventy", "eighty", "ninety"
};

char *fmt_ordinal[] = {
	"zeroth", "first", "second", "third", "fourth",
	"fifth", "sixth", "seventh", "eighth", "ninth",
	"tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
	"fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
	"zeroth", "tenth", "twentieth", "thirtieth", "fortieth",
	"fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"
};


fixnum fmt_spare_spaces;
fixnum fmt_line_length;


static int
fmt_tempstr(int s)
{
	return(fmt_temporary_string->st.st_self[s]);
}

static int
ctl_advance(void)
{
	if (ctl_index >= ctl_end)
		fmt_error("unexpected end of control string");
	return(ctl_string[ctl_index++]);
}

static object
fmt_advance_base(void)
{
	if (fmt_index >= fmt_end)
		fmt_error("arguments exhausted");
	return(fmt_base[fmt_index++]);
}

static object
fmt_advance_pprint_pop(void) {

  object x;

  if (ifuncall4(sSpprint_quit,fmt_base[0],fmt_base[1],fmt_stream,fmt_base[2])!=Cnil)
    longjmp(*fmt_jmp_bufp, 1);/*FIXME :*/
  fmt_base[2]=number_plus(fmt_base[2],make_fixnum(1));
  x=fmt_base[0]->c.c_car;
  fmt_base[0]=fmt_base[0]->c.c_cdr;
  return x;
}

static object (*fmt_advance)(void)=fmt_advance_base;
static void (*fmt_lt)(volatile bool,bool)=fmt_justification;

static int
rd_ex_ch(int f,int *s) {
  char *p1[]={"Return","Space","Rubout","Page","Tab","Backspace","Linefeed","Newline",0},**p,*ch;
  int c1[]={'\r',' ','\177','\f','\t','\b','\n','\n',0},*c,i;
  for (p=p1,c=c1,i=ctl_index;*p && *c;p++,c++) {
    if (f==(*p)[0] && *s==(*p)[1]) {
      for (ch=*p+2,ctl_index=i;*ch && *ch==ctl_advance();ch++);
      if (!*ch) {
	*s=ctl_advance();
	return *c;
      }
    }
  }
  ctl_index=i;
  return f;
}
  

static void
format(object fmt_stream0, int ctl_origin0, int ctl_end0)
{
	int c, n;
	fixnum i,j,sn;
	bool colon, atsign;
	object x;
	fmt_paramp = fmt_param;

	/* could eliminate the no interrupt if made the
	   temporary stream on the stack... */
       {BEGIN_NO_INTERRUPT;
	fmt_stream = fmt_stream0;
	ctl_origin = ctl_origin0;
	ctl_index = 0;
	ctl_end = ctl_end0;

LOOP:
	if (ctl_index >= ctl_end)
	  { END_NO_INTERRUPT;
		return;}
	if ((c = ctl_advance()) != '~') {
		writec_stream(c, fmt_stream);
		goto LOOP;
	}
	n = 0;
	for (;;) {
		switch (c = ctl_advance()) {
		case ',':
			fmt_param[n].fmt_param_type = fmt_null;
			break;

		case '0':  case '1':  case '2':  case '3':  case '4':
		case '5':  case '6':  case '7':  case '8':  case '9':
   		        sn=1;
		DIGIT:
			i = 0;
			do {
				j = i*10 + (c - '0');
				i=j>=i ? j : MOST_POSITIVE_FIX;
				c = ctl_advance();
			} while (isDigit(c));
			fmt_param[n].fmt_param_type = fmt_int;
			fmt_param[n].fmt_param_value = sn*i;
			fmt_param[n].fmt_param_object=make_fixnum(fmt_param[n].fmt_param_value);
			break;

		case '+':
		case '-':
   		        sn=c=='+' ? 1 : -1;
			c = ctl_advance();
			if (!isDigit(c))
				fmt_error("digit expected");
			goto DIGIT;

		case '\'':
			fmt_param[n].fmt_param_type = fmt_char;
			fmt_param[n].fmt_param_value = ctl_advance();
			c = ctl_advance();
			if (c != ',')
			  fmt_param[n].fmt_param_value=rd_ex_ch(fmt_param[n].fmt_param_value,&c);
			fmt_param[n].fmt_param_object = code_char(fmt_param[n].fmt_param_value);
			break;

		case 'v':  case 'V':
			x = fmt_advance();
			if (type_of(x) == t_fixnum) {
				fmt_param[n].fmt_param_type = fmt_int;
				fmt_param[n].fmt_param_value = fix(x);
				/* if (fmt_param[n].fmt_param_value==MOST_NEGATIVE_FIX) */
				fmt_param[n].fmt_param_object=x;
			} else if (type_of(x) == t_character) {
				fmt_param[n].fmt_param_type = fmt_char;
				fmt_param[n].fmt_param_value = x->ch.ch_code;
				fmt_param[n].fmt_param_object=x;
			} else if (type_of(x) == t_bignum) {
				fmt_param[n].fmt_param_type = fmt_int;
				fmt_param[n].fmt_param_value = MOST_NEGATIVE_FIX;
				fmt_param[n].fmt_param_object = x;
                        } else if (x == Cnil) {
                                 fmt_param[n].fmt_param_type = fmt_null;				
			} else
				fmt_error("illegal V parameter");
			c = ctl_advance();
			break;

		case '#':
			fmt_param[n].fmt_param_type = fmt_int;
			fmt_param[n].fmt_param_value = fmt_end - fmt_index;
			fmt_param[n].fmt_param_object=make_fixnum(fmt_param[n].fmt_param_value);
			c = ctl_advance();
			break;

		default:
/*			if (n > 0)
				fmt_error("illegal ,");
			else
*/
            /* allow (FORMAT NIL "~5,,X" 10) ; ie ,just before directive */ 

				goto DIRECTIVE;
		}
		n++;
		if (c != ',')
			break;
	}

DIRECTIVE:
	colon = atsign = FALSE;
	if (c == ':') {
		colon = TRUE;
		c = ctl_advance();
	}
	if (c == '@') {
		atsign = TRUE;
		c = ctl_advance();
		if (!colon)
		  if (c == ':') {
		    colon = TRUE;
		    c = ctl_advance();
		  }
	}
	fmt_nparam = n;
	switch (c) {
	case 'a':  case 'A':
		fmt_ascii(colon, atsign);
		break;

	case 's':  case 'S':
		fmt_S_expression(colon, atsign);
		break;

	case 'w':  case 'W':
		fmt_write(colon, atsign);
		break;

	case 'd':  case 'D':
		fmt_decimal(colon, atsign);
		break;

	case 'b':  case 'B':
		fmt_binary(colon, atsign);
		break;

	case 'o':  case 'O':
		fmt_octal(colon, atsign);
		break;

	case 'x':  case 'X':
		fmt_hexadecimal(colon, atsign);
		break;

	case 'r':  case 'R':
		fmt_radix(colon, atsign);
		break;

	case 'p':  case 'P':
		fmt_plural(colon, atsign);
		break;

	case 'c':  case 'C':
		fmt_character(colon, atsign);
		break;

	case 'f':  case 'F':
		fmt_fix_float(colon, atsign);
		break;

	case 'e':  case 'E':
		fmt_exponential_float(colon, atsign);
		break;

	case 'g':  case 'G':
		fmt_general_float(colon, atsign);
		break;

	case '$':
		fmt_dollars_float(colon, atsign);
		break;

	case '%':
		fmt_percent(colon, atsign);
		break;

	case '&':
		fmt_ampersand(colon, atsign);
		break;

	case '|':
		fmt_bar(colon, atsign);
		break;

	case '~':
		fmt_tilde(colon, atsign);
		break;

	case '_':
		fmt_ppnewline(colon, atsign);
		break;

	case 'I':
	case 'i':
		fmt_ppindent(colon, atsign);
		break;

	case '\n':
	case '\r':	
		fmt_newline(colon, atsign);
		break;

	case 't':  case 'T':
		fmt_tabulate(colon, atsign);
		break;

	case '*':
		fmt_asterisk(colon, atsign);
		break;

	case '?':
		fmt_indirection(colon, atsign);
		break;

	case '(':
		fmt_case(colon, atsign);
		break;

	case '[':
		fmt_conditional(colon, atsign);
		break;

	case '{':
		fmt_iteration(colon, atsign);
		break;

	case '/':
		fmt_function(colon, atsign);
		break;

	case '<':
		fmt_lt(colon, atsign);
		break;

	case '^':
		fmt_up_and_out(colon, atsign);
		break;

	case ';':
		fmt_semicolon(colon, atsign);
		break;

	default:
   {object user_fmt=getf(sSAindent_formatted_outputA->s.s_plist,make_fixnum(c),Cnil);
    
    if (user_fmt!=Cnil)
     {object *oldbase=vs_base;
      object *oldtop=vs_top;
      vs_base=vs_top;
      vs_push(fmt_advance());
      vs_push(fmt_stream);
      vs_push(make_fixnum(colon));
      vs_push(make_fixnum(atsign));
      if (type_of(user_fmt)==t_symbol) user_fmt=symbol_function(user_fmt);
      funcall(user_fmt);
      vs_base=oldbase; vs_top=oldtop; break;}}
		fmt_error("illegal directive");
	}
	goto LOOP;
}}



static int
fmt_skip(void)
{
	int c, level = 0;
	
LOOP:
	if (ctl_advance() != '~')
		goto LOOP;
	for (;;)
		switch (c = ctl_advance()) {
		case '\'':
			ctl_advance();

		case ',':
		case '0':  case '1':  case '2':  case '3':  case '4':
		case '5':  case '6':  case '7':  case '8':  case '9':
		case '+':
		case '-':
		case 'v':  case 'V':
		case '#':
		case ':':  case '@':
			continue;

		default:
			goto DIRECTIVE;
		}

DIRECTIVE:
	switch (c) {
	case '(':  case '[':  case '<':  case '{':
		level++;
		break;

	case ')':  case ']':  case '>':  case '}':
		if (level == 0)
			return(ctl_index);
		else
			--level;
		break;

	case ';':
		if (level == 0)
			return(ctl_index);
		break;
	}
	goto LOOP;
}


static void
fmt_max_param(int n)
{
	if (fmt_nparam > n)
		fmt_error("too many parameters");
}

static void
fmt_not_colon(bool colon)
{
	if (colon)
		fmt_error("illegal :");
}

static void
fmt_not_atsign(bool atsign)
{
	if (atsign)
		fmt_error("illegal @");
}

static void
fmt_not_colon_atsign(bool colon, bool atsign)
{
	if (colon && atsign)
		fmt_error("illegal :@");
}

static void
fmt_set_param(fixnum i, fixnum *p, fixnum t, fixnum v)
{
	if (i >= fmt_nparam || FMT_PARAM[i].fmt_param_type == fmt_null)
		*p = v;
	else if (FMT_PARAM[i].fmt_param_type != t)
		fmt_error("illegal parameter type");
	else
		*p = FMT_PARAM[i].fmt_param_value;
}	


static void
fmt_ascii(bool colon, bool atsign)
{
	fixnum mincol=0, colinc=0, minpad=0, padchar=0;
	object x;
	int l, i;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	mincol=BOUND_MINCOL(mincol);
	fmt_set_param(1, &colinc, fmt_int, 1);
	fmt_set_param(2, &minpad, fmt_int, 0);
	fmt_set_param(3, &padchar, fmt_char, ' ');

	fmt_temporary_string->st.st_fillp = 0;
	/* fmt_temporary_stream->sm.sm_int0 = file_column(fmt_stream); */
	STREAM_FILE_COLUMN(fmt_temporary_stream) = file_column(fmt_stream);
	x = fmt_advance();
	if (colon && x == Cnil)
		writestr_stream("()", fmt_temporary_stream);
	else if (mincol == 0 && minpad == 0) {
		princ(x, fmt_stream);
		return;
	} else
		princ(x, fmt_temporary_stream);
	l = fmt_temporary_string->st.st_fillp;
	for (i = minpad;  l + i < mincol;  i += colinc)
		;
	if (!atsign) {
		write_string(fmt_temporary_string, fmt_stream);
		while (i-- > 0)
			writec_stream(padchar, fmt_stream);
	} else {
		while (i-- > 0)
			writec_stream(padchar, fmt_stream);
		write_string(fmt_temporary_string, fmt_stream);
	}
}

static void
fmt_write(bool colon, bool atsign) {

  object x;
  bds_ptr old_bds_top=bds_top;

  fmt_max_param(0);

  x = fmt_advance();

  if (colon)
    bds_bind(sLAprint_prettyA,Ct);
  if (atsign) {
    bds_bind(sLAprint_levelA,Cnil);
    bds_bind(sLAprint_lengthA,Cnil);
  }
  fSwrite_int(x,fmt_stream);

  bds_unwind(old_bds_top);

}

static void
fmt_S_expression(bool colon, bool atsign)
{
	fixnum mincol=0, colinc=0, minpad=0, padchar=0;
	object x;
	int l, i;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	mincol=BOUND_MINCOL(mincol);
	fmt_set_param(1, &colinc, fmt_int, 1);
	fmt_set_param(2, &minpad, fmt_int, 0);
	fmt_set_param(3, &padchar, fmt_char, ' ');

	fmt_temporary_string->st.st_fillp = 0;
	/* fmt_temporary_stream->sm.sm_int0 = file_column(fmt_stream); */
	STREAM_FILE_COLUMN(fmt_temporary_stream) = file_column(fmt_stream);
	x = fmt_advance();
	if (colon && x == Cnil)
		writestr_stream("()", fmt_temporary_stream);
	else if (type_of(x)==t_character)/*FIXME*/
	  return fmt_proc_character(x,0,1);
	else if (mincol == 0 && minpad == 0) {
		prin1(x, fmt_stream);
		return;
	} else
		prin1(x, fmt_temporary_stream);
	l = fmt_temporary_string->st.st_fillp;
	for (i = minpad;  l + i < mincol;  i += colinc)
		;
	if (!atsign) {
		write_string(fmt_temporary_string, fmt_stream);
		while (i-- > 0)
			writec_stream(padchar, fmt_stream);
	} else {
		while (i-- > 0)
			writec_stream(padchar, fmt_stream);
		write_string(fmt_temporary_string, fmt_stream);
	}
}

static void
fmt_decimal(bool colon, bool atsign)
{
	fixnum mincol=0, padchar=0, commachar=0, commainterval=0;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	fmt_set_param(1, &padchar, fmt_char, ' ');
	fmt_set_param(2, &commachar, fmt_char, ',');
	fmt_set_param(3, &commainterval, fmt_int, 3);
	fmt_integer(fmt_advance(), colon, atsign,
		    10, mincol, padchar, commachar, commainterval);
}

static void
fmt_binary(bool colon, bool atsign)
{
	fixnum mincol=0, padchar=0, commachar=0, commainterval=0;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	fmt_set_param(1, &padchar, fmt_char, ' ');
	fmt_set_param(2, &commachar, fmt_char, ',');
	fmt_set_param(3, &commainterval, fmt_int, 3);
	fmt_integer(fmt_advance(), colon, atsign,
		    2, mincol, padchar, commachar, commainterval);
}

static void
fmt_octal(bool colon, bool atsign)
{
	fixnum mincol=0, padchar=0, commachar=0, commainterval=0;;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	fmt_set_param(1, &padchar, fmt_char, ' ');
	fmt_set_param(2, &commachar, fmt_char, ',');
	fmt_set_param(3, &commainterval, fmt_int, 3);
	fmt_integer(fmt_advance(), colon, atsign,
		    8, mincol, padchar, commachar, commainterval);
}

static void
fmt_hexadecimal(bool colon, bool atsign)
{
	fixnum mincol=0, padchar=0, commachar=0, commainterval=0;;

	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	fmt_set_param(1, &padchar, fmt_char, ' ');
	fmt_set_param(2, &commachar, fmt_char, ',');
	fmt_set_param(3, &commainterval, fmt_int, 3);
	fmt_integer(fmt_advance(), colon, atsign,
		    16, mincol, padchar, commachar, commainterval);
}

static void
fmt_radix(bool colon, bool atsign)
{
	fixnum radix=0, mincol=0, padchar=0, commachar=0, commainterval=0;
	object x;
	int i, j, k;
	int s, t;
	bool b;

	fmt_max_param(5);
	fmt_set_param(0, &radix, fmt_int, -1);
	if (radix==-1) {
		x = fmt_advance();
		check_type_integer(&x);
		if (atsign) {
			if (type_of(x) == t_fixnum)
				i = fix(x);
			else
				i = -1;
			if ((!colon && (i <= 0 || i >= 4000)) ||
			    (colon && (i <= 0 || i >= 5000))) {
				fmt_integer(x, FALSE, FALSE, 10, 0, ' ', ',', 3);
				return;
			}
			fmt_roman(i/1000, 'M', '*', '*', colon);
			fmt_roman(i%1000/100, 'C', 'D', 'M', colon);
			fmt_roman(i%100/10, 'X', 'L', 'C', colon);
			fmt_roman(i%10, 'I', 'V', 'X', colon);
			return;
		}
		fmt_temporary_string->st.st_fillp = 0;
		STREAM_FILE_COLUMN(fmt_temporary_stream) = file_column(fmt_stream);
		bds_bind(sLAprint_radixA,Cnil);
		bds_bind(sLAprint_baseA,make_fixnum(10));
		princ(x,fmt_temporary_stream);
		bds_unwind1;
		bds_unwind1;
		s = 0;
		i = fmt_temporary_string->st.st_fillp;
		if (i == 1 && fmt_tempstr(s) == '0') {
			writestr_stream("zero", fmt_stream);
			if (colon)
				writestr_stream("th", fmt_stream);
			return;
		} else if (fmt_tempstr(s) == '-') {
			writestr_stream("minus ", fmt_stream);
			--i;
			s++;
		}
		t = fmt_temporary_string->st.st_fillp;
		for (;;)
			if (fmt_tempstr(--t) != '0')
				break;
		for (b = FALSE;  i > 0;  i -= j) {
			b = fmt_nonillion(s, j = (i+29)%30+1, b,
					  i<=30&&colon, t);
			s += j;
			if (b && i > 30) {
				for (k = (i - 1)/30;  k > 0;  --k)
					writestr_stream(" nonillion",
							fmt_stream);
				if (colon && s > t)
					writestr_stream("th", fmt_stream);
			}
		}
		return;
	}
	fmt_set_param(0, &radix, fmt_int, -1);
	fmt_set_param(1, &mincol, fmt_int, 0);
	fmt_set_param(2, &padchar, fmt_char, ' ');
	fmt_set_param(3, &commachar, fmt_char, ',');
	fmt_set_param(4, &commainterval, fmt_int, 3);
	x = fmt_advance();
	check_type_integer(&x);
	if (radix < 0 || radix > 36) {
		vs_push(make_fixnum(radix));
		FEerror("~D is illegal as a radix.", 1, vs_head);
	}
	fmt_integer(x, colon, atsign, radix, mincol, padchar, commachar, commainterval);
}	

static void
fmt_integer(object x, bool colon, bool atsign, fixnum radix, fixnum mincol, fixnum padchar, fixnum commachar, fixnum commainterval)
{
	int l, l1;
	int s;

	mincol=BOUND_MINCOL(mincol);
	if (type_of(x) != t_fixnum && type_of(x) != t_bignum) {
	        object fts,ftm;/*FIXME more comprehensive solution
				 here, but this avoids some recursive
				 use of the temporaries*/
	        ftm=make_string_output_stream(64);
	        fts=ftm->sm.sm_object0;
	        fts->st.st_fillp = 0;
		/* ftm->sm.sm_int0 = file_column(fmt_stream); */
		STREAM_FILE_COLUMN(ftm) = file_column(fmt_stream);
		bds_bind(sLAprint_baseA,make_fixnum(radix));
		princ(x,ftm);
		bds_unwind1;
		l = fts->st.st_fillp;
		mincol -= l;
		while (mincol-- > 0)
			writec_stream(padchar, fmt_stream);
		for (s = 0;  l > 0;  --l, s++)
			writec_stream(fts->st.st_self[s], fmt_stream);
		return;
	}
	fmt_temporary_string->st.st_fillp = 0;
	STREAM_FILE_COLUMN(fmt_temporary_stream) = file_column(fmt_stream);
	bds_bind(sLAprint_baseA,make_fixnum(radix));
	princ(x,fmt_temporary_stream);
	bds_unwind1;
	if (fmt_temporary_string->st.st_fillp>0&& fmt_temporary_string->st.st_self[fmt_temporary_string->st.st_fillp-1]=='.')/*FIXME*/
	  fmt_temporary_string->st.st_fillp--;
	l = l1 = fmt_temporary_string->st.st_fillp;
	s = 0;
	if (fmt_tempstr(s) == '-')
		--l1;
	mincol -= l;
	if (colon)
		mincol -= (l1 - 1)/3;
	if (atsign && fmt_tempstr(s) != '-')
		--mincol;
	while (mincol-- > 0)
		writec_stream(padchar, fmt_stream);
	if (fmt_tempstr(s) == '-') {
		s++;
		writec_stream('-', fmt_stream);
	} else if (atsign)
		writec_stream('+', fmt_stream);
	while (l1-- > 0) {
		writec_stream(fmt_tempstr(s++), fmt_stream);
		if (colon && l1 > 0 && l1%(commainterval) == 0)
			writec_stream(commachar, fmt_stream);
	}
}

static int
fmt_nonillion(int s, int i, bool b, bool o, int t)
{
	int j;

	for (;  i > 3;  i -= j) {
		b = fmt_thousand(s, j = (i+2)%3+1, b, FALSE, t);
		if (j != 3 || fmt_tempstr(s) != '0' ||
		    fmt_tempstr(s+1) != '0' || fmt_tempstr(s+2) != '0') {
			writec_stream(' ', fmt_stream);
			writestr_stream(fmt_big_numeral[(i - 1)/3 - 1],
					fmt_stream);
			s += j;
			if (o && s > t)
				writestr_stream("th", fmt_stream);
		} else
			s += j;
	}
	return(fmt_thousand(s, i, b, o, t));
}		

static int
fmt_thousand(int s, int i, bool b, bool o, int t)
{
	if (i == 3 && fmt_tempstr(s) > '0') {
		if (b)
			writec_stream(' ', fmt_stream);
		fmt_write_numeral(s, 0);
		writestr_stream(" hundred", fmt_stream);
		--i;
		s++;
		b = TRUE;
		if (o && s > t)
			writestr_stream("th", fmt_stream);
	}
	if (i == 3) {
		--i;
		s++;
	}
	if (i == 2 && fmt_tempstr(s) > '0') {
		if (b)
			writec_stream(' ', fmt_stream);
		if (fmt_tempstr(s) == '1') {
			if (o && s + 2 > t)
				fmt_write_ordinal(++s, 10);
			else
				fmt_write_numeral(++s, 10);
			return(TRUE);
		} else {
			if (o && s + 1 > t)
				fmt_write_ordinal(s, 20);
			else
				fmt_write_numeral(s, 20);
			s++;
			if (fmt_tempstr(s) > '0') {
				writec_stream('-', fmt_stream);
				if (o && s + 1 > t)
					fmt_write_ordinal(s, 0);
				else
					fmt_write_numeral(s, 0);
			}
			return(TRUE);
		}
	}
	if (i == 2)
		s++;
	if (fmt_tempstr(s) > '0') {
		if (b)
			writec_stream(' ', fmt_stream);
		if (o && s + 1 > t)
			fmt_write_ordinal(s, 0);
		else
			fmt_write_numeral(s, 0);
		return(TRUE);
	}
	return(b);
}
	
static void
fmt_write_numeral(int s, int i)
{
	writestr_stream(fmt_numeral[fmt_tempstr(s) - '0' + i], fmt_stream);
}

static void
fmt_write_ordinal(int s, int i)
{
	writestr_stream(fmt_ordinal[fmt_tempstr(s) - '0' + i], fmt_stream);
}

static void
fmt_roman(int i, int one, int five, int ten, int colon)
{
	int j;

	if (i == 0)
		return;
	if ((!colon && i < 4) || (colon && i < 5))
		for (j = 0;  j < i;  j++)
			writec_stream(one, fmt_stream);
	else if (!colon && i == 4) {
		writec_stream(one, fmt_stream);
		writec_stream(five, fmt_stream);
	} else if ((!colon && i < 9) || colon) {
		writec_stream(five, fmt_stream);
		for (j = 5;  j < i;  j++)
			writec_stream(one, fmt_stream);
	} else if (!colon && i == 9) {
		writec_stream(one, fmt_stream);
		writec_stream(ten, fmt_stream);
	}
}

static void
fmt_plural(bool colon, bool atsign)
{
	fmt_max_param(0);
	if (colon) {
		if (fmt_index == 0)
			fmt_error("can't back up");
		--fmt_index;
	}
	if (eql(fmt_advance(), make_fixnum(1)))
		if (atsign)
			writec_stream('y', fmt_stream);
		else
			;
	else
		if (atsign)
			writestr_stream("ies", fmt_stream);
		else
			writec_stream('s', fmt_stream);
}

static void
fmt_proc_character(object x,bool colon,bool atsign) {

  if (colon || atsign) {

    int i=colon ? 2 : 0;

    fmt_temporary_string->st.st_fillp = 0;
    STREAM_FILE_COLUMN(fmt_temporary_stream) = 0;
    if (x->ch.ch_code==' ')
      writestr_stream("#\\Space",fmt_temporary_stream);
    else
      prin1(x, fmt_temporary_stream);

    for (;  i < fmt_temporary_string->st.st_fillp;  i++)
      writec_stream(fmt_tempstr(i), fmt_stream);



  } else
    writec_stream(x->ch.ch_code, fmt_stream);

}

static void
fmt_character(bool colon, bool atsign) {

  object x;

  fmt_max_param(0);
  x = fmt_advance();
  check_type_character(&x);

  fmt_proc_character(x,colon,atsign);

}

static void
fmt_fix_float(bool colon, bool atsign)
{
        fixnum w=0, d=0, k=0, overflowchar=0, padchar=0,dp;
	double f;
	int sign;
	char *buff, *b, *buff1;
	int exp;
	int i, j;
	object x;
	int n, m;
	vs_mark;

	massert(buff=alloca(256)); /*from automatic array -- work around for persistent gcc alpha bug*/
	massert(buff1=alloca(256));

	b = buff1 + 1;

	fmt_not_colon(colon);
	fmt_max_param(5);
	fmt_set_param(0, &w, fmt_int, 0);
	if (w < 0)
		fmt_error("illegal width");
	fmt_set_param(0, &w, fmt_int, -1);
	fmt_set_param(1, &d, fmt_int, 0);
	if (d < 0)
		fmt_error("illegal number of digits");
	fmt_set_param(1, &d, fmt_int, -1);
	fmt_set_param(2, &k, fmt_int, 0);
	fmt_set_param(3, &overflowchar, fmt_char, -1);
	fmt_set_param(4, &padchar, fmt_char, ' ');

	x = fmt_advance();
	if (type_of(x) == t_fixnum ||
	    type_of(x) == t_bignum ||
	    type_of(x) == t_ratio) {
		x = make_shortfloat((shortfloat)number_to_double(x));
		vs_push(x);
	}
	if (type_of(x) == t_complex) {
		if (w < 0)
			prin1(x, fmt_stream);
		else {
			fmt_nparam = 1;
			--fmt_index;
			fmt_decimal(colon, atsign);
		}
		vs_reset;
		return;
	}
	if (type_of(x) == t_longfloat) {
	  n = 17;
	  dp=1;
	} else {
	  n = 10;/*FIXME*/
	  dp=0;
	}
	f = number_to_double(x);
	edit_double(n, f, &sign, buff, &exp, dp);
	if (sign==2) {
		prin1(x, fmt_stream);
		vs_reset;
		return;
	}
	if (d >= 0)
		m = d + exp + k + 1;
	else if (w >= 0) {
		if (exp + k >= 0)
			m = w - 1;
		else
			m = w + exp + k - 2;
		if (sign < 0 || atsign)
			--m;
		if (m == 0)
			m = 1;
	} else
		m = n;
	if (m <= 0) {
		if (m == 0 && buff[0] >= '5') {
			exp++;
			n = m = 1;
			buff[0] = '1';
		} else
			n = m = 0;
	} else if (m < n) {
		n = m;
		edit_double(n, f, &sign, buff, &exp, dp);
	}
	while (n >= 0)
		if (buff[n - 1] == '0')
			--n;
		else
			break;
	exp += k;
	j = 0;
	if (exp >= 0) {
		for (i = 0;  i <= exp;  i++)
			b[j++] = i < n ? buff[i] : '0';
		b[j++] = '.';
		if (d >= 0)
			for (m = i + d;  i < m;  i++)
				b[j++] = i < n ? buff[i] : '0';
		else
			for (;  i < n;  i++)
				b[j++] = buff[i];
	} else {
		b[j++] = '.';
		if (d >= 0) {
			for (i = 0;  i < (-exp) - 1 && i < d;  i++)
				b[j++] = '0';
			for (m = d - i, i = 0;  i < m;  i++)
				b[j++] = i < n ? buff[i] : '0';
		} else if (n > 0) {
			for (i = 0;  i < (-exp) - 1;  i++)
				b[j++] = '0';
			for (i = 0;  i < n;  i++)
				b[j++] = buff[i];
		}
	}
	b[j] = '\0';
	if (w >= 0) {
		if (sign < 0 || atsign)
			--w;
		if (j > w && overflowchar >= 0)
			goto OVER;
		if (j < w && b[j-1] == '.' && d) {
			b[j++] = '0';
			b[j] = '\0';
		}
		if (j < w && b[0] == '.') {
			*--b = '0';
			j++;
		}
		for (i = j;  i < w;  i++)
			writec_stream(padchar, fmt_stream);
	} else {
		if (b[0] == '.') {
			*--b = '0';
			j++;
		}
		if (d < 0 && b[j-1] == '.') {
			b[j++] = '0';
			b[j] = '\0';
		}
	}
	if (sign < 0)
		writec_stream('-', fmt_stream);
	else if (atsign)
		writec_stream('+', fmt_stream);
	writestr_stream(b, fmt_stream);
	vs_reset;
	return;

OVER:
	fmt_set_param(0, &w, fmt_int, 0);
	for (i = 0;  i < w;  i++)
		writec_stream(overflowchar, fmt_stream);
	vs_reset;
	return;
}

static int
fmt_exponent_length(int e)
{
	int i;

	if (e == 0)
		return(1);
	if (e < 0)
		e = -e;
	for (i = 0;  e > 0;  i++, e /= 10)
		;
	return(i);
}

static void
fmt_exponent(int e)
{
	if (e == 0) {
		writec_stream('0', fmt_stream);
		return;
	}
	if (e < 0)
		e = -e;
	fmt_exponent1(e);
}
	
static void
fmt_exponent1(int e)
{
	if (e == 0)
		return;
	fmt_exponent1(e/10);
	writec_stream('0' + e%10, fmt_stream);
}

static void
fmt_exponential_float(bool colon, bool atsign)
{
        fixnum w=0, d=0, e=0, k=0, overflowchar=0, padchar=0, exponentchar=0,dp;
	double f;
	int sign;
	char buff[256], *b, buff1[256];
	int exp;
	int i, j;
	object x, y;
	int n, m;
	enum type t;
	vs_mark;

	b = buff1 + 1;

	fmt_not_colon(colon);
	fmt_max_param(7);
	fmt_set_param(0, &w, fmt_int, 0);
	if (w < 0)
		fmt_error("illegal width");
	fmt_set_param(0, &w, fmt_int, -1);
	fmt_set_param(1, &d, fmt_int, 0);
	if (d < 0)
		fmt_error("illegal number of digits");
	fmt_set_param(1, &d, fmt_int, -1);
	fmt_set_param(2, &e, fmt_int, 0);
	if (e < 0)
		fmt_error("illegal number of digits in exponent");
	fmt_set_param(2, &e, fmt_int, -1);
	fmt_set_param(3, &k, fmt_int, 1);
	fmt_set_param(4, &overflowchar, fmt_char, -1);
	fmt_set_param(5, &padchar, fmt_char, ' ');
	fmt_set_param(6, &exponentchar, fmt_char, -1);

	x = fmt_advance();
	if (type_of(x) == t_fixnum ||
	    type_of(x) == t_bignum ||
	    type_of(x) == t_ratio) {
		x = make_shortfloat((shortfloat)number_to_double(x));
		vs_push(x);
	}
	if (type_of(x) == t_complex) {
		if (w < 0)
			prin1(x, fmt_stream);
		else {
			fmt_nparam = 1;
			--fmt_index;
			fmt_decimal(colon, atsign);
		}
		vs_reset;
		return;
	}
	if (type_of(x) == t_longfloat) {
	  n = 17;
	  dp=1;
	} else {
	  n = 9;
	  dp=0;
	}
	f = number_to_double(x);
	edit_double(n, f, &sign, buff, &exp, dp);
	if (sign==2) {
		prin1(x, fmt_stream);
		vs_reset;
		return;
	}
	if (d >= 0) {
		if (k > 0) {
			if (!(k < d + 2))
				fmt_error("illegal scale factor");
			m = d + 1;
		} else {
			if (!(k > -d))
				fmt_error("illegal scale factor");
			m = d + k;
		}
	} else if (w >= 0) {
		if (k > 0)
			m = w - 1;
		else
			m = w + k - 1;
		if (sign < 0 || atsign)
			--m;
		if (e >= 0)
			m -= e + 2;
		else
			m -= fmt_exponent_length(e - k + 1) + 2;
	} else
		m = n;
	if (m <= 0) {
		if (m == 0 && buff[0] >= '5') {
			exp++;
			n = m = 1;
			buff[0] = '1';
		} else
			n = m = 0;
	} else if (m < n) {
		n = m;
		edit_double(n, f, &sign, buff, &exp, dp);
	}
	while (n >= 0)
		if (buff[n - 1] == '0')
			--n;
		else
			break;
	exp = exp - k + 1;
	j = 0;
	if (k > 0) {
		for (i = 0;  i < k;  i++)
			b[j++] = i < n ? buff[i] : '0';
		b[j++] = '.';
		if (d >= 0)
			for (m = i + (d - k + 1);  i < m;  i++)
				b[j++] = i < n ? buff[i] : '0';
		else
			for (;  i < n;  i++)
				b[j++] = buff[i];
	} else {
		b[j++] = '.';
		if (d >= 0) {
			for (i = 0;  i < -k && i < d;  i++)
				b[j++] = '0';
			for (m = d - i, i = 0;  i < m;  i++)
				b[j++] = i < n ? buff[i] : '0';
		} else if (n > 0) {
			for (i = 0;  i < -k;  i++)
				b[j++] = '0';
			for (i = 0;  i < n;  i++)
				b[j++] = buff[i];
		}
	}
	b[j] = '\0';
	if (w >= 0) {
		if (sign < 0 || atsign)
			--w;
		i = fmt_exponent_length(exp);
		if (e >= 0) {
			if (i > e) {
				if (overflowchar >= 0)
					goto OVER;
				else
					e = i;
			}
			w -= e + 2;
		} else
			w -= i + 2;
		if (j > w && overflowchar >= 0)
			goto OVER;
		if (j < w && b[j-1] == '.') {
			b[j++] = '0';
			b[j] = '\0';
		}
		if (j < w && b[0] == '.') {
			*--b = '0';
			j++;
		}
		for (i = j;  i < w;  i++)
			writec_stream(padchar, fmt_stream);
	} else {
		if (b[j-1] == '.') {
			b[j++] = '0';
			b[j] = '\0';
		}
		if (d < 0 && b[0] == '.') {
			*--b = '0';
			j++;
		}
	}
	if (sign < 0)
		writec_stream('-', fmt_stream);
	else if (atsign)
		writec_stream('+', fmt_stream);
	writestr_stream(b, fmt_stream);
	y = symbol_value(sLAread_default_float_formatA);
	if (exponentchar < 0) {
		if (y == sLlong_float || y == sLdouble_float
		    || y == sLsingle_float 

		    )
			t = t_longfloat;
		else
			t = t_shortfloat;
		if (type_of(x) == t)
			exponentchar = 'E';
		else if (type_of(x) == t_shortfloat)
			exponentchar = 'S';
		else
			exponentchar = 'L';
	}
	writec_stream(exponentchar, fmt_stream);
	if (exp < 0)
		writec_stream('-', fmt_stream);
	else
		writec_stream('+', fmt_stream);
	if (e >= 0)
		for (i = e - fmt_exponent_length(exp);  i > 0;  --i)
			writec_stream('0', fmt_stream);
	fmt_exponent(exp);
	vs_reset;
	return;

OVER:
	fmt_set_param(0, &w, fmt_int, -1);
	for (i = 0;  i < w;  i++)
		writec_stream(overflowchar, fmt_stream);
	vs_reset;
	return;
}

static void
fmt_general_float(bool colon, bool atsign)
{
        fixnum w=0, d=0, e=0, k, overflowchar, padchar=0, exponentchar,dp;
	int sign, exp;
	char buff[256];
	object x;
	int n, ee, ww, q, dd;
	vs_mark;

	fmt_not_colon(colon);
	fmt_max_param(7);
	fmt_set_param(0, &w, fmt_int, 0);
	if (w < 0)
		fmt_error("illegal width");
	fmt_set_param(0, &w, fmt_int, -1);
	fmt_set_param(1, &d, fmt_int, 0);
	if (d < 0)
		fmt_error("illegal number of digits");
	fmt_set_param(1, &d, fmt_int, -1);
	fmt_set_param(2, &e, fmt_int, 0);
	if (e < 0)
		fmt_error("illegal number of digits in exponent");
	fmt_set_param(2, &e, fmt_int, -1);
	fmt_set_param(3, &k, fmt_int, 1);
	fmt_set_param(4, &overflowchar, fmt_char, -1);
	fmt_set_param(5, &padchar, fmt_char, ' ');
	fmt_set_param(6, &exponentchar, fmt_char, -1);

	x = fmt_advance();
	if (type_of(x) == t_complex) {
		if (w < 0)
			prin1(x, fmt_stream);
		else {
			fmt_nparam = 1;
			--fmt_index;
			fmt_decimal(colon, atsign);
		}
		vs_reset;
		return;
	}
	if (type_of(x) == t_longfloat) {
	  q = 17;
	  dp=1;
	} else {
	  q = 8;
	  dp=0;
	}
	edit_double(q, number_to_double(x), &sign, buff, &exp, dp);
	n = exp + 1;
	while (q > 0)
		if (buff[q - 1] == '0')
			--q;
		else
			break;
	if (e >= 0)
		ee = e + 2;
	else
		ee = 4;
	ww = w - ee;
	if (d < 0) {
		d = n < 7 ? n : 7;
		d = q > d ? q : d;
	}
	dd = d - n;
	if (0 <= dd && dd <= d) {
		FMT_PARAM[0].fmt_param_value = ww;
		if (w < 0) FMT_PARAM[0].fmt_param_type = fmt_null;
		FMT_PARAM[1].fmt_param_value = dd;
		FMT_PARAM[1].fmt_param_type = fmt_int;
		FMT_PARAM[2].fmt_param_type = fmt_null;
		if (fmt_nparam > 4)
		  {FMT_PARAM[3] =    FMT_PARAM[4]; }
		else FMT_PARAM[3].fmt_param_type = fmt_null;
		if (fmt_nparam > 5)
		  {FMT_PARAM[4] = FMT_PARAM[5];}
		else FMT_PARAM[4].fmt_param_type = fmt_null;
		fmt_nparam = 5;
		--fmt_index;
		fmt_fix_float(colon, atsign);
		if (w >= 0)
			while (ww++ < w)
				writec_stream(padchar, fmt_stream);
		vs_reset;
		return;
	}
	FMT_PARAM[1].fmt_param_value = d;
	FMT_PARAM[1].fmt_param_type = fmt_int;
	--fmt_index;
	fmt_exponential_float(colon, atsign);
	vs_reset;
}

static void
fmt_dollars_float(bool colon, bool atsign)
{
        fixnum d=0, n=0, w=0, padchar=0,dp;
	double f;
	int sign;
	char buff[256];
	int exp;
	int q, i;
	object x;
	vs_mark;

	fmt_max_param(4);
	fmt_set_param(0, &d, fmt_int, 2);
	if (d < 0)
		fmt_error("illegal number of digits");
	fmt_set_param(1, &n, fmt_int, 1);
	if (n < 0)
		fmt_error("illegal number of digits");
	fmt_set_param(2, &w, fmt_int, 0);
	if (w < 0)
		fmt_error("illegal width");
	fmt_set_param(3, &padchar, fmt_char, ' ');
	x = fmt_advance();
	if (type_of(x) == t_complex) {
		if (w < 0)
			prin1(x, fmt_stream);
		else {
			fmt_nparam = 1;
			FMT_PARAM[0] = FMT_PARAM[2];
			--fmt_index;
			fmt_decimal(colon, atsign);
		}
		vs_reset;
		return;
	}
	q = 8;
	dp=0;
	if (type_of(x) == t_longfloat) {
		q = 17;
		dp=1;
	}
	f = number_to_double(x);
	edit_double(q, f, &sign, buff, &exp, dp);
	if ((q = exp + d + 1) > 0)
	  edit_double(q, f, &sign, buff, &exp, dp);
	exp++;
	if (w > 100 || exp > 100 || exp < -100) {
		fmt_nparam = 6;
		FMT_PARAM[0] = FMT_PARAM[2];
		FMT_PARAM[1].fmt_param_value = d + n - 1;
		FMT_PARAM[1].fmt_param_type = fmt_int;
		FMT_PARAM[2].fmt_param_type =
		FMT_PARAM[3].fmt_param_type =
		FMT_PARAM[4].fmt_param_type = fmt_null;
		FMT_PARAM[5] = FMT_PARAM[3];
		--fmt_index;
		fmt_exponential_float(colon, atsign);
	}
	if (exp > n)
		n = exp;
	if (sign < 0 || atsign)
		--w;
	if (colon) {
		if (sign < 0)
			writec_stream('-', fmt_stream);
		else if (atsign)
			writec_stream('+', fmt_stream);
		while (--w > n + d)
			writec_stream(padchar, fmt_stream);
	} else {
		while (--w > n + d)
			writec_stream(padchar, fmt_stream);
		if (sign < 0)
			writec_stream('-', fmt_stream);
		else if (atsign)
			writec_stream('+', fmt_stream);
	}
	for (i = n - exp;  i > 0;  --i)
		writec_stream('0', fmt_stream);
	for (i = 0;  i < exp;  i++)
		writec_stream((i < q ? buff[i] : '0'), fmt_stream);
	writec_stream('.', fmt_stream);
	for (d += i;  i < d;  i++)
		writec_stream((i < q ? buff[i] : '0'), fmt_stream);
	vs_reset;
}

static void
fmt_percent(bool colon, bool atsign)
{
	fixnum n=0, i;

	fmt_max_param(1);
	fmt_set_param(0, &n, fmt_int, 1);
	fmt_not_colon(colon);
	fmt_not_atsign(atsign);
	while (n-- > 0) {
                WRITEC_NEWLINE(fmt_stream);
		if (n == 0)
			for (i = fmt_indents;  i > 0;  --i)
				writec_stream(' ', fmt_stream);
	}
}

static void
fmt_ampersand(bool colon, bool atsign)
{
	fixnum n=0;

	fmt_max_param(1);
	fmt_set_param(0, &n, fmt_int, 1);
	fmt_not_colon(colon);
	fmt_not_atsign(atsign);
	if (n == 0)
		return;
	if (file_column(fmt_stream) != 0)
	  WRITEC_NEWLINE(fmt_stream);
	while (--n > 0)
	  	  WRITEC_NEWLINE(fmt_stream);
	fmt_indents = 0;
}

static void
fmt_bar(bool colon, bool atsign)
{
	fixnum n=0;

	fmt_max_param(1);
	fmt_set_param(0, &n, fmt_int, 1);
	fmt_not_colon(colon);
	fmt_not_atsign(atsign);
	while (n-- > 0)
		writec_stream('\f', fmt_stream);
}

static void
fmt_tilde(bool colon, bool atsign)
{
	fixnum n=0;

	fmt_max_param(1);
	fmt_set_param(0, &n, fmt_int, 1);
	fmt_not_colon(colon);
	fmt_not_atsign(atsign);
	while (n-- > 0)
		writec_stream('~', fmt_stream);
}

static void
fmt_newline(bool colon, bool atsign)
{

	fmt_max_param(0);
	fmt_not_colon_atsign(colon, atsign);
	if (atsign)
	  WRITEC_NEWLINE(fmt_stream);
	while (ctl_index < ctl_end && isspace((int)ctl_string[ctl_index])) {
		if (colon)
			writec_stream(ctl_string[ctl_index], fmt_stream);
		ctl_index++;
	}
}

static void
fmt_ppnewline(bool colon, bool atsign) {

  object k=colon ? (atsign ? sKmandatory : sKfill) : (atsign ? sKmiser : sKlinear);
  object f=get(k,sLfixnum,Cnil);

  fmt_max_param(0);
  massert(type_of(f)==t_fixnum);
  ifuncall2(find_symbol(make_simple_string("PPRINT-NEWLINE"),lisp_package),k,fmt_stream);

}

static void
fmt_ppindent(bool colon, bool atsign) {

  object k=colon ? sKcurrent : sKblock;
  object f=get(k,sLfixnum,Cnil);
  fixnum n;

  fmt_max_param(1);
  fmt_set_param(0, &n, fmt_int, 0);
  massert(type_of(f)==t_fixnum);
  ifuncall3(find_symbol(make_simple_string("PPRINT-INDENT"),lisp_package),k,make_fixnum(n),fmt_stream);

}

static void
fmt_pptab(bool colon, bool atsign) {

  object k=colon ? (atsign ? sKsection_relative : sKsection) : (atsign ? sKline_relative : sKline);
  object f=get(k,sLfixnum,Cnil);
  fixnum colnum=0, colinc=0,n;

  fmt_max_param(2);
  fmt_set_param(0, &n, fmt_int, 0);
  fmt_set_param(0, &colnum, fmt_int, 1);
  fmt_set_param(1, &colinc, fmt_int, 1);
  massert(type_of(f)==t_fixnum);
  if (colon)
    ifuncall4(find_symbol(make_simple_string("PPRINT-TAB"),lisp_package),k,make_fixnum(colnum),make_fixnum(colinc),fmt_stream);
  else {
    bds_bind(sLAprint_prettyA,Ct);
    write_codes_pstream(fmt_stream,fix(f),2,colnum,colinc);
    bds_unwind1;
  }

}

static void
fmt_tabulate(bool colon, bool atsign)
{
	fixnum colnum=0, colinc=0;
	fixnum c, i;

	return fmt_pptab(colon,atsign);

	if (!atsign) {
		c = file_column(fmt_stream);
		if (c < 0) {
			writestr_stream("  ", fmt_stream);
			return;
		}
		if (c > colnum && colinc <= 0)
			return;
		while (c > colnum)
			colnum += colinc;
		for (i = colnum - c;  i > 0;  --i)
			writec_stream(' ', fmt_stream);
	} else {
		for (i = colnum;  i > 0;  --i)
			writec_stream(' ', fmt_stream);
		c = file_column(fmt_stream);
		if (c < 0 || colinc <= 0)
			return;
		colnum = 0;
		while (c > colnum)
			colnum += colinc;
		for (i = colnum - c;  i > 0;  --i)
			writec_stream(' ', fmt_stream);
	}
}

static void
fmt_asterisk(bool colon, bool atsign)
{
	fixnum n=0;

	fmt_max_param(1);
	fmt_not_colon_atsign(colon, atsign);
	if (atsign) {
		fmt_set_param(0, &n, fmt_int, 0);
		if (n < 0 || n >= fmt_end)
			fmt_error("can't goto");
		fmt_index = n;
	} else if (colon) {
		fmt_set_param(0, &n, fmt_int, 1);
		if (n > fmt_index)
			fmt_error("can't back up");
		fmt_index -= n;
	} else {
		fmt_set_param(0, &n, fmt_int, 1);
		while (n-- > 0)
			fmt_advance();
	}
}	

static void
fmt_indirection(bool colon, bool atsign) {
	object s, l;
	fmt_old;
	jmp_buf fmt_jmp_buf0;
	int up_colon;

	/* to prevent longjmp clobber */
	up_colon=(long)&old_fmt_paramp;
	fmt_max_param(0);
	fmt_not_colon(colon);
	s = fmt_advance();
	if (!stringp(s))
		fmt_error("control string expected");
	if (atsign) {
		fmt_save;
		fmt_jmp_bufp = &fmt_jmp_buf0;
		fmt_string = s;
		if ((up_colon = setjmp(*fmt_jmp_bufp))) {
			if (--up_colon)
				fmt_error("illegal ~:^");
		} else
		  format(fmt_stream, 0, VLEN(s));
		fmt_restore1;  /*FIXME restore?*/
	} else {
		l = fmt_advance();
		fmt_save;
		fmt_base = vs_top;
		fmt_index = 0;
		for (fmt_end = 0;  !endp(l);  fmt_end++, l = l->c.c_cdr)
			vs_check_push(l->c.c_car);
		fmt_jmp_bufp = &fmt_jmp_buf0;
		fmt_string = s;
		if ((up_colon = setjmp(*fmt_jmp_bufp))) {
			if (--up_colon)
				fmt_error("illegal ~:^");
		} else
		  format(fmt_stream, 0, VLEN(s));
		vs_top = fmt_base;
		fmt_restore;
	}
}

static void
fmt_case(bool colon, bool atsign)
{
	VOL object x;
	VOL int i, j;
	fmt_old1;
	jmp_buf fmt_jmp_buf0;
	int up_colon;
	bool b;

	x = make_string_output_stream(64);
	vs_push(x);
	i = ctl_index;
	j = fmt_skip();
	if (ctl_string[--j] != ')' || ctl_string[--j] != '~')
		fmt_error("~) expected");
	fmt_save1;
	fmt_jmp_bufp = &fmt_jmp_buf0;
	if ((up_colon = setjmp(*fmt_jmp_bufp)))
		;
	else
		format(x, ctl_origin + i, j - i);
	fmt_restore1;
	x = x->sm.sm_object0;
	if (!colon && !atsign)
		for (i = 0;  i < x->st.st_fillp;  i++) {
		  j = x->st.st_self[i];
		  if (isUpper(j))
		    j += 'a' - 'A';
		  writec_stream(j, fmt_stream);
		}
	else if (colon && !atsign)
		for (b = TRUE, i = 0;  i < x->st.st_fillp;  i++) {
		  j = x->st.st_self[i];
		  if (isLower(j)) {
		    if (b)
		      j -= 'a' - 'A';
		    b = FALSE;
		  } else if (isUpper(j)) {
		    if (!b)
		      j += 'a' - 'A';
		    b = FALSE;
		  } else if (!isDigit(j))
		    b = TRUE;
		  writec_stream(j, fmt_stream);
		}
	else if (!colon && atsign)
		for (b = TRUE, i = 0;  i < x->st.st_fillp;  i++) {
		  j = x->st.st_self[i];
		  if (isLower(j)) {
		    if (b)
		      j -= 'a' - 'A';
		    b = FALSE;
		  } else if (isUpper(j)) {
		    if (!b)
		      j += 'a' - 'A';
		    b = FALSE;
		  }
		  writec_stream(j, fmt_stream);
		}
	else
		for (i = 0;  i < x->st.st_fillp;  i++) {
		  j = x->st.st_self[i];
		  if (isLower(j))
		    j -= 'a' - 'A';
		  writec_stream(j, fmt_stream);
		}
	vs_popp;
	if (up_colon)
		longjmp(*fmt_jmp_bufp, up_colon);
}

static void
fmt_conditional(bool colon, bool atsign)
{
	int i, j, k;
	object x;
	fixnum n=0;
	bool done;
	fmt_old1;

	fmt_not_colon_atsign(colon, atsign);
	if (colon) {
		fmt_max_param(0);
		i = ctl_index;
		j = fmt_skip();
		if (ctl_string[--j] != ';' || ctl_string[--j] != '~')
			fmt_error("~; expected");
		k = fmt_skip();
		if (ctl_string[--k] != ']' || ctl_string[--k] != '~')
			fmt_error("~] expected");
		if (fmt_advance() == Cnil) {
			fmt_save1;
			format(fmt_stream, ctl_origin + i, j - i);
			fmt_restore1;
		} else {
			fmt_save1;
			format(fmt_stream, ctl_origin + j + 2, k - (j + 2));
			fmt_restore1;
		}
	} else if (atsign) {
		i = ctl_index;
		j = fmt_skip();
		if (ctl_string[--j] != ']' || ctl_string[--j] != '~')
			fmt_error("~] expected");
		if (fmt_advance() == Cnil)
			;
		else {
			--fmt_index;
			fmt_save1;
			format(fmt_stream, ctl_origin + i, j - i);
			fmt_restore1;
		}
	} else {
		fmt_max_param(1);
		if (fmt_nparam == 0 || FMT_PARAM[0].fmt_param_type==fmt_null) {
			x = fmt_advance();
			switch(type_of(x)) {
			case t_fixnum:
			  n=fix(x);break;
			case t_bignum:
			  n=MOST_NEGATIVE_FIX;break;/*FIXME*/
			default:
			  fmt_error("illegal argument for conditional");
			}
		} else
			fmt_set_param(0, &n, fmt_int, 0);
		i = ctl_index;
		for (done = FALSE;;  --n) {
			j = fmt_skip();
			for (k = j;  ctl_string[--k] != '~';)
				;
			if (n == 0) {
				fmt_save1;
				format(fmt_stream, ctl_origin + i, k - i);
				fmt_restore1;
				done = TRUE;
			}
			i = j;
			if (ctl_string[--j] == ']') {
				if (ctl_string[--j] != '~')
					fmt_error("~] expected");
				return;
			}
			if (ctl_string[j] == ';') {
				if (ctl_string[--j] == '~')
					continue;
				if (ctl_string[j] == ':')
					goto ELSE;
			}
			fmt_error("~; or ~] expected");
		}
	ELSE:
		if (ctl_string[--j] != '~')
			fmt_error("~:; expected");
		j = fmt_skip();
		if (ctl_string[--j] != ']' || ctl_string[--j] != '~')
			fmt_error("~] expected");
		if (!done) {
			fmt_save1;
			format(fmt_stream, ctl_origin + i, j - i);
			fmt_restore1;
		}
	}
}	

static object
fmt_copy_ctl_string(fixnum i,fixnum j) {

  object x=alloc_simple_string(j-i);
  x->sst.sst_self=alloc_relblock(j-i);
  memcpy(x->sst.sst_self,ctl_string+i,j-i);

  return x;

}

static void
fmt_function(bool colon, bool atsign) {

  fixnum i,j,c,n;
  object x,y,z,s,p=user_package;

  i=ctl_index;
  for (;(c=ctl_advance())!='/' && c!=':';);
  j=ctl_index;

  if (c==':') {
    if (ctl_string[ctl_index]==':')
      ctl_index++;
    s=fmt_copy_ctl_string(i,j-1);
    for (i=0;i<VLEN(s);i++)
      if (islower(s->sst.sst_self[i]))
	s->sst.sst_self[i]=toupper(s->sst.sst_self[i]);
    p=find_package(s);
    p=p==Cnil ? user_package : p;
    i=ctl_index;
    for (;(c=ctl_advance())!='/';);
    j=ctl_index;
  }

  s=fmt_copy_ctl_string(i,j-1);
  for (i=0;i<VLEN(s);i++)
    if (islower(s->sst.sst_self[i]))
      s->sst.sst_self[i]=toupper(s->sst.sst_self[i]);
  x=find_symbol(s,p);

  for (y=Cnil,n=fmt_nparam;n;)
    if (FMT_PARAM[--n].fmt_param_type!=fmt_null)
      y=MMcons(FMT_PARAM[n].fmt_param_object,y);

  z=fmt_advance();

  VFUN_NARGS=6;
  apply_format_function(x,fmt_stream,z,colon ? Ct : Cnil,atsign ? Ct : Cnil,y);

}

static void
fmt_proc_iteration(object control,fixnum o,fixnum i,fixnum j) {

  if (stringp(control))
    format(fmt_stream, o + i, j - i);
  else {
    object y,*p=ZALLOCA((1+(fmt_end-fmt_index))*sizeof(*p));
    *p=fmt_stream;
    memcpy(p+1,fmt_base+fmt_index,(fmt_end-fmt_index)*sizeof(*p));
    y=(fcall.valp=0,funcall_vec(coerce_funcall_object_to_function(control),1+fmt_end-fmt_index,p));
    fmt_index=fmt_end-length(y);
  }

}

static void
fmt_iteration(bool colon, bool atsign) {
	fixnum i,n=0;
	VOL int j;
	int o;
	bool colon_close = FALSE;
	object l;
	VOL object l0,control=fmt_string;
	fmt_old;
	jmp_buf fmt_jmp_buf0;
	int up_colon;

	/* to prevent longjmp clobber */
	up_colon=(long)&old_fmt_paramp;
	fmt_max_param(1);
	fmt_set_param(0, &n, fmt_int, 1000000);
	i = ctl_index;
	j = fmt_skip();
	if (ctl_string[--j] != '}')
		fmt_error("~} expected");
	if (ctl_string[--j] == ':') {
		colon_close = TRUE;
		--j;
	}
	if (ctl_string[j] != '~')
		fmt_error("syntax error");
	o = ctl_origin;
	if (i==j) {
	  switch (type_of(control=fmt_advance())) {
	  case t_string:
	  case t_simple_string:
	    fmt_string=control;
	    i=o=0;
	    j=VLEN(fmt_string);
	    break;
	  case t_symbol:
	  case t_function:
	    i=o=j=0;
	    break;
	  default:
	    fmt_error("control string expected");
	  }
	}
	if (!colon && !atsign) {
		l = fmt_advance();
		fmt_save;
		fmt_base = vs_top;
		fmt_index = 0;
		for (fmt_end = 0;  !endp(l);  fmt_end++, l = l->c.c_cdr)
			vs_check_push(l->c.c_car);
		fmt_jmp_bufp = &fmt_jmp_buf0;
		if (colon_close)
			goto L1;
		while (fmt_index < fmt_end) {
		L1:
			if (n-- <= 0)
				break;
			if ((up_colon = setjmp(*fmt_jmp_bufp))) {
				if (--up_colon)
					fmt_error("illegal ~:^");
				break;
			}
			fmt_proc_iteration(control,o,i,j);
		}
		vs_top = fmt_base;
		fmt_restore;
	} else if (colon && !atsign) {
		l0 = fmt_advance();
		fmt_save;
		fmt_iteration_list=l0;
		fmt_base = vs_top;
		fmt_jmp_bufp = &fmt_jmp_buf0;
		if (colon_close)
			goto L2;
		while (!endp(l0)) {
		L2:
			if (n-- <= 0)
				break;
			l = l0->c.c_car;
			fmt_iteration_list=l0 = l0->c.c_cdr;
			fmt_index = 0;
			for (fmt_end = 0; !endp(l); fmt_end++, l = l->c.c_cdr)
				vs_check_push(l->c.c_car);
			if ((up_colon = setjmp(*fmt_jmp_bufp))) {
				vs_top = fmt_base;
				if (--up_colon)
					break;
				else
					continue;
			}
			fmt_proc_iteration(control,o,i,j);
			vs_top = fmt_base;
		}
		fmt_restore;
	} else if (!colon && atsign) {
		fmt_save;
		fmt_jmp_bufp = &fmt_jmp_buf0;
		if (colon_close)
			goto L3;
		while (fmt_index < fmt_end) {
		L3:
			if (n-- <= 0)
				break;
			if ((up_colon = setjmp(*fmt_jmp_bufp))) {
				if (--up_colon)
					fmt_error("illegal ~:^");
				break;
			}
			fmt_proc_iteration(control,o,i,j);
		}
		fmt_restore1; /*FIXME restore?*/
	} else if (colon && atsign) {
		if (colon_close)
			goto L4;
		while (fmt_index < fmt_end) {
		L4:
		  fmt_iteration_list=fmt_index>=fmt_end-1 ? Cnil : Ct;
			if (n-- <= 0)
				break;
			l = fmt_advance();
			fmt_save;
			fmt_base = vs_top;
			fmt_index = 0;
			for (fmt_end = 0; !endp(l); fmt_end++, l = l->c.c_cdr)
				vs_check_push(l->c.c_car);
			fmt_jmp_bufp = &fmt_jmp_buf0;
			if ((up_colon = setjmp(*fmt_jmp_bufp))) {
				vs_top = fmt_base;
				fmt_restore;
				if (--up_colon)
					break;
				else
					continue;
			}
			fmt_proc_iteration(control,o,i,j);
			vs_top = fmt_base;
			fmt_restore;
		}
	}
}


DEFUN("FORMAT-LOGICAL-BLOCK-PREFIX",object,fSformat_logical_block_prefix,SI,2,2,NONE,OO,OO,OO,OO,
      (object x,object h),"") {

  if (ifuncall4(sSpprint_quit,x->c.c_cdr,h,fmt_stream,make_fixnum(-1))!=Cnil)
    RETURN1(Cnil);

  write_string(x->c.c_car->c.c_car,fmt_stream);

  RETURN1(Ct);

}

DEFUN("FORMAT-LOGICAL-BLOCK-BODY",object,fSformat_logical_block_body,SI,2,2,NONE,OO,OO,OO,OO,
      (object x,object h),"") {

  jmp_buf fmt_jmp_buf0;
  int up_colon;
  fmt_old;

  vs_mark;

  fmt_save;

  fmt_jmp_bufp = &fmt_jmp_buf0;
  if (!(up_colon = setjmp(*fmt_jmp_bufp))) {

    fmt_base=vs_top;
    fmt_index=0;
    vs_push(x->c.c_cdr);
    vs_push(h);
    vs_push(make_fixnum(0));
    fmt_end=3;
    fmt_string=x->c.c_car->c.c_cdr->c.c_car;
    fmt_advance=fmt_advance_pprint_pop;

    vs_push(fmt_stream);
    format(fmt_stream,0,VLEN(fmt_string));

  } else if (--up_colon)
    fmt_error("illegal ~:^");

  fmt_restore;

  vs_reset;

  RETURN1(Ct);

}

DEFUN("FORMAT-LOGICAL-BLOCK-SUFFIX",object,fSformat_logical_block_suffix,SI,2,2,NONE,OO,OO,OO,OO,
      (object x,object h),"") {

  write_string(x->c.c_car->c.c_cdr->c.c_cdr->c.c_car,fmt_stream);

  RETURN1(Ct);

}

static void
fmt_logical_block(volatile bool colon, bool atsign) {

  object per_line_prefix=Cnil,x,prefix,body,suffix;
  VOL int i,j,j0;
  int ax=0;
  VOL int special = 0;
  bds_ptr old_bds_top=bds_top;

  if (atsign) {
    object pp;
    for (x=pp=OBJNULL;fmt_index<fmt_end;) {
      object p=MMcons(fmt_advance(),Cnil);
      if (pp!=OBJNULL) pp->c.c_cdr=p; else x=p;
      pp=p;
    }
  } else
    x=fmt_advance();

  if (atom(x)) {
    x=list(1,x);
    ax=1;
  }

  i = ctl_index;
  j0 = j = fmt_skip()-1;
  while (ctl_string[--j] != '~');

  if (ctl_string[j0]==';') { /*prefix*/

    int k;

    for (k=i;k<j && ctl_string[k]!='~';k++);
    if (k<j)
      fmt_error("Directive found in prefix");

    prefix=fmt_copy_ctl_string(i,j);

    if (ctl_string[--j0]=='@') {
      per_line_prefix=prefix;
      prefix=make_simple_string("");
      special=1;
    } else if (ctl_string[j0]!='~')
      fmt_error("~ expected");

    i=ctl_index;
    j0=j=fmt_skip()-1;
    while (ctl_string[--j] != '~');

  } else if (colon && !ax)
    prefix=make_simple_string("(");
  else
    prefix=make_simple_string("");

  body=fmt_copy_ctl_string(i,j);

  if (ctl_string[j0]==';') { /*suffix*/

    int k;

    if (ctl_string[--j0]!='~')
      fmt_error("~ expected");

    i=ctl_index;
    j0=j=fmt_skip()-1;
    while (ctl_string[--j] != '~');

    for (k=i;k<j && ctl_string[k]!='~';k++);
    if (k<j)
      fmt_error("Directive found in suffix");

    suffix=fmt_copy_ctl_string(i,j);

  } else if (colon && !ax)
    suffix=make_simple_string(")");
  else
    suffix=make_simple_string("");

  if (ctl_string[j0]!='>')
    fmt_error("Terminating > expected");
  if (ctl_string[j0-1]=='@') {
    body=ifuncall1(sSpprint_insert_conditional_newlines,body);
    j0--;
  }
  if (ctl_string[--j0]!=':')
    fmt_error("Terminating :> expected");
  if (ctl_string[--j0]!='~')
    fmt_error("~ expected");

  if (per_line_prefix!=Cnil)
    bds_bind(sSAprint_line_prefixA,per_line_prefix);

  fSwrite_int1(MMcons(list(3,prefix,body,suffix),x),fmt_stream,sSformat_logical_block_body,
	       sSformat_logical_block_prefix,sSformat_logical_block_suffix);

  bds_unwind(old_bds_top);

}

#define FORMAT_DIRECTIVE_LIMIT 100

static void
fmt_justification(volatile bool colon, bool atsign)
{
	fixnum mincol=0, colinc=0, minpad=0, padchar=0;
	object fields[FORMAT_DIRECTIVE_LIMIT];
	fmt_old1;
	jmp_buf fmt_jmp_buf0;
	VOL int i,j,n,j0;
	int k,l,m,l0;
	int up_colon;
	VOL int special = 0;
	volatile int spare_spaces=0, line_length=0;
	vs_mark;

	/* to prevent longjmp clobber */
	up_colon=(long)&old_fmt_paramp;
	fmt_max_param(4);
	fmt_set_param(0, &mincol, fmt_int, 0);
	mincol=BOUND_MINCOL(mincol);
	fmt_set_param(1, &colinc, fmt_int, 1);
	fmt_set_param(2, &minpad, fmt_int, 0);
	fmt_set_param(3, &padchar, fmt_char, ' ');

	n = 0;
	for (;;) {
		if (n >= FORMAT_DIRECTIVE_LIMIT)
			fmt_error("too many fields");
		i = ctl_index;
		j0 = j = fmt_skip();
		while (ctl_string[--j] != '~')
			;
		fields[n] = make_string_output_stream(64);
		vs_push(fields[n]);
		fmt_save1;
		fmt_jmp_bufp = &fmt_jmp_buf0;
		if ((up_colon = setjmp(*fmt_jmp_bufp))) {
			--n;
			if (--up_colon)
				fmt_error("illegal ~:^");
			fmt_restore1;
			while (ctl_string[--j0] != '>')
				j0 = fmt_skip();
			if (ctl_string[j0-1] == '@') {
			    j0--;
			    if (ctl_string[j0-1] == ':') j0--;
			} else
			if (ctl_string[j0-1] == ':') {
			    j0--;
			    if (ctl_string[j0-1] == '@') j0--;
			}
			if (ctl_string[--j0] != '~')
				fmt_error("~> expected");
			break;
		}
		format(fields[n++], ctl_origin + i, j - i);
		fmt_restore1;
		if (ctl_string[--j0] == '>') {
			if (ctl_string[j0-1] == '@') {
			    j0--;
			    if (ctl_string[j0-1] == ':') j0--;
			} else
			if (ctl_string[j0-1] == ':') {
			    j0--;
			    if (ctl_string[j0-1] == '@') j0--;
			}
			if (ctl_string[--j0] != '~')
				fmt_error("~> expected");
			break;
		} else if (ctl_string[j0] != ';')
			fmt_error("~; expected");
		else {
		    if (ctl_string[j0] == '@')
			--j0;
		    if (ctl_string[--j0] == ':') {
			if (n != 1)
				fmt_error("illegal ~:;");
			special = 1;
			for (j = j0;  ctl_string[j] != '~';  --j)
				;
			fmt_save1;
			format(fmt_stream, ctl_origin + j, j0 - j + 2);
			fmt_restore1;
			spare_spaces = fmt_spare_spaces;
			line_length = fmt_line_length;
		    } else {
			if (ctl_string[j0] == '@')
			    --j0;
			if (ctl_string[j0] != '~')
			    fmt_error("~; expected");
		    }
		}
	}
	for (i = special, l = 0;  i < n;  i++)
		l += fields[i]->sm.sm_object0->st.st_fillp;
	m = n - 1 - special;
	l0 = l;
	l += minpad * m;
	if (m <= 0 && !colon && !atsign) {
		m = 0;
		colon = TRUE;
	}
	if (colon)
		m++;
	if (atsign)
		m++;
	for (k = 0;  mincol + k * colinc < l;  k++);
	l = mincol + k * colinc;
	if (special != 0 &&
	    file_column(fmt_stream) + l + spare_spaces >= line_length)
		princ(fields[0]->sm.sm_object0, fmt_stream);
	l -= l0;
	for (i = special;  i < n;  i++) {
		if (m > 0 && (i > special || colon))
			for (j = l / m, l -= j, --m;  j > 0;  --j)
				writec_stream(padchar, fmt_stream);
		princ(fields[i]->sm.sm_object0, fmt_stream);
	}
	if (atsign)
		for (j = l;  j > 0;  --j)
			writec_stream(padchar, fmt_stream);
	vs_reset;
}


static void
fmt_up_and_out(bool colon, bool atsign)
{
  fixnum j,n;
  object x[3];

  fmt_max_param(3);
  fmt_not_atsign(atsign);

  for (n=j=0;j<fmt_nparam;j++) {
    if (FMT_PARAM[j].fmt_param_type==fmt_null)
      continue;
    x[n++]=FMT_PARAM[j].fmt_param_value==MOST_NEGATIVE_FIX ? FMT_PARAM[j].fmt_param_object : make_fixnum(FMT_PARAM[j].fmt_param_value);
  }

  switch(n) {
  case 0:
    if (fmt_advance==fmt_advance_base) {
      if (colon ? fmt_iteration_list == Cnil : fmt_index >= fmt_end)
	longjmp(*fmt_jmp_bufp, ++colon);
    } else {
      if (fmt_base[0]==Cnil)
	longjmp(*fmt_jmp_bufp, ++colon);
    }
    break;
  case 1:
    if (!fix(x[0]))
      longjmp(*fmt_jmp_bufp, ++colon);
    break;
  case 2:
    if (number_compare(x[0],x[1])==0)
      longjmp(*fmt_jmp_bufp, ++colon);
    break;
  default:
    if (number_compare(x[0],x[1])<=0 && number_compare(x[1],x[2])<=0)
      longjmp(*fmt_jmp_bufp, ++colon);
    break;
  }

}


static void
fmt_semicolon(bool colon, bool atsign)
{
	fmt_not_atsign(atsign);
	if (!colon)
		fmt_error("~:; expected");
	fmt_max_param(2);
	fmt_set_param(0, &fmt_spare_spaces, fmt_int, 0);
	fmt_set_param(1, &fmt_line_length, fmt_int, 72);
}

DEFVAR("*FORMAT-UNUSED-ARGS*",sSAformat_unused_argsA,SI,OBJNULL,"");

static object justification_regexp=OBJNULL,logical_block_regexp=OBJNULL;

static int
fmt_pp_string(object control) {

  fixnum just,pp;
  if (justification_regexp==OBJNULL)
    justification_regexp=fScompile_regexp(make_simple_string("~>"));
  if (logical_block_regexp==OBJNULL)
    logical_block_regexp=fScompile_regexp(make_simple_string("~@?:@?>|~[:@]?[:@]?_|~[0-9]*:?[Ii]|~[0-9]+,[0-9]+[:@]?[:@]?[Tt]|~[:@]?[:@]?[Ww]"));
  VFUN_NARGS=2;
  just=(fixnum)fSstring_match2(justification_regexp,control);
  pp=(fixnum)fSstring_match2(logical_block_regexp,control);
  if (just>=0 && pp>=0)
    fmt_error("Mixed justification syntax");

  return pp>=0;

}


DEFUN("FORMAT",object,fLformat,LISP,2,F_ARG_LIMIT,NONE,OO,OO,OO,OO,(object strm, object control,...),"") {

  va_list ap; 
  VOL object x = OBJNULL;
  jmp_buf fmt_jmp_buf0;
  bool colon, e;
  VOL fixnum nargs=INIT_NARGS(2);
  
  fmt_old;
  
  if (strm == Cnil) {
    strm = make_string_output_stream(64);
    x = strm->sm.sm_object0;
  } else if (strm == Ct)
    strm = symbol_value(sLAstandard_outputA);
  else if (stringp(strm)) {
    x = strm;
    if (!x->st.st_hasfillp)
      FEerror("The string ~S doesn't have a fill-pointer.", 1, x);
    strm = make_string_output_stream(0);
    strm->sm.sm_object0 = x;
  } else
    check_type_stream(&strm);
  
  /* check_type_string(&control); */
  if (stringp(control)) {
    fmt_save;
    va_start(ap,control);
    frs_push(FRS_PROTECT, Cnil);
    if (nlj_active) {
      e = TRUE;
      goto L;
    }
    {
      object l[MAX_ARGS],ll=Cnil,f=OBJNULL;
      ufixnum i;
      for (i=0;(l[i]=NEXT_ARG(nargs,ap,ll,f,OBJNULL))!=OBJNULL;i++);
      fmt_base = l;
      fmt_index = 0;
      fmt_end = i;
      fmt_jmp_bufp = & fmt_jmp_buf0;
      if (symbol_value(sSAindent_formatted_outputA) != Cnil)
	fmt_indents = file_column(strm);
      else
	fmt_indents = 0;
      fmt_string = control;
      fmt_lt=fmt_pp_string(control) ? fmt_logical_block : fmt_justification;
      if ((colon = setjmp(*fmt_jmp_bufp))) {
	if (--colon)
	  fmt_error("illegal ~:^");
	vs_base = vs_top;
	if (x != OBJNULL)
	  vs_push(x);
	else
	  vs_push(Cnil);
	e = FALSE;
	goto L;
      }
      format(strm, 0, VLEN(control));
      if (sSAformat_unused_argsA->s.s_dbind!=OBJNULL) {
	int i;
	for (i=fmt_end-1;i>=fmt_index;i--)
	  sSAformat_unused_argsA->s.s_dbind=MMcons(fmt_base[i],sSAformat_unused_argsA->s.s_dbind);
      }
      
      flush_stream(strm);
    }
    e = FALSE;
  L:
    va_end(ap);
    frs_pop();
    fmt_restore;
    if (e) {
      nlj_active = FALSE;
      unwind(nlj_fr, nlj_tag);
    }
  } else
    switch (type_of(control)) {
    /* case t_cfun: */
    /* case t_ifun: */
    case t_function:
    case t_symbol:
    case t_cons:
      if (nargs >= 64) FEerror("Too many arguments",0);
      {	
	int i;
	object Xxvl[MAX_ARGS+1],ll=Cnil,f=OBJNULL;
	vs_mark;

	va_start(ap,control);
	Xxvl[0] = strm;
	for (i=1;(Xxvl[i]=NEXT_ARG(nargs,ap,ll,f,OBJNULL))!=OBJNULL;i++);
	va_end(ap);
	fcall.valp=0,funcall_vec(coerce_funcall_object_to_function(control),i,Xxvl);
	vs_reset;

      }
      break;
    default:
      FEwrong_type_argument(sLstring,control);
    }
  
  RETURN1 (x ==0 ? Cnil : x);  

}

object 
fLformat_1(object strm, object control,object x) {
  VFUN_NARGS=3;
  return FFN(fLformat)(strm,control,x);

}

/*  object c_apply_n(long int (*fn) (), int n, object *x); */

static void
fmt_error(char *s)
{
  fmt_advance=fmt_advance_base;/*FIXME*/
  vs_push(make_simple_string(s));
  vs_push(make_fixnum(&ctl_string[ctl_index] - fmt_string->st.st_self));
  FEerror("Format error: ~A.~%~V@TV~%\"~A\"~%",
	  3, vs_top[-2], vs_top[-1], fmt_string);
}

DEFVAR("*INDENT-FORMATTED-OUTPUT*",sSAindent_formatted_outputA,SI,Cnil,"");
void
gcl_init_format(void)
{
	fmt_temporary_stream = make_string_output_stream(64);
	enter_mark_origin(&fmt_temporary_stream);
	fmt_temporary_string = fmt_temporary_stream->sm.sm_object0;
	enter_mark_origin(&justification_regexp);
	enter_mark_origin(&logical_block_regexp);
}
