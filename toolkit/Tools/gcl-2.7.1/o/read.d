/*-*-C-*-*/
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
	read.d
*/

#define NEED_ISFINITE
#include "include.h"
#include "num_include.h"
#include "page.h"

static object
current_readtable(void);

DEFVAR("*SHARP-EQ-CONTEXT*",sSAsharp_eq_contextA,SI,sLnil,"");

DEFUN("TYPE_OF",object,fStype_of,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  return (object)(fixnum)type_of(x);
}
DEFUN("ALLOC-SPICE",object,fSalloc_spice,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  return alloc_object(t_spice);
}

DEFVAR("PATCH-SHARP",sSpatch_sharp,SI,sLnil,"");
DEFVAR("NEW-CONTEXT",sSnew_context,SI,sLnil,"");

static object
patch_sharp(object x) {return ifuncall1(sSpatch_sharp,x);}


#define digitp digitp1

static inline int
digitp(int i,int r) {
  
  if ( r<=10 || i<='9' )
    i-='0';
  else {
    i=tolower(i)-'a';
    i=i<0 ? i : i+10;
  }

  return i<r ? i : -1;

}

static inline object
parse_unsigned_integer_negate(char *s1,char **ep,int radix,int neg) {

  fixnum f,o,u,l=MOST_POSITIVE_FIX/radix-1;
  int r,d;
  char ch,*s;

  if (radix==10)
    for (o=u=1,f=0,s=s1;*s && *s<='9' && (d=*s-'0')>=0;u=0,o=o && f<l,f=f*10+d,s++);
  else
    for (o=u=1,f=0,s=s1;*s && (d=digitp(*s,radix))>=0;u=0,o=o && f<l,f=f*radix+d,s++);

  if (ep) *ep=s;

  if (u) return OBJNULL;
  if (o && !*s) return make_fixnum(neg ? -f : f);

  ch=*s;
  *s=0;
  r=mpz_set_str(MP(big_fixnum1),s1,radix);
  *s=ch;

  if (r) return OBJNULL;

  if (neg) set_big_sign(big_fixnum1,-1);

  return normalize_big_to_object(big_fixnum1);

}

static inline object
parse_unsigned_integer(char *s,char **ep,int radix) {

  return parse_unsigned_integer_negate(s,ep,radix,0);

}


static inline object
parse_integer(char *s,char **ep,int radix) {

  int negate=0;

  switch (*s) {
  case '-':
    negate=1;
  case '+':
    s++;
  default:
    break;
  }

  return parse_unsigned_integer_negate(s,ep,radix,negate);

}


static inline object
parse_number(object in,char *s,int radix) {

  object x,y;
  char *q,ch,c;
  int n,m;
  float f;
  double d;

  x=parse_integer(s,&q,radix);

  switch (*q) {
  case 0:
    return x;
  case '/':
    y=parse_unsigned_integer(q+1,&q,radix);
    return (x==OBJNULL || y==OBJNULL || *q) ? OBJNULL :
      (number_zerop(y) ? (READER_ERROR(in,"Ratio denominator is zero"),Cnil) : make_ratio(x,y,0));
  default:
    if (radix!=10)
      x=parse_integer(s,&q,10);

    if ((ch=*q)=='.') {
      if (!*++q) return x;
      parse_unsigned_integer(q,&q,10);
      ch=*q ? *q : 'E';
    }
    ch=ch=='e' || ch=='E' ? READdefault_float_format : ch;

    if ((c=*q)) {
      if (parse_integer(q+1,NULL,10)==OBJNULL)
      	return OBJNULL;
      *q='E';
    }
    if (ch=='s' || ch=='S')
      BLOCK_EXCEPTIONS(n=sscanf(s,"%f%n",&f,&m));
    else
      BLOCK_EXCEPTIONS(n=sscanf(s,"%lf%n",&d,&m));
    *q=c;
    if (n!=1||s[m]) return OBJNULL;

    switch (ch=='e' || ch=='E' ? READdefault_float_format : ch) {
    case 's':case 'S':
      return make_shortfloat(f);
    case 'f':case 'F':case 'd':case 'D':case 'l':case 'L':
      return make_longfloat(d);
    default:
      return OBJNULL;
    }
  }
}

static inline void
too_long_token(void) {
  char *q;
  int i;
  
  BEGIN_NO_INTERRUPT;
  q = alloc_contblock(token->st.st_dim*2);
  for (i = 0;  i < token->st.st_dim;  i++)
    q[i] = token->st.st_self[i];
  token->st.st_self = q;
  token->st.st_dim *= 2;
  END_NO_INTERRUPT;

}

static inline void
null_terminate_token(void) {

  if (token->st.st_fillp==token->st.st_dim)
    too_long_token();
  token->st.st_self[token->st.st_fillp]=0;

}


#define	token_buffer	token->st.st_self
/* the active length of the token */
int tok_leng;



object dispatch_reader;


#define	cat(c)	(READtable->rt.rt_self[char_code((c))] \
		 .rte_chattrib)
#define	trt(c)	(READtable->rt.rt_self[char_code((c))] \
		 .rte_chatrait)

static void
setup_READtable()
{
	READtable = current_readtable();
}

/* FIXME What should this be? Apparently no reliable way to use value stack */ 
#define MAX_PACKAGE_STACK 1024
static object P0[MAX_PACKAGE_STACK],*PP0=P0,LP;

static int inlp;

static void
setup_READ()
{
	object x;

	READtable = current_readtable();
	x = symbol_value(sLAread_default_float_formatA);
	if (x == sLshort_float)
		READdefault_float_format = 'S';
	else if (x == sLsingle_float || x == sLdouble_float || x == sLlong_float)
		READdefault_float_format = 'F';
	else {
		vs_push(x);
		sLAread_default_float_formatA->s.s_dbind = sLsingle_float;
	FEerror("The value of *READ-DEFAULT-FLOAT-FORMAT*, ~S, was illegal.",
			1, x);
	}
	x = symbol_value(sLAread_baseA);
	if (type_of(x) != t_fixnum || fix(x) < 2 || fix(x) > 36) {
		vs_push(x);
		sLAread_baseA->s.s_dbind = make_fixnum(10);
		FEerror("The value of *READ-BASE*, ~S, was illegal.", 1, x);
	}
	READbase = fix(x);
	READsuppress = symbol_value(sLAread_suppressA) != Cnil;
	READeval = symbol_value(sLAread_evalA) != Cnil;
	sSAsharp_eq_contextA->s.s_dbind=Cnil;

	backq_level = 0;

	PP0=P0;
	LP=NULL;
	inlp=0;

}

object
read_char(in)
object in;
{
	return(code_char(readc_stream(in)));
}


static object
read_char_no_echo(object in) {

  int c;

  if (type_of(in)==t_stream && in->sm.sm_mode==smm_echo) {
    c = readc_stream(STREAM_INPUT_STREAM(in));
    if (ECHO_STREAM_N_UNREAD(in) != 0)
      --(ECHO_STREAM_N_UNREAD(in));
    return(code_char(c));
  } else
    return(code_char(readc_stream(in)));

}

#define	read_char(in)	code_char(readc_stream(in))

static void
unread_char(c, in)
object c, in;
{
	if (type_of(c) != t_character)
		FEwrong_type_argument(sLcharacter, c);
	unreadc_stream(char_code(c), in);
}

/*
	Peek_char corresponds to COMMON Lisp function PEEK-CHAR.
	When pt is TRUE, preceeding whitespaces are ignored.
*/
object
peek_char(pt, in)
bool pt;
object in;
{
	object c;

	if (pt) {
		do
			c = read_char(in);
		while (cat(c) == cat_whitespace);
		unread_char(c, in);
		return(c);
	} else {
		c = read_char(in);
		unread_char(c, in);
		return(c);
	}
}
		

static object
read_object_recursive(in)
object in;
{
	VOL object x;
	bool e;

	object old_READtable = READtable;
	int old_READdefault_float_format = READdefault_float_format;
	int old_READbase = READbase;
	bool old_READsuppress = READsuppress;
	bool old_READeval = READeval;

	/* BUG FIX by Toshiba */
	vs_push(old_READtable);

	frs_push(FRS_PROTECT, Cnil);
	if (nlj_active) {
		e = TRUE;
		goto L;
	}

	READtable = current_readtable();
	x = symbol_value(sLAread_default_float_formatA);
	if (x == sLshort_float)
		READdefault_float_format = 'S';
	else if (x == sLsingle_float || x == sLdouble_float || x == sLlong_float)
		READdefault_float_format = 'F';
	else {
		vs_push(x);
		sLAread_default_float_formatA->s.s_dbind = sLsingle_float;
	FEerror("The value of *READ-DEFAULT-FLOAT-FORMAT*, ~S, was illegal.",
			1, x);
	}
	x = symbol_value(sLAread_baseA);
	if (type_of(x) != t_fixnum || fix(x) < 2 || fix(x) > 36) {
		vs_push(x);
		sLAread_baseA->s.s_dbind = make_fixnum(10);
		FEerror("The value of *READ-BASE*, ~S, was illegal.", 1, x);
	}
	READbase = fix(x);
	READsuppress = symbol_value(sLAread_suppressA) != Cnil;
	READeval = symbol_value(sLAread_evalA) != Cnil;

	x = read_object(in);
	e = FALSE;

L:
	frs_pop();

	READtable = old_READtable;
	READdefault_float_format = old_READdefault_float_format;
	READbase = old_READbase;
	READsuppress = old_READsuppress;
	READeval = old_READeval;

	/* BUG FIX by Toshiba */
	vs_popp;

	if (e) {
		nlj_active = FALSE;
		unwind(nlj_fr, nlj_tag);
	}

	return(x);
}


object
read_object_non_recursive(in)
object in;
{
	VOL object x;
	bool e;
	object old_READtable;
	int old_READdefault_float_format;
	int old_READbase;
	int old_READsuppress;
	int old_READeval;
	object old_READcontext;
	int old_backq_level;

	old_READtable = READtable;
	old_READdefault_float_format = READdefault_float_format;
	old_READbase = READbase;
	old_READsuppress = READsuppress;
	old_READeval = READeval;
	old_READcontext=sSAsharp_eq_contextA->s.s_dbind;

	/* BUG FIX by Toshiba */
	vs_push(old_READtable);
	old_backq_level = backq_level;
	setup_READ();

	frs_push(FRS_PROTECT, Cnil);
	if (nlj_active) {
		e = TRUE;
		goto L;
	}

	x = read_object(in);
	vs_push(x);

#ifndef _WIN32        
	while (fLinteractive_stream_p(in)!=Cnil && listen_stream(in)) {
	  object c=read_char(in);
	  if (cat(c)!=cat_whitespace) {
	    unread_char(c,in);
	    break;
	  }
	}
#endif        

	if (sSAsharp_eq_contextA->s.s_dbind!=Cnil)
	  x = vs_head = patch_sharp(x);

	e = FALSE;

L:
	frs_pop();

	READtable = old_READtable;
	READdefault_float_format = old_READdefault_float_format;
	READbase = old_READbase;
	READsuppress = old_READsuppress;
	READeval = old_READeval;
	sSAsharp_eq_contextA->s.s_dbind=old_READcontext;
	backq_level = old_backq_level;
	if (e) {
		nlj_active = FALSE;
		unwind(nlj_fr, nlj_tag);
	}
	vs_popp;
	/* BUG FIX by Toshiba */
	vs_popp;
	return(x);
}

#ifdef UNIX  /* faster code for inner loop from file stream */
#define xxxread_char_to(res,in,eof_code) \
  do{FILE *fp; \
      if(fp=in->sm.sm_fp) \
	{int ch = getc(fp); \
      if (ch==EOF) { \
	if (feof(fp)) { eof_code;} \
          else if (in->sm.sm_mode==smm_socket) \
             {  ch = getOneChar(fp); \
	       if (ch==EOF) { eof_code;}}} \
       else res=code_char(ch);} \
      else \
	{ if (stream_at_end(in)) \
	    {eof_code;} \
	else res=read_char(in);}} while(0)

#define read_char_to(res,in,eof_code)		\
  do{FILE *fp;					\
    if((fp=in->sm.sm_fp))			\
      {int ch = getc(fp);			\
	if (ch==EOF && feof(fp))		\
	  { eof_code;}				\
	else res=code_char(ch);}		\
    else					\
      {int ch ;					\
	if(stream_at_end(in)) {eof_code ;}	\
	ch = readc_stream(in);						\
	if (ch == EOF && stream_at_end(in)) { eof_code;}		\
	res = code_char(ch);						\
      }} while(0)
#define read_char_to2(in,eof_code)					\
  ({FILE *fp=in->sm.sm_fp;int ch;					\
    fp ?								\
      ((ch=getc(fp))==EOF && feof(fp) ? ({eof_code;Cnil;}) : code_char(ch)) : \
      (stream_at_end(in) || ((ch=readc_stream(in))==EOF && stream_at_end(in)) ? \
       ({eof_code;Cnil;}) : code_char(ch));})
#else
#define read_char_to(res,in,eof_code)	\
  do {if(stream_at_end(in)) {eof_code ;}	\
    else { int ch = readc_stream(in);		\
      if (ch == EOF) { eof_code;}		\
      res = code_char(ch);			\
    }						\
  } while(0)
#endif

/*
	Read_object(in) reads an object from stream in.
	This routine corresponds to COMMON Lisp function READ.
*/

object
read_object(in)
object in;
{
	object x;
	object c=Cnil;
	enum chattrib a;
	object *old_vs_base;
	object result;
	object p;
	int  colon=0, colon_type;
	int i;
	bool df, ilf;
	VOL int length;
	vs_mark;

	cs_check(in);

	vs_check_push(delimiting_char);
	delimiting_char = OBJNULL;
	df = detect_eos_flag;
	detect_eos_flag = FALSE;
	ilf = in_list_flag;
	in_list_flag = FALSE;
	dot_flag = FALSE;

BEGIN:
	do { read_char_to(c,in, {
	  if (df) {
	    vs_reset;
	    return(OBJNULL);
	  } else 
	    end_of_stream(in);
	});
		a = cat(c);
	} while (a == cat_whitespace);

	delimiting_char = vs_head;
	if (delimiting_char != OBJNULL && c == delimiting_char) {
		delimiting_char = OBJNULL;
		vs_reset;
		LP=NULL;
		return(OBJNULL);
	}
	delimiting_char = OBJNULL;
	if (a == cat_terminating || a == cat_non_terminating)
	{
		object *fun_box = vs_top;

		old_vs_base = vs_base;
		vs_push(Cnil);
		vs_base = vs_top;
		vs_push(in);
		vs_push(c);

		x =
		READtable->rt.rt_self[char_code(c)].rte_macro;
		fun_box[0] = x;
		super_funcall(x);

		LP=NULL;
		i = vs_top - vs_base;
		if (i == 0) {
			vs_base = old_vs_base;
			vs_top = old_vs_top + 1;
			goto BEGIN;
		}
		if (i > 1) {
			vs_push(make_fixnum(i));
			FEerror("The readmacro ~S returned ~D values.",
				 2, fun_box[0], vs_top[-1]);
		}
		result = READsuppress ? Cnil : vs_base[0]; /*FIXME, centralize here rather than in
							     reader macros ??*/
		vs_base = old_vs_base;
		vs_reset;
		return(result);
	}
	escape_flag = FALSE;
	length = 0; tok_leng=0;
	colon_type = 0;
	goto L;
	for (;;) {
		if (length >= token->st.st_dim)
			too_long_token();
		token_buffer[(tok_leng++,length++)] = char_code(c);
	K:
		read_char_to(c,in,goto M);
		a = cat(c);
	L:
		if (a == cat_single_escape) {
			c = read_char(in);
			a = cat_constituent;
			escape_flag = TRUE;
		} else if (a == cat_multiple_escape) {
			escape_flag = TRUE;
			for (;;) {
				if (stream_at_end(in))
					end_of_stream(in);
				c = read_char(in);
				a = cat(c);
				if (a == cat_single_escape) {
					c = read_char(in);
					a = cat_constituent;
				} else if (a == cat_multiple_escape)
					break;
				if (length >= token->st.st_dim)
					too_long_token();
				token_buffer[(tok_leng++,length++)] = char_code(c);
			}
			goto K;
		} else if (a == cat_terminating) {
			break;
        	} else if (a == cat_whitespace) {
		  /* skip all whitespace after trailing colon if no escape seen */
		  if (colon+colon_type==length && !escape_flag)
		    goto K;
		  else
		    break;
		} else {

		  if (trt(c)==trait_invalid)
		    READER_ERROR(in,"Cannot read character");

		  if ('A' <= char_code(c) && char_code(c) <= 'z') {
		    if ('a' <= char_code(c) && char_code(c) <= 'z' && 
			(READtable->rt.rt_case==sKupcase || READtable->rt.rt_case==sKinvert))
		      c = code_char(char_code(c) - ('a' - 'A'));
		    else if ('A' <= char_code(c) && char_code(c) <= 'Z' && 
			     (READtable->rt.rt_case==sKdowncase || READtable->rt.rt_case==sKinvert))
		      c = code_char(char_code(c) + ('a' - 'A'));
		  } else if (char_code(c) == ':') {
		    if (colon_type == 0) {
		      colon_type = 1;
		      colon = length;
		    } else if (colon_type == 1 && colon == length-1)
		      colon_type = 2;
		    else
		      colon_type = -1;
		    /*  Colon has appeared twice.  */
		  }
		}
        }
	if (preserving_whitespace_flag || cat(c) != cat_whitespace)
		unread_char(c, in);

M:
	if (READsuppress) {
		token->st.st_fillp = length;
		vs_reset;
		return(Cnil);
	}
	if (ilf && !escape_flag &&
	    length == 1 && token->st.st_self[0] == '.') {
		dot_flag = TRUE;
		vs_reset;
		return(Cnil);
	 } else if (!escape_flag && length > 0) {
	 	for (i = 0;  i < length;  i++)
	 		if (token->st.st_self[i] != '.')
	 			goto N;
	 	READER_ERROR(in,"Dots appeared illegally.");
	}

N:
	token->st.st_fillp = length;
	if (escape_flag || (READbase<=10 && token_buffer[0]>'9'))
	  goto SYMBOL;
	null_terminate_token();
	x = parse_number(in,token_buffer, READbase);
	if (x != OBJNULL) {
		vs_reset;
		return(x);
	}

SYMBOL:
	if (colon_type == 1 /* && length > colon + 1 */) {
		if (colon == 0)
			p = keyword_package;
		else {
			token->st.st_fillp = colon;
			p = find_package(token);
			if (p == Cnil) {
			    vs_push(copy_simple_string(token));
			    FEerror("There is no package with the name \"~A\".",
				    1, vs_head);
			}
		}
		for (i = colon + 1;  i < length;  i++)
			token_buffer[i - (colon + 1)]
			= token_buffer[i];
		token->st.st_fillp = length - (colon + 1);
		if (colon > 0) {
			x = find_symbol(token, p);
			if (intern_flag != EXTERNAL) {
				vs_push(copy_simple_string(token));
			FEerror("Cannot find the external symbol ~A in ~S.",
						2, vs_head, p);
				/*  no need to push a package  */
			}
			vs_reset;
			return(x);
		}
	} else if (colon_type == 2 /* && colon > 0 && length > colon + 2 */) {
		token->st.st_fillp = colon;
		p = find_package(token);
		if (p == Cnil) {
			vs_push(copy_simple_string(token));
			FEerror("There is no package with the name \"~A\".",
				1, vs_head);
		}
		for (i = colon + 2;  i < length;  i++)
			token_buffer[i - (colon + 2)]
			= token_buffer[i];
		token->st.st_fillp = length - (colon + 2);
	} else
		p = current_package();
	/* loose package is an empty token following a non-beginning 
	   colon with no escape, to allow for ||*/
	if (!token->st.st_fillp && colon && !escape_flag) {
	  LP=p;
	  goto BEGIN;
	}
	/* unless package specified for this symbol, use loose package if present */
	if (PP0>P0 && !colon_type)
	  p=PP0[-1];
	vs_push(p);
	x = intern(token, p);
	vs_push(x);
	vs_reset;
	return(x);
}

static void
Lleft_parenthesis_reader()
{
	object in, x;
	object *p;

	check_arg(2);
	in = vs_base[0];
	vs_head = Cnil;
	p = &vs_head;
	inlp=1;
	LP=LP || PP0==P0 ? LP : PP0[-1]; /* push loose packages into nested lists */
	if (LP) {
	  if (PP0-P0>=MAX_PACKAGE_STACK)
	    FEerror("Too many nested package specifiers",0);
	  *PP0++=LP;
	  LP=NULL;
	}
	for (;;) {
		delimiting_char = code_char(')');
		in_list_flag = TRUE;
		x = read_object(in);
		if (x == OBJNULL)
			goto ENDUP;
		if (dot_flag) {
		  if (p == &vs_head) 
		    READER_ERROR(in,"A dot appeared after a left parenthesis.");
		  delimiting_char = code_char(')');
		  in_list_flag = TRUE;
		  *p = read_object(in);
		  if (dot_flag)
		    READER_ERROR(in,"Two dots appeared consecutively.");
		  if (*p==OBJNULL) 
		    READER_ERROR(in,"Object missing after dot.");
		  delimiting_char = code_char(')');
		  in_list_flag = TRUE;
		  if (read_object(in)!=OBJNULL)
		    READER_ERROR(in,"Two objects after dot.");
		  goto ENDUP;
		}
		vs_push(x);
		*p = make_cons(x, Cnil);
		vs_popp;
		p = &((*p)->c.c_cdr);
	}

ENDUP:
	if (PP0>P0) PP0--;
	vs_base[0] = vs_pop;
	inlp=0;
	return;
}

/*
	Read_string(delim, in) reads
	a simple string	terminated by character code delim
	and places it in token.
	Delim is not included in the string but discarded.
*/
static void
read_string(delim, in)
int delim;
object in;
{
	int i;
	object c;

	i = 0;
	for (;;) {
		c = read_char(in);
		if (char_code(c) == delim)
			break;
		else if (cat(c) == cat_single_escape)
		  c = read_char(in);/*FIXME continue*/
		if (i >= token->st.st_dim)
			too_long_token();
		token_buffer[i++] = char_code(c);
	}
	token->st.st_fillp = i;
}

/*
	Read_constituent(in) reads
	a sequence of constituent characters from stream in
	and places it in token_buffer.
*/
static void
read_constituent(in)
object in;
{
	int i, j;
	object c;

	i = 0;
	for (;;) {
                read_char_to(c,in,goto FIN);
		if (cat(c) != cat_constituent && !(READsuppress && cat(c)==cat_non_terminating)) {
			unread_char(c, in);
			break;
		}
		j = char_code(c);
		token_buffer[i++] = j;
	}
      FIN:
	token->st.st_fillp = i;
	
}

DEFUN("DOUBLE-QUOTE-READER",object,fSdouble_quote_reader,SI,2,2,NONE,OO,OO,OO,OO,
	  (object s,object c),"") {

  read_string(char_code(c),s);
  RETURN1(copy_simple_string(token));

}

/* static void */
/* Ldouble_quote_reader() */
/* { */
/*   object c; */
/*   check_arg(2); */
/*   c=vs_base[1]; */
/*   vs_popp; */
/*   read_string(char_code(c), vs_base[0]); */
/*   vs_base[0] = copy_simple_string(token); */

/* } */

static void
Ldispatch_reader()
{
	object c, x;
	int i, j;
	object in;

	check_arg(2);
	
	in = vs_base[0];
	c = vs_base[1];

	if (READtable->rt.rt_self[char_code(c)].rte_dtab == NULL)
		FEerror("~C is not a dispatching macro character", 1, c);


	for (i=0;i<token->st.st_dim;i++) {
	  c=read_char(in);
	  j=char_code(c);
	  if (digitp(j,10)<0)
	    break;
	  token->st.st_self[i]=j;
	}
	if (i==token->st.st_dim)
	  FEerror("Dispatch number too long", 0);
	if (i) {
	  token->st.st_fillp=i;
	  null_terminate_token();
	  x=parse_number(in,token->st.st_self,10);
	  if (x == OBJNULL)
	    FEerror("Cannot parse the dispatch macro number.", 0);
	} else
	  x=Cnil;
	vs_push(x);
	
	x =
	READtable->rt.rt_self[char_code(vs_base[1])].rte_dtab[char_code(c)];
	vs_base[1] = c;
	super_funcall(x);
}

DEFUN("SINGLE-QUOTE-READER",object,fSsingle_quote_reader,SI,2,2,NONE,OO,OO,OO,OO,
	  (object s,object c),"") {
  RETURN1(MMcons(sLquote,MMcons(read_object(s),Cnil)));
}

/* static void */
/* Lsingle_quote_reader() */
/* { */
/* 	check_arg(2); */
/* 	vs_popp; */
/* 	vs_push(sLquote); */
/* 	vs_push(read_object(vs_base[0])); */
/* 	vs_push(Cnil); */
/* 	stack_cons(); */
/* 	stack_cons(); */
/* 	vs_base[0] = vs_pop; */
/* } */

DEFUN("RIGHT-PARENTHESIS-READER",object,fSright_parenthesis_reader,SI,2,2,NONE,OO,OO,OO,OO,
	  (object s,object c),"") {
  if (!inlp)
    READER_ERROR(vs_base[0],"Right paren found with no left.");
  RETURN1(Cnil);
}

/* static void */
/* Lright_parenthesis_reader() */
/* { */
/* 	check_arg(2); */
/* 	if (!inlp) */
/* 	  READER_ERROR(vs_base[0],"Right paren found with no left."); */
/* 	vs_popp; */
/* 	vs_popp; */
/* 		/\*  no result  *\/ */
/* } */

/*
Lcomma_reader(){}
*/

DEFUNM("SEMICOLON-READER",object,fSsemicolon_reader,SI,2,2,NONE,OO,OO,OO,OO,
	  (object str,object c),"") {

  fixnum vals=fcall.valp;
  object *base=vs_top;

  do
    { read_char_to(c,str, goto L); }
  while (char_code(c) != '\n');
 L:	
  RETURN0;
}

/* static void */
/* Lsemicolon_reader() */
/* { */
/* 	object c; */
/* 	object str= vs_base[0]; */
/* 	check_arg(2); */
/* 	vs_popp; */
/* 	do */
/* 	{ read_char_to(c,str, goto L); } */
/* 		while (char_code(c) != '\n'); */
/* L:	 */
/* 	vs_popp; */
/* 	vs_base[0] = Cnil; */
/* 	/\*  no result  *\/ */
/* } */

/*
Lbackquote_reader(){}
*/

/*
	sharpmacro routines
*/
static void
extra_argument(int);

DEFUN("SHARP-C-READER",object,fSsharp_c_reader,SI,3,3,NONE,OO,OO,OO,OO,(object s,object x,object y),"") {
  
  object r,i;

  x=OBJNULL;
  if (y!=Cnil && !READsuppress)
    extra_argument('C');
  if (READsuppress) {
    read_object(s);
    s= Cnil;
  } else {
    do {read_char_to(x,s, {READER_ERROR(s,"A left parenthesis is expected.");});
    } while (cat(x) == cat_whitespace);
    if (char_code(x) != '(')
      READER_ERROR(s,"A left parenthesis is expected.");
    delimiting_char = code_char(')');
    x = read_object(s);
    if (x==OBJNULL || !realp(x))
      TYPE_ERROR(x,TSor_rational_float);
    r=x;
    delimiting_char = code_char(')');
    x = read_object(s);
    if (x==OBJNULL || !realp(x))
      TYPE_ERROR(x,TSor_rational_float);
    i=x;
    delimiting_char = code_char(')');
    x = read_object(s);
    if (x != OBJNULL)
      READER_ERROR(s,"A right parenthesis is expected.");
    if (contains_sharp_comma(r) || contains_sharp_comma(i)) {
      s = alloc_object(t_complex);
      s->cmp.cmp_real = r;
      s->cmp.cmp_imag = i;
    } else {
      check_type_number(&r);
      check_type_number(&i);
      s = make_complex(r, i);
    }
  }
  RETURN1(s);

}

/* static void */
/* Lsharp_C_reader() */
/* { */
/* 	object x=OBJNULL; */

/* 	check_arg(3); */
/* 	if (vs_base[2] != Cnil && !READsuppress) */
/* 		extra_argument('C'); */
/* 	vs_popp; */
/* 	vs_popp; */

/* 	if (READsuppress) { */
/* 	  read_object(vs_base[0]); */
/* 	  vs_base[0]= Cnil; */
/* 	} else { */
/* 	  do {read_char_to(x,vs_base[0], { */
/* 	    READER_ERROR(vs_base[0],"A left parenthesis is expected."); */
/* 	  }); */
/* 	  } while (cat(x) == cat_whitespace); */
/* 	  if (char_code(x) != '(') */
/* 	    READER_ERROR(vs_base[0],"A left parenthesis is expected."); */
/* 	  delimiting_char = code_char(')'); */
/* 	  x = read_object(vs_base[0]); */
/* 	  if (x==OBJNULL || !realp(x)) */
/* 	    TYPE_ERROR(x,TSor_rational_float); */
/* 	  vs_push(x); */
/* 	  delimiting_char = code_char(')'); */
/* 	  x = read_object(vs_base[0]); */
/* 	  if (x==OBJNULL || !realp(x)) */
/* 	    TYPE_ERROR(x,TSor_rational_float); */
/* 	  vs_push(x); */
/* 	  delimiting_char = code_char(')'); */
/* 	  x = read_object(vs_base[0]); */
/* 	  if (x != OBJNULL) */
/* 	    READER_ERROR(vs_base[0],"A right parenthesis is expected."); */
/* 	  if (contains_sharp_comma(vs_base[1]) || */
/* 	      contains_sharp_comma(vs_base[2])) { */
/* 	    vs_base[0] = alloc_object(t_complex); */
/* 	    vs_base[0]->cmp.cmp_real = vs_base[1]; */
/* 	    vs_base[0]->cmp.cmp_imag = vs_base[2]; */
/* 	  } else { */
/* 	    check_type_number(&vs_base[1]); */
/* 	    check_type_number(&vs_base[2]); */
/* 	    vs_base[0] = make_complex(vs_base[1], vs_base[2]); */
/* 	  } */
/* 	} */
/* 	vs_top = vs_base + 1; */
/* } */

DEFUN("SHARP-\\-READER",object,fSsharp_sl_reader,SI,3,3,NONE,OO,OO,OO,OO,(object s,object x,object y),"") {

  object c,u;
  
  /*FIXME, read_token function*/
  token->st.st_fillp=0;
  token_buffer[token->st.st_fillp++]=char_code((c=read_char(s)));/*FIXME, eof?*/
  for (u=Cnil;u!=c;)
    switch (cat((c=read_char_to2(s,break)))) {
    case cat_single_escape: c=read_char_to2(s,u=c;break);
    case cat_constituent: token_buffer[token->st.st_fillp++]=char_code(c);
    case cat_multiple_escape: continue;
    default: unread_char(u=c,s);
    }

  if (READsuppress)
    RETURN1(Cnil);
  c = token;
  if (c->st.st_fillp == 1) {
    RETURN1(code_char(c->ust.ust_self[0]));
  }
  if (string_equal(c, STreturn))
    s = code_char('\r');
  else if (string_equal(c, STspace))
    s = code_char(' ');
  else if (string_equal(c, STrubout))
    s = code_char('\177');
  else if (string_equal(c, STpage))
    s = code_char('\f');
  else if (string_equal(c, STtab))
    s = code_char('\t');
  else if (string_equal(c, STbackspace))
    s = code_char('\b');
  else if (string_equal(c, STlinefeed) || string_equal(c, STnewline))
    s = code_char('\n');
  else if (c->st.st_fillp == 2 && c->st.st_self[0] == '^')
    s = code_char(c->st.st_self[1] & 037);
  else if (c->st.st_self[0] =='\\' && c->st.st_fillp > 1) {
    int i, n;
    for (n = 0, i = 1;  i < c->st.st_fillp;  i++)
      if (c->st.st_self[i] < '0' || '7' < c->st.st_self[i])
	FEerror("Octal digit expected.", 0);
      else
	n = 8*n + c->st.st_self[i] - '0';
    s = code_char(n & 0377);
  } else
    FEerror("~S is an illegal character name.", 1, c);
  RETURN1(s);
}

/* static void */
/* Lsharp_backslash_reader() */
/* { */
/* 	object c; */

/* 	check_arg(3); */
/* 	if (vs_base[2] != Cnil && !READsuppress) */
/* 		if (type_of(vs_base[2]) != t_fixnum || */
/* 		    fix(vs_base[2]) != 0) */
/* 			FEerror("~S is an illegal CHAR-FONT.", 1, vs_base[2]); */
/* 			/\*  assuming that CHAR-FONT-LIMIT is 1  *\/ */
/* 	vs_popp; */
/* 	vs_popp; */
/* 	unread_char(code_char('\\'), vs_base[0]); */
/* 	if (READsuppress) { */
/* 		(void)read_object(vs_base[0]); */
/* 		vs_base[0] = Cnil; */
/* 		return; */
/* 	} */
/* 	READsuppress = TRUE; */
/* 	(void)read_object(vs_base[0]); */
/* 	READsuppress = FALSE; */
/* 	c = token; */
/* 	if (c->s.s_fillp == 1) { */
/* 		vs_base[0] = code_char(c->ust.ust_self[0]); */
/* 		return; */
/* 	} */
/* 	if (string_equal(c, STreturn)) */
/* 		vs_base[0] = code_char('\r'); */
/* 	else if (string_equal(c, STspace)) */
/* 		vs_base[0] = code_char(' '); */
/* 	else if (string_equal(c, STrubout)) */
/* 		vs_base[0] = code_char('\177'); */
/* 	else if (string_equal(c, STpage)) */
/* 		vs_base[0] = code_char('\f'); */
/* 	else if (string_equal(c, STtab)) */
/* 		vs_base[0] = code_char('\t'); */
/* 	else if (string_equal(c, STbackspace)) */
/* 		vs_base[0] = code_char('\b'); */
/* 	else if (string_equal(c, STlinefeed) || string_equal(c, STnewline)) */
/* 		vs_base[0] = code_char('\n'); */
/* 	else if (c->s.s_fillp == 2 && c->s.s_self[0] == '^') */
/* 		vs_base[0] = code_char(c->s.s_self[1] & 037); */
/* 	else if (c->s.s_self[0] =='\\' && c->s.s_fillp > 1) { */
/* 		int i, n; */
/* 		for (n = 0, i = 1;  i < c->s.s_fillp;  i++) */
/* 			if (c->s.s_self[i] < '0' || '7' < c->s.s_self[i]) */
/* 				FEerror("Octal digit expected.", 0); */
/* 			else */
/* 				n = 8*n + c->s.s_self[i] - '0'; */
/* 		vs_base[0] = code_char(n & 0377); */
/* 	} else */
/* 		FEerror("~S is an illegal character name.", 1, c); */
/* } */

DEFUN("SHARP-'-READER",object,fSsharp_q_reader,SI,3,3,NONE,OO,OO,OO,OO,(object s,object x,object y),"") {

  if(y != Cnil && !READsuppress)
    extra_argument('#');
  RETURN1(MMcons(sLfunction,MMcons(read_object(s),Cnil)));

}

static void
Lsharp_single_quote_reader()
{

	check_arg(3);
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument('#');
	vs_base[0] = list(2,sLfunction,read_object(vs_base[0]));
	vs_top=vs_base+1;
}

#define	QUOTE	1
#define	EVAL	2
#define	LIST	3
#define	LISTA	4
#define	APPEND	5
#define	NCONC	6

object siScomma;

static void
Lsharp_left_parenthesis_reader()
{

	int dim=0;
	int dimcount;
	object in, x;
	int a;
	object *vsp;		

	check_arg(3);
	if (vs_base[2] == Cnil || READsuppress)
		dim = -1;
	else if (type_of(vs_base[2]) == t_fixnum)
		dim = fix(vs_base[2]);
	vs_popp;
	vs_popp;
	in = vs_base[0];
	if (backq_level > 0) {
		unreadc_stream('(', in);
		vs_push(read_object(in));
		a = backq_car(vs_base[1]);
		if (a == APPEND || a == NCONC)
		  FEerror(",at or ,. has appeared in an illegal position.", 0);
		if (a == QUOTE) {
			vsp = vs_top;
			dimcount = 0;
			for (x = vs_base[2];  !endp(x);  x = x->c.c_cdr) {
				vs_check_push(x->c.c_car);
				dimcount++;
			}	
			goto L;
		}
		vs_base[0]=list(4,siScomma,sLapply,list(2,sLquote,sLvector),vs_base[2]);
		vs_top=vs_base+1;
		return;
	}
	vsp = vs_top;
	dimcount = 0;
	for (;;) {
		delimiting_char = code_char(')');
		x = read_object(in);
		if (x == OBJNULL)
			break;
		vs_check_push(x);
		dimcount++;
	}	
L:
	if (dim >= 0) {
		if (dimcount > dim)
			FEerror("Too many elements in #(...).", 0);
		else {
/* 			if (dimcount == 0) */
/* 				FEerror("Cannot fill the vector #().", 0); */
			x = vs_head;
			for (;  dimcount < dim;  dimcount++)
				vs_push(x);
		}
	}
        {BEGIN_NO_INTERRUPT;
	x = alloc_simple_vector(dimcount);
	vs_push(x);
	x->v.v_self
	= (object *)alloc_relblock(dimcount * sizeof(object));
	vs_popp;
	for (dim = 0; dim < dimcount; dim++)
		x->v.v_self[dim] = vsp[dim];
	vs_top = vs_base;
	END_NO_INTERRUPT;}
	vs_push(x);
}

static void
Lsharp_asterisk_reader()
{
	int dim=0;
	int dimcount;
	object in, x;
	object *vsp;		

	check_arg(3);
	if (READsuppress) {
		read_constituent(vs_base[0]);
		vs_popp;
		vs_popp;
		vs_base[0] = Cnil;
		return;
	}
	if (vs_head == Cnil)
		dim = -1;
	else if (type_of(vs_head) == t_fixnum)
		dim = fix(vs_head);
	vs_popp;
	vs_popp;
	in = vs_head;
	vsp = vs_top;
	dimcount = 0;
	for (;;) {
		if (stream_at_end(in))
			break;
		x = read_char(in);
		if (char_code(x) != '0' && char_code(x) != '1') {
		  if (cat(x)==cat_constituent)
		    READER_ERROR(in,"Invalid bit vector element");
		  unreadc_stream(char_code(x),in);
		  break;
		}
		vs_check_push(x);
		dimcount++;
	}	
	if (dim >= 0) {
	  if (dimcount > dim) {
	    READER_ERROR(in,"Too many elements in #*....");
	  } else {
/* 			if (dimcount == 0) */
/* 				FEerror("Cannot fill the bit-vector #*.",0); */
			x = vs_head;
			for (;  dimcount < dim;  dimcount++)
				vs_push(x);
		}
	}
	{BEGIN_NO_INTERRUPT;
	x = alloc_simple_bitvector(dimcount);
	vs_push(x);
	x->bv.bv_self = alloc_relblock(ceil(dimcount,BV_ALLOC)*sizeof(*x->bv.bv_self));
	vs_popp;
	for (dim = 0; dim < dimcount; dim++)
	  switch(char_code(vsp[dim])) {
	  case '0':
	    CLEAR_BITREF(x,dim);
	    break;
	  case '1':
	    SET_BITREF(x,dim);
	    break;
	  default:
	    READER_ERROR(in,"Invalid bit vector entry");
	    break;
	  }
	END_NO_INTERRUPT;}
	vs_top = vs_base;
	vs_push(x);
}

static void
Lsharp_colon_reader()
{
	object in;
	int length;
	object c;
	enum chattrib a;

	if (vs_base[2] != Cnil && !READsuppress)
		extra_argument(':');
	vs_popp;
	vs_popp;
	in = vs_base[0];
	c = read_char(in);
	a = cat(c);
	escape_flag = FALSE;
	length = 0; tok_leng=0;
	goto L;
	for (;;) {
		if (length >= token->st.st_dim)
			too_long_token();
		token_buffer[(tok_leng++,length++)] = char_code(c);
	K:
		if (stream_at_end(in))
			goto M;
		c = read_char(in);
		a = cat(c);
	L:
		if (a == cat_single_escape) {
			c = read_char(in);
			a = cat_constituent;
			escape_flag = TRUE;
		} else if (a == cat_multiple_escape) {
			escape_flag = TRUE;
			for (;;) {
				if (stream_at_end(in))
					end_of_stream(in);
				c = read_char(in);
				a = cat(c);
				if (a == cat_single_escape) {
					c = read_char(in);
					a = cat_constituent;
				} else if (a == cat_multiple_escape)
					break;
				if (length >= token->st.st_dim)
					too_long_token();
				token_buffer[(tok_leng++,length++)] = char_code(c);
			}
			goto K;
		} else if ('a' <= char_code(c) && char_code(c) <= 'z')
			c = code_char(char_code(c) - ('a' - 'A'));
		if (a == cat_whitespace || a == cat_terminating)
			break;
	}
	if (preserving_whitespace_flag || cat(c) != cat_whitespace)
		unread_char(c, in);

M:
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	token->st.st_fillp = length;
	vs_base[0] = copy_simple_string(token);
	vs_base[0] = make_symbol(vs_base[0]);
}

static void
Lsharp_dot_reader()
{
	check_arg(3);
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument('.');
	vs_popp;
	vs_popp;
	if (READsuppress) {
		read_object(vs_base[0]);	
		vs_base[0] = Cnil;
		return;
	}
	if (READeval) {
	  vs_base[0] = read_object(vs_base[0]);
	  /*FIXME: assumes no sharing until patch-sharp appears.  Bootstrap requires *fasd-data**/
	  vs_base[0] = ieval(sSpatch_sharp->s.s_gfdef!=OBJNULL ? patch_sharp(vs_base[0]) : vs_base[0]);
	} else {
	  READER_ERROR(vs_base[0],"Sharp dot found with *read-eval* set to nil");
	}
}

static void
Lsharp_comma_reader()
{
	check_arg(3);
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument(',');
	vs_popp;
	vs_popp;
	if (READsuppress) {
		read_object(vs_base[0]);
		vs_base[0] = Cnil;
		return;
	}
	vs_base[0] = read_object(vs_base[0]);
	vs_base[0] = ieval(vs_base[0]);
}

static void
FFN(siLsharp_comma_reader_for_compiler)()
{
	check_arg(3);
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument(',');
	vs_popp;
	vs_popp;
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	vs_base[0] = read_object(vs_base[0]);
	vs_base[0] = make_cons(siSsharp_comma, vs_base[0]);
}

static void
Lsharp_B_reader()
{
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument('B');
	vs_popp;
	vs_popp;
	read_constituent(vs_base[0]);
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	null_terminate_token();
	vs_base[0]
	  = parse_number(vs_base[0],token_buffer, 2);
	if (vs_base[0] == OBJNULL)
		FEerror("Cannot parse the #B readmacro.", 0);
	if (type_of(vs_base[0]) == t_shortfloat ||
	    type_of(vs_base[0]) == t_longfloat)
		FEerror("The float ~S appeared after the #B readmacro.",
			1, vs_base[0]);
}

static void
Lsharp_O_reader()
{
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument('O');
	vs_popp;
	vs_popp;
	read_constituent(vs_base[0]);
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	null_terminate_token();
	vs_base[0]
	  = parse_number(vs_base[0],token_buffer, 8);
	if (vs_base[0] == OBJNULL)
		FEerror("Cannot parse the #O readmacro.", 0);
	if (type_of(vs_base[0]) == t_shortfloat ||
	    type_of(vs_base[0]) == t_longfloat)
		FEerror("The float ~S appeared after the #O readmacro.",
			1, vs_base[0]);
}

static void
Lsharp_X_reader()
{
	if(vs_base[2] != Cnil && !READsuppress)
		extra_argument('X');
	vs_popp;
	vs_popp;
	read_constituent(vs_base[0]);
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	null_terminate_token();
	vs_base[0]
	  = parse_number(vs_base[0],token_buffer, 16);
	if (vs_base[0] == OBJNULL)
		FEerror("Cannot parse the #X readmacro.", 0);
	if (type_of(vs_base[0]) == t_shortfloat ||
	    type_of(vs_base[0]) == t_longfloat)
		FEerror("The float ~S appeared after the #X readmacro.",
			1, vs_base[0]);
}

static void
Lsharp_R_reader()
{
	int radix=0;

	check_arg(3);
	if (READsuppress)
		radix = 10;
	else if (type_of(vs_base[2]) == t_fixnum) {
		radix = fix(vs_base[2]);
		if (radix > 36 || radix < 2)
			FEerror("~S is an illegal radix.", 1, vs_base[2]);
	} else
		FEerror("No radix was supplied in the #R readmacro.", 0);
	vs_popp;
	vs_popp;
	read_constituent(vs_base[0]);
	if (READsuppress) {
		vs_base[0] = Cnil;
		return;
	}
	null_terminate_token();
	vs_base[0]
	  = parse_number(vs_base[0],token_buffer, radix);
	null_terminate_token();
	vs_base[0]
	  = parse_number(vs_base[0],token_buffer, radix);
	if (vs_base[0] == OBJNULL)
		FEerror("Cannot parse the #R readmacro.", 0);
	if (type_of(vs_base[0]) == t_shortfloat ||
	    type_of(vs_base[0]) == t_longfloat)
		FEerror("The float ~S appeared after the #R readmacro.",
			1, vs_base[0]);
}

static void Lsharp_plus_reader(){}

static void Lsharp_minus_reader(){}

static void
Lsharp_vertical_bar_reader() {

  object c=OBJNULL;
  int level = 0;
  
  check_arg(3);
  if (vs_base[2] != Cnil && !READsuppress)
    extra_argument('|');
  vs_popp;
  vs_popp;
  for (;;) {
    read_char_to(c,vs_base[0],goto END);
  L:
    if (char_code(c) == '#') {
      read_char_to(c,vs_base[0],goto END);
      if (char_code(c) == '|')
	level++;
    } else if (char_code(c) == '|') {
      read_char_to(c,vs_base[0],goto END);
      if (char_code(c) == '#') {
	if (level == 0)
	  break;
	else
	  --level;
      } else
	goto L;
    }
  }
 END:
  vs_popp;
  vs_base[0] = Cnil;
  /*  no result  */
}

static void
Ldefault_dispatch_macro()
{
	READER_ERROR(vs_base[0],"The default dispatch macro signalled an error.");
}

/*
	#$ fixnum returns a random-state with the fixnum
	as its content.
*/
static void
Lsharp_dollar_reader()
{
	object x;
	enum type tx;

	check_arg(3);
	if (vs_base[2] != Cnil && !READsuppress)
		extra_argument('$');
	vs_popp;
	vs_popp;
	x = read_object(vs_base[0]);
	tx=type_of(x);
	vs_base[0] = alloc_object(t_random);
	init_gmp_rnd_state(&vs_base[0]->rnd.rnd_state);
	if (tx!=t_fixnum || fix(x)) {
	  if (tx==t_fixnum) {
	    if (vs_base[0]->rnd.rnd_state._mp_seed->_mp_size!=1)
	      FEerror("Cannot make a random-state with the value ~S.",1, x);
	    mpz_set_ui(vs_base[0]->rnd.rnd_state._mp_seed,fix(x));
	  } else {
	    if (x->big.big_mpz_t._mp_size!=vs_base[0]->rnd.rnd_state._mp_seed->_mp_size)
	      FEerror("Cannot make a random-state with the value ~S.",1, x);
	    memcpy(vs_base[0]->rnd.rnd_state._mp_seed->_mp_d,x->big.big_mpz_t._mp_d,
		   vs_base[0]->rnd.rnd_state._mp_seed->_mp_size*sizeof(*vs_base[0]->rnd.rnd_state._mp_seed->_mp_d));
	  }
	}

}

/*
	readtable routines
*/

static object
copy_readtable(object from,object to) {

  struct rtent *rtab;
  int i, j;
  vs_mark;

  {BEGIN_NO_INTERRUPT;

  if (to == Cnil) {
    to = alloc_object(t_readtable);
    to->rt.rt_self = NULL;
    to->rt.rt_case = OBJNULL;
    /*  For GBC not to go mad.  */
    vs_push(to);
    /*  Saving for GBC.  */
    to->rt.rt_self
      = rtab
      = (struct rtent *)
      alloc_contblock(RTABSIZE * sizeof(struct rtent));
    /* 		for (i = 0;  i < RTABSIZE;  i++) */
    /* 			rtab[i] = from->rt.rt_self[i]; */
  } else
    rtab=to->rt.rt_self;

  /*  structure assignment  */
  for (i = 0;  i < RTABSIZE;  i++)
    rtab[i] = from->rt.rt_self[i];

  for (i = 0;  i < RTABSIZE;  i++)
    if (from->rt.rt_self[i].rte_dtab != NULL) {
      rtab[i].rte_dtab
	= (object *)
	alloc_contblock(RTABSIZE * sizeof(object));
      for (j = 0;  j < RTABSIZE;  j++)
	rtab[i].rte_dtab[j]
	  = from->rt.rt_self[i].rte_dtab[j];
    }
  to->rt.rt_case=from->rt.rt_case;
  vs_reset;
  END_NO_INTERRUPT;}
  return(to);

}

static object
current_readtable()
{
	object r;

	r = symbol_value(Vreadtable);
	if (type_of(r) != t_readtable) {
		Vreadtable->s.s_dbind = copy_readtable(standard_readtable,sLnil);
		FEerror("The value of *READTABLE*, ~S, was not a readtable.",
			1, r);
	}
	return(r);
}



DEFUN("READ",object,fLread,LISP,0,4,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,x,strm,eof_errorp,eof_value,recursivep;
  va_list ap;
  
  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  if (recursivep == Cnil)
    preserving_whitespace_flag = FALSE;
  detect_eos_flag = TRUE;
  if (recursivep == Cnil)
    x = read_object_non_recursive(strm);
  else
    x = read_object_recursive(strm);
  if (x == OBJNULL) {
    if (eof_errorp == Cnil && recursivep == Cnil)
      RETURN1(eof_value);
    end_of_stream(strm);
  }
  
  RETURN1(x);
  
}

DEFUN("READ-PRESERVING-WHITESPACE",object,fLread_preserving_whitespace,LISP,
	  0,4,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,c,x,strm,eof_errorp,eof_value,recursivep;
  va_list ap;

  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  while (!stream_at_end(strm)) {
    c = read_char(strm);
    if (cat(c) != cat_whitespace) {
      unread_char(c, strm);
      goto READ;
    }
  }
  if (eof_errorp == Cnil && recursivep == Cnil)
    RETURN1(eof_value);
  end_of_stream(strm);
  
 READ:
  if (recursivep == Cnil)
    preserving_whitespace_flag = TRUE;
  if (recursivep == Cnil)
    x = read_object_non_recursive(strm);
  else
    x = read_object_recursive(strm);
  RETURN1(x);
}
    
DEFUN("READ-DELIMITED-LIST",object,fLread_delimited_list,LISP,1,3,NONE,OO,OO,OO,OO,(object d,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object l=Cnil,x,f=OBJNULL,strm,recursivep,*p;
  va_list ap;
  bool e;
  volatile object old_READcontext;
  volatile int old_backq_level=0;


  va_start(ap,d);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);

  check_type_character(&d);
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  if (recursivep == Cnil) {

    old_READcontext=sSAsharp_eq_contextA->s.s_dbind;
    old_backq_level = backq_level;
    setup_READ();
    frs_push(FRS_PROTECT, Cnil);
    if (nlj_active) {
      e = TRUE;
      goto L;
    }
  }
  l = Cnil;
  p = &l;
  preserving_whitespace_flag = FALSE;	/*  necessary?  */
  for (;;) {
    delimiting_char = d;
    x = read_object_recursive(strm);
    if (x == OBJNULL)
      break;
    *p = make_cons(x, Cnil);
    p = &((*p)->c.c_cdr);
  }
  if (recursivep == Cnil) {
    if (sSAsharp_eq_contextA->s.s_dbind!=Cnil)
      l = patch_sharp(l);
    e = FALSE;
  L:
    frs_pop();
    sSAsharp_eq_contextA->s.s_dbind=old_READcontext;
    backq_level = old_backq_level;
    if (e) {
      nlj_active = FALSE;
      unwind(nlj_fr, nlj_tag);
    }
  }
  RETURN1(l);
}

DEFUNM("READ-LINE",object,fLread_line,LISP,0,4,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum vals=(fixnum)fcall.valp,nargs=INIT_NARGS(0),i;
  object l=Cnil,c,strm,eof_errorp,eof_value,recursivep,*base=vs_top;
  va_list ap;

  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  if (stream_at_end(strm)) {
    if (eof_errorp == Cnil && recursivep == Cnil)
      RETURN2(eof_value,Ct);
    else
      end_of_stream(strm);
  }
  i = 0;
  for (;;) {
    read_char_to(c,strm,c = Ct; goto FINISH);
    if (char_code(c) == '\n') {
      c = Cnil;
      break;
    }
    if (i >= token->st.st_dim)
      too_long_token();
    token->st.st_self[i++] = char_code(c);
  }
 FINISH:
#ifdef DOES_CRLF
  if (i > 0 && token->st.st_self[i-1] == '\r') i--;
#endif
  token->st.st_fillp = i;
  /* no disadvantage to returning an adjustable string */
  
  RETURN2(copy_simple_string(token),c);

}

DEFUN("READ-CHAR",object,fLread_char,LISP,0,4,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,strm,eof_errorp,eof_value,recursivep;
  va_list ap;

  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  {object x ;
    read_char_to(x,strm,goto AT_EOF);
    RETURN1(x);
  AT_EOF:
    if (eof_errorp == Cnil && recursivep == Cnil)
      RETURN1(eof_value);
    else
      end_of_stream(strm);
    RETURN1(Cnil);
  }
}

DEFUN("UNREAD-CHAR",object,fLunread_char,LISP,1,2,NONE,OO,OO,OO,OO,(object c,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object l=Cnil,f=OBJNULL,strm;
  va_list ap;

  va_start(ap,c);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  va_end(ap);
  
  check_type_character(&c);
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  unread_char(c, strm);
  RETURN1(Cnil);
}

DEFUN("PEEK-CHAR",object,fLpeek_char,LISP,0,5,NONE,OO,OO,OO,OO,(object peek_type,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,c,f=peek_type,strm,eof_errorp,eof_value,recursivep;
  va_list ap;

  va_start(ap,peek_type);
  peek_type=NEXT_ARG(nargs,ap,l,f,Cnil);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  recursivep=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  setup_READtable();
  if (peek_type == Cnil) {
    if (stream_at_end(strm)) {
      if (eof_errorp == Cnil && recursivep == Cnil)
	RETURN1(eof_value);
      else
	end_of_stream(strm);
    }
    c = read_char_no_echo(strm);
    unread_char(c, strm);
    RETURN1(c);
  }
  if (peek_type == Ct) {
    while (!stream_at_end(strm)) {
      c = read_char(strm);
      if (cat(c) != cat_whitespace) {
	unread_char(c, strm);
	RETURN1(c);
      }
    }
    if (eof_errorp == Cnil)
      RETURN1(eof_value);
    else
      end_of_stream(strm);
  }
  check_type_character(&peek_type);
  while (!stream_at_end(strm)) {
    c = read_char(strm);
    if (char_eq(c, peek_type)) {
      unread_char(c, strm);
      RETURN1(c);
    }
  }
  if (eof_errorp == Cnil)
    RETURN1(eof_value);
  else
    end_of_stream(strm);
  RETURN1(Cnil);
}


DEFUN("LISTEN",object,fLlisten,LISP,0,1,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,strm;
  va_list ap;

  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  va_end(ap);
  
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  RETURN1(listen_stream(strm) ? Ct : Cnil);

}

DEFUN("READ-CHAR-NO-HANG",object,fLread_char_no_hang,LISP,0,4,NONE,OO,OO,OO,OO,(object f,...),"") {

  fixnum nargs=INIT_NARGS(0);
  object l=Cnil,strm,eof_errorp,eof_value/* ,recursivep */;
  va_list ap;

  va_start(ap,f);
  strm=NEXT_ARG(nargs,ap,l,f,sLAstandard_inputA->s.s_dbind);
  eof_errorp=NEXT_ARG(nargs,ap,l,f,Ct);
  eof_value= NEXT_ARG(nargs,ap,l,f,Cnil);
  /* FIXME: recursivep=NEXT_ARG(nargs,ap,l,f,Cnil); */
  va_end(ap);

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  if (stream_at_end(strm)) {
    if (eof_errorp == Cnil)
      RETURN1(eof_value);
    else
      end_of_stream(strm);
  }
  RETURN1(listen_stream(strm) ? read_char(strm) : Cnil);
}

@(defun clear_input (&optional (strm `symbol_value(sLAstandard_inputA)`))
@
	if (strm == Cnil)
		strm = symbol_value(sLAstandard_inputA);
	else if (strm == Ct)
		strm = symbol_value(sLAterminal_ioA);
	check_type_stream(&strm);
#ifdef LISTEN_FOR_INPUT
	while(listen_stream(strm)) {readc_stream(strm);}
#endif
	@(return Cnil)
@)

DEFUNM("PARSE-INTEGER-INT",object,fSparse_integer_int,SI,5,5,NONE,OO,OO,IO,OO,
	  (object strng,object start,object end,fixnum radix,object junk_allowed),"") {

  fixnum vals=(fixnum)fcall.valp;
  int s,e,ep;
  object *base=vs_top,x;

  if (junk_allowed==Cnil)
    check_type_string(&strng);
  get_string_start_end(strng, start, end, &s, &e);
  if (radix < 2 || radix > 36)
    FEerror("~S is an illegal radix.", 1, radix);
  setup_READtable();
  while (READtable->rt.rt_self[(unsigned char)strng->st.st_self[s]].rte_chattrib
	 == cat_whitespace && s < e)
    s++;
  if (s >= e) {
    if (junk_allowed != Cnil)
      RETURN2(Cnil,make_fixnum(s));
    else
      goto CANNOT_PARSE;
  }
  {
    char *q;
    while (token->st.st_dim<e-s)
      too_long_token();
    memcpy(token->st.st_self,strng->st.st_self+s,e-s);
    token->st.st_fillp=e-s;
    null_terminate_token();
    x = parse_integer(token->st.st_self, &q, radix);
    ep=q-token->st.st_self;
  }
  if (x == OBJNULL) {
    if (junk_allowed != Cnil)
      RETURN2(Cnil,make_fixnum(ep+s));
    else
      goto CANNOT_PARSE;
  }
  if (junk_allowed != Cnil)
    RETURN2(x,make_fixnum(ep+s));
  for (s += ep ;  s < e;  s++)
    if (READtable->rt.rt_self[(unsigned char)strng->st.st_self[s]]
	.rte_chattrib
	!= cat_whitespace)
      goto CANNOT_PARSE;
  RETURN2(x,make_fixnum(e));
  
 CANNOT_PARSE:
  PARSE_ERROR("Cannot parse integer from string");
  RETURN2(Cnil,make_fixnum(0));
}



/* @(defun read_byte (binary_input_stream */
/* 		   &optional (eof_errorp Ct) eof_value) */
/* 	int c; */
/* @ */
/* 	check_type_stream(&binary_input_stream); */
/* 	if (stream_at_end(binary_input_stream)) { */
/* 		if (eof_errorp == Cnil) */
/* 			@(return eof_value) */
/* 		else */
/* 			end_of_stream(binary_input_stream); */
/* 	} */
/* 	c = readc_stream(binary_input_stream); */
/* 	@(return `make_fixnum(c)`) */
/* @) */

/* object */
/* read_byte1(strm,eof) */
/* object strm,eof; */
/* { */
/*   if (strm == Cnil) */
/*     strm = symbol_value(sLAstandard_inputA); */
/*   else if (strm == Ct) */
/*     strm = symbol_value(sLAterminal_ioA); */
/*   if (stream_at_end(strm)) */
/*     return eof; */
/*   return make_fixnum(readc_stream(strm)); */
/* } */

object
read_char1(strm,eof)
object strm,eof;
{
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_inputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (stream_at_end(strm))
    return eof;
  return code_char(readc_stream(strm));
}

@(defun copy_readtable (&optional (from `current_readtable()`) to)
@
	if (from == Cnil) {
		from = standard_readtable;
		if (to != Cnil)
			check_type_readtable(&to);
		to = copy_readtable(from, to);
		to->rt.rt_self['#'].rte_dtab['!']
		= default_dispatch_macro;
		/*  We must forget #! macro.  */
		@(return to)
	}
	check_type_readtable(&from);
	if (to != Cnil)
		check_type_readtable(&to);
	@(return `copy_readtable(from, to)`)
@)

LFD(Lreadtablep)()
{
	check_arg(1);

	if (type_of(vs_base[0]) == t_readtable)
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

@(defun set_syntax_from_char (tochr fromchr
			      &optional (tordtbl `current_readtable()`)
				 fromrdtbl)
	int i;
@
	check_type_character(&tochr);
	check_type_character(&fromchr);
	check_type_readtable(&tordtbl);
	{BEGIN_NO_INTERRUPT;	
	if (fromrdtbl == Cnil)
		fromrdtbl = standard_readtable;
	else
		check_type_readtable(&fromrdtbl);
	tordtbl->rt.rt_self[char_code(tochr)].rte_chattrib 
	  = fromrdtbl->rt.rt_self[char_code(fromchr)].rte_chattrib;
	tordtbl->rt.rt_self[char_code(tochr)].rte_macro
	= fromrdtbl->rt.rt_self[char_code(fromchr)].rte_macro;
	if ((tordtbl->rt.rt_self[char_code(tochr)].rte_dtab
	     = fromrdtbl->rt.rt_self[char_code(fromchr)].rte_dtab)
	    != NULL) {
		tordtbl->rt.rt_self[char_code(tochr)].rte_dtab
		= (object *)
		  alloc_contblock(RTABSIZE * sizeof(object));
		for (i = 0;  i < RTABSIZE;  i++)
			tordtbl->rt.rt_self[char_code(tochr)]
			.rte_dtab[i]
			= fromrdtbl->rt.rt_self[char_code(fromchr)]
			  .rte_dtab[i];
	}
	END_NO_INTERRUPT;}
	@(return Ct)
@)

@(defun set_macro_character (chr fnc
			     &optional ntp
				       (rdtbl `current_readtable()`))
	int c;
@
	check_type_character(&chr);
	check_type_readtable(&rdtbl);
	c = char_code(chr);
	if (ntp != Cnil)
		rdtbl->rt.rt_self[c].rte_chattrib
		= cat_non_terminating;
	else
		rdtbl->rt.rt_self[c].rte_chattrib
		= cat_terminating;
	rdtbl->rt.rt_self[c].rte_macro = fnc;
        SGC_TOUCH(rdtbl);
        @(return Ct)
@)

@(defun get_macro_character (chr &optional (rdtbl `current_readtable()`))
	object m;
@
	check_type_character(&chr);
	check_type_readtable(&rdtbl);
	if ((m = rdtbl->rt.rt_self[char_code(chr)].rte_macro)
	    == OBJNULL)
		@(return Cnil)
	if (rdtbl->rt.rt_self[char_code(chr)].rte_chattrib
	    == cat_non_terminating)
		@(return m Ct)
	else
		@(return m Cnil)
@)

@(static defun make_dispatch_macro_character (chr
	&optional ntp (rdtbl `current_readtable()`))
	int i;
@
	check_type_character(&chr);
	check_type_readtable(&rdtbl);
	{BEGIN_NO_INTERRUPT;
	if (ntp != Cnil)
		rdtbl->rt.rt_self[char_code(chr)].rte_chattrib
		= cat_non_terminating;
	else
		rdtbl->rt.rt_self[char_code(chr)].rte_chattrib
		= cat_terminating;
	rdtbl->rt.rt_self[char_code(chr)].rte_dtab
	= (object *)
	  alloc_contblock(RTABSIZE * sizeof(object));
	for (i = 0;  i < RTABSIZE;  i++)
		rdtbl->rt.rt_self[char_code(chr)].rte_dtab[i]
		= default_dispatch_macro;
	rdtbl->rt.rt_self[char_code(chr)].rte_macro = dispatch_reader;
	END_NO_INTERRUPT;}
	@(return Ct)
@)

@(static defun set_dispatch_macro_character (dspchr subchr fnc
	&optional (rdtbl `current_readtable()`))
@
	check_type_character(&dspchr);
	check_type_character(&subchr);
	check_type_readtable(&rdtbl);
        SGC_TOUCH(rdtbl);
	if (rdtbl->rt.rt_self[char_code(dspchr)].rte_macro != dispatch_reader
	    || rdtbl->rt.rt_self[char_code(dspchr)].rte_dtab == NULL)
		FEerror("~S is not a dispatch character.", 1, dspchr);
	rdtbl->rt.rt_self[char_code(dspchr)]
	.rte_dtab[char_code(subchr)] = fnc;
	if ('a' <= char_code(subchr) && char_code(subchr) <= 'z')
		rdtbl->rt.rt_self[char_code(dspchr)]
		.rte_dtab[char_code(subchr) - ('a' - 'A')] = fnc;

	@(return Ct)
@)

DEFUN("READTABLE-CASE",object,fLreadtable_case,LISP,1,1,NONE,OO,OO,OO,OO,(object rt),"") {
  check_type_readtable_no_default(&rt);
  RETURN1(rt->rt.rt_case);
}

DEFUN("SET-READTABLE-CASE",object,fSset_readtable_case,SI,2,2,NONE,OO,OO,OO,OO,(object rt,object cas),"") {
  check_type_readtable_no_default(&rt);
  if (cas!=sKupcase && cas!=sKdowncase && cas!=sKpreserve && cas!=sKinvert)
    TYPE_ERROR(cas,list(5,sLmember,sKupcase,sKdowncase,sKpreserve,sKinvert));
  RETURN1(rt->rt.rt_case=cas);
}



@(static defun get_dispatch_macro_character (dspchr subchr
	&optional (rdtbl `current_readtable()`))
@
	check_type_character(&dspchr);
	check_type_character(&subchr);
	check_type_readtable(&rdtbl);
	if (rdtbl->rt.rt_self[char_code(dspchr)].rte_macro != dispatch_reader
	    || rdtbl->rt.rt_self[char_code(dspchr)].rte_dtab == NULL)
		FEerror("~S is not a dispatch character.", 1, dspchr);
	if (digitp(char_code(subchr),10) >= 0) @(return Cnil)
	  else {
	    object x=rdtbl->rt.rt_self[char_code(dspchr)].rte_dtab[char_code(subchr)];
	    @(return `x==default_dispatch_macro ? Cnil : x`)
	      }
@)

static object
string_to_object(x)
object x;
{
	object in;
	vs_mark;

	in = fSmake_string_input_stream_int(x, 0, VLEN(x));
	vs_push(in);
	preserving_whitespace_flag = FALSE;
	detect_eos_flag = FALSE;
	x = read_object_non_recursive(in);
	vs_reset;
	return(x);
}
	
LFD(siLstring_to_object)()
{
	check_arg(1);

	check_type_string(&vs_base[0]);
	vs_base[0] = string_to_object(vs_base[0]);
}


static void
FFN(siLstandard_readtable)()
{
	check_arg(0);

	vs_push(standard_readtable);
}

static void
extra_argument(int c) {
  FEerror("~S is an extra argument for the #~C readmacro.",2, vs_base[2], code_char(c));
}


#define	make_cf(f)	make_cfun((f), Cnil, Cnil, NULL, 0)
#define make_f(f)       find_symbol(make_simple_string(f),system_package)->s.s_gfdef;

DEFVAR("*READ-DEFAULT-FLOAT-FORMAT*",sLAread_default_float_formatA,LISP,sLsingle_float,"");
DEFVAR("*READ-BASE*",sLAread_baseA,LISP,make_fixnum(10),"");
DEFVAR("*READ-SUPPRESS*",sLAread_suppressA,LISP,Cnil,"");


void
gcl_init_read()
{
	struct rtent *rtab;
	object *dtab;
	int i;

	standard_readtable = alloc_object(t_readtable);
	enter_mark_origin(&standard_readtable);

	standard_readtable->rt.rt_self
	= rtab
	= (struct rtent *)
	  alloc_contblock(RTABSIZE * sizeof(struct rtent));
	for (i = 0;  i < RTABSIZE;  i++) {
		rtab[i].rte_chattrib = cat_constituent;
		rtab[i].rte_macro = OBJNULL;
		rtab[i].rte_dtab = NULL;
	}

	dispatch_reader = make_cf(Ldispatch_reader);
	enter_mark_origin(&dispatch_reader);


	rtab['\b'].rte_chatrait = trait_invalid;
	rtab['\t'].rte_chatrait = trait_invalid;
	rtab['\n'].rte_chatrait = trait_invalid;
	rtab['\r'].rte_chatrait = trait_invalid;
	rtab['\f'].rte_chatrait = trait_invalid;
	rtab[' '].rte_chatrait = trait_invalid;
	rtab['\177'].rte_chatrait = trait_invalid;


	rtab['\t'].rte_chattrib = cat_whitespace;
	rtab['\r'].rte_chattrib = cat_whitespace;
	rtab['\n'].rte_chattrib = cat_whitespace;
	rtab['\f'].rte_chattrib = cat_whitespace;
	rtab[' '].rte_chattrib = cat_whitespace;
	rtab['"'].rte_chattrib = cat_terminating;
	rtab['"'].rte_macro = make_f("DOUBLE-QUOTE-READER");
	rtab['#'].rte_chattrib = cat_non_terminating;
	rtab['#'].rte_macro = dispatch_reader;
	rtab['\''].rte_chattrib = cat_terminating;
	rtab['\''].rte_macro = make_f("SINGLE-QUOTE-READER");
	rtab['('].rte_chattrib = cat_terminating;
	rtab['('].rte_macro = make_cf(Lleft_parenthesis_reader);
	rtab[')'].rte_chattrib = cat_terminating;
	rtab[')'].rte_macro = make_f("RIGHT-PARENTHESIS-READER");
/*
	rtab[','].rte_chattrib = cat_terminating;
	rtab[','].rte_macro = make_cf(Lcomma_reader);
*/
	rtab[';'].rte_chattrib = cat_terminating;
	rtab[';'].rte_macro = find_symbol(make_simple_string("SEMICOLON-READER"),system_package)->s.s_gfdef;
	rtab['\\'].rte_chattrib = cat_single_escape;
/*
	rtab['`'].rte_chattrib = cat_terminating;
	rtab['`'].rte_macro = make_cf(Lbackquote_reader);
*/
	rtab['|'].rte_chattrib = cat_multiple_escape;
/*
	rtab['|'].rte_macro = make_cf(Lvertical_bar_reader);
*/

	default_dispatch_macro = make_cf(Ldefault_dispatch_macro);

	rtab['#'].rte_dtab
	= dtab
	= (object *)alloc_contblock(RTABSIZE * sizeof(object));
	for (i = 0;  i < RTABSIZE;  i++)
		dtab[i] = default_dispatch_macro;
	dtab['C'] = dtab['c'] = make_f("SHARP-C-READER");
	dtab['\\'] = make_f("SHARP-\\-READER");
	dtab['\''] = make_cf(Lsharp_single_quote_reader);
	dtab['('] = make_cf(Lsharp_left_parenthesis_reader);
	dtab['*'] = make_cf(Lsharp_asterisk_reader);
	dtab[':'] = make_cf(Lsharp_colon_reader);
	dtab['.'] = make_cf(Lsharp_dot_reader);
	dtab[','] = make_cf(Lsharp_comma_reader);
	dtab['B'] = dtab['b'] = make_cf(Lsharp_B_reader);
	dtab['O'] = dtab['o'] = make_cf(Lsharp_O_reader);
	dtab['X'] = dtab['x'] = make_cf(Lsharp_X_reader);
	dtab['R'] = dtab['r'] = make_cf(Lsharp_R_reader);
/*
	dtab['A'] = dtab['a'] = make_cf(Lsharp_A_reader);
	dtab['S'] = dtab['s'] = make_cf(Lsharp_S_reader);
*/
	dtab['A'] = dtab['a'] = make_si_ordinary("SHARP-A-READER");
	dtab['S'] = dtab['s'] = make_si_ordinary("SHARP-S-READER");

	dtab['='] = make_si_ordinary("SHARP-EQ-READER");
	dtab['#'] = make_si_ordinary("SHARP-SHARP-READER");
	dtab['+'] = make_cf(Lsharp_plus_reader);
	dtab['-'] = make_cf(Lsharp_minus_reader);
/*
	dtab['<'] = make_cf(Lsharp_less_than_reader);
*/
	dtab['|'] = make_cf(Lsharp_vertical_bar_reader);
	dtab['p'] = make_si_ordinary("SHARP-P-READER");
	dtab['P'] = make_si_ordinary("SHARP-P-READER");
	/*  This is specific to this implimentation  */
	dtab['$'] = make_cf(Lsharp_dollar_reader);
	/*  This is specific to this implimentation  */
/*
	dtab[' '] = dtab['\t'] = dtab['\n'] = dtab['\f']
	= make_cf(Lsharp_whitespace_reader);
	dtab[')'] = make_cf(Lsharp_right_parenthesis_reader);
*/

	gcl_init_backq();

	sKupcase = make_keyword("UPCASE");
	sKdowncase = make_keyword("DOWNCASE");
	sKpreserve = make_keyword("PRESERVE");
	sKinvert = make_keyword("INVERT");

	standard_readtable->rt.rt_case=sKupcase;

	Vreadtable
 	= make_special("*READTABLE*",
		       copy_readtable(standard_readtable, Cnil));
	Vreadtable->s.s_dbind->rt.rt_self['#'].rte_dtab['!']
	= default_dispatch_macro;
	/*  We must forget #! macro.  */


	sKstart = make_keyword("START");
	sKend = make_keyword("END");
	sKradix = make_keyword("RADIX");
	sKjunk_allowed = make_keyword("JUNK-ALLOWED");

	READtable = symbol_value(Vreadtable);
	enter_mark_origin(&READtable);
	READdefault_float_format = 'F';
	READbase = 10;
	READsuppress = FALSE;
	READeval = TRUE;

	sSAsharp_eq_contextA->s.s_dbind=Cnil;

	siSsharp_comma = make_si_ordinary("#,");
	enter_mark_origin(&siSsharp_comma);

	delimiting_char = OBJNULL;
	enter_mark_origin(&delimiting_char);

	detect_eos_flag = FALSE;
	in_list_flag = FALSE;
	dot_flag = FALSE;

}

void
gcl_init_read_function()
{
/* 	make_function("READ", Lread); */
/* 	make_function("READ-PRESERVING-WHITESPACE",Lread_preserving_whitespace); */
/* 	make_function("READ-DELIMITED-LIST", Lread_delimited_list); */
/* 	make_function("READ-LINE", Lread_line); */
/* 	make_function("READ-CHAR", Lread_char); */
/* 	make_function("UNREAD-CHAR", Lunread_char); */
/* 	make_function("PEEK-CHAR", Lpeek_char); */
/* 	make_function("LISTEN", Llisten); */
/* 	make_function("READ-CHAR-NO-HANG", Lread_char_no_hang); */
	make_function("CLEAR-INPUT", Lclear_input);

/* 	make_function("PARSE-INTEGER", Lparse_integer); */

/* 	make_function("READ-BYTE", Lread_byte); */

	make_function("COPY-READTABLE", Lcopy_readtable);
	make_function("READTABLEP", Lreadtablep);
	make_function("SET-SYNTAX-FROM-CHAR", Lset_syntax_from_char);
	make_function("SET-MACRO-CHARACTER", Lset_macro_character);
	make_function("GET-MACRO-CHARACTER", Lget_macro_character);
	make_function("MAKE-DISPATCH-MACRO-CHARACTER",Lmake_dispatch_macro_character);
	make_function("SET-DISPATCH-MACRO-CHARACTER",Lset_dispatch_macro_character);
	make_function("GET-DISPATCH-MACRO-CHARACTER",Lget_dispatch_macro_character);

	make_si_function("SHARP-COMMA-READER-FOR-COMPILER",siLsharp_comma_reader_for_compiler);

	make_si_function("STRING-TO-OBJECT",siLstring_to_object);

	make_si_function("STANDARD-READTABLE",siLstandard_readtable);
}

object sSPinit;
