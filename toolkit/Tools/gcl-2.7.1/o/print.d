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
	print.d
*/

/* hacked by Michael Koehne (c) GNU LGPL
 *           kraehe (at) copyleft.de
 *	     Sun Apr 25 07:43:08 CEST 2004
 *
 * beware of new bugs^h^h^h^h features !
 *
 * many thanks to pfdietz - not only for ircing at #lisp to explain a
 * few bits to me, but even more for writing the ansi-test. This hack
 * would never been possible without his regression test !
 * ------------------------------------------------------------------------- */

#define NEED_ISFINITE

#include "include.h"
#include <unistd.h>
#include "num_include.h"

#define MINIMUM_RIGHT_MARGIN 1
#define DEFAULT_RIGHT_MARGIN 72

#define  PRINTreadably  (sLAprint_readablyA->s.s_dbind != Cnil)
#define  PRINTescape    (PRINTreadably || (sLAprint_escapeA->s.s_dbind != Cnil))
#define  PRINTpretty    (sLAprint_prettyA->s.s_dbind != Cnil)
#define  PRINTcircle    (sLAprint_circleA->s.s_dbind != Cnil)
#define  PRINTarray     (PRINTreadably || (sLAprint_arrayA->s.s_dbind != Cnil))
#define  PRINTgensym    (PRINTreadably || (sLAprint_gensymA->s.s_dbind != Cnil))
#define  PRINTradix     (sLAprint_radixA->s.s_dbind != Cnil)

#define  PRINTpackage   (sSAprint_packageA->s.s_dbind != Cnil)
#define  PRINTstructure (sSAprint_structureA->s.s_dbind != Cnil)

#define  PRINTbase      fixint(sLAprint_baseA->s.s_dbind)
#define  PRINTcase      sLAprint_caseA->s.s_dbind

#define  PRINTlevel     (PRINTreadably || sLAprint_levelA->s.s_dbind==Cnil || type_of(sLAprint_levelA->s.s_dbind)!=t_fixnum ? -1 : fix(sLAprint_levelA->s.s_dbind))
#define  PRINTlength    (PRINTreadably || sLAprint_lengthA->s.s_dbind==Cnil || type_of(sLAprint_lengthA->s.s_dbind)!=t_fixnum ? -1 : fixint(sLAprint_lengthA->s.s_dbind))
#define  PRINTlines     (PRINTreadably || sLAprint_linesA->s.s_dbind==Cnil || type_of(sLAprint_linesA->s.s_dbind)!=t_fixnum ? -1 : fixint(sLAprint_linesA->s.s_dbind))

DEFVAR("*PRINT-CONTEXT*",sSAprint_contextA,SI,make_fixnum(0),"");
DEFVAR("*PRINT-CONTEXT-HEAD*",sSAprint_context_headA,SI,make_fixnum(0),"");

#define	Q_SIZE		256
#define IS_SIZE		256

struct printStruct {
  unsigned short p_queue[Q_SIZE];
  unsigned short p_indent_stack[IS_SIZE];
  int p_qh;
  int p_qt;
  int p_qc;
  int p_isp;
  int p_iisp;
  int p_lb;
  int p_col;
  int p_sn;
  int p_ll;
};

struct printContext {
  struct printContext *next;
  struct printContext *pp;
  object s,h;
  void (*write_ch_fun)(int,void *);
  int (*write_stream_fun)(int,object);
  int ll,ms;
  struct printStruct b;
};

struct printContextshort {
  struct printContext *next;
  struct printContext *pp;
  object s,h;
  void (*write_ch_fun)(int,void *);
  int (*write_stream_fun)(int,object);
};

struct printContext *
lookup_print_context(object strm) {
  struct printContext *p=(void *)fix(sSAprint_context_headA->s.s_dbind);
  for (;p && p->s!=strm;p=p->next);
  return p;
}

  /* object y=output_stream(_y);						\ */
    /* pp->write_stream_fun=writec_stream_fun(y);				\ */
#define SETUP_PRINT_DEFAULT(_x,_y,_z,_s)				\
  bds_ptr old_bds_top=bds_top;						\
  struct printContext *p,*pp=lookup_print_context(_y);			\
  pp=pp ? pp : ZALLOCA(sizeof(struct printContextshort));		\
  if (!pp->s) {								\
    pp->s=_y;								\
    pp->h=((PRINTcircle&&!_s) ? setupPRINTcircle1(_x,1) : Cnil);	\
    pp->write_ch_fun=writec_PRINTstream;				\
    pp->write_stream_fun=writec_stream;					\
    pp->next=(void *)fix(sSAprint_context_headA->s.s_dbind);		\
    bds_bind(sSAprint_context_headA,make_fixnum((fixnum)(void *)pp));	\
  }									\
  p=(!PRINTpretty || _s || pp->write_ch_fun==writec_queue)&&_z ?	\
    pp : ZALLOCA(sizeof(*p));						\
  if (!p->s) {								\
    p->s=_y;								\
    p->pp=pp;								\
    p->h=pp->h;								\
    p->write_ch_fun=writec_queue;					\
    p->b.p_col=file_column(_y);						\
    p->b.p_ll=get_line_length();					\
    p->ms=get_miser_style(p->b.p_col,p->b.p_ll);			\
    p->next=(void *)fix(sSAprint_context_headA->s.s_dbind);		\
    bds_bind(sSAprint_context_headA,make_fixnum((fixnum)(void *)p));	\
  }									\
  bds_bind(sSAprint_contextA,make_fixnum((fixnum)(void *)p))		\

#define CLEANUP_PRINT_DEFAULT()				\
  bds_unwind(old_bds_top);				\
  if (p!=pp)						\
    flush_queue(TRUE,p)					\

void
write_ch_fun(int c) {
  struct printContext *p=(struct printContext *)(void *)fix(sSAprint_contextA->s.s_dbind);
  p->write_ch_fun(c,p);
}

#define DONE 1
#define FOUND -1

#define	write_ch	(*write_ch_fun)

static void
write_decimal1(int);

static void
write_decimal(i)
int i;
{
	if (i == 0) {
		write_ch('0');
		return;
	}
	write_decimal1(i);
}

static int
do_write_sharp_eq(struct cons *e,bool dot) {

  fixnum val=fix(e->c_car);
  bool defined=val&1;

  if (dot) {
    write_str(" . ");
    if (!defined) return FOUND;
  }

  if (!defined) e->c_car=make_fixnum(val|1);
  write_ch('#');
  write_decimal(val>>1);
  write_ch(defined ? '#' : '=');

  return defined ? DONE : FOUND;

}

static int
write_sharp_eq1(object x,bool dot,object h) {

  struct cons *e;

  return h!=Cnil && (e=gethash(x,h))->c_cdr!=OBJNULL ? do_write_sharp_eq(e,dot) : 0;

}

int
write_sharp_eq(object x,bool dot) {
  struct printContext *p=(struct printContext *)(void *)fix(sSAprint_contextA->s.s_dbind);
  return write_sharp_eq1(x,dot,p->h);
}


static void
per_line_prefix_context(struct printContext *p) {
  int i;
  if (stringp(sSAprint_line_prefixA->s.s_dbind))
    for (i=0;i<VLEN(sSAprint_line_prefixA->s.s_dbind);i++)
      p->write_ch_fun(sSAprint_line_prefixA->s.s_dbind->st.st_self[i],p);
}


#define READ_TABLE_CASE (Vreadtable->s.s_dbind->rt.rt_case)

#define	mod(x)		((x)%Q_SIZE)


object sSAprint_packageA;
object sSAprint_structureA;


#define	MARK	          0400
#define	UNMARK		  0401

#define	LINEAR		  0406
#define	MISER		  0407
#define	FILL		  0410
#define	MANDATORY	  0411
#define CURRENT           0412
#define BLOCK             0413
#define LINE              0414
#define SECTION           0415
#define LINE_RELATIVE     0416
#define SECTION_RELATIVE  0417

#define	CONTINUATION 	  0x8000

extern object coerce_stream(object,int);

DEFVAR("*PRINT-LINE-PREFIX*",sSAprint_line_prefixA,SI,Cnil,"");

void
writec_PRINTstream(int c,void *v) {
  struct printContext *p=v;

  p->write_stream_fun(c,p->s);

}

static int dgs,dga;
static fixnum mlen,mlev;

#include "page.h"

static void
travel_push(object x,fixnum lev,fixnum len) {

  int i;

  if (is_imm_fixnum(x))
    return;

  if (lev>=mlev||len>=mlen)
    return;

  if (is_marked(x)) {

    if (imcdr(x) || !x->d.f)
      vs_check_push(x);
    if (!imcdr(x))
      x->d.f=1;

  } else switch (type_of(x)) {

    case t_symbol:

      if (dgs && x->s.s_hpack==Cnil) {
    	mark(x);
      }
      break;

    case t_cons:

      {
	object y=x->c.c_cdr;
	mark(x);
	travel_push(x->c.c_car,lev+1,0);
	travel_push(y,lev,len+1);
      }
      break;

    case t_vector:
    case t_array:
    case t_string:
    case t_bitvector:
    case t_simple_vector:
    case t_simple_array:
    case t_simple_string:
    case t_simple_bitvector:

      mark(x);
      if (dga && (enum aelttype)x->a.a_elttype==aet_object)
	for (i=0;i<x->a.a_dim;i++)
	  travel_push(x->a.a_self[i],lev+1,i);
      break;

    case t_structure:

      mark(x);
      for (i = 0;  i < S_DATA(x->str.str_def)->length;  i++)
	travel_push(structure_ref(x,x->str.str_def,i),lev+1,i);
      break;

    default:

      break;

    }

}


static void
travel_clear(object x) {

  int i;

  if (is_imm_fixnum(x))
    return;

  if (!is_marked(x))
    return;

  unmark(x);
  if (!imcdr(x))
    x->d.f=0;

  switch (type_of(x)) {

  case t_cons:

    travel_clear(x->c.c_car);
    travel_clear(x->c.c_cdr);
    break;

  case t_vector:
  case t_array:
  case t_simple_vector:
  case t_simple_array:

    if (dga && (enum aelttype)x->a.a_elttype == aet_object)
      for (i=0;i<x->a.a_dim;i++)
	travel_clear(x->a.a_self[i]);
    break;

  case t_structure:

    for (i = 0;  i < S_DATA(x->str.str_def)->length;  i++)
      travel_clear(structure_ref(x,x->str.str_def,i));
    break;

  default:

    break;

  }

}

static void
travel(object x,int mdgs,int mdga,fixnum lev,fixnum len) {

  BEGIN_NO_INTERRUPT;
  dgs=mdgs;
  dga=mdga;
  mlev=lev;
  mlen=len;
  travel_push(x,0,0);
  travel_clear(x);
  END_NO_INTERRUPT;

}

object sLeq;

static object
setupPRINTcircle1(object x,int dogensyms) {

  object *vp=vs_top,*v=vp,h;
  fixnum j;

  travel(x,dogensyms,PRINTarray,
	 PRINTlevel>=0 ? PRINTlevel : ARRAY_DIMENSION_LIMIT,
	 PRINTlength>=0 ? PRINTlength : ARRAY_DIMENSION_LIMIT);

  h=vs_top>vp ? gcl_make_hash_table(sLeq) : Cnil;
  for (j=0;v<vs_top;v++)
    if (!imcdr(*v) || gethash(*v,h)->c_cdr==OBJNULL)
      sethash(*v,h,make_fixnum((++j)<<1));

  vs_top=vp;

  return h;

}

static int
get_miser_style(int col,int ll) {
  object o=symbol_value(sLAprint_miser_widthA);
  return o!=Cnil && col>=(ll-fixint(o));
}

static ushort
flush_queue_tab_nspaces(ushort c,ushort num,ushort inc,ushort pos,ushort s) {
  num<<=1;num>>=1;inc<<=1;inc>>=1;
  switch (c) {
  case LINE_RELATIVE:
    num+=pos;
    num+=(inc && num%inc) ? inc-(num%inc) : 0;
    break;
  case SECTION_RELATIVE:
    num+=pos-s;
    num+=(inc && num%inc) ? inc-(num%inc) : 0;
    num+=s;
    break;
  case SECTION:
    num+=s;
  case LINE:
    while (num<=pos)
      if (inc) num+=inc; else break;
    break;
  }

  return num>=pos ? num-pos : 0;

}

static int
flush_queue_flush(int force,int i,struct printContext *p) {

  int j,c;

  for (j=0;j<i;j++) {
    c = p->b.p_queue[p->b.p_qh];
    if (c==' ')
      p->b.p_lb++;
    else if (c==LINE||c==SECTION||c==LINE_RELATIVE||c==SECTION_RELATIVE)
      p->b.p_lb+=flush_queue_tab_nspaces(c,p->b.p_queue[mod(p->b.p_qh+1)],p->b.p_queue[mod(p->b.p_qh+2)],
					 p->b.p_col+p->b.p_lb,p->b.p_indent_stack[p->b.p_isp-1]);
    else if (c<0400) {
      for (;p->b.p_lb;p->b.p_lb--) {
	p->pp->write_ch_fun(' ',p->pp);
	p->b.p_col++;
      }
      p->pp->write_ch_fun(c,p->pp);
      p->b.p_col=c=='\n' ? 0 : (c=='\t' ? (p->b.p_col&-07)+8 : p->b.p_col+1);
      p->b.p_sn=c=='\n' || p->b.p_sn;
      if (!p->b.p_col)
	per_line_prefix_context(p->pp);
    }
    p->b.p_qh = mod(p->b.p_qh+1);
    --p->b.p_qc;
  }

  if (!p->b.p_qc)
    for (;p->b.p_lb;p->b.p_lb--)
      p->pp->write_ch_fun(' ',p->pp);

  return 0;

}

static int
flush_queue_put_indent(int force,struct printContext *p) {

  p->pp->write_ch_fun('\n',p->pp);
  p->b.p_col=0;
  p->b.p_sn=0;
  per_line_prefix_context(p->pp);
  p->b.p_lb=p->b.p_indent_stack[p->b.p_isp];
  p->b.p_iisp = p->b.p_isp;

  p->b.p_qh = mod(p->b.p_qh+1);
  --p->b.p_qc;

  return 0;

}

static int
flush_queue_proc(struct printContext *p,int i,int *l,int *i0,int *j,int *nb) {

  ushort c,s=p->b.p_indent_stack[p->b.p_isp-1];

  switch((c=p->b.p_queue[mod(p->b.p_qh+i)])) {
  case MARK: (*l)++;return 0;
  case UNMARK: if (--(*l) == 0) *i0=i;return (*l==0);
  case FILL: if (*l==1) *i0=i;return 0;
  case MANDATORY: case LINEAR:if (*l==1) *i0=i;return (*l == 1);
  case LINE:case SECTION:case LINE_RELATIVE:case SECTION_RELATIVE:
    (*nb)+=flush_queue_tab_nspaces(c,p->b.p_queue[mod(p->b.p_qh+i+1)],p->b.p_queue[mod(p->b.p_qh+i+2)],*j,s);
    return 0;
  case ' ':(*nb)++;return 0;
  default: if (c < 0400) {(*j)+=1+*nb;*nb=0;} return 0;
  }

}

static int
flush_queue_indent(int force,struct printContext *p) {

  int i,j,k,l,i0,nb;

  if (p->b.p_iisp > p->b.p_isp)
    return flush_queue_put_indent(force,p);

  k = p->b.p_ll-1;
  for (i0=0,j=p->b.p_col,nb=p->b.p_lb,l=1,i=1;i<p->b.p_qc && j<=k;i++)
    if (flush_queue_proc(p,i,&l,&i0,&j,&nb))
      break;

  if (i == p->b.p_qc && !force)
    return 1;

  if (i0 && !p->b.p_sn && p->b.p_queue[mod(p->b.p_qh)]==FILL)
    return flush_queue_flush(force,i0,p);

  return flush_queue_put_indent(force,p);

}

static int
flush_queue_mark(int force,struct printContext *p) {

  int i,j,k,l,c;

  k = p->b.p_ll - 1 - p->b.p_col;

  for (i=l=1,j=p->b.p_lb;l>0 && i<p->b.p_qc && j<k;i++) {
    c=p->b.p_queue[mod(p->b.p_qh + i)];
    if (c=='\n' || c==MANDATORY || c==LINE || c==SECTION || c==LINE_RELATIVE || c==SECTION_RELATIVE) break;
    switch(c) {
    case MARK:l++;break;
    case UNMARK:--l;break;
    default: if (c<0400) j++;break;
    }
  }

  if (l == 0 && c!='\n' && c!=MANDATORY && c!=LINE && c!=SECTION && c!=LINE_RELATIVE && c!=SECTION_RELATIVE)
    return flush_queue_flush(force,i,p);

  if (i == p->b.p_qc && !force)
    return 1;

  if (++p->b.p_isp >= IS_SIZE-1)
    FEerror("Can't pretty-print.", 0);

  p->b.p_indent_stack[p->b.p_isp++] = p->b.p_col+p->b.p_lb;
  p->b.p_indent_stack[p->b.p_isp] = p->b.p_indent_stack[p->b.p_isp-1];

  return flush_queue_flush(force,1,p);

}

static void
flush_queue(int force,struct printContext *p) {

  int c;

  if (!p->b.p_col)
    per_line_prefix_context(p->pp);

  while (p->b.p_qc > 0) {
    switch ((c = p->b.p_queue[p->b.p_qh])) {
    case MARK:
      if (flush_queue_mark(force,p)) return;
      break;
    case UNMARK:
      p->b.p_isp -= 2;
      flush_queue_flush(force,1,p);
      break;
    case FILL: case LINEAR:case MANDATORY:
      if (flush_queue_indent(force,p)) return;
      break;
    case CURRENT: case BLOCK:
      {
	short sh=p->b.p_queue[mod(p->b.p_qh+1)];
	if (p->b.p_qc<2) return;
	sh<<=1;sh>>=1;
	sh+=(c==CURRENT ? p->b.p_col : p->b.p_indent_stack[p->b.p_isp-1]);/*lb*/
	sh=sh<0 ? 0 : sh;
	p->b.p_indent_stack[p->b.p_isp] = sh;
	flush_queue_flush(force,2,p);
	break;
      }
    case LINE:case SECTION:case LINE_RELATIVE:case SECTION_RELATIVE:
      if (p->b.p_qc<3) return;
      flush_queue_flush(force,3,p);
      break;
    default:
      flush_queue_flush(force,1,p);
      break;
    }
  }
  flush_queue_flush(force,0,p);

}

static void
writec_queue(int c,void *v) {
  struct printContext *p=v;
  struct printStruct *b=&p->b;

  if (b->p_qc >= Q_SIZE)
    flush_queue(FALSE,p);
  if (b->p_qc >= Q_SIZE)
    FEerror("Can't pretty-print.", 0);
  b->p_queue[b->p_qt] = c;
  b->p_qt = mod(b->p_qt+1);
  b->p_qc++;
}


void
write_str(s)
char *s;
{
	while (*s != '\0')
		write_ch(*s++);
}

static void
write_decimal1(i)
int i;
{
	if (i == 0)
		return;
	write_decimal1(i/10);
	write_ch(i%10 + '0');
}

static void
write_addr(x)
object x;
{
	long i;
	int j, k;

	i = (long)x;
	for (j = CHAR_SIZE*sizeof(i)-4;  j >= 0;  j -= 4) {
		k = (i>>j) & 0xf;
		if (k < 10)
			write_ch('0' + k);
		else
			write_ch('a' + k - 10);
	}
}

static void
write_base(void)
{
	if (PRINTbase == 2)
		write_str("#b");
	else if (PRINTbase == 8)
		write_str("#o");
	else if (PRINTbase == 16)
		write_str("#x");
	else if (PRINTbase >= 10) {
		write_ch('#');
		write_ch(PRINTbase/10+'0');
		write_ch(PRINTbase%10+'0');
		write_ch('r');
	} else {
		write_ch('#');
		write_ch(PRINTbase+'0');
		write_ch('r');
	}
}

/* The floating point precision required to make the most-positive-long-float
   printed expression readable.   If this is too small, then the rounded
   off fraction, may be too big to read */

#ifndef FPRC 
#define FPRC 16
#endif

object sSAprint_nansA;


static int
char_inc(char *b,char *p) {

  if (b==p) {
    *p='1';
  } else if (*p=='9') {
    *p='0';
    char_inc(b,p-1);
  } else if (*p=='.')
    char_inc(b,p-1);
  else (*p)++;

  return 1;

}

#define COMP(a_,b_,c_,d_)						\
  ({fixnum _r;								\
    BLOCK_EXCEPTIONS(_r=((d_) ? strtod((a_),(b_))==(c_) : strtof((a_),(b_))==(float)(c_))); \
    _r;})

static int
truncate_double(char *b,double d,int dp) {

  char c[FPRC+9],c1[FPRC+9],*p,*pp,*n;
  int j,k;

  n=b;
  k=strlen(n);

  strcpy(c1,b);
  for (p=c1;*p && *p!='e';p++);
  pp=p>c1 && p[-1]!='.' ? p-1 : p;
  for (;pp>c1 && pp[-1]=='0';pp--);
  memmove(pp,p,strlen(p)+1);
  if (pp!=p && COMP(c1,&pp,d,dp))
    k=truncate_double(n=c1,d,dp);

  strcpy(c,n);
  for (p=c;*p && *p!='e';p++);
  if (p>c && p[-1]!='.' && char_inc(c,p-1) && COMP(c,&pp,d,dp)) {
    j=truncate_double(c,d,dp);
    if (j<=k) {
      k=j;
      n=c;
    }
  }

  if (n!=b) strcpy(b,n);
  return k;

}

void
edit_double(int n,double d,int *sp,char *s,int *ep,int dp) {

  char *p, b[FPRC+9];
  int i;
  
  if (!ISFINITE(d)) {
    if (1 /* sSAprint_nansA->s.s_dbind !=Cnil */) {
      sprintf(s, "%e",d);
      *sp=2;
      return;
    }
  } else
    sprintf(b, "%*.*e",FPRC+8,FPRC,d);
  if (b[FPRC+3] != 'e') {
    sprintf(b, "%*.*e",FPRC+7,FPRC,d);
    *ep = (b[FPRC+5]-'0')*10 + (b[FPRC+6]-'0');
  } else
    *ep = (b[FPRC+5]-'0')*100 + (b[FPRC+6]-'0')*10 + (b[FPRC+7]-'0');

  *sp = 1;
  if (b[0] == '-') {
    *sp *= -1;
    b[0]=' ';
  }
  if (b[FPRC+4] == '-')
    *ep *= -1;

  truncate_double(b,d,dp);
  if ((p=strchr(b,'e')))
    *p=0;

  if (n+2<strlen(b) && b[n+2]>='5')
    char_inc(b,b+n+1);

  if (isdigit(b[0])) {
    b[1]=b[0];
    (*ep)++;
  }
  b[2] = b[1];

  for (i=0,p=b+2;i<n && p[i];i++)
      s[i] = p[i];
  for (;i<n;i++)
    s[i] = '0';
  s[n] = '\0';

}

static void
write_unreadable_str(object x,char *str) {
  if (PRINTreadably)
    PRINT_NOT_READABLE(x,"No readable print representation.");
  write_str(str);
}

static void
write_double(d, e, shortp)
double d;
int e;
bool shortp;
{
	int sign;
	char buff[FPRC+5];
	int exp;
	int i;
	int n = FPRC+1;

	if (shortp)
	  n = 10;
	edit_double(n, d, &sign, buff, &exp, !shortp);
	if (sign==2) {write_unreadable_str(make_longfloat(d),"#<");
		      write_str(buff);
		      write_ch('>');
		      return;
		    }
	if (sign < 0)
		write_ch('-');
	if (-3 <= exp && exp < 7) {
		if (exp < 0) {
			write_ch('0');
			write_ch('.');
			exp = (-exp) - 1;
			for (i = 0;  i < exp;  i++)
				write_ch('0');
			for (;  n > 0;  --n)
				if (buff[n-1] != '0' && buff[n-1])
					break;
			if (exp == 0 && n == 0)
				n = 1;
			for (i = 0;  i < n;  i++)
				write_ch(buff[i]);
		} else {
			exp++;
			for (i = 0;  i < exp;  i++)
				if (i < n)
					write_ch(buff[i]);
				else
					write_ch('0');
			write_ch('.');
			if (i < n)
				write_ch(buff[i]);
			else
				write_ch('0');
			i++;
			for (;  n > i;  --n)
				if (buff[n-1] != '0' && buff[n-1])
					break;
			for (;  i < n;  i++)
				write_ch(buff[i]);
		}
		exp = 0;
	} else {
		write_ch(buff[0]);
		write_ch('.');
		write_ch(buff[1]);
		for (;  n > 2;  --n)
			if (buff[n-1] != '0' && buff[n-1])
				break;
		for (i = 2;  i < n;  i++)
			write_ch(buff[i]);
	}
	if (exp == 0 && e == 0)
		return;
	if (e == 0)
		e = 'E';
	write_ch(e);
	if (exp < 0) {
		write_ch('-');
		exp *= -1;
	}
	write_decimal(exp);
}

static void
call_structure_print_function(object x,int level) {
  struct printContext *p=(struct printContext *)(void *)fix(sSAprint_contextA->s.s_dbind);
  ifuncall3(S_DATA(x->str.str_def)->print_function,x,p->s,make_fixnum(level));
}

object copy_big();
object coerce_big_to_string(object,int);
extern object cLtype_of(object);
static bool potential_number_p(object,int);

DEF_ORDINARY("PPRINT-DISPATCH",sLpprint_dispatch,LISP,"");
DEF_ORDINARY("DEFAULT-PPRINT-OBJECT",sSdefault_pprint_object,SI,"");

object
print(object obj,object strm) {
  terpri(strm);
  prin1(obj,strm);
  princ(code_char(' '),strm);
  return(obj);
}

object
terpri(object strm) {
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (type_of(strm) != t_stream)
    FEerror("~S is not a stream.", 1, strm);
  writec_pstream('\n',strm);
  return(Cnil);
}

static int
get_line_length(void) {
  int l=0;
  object o=symbol_value(sLAprint_right_marginA);
  if ((o!=Cnil) && (type_of(o)==t_fixnum))
    l=fix(o);
  if (l<MINIMUM_RIGHT_MARGIN)
    l=DEFAULT_RIGHT_MARGIN;
  return l;
}

int
writec_pstream(int c,object s) {
  SETUP_PRINT_DEFAULT(Cnil,s,1,1);
  write_ch(c);
  CLEANUP_PRINT_DEFAULT();
  return c;
}

void
writestr_pstream(char *s,object strm) {
  SETUP_PRINT_DEFAULT(Cnil,strm,1,1);
  while (*s != '\0')
    write_ch(*s++);
  CLEANUP_PRINT_DEFAULT();
}

void
write_bounded_string_pstream(object str,int b,int e,object strm) {

  SETUP_PRINT_DEFAULT(str,strm,1,1);
  for (;b<e;)
    write_ch(str->st.st_self[b++]);
  CLEANUP_PRINT_DEFAULT();
}

void
write_string_pstream(object str,object strm) {
  write_bounded_string_pstream(str,0,VLEN(str),strm);
}

void
write_string(object strng,object strm) {

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_string(&strng);
  check_type_stream(&strm);
  write_string_pstream(strng,strm);
  flush_stream(strm);

}


void
princ_str(char *s,object sym) {
  sym = symbol_value(sym);
  if (sym == Cnil)
    sym = symbol_value(sLAstandard_outputA);
  else if (sym == Ct)
    sym = symbol_value(sLAterminal_ioA);
  check_type_stream(&sym);
  writestr_pstream(s, sym);
}

void
princ_char(int c,object sym) {
  sym = symbol_value(sym);
  if (sym == Cnil)
    sym = symbol_value(sLAstandard_outputA);
  else if (sym == Ct)
    sym = symbol_value(sLAterminal_ioA);
  check_type_stream(&sym);
  writec_pstream(c,sym);

}

void
pp(object x) {
  princ(x,Cnil);
  flush_stream(symbol_value(sLAstandard_outputA));
}

static int
constant_case(object x) {

  fixnum i,j=0,jj;
  for (i=0;i<VLEN(x);i++,j=jj ? jj : j) {
    jj=isUpper(x->st.st_self[i]) ? 1 : (isLower(x->st.st_self[i]) ? -1 : 0);
    if (j*jj==-1)
      return 0;
  }
  return j;

}
    
static int
needs_escape (object x) {

  fixnum i,all_dots=1;
  unsigned char ch;

  for (i=0;i<VLEN(x);i++)
    switch((ch=x->st.st_self[i])) {
    case ' ':
    case '#':
    case '(':
    case ')':
    case ':':
    case '`':
    case '\'':
    case '"':
    case ';':
    case ',':
    case '\n':
      return 1;
    case '.':
      break;
    default:
      all_dots=0;
      if (Vreadtable->s.s_dbind->rt.rt_self[ch].rte_chattrib!=cat_constituent)
	return 1;
      break;
    }

  if (all_dots)
    return 1;

  if (READ_TABLE_CASE==sKupcase || PRINTreadably) {
    for (i=0;i<VLEN(x);i++)
      if (isLower(x->st.st_self[i]))
	return 1;
  } else if (READ_TABLE_CASE==sKdowncase) {
    for (i=0;i<VLEN(x);i++)
      if (isUpper(x->st.st_self[i]))
	return 1;
  }

  if (potential_number_p(x, PRINTbase))
    return 1;

  return !VLEN(x);

}

#define convertible_upper(c) ((READ_TABLE_CASE==sKupcase||READ_TABLE_CASE==sKinvert)&& isUpper(c))
#define convertible_lower(c) ((READ_TABLE_CASE==sKdowncase||READ_TABLE_CASE==sKinvert)&& isLower(c))

static void
print_symbol_name_body(object x,int pp) {

  int i,j,fc,tc,lw,k,cc;

  cc=constant_case(x);
  k=needs_escape(x);
  k=PRINTescape ? k : 0;
  pp=k&&pp ? 0 : 1;


  if (k)
    write_ch('|');

  for (lw=i=0;i<VLEN(x);i++) {
    j = x->st.st_self[i];
    if (PRINTescape && (j == '|' || j == '\\'))
      write_ch('\\');
    fc=convertible_upper(j) ? 1 : 
        (convertible_lower(j) ? -1 : 0);
    tc=(READ_TABLE_CASE==sKinvert ? -cc :
	 (PRINTcase == sKupcase ? 1 : 
	  (PRINTcase == sKdowncase ? -1 : 
	   (PRINTcase == sKcapitalize ? (i==lw ? 1 : -1) : 0))));
    if (ispunct(j)||isspace(j)) lw=i+1;
    tc*=pp*fc*fc;
    fc=tc*tc*(tc-fc)>>1;
    j+=fc*('A'-'a');
    write_ch(j);
    
  }

  if (k)
    write_ch('|');

}

static int
write_level(void) {

  return type_of(sSAprin_levelA->s.s_dbind)==t_fixnum ? fix(sSAprin_levelA->s.s_dbind) : 0;

}


static void
write_object(object x,int level) {

	object r, y;
	fixnum i, j, k;

	cs_check(x);

	if (x == OBJNULL) {
	        write_unreadable_str(x,"#<OBJNULL>");
		return;
	}
	if (is_free(x)) {
	        write_unreadable_str(x,"#<FREE OBJECT ");
		write_addr(x);
		write_str(">");
		return;
	}

	switch (type_of(x)) {

	case t_fixnum:
	{
		object *vsp;

		if (PRINTradix && PRINTbase != 10)
			write_base();
		i = fix(x);
		if (i == 0) {
			write_ch('0');
			if (PRINTradix && PRINTbase == 10)
				write_ch('.');
			break;
		}
		vsp = vs_top;
		if (i < 0) {
			write_ch('-');
			if (i == MOST_NEGATIVE_FIX) {
			  vs_push(code_char(digit_weight(labs(i%PRINTbase),PRINTbase)));
			  i/=PRINTbase;
			}
			i = -i;
		}
		for (;i;i/=PRINTbase)
		  vs_push(code_char(digit_weight(i%PRINTbase,PRINTbase)));
		while (vs_top > vsp)
		  write_ch(char_code((vs_pop)));
		if (PRINTradix && PRINTbase == 10)
		  write_ch('.');
		break;
	}

	case t_bignum:
	{
		if (PRINTradix && PRINTbase != 10)
			write_base();
		i = big_sign(x);
		if (i == 0) {
			write_ch('0');
			if (PRINTradix && PRINTbase == 10)
				write_ch('.');
			break;
		}
		{ object s = coerce_big_to_string(x,PRINTbase);
                  int i=0;
                  while (i<VLEN(s)) { write_ch(s->ust.ust_self[i++]); }
                 } 
		if (PRINTradix && PRINTbase == 10)
			write_ch('.');
		break;
	}

	case t_ratio:
		if (PRINTradix) {
			write_base();
			bds_bind(sLAprint_radixA,Cnil);
			write_object(x->rat.rat_num, level);
			write_ch('/');
			write_object(x->rat.rat_den, level);
			bds_unwind1;
		} else {
			write_object(x->rat.rat_num, level);
			write_ch('/');
			write_object(x->rat.rat_den, level);
		}
		break;

	case t_shortfloat:
		r = symbol_value(sLAread_default_float_formatA);
		if (r == sLshort_float)
			write_double((double)sf(x), 0, TRUE);
		else
			write_double((double)sf(x), 'S', TRUE);
		break;

	case t_longfloat:
		r = symbol_value(sLAread_default_float_formatA);
		if (r == sLsingle_float ||
		    r == sLlong_float || r == sLdouble_float)
			write_double(lf(x), 0, FALSE);
		else
			write_double(lf(x), 'F', FALSE);
		break;

	case t_complex:
		write_str("#C(");
		write_object(x->cmp.cmp_real, level);
		write_ch(' ');
		write_object(x->cmp.cmp_imag, level);
		write_ch(')');
		break;

	case t_character:
		if (!PRINTescape) {
		  write_ch(char_code(x));
		  break;
		}
		write_str("#\\");
		switch (char_code(x)) {

		case '\r':
			write_str("Return");
			break;

		case ' ':
			write_str(" ");
			/* write_str("Space"); */
			break;

		case '\177':
			write_str("Rubout");
			break;

		case '\f':
			write_str("Page");
			break;

		case '\t':
			write_str("Tab");
			break;

		case '\b':
			write_str("Backspace");
			break;

		case '\n':
			write_str("Newline");
			break;

		default:
			if (char_code(x) & 0200) {
				write_ch('\\');
				i = char_code(x);
				write_ch(((i>>6)&7) + '0');
				write_ch(((i>>3)&7) + '0');
				write_ch(((i>>0)&7) + '0');
			} else if (char_code(x) < 040) {
				write_ch('^');
				write_ch(char_code(x) + 0100);
				if (char_code(x)==28)
				  write_ch(char_code(x) + 0100);
			} else
				write_ch(char_code(x));
			break;
		}
		break;

	case t_symbol:
	  {
	    object y=vs_head;
	    y=y!=OBJNULL && consp(y) && y->c.c_car==sSstructure_list ? y->c.c_cdr: Cnil;
	    for (;consp(y) && y->c.c_car!=x;y=y->c.c_cdr);

	    if (PRINTescape || consp(y)) {
	      if (x->s.s_hpack == Cnil || x->s.s_hpack->p.p_name==Cnil) {
		if (PRINTcircle)
		  if (write_sharp_eq(x,FALSE)==DONE) return;
		if (PRINTgensym)
		  write_str("#:");
	      } else if (x->s.s_hpack == keyword_package) {
		write_ch(':');
	      } else if (PRINTpackage||find_symbol(x,current_package())!=x || !intern_flag) {
		
		print_symbol_name_body(x->s.s_hpack->p.p_name,1);
		
		if (find_symbol(x, x->s.s_hpack) != x)
		  error("can't print symbol");
		if (PRINTpackage || intern_flag == INTERNAL)
		  write_str("::");
		else if (intern_flag == EXTERNAL)
		  write_ch(':');
		else
		  FEerror("Pathological symbol --- cannot print.", 0);
		
	      }

	    }
	    print_symbol_name_body(x->s.s_name,1);
	    break;
	  }
	case t_array:
	case t_simple_array:
	{
		int subscripts[ARRAY_RANK_LIMIT];
		int n, m;

		if (!PRINTarray) {
		        write_unreadable_str(x,"#<array ");
			write_addr(x);
			write_str(">");
			break;
		} else if (x->v.v_elttype!=aet_object)
		  write_unreadable_str(x,"");
		if (PRINTcircle)
		  if (write_sharp_eq(x,FALSE)==DONE) return;
		if (PRINTlevel >= 0 && level >= PRINTlevel) {
			write_ch('#');
			break;
		}
		n = x->a.a_rank;
		write_ch('#');
		write_decimal(n);
		write_ch('A');
		if (PRINTlevel >= 0 && level+n >= PRINTlevel)
			n = PRINTlevel - level;
		for (i = 0;  i < n;  i++)
			subscripts[i] = 0;
		m = 0;
		j = 0;
		for (;;) {
			for (i = j;  i < n;  i++) {
				if (subscripts[i] == 0) {
					if (PRINTpretty) write_ch(MARK);
					write_ch('(');
					if (PRINTpretty) {write_ch(CURRENT);write_ch(CONTINUATION);}
					if (x->a.a_dims[i] == 0) {
						write_ch(')');
						if (PRINTpretty) write_ch(UNMARK);
						j = i-1;
						k = 0;
						if (PRINTreadably)
						  PRINT_NOT_READABLE(x,"Array has a zero dimension.");
						goto INC;
					}
				}
				if (subscripts[i] > 0) {
				  write_ch(' ');
				  if (PRINTpretty) write_ch(FILL);
				}
				if (PRINTlength >= 0 &&
				    subscripts[i] >= PRINTlength) {
					write_str("...)");
					if (PRINTpretty) write_ch(UNMARK);
					k=x->a.a_dims[i]-subscripts[i];
					subscripts[i] = 0;
					for (j = i+1;  j < n;  j++)
						k *= x->a.a_dims[j];
					j = i-1;
					goto INC;
				}
			}
			if (n == x->a.a_rank) {
				vs_push(aref(x, m));
				write_object(vs_head, level+n);
				vs_popp;
			} else
				write_ch('#');
			j = n-1;
			k = 1;

		INC:
			while (j >= 0) {
				if (++subscripts[j] < x->a.a_dims[j])
					break;
				subscripts[j] = 0;
				write_ch(')');
				if (PRINTpretty) write_ch(UNMARK);
				--j;
			}
			if (j < 0)
				break;
			m += k;
		}
		break;
	}

	case t_vector:
	case t_simple_vector:
		if (!PRINTarray) {
		        write_unreadable_str(x,"#<vector ");
			write_addr(x);
			write_str(">");
			break;
		} else if (x->v.v_elttype!=aet_object)
		  write_unreadable_str(x,"");
		if (PRINTcircle)
		  if (write_sharp_eq(x,FALSE)==DONE) return;
		if (PRINTlevel >= 0 && level >= PRINTlevel) {
			write_ch('#');
			break;
		}
		write_ch('#');
		if (PRINTpretty) write_ch(MARK);
		write_ch('(');
		if (PRINTpretty) {write_ch(CURRENT);write_ch(CONTINUATION);}
		if (VLEN(x) > 0) {
			if (PRINTlength == 0) {
				write_str("...)");
				if (PRINTpretty) write_ch(UNMARK);
				break;
			}
			vs_push(aref(x, 0));
			write_object(vs_head, level+1);
			vs_popp;
			for (i = 1;  i < VLEN(x);  i++) {
			  write_ch(' ');
			  if (PRINTpretty) write_ch(FILL);
				if (PRINTlength>=0 && i>=PRINTlength){
					write_str("...");
					break;
				}
				vs_push(aref(x, i));
				write_object(vs_head, level+1);
				vs_popp;
			}
		}
		write_ch(')');
		if (PRINTpretty) write_ch(UNMARK);
		break;

	case t_simple_string:
	case t_string:
	  if (PRINTcircle)
	    if (write_sharp_eq(x,FALSE)==DONE) return;
	  if (!PRINTescape) {
		  for (i = 0;  i < VLEN(x);  i++)
		    write_ch((uchar)x->st.st_self[i]);
			break;
		}
		write_ch('"');
		for (i = 0;  i < VLEN(x);  i++) {
			if (x->st.st_self[i] == '"' ||
			    x->st.st_self[i] == '\\')
				write_ch('\\');
			write_ch((uchar)x->st.st_self[i]);
		}
		write_ch('"');
		break;

	case t_bitvector:
	case t_simple_bitvector:
	  if (PRINTcircle)
	    if (write_sharp_eq(x,FALSE)==DONE) return;
		if (!PRINTarray) {
		        write_unreadable_str(x,"#<bit-vector ");
			write_addr(x);
			write_str(">");
			break;
		}
		write_str("#*");
		for (i = x->bv.bv_offset;  i < VLEN(x) + x->bv.bv_offset;  i++)
		  write_ch(BITREF(x,i) ? '1' : '0');
		break;

	case t_cons:
		if (x->c.c_car == siSsharp_comma) {
			write_str("#.");
			write_object(x->c.c_cdr, level);
			break;
		}
		if (PRINTcircle)
		  if (write_sharp_eq(x,FALSE)==DONE) return;
                if (PRINTpretty) {
		  if (x->c.c_car == sLquote &&
		      consp(x->c.c_cdr) &&
		      x->c.c_cdr->c.c_cdr == Cnil) {
		    write_ch('\'');
		    write_object(x->c.c_cdr->c.c_car, level);
		    break;
		  }
		  if (x->c.c_car == sLfunction &&
		      consp(x->c.c_cdr) &&
		      x->c.c_cdr->c.c_cdr == Cnil) {
		    write_ch('#');
		    write_ch('\'');
		    write_object(x->c.c_cdr->c.c_car, level);
		    break;
		  }
                }
		if (PRINTlevel >= 0 && level >= PRINTlevel) {
			write_ch('#');
			break;
		}
		if (PRINTpretty) write_ch(MARK);
		write_ch('(');
		if (PRINTpretty) {write_ch(CURRENT);write_ch(CONTINUATION);}
		if (PRINTpretty && x->c.c_car != OBJNULL &&
		    type_of(x->c.c_car) == t_symbol &&
		    (r = getf(x->c.c_car->s.s_plist,
		              sSpretty_print_format, Cnil)) != Cnil)
			goto PRETTY_PRINT_FORMAT;
		for (i = 0;  ;  i++) {
			if (PRINTlength >= 0 && i >= PRINTlength) {
				write_str("...");
				break;
			}
			y = x->c.c_car;
			x = x->c.c_cdr;
			write_object(y, level+1);
			if (!x || !consp(x)) {
				if (x != Cnil) {
				  write_ch(' ');
				  if (PRINTpretty) write_ch(FILL);
					write_str(". ");
					write_object(x, level);
				}
				break;
			}
			if (PRINTcircle)
			  switch (write_sharp_eq(x,TRUE)) {
			  case FOUND:
			    write_object(x, level);
			  case DONE:
			    goto RIGHT_PAREN;
			  default:
			    break;
			  }
			write_ch(' ');
			if (PRINTpretty)
			  write_ch(i==0 && y!=OBJNULL && type_of(y)==t_symbol ? LINEAR : FILL);
		}

	RIGHT_PAREN:
		write_ch(')');
		if (PRINTpretty) write_ch(UNMARK);
		break;

	PRETTY_PRINT_FORMAT:
		j = fixint(r);
		for (i = 0;  ;  i++) {
			if (PRINTlength >= 0 && i >= PRINTlength) {
				write_str("...");
				break;
			}
			y = x->c.c_car;
			x = x->c.c_cdr;
			if (i <= j && y == Cnil)
				write_str("()");
			else
				write_object(y, level+1);
			if (!consp(x)) {
				if (x != Cnil) {
				  write_ch(' ');
				  if (PRINTpretty) write_ch(FILL);
					write_str(". ");
					write_object(x, level);
				}
				break;
			}
			write_ch(' ');
			if (PRINTpretty)
			  write_ch(i>=j ? MANDATORY : (!i ? LINEAR : FILL));
		}
		goto RIGHT_PAREN;

	case t_package:
	        write_unreadable_str(x,"#<");
		write_object(x->p.p_name, level);
 		write_str(" package>");
		break;

	case t_hashtable:
	        write_unreadable_str(x,"#<hash-table ");
		write_addr(x);
		write_str(">");
		break;

	case t_stream:
		switch (x->sm.sm_mode) {
		case smm_input:
		        write_unreadable_str(x,"#<input stream ");
			write_object(x->sm.sm_object1, level);
			write_ch('>');
			break;

		case smm_output:
		        write_unreadable_str(x,"#<output stream ");
			write_object(x->sm.sm_object1, level);
			write_ch('>');
			break;

		case smm_io:
		        write_unreadable_str(x,"#<io stream ");
			write_object(x->sm.sm_object1, level);
			write_ch('>');
			break;

		case smm_socket:
		        write_unreadable_str(x,"#<socket stream ");
			write_object(x->sm.sm_object0, level);
			write_ch('>');
			break;


		case smm_probe:
		        write_unreadable_str(x,"#<probe stream ");
			write_object(x->sm.sm_object1, level);
			write_ch('>');
			break;

		case smm_file_synonym:
		case smm_synonym:
		        write_unreadable_str(x,"#<synonym stream to ");
			write_object(x->sm.sm_object0, level);
			write_ch('>');
			break;

		case smm_broadcast:
		        write_unreadable_str(x,"#<broadcast stream ");
			write_addr(x);
			write_str(">");
			break;

		case smm_concatenated:
		        write_unreadable_str(x,"#<concatenated stream ");
			write_addr(x);
			write_str(">");
			break;

		case smm_two_way:
		        write_unreadable_str(x,"#<two-way stream ");
			write_addr(x);
			write_str(">");
			break;

		case smm_echo:
		        write_unreadable_str(x,"#<echo stream ");
			write_addr(x);
			write_str(">");
			break;

		case smm_string_input:
		        write_unreadable_str(x,"#<string-input stream ");
			y = x->sm.sm_object0;
			if (y!=OBJNULL) {
			  write_str(" from \"");
			  j = VLEN(y);
			  for (i = 0;  i < j && i < 16;  i++)
			    write_ch(y->st.st_self[i]);
			  if (j > 16)
			    write_str("...");
			} else
			  write_str("(closed)");
			write_str("\">");
			break;
#ifdef USER_DEFINED_STREAMS
	        case smm_user_defined:
		        write_unreadable_str(x,"#<use-define stream");
			write_addr(x);
			write_str(">");
			break;
#endif

		case smm_string_output:
		        write_unreadable_str(x,"#<string-output stream ");
			write_addr(x);
			write_str(">");
			break;

		default:
			error("illegal stream mode");
		}
		break;

#define FRESH_COPY(a_,b_) {(b_)->_mp_alloc=(a_)->_mp_alloc;\
                           (b_)->_mp_d=gcl_gmp_alloc((b_)->_mp_alloc*sizeof(*(b_)->_mp_d));\
                           (b_)->_mp_size=(a_)->_mp_size;\
                           memcpy((b_)->_mp_d,(a_)->_mp_d,(b_)->_mp_alloc*sizeof(*(b_)->_mp_d));}

	case t_random:
		write_str("#$");
		y = new_bignum();
		FRESH_COPY(x->rnd.rnd_state._mp_seed,MP(y));
		y=normalize_big(y);
		vs_push(y);
		write_object(y, level);
		vs_popp;
		break;

	case t_structure:
	  {
	    object y=structure_to_list(x);
	    if (PRINTcircle)
	      if (write_sharp_eq(x,FALSE)==DONE) return;
	    if (y->c.c_cdr==Cnil) {/*FIXME: Where is this specified?*/
	      write_str("#S(");
	      print_symbol_name_body(y->c.c_car->s.s_name,1);
	      write_ch(')');
	      break;
	    }
	    if (PRINTlevel >= 0 && level >= PRINTlevel) {
	      write_ch('#');
	      break;
	    }
	    if (type_of(x->str.str_def) != t_structure)
	      FEwrong_type_argument(sLstructure, x->str.str_def);
	    if (S_DATA(x->str.str_def)->print_function != Cnil) {
	      call_structure_print_function(x, level);
	      break;
	    }
	    if (PRINTstructure) {
	      write_str("#S");
	      vs_push(MMcons(sSstructure_list,y));/*FIXME alloc etc.*/
	      write_object(y, level);
	      vs_popp;
	      break;
	    }
	    break;
	  }

	case t_readtable:
	        write_unreadable_str(x,"#<readtable ");
		write_addr(x);
		write_str(">");
		break;

	case t_pathname:
	  if (PRINTreadably && x->pn.pn_version!=Cnil && x->pn.tt==0) {
	    write_str("#.(MAKE-PATHNAME ");
	    write_str(" :HOST ");write_object(x->pn.pn_host,level);
	    write_str(" :DEVICE ");write_object(x->pn.pn_device,level);
	    write_str(" :DIRECTORY '");write_object(x->pn.pn_directory,level);
	    write_str(" :NAME ");write_object(x->pn.pn_name,level);
	    write_str(" :TYPE ");write_object(x->pn.pn_type,level);
	    write_str(" :VERSION ");write_object(x->pn.pn_version,level);
	    write_str(")");
	    break;
	  }
	    /* PRINT_NOT_READABLE(x,"Physical pathname has non-nil version."); */
	  if (PRINTescape) {
	    write_ch('#');
	    write_ch('P');
	    vs_push(x->pn.pn_namestring);
	    write_object(vs_head, level);
	    vs_popp;
	  } else {
	    vs_push(x->pn.pn_namestring);
	    write_object(vs_head, level);
	    vs_popp;
	  }
	  break;

	case t_function:
	        write_unreadable_str(x,"#<function ");
		write_addr(x);
		write_str(">");
		break;

	case t_spice:
	        write_unreadable_str(x,"#<\100");
		for (i = CHAR_SIZE*sizeof(long)-4;  i >= 0;  i -= 4) {
			j = ((long)x >> i) & 0xf;
			if (j < 10)
				write_ch('0' + j);
			else
				write_ch('A' + (j - 10));
		}
		write_ch('>');
		break;

	default:
		error("illegal type --- cannot print");
	}
}

void
write_object_pstream(object obj,object strm) {
  object ppfun;

  SETUP_PRINT_DEFAULT(obj,strm,1,0);

  if (PRINTpretty &&
      sLAprint_pprint_dispatchA->s.s_dbind->c.c_cdr!=Cnil &&
      (ppfun=ifuncall1(sLpprint_dispatch,obj))!=Cnil &&
      ppfun!=sSdefault_pprint_object) {
    ifuncall2(ppfun,p->s,obj);
  } else
    write_object(obj, write_level());

  CLEANUP_PRINT_DEFAULT();
}

object
princ(object obj,object strm) {

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (type_of(strm) != t_stream)
    FEerror("~S is not a stream.", 1, strm);
  bds_bind(sLAprint_readablyA,Cnil);
  bds_bind(sLAprint_escapeA,Cnil);
  write_object_pstream(obj,strm);
  bds_unwind1;
  bds_unwind1;

  return(obj);

}

object
prin1(object obj,object strm) {
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (type_of(strm) != t_stream)
    FEerror("~S is not a stream.", 1, strm);

  bds_bind(sLAprint_escapeA,Ct);
  write_object_pstream(obj,strm);
  bds_unwind1;

  flush_stream(strm);
  return(obj);

}

void
travel_find_sharing(object x,object table) {

  object *vp=vs_top;

  travel(x,1,1,ARRAY_DIMENSION_LIMIT,ARRAY_DIMENSION_LIMIT);

  for (;vs_top>vp;vs_top--)
      sethash(vs_head,table,make_fixnum(-2));

}

static bool
potential_number_p(strng, base)
object strng;
int base;
{
	int i, l, c, dc;
	char *s;

	l = VLEN(strng);
	if (l == 0)
		return(FALSE);
	s = strng->st.st_self;
	dc = 0;
	c = s[0];
	if (digitp(c, base) >= 0)
		dc++;
	else if (c != '+' && c != '-' && c != '^' && c != '_')
		return(FALSE);
	if (s[l-1] == '+' || s[l-1] == '-')
		return(FALSE);
	for (i = 1;  i < l;  i++) {
		c = s[i];
		if (digitp(c, base) >= 0) {
			dc++;
			continue;
		}
		if (c != '+' && c != '-' && c != '/' && c != '.' &&
		    c != '^' && c != '_' &&
		    c != 'e' && c != 'E' &&
		    c != 's' && c != 'S' && c != 'l' && c != 'L')
			return(FALSE);
	}
	if (dc == 0)
		return(FALSE);
	return(TRUE);
}

DEFUN("WRITE-CH",object,fSwrite_ch,SI,1,1,NONE,OI,OO,OO,OO,(fixnum x),"") {
  write_ch(x);
  RETURN1(Cnil);
}

DEFUN("WRITE-INT",object,fSwrite_int,SI,2,2,NONE,OO,OO,OO,OO,(object x,object strm),"") {

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (type_of(strm) != t_stream)
    FEerror("~S is not a stream.", 1, strm);
  write_object_pstream(x,strm);
  flush_stream(strm);
  
  RETURN1(x);
  
}
#ifdef STATIC_FUNCTION_POINTERS
object
fSwrite_int(object x,object y) {
  return FFN(fSwrite_int)(x,y);
}
#endif



DEFUN("PPRINT-MISER-STYLE",object,fSpprint_miser_style,SI,1,1,NONE,OO,OO,OO,OO,(object strm),"") {

  struct printContext *p=lookup_print_context(strm);/*output_stream(strm)*/

  RETURN1(p && p->ms ? Ct : Cnil);

}

static void
queue_continuation(fixnum x,struct printContext *p) {

  short sh=x;
  sh<<=1;sh>>=1;sh|=CONTINUATION;
  p->write_ch_fun(sh,p);

}

static void
queue_codes(object strm,fixnum code,fixnum n,fixnum c1,fixnum c2) {
  struct printContext *p=lookup_print_context(strm);/*output_stream(strm)*/

  if (p && p->write_ch_fun==writec_queue) {

    p->write_ch_fun(code,p);
    if (n--) {
      queue_continuation(c1,p);
      if (n--)
	queue_continuation(c2,p);
    }
  }
}

void
write_codes_pstream(object strm,fixnum code,fixnum nc,fixnum c1,fixnum c2) {

  SETUP_PRINT_DEFAULT(Cnil,strm,1,0);
  if (PRINTpretty)
    queue_codes(strm,code,nc,c1,c2);
  CLEANUP_PRINT_DEFAULT();

}



DEFUN("PPRINT-QUEUE-CODES",object,fSpprint_queue_codes,SI,2,4,NONE,OO,IO,OO,OO,(object strm,fixnum off,...),"") {

  fixnum n=INIT_NARGS(2),nc=0,c1=0,c2=0;
  object l=Cnil,f=OBJNULL,x;
  va_list ap;

  va_start(ap,off);
  if ((x=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL) {
    c1=fixint(x);
    nc++;
  }
  if ((x=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL) {
    c2=fixint(x);
    nc++;
  }
  va_end(ap);

  queue_codes(strm,off,nc,c1,c2);

  RETURN1(Cnil);

}

DEFUN("WRITE-INT1",object,fSwrite_int1,SI,5,5,NONE,OO,OO,OO,OO,
      (object x,object strm,object fun,object pref,object suf),"") {

  object s;

  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  if (type_of(strm) != t_stream)
    FEerror("~S is not a stream.", 1, strm);

  {
    SETUP_PRINT_DEFAULT(x,strm,0,0);

    if ((s=ifuncall2(pref,x,p->h))!=Cnil) {

      p->ms=get_miser_style(file_column(strm)+VLEN(s),p->b.p_ll);

      write_ch(MARK);

      ifuncall2(fun,x,p->h);

      write_ch(UNMARK);

      ifuncall2(suf,x,p->h);
    }

    CLEANUP_PRINT_DEFAULT();

  }

  RETURN1(Cnil);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSwrite_int1(object a,object b,object c,object d,object e) {
  return FFN(fSwrite_int1)(a,b,c,d,e);
}
#endif


@(defun write (x
	       &key ((:stream strm) Cnil)
	       (escape `symbol_value(sLAprint_escapeA)`)
	       (readably `symbol_value(sLAprint_readablyA)`)
	       (radix `symbol_value(sLAprint_radixA)`)
	       (base `symbol_value(sLAprint_baseA)`)
	       (circle `symbol_value(sLAprint_circleA)`)
	       (pretty `symbol_value(sLAprint_prettyA)`)
	       (level `symbol_value(sLAprint_levelA)`)
	       (length `symbol_value(sLAprint_lengthA)`)
	       ((:case cas) `symbol_value(sLAprint_caseA)`)
	       (gensym `symbol_value(sLAprint_gensymA)`)
	       (array `symbol_value(sLAprint_arrayA)`)
	       (pprint_dispatch `symbol_value(sLAprint_pprint_dispatchA)`)
	       (lines `symbol_value(sLAprint_linesA)`)
	       (right_margin `symbol_value(sLAprint_right_marginA)`)
	       (miser_width `symbol_value(sLAprint_miser_widthA)`))

  @
  x=FFN(fSwrite_int)(x,strm);
  
  @(return x)
  
@)



@(defun prin1 (obj &optional strm)
@
	prin1(obj, strm);
	@(return obj)
@)

@(defun print (obj &optional strm)
@
	print(obj, strm);
	@(return obj)
@)

@(defun pprint (obj &optional strm)
@
  if (strm == Cnil)
  strm = symbol_value(sLAstandard_outputA);
 else if (strm == Ct)
   strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  terpri(strm);
  bds_bind(sLAprint_prettyA,Ct);
  prin1(obj,strm);
  bds_unwind1;
  @(return)
@)

@(defun default_pprint_object (strm obj)
@
  if (strm == Cnil)
  strm = symbol_value(sLAstandard_outputA);
 else if (strm == Ct)
   strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  bds_bind(sLAprint_prettyA,Cnil);
  write_object_pstream(obj,strm);
  bds_unwind1;
  @(return)
@)

@(defun princ (obj &optional strm)
@
	princ(obj, strm);
	@(return obj)
@)

@(defun write_char (c &optional strm)
@
  if (strm == Cnil)
  strm = symbol_value(sLAstandard_outputA);
 else if (strm == Ct)
   strm = symbol_value(sLAterminal_ioA);
  check_type_character(&c);
  check_type_stream(&strm);
  writec_pstream(char_code(c),strm);
  @(return c)
@)

@(defun write_string (strng &o strm &k start end)
	int s, e;
@
  check_type_string(&strng);
  get_string_start_end(strng, start, end, &s, &e);
  if (strm == Cnil)
    strm = symbol_value(sLAstandard_outputA);
  else if (strm == Ct)
    strm = symbol_value(sLAterminal_ioA);
  check_type_stream(&strm);
  write_bounded_string_pstream(strng,s,e,strm);
  flush_stream(strm);
  @(return strng)
@)

@(defun write_line (strng &o strm &k start end)
	int s, e;
@
  if (strm == Cnil)
  strm = symbol_value(sLAstandard_outputA);
 else if (strm == Ct)
   strm = symbol_value(sLAterminal_ioA);
  check_type_string(&strng);
  check_type_stream(&strm);
  get_string_start_end(strng, start, end, &s, &e);
  write_bounded_string_pstream(strng,s,e,strm);
  writec_pstream('\n',strm);
  flush_stream(strm);
  @(return strng)
@)

@(defun terpri (&optional strm)
@
	terpri(strm);
	@(return Cnil)
@)

@(defun fresh_line (&optional strm)
@
	if (strm == Cnil)
		strm = symbol_value(sLAstandard_outputA);
	else if (strm == Ct)
		strm = symbol_value(sLAterminal_ioA);
        /* we need to get the real output stream, if possible */
        {object tmp=coerce_stream(strm,1);
           if(tmp != Cnil) strm = tmp ;
         else 
          check_type_stream(&strm);
         }
	if (file_column(strm) == 0)
		@(return Cnil)
        if (strm->sm.sm_mode==smm_broadcast && strm->sm.sm_object0==Cnil)
           @(return Cnil)
	     writec_pstream('\n',strm);
	flush_stream(strm);
	@(return Ct)
@)

@(defun finish_output (&o strm)
@
	if (strm == Cnil)
		strm = symbol_value(sLAstandard_outputA);
	else if (strm == Ct)
		strm = symbol_value(sLAterminal_ioA);
	check_type_stream(&strm);
	flush_stream(strm);
	@(return Cnil)
@)

@(defun force_output (&o strm)
@
	if (strm == Cnil)
		strm = symbol_value(sLAstandard_outputA);
	else if (strm == Ct)
		strm = symbol_value(sLAterminal_ioA);
	check_type_stream(&strm);
	flush_stream(strm);
	@(return Cnil)
@)

@(defun clear_output (&o strm)
@
	if (strm == Cnil)
		strm = symbol_value(sLAstandard_outputA);
	else if (strm == Ct)
		strm = symbol_value(sLAterminal_ioA);
	check_type_stream(&strm);
	@(return Cnil)
@)

DEF_ORDINARY("STRUCTURE-LIST",sSstructure_list,SI,"");
DEF_ORDINARY("PPRINT-QUIT",sSpprint_quit,SI,"");
DEF_ORDINARY("PPRINT-INSERT-CONDITIONAL-NEWLINES",sSpprint_insert_conditional_newlines,SI,"");
DEF_ORDINARY("FORMAT-LOGICAL-BLOCK-PREFIX",sSformat_logical_block_prefix,SI,"");
DEF_ORDINARY("FORMAT-LOGICAL-BLOCK-BODY",sSformat_logical_block_body,SI,"");
DEF_ORDINARY("FORMAT-LOGICAL-BLOCK-SUFFIX",sSformat_logical_block_suffix,SI,"");


DEF_ORDINARY("OBJECT",sKobject,KEYWORD,"");
DEF_ORDINARY("LINEAR",sKlinear,KEYWORD,"");
DEF_ORDINARY("MISER",sKmiser,KEYWORD,"");
DEF_ORDINARY("FILL",sKfill,KEYWORD,"");
DEF_ORDINARY("MANDATORY",sKmandatory,KEYWORD,"");
DEF_ORDINARY("CURRENT",sKcurrent,KEYWORD,"");
DEF_ORDINARY("BLOCK",sKblock,KEYWORD,"");
DEF_ORDINARY("LINE",sKline,KEYWORD,"");
DEF_ORDINARY("SECTION",sKsection,KEYWORD,"");
DEF_ORDINARY("LINE-RELATIVE",sKline_relative,KEYWORD,"");
DEF_ORDINARY("SECTION-RELATIVE",sKsection_relative,KEYWORD,"");
DEF_ORDINARY("UPCASE",sKupcase,KEYWORD,"");
DEF_ORDINARY("DOWNCASE",sKdowncase,KEYWORD,"");
DEF_ORDINARY("CAPITALIZE",sKcapitalize,KEYWORD,"");
DEF_ORDINARY("STREAM",sKstream,KEYWORD,"");
DEF_ORDINARY("ESCAPE",sKescape,KEYWORD,"");
DEF_ORDINARY("READABLY",sKreadably,KEYWORD,"");
DEF_ORDINARY("PRETTY",sKpretty,KEYWORD,"");
DEF_ORDINARY("CIRCLE",sKcircle,KEYWORD,"");
DEF_ORDINARY("BASE",sKbase,KEYWORD,"");
DEF_ORDINARY("RADIX",sKradix,KEYWORD,"");
DEF_ORDINARY("CASE",sKcase,KEYWORD,"");
DEF_ORDINARY("GENSYM",sKgensym,KEYWORD,"");
DEF_ORDINARY("LEVEL",sKlevel,KEYWORD,"");
DEF_ORDINARY("LENGTH",sKlength,KEYWORD,"");
DEF_ORDINARY("PPRINT-DISPATCH",sKpprint_dispatch,KEYWORD,"");
DEF_ORDINARY("ARRAY",sKarray,KEYWORD,"");
DEF_ORDINARY("LINES",sKlines,KEYWORD,"");
DEF_ORDINARY("RIGHT-MARGIN",sKright_margin,KEYWORD,"");
DEF_ORDINARY("MISER-WIDTH",sKmiser_width,KEYWORD,"");
DEF_ORDINARY("LINEAR",sKlinear,KEYWORD,"");
DEF_ORDINARY("MISER",sKmiser,KEYWORD,"");
DEF_ORDINARY("FILL",sKfill,KEYWORD,"");
DEF_ORDINARY("MANDATORY",sKmandatory,KEYWORD,"");
DEFVAR("*PRIN-LEVEL*",sSAprin_levelA,SI,make_fixnum(0),"");
DEFVAR("*PRINT-ESCAPE*",sLAprint_escapeA,LISP,Ct,"");
DEFVAR("*PRINT-READABLY*",sLAprint_readablyA,LISP,Cnil,"");
DEFVAR("*PRINT-PRETTY*",sLAprint_prettyA,LISP,Ct,"");
DEFVAR("*PRINT-CIRCLE*",sLAprint_circleA,LISP,Cnil,"");
DEFVAR("*PRINT-BASE*",sLAprint_baseA,LISP,make_fixnum(10),"");
DEFVAR("*PRINT-RADIX*",sLAprint_radixA,LISP,Cnil,"");
DEFVAR("*PRINT-CASE*",sLAprint_caseA,LISP,sKupcase,"");
DEFVAR("*PRINT-GENSYM*",sLAprint_gensymA,LISP,Ct,"");
DEFVAR("*PRINT-LEVEL*",sLAprint_levelA,LISP,Cnil,"");
DEFVAR("*PRINT-LENGTH*",sLAprint_lengthA,LISP,Cnil,"");
DEFVAR("*PRINT-ARRAY*",sLAprint_arrayA,LISP,Ct,"");
DEFVAR("*PRINT-PACKAGE*",sSAprint_packageA,SI,Cnil,"");
DEFVAR("*PRINT-STRUCTURE*",sSAprint_structureA,SI,Ct,"");
DEF_ORDINARY("PRETTY-PRINT-FORMAT",sSpretty_print_format,SI,"");
DEFVAR("*PRINT-NANS*",sSAprint_nansA,SI,Ct,"");


DEFVAR("*PRINT-PPRINT-DISPATCH*",sLAprint_pprint_dispatchA,LISP,MMcons(Cnil,Cnil),"");
DEFVAR("*PRINT-LINES*",sLAprint_linesA,LISP,Cnil,"");
DEFVAR("*PRINT-MISER-WIDTH*",sLAprint_miser_widthA,LISP,Cnil,"");
DEFVAR("*PRINT-RIGHT-MARGIN*",sLAprint_right_marginA,LISP,Cnil,"");
DEFVAR("*READ-EVAL*",sLAread_evalA,LISP,Ct,"");

void
gcl_init_print(void) {

}



LFD(Lset_line_length)(void)
{
  check_arg(1);

  if ((vs_base[0] == Cnil) || (type_of(vs_base[0]) == t_fixnum))
      sLAprint_right_marginA->s.s_dbind = vs_base[0];
}

DEFVAR("*PRINT-NANS*",sSAprint_nansA,SI,Cnil,"");

void
gcl_init_print_function()
{
	make_function("WRITE", Lwrite);
	make_function("PRIN1", Lprin1);
	make_function("PRINT", Lprint);
	make_function("PPRINT", Lpprint);
	make_function("PRINC", Lprinc);

	make_function("WRITE-CHAR", Lwrite_char);
	make_function("WRITE-STRING", Lwrite_string);
	make_function("WRITE-LINE", Lwrite_line);
	make_function("TERPRI", Lterpri);
	make_function("FRESH-LINE", Lfresh_line);
	make_function("FINISH-OUTPUT", Lfinish_output);
	make_function("FORCE-OUTPUT", Lforce_output);
	make_function("CLEAR-OUTPUT", Lclear_output);
	make_si_function("DEFAULT-PPRINT-OBJECT", Ldefault_pprint_object);

	/* KCL compatibility function */
        make_si_function("SET-LINE-LENGTH",Lset_line_length);

	sKlinear->s.s_plist=putf(sKlinear->s.s_plist,make_fixnum(LINEAR),sLfixnum);
	sKmiser->s.s_plist=putf(sKmiser->s.s_plist,make_fixnum(MISER),sLfixnum);
	sKfill->s.s_plist=putf(sKfill->s.s_plist,make_fixnum(FILL),sLfixnum);
	sKmandatory->s.s_plist=putf(sKmandatory->s.s_plist,make_fixnum(MANDATORY),sLfixnum);

	sKcurrent->s.s_plist=putf(sKcurrent->s.s_plist,make_fixnum(CURRENT),sLfixnum);
	sKblock->s.s_plist=putf(sKblock->s.s_plist,make_fixnum(BLOCK),sLfixnum);

	sKline->s.s_plist=putf(sKline->s.s_plist,make_fixnum(LINE),sLfixnum);
	sKsection->s.s_plist=putf(sKsection->s.s_plist,make_fixnum(SECTION),sLfixnum);
	sKline_relative->s.s_plist=putf(sKline_relative->s.s_plist,make_fixnum(LINE_RELATIVE),sLfixnum);
	sKsection_relative->s.s_plist=putf(sKsection_relative->s.s_plist,make_fixnum(SECTION_RELATIVE),sLfixnum);

}
