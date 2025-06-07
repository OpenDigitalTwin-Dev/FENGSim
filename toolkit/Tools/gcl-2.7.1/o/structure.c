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
	structure.c

	structure interface
*/

#include "include.h"


#define COERCE_DEF(x) if (type_of(x)==t_symbol) \
  x=getf(x->s.s_plist,sSs_data,Cnil)

#define check_type_structure(x) \
  if(type_of((x))!=t_structure) \
    FEwrong_type_argument(sLstructure,(x)) 


static bool
structure_subtypep(object x, object y)
{ if (x==y) return 1;
  if (type_of(x)!= t_structure
      || type_of(y)!=t_structure)
    FEerror("bad call to structure_subtypep",0);
  {if (S_DATA(y)->included == Cnil) return 0;
   while ((x=S_DATA(x)->includes) != Cnil)
     { if (x==y) return 1;}
   return 0;
 }}

static void
bad_raw_type(void)
{     	  FEerror("Bad raw struct type",0);}


DEFUN("STRUCTURE-DEF",object,fSstructure_def,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  check_type_structure(x);
  return (x)->str.str_def;
}

DEFUN("STRUCTURE-REF",object,structure_ref,SI,3,3,NONE,OO,OI,OO,OO,(object x,object name,fixnum i),"") {

  unsigned short *s_pos;
  COERCE_DEF(name);
  if (type_of(x) != t_structure ||
      (type_of(name)!=t_structure) ||
      !structure_subtypep(x->str.str_def, name))
    FEwrong_type_argument((type_of(name)==t_structure ?
			   S_DATA(name)->name : name),
			  x);
  s_pos = &SLOT_POS(x->str.str_def,0);
  switch((SLOT_TYPE(x->str.str_def,i)))
    {
    case aet_object: return(STREF(object,x,s_pos[i]));
    case aet_nnfix: case aet_fix:  return(make_fixnum((STREF(fixnum,x,s_pos[i]))));
    case aet_ch:  return(code_char(STREF(char,x,s_pos[i])));
    case aet_bit:
    case aet_nnchar: case aet_char: return(small_fixnum(STREF(char,x,s_pos[i])));
    case aet_sf: return(make_shortfloat(STREF(shortfloat,x,s_pos[i])));
    case aet_lf: return(make_longfloat(STREF(longfloat,x,s_pos[i])));
    case aet_uchar: return(small_fixnum(STREF(unsigned char,x,s_pos[i])));
    case aet_ushort: return(make_fixnum(STREF(unsigned short,x,s_pos[i])));
    case aet_nnshort: case aet_short: return(make_fixnum(STREF(short,x,s_pos[i])));
    case aet_uint: return(make_fixnum(STREF(unsigned int,x,s_pos[i])));
    case aet_nnint: case aet_int: return(make_fixnum(STREF(int,x,s_pos[i])));
    default:
      bad_raw_type();
      return 0;
    }
}
#ifdef STATIC_FUNCTION_POINTERS
object
structure_ref(object x,object name,fixnum i) {
  return FFN(structure_ref)(x,name,i);
}
#endif


static void
FFN(siLstructure_ref1)(void)
{object x=vs_base[0];
 int n=fix(vs_base[1]);
 object def;
 check_type_structure(x);
 def=x->str.str_def;
 if(n>= S_DATA(def)->length)
   FEerror("Structure ref out of bounds",0);
 vs_base[0]=structure_ref(x,x->str.str_def,n);
 vs_top=vs_base+1;
}

DEFUN("STRUCTURE-SET",object,structure_set,SI,4,4,NONE,OO,OI,OO,OO,(object x,object name,fixnum i,object v),"") {

  unsigned short *s_pos;

  COERCE_DEF(name);
  if (type_of(x) != t_structure ||
      type_of(name) != t_structure ||
      !structure_subtypep(x->str.str_def, name))
    FEwrong_type_argument((type_of(name)==t_structure ?
			   S_DATA(name)->name : name)
			  , x);

#ifdef SGC
  /* make sure the structure header is on a writable page */
  if (is_marked(x)) FEerror("bad gc field",0); else  unmark(x);
#endif   
 
 s_pos= & SLOT_POS(x->str.str_def,0);
 switch(SLOT_TYPE(x->str.str_def,i)){
   
   case aet_object: STREF(object,x,s_pos[i])=v; break;
   case aet_nnfix:case aet_fix:  (STREF(fixnum,x,s_pos[i]))=fix(v); break;
   case aet_ch:  STREF(char,x,s_pos[i])=char_code(v); break;
   case aet_bit:
   case aet_nnchar:case aet_char: STREF(char,x,s_pos[i])=fix(v); break;
   case aet_sf: STREF(shortfloat,x,s_pos[i])=sf(v); break;
   case aet_lf: STREF(longfloat,x,s_pos[i])=lf(v); break;
   case aet_uchar: STREF(unsigned char,x,s_pos[i])=fix(v); break;
   case aet_ushort: STREF(unsigned short,x,s_pos[i])=fix(v); break;
   case aet_nnshort:case aet_short: STREF(short,x,s_pos[i])=fix(v); break;
   case aet_uint: STREF(unsigned int,x,s_pos[i])=fix(v); break;
   case aet_nnint:case aet_int: STREF(int,x,s_pos[i])=fix(v); break;
 default:
   bad_raw_type();

   }
 return(v);
}
#ifdef STATIC_FUNCTION_POINTERS
object
structure_set(object x,object name,fixnum i,object v) {
  return FFN(structure_set)(x,name,i,v);
}
#endif

DEFUN("STRUCTURE-LENGTH",object,fSstructure_length,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  check_type_structure(x);
  return (object)S_DATA(x)->length;
}

DEFUN("STRUCTURE-SUBTYPE-P",object,fSstructure_subtype_p,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

/* static void */
/* FFN(siLstructure_subtype_p)(void) */
/* {object x,y; */
/*  check_arg(2); */
/*  x=vs_base[0]; */
/*  y=vs_base[1]; */
  if (type_of(x)!=t_structure)
    RETURN1(Cnil);
  x=x->str.str_def;
  COERCE_DEF(y);
  RETURN1(structure_subtypep(x,y) ? Ct : Cnil);

}
 
     

object
structure_to_list(object x)
{

  object *p,s,v;
  struct s_data *def=S_DATA(x->str.str_def);
  int i,n;

  s=def->slot_descriptions;
  for (p=&v,i=0,n=def->length;!endp(s)&&i<n;s=s->c.c_cdr,i++) {
    collect(p,make_cons(intern(car(s->c.c_car),keyword_package),Cnil));
    collect(p,make_cons(structure_ref(x,x->str.str_def,i),Cnil));
  }
  *p=Cnil;

  return make_cons(def->name,v);

}

DEFUN("MAKE-DUMMY-STRUCTURE",object,fSmake_dummy_structure,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  
  object x;

  BEGIN_NO_INTERRUPT;
  x = alloc_object(t_structure);
  x->str.str_def=NULL;
  x->str.str_self=NULL;
  END_NO_INTERRUPT;
  return x;
}

DEFUN("MAKE-STRUCTURE",object,fSmake_structure,SI,1,63,NONE,OO,OO,OO,OO,(object name,...),"") {/*FIXME*/

  fixnum narg=INIT_NARGS(1),i,size;
  object l=Cnil,f=OBJNULL,v,x;
  struct s_data *def=NULL;
  va_list ap;
  unsigned char *s_type;
  unsigned short *s_pos;

  {
    BEGIN_NO_INTERRUPT;
    x = alloc_object(t_structure);
    COERCE_DEF(name);
    if (type_of(name)!=t_structure  ||
	(def=S_DATA(name))->length != narg)
      FEerror("Bad make_structure args for type ~a",1,name);
    x->str.str_def = name;
    x->str.str_self = NULL;
    size=S_DATA(name)->size;
    x->str.str_self=(object *)(def->staticp == Cnil ? alloc_relblock(size) : alloc_contblock(size));
  /* There may be holes in the structure.
     We want them zero, so that equal can work better.
     */
    if (S_DATA(name)->has_holes != Cnil)
      bzero(x->str.str_self,size);

    s_pos= (&SLOT_POS(x->str.str_def,0));
    s_type = (&(SLOT_TYPE(x->str.str_def,0)));

    va_start(ap,name);
    for (i=0;(v=NEXT_ARG(narg,ap,l,f,OBJNULL))!=OBJNULL;i++) {

      switch(s_type[i]) {
	     
      case aet_object: STREF(object,x,s_pos[i])=v; break;
      case aet_nnfix:case aet_fix:  (STREF(fixnum,x,s_pos[i]))=fix(v); break;
      case aet_ch:  STREF(char,x,s_pos[i])=char_code(v); break;
      case aet_bit:
      case aet_nnchar:case aet_char: STREF(char,x,s_pos[i])=fix(v); break;
      case aet_sf: STREF(shortfloat,x,s_pos[i])=sf(v); break;
      case aet_lf: STREF(longfloat,x,s_pos[i])=lf(v); break;
      case aet_uchar: STREF(unsigned char,x,s_pos[i])=fix(v); break;
      case aet_ushort: STREF(unsigned short,x,s_pos[i])=fix(v); break;
      case aet_nnshort:case aet_short: STREF(short,x,s_pos[i])=fix(v); break;
	/*FIXME uint on 32bit really should not be here*/
      case aet_uint: STREF(unsigned int,x,s_pos[i])=fix(v); break;
      case aet_nnint:case aet_int: STREF(int,x,s_pos[i])=fix(v); break;
      default:
	bad_raw_type();

      }
    }

    va_end(ap);
    END_NO_INTERRUPT;

  }

  RETURN1(x);

}

DEFUN("COPY-STRUCTURE",object,fLcopy_structure,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  object y;
  struct s_data *def;
  
  check_type_structure(x);
  {
    BEGIN_NO_INTERRUPT;
    y = alloc_object(t_structure);
    def=S_DATA(y->str.str_def = x->str.str_def);
    y->str.str_self = NULL;
    y->str.str_self = (object *)alloc_relblock(def->size);
    memcpy(y->str.str_self,x->str.str_self,def->size);
    END_NO_INTERRUPT;
  }

  return y;

}

LFD(siLstructure_name)(void)
{
	check_arg(1);
	check_type_structure(vs_base[0]);
	vs_base[0] = S_DATA(vs_base[0]->str.str_def)->name;
}

LFD(siLstructure_ref)(void)
{
	check_arg(3);
	vs_base[0]=structure_ref(vs_base[0],vs_base[1],fix(vs_base[2]));
	vs_top=vs_base+1;
}

LFD(siLstructure_set)(void)
{
	check_arg(4);
	structure_set(vs_base[0],vs_base[1],fix(vs_base[2]),vs_base[3]);
	vs_base = vs_top-1;
}

LFD(siLstructurep)(void)
{
	check_arg(1);
	if (type_of(vs_base[0]) == t_structure && !vs_base[0]->d.tt)
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

/* LFD(siLrplaca_nthcdr)(void) */
/* { */

/* /\* */
/* 	Used in DEFSETF forms generated by DEFSTRUCT. */
/* 	(si:rplaca-nthcdr x i v) is equivalent to  */
/* 	(progn (rplaca (nthcdr i x) v) v). */
/* *\/ */
/* 	int i; */
/* 	object l; */

/* 	check_arg(3); */
/* 	if (type_of(vs_base[1]) != t_fixnum || fix(vs_base[1]) < 0) */
/* 		FEerror("~S is not a non-negative fixnum.", 1, vs_base[1]); */
/* 	if (!consp(vs_base[0])) */
/* 		FEerror("~S is not a cons.", 1, vs_base[0]); */

/* 	for (i = fix(vs_base[1]), l = vs_base[0];  i > 0; --i) { */
/* 		l = l->c.c_cdr; */
/* 		if (endp(l)) */
/* 			FEerror("The offset ~S is too big.", 1, vs_base[1]); */
/* 	} */
/* 	take_care(vs_base[2]); */
/* 	l->c.c_car = vs_base[2]; */
/* 	vs_base = vs_base + 2; */
/* } */

/* LFD(siLlist_nth)(void) */
/* { */

/* /\* */
/* 	Used in structure access functions generated by DEFSTRUCT. */
/* 	si:list-nth is similar to nth except that */
/* 	(si:list-nth i x) is error if the length of the list x is less than i. */
/* *\/ */
/* 	int i; */
/* 	object l; */

/* 	check_arg(2); */
/* 	if (type_of(vs_base[0]) != t_fixnum || fix(vs_base[0]) < 0) */
/* 		FEerror("~S is not a non-negative fixnum.", 1, vs_base[0]); */
/* 	if (!consp(vs_base[1])) */
/* 		FEerror("~S is not a cons.", 1, vs_base[1]); */

/* 	for (i = fix(vs_base[0]), l = vs_base[1];  i > 0; --i) { */
/* 		l = l->c.c_cdr; */
/* 		if (endp(l)) */
/* 			FEerror("The offset ~S is too big.", 1, vs_base[0]); */
/* 	} */

/* 	vs_base[0] = l->c.c_car; */
/* 	vs_popp; */
/* } */


static void
FFN(siLmake_s_data_structure)(void)
{object x,y,raw,*base;
 int i;
 check_arg(5);
 x=vs_base[0];
 base=vs_base;
 raw=vs_base[1];
 y=alloc_object(t_structure);
 y->str.str_def=y;
 y->str.str_self = (object *)(x->v.v_self);
 S_DATA(y)->name  =sSs_data;
 S_DATA(y)->length=(raw->v.v_dim);
 S_DATA(y)->raw   =raw;
 for(i=3; i<raw->v.v_dim; i++)
   y->str.str_self[i]=Cnil;
 S_DATA(y)->slot_position=base[2];
 S_DATA(y)->slot_descriptions=base[3];
 S_DATA(y)->staticp=base[4];
 S_DATA(y)->size = (raw->v.v_dim)*sizeof(object);
 vs_base[0]=y;
 vs_top=vs_base+1;
}

extern aet_type_struct aet_types[];

static void
FFN(siLsize_of)(void)
{ object x= vs_base[0];
  int i;
  i= aet_types[fix(fSget_aelttype(x))].size;
  vs_base[0]=make_fixnum(i);
}
  
static void
FFN(siLaet_type)(void)
{vs_base[0]=fSget_aelttype(vs_base[0]);}


/* Return N such that something of type ARG can be aligned on
   an address which is a multiple of N */


static void
FFN(siLalignment)(void)
{struct {double x; int y; double z;
	 float x1; int y1; float z1;}
 joe;
 joe.z=3.0;
 
 if (vs_base[0]==sLlong_float)
   {vs_base[0]=make_fixnum((long)&joe.z- (long)&joe.y); return;}
 else
   if (vs_base[0]==sLshort_float)
     {vs_base[0]=make_fixnum((long)&(joe.z1)-(long)&(joe.y1)); return;}
   else
     {FFN(siLsize_of)();}
}
   
 
DEF_ORDINARY("S-DATA",sSs_data,SI,"");

void
gcl_init_structure_function(void)
{


/* 	make_si_function("MAKE-STRUCTURE", siLmake_structure); */
	make_si_function("MAKE-S-DATA-STRUCTURE",siLmake_s_data_structure);
/* 	make_si_function("COPY-STRUCTURE", siLcopy_structure); */
	make_si_function("STRUCTURE-NAME", siLstructure_name);
	make_si_function("STRUCTURE-REF1", siLstructure_ref1);
	make_si_function("STRUCTUREP", siLstructurep);
	make_si_function("SIZE-OF", siLsize_of);
	make_si_function("ALIGNMENT",siLalignment);
/* 	make_si_function("STRUCTURE-SUBTYPE-P",siLstructure_subtype_p); */
	/* make_si_function("RPLACA-NTHCDR", siLrplaca_nthcdr); */
	/* make_si_function("LIST-NTH", siLlist_nth); */
	make_si_function("AET-TYPE",siLaet_type);
}
