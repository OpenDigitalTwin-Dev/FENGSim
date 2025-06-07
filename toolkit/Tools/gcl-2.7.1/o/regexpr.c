/*
 Copyright (C) 1994 W. Schelter
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

#include "include.h"
#include "page.h"

#undef STATIC
#define regerror gcl_regerror
static void
gcl_regerror(char *s)
{ 
  FEerror("Regexp Error: ~a",1,make_simple_string(s));
}
#undef endp
#include "regexp.c"
#define check_string(x) \
	if (!stringp(x)) \
		not_a_string(x)


DEFVAR("*COMPILED-REGEXP-CACHE*",sSAcompiled_regexp_cacheA,SI,MMcons(MMcons(sLnil,sLnil),sLnil),"");
DEFVAR("*MATCH-DATA*",sSAmatch_dataA,SI,sLnil,"");
DEFVAR("*CASE-FOLD-SEARCH*",sSAcase_fold_searchA,SI,sLnil,
       "Non nil means that a string-match should ignore case");

DEFUN("MATCH-BEGINNING",object,fSmatch_beginning,SI,1,1,NONE,OI,OO,OO,OO,(fixnum i),
   "Returns the beginning of the I'th match from the previous STRING-MATCH, \
where the 0th is for the whole regexp and the subsequent ones match parenthetical expressions.  -1 is returned if there is no match, or if the *match-data* \
vector is not a fixnum array.")
{ object v = sSAmatch_dataA->s.s_dbind;
  if (type_of(v)==t_vector
      && (v->v.v_elttype == aet_fix))
    RETURN1(make_fixnum(((fixnum *)sSAmatch_dataA->s.s_dbind->a.a_self)[i]));
  RETURN1(make_fixnum(-1));
}

DEFUN("MATCH-END",object,fSmatch_end,SI,1,1,NONE,OI,OO,OO,OO,(fixnum i),
   "Returns the end of the I'th match from the previous STRING-MATCH")
{ object v = sSAmatch_dataA->s.s_dbind;
  if (type_of(v)==t_vector
      && (v->v.v_elttype == aet_fix))
    RETURN1(make_fixnum(((fixnum *)sSAmatch_dataA->s.s_dbind->a.a_self)[i+NSUBEXP]));
  RETURN1(make_fixnum(-1));
}

DEFUN("COMPILE-REGEXP",object,fScompile_regexp,SI,1,1,NONE,OO,OO,OO,OO,(object p),
	  "Provide handle to export pre-compiled regexp's to string-match") {

  char *tmp;
  object res;
  void *v;
  ufixnum sz=0;

  p=coerce_to_string(p);
  if (!(tmp=alloca(VLEN(p)+1)))
    FEerror("out of C stack",0);
  memcpy(tmp,p->st.st_self,VLEN(p));
  tmp[VLEN(p)]=0;

  if (!(v=(void *)regcomp(tmp,&sz)))
    FEerror("regcomp failure",0);

  res=alloc_object(t_vector);
  res->v.v_adjustable=1;
  res->v.v_hasfillp=1;
  SET_ADISP(res,Cnil);
  set_array_elttype(res,aet_uchar);
  res->v.v_rank=1;
  res->v.v_self=v;
  res->v.v_dim=sz;
  VSET_MAX_FILLP(res);

  RETURN1(res);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fScompile_regexp(object x) {
  return FFN(fScompile_regexp)(x);
}
#endif

DEFUN("STRING-MATCH",object,fSstring_match,SI,2,4,NONE,IO,OO,OO,OO,
	  (object pattern,object string,...),
      "Match regexp PATTERN in STRING starting in string starting at START \
and ending at END.  Return -1 if match not found, otherwise \
return the start index  of the first matchs.  The variable \
*MATCH-DATA* will be set to a fixnum array of sufficient size to hold \
the matches, to be obtained with match-beginning and match-end. \
If it already contains such an array, then the contents of it will \
be over written.   \
") {  

  fixnum nargs=INIT_NARGS(2);
  int i,ans;
  int len,start,end;
  va_list ap;
  object v=sSAmatch_dataA->s.s_dbind,l=Cnil,f=OBJNULL;
  char **pp,*str,save_c=0;

  if (!stringp(pattern) && type_of(pattern)!=t_symbol &&
      (type_of(pattern)!=t_vector || pattern->v.v_elttype!=aet_uchar))
    FEerror("~S is not a regexp pattern", 1 , pattern);
  if (!stringp(string) && type_of(string)!=t_symbol)
    not_a_string_or_symbol(string);
  
  if (type_of(v) != t_vector || v->v.v_elttype != aet_fix || v->v.v_dim < NSUBEXP*2)
    /* v=sSAmatch_dataA->s.s_dbind=fSmake_vector1_1((NSUBEXP *2),aet_fix,sLnil); */
    v=sSAmatch_dataA->s.s_dbind=fSmake_vector(sLfixnum,(NSUBEXP *2),Ct,Cnil,Cnil,0,Cnil,Cnil);
  
  va_start(ap,string);
  start=fixint(NEXT_ARG(nargs,ap,l,f,make_fixnum(0)));
  end=fixint(NEXT_ARG(nargs,ap,l,f,make_fixnum(VLEN(string))));
  va_end(ap);
  if (start < 0 || end > VLEN(string) || start > end)
     FEerror("Bad start or end",0);

  len=VLEN(pattern);
  if (len==0) {
    /* trivial case of empty pattern */
    for (i=0;i<NSUBEXP;i++)
      ((fixnum *)v->a.a_self)[i]=i ? -1 : 0;
    memcpy(((fixnum *)v->a.a_self)+NSUBEXP,((fixnum *)v->a.a_self),NSUBEXP*sizeof(*((fixnum *)v->a.a_self)));
    RETURN1(0);
  }

  {

    regexp *compiled_regexp;

    BEGIN_NO_INTERRUPT;

    if (type_of(pattern)==t_vector)

      compiled_regexp=(void *)pattern->ust.ust_self;

    else {

      object cache=sSAcompiled_regexp_cacheA->s.s_dbind;

      if (cache->c.c_car->c.c_car!=pattern || cache->c.c_car->c.c_cdr!=sSAcase_fold_searchA->s.s_dbind) {
	cache->c.c_car->c.c_car=pattern;
	cache->c.c_car->c.c_cdr=sSAcase_fold_searchA->s.s_dbind;
	cache->c.c_cdr=FFN(fScompile_regexp)(pattern);
      }

      compiled_regexp=(regexp *)cache->c.c_cdr->v.v_self;

    }

    str=string->st.st_self;
    if (NULL_OR_ON_C_STACK(str+end) || str+end==(void *)compiled_regexp) {

      if (!(str=alloca(VLEN(string)+1)))
	FEerror("Cannot allocate memory on C stack",0);
      memcpy(str,string->st.st_self,VLEN(string));

    } else
      save_c=str[end];
    str[end]=0;

    ans = regexec(compiled_regexp,str+start,str,end-start);

    str[end] = save_c;

    if (!ans ) {
      END_NO_INTERRUPT;
      RETURN1((object)-1);
    }

    pp=compiled_regexp->startp;
    for (i=0;i<NSUBEXP;i++,pp++)
      ((fixnum *)v->a.a_self)[i]=*pp ? *pp-str : -1;
    pp=compiled_regexp->endp;
    for (;i<2*NSUBEXP;i++,pp++)
      ((fixnum *)v->a.a_self)[i]=*pp ? *pp-str : -1;

    END_NO_INTERRUPT;
    RETURN1((object)((fixnum *)v->a.a_self)[0]);

  }

}
object
fSstring_match2(object x,object y) {
  return FFN(fSstring_match)(x,y);
}
