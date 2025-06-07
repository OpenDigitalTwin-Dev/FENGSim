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
	package.d
*/

#include "include.h"

#define	HASHCOEF	12345		/*  hashing coefficient  */

void check_type_or_symbol_string_package(object *);

#define	INTERNAL	1
#define	EXTERNAL	2
#define	INHERITED	3

#define P_INTERNAL(x,j) ((x)->p.p_internal[(j) % (x)->p.p_internal_size])
#define P_EXTERNAL(x,j) ((x)->p.p_external[(j) % (x)->p.p_external_size])


static bool
member_string_eq(x, l)
object x, l;
{
	for (;  consp(l);  l = l->c.c_cdr)
		if (string_eq(x, l->c.c_car))
			return(TRUE);
	return(FALSE);
}

static bool
designate_package(object x,struct package *p) {

  switch(type_of(x)) {
  case t_simple_string:
  case t_string:
    return string_eq(x,p->p_name) || member_string_eq(x, p->p_nicknames);
    break;
  case t_character: case t_symbol:
    return designate_package(coerce_to_string(x),p);
    break;
  case t_package:
    return x==(object)p;
    break;
  default:
    FEwrong_type_argument(TSor_symbol_string_package,x);
    break;
  }
  return FALSE;

}

/* #define bad_package_name(a) (type_of(a)==t_string &&\ */
/*                              (memchr((a)->st.st_self,'-',(a)->st.st_fillp) || \ */
/* 			      ((a)->st.st_self[0]=='*' && (a)->st.st_fillp==1))) */

#define check_package_designator(a) if (!stringp(a) && \
                                        type_of(a)!=t_character && \
					type_of(a)!=t_symbol && \
					type_of(a)!=t_package) \
					   FEwrong_type_argument(TSor_symbol_string_package,(a))
#define check_type_or_symbol_string_package(a) check_package_designator(*a)

static void
rehash_pack(object **ptab,ufixnum *n,fixnum m) { 
  object *ntab;
  object *tab = *ptab;
  object l,ll;
  fixnum k,i;
  i=0;
  k = *n;
  {BEGIN_NO_INTERRUPT;
  ntab= AR_ALLOC(alloc_contblock,m,object);
  *ptab = ntab;
  *n=m;
  while(i<m) ntab[i++]=Cnil;
   for(i=0 ; i< k; i++)
   	for (l = tab[i];  consp(l);)
	  {int j =pack_hash(l->c.c_car)%m;
	   ll=l->c.c_cdr;
	   l->c.c_cdr = ntab[j];
	   ntab[j]=l;
	   l=ll;
	 }
   END_NO_INTERRUPT;}
}

/* some prime numbers suitable for package sizes */

static int package_sizes[]={
  97,251, 509, 1021, 2039, 4093, 8191, 16381,
  32749, 65521, 131071, 262139,   524287, 1048573};

static int
suitable_package_size(ufixnum n)
{int *i=package_sizes;
 if (n>= 1000000) return 1048573;
 while(*i < n) { i++;}
 return *i;}
   
/*
	Make_package(n, ns, ul, isize , esize) makes a package with name n,
	which must be a string or a symbol,
	and nicknames ns, which must be a list of strings or symbols,
	and uses packages in list ul, which must be a list of packages
	or package names i.e. strings or symbols.
*/
static object
make_package(n, ns, ul,isize,esize)
object n, ns, ul;
int isize,esize;
{

	object x, y;
	int i;
	vs_mark;
	{ BEGIN_NO_INTERRUPT;
	BEGIN:
	  n=coerce_to_string(n);
	if (find_package(n) != Cnil) {
	  PACKAGE_CERROR(n,"Input new package","Package already exists",0);
	  NEW_INPUT(n);
	  goto BEGIN;
	}
	x = alloc_object(t_package);
	x->p.p_name = n;
	x->p.p_nicknames = Cnil;
	x->p.p_shadowings = Cnil;
	x->p.p_uselist = Cnil;
	x->p.p_usedbylist = Cnil;
	x->p.p_internal = NULL;
	x->p.p_external = NULL;
	x->p.p_internal_size = (isize ? isize : suitable_package_size(200));
	x->p.p_external_size = (esize ? esize : suitable_package_size(60));
	x->p.p_internal_fp =0;   
	x->p.p_external_fp =0;
	
	vs_push(x);
	for (;  !endp(ns);  ns = ns->c.c_cdr) {
	        n = ns->c.c_car;
		n=coerce_to_string(n);
		if (find_package(n) != Cnil) {
			vs_reset;
			PACKAGE_CERROR(n,"Input new nicknames list","Package already exists",0);
			NEW_INPUT(ns);
			goto BEGIN;
		}
		x->p.p_nicknames = make_cons(n, x->p.p_nicknames);
	}
	for (;  !endp(ul);  ul = ul->c.c_cdr) {
		if (type_of(ul->c.c_car) == t_package)
			y = ul->c.c_car;
		else {
			y = find_package(ul->c.c_car);
			if (y == Cnil) {
			  PACKAGE_CERROR(ul->c.c_car,"Continue anyway","No such package",0);
			  continue;
			}
		}
		x->p.p_uselist = make_cons(y, x->p.p_uselist);
		y->p.p_usedbylist = make_cons(x, y->p.p_usedbylist);
	}
	x->p.p_internal
	= AR_ALLOC(alloc_contblock,x->p.p_internal_size,object);
	for (i = 0;  i < x->p.p_internal_size;  i++)
		x->p.p_internal[i] = Cnil;
	x->p.p_external
	= AR_ALLOC(alloc_contblock,x->p.p_external_size,object);
	for (i = 0;  i < x->p.p_external_size;  i++)
		x->p.p_external[i] = Cnil;
	x->p.p_link = pack_pointer;
	pack_pointer = &(x->p);
	vs_reset;
        END_NO_INTERRUPT;}	
	return(x);
}

static void
use_package(object,object);

static object
in_package(n, ns, ul,isize,esize)
object n, ns, ul;
int isize,esize;
{

	object x, y;
	vs_mark;

 BEGIN:
	x = find_package(n);
	if (x == Cnil) {
	  x = make_package(n, ns, ul,isize,esize);
	  goto L;
	}
	if (isize) rehash_pack(&(x->p.p_internal),
		&x->p.p_internal_size,isize);
	for (;  !endp(ns);  ns = ns->c.c_cdr) {
		n = ns->c.c_car;
		n=coerce_to_string(n);
		y = find_package(n);
		if (x == y)
			continue;
		if (y != Cnil) {
		  PACKAGE_CERROR(n,"Input new nicknames list","Package already exists",0);
		  NEW_INPUT(ns);
		  goto BEGIN;
		}
		x->p.p_nicknames = make_cons(n, x->p.p_nicknames);
	}
	for (;  !endp(ul);  ul = ul->c.c_cdr)
		use_package(ul->c.c_car, x);
L:
	sLApackageA->s.s_dbind = x;
	vs_reset;
	return(x);
}

static object
rename_package(x, n, ns)
object x, n, ns;
{

	object y;
	vs_mark;

 BEGIN:
	n=coerce_to_string(n);
   	if (!(equal(x->p.p_name,n)) &&
	    find_package(n) != Cnil) {
	  PACKAGE_CERROR(n,"Input new package","Package already exists",0);
	  NEW_INPUT(n);
	  goto BEGIN;
	}
	x->p.p_name = n;
	x->p.p_nicknames = Cnil;
	for (;  !endp(ns);  ns = ns->c.c_cdr) {
		n = ns->c.c_car;
		n=coerce_to_string(n);
		y = find_package(n);
		if (x == y)
			continue;
		if (y != Cnil) {
		  PACKAGE_CERROR(n,"Input nicknames list","Package already exists",0);
		  NEW_INPUT(ns);
		  goto BEGIN;
		}
		x->p.p_nicknames = make_cons(n, x->p.p_nicknames);
	}
	vs_reset;
	return(x);
}


/*
	Find_package(n) seaches for a package with name n,
	which is a string or a symbol.
	If not so, an error is signaled.
*/
object
find_package(n)
object n;
{
	struct package *p;

	check_package_designator(n);
	for (p = pack_pointer;  p != NULL;  p = p->p_link) 
	  if (designate_package(n,p))
	    return ((object)p);
	return(Cnil);
}

static object
coerce_to_package(p)
object p;
{
	object pp;

	if (type_of(p) == t_package)
		return(p);
	pp = find_package(p);
	if (pp == Cnil) {
	  PACKAGE_CERROR(p,"Input new package","No such package",0);
	  NEW_INPUT(p);
	  return coerce_to_package(p);
	}
	return(pp);
}

object
current_package()
{
	object x;

	x = symbol_value(sLApackageA);
	if (type_of(x) != t_package) {
		sLApackageA->s.s_dbind = user_package;
		FEerror("The value of *PACKAGE*, ~S, was not a package.",
			1, x);
	}
	return(x);
}

/*
	Pack_hash(st) hashes string st
	and returns the index for a hash table of a package.
*/

int
pack_hash(x)
object x;
{unsigned int h=0;
  x=coerce_to_string(x);
  {int len=VLEN(x);
  char *s;
#define HADD(i,j,k,l) (h+=s[i],h+=s[j]<<8,h+=s[k]<<13,h+=s[l]<<23)
#define HADD2(i,j) (h+=s[i]<<5,h+=s[j]<<15)
  s=x->st.st_self;
  switch(len) {
  case 0: break;
  case 10: 
  case 9: HADD(1,4,6,8); HADD2(5,7); goto END;
  case 8: HADD(1,3,5,7); HADD2(2,4); goto END;
  case 7: HADD(1,3,4,5); HADD2(6,2); goto END;
  case 6: HADD(1,3,4,5); HADD2(0,2); goto END;
  case 5: h+= s[4] << 13;
  case 4: h+= s[3] << 24;
  case 3: h+= s[2]<< 16;
  case 2: h+= s[1] << 8;
  case 1: h+= s[0] ;
    break;
  default:
    HADD(3,6,len-2,len-4); HADD2(1,len-1);
    if (len > 15) {HADD2(7,10);		   
		 }
  }
 END:
  h &= 0x7fffffff; 
  return(h);
}}

DEFUN("PACK-HASH",object,fSpack_hash,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)(fixnum)pack_hash(x));
}

DEFUN("SET-SYMBOL-HPACK",object,fSset_symbol_hpack,SI,2,2,NONE,OO,OO,OO,OO,(object p,object s),"") { 
  check_type_package(&p); 
  check_type_sym(&s);
  RETURN1(s->s.s_hpack=p); 
} 

DEFUN("PACKAGE-INTERNAL",object,fSpackage_internal,SI,2,2,NONE,OO,IO,OO,OO,(object x,fixnum i),"") {
  check_type_package(&x);
  RETURN1(x->p.p_internal[i]);
}

DEFUN("PACKAGE-INTERNAL_SIZE",object,fSpackage_internal_size,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  check_type_package(&x);
  RETURN1((object)x->p.p_internal_size);
}

DEFUN("PACKAGE-EXTERNAL",object,fSpackage_external,SI,2,2,NONE,OO,IO,OO,OO,(object x,fixnum i),"") {
  check_type_package(&x);
  RETURN1(x->p.p_external[i]);
}

DEFUN("PACKAGE-EXTERNAL_SIZE",object,fSpackage_external_size,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  check_type_package(&x);
  RETURN1((object)x->p.p_external_size);
}

/*
	Intern(st, p) interns string st in package p.
*/
object
intern(st, p)
object st, p;
{
	int j;
	object x, *ip, *ep, l, ul;
	vs_mark;

	st=coerce_to_string(st);
	{BEGIN_NO_INTERRUPT;
	j = pack_hash(st);
	ip = &P_INTERNAL(p ,j);

	for (l = *ip;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, st)) {
			intern_flag = INTERNAL;
			END_NO_INTERRUPT;return(l->c.c_car);
		}
	ep = &P_EXTERNAL(p,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, st)) {
			intern_flag = EXTERNAL;
			END_NO_INTERRUPT;return(l->c.c_car);
		}
	for (ul=p->p.p_uselist; consp(ul); ul=ul->c.c_cdr)
		for (l = P_EXTERNAL(ul->c.c_car,j);
		     consp(l);
		     l = l->c.c_cdr)
			if (string_eq(l->c.c_car->s.s_name, st)) {
				intern_flag = INHERITED;
				END_NO_INTERRUPT;return(l->c.c_car);
			}
	x = make_symbol(st);
	vs_push(x);
	if (p == keyword_package) {
		x->s.s_stype = (short)stp_constant;
		x->s.tt=2;
		x->s.s_dbind = x;
		*ep = make_cons(x, *ep);
		keyword_package->p.p_external_fp ++;
		intern_flag = 0;
	} else {
		*ip = make_cons(x, *ip);
		if (p->p.p_internal_fp++>(p->p.p_internal_size << 1))
			rehash_pack(&(p->p.p_internal),&p->p.p_internal_size,
				    suitable_package_size(p->p.p_internal_fp));
		intern_flag = 0;
	}
	if (x->s.s_hpack == Cnil)
		x->s.s_hpack = p;
	vs_reset;
	END_NO_INTERRUPT;return(x);
}}

/*
	Find_symbol(st, p) searches for string st in package p.
*/
object
find_symbol(st, p)
object st, p;
{
	int j;
	object *ip, *ep, l, ul;
	{BEGIN_NO_INTERRUPT;
	st=coerce_to_string(st);
	j = pack_hash(st);
	ip = &P_INTERNAL(p ,j);
	for (l = *ip;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, st)) {
			intern_flag = INTERNAL;
			END_NO_INTERRUPT;return(l->c.c_car);
		}
	ep = &P_EXTERNAL(p,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, st)) {
			intern_flag = EXTERNAL;
			END_NO_INTERRUPT;return(l->c.c_car);
		}
	for (ul=p->p.p_uselist; consp(ul); ul=ul->c.c_cdr)
		for (l = P_EXTERNAL(ul->c.c_car,j);
		     consp(l);
		     l = l->c.c_cdr)
			if (string_eq(l->c.c_car->s.s_name, st)) {
				intern_flag = INHERITED;
				END_NO_INTERRUPT;return(l->c.c_car);
			}
	intern_flag = 0;
	END_NO_INTERRUPT;return(Cnil);
}}

static bool
unintern(s, p)
object s, p;
{
	object x, y, l, *lp;
	int j;
	{BEGIN_NO_INTERRUPT;
	j = pack_hash(s);
	x = find_symbol(s, p);
	if (intern_flag == INTERNAL && s == x) {
		lp = &P_INTERNAL(p ,j);
		if (member_eq(s, p->p.p_shadowings))
			goto L;
		goto UNINTERN;
	}
	if (intern_flag == EXTERNAL && s == x) {
		lp = &P_EXTERNAL(p,j);
		if (member_eq(s, p->p.p_shadowings))
			goto L;
		goto UNINTERN;
	}
	END_NO_INTERRUPT;return(FALSE);

L:
	x = OBJNULL;
	for (l = p->p.p_uselist; consp(l); l = l->c.c_cdr) {
		y = find_symbol(s, l->c.c_car);
		if (intern_flag == EXTERNAL) {
			if (x == OBJNULL)
				x = y;
			else if (x != y) {
			  PACKAGE_CERROR(p,"Input new symbol","Name conflict on unintern of shadowing symbol ~s",1,s);
			  NEW_INPUT(s);
			  goto L;
			}
		}
	}
	delete_eq(s, &p->p.p_shadowings);

UNINTERN:
	delete_eq(s, lp);
	if (s->s.s_hpack == p) {
		s->s.s_hpack = Cnil;
		s->s.tt=0;
	}
	if ((enum stype)s->s.s_stype != stp_ordinary)
		uninterned_list = make_cons(s, uninterned_list);
	END_NO_INTERRUPT;return(TRUE);
}}

void
export(s, p)
object s, p;
{
	object x;
	int j;
	object *ep, *ip, l;

BEGIN:
	ip = NULL;
	j = pack_hash(s);
	x = find_symbol(s, p);
	if (intern_flag) {
		if (x != s) {
			import(s, p);	/*  signals an error  */
			goto BEGIN;
		}
		if (intern_flag == INTERNAL)
			ip = &P_INTERNAL(p ,j);
		else if (intern_flag == EXTERNAL)
			return;
	} else {
	  PACKAGE_CERROR(p,"Input new symbol","Symbol ~s not accessible",1,s);
	  NEW_INPUT(s);
	  goto BEGIN;
	}
	for (l = p->p.p_usedbylist;
	     consp(l);
	     l = l->c.c_cdr) {
	        x = find_symbol(s, l->c.c_car);
		if (intern_flag && s != x &&
		    !member_eq(x, l->c.c_car->p.p_shadowings)) {
		  PACKAGE_CERROR(p,"Input new symbol","Name conflict on exporting ~s",1,s);
		  NEW_INPUT(s);
		  goto BEGIN;
		}
	}
	if (ip != NULL)
		{delete_eq(s, ip);
		 p->p.p_internal_fp--;}
	ep = &P_EXTERNAL(p,j);
	p->p.p_external_fp++;
	*ep = make_cons(s, *ep);
}

static void
unexport(s, p)
object s, p;
{
	object x, *ep, *ip;
	int j;

 BEGIN:
	if (p == keyword_package) {
	  PACKAGE_CERROR(p,"Input new package","Cannot unexport a symbol from the keyword",0);
	  NEW_INPUT(p);
	  goto BEGIN;
	}
	x = find_symbol(s, p);
	if (/* intern_flag != EXTERNAL || */ x != s) {
	  PACKAGE_CERROR(p,"Input new symbol","Symbol ~s not in package.",1,s);
	  NEW_INPUT(s);
	  goto BEGIN;
	}
/* "Cannot unexport the symbol ~S~%\ */
/* from ~S,~%\ */
/* because the symbol is not an external symbol~%\ */
/* of the package.", 2, s, p); */
	j = pack_hash(s);
	ep = &P_EXTERNAL(p,j);
	delete_eq(s, ep);
	ip = &P_INTERNAL(p ,j);
	p->p.p_internal_fp++;
	*ip = make_cons(s, *ip);
}

void
import(s, p)
object s, p;
{
	object x;
	int j;
	object *ip;

 BEGIN:
	x = find_symbol(s, p);
	if (intern_flag) {
	  if (x != s) {
	    PACKAGE_CERROR(p,"Input new symbol","Name conflict on importing ~s",1,s);
	    NEW_INPUT(s);
	    goto BEGIN;
	  }
	  if (intern_flag == INTERNAL || intern_flag == EXTERNAL)
	    return;
	}
	j = pack_hash(s);
	ip = &P_INTERNAL(p ,j);
	p->p.p_internal_fp++;
	*ip = make_cons(s, *ip);
	if (s->s.s_hpack==Cnil) {if (p==keyword_package) s->s.tt=2;s->s.s_hpack=p;}
}

static void
shadowing_import(s, p)
object s, p;
{
	object x, *ip;

	x=find_symbol(s, p);
	if (intern_flag && intern_flag != INHERITED) {
		if (x == s) {
			if (!member_eq(x, p->p.p_shadowings))
				p->p.p_shadowings
				= make_cons(x, p->p.p_shadowings);
			return;
		}
		if(member_eq(x, p->p.p_shadowings))
			delete_eq(x, &p->p.p_shadowings);
		if (intern_flag == INTERNAL)
			delete_eq(x, &P_INTERNAL(p,pack_hash(x)));
		else
			delete_eq(x, &P_EXTERNAL(p ,pack_hash(x)));
		if (x->s.s_hpack == p) {
			x->s.s_hpack = Cnil;
			x->s.tt=0;
		}
		if ((enum stype)x->s.s_stype != stp_ordinary)
			uninterned_list = make_cons(x, uninterned_list);
	}
	ip = &P_INTERNAL(p ,pack_hash(s));
	*ip = make_cons(s, *ip);
	p->p.p_internal_fp++;	
	p->p.p_shadowings = make_cons(s, p->p.p_shadowings);
}

static void
shadow(s, p)
object s, p;
{
	int j;
	object *ip,x;

	s=coerce_to_string(s);
	x=find_symbol(s, p);
	if (intern_flag == INTERNAL || intern_flag == EXTERNAL) {
		p->p.p_shadowings = make_cons(x, p->p.p_shadowings);
		return;
	}
	j = pack_hash(s);
	ip = &P_INTERNAL(p ,j);
	vs_push(make_symbol(s));
	vs_head->s.s_hpack = p;
	if (p==keyword_package) vs_head->s.tt=2;
	*ip = make_cons(vs_head, *ip);
	p->p.p_internal_fp++;
	p->p.p_shadowings = make_cons(vs_head, p->p.p_shadowings);
	vs_popp;
}

static void
use_package(x0, p)
object x0, p;
{
	object x = x0;
	int i;
	object y, l;

 BEGIN:
	if (type_of(x) != t_package) {
		x = find_package(x);
		if (x == Cnil) {
		  PACKAGE_CERROR(x0,"Input new package","No such package",0);
		  NEW_INPUT(x0);
		  goto BEGIN;
		}
	}
	if (x == keyword_package) {
		PACKAGE_CERROR(x,"Input new package","Cannot use keyword package",0);
		NEW_INPUT(x);
		goto BEGIN;
	}
	if (p == x)
		return;
	if (member_eq(x, p->p.p_uselist))
		return;
	for (i = 0;  i < x->p.p_external_size;  i++)
		for (l = P_EXTERNAL(x ,i);
		     consp(l);
		     l = l->c.c_cdr) {
			y = find_symbol(l->c.c_car, p);
			if (intern_flag && l->c.c_car != y
			    && ! member_eq(y,p->p.p_shadowings)
			    ) {
			  PACKAGE_CERROR(p,"Input new package","Name conflict on using ~s from ~s",2,p,y);
			  NEW_INPUT(p);
			  goto BEGIN;
			}
		}
	p->p.p_uselist = make_cons(x, p->p.p_uselist);
	x->p.p_usedbylist = make_cons(p, x->p.p_usedbylist);
}

static void
unuse_package(x0, p)
object x0, p;
{
	object x = x0;

 BEGIN:
	if (type_of(x) != t_package) {
		x = find_package(x);
		if (x == Cnil) {
		  PACKAGE_CERROR(x0,"Input new package","No such package",0);
		  NEW_INPUT(x0);
		  goto BEGIN;
		}
	}
	delete_eq(x, &p->p.p_uselist);
	delete_eq(p, &x->p.p_usedbylist);
}



static object
delete_package(object n) {

  struct package *p,*pp;
  object t;

  for (p = pack_pointer,pp=NULL;  p != NULL;  pp=p,p = p->p_link) 
    
    if (designate_package(n,p)) {
      
      if (p->p_usedbylist!=Cnil) {
 	
	PACKAGE_CERROR((object)n,"Delete anyway","Package used by other packages",0);

	for (t=p->p_usedbylist;!endp(t);t=t->c.c_cdr)
	  unuse_package((object)p,t->c.c_car);

      }

      if (p->p_uselist!=Cnil) {
	for (t=p->p_uselist;!endp(t);t=t->c.c_cdr)
	  unuse_package(t->c.c_car,(object)p);
      }

      p->p_name=Cnil;

      if (pp) 
	pp->p_link=p->p_link;
      else
	pack_pointer=p->p_link;
	
      return(Ct);
      
    }
  
  if (type_of(n)!=t_package) {
    PACKAGE_CERROR(n,"Input new package","No such package",0);
    NEW_INPUT(n);
    return delete_package(n);
  }

  return(Cnil);
  
}
  
/*  			   (use `make_cons(lisp_package, Cnil)`) */

DEFUN("MAKE-PACKAGE-INT",object,fSmake_package_int,SI,3,3,NONE,OO,OO,OO,OO,(object name,object nicknames,object use),"") {

  RETURN1(make_package(name,nicknames,use,0,0));

}


@(defun in_package (pack_name &key nicknames (use Cnil use_sp)
		      (internal `small_fixnum(0)`)
		      (external `small_fixnum(0)`)
		    )
@
        pack_name=coerce_to_string(pack_name);
	check_type_or_string_symbol(&pack_name);
	if (find_package(pack_name) == Cnil && !(use_sp))
		use = make_cons(lisp_package, Cnil);
	@(return `in_package(pack_name, nicknames, use,fix(internal),fix(external))`)
@)

extern object sKuse;
extern object sKnicknames;
DEF_ORDINARY("IN-PACKAGE-INTERNAL",sSin_package_internal,SI,"");
DEFUN("IN-PACKAGE-INTERNAL",object,fSin_package_internal,SI,1,2,NONE,OO,OO,OO,OO,(object p,...),"") {

  fixnum nargs=INIT_NARGS(1);
  va_list ap;
  object use=Cnil,nick=Cnil,l=Cnil,f=OBJNULL,r;

  p = coerce_to_string(p);
  va_start(ap,p);
  r = NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  /*fixme non-std error check?*/
  for (;consp(r) && consp(r->c.c_cdr);r=r->c.c_cdr->c.c_cdr) {
    if (r->c.c_car==sKuse)
      use=Ieval1(r->c.c_cdr->c.c_car);
    if (r->c.c_car==sKnicknames)
      nick=Ieval1(r->c.c_cdr->c.c_car);
  }

  RETURN1(in_package(p,nick,use,0,0));

}

DEFUN("FIND-PACKAGE",object,fLfind_package,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(find_package(x));/*FIXME p->p_link not exposable in lisp*/
}

LFD(Ldelete_package_internal)()
{
	check_arg(1);

	vs_base[0] = delete_package(vs_base[0]);
}

LFD(Lpackage_name)()
{
  object t;

  check_arg(1);

  check_package_designator(vs_base[0]);
  t=coerce_to_package(vs_base[0]);
  vs_base[0]=t==Cnil ? t : t->p.p_name;

}

LFD(Lpackage_nicknames)()
{
	check_arg(1);

	check_package_designator(vs_base[0]);
	vs_base[0] = coerce_to_package(vs_base[0]);
	vs_base[0] = vs_base[0]->p.p_nicknames;
}

@(defun rename_package (pack new_name &o new_nicknames)
@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
        new_name=coerce_to_string(new_name);
	check_type_or_string_symbol(&new_name);
	@(return `rename_package(pack, new_name, new_nicknames)`)
@)

LFD(Lpackage_use_list)()
{
	check_arg(1);

	check_package_designator(vs_base[0]);
	vs_base[0] = coerce_to_package(vs_base[0]);
	vs_base[0] = vs_base[0]->p.p_uselist;
}

LFD(Lpackage_used_by_list)()
{
	check_arg(1);

	check_package_designator(vs_base[0]);
	vs_base[0] = coerce_to_package(vs_base[0]);
	vs_base[0] = vs_base[0]->p.p_usedbylist;
}

static void
FFN(Lpackage_shadowing_symbols)()
{
	check_arg(1);

	check_package_designator(vs_base[0]);
	vs_base[0] = coerce_to_package(vs_base[0]);
	vs_base[0] = vs_base[0]->p.p_shadowings;
}

LFD(Llist_all_packages)() {

  struct package *p;
  object x,*l;
  int i;

  check_arg(0);

  for (l=&x,p=pack_pointer,i=0;p!=NULL;p=p->p_link,i++)
    collect(l,make_cons((object)p,Cnil));
  *l=Cnil;
  vs_push(x);

}

@(defun intern (strng &optional (p `current_package()`) &aux sym)
@
	check_type_string(&strng);
	check_package_designator(p);
	p = coerce_to_package(p);
	sym = intern(strng, p);
	if (intern_flag == INTERNAL)
		@(return sym sKinternal)
	if (intern_flag == EXTERNAL)
		@(return sym sKexternal)
	if (intern_flag == INHERITED)
		@(return sym sKinherited)
	@(return sym Cnil)
@)

@(defun find_symbol (strng &optional (p `current_package()`))
	object x;
@
	check_type_string(&strng);
	check_package_designator(p);
	p = coerce_to_package(p);
	x = find_symbol(strng, p);
	if (intern_flag == INTERNAL)
		@(return x sKinternal)
	if (intern_flag == EXTERNAL)
		@(return x sKexternal)
	if (intern_flag == INHERITED)
		@(return x sKinherited)
	@(return Cnil Cnil)
@)

@(defun unintern (symbl &optional (p `current_package()`))
@
	check_type_sym(&symbl);
	check_package_designator(p);
	p = coerce_to_package(p);
	if (unintern(symbl, p))
		@(return Ct)
	else
		@(return Cnil)
@)

@(defun export (symbols &o (pack `current_package()`))
	object l;

@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
BEGIN:
	switch (type_of(symbols)) {
	case t_symbol:
		if (symbols == Cnil)
			break;
		export(symbols, pack);
		break;

	case t_cons:
	  for (l = symbols;  !endp(l);  l = l->c.c_cdr) {
	    check_type_sym(&l->c.c_car);
	    export(l->c.c_car, pack);
	  }
	  break;

	default:
		check_type_sym(&symbols);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun unexport (symbols &o (pack `current_package()`))
	object l;

@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
BEGIN:
	switch (type_of(symbols)) {
	case t_symbol:
		if (symbols == Cnil)
			break;
		unexport(symbols, pack);
		break;

	case t_cons:
	  for (l = symbols;  !endp(l);  l = l->c.c_cdr) {
	    check_type_sym(&l->c.c_car);
	    unexport(l->c.c_car, pack);
	  }
		break;

	default:
		check_type_sym(&symbols);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun import_internal (symbols &o (pack `current_package()`))
	object l;
@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
BEGIN:
	switch (type_of(symbols)) {
	case t_symbol:
		if (symbols == Cnil)
			break;
		import(symbols, pack);
		break;

	case t_cons:
		for (l = symbols;  !endp(l);  l = l->c.c_cdr)
			import(l->c.c_car, pack);
		break;

	default:
		check_type_sym(&symbols);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun shadowing_import (symbols &o (pack `current_package()`))
	object l;
@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
BEGIN:
	switch (type_of(symbols)) {
	case t_symbol:
		if (symbols == Cnil)
			break;
		shadowing_import(symbols, pack);
		break;

	case t_cons:
		for (l = symbols;  !endp(l);  l = l->c.c_cdr)
			shadowing_import(l->c.c_car, pack);
		break;

	default:
		check_type_sym(&symbols);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun shadow (symbols &o (pack `current_package()`))
	object l;
@
	check_package_designator(pack);
	pack = coerce_to_package(pack);
BEGIN:
	switch (type_of(symbols)) {
	case t_symbol: 
	case t_simple_string:
	case t_string: 
	case t_character:
		if (symbols == Cnil)
			break;
		shadow(symbols, pack);
		break;

	case t_cons:
		for (l = symbols;  !endp(l);  l = l->c.c_cdr)
			shadow(l->c.c_car, pack);
		break;

	default:
		check_type_or_symbol_string(&symbols);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun use_package (pack &o (pa `current_package()`))
	object l;
@
	check_package_designator(pa);
	pa = coerce_to_package(pa);
BEGIN:
	switch (type_of(pack)) {
	case t_symbol:
		if (pack == Cnil)
			break;

	case t_simple_string:
	case t_string:
	case t_package:
	case t_character:
		use_package(pack, pa);
		break;

	case t_cons:
		for (l = pack;  !endp(l);  l = l->c.c_cdr)
			use_package(l->c.c_car, pa);
		break;

	default:
		check_type_package(&pack);
		goto BEGIN;
	}
	@(return Ct)
@)

@(defun unuse_package (pack &o (pa `current_package()`))
	object l;
@
	check_package_designator(pa);
	pa = coerce_to_package(pa);
BEGIN:
	switch (type_of(pack)) {
	case t_symbol:
		if (pack == Cnil)
			break;

	case t_simple_string:
	case t_string:
	case t_package:
	case t_character:
		unuse_package(pack, pa);
		break;

	case t_cons:
		for (l = pack;  !endp(l);  l = l->c.c_cdr)
			unuse_package(l->c.c_car, pa);
		break;

	default:
		check_type_package(&pack);
		goto BEGIN;
	}
	@(return Ct)
@)

/* LFD(siLpackage_internal)() */
/* { */

/* 	int j=0; */

/* 	check_arg(2); */
/* 	check_type_package(&vs_base[0]); */
/* 	if (type_of(vs_base[1]) != t_fixnum || */
/* 	    (j = fix(vs_base[1])) < 0 || j >= vs_base[0]->p.p_internal_size) */
/* 		FEerror("~S is an illegal index to a package hashtable.", */
/* 			1, vs_base[1]); */
/* 	vs_base[0] = P_INTERNAL(vs_base[0],j); */
/* 	vs_popp; */
/* } */

/* LFD(siLpackage_external)() */
/* { */
/* 	int j=0; */

/* 	check_arg(2); */
/* 	check_type_package(&vs_base[0]); */
/* 	if (type_of(vs_base[1]) != t_fixnum || */
/* 	    (j = fix(vs_base[1])) < 0 || j >= vs_base[0]->p.p_external_size) */
/* 		FEerror("~S is an illegal index to a package hashtable.", */
/* 			1, vs_base[1]); */
/* 	vs_base[0] = P_EXTERNAL(vs_base[0],j); */
/* 	vs_popp; */
/* } */

static void
FFN(siLpackage_size)()
{object p;
 p=vs_base[0];
 check_type_package(&p);
 check_arg(1);
 vs_base[0]=make_fixnum(p->p.p_external_size);
 vs_base[1]=make_fixnum(p->p.p_internal_size);
 vs_top=vs_base+2;
 return;
}
 
DEF_ORDINARY("EXTERNAL",sKexternal,KEYWORD,"");
DEF_ORDINARY("INHERITED",sKinherited,KEYWORD,"");
DEF_ORDINARY("INTERNAL",sKinternal,KEYWORD,"");
DEF_ORDINARY("NICKNAMES",sKnicknames,KEYWORD,"");
DEF_ORDINARY("USE",sKuse,KEYWORD,"");
DEFVAR("*PACKAGE*",sLApackageA,LISP,lisp_package,"");


void
gcl_init_package()
{

	lisp_package
	= make_package(make_simple_string("COMMON-LISP"),
		       list(2,make_simple_string("CL"),make_simple_string("LISP")),Cnil,47,509);
	user_package
	= make_package(make_simple_string("COMMON-LISP-USER"),
		       list(2,make_simple_string("CL-USER"),make_simple_string("USER")),
		       make_cons(lisp_package, Cnil),509,97);
	keyword_package
	= make_package(make_simple_string("KEYWORD"),
		       Cnil, Cnil,11,509);
	system_package
	= make_package(make_simple_string("SYSTEM"),
		       make_cons(make_simple_string("SI"),
			         make_cons(make_simple_string("SYS"),
					   Cnil)),
		       make_cons(lisp_package, Cnil),251,157);

	gmp_package=make_package(make_simple_string("GMP"),Cnil,Cnil,0,0);

	/*  There is no need to enter a package as a mark origin.  */

	uninterned_list = Cnil;
	enter_mark_origin(&uninterned_list);
}

void
gcl_init_package_function()
{
	make_si_function("DELETE-PACKAGE-INTERNAL", Ldelete_package_internal);
	make_si_function("KCL-IN-PACKAGE", Lin_package);
	make_function("IN-PACKAGE", Lin_package);
/* 	make_function("FIND-PACKAGE", Lfind_package); */
	make_function("PACKAGE-NAME", Lpackage_name);
	make_function("PACKAGE-NICKNAMES", Lpackage_nicknames);
	make_function("RENAME-PACKAGE", Lrename_package);
	make_function("PACKAGE-USE-LIST", Lpackage_use_list);
	make_function("PACKAGE-USED-BY-LIST", Lpackage_used_by_list);
	make_function("PACKAGE-SHADOWING-SYMBOLS",Lpackage_shadowing_symbols);
	make_function("LIST-ALL-PACKAGES", Llist_all_packages);
	make_function("INTERN", Lintern);
	make_function("FIND-SYMBOL", Lfind_symbol);
	make_function("UNINTERN", Lunintern);
	make_function("EXPORT", Lexport);
	make_function("UNEXPORT", Lunexport);
	make_si_function("IMPORT-INTERNAL", Limport_internal);
	make_function("SHADOWING-IMPORT", Lshadowing_import);
	make_function("SHADOW", Lshadow);
	make_function("USE-PACKAGE", Luse_package);
	make_function("UNUSE-PACKAGE", Lunuse_package);

	make_si_function("PACKAGE-SIZE",siLpackage_size);
	/* make_si_function("SET-SYMBOL-HPACK", siLset_symbol_hpack); */
	/* make_si_function("PACKAGE-INTERNAL", siLpackage_internal); */
	/* make_si_function("PACKAGE-EXTERNAL", siLpackage_external); */
}
