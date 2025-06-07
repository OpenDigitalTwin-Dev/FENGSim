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

  hash.d                  06/2005 Update           Boyer & Hunt

For a number of reasons, we have modified this file from its
long-standing form to improve the performance of the Common Lisp
GETHASH and the SETHASH procedures.  These changes extend to the GCL
header file "h/object.h" where we included an additional field.

In the spirit of the comment just above, we have attempted to write
down some observations, comments, and invariants, about the modified
code contained in this file.  The two C-code procedures that were
substantially modified are "gethash" and "sethash", which in turn,
required additional changes to be made in "make_hash_table" and
"extend_hashtable".

  - We never allow a hashtable to become completely full -- no matter
    what the REHASH-THRESHOLD is; we require that there is always at
    least one empty table entry.

*/


#define NEED_MP_H
#include <ctype.h>
#include "include.h"


object sLeq;
object sLeql;
object sLequal;
object sLequalp;

object sKsize;
object sKrehash_size;
object sKrehash_threshold;
object sKstatic;

#define MHSH(a_) ((a_) & ~(1UL<<(sizeof(a_)*CHAR_SIZE-1)))

typedef union {/*FIXME size checks*/
  float f;
  unsigned int ul;
} F2ul;

typedef union {
  double d;
  unsigned int ul[2];
} D2ul;


static ufixnum rtb[256];

#define MASK(n) (~(~0UL << (n)))

static ufixnum
ufixhash(ufixnum g) {
  ufixnum i,h;
  for (h=i=0;i<sizeof(g);g>>=CHAR_SIZE,i++)
    h^=rtb[g&MASK(CHAR_SIZE)];
  return h;
}

static ufixnum
uarrhash(void *v,void *ve,uchar off,uchar bits) {

  uchar *c=v,*ce=ve-(bits+(off ? off : CHAR_SIZE)>CHAR_SIZE ? 1 : 0),i;
  ufixnum h=0,*u=v,*ue=u+(ce-c)/sizeof(*u);
  
  if (!off)
    for (;u<ue;) h^=ufixhash(*u++);

  for (c=(void *)u;c+(off ? 1 : 0)<ce;c++)
    h^=rtb[(uchar)(((*c)<<off)|(off ? c[1]>>(CHAR_SIZE*sizeof(*c)-off) : 0))];

  for (i=off;bits--;i=(i+1)%CHAR_SIZE,c=i ? c : c+1)
    h^=rtb[((*c)>>(CHAR_SIZE-1-i))&0x1];

  return h;

}

#define hash_eq1(x) ufixhash((ufixnum)x/sizeof(x))
#define hash_eq(x)  MHSH(hash_eq1(x))


static ufixnum
hash_eql(object x) {

  ufixnum h;

  switch (type_of(x)) {

  case t_fixnum:
    h=ufixhash(fix(x));
    break;

  case t_character:
    h = rtb[char_code(x)];
    break;
    
  case t_bignum:
    { 
      MP_INT *mp = MP(x);
      void *v1=mp->_mp_d,*ve=v1+mpz_size(mp);

      h=uarrhash(v1,ve,0,0);
    }
    break;

  case t_ratio:
    h=hash_eql(x->rat.rat_num) + hash_eql(x->rat.rat_den);
    break;

  case t_shortfloat:  /*FIXME, sizeof int = sizeof float*/
    { 
      F2ul u;
      u.f=sf(x);
      h=ufixhash(u.ul);
    }
    break;
    
  case t_longfloat:
    { 
      D2ul u;
      u.d=lf(x);
      h=ufixhash(u.ul[0])^ufixhash(u.ul[1]);
    }
    break;

  case t_complex:
    h=hash_eql(x->cmp.cmp_real) + hash_eql(x->cmp.cmp_imag);
    break;

  default:
    h=hash_eq1(x);
    break;

  }

  return MHSH(h);

}


#define ihash_equal(a_,b_) ((type_of(a_)==t_symbol && (a_)->s.s_hash) ? (a_)->s.s_hash : ihash_equal1(a_,b_))
ufixnum
ihash_equal1(object x,int depth) {

  enum type tx;
  ufixnum h=0;
  
  cs_check(x);

BEGIN:
  if (depth++<=3)
    switch ((tx=type_of(x))) {
    case t_cons:
      h^=ihash_equal(x->c.c_car,depth)^rtb[labs(depth%(sizeof(rtb)/sizeof(*rtb)))];
      x = x->c.c_cdr;
      goto BEGIN;
      break;
    case t_symbol:
      x=coerce_to_string(x);
    case t_simple_string:
    case t_string:
      h^=uarrhash(x->st.st_self,x->st.st_self+VLEN(x),0,0);
      break;
    case t_package: 
      break;
    case t_bitvector:
    case t_simple_bitvector:
      {
	ufixnum *u=x->bv.bv_self+x->bv.bv_offset/BV_BITS;
	ufixnum *ue=x->bv.bv_self+(VLEN(x)+x->bv.bv_offset)/BV_BITS;
	uchar s=x->bv.bv_offset%BV_BITS;
	uchar m=((VLEN(x)+x->bv.bv_offset)%BV_BITS);

	for (;u<ue;) {
	  ufixnum v=(*u++)>>s;
	  if (u<ue||m) {
	    ufixnum w=(*u);
	    if (u==ue)
	      w&=BIT_MASK(m);
	    v|=w<<(sizeof(*u)-s);
	  }
	  h^=ufixhash(v);
	}
      }
      break;
    case t_pathname:
      h^=ihash_equal(x->pn.pn_host,depth);
      h^=ihash_equal(x->pn.pn_device,depth);
      h^=ihash_equal(x->pn.pn_directory,depth);
      h^=ihash_equal(x->pn.pn_name,depth);
      h^=ihash_equal(x->pn.pn_type,depth);
      /* version is ignored unless logical host */
      /* if ((type_of(x->pn.pn_host) == t_string) && */
      /* 	  (pathname_lookup(x->pn.pn_host,sSApathname_logicalA) != Cnil)) */
      /* 	h^=ihash_equal(x->pn.pn_version,depth); */
      h^=ihash_equal(x->pn.pn_version,depth);
      break;
    default:
      h^=hash_eql(x);
      break;
    }
  
  return MHSH(h);

}

DEFUN("HASH-EQUAL",object,fShash_equal,SI,2,2,NONE,IO,IO,OO,OO,(object x,fixnum depth),"") {
  return (object)ihash_equal(x,depth);
}

#define ihash_equalp(a_,b_) ((type_of(a_)==t_symbol && (a_)->s.s_hash) ? (a_)->s.s_hash : ihash_equalp1(a_,b_))
unsigned long
ihash_equalp1(object x,int depth) {

  enum type tx;
  unsigned long h = 0,j;
  long i;
  
  cs_check(x);

BEGIN:
  if (depth++ <=3)
    switch ((tx=type_of(x))) {
    case t_cons:
      h += ihash_equalp(x->c.c_car,depth)^rtb[labs(depth%(sizeof(rtb)/sizeof(*rtb)))];
      x = x->c.c_cdr;
      goto BEGIN;
      break;
    case t_symbol:
      x=coerce_to_string(x);
    case t_simple_string:
    case t_string:
      {
	ufixnum len=VLEN(x);
	uchar *s=(void *)x->st.st_self;
	for (;len--;)
	  h^=rtb[toupper(*s++)];
      }
      break;
    case t_package: 
      break;
    case t_bitvector:
    case t_simple_bitvector:
      {
	ufixnum *u=x->bv.bv_self+x->bv.bv_offset/BV_BITS;
	ufixnum *ue=x->bv.bv_self+(VLEN(x)+x->bv.bv_offset)/BV_BITS;
	uchar s=x->bv.bv_offset%BV_BITS;
	uchar m=((VLEN(x)+x->bv.bv_offset)%BV_BITS);

	for (;u<ue;) {
	  ufixnum v=(*u++)>>s;
	  if (u<ue||m) {
	    ufixnum w=(*u);
	    if (u==ue)
	      w&=BIT_MASK(m);
	    v|=w<<(sizeof(*u)-s);
	  }
	  h^=ufixhash(v);
	}
      }
      break;

    case t_simple_vector:
    case t_vector:
      h^=ufixhash(j=VLEN(x));
      j=j>10 ? 10 : j;
      for (i=0;i<j;i++)
	h^=ihash_equalp(aref(x,i),depth);
      break;
			
    case t_array:
      h^=ufixhash(j=x->a.a_rank);
      for (i=0;i<j-1;i++)
	h^=ufixhash(x->a.a_dims[i]);
      j=x->a.a_dim;
      j=j>10 ? 10 : j;
      for (i=0;i<j;i++)
	h^=ihash_equalp(aref(x,i),depth);
      break;
			
    case t_hashtable:
      h^=ufixhash(j=x->ht.ht_nent);
      h^=ufixhash(x->ht.ht_test);
      j=j>10 ? 10 : j;
      for (i=0;i<j;i++)
	if (x->ht.ht_self[i].c_cdr!=OBJNULL)
	  switch (x->ht.ht_test) {
	  case htt_eq:
	    h^=(((unsigned long)x->ht.ht_self[i].c_cdr)>>3) ^
	      ihash_equalp(x->ht.ht_self[i].c_car,depth);
	    break;
	  case htt_eql:
	    h^=hash_eql(x->ht.ht_self[i].c_cdr) ^
	      ihash_equalp(x->ht.ht_self[i].c_car,depth);
	    break;
	  case htt_equal:
	    h^=ihash_equal(x->ht.ht_self[i].c_cdr,depth) ^
	      ihash_equalp(x->ht.ht_self[i].c_car,depth);
	    break;
	  case htt_equalp:
	    h^=ihash_equalp(x->ht.ht_self[i].c_cdr,depth) ^
	      ihash_equalp(x->ht.ht_self[i].c_car,depth);
	    break;
	  }
      break;

    case t_pathname:
      h^=ihash_equalp(x->pn.pn_host,depth);
      h^=ihash_equalp(x->pn.pn_device,depth);
      h^=ihash_equalp(x->pn.pn_directory,depth);
      h^=ihash_equalp(x->pn.pn_name,depth);
      h^=ihash_equalp(x->pn.pn_type,depth);
      h^=ihash_equalp(x->pn.pn_version,depth);
      break;

    case t_structure:
      {
	unsigned char *s_type;
	struct s_data *def;
	def=S_DATA(x->str.str_def);
	s_type= & SLOT_TYPE(x->str.str_def,0);
	h^=ihash_equalp(def->name,depth);
	for (i=0;i<def->length;i++)
	  if (s_type[i]==aet_object)
	    h^=ihash_equalp(x->str.str_self[i],depth);
	  else
	    h^=ufixhash((long)x->str.str_self[i]);
	break;
      }

    case t_character:
      h^=rtb[toupper(x->ch.ch_code)];
      break;

    default:
      h^=hash_eql(x);
      break;
    }
  
  return MHSH(h);

}

	
DEFUN("HASH-EQUALP",object,fShash_equalp,SI,2,2,NONE,OO,IO,OO,OO,(object x,fixnum depth),"") {
  RETURN1(make_fixnum(ihash_equalp(x,depth)));
}

/* 

gethash

Here are conditions on the two inputs, key and hashtable, and the
value returned.

Condition 1.  key may not be OBJNULL.

Definition.  i is an "open" location in hashtable iff its key slot
holds OBJNULL.

Condition 2.  There is an open location in hashtable.

Definition.  i is an "initially open" location in hashtable iff it is
open and furthermore its value slot also holds OBJNULL.

Condition 3.  If there is an entry for key in hashtable, then starting
at the "init_hash_index" location of hashtable for key and searching
to the top of the hashtable, and then wrapping to start the search at
the beginning of the hashtable, will find the entry for key before
encountering an initially open location.  (What this means in practice
is that remhash must set the value field to something other than
OBJNULL, e.g., to NIL.)

Output condition.  If there is an entry in hashtable whose key slot
holds key, then the value returned is the address of that entry.  On
the otherhand, if there is no entry in hashtable whose key slot holds
key, then the value returned is the first open (not necessarily
intially open) slot encounterd starting at the hkey generated for key,
and wrappping if necessary.

*/

struct cons *
gethash(object key, object ht) {

  long s,q;
  struct cons *e,*ee,*first_open=NULL;
  static struct cons dummy={OBJNULL,OBJNULL};
  
  if (ht->ht.ht_cache && ht->ht.ht_cache->c_cdr==key)
    return ht->ht.ht_cache;
  ht->ht.ht_cache=NULL;

#define eq(x,y) x==y
#define hash_loop(t_,i_)						\
  for (q=ht->ht.ht_size,s=i_%q;s>=0;q=s,s=s?0:-1)			\
    for (e=ht->ht.ht_self,ee=e+q,e+=s;e<ee;e++) {			\
      object hkey=e->c_cdr;						\
      if (hkey==OBJNULL) {						\
	if (e->c_car==OBJNULL) return first_open ? first_open : e;	\
	if (!first_open) first_open=e;					\
      } else if (t_(key,hkey)) return ht->ht.ht_cache=e;		\
    }

  switch (ht->ht.ht_test) {
  case htt_eq:
    hash_loop(eq,hash_eq(key));
    break;
  case htt_eql:
    hash_loop(eql,hash_eql(key));
    break;
  case htt_equal:
    hash_loop(equal,ihash_equal(key,0));
    break;
  case htt_equalp:
    hash_loop(equalp,ihash_equalp(key,0));
    break;
  default:
    FEerror( "gethash:  Hash table not of type EQ, EQL, or EQUAL." ,0);
    return &dummy;
  }
  
  return first_open ? first_open : (FEerror("No free spot in hashtable ~S.", 1, ht),&dummy);

}

static void
extend_hashtable(object hashtable) {

  object old;
  fixnum new_size=0,new_max_ent=0,i;
  struct cons *hte;
  
  /* Compute new size for the larger hashtable */
  
  new_size=hashtable->ht.ht_size+1;
  switch (type_of(hashtable->ht.ht_rhsize)) {
  case t_fixnum:
    new_size *= fix(hashtable->ht.ht_rhsize);
    break;
  case t_shortfloat:
    new_size *= sf(hashtable->ht.ht_rhsize);
    break;
  case t_longfloat:
    new_size *= lf(hashtable->ht.ht_rhsize);
    break;
  }
  
  /* Compute the maximum number of entries */
  
  switch (type_of(hashtable->ht.ht_rhthresh)) {
  case t_fixnum:
    new_max_ent = fix(hashtable->ht.ht_rhthresh) + ( new_size - hashtable->ht.ht_size );
    break;
  case t_shortfloat:
    new_max_ent = (fixnum)(( new_size * sf(hashtable->ht.ht_rhthresh)) + 0.5 );
    break;
  case t_longfloat:
    new_max_ent = (fixnum)(( new_size * lf(hashtable->ht.ht_rhthresh)) + 0.5 );
    break;
  }
  
  if (new_max_ent>=new_size || new_max_ent<=0) 
    new_max_ent = new_size - 1;

  {
    BEGIN_NO_INTERRUPT;	
    old = alloc_object(t_hashtable);
    old->ht = hashtable->ht;
    vs_push(old);
    hashtable->ht.ht_cache = hashtable->ht.ht_self = NULL;
    hashtable->ht.ht_size = new_size;
    if (type_of(hashtable->ht.ht_rhthresh) == t_fixnum)
      hashtable->ht.ht_rhthresh =
	make_fixnum(fix(hashtable->ht.ht_rhthresh) +
		    (new_size - old->ht.ht_size));
    hashtable->ht.ht_self =
      (struct cons *)alloc_relblock(new_size * sizeof(struct cons));
    for (i = 0;  i < new_size;  i++) {
      hashtable->ht.ht_self[i].c_cdr = OBJNULL;
      hashtable->ht.ht_self[i].c_car = OBJNULL;
    }

    for (i=0;i<old->ht.ht_size;i++) 

      if (old->ht.ht_self[i].c_cdr != OBJNULL) {

	hte = gethash(old->ht.ht_self[i].c_cdr, hashtable);
	/* Initially empty, only empty locations. */
	hte->c_cdr = old->ht.ht_self[i].c_cdr;
	hte->c_car = old->ht.ht_self[i].c_car;

      }

    hashtable->ht.ht_nent = old->ht.ht_nent;
    hashtable->ht.ht_max_ent = new_max_ent;
    vs_popp;
    END_NO_INTERRUPT;

  }

}

void
sethash(object key, object hashtable, object value) {

  struct cons *e;
  
  if (hashtable->ht.ht_nent+1>=hashtable->ht.ht_max_ent)
    extend_hashtable(hashtable);

  e = gethash(key, hashtable);
  if (e->c_cdr == OBJNULL)
    hashtable->ht.ht_nent++;
  e->c_cdr = key;
  e->c_car = value;

}
	
DEFUN("MAKE-HASH-TABLE-INT",object,fSmake_hash_table_int,SI,5,5,NONE,OO,OO,OO,OO,
	  (object test,object size,object rehash_size,
	   object rehash_threshold,object staticp),"") {

  enum httest htt=0;
  fixnum i,max_ent=0,err;
  object h;

  if (test == sLeq || test == sLeq->s.s_gfdef)
    htt = htt_eq;
  else if (test == sLeql || test == sLeql->s.s_gfdef)
    htt = htt_eql;
  else if (test == sLequal || test == sLequal->s.s_gfdef)
     htt = htt_equal;
  else if (test == sLequalp || test == sLequalp->s.s_gfdef)
     htt = htt_equalp;
  else
     FEerror("~S is an illegal hash-table test function.",1, test);

  if (type_of(size)!=t_fixnum || fix(size)<0)
     FEerror("~S is an illegal hash-table size.", 1, size);

  err=0;
  switch(type_of(rehash_size)) {
  case t_fixnum:
    if (fix(rehash_size)<=0) err=1;
    break;
  case t_shortfloat:
    if (sf(rehash_size)<=1.0) err=1;
    break;
  case t_longfloat:
    if (lf(rehash_size)<=1.0) err=1;
    break;
  default:
    err=1;
  }
  if (err)
     FEerror("~S is an illegal hash-table rehash-size.",1, rehash_size);

  err=0;
  switch(type_of(rehash_threshold)) {
  case t_fixnum:
    max_ent=fix(rehash_threshold);
    if (max_ent<0 || max_ent>fix(size)) err=1;
    break;
  case t_shortfloat:
    BLOCK_EXCEPTIONS(max_ent=sf(rehash_threshold)*fix(size)+0.5);
    if (sf(rehash_threshold)<0.0 || sf(rehash_threshold)>1.0) err=1;
    break;
  case t_longfloat:
    BLOCK_EXCEPTIONS(max_ent=lf(rehash_threshold)*fix(size)+0.5);
    if (lf(rehash_threshold)<0.0 || lf(rehash_threshold)>1.0) err=1;
    break;
  case t_ratio:
    {
      double d=number_to_double(rehash_threshold);
      max_ent=(fixnum)(d*fix(size)+0.5);
      if (d<0.0 || d>1.0) err=1;
      break;
    }
      
  default:
    err=1;
    break;
  }
  if (err)
     FEerror("~S is an illegal hash-table rehash-threshold.",1,rehash_threshold);
	 
  {
    BEGIN_NO_INTERRUPT;
    h = alloc_object(t_hashtable);
    h->ht.tt=h->ht.ht_test = (short)htt;
    h->ht.ht_size = fix(size);
    h->ht.ht_rhsize = rehash_size;
    h->ht.ht_rhthresh = rehash_threshold;
    h->ht.ht_cache = NULL;
    h->ht.ht_nent = 0;
    h->ht.ht_max_ent = max_ent;
    h->ht.ht_static=staticp==Cnil ? 0 : 1;
    h->ht.ht_self = NULL;
    h->ht.ht_self = h->ht.ht_static ?
      (struct cons *)alloc_contblock(fix(size) * sizeof(struct cons)) :
      (struct cons *)alloc_relblock(fix(size) * sizeof(struct cons));
    for(i = 0;  i < fix(size);  i++) {
      h->ht.ht_self[i].c_cdr = OBJNULL;
      h->ht.ht_self[i].c_car = OBJNULL;
    }
    END_NO_INTERRUPT;
  }
  RETURN1(h);
}

DEFVAR("*DEFAULT-HASH-TABLE-SIZE*",sSAdefault_hash_table_sizeA,SI,make_fixnum(1024),"");
DEFVAR("*DEFAULT-HASH-TABLE-REHASH-SIZE*",sSAdefault_hash_table_rehash_sizeA,SI,make_shortfloat((shortfloat)1.5),"");
DEFVAR("*DEFAULT-HASH-TABLE-REHASH-THRESHOLD*",sSAdefault_hash_table_rehash_thresholdA,SI,make_shortfloat((shortfloat)0.7),"");

object
gcl_make_hash_table(object test) {
  return FFN(fSmake_hash_table_int)(test,
				    sSAdefault_hash_table_sizeA->s.s_dbind,
				    sSAdefault_hash_table_rehash_sizeA->s.s_dbind,
				    sSAdefault_hash_table_rehash_thresholdA->s.s_dbind,
				    Cnil);
}

DEFUN("GCL-MAKE-HASH-TABLE",object,fSgcl_make_hash_table,SI,1,1,NONE,OO,OO,OO,OO,(object test),"") {

  return gcl_make_hash_table(test);

}
  

DEFUN("GETHASH-INT",object,fSgethash_int,SI,2,2,NONE,IO,OO,OO,OO,(object x,object y),"") {
  return (object)gethash(x,y);
}

DEFUN("EXTEND-HASHTABLE",object,fSextent_hashtable,SI,1,1,NONE,IO,OO,OO,OO,(object table),"") {
  extend_hashtable(table);
  return (object)(fixnum)table->ht.ht_size;
}

void
gcl_init_hash() {

  ufixnum i;

  sLeq = make_ordinary("EQ");
  sLeql = make_ordinary("EQL");
  sLequal = make_ordinary("EQUAL");
  sLequalp = make_ordinary("EQUALP");
  sKsize = make_keyword("SIZE");
  sKtest = make_keyword("TEST");
  sKrehash_size = make_keyword("REHASH-SIZE");
  sKrehash_threshold = make_keyword("REHASH-THRESHOLD");
  sKstatic = make_keyword("STATIC");
  
  {
    object x=find_symbol(make_simple_string("MOST-NEGATIVE-FIXNUM"),system_package);
    x=number_negate(x->s.s_dbind);
    for (i=0;i<sizeof(rtb)/sizeof(*rtb);i++) {
      vs_push(x);
      Lrandom();
      rtb[i]=fixint(vs_pop);
    }
  }
}

