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
	Random numbers
*/

#include <time.h>
#include <string.h>

#include "include.h"
#include "num_include.h"

#ifdef AOSVS

#endif

static object
rando(object x, object rs) {

  enum type tx;
  object base,out,z;
  fixnum fbase;
  double d;
  
  tx = type_of(x);
  if (number_compare(x, small_fixnum(0)) != 1)
    FEwrong_type_argument(TSpositive_number, x);
  
  if (tx==t_bignum) {
    out=new_bignum();
    base=x;
    fbase=-1;
  } else {
    out=big_fixnum1;
    fbase=tx==t_fixnum ? fix(x) : MOST_POSITIVE_FIX;
    mpz_set_si(MP(big_fixnum2),fbase);
    base=big_fixnum2;
  }
  
  mpz_urandomm(MP(out),&rs->rnd.rnd_state,MP(base));
  
  switch (tx) {
    
  case t_fixnum:
    return make_fixnum(mpz_get_si(MP(out)));
  case t_bignum:
    return normalize_big(out);
  case t_shortfloat: case t_longfloat:
    d=mpz_get_d(MP(out));
    d/=(double)fbase;
    z=alloc_object(tx);
    BLOCK_EXCEPTIONS(if (tx==t_shortfloat) sf(z)=sf(x)*d; else lf(z)=lf(x)*d);
    return z;
  default:
    FEerror("~S is not an integer nor a floating-point number.", 1, x);
    return(Cnil);
  }
}


#ifdef UNIX
#define RS_DEF_INIT time(0)
#else
#define RS_DEF_INIT 0
#endif

#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
extern void * (*gcl_gmp_allocfun) (size_t);
static void * (*old_gcl_gmp_allocfun) (size_t);
static void * trap_result;
static size_t trap_size;

static void *
trap_gcl_gmp_allocfun(size_t size){

  size+=size%MP_LIMB_SIZE;
  if (trap_size)
    return old_gcl_gmp_allocfun(size);
  else {
    trap_size=size/MP_LIMB_SIZE;
    trap_result=old_gcl_gmp_allocfun(size);
    return trap_result;
  }

}
#endif

void
reinit_gmp() {

#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
  Mersenne_Twister_Generator_Noseed.b=__gmp_randget_mt;
  Mersenne_Twister_Generator_Noseed.c=__gmp_randclear_mt;
  Mersenne_Twister_Generator_Noseed.d=__gmp_randiset_mt;
#endif

}

void
init_gmp_rnd_state(__gmp_randstate_struct *x) {

  static int n;

  bzero(x,sizeof(*x));
  
#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
/*   if (!trap_size) { */
  old_gcl_gmp_allocfun=gcl_gmp_allocfun;
  gcl_gmp_allocfun=trap_gcl_gmp_allocfun;
/*   } */
#endif
  gmp_randinit_default(x);
#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
  if (!n) {

    if (x->_mp_seed->_mp_d!=trap_result)
      FEerror("Unknown pointer in rnd_state!",0);
/* #ifndef __hppa__ /\*FIXME*\/ */
/*     if (((gmp_randfnptr_t *)x->_mp_algdata._mp_lc)->b!=Mersenne_Twister_Generator_Noseed.b || */
/* 	((gmp_randfnptr_t *)x->_mp_algdata._mp_lc)->c!=Mersenne_Twister_Generator_Noseed.c || */
/* 	((gmp_randfnptr_t *)x->_mp_algdata._mp_lc)->d!=Mersenne_Twister_Generator_Noseed.d) */
/*       FEerror("Unknown pointer data in rnd_state!",0); */
/* #endif */

    n=1;

  }
  gcl_gmp_allocfun=old_gcl_gmp_allocfun;
  x->_mp_seed->_mp_alloc=x->_mp_seed->_mp_size=trap_size;
#endif
    

}



static object
make_random_state(object rs) {

  object z;
  
  if (rs==Cnil)
    rs=symbol_value(Vrandom_state);
  
  if (rs!=Ct && type_of(rs) != t_random) {
    FEwrong_type_argument(sLrandom_state, rs);
    return(Cnil);
  }
  
  z = alloc_object(t_random);
  init_gmp_rnd_state(&z->rnd.rnd_state);

    
  if (rs == Ct) 
    gmp_randseed_ui(&z->rnd.rnd_state,RS_DEF_INIT);
  else
    memcpy(z->rnd.rnd_state._mp_seed->_mp_d,rs->rnd.rnd_state._mp_seed->_mp_d,
	   rs->rnd.rnd_state._mp_seed->_mp_alloc*sizeof(*z->rnd.rnd_state._mp_seed->_mp_d));
  
#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
  z->rnd.rnd_state._mp_algdata._mp_lc=&Mersenne_Twister_Generator_Noseed;
#endif
  return(z);

}

LFD(Lrandom)(void)
{
	int j;
        object x;
	
	j = vs_top - vs_base;
	if (j == 1)
		vs_push(symbol_value(Vrandom_state));
	check_arg(2);
	check_type_random_state(&vs_base[1]);
	x = rando(vs_base[0], vs_base[1]);
	vs_top = vs_base;
	vs_push(x);
}

LFD(Lmake_random_state)(void)
{
	int j;
	object x;

	j = vs_top - vs_base;
	if (j == 0)
		vs_push(Cnil);
	check_arg(1);
	x = make_random_state(vs_head);
	vs_top = vs_base;
	vs_push(x);
}

LFD(Lrandom_state_p)(void)
{
	check_arg(1);
	if (type_of(vs_pop) == t_random)
		vs_push(Ct);
	else
		vs_push(Cnil);
}

void
gcl_init_num_rand(void)
{
        Vrandom_state = make_special("*RANDOM-STATE*",
				     make_random_state(Ct));

	make_function("RANDOM", Lrandom);
	make_function("MAKE-RANDOM-STATE", Lmake_random_state);
	make_function("RANDOM-STATE-P", Lrandom_state_p);
}
