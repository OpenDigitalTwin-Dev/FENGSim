/* 
Copyright William Schelter. All rights reserved.
Copyright 2024 Camm Maguire
There is a companion file rsym.c which is used to build
a list of the external symbols in a COFF or A.OUT object file, for
example saved_kcl.  These are loaded into kcl, and the
linking is done directly inside kcl.  This saves a good 
deal of time.   For example a tiny file foo.o with one definition
can be loaded in .04 seconds.  This is much faster than
previously possible in kcl.
The function fasload from unixfasl.c is replaced by the fasload
in this file.
this file is included in unixfasl.c
via #include "../c/sfasl.c" 
*/


/* for testing in standalone manner define STAND
 You may then compile this file cc -g -DSTAND -DDEBUG -I../hn
 a.out /tmp/foo.o /public/gcl/unixport/saved_kcl /public/gcl/unixport/
 will write a /tmp/sfasltest file
 which you can use comp to compare with one produced by ld.
 */

#define IN_SFASL

/*  #ifdef STAND */
/*  #include "config.h" */
/*  #include "gclincl.h" */
/*  #define OUR_ALLOCA alloca */
/*  #include <stdio.h> */
/*  #include "mdefs.h" */

/*  #else */
#include "gclincl.h"
#include "include.h"
#undef S_DATA
/*  #endif */


#if defined(SPECIAL_RSYM) && !defined(USE_DLOPEN)

#include <string.h>

#include "ptable.h"

static int
node_compare(const void *v1,const void *v2) {
  const struct node *a1=v1,*a2=v2;
  
  return strcmp(a1->string,a2->string);

}

static struct node *
find_sym_ptable(const char *name) {

  struct node joe;
  joe.string=name;
  return bsearch(&joe,c_table.ptable,c_table.length,sizeof(joe),node_compare);

}

DEFUN("FIND-SYM-PTABLE",object,fSfind_sym_ptable,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  char c;
  struct node *a;

  check_type_string(&x);

  c=x->st.st_self[VLEN(x)];
  x->st.st_self[VLEN(x)]=0;
  a=find_sym_ptable(x->st.st_self);
  x->st.st_self[VLEN(x)]=c;

  return (object)(a ? a->address : 0);

}

#endif

#ifdef SEPARATE_SFASL_FILE
#include SEPARATE_SFASL_FILE
#else
#error must define SEPARATE_SFASL_FILE
#endif /* SEPARATE_SFASL_FILE */
