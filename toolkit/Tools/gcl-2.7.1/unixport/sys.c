#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "../h/include.h"

#if defined(pre_gcl)
#define libname "libpre_gcl.a"
#elif defined(gcl)
#define libname "libgcl.a"
#elif defined(mod_gcl)
#define libname "libmod_gcl.a"
#elif defined(pcl_gcl)
#define libname "libpcl_gcl.a"
#elif defined(ansi_gcl)
#define libname "libansi_gcl.a"
#else
#define libname ""
#error "No flavor defined"
#endif

#ifndef pre_gcl
static void
ar_init_fn(void (fn)(void),const char *s) {

  struct stat ss;
  
  if (stat(s,&ss)) {

    char *sysd=getenv("GCL_SYSDIR");
    char *d=sysd ? sysd : kcl_self;
    int n=sysd ? strlen(sysd) : dir_name_length(d);

    assert(snprintf(FN1,sizeof(FN1),"ar x %-.*s%s %s",n,d,libname,s)>0);
    assert(!msystem(FN1));

  }
  gcl_init_or_load1(fn,s);
  assert(!unlink(s));
  
}

static void
ar_check_init_fn(void (fn)(void),char *s,object b,char *o) {

   object t;

   for (t=b->s.s_dbind;
	!endp(t) &&
	  type_of(t->c.c_car)==t_string &&
	  strcmp(s,t->c.c_car->st.st_self);t=t->c.c_cdr);
   if (endp(t))
     ar_init_fn(fn,o);

}
#endif

#define proc(init,fn,args...) {extern void init(void);fn(init,##args);}

#define ar_init(a) proc(Mjoin(init_,a),ar_init_fn,#a ".o")
#define ar_check_init(a,b) proc(Mjoin(init_,a),ar_check_init_fn,#a,b,#a ".o")


static void
lsp_init(const char *a,const char *b) {

  char *d,*sysd=getenv("GCL_LSPSYSDIR");
  int n;

  d=sysd ? sysd : kcl_self;
  n=sysd ? strlen(sysd) : dir_name_length(d);
  assert(snprintf(FN1,sizeof(FN1),"%-.*s../%s/%s%s",n,d,a,b,strchr(b,'.') ? "" : ".lsp")>0);
  printf("loading %s\n",FN1);
  fflush(stdout);
  load(FN1);

}

#ifdef pre_gcl
#define init(a,b) lsp_init(a,b)
#define check_init(a,b,c) lsp_init(#a,#b)
#else
#define init(a,b) ar_init(b)
#define check_init(a,b,c) ar_check_init(b,c)
#endif


