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
	main.c
	IMPLEMENTATION-DEPENDENT
*/

#include <unistd.h>
#include <fcntl.h>

static void init_main(void);
static void initlisp(void);
static int  multiply_stacks(int);
void install_segmentation_catcher(void);

#ifdef KCLOVM
#include <ovm/ovm.h>
void change_contexts();
int ovm_process_created; 
void initialize_process();
#endif

#define EXTER
#define INLINE

#include "include.h"
#include <signal.h>
#include "page.h"

bool saving_system=FALSE;

#ifdef BSD
#include <sys/time.h>
#ifndef SGI
#include <sys/resource.h>
#endif
#endif

#ifdef AOSVS

#endif

#ifdef _WIN32
#include <fcntl.h>
#endif

#define LISP_IMPLEMENTATION_VERSION "April 1994"

char *system_directory;

#define EXTRA_BUFSIZE 8
char stdin_buf[BUFSIZ + EXTRA_BUFSIZE];
char stdout_buf[BUFSIZ + EXTRA_BUFSIZE];
char stderr_buf[BUFSIZ + EXTRA_BUFSIZE];

#include "stacks.h"

int debug;			/* debug switch */
int raw_image = TRUE;		/* raw or saved image -- CYGWIN will only place this in .data and not in .bss if initialized to non-zero */
bool GBC_enable=FALSE;

long real_maxpage;
object sSAlisp_maxpagesA;

object siClisp_pagesize;

object sStop_level;


object sSAmultiply_stacksA;
int stack_multiple=1;
static object stack_space;

#ifdef _WIN32
unsigned int _dbegin = 0x10100000;
#endif
#ifdef __CYGWIN__
unsigned long _dbegin = 0;
#endif

#ifdef SGC
int sgc_enabled;
#endif
void install_segmentation_catcher(void);

#ifdef NEED_STACK_CHK_GUARD

unsigned long __stack_chk_guard=0;

static unsigned long
random_ulong() {

  object y;

  vs_top=vs_base;
  vs_push(Ct);
  Lmake_random_state();
  y=vs_pop;
  vs_push(number_negate(find_symbol(make_simple_string("MOST-NEGATIVE-FIXNUM"),system_package)->s.s_dbind));
  vs_push(y);
  Lrandom();

  return fixint(vs_pop);

}
#endif

int cstack_dir=0;

int
get_cstack_dir(VOL fixnum j) {
  static fixnum n;
  fixnum q=n;
  n=1-n;
  return q ? ((fixnum)&j<j ? -1 : 1) : get_cstack_dir((fixnum)&j);
}

void
wipe_stack(VOL void *l) {

  if (cstack_dir==-1) {
    if (l>(void *)&l) bzero((void *)&l,l-(void *)&l);
  } else {
     l+=sizeof(l);
     if ((void *)&l>l) bzero((void *)l,(void *)&l-l);
   }
}

void
clear_c_stack(VOL unsigned n) {

  void *v=OBJNULL;
  alloca(n);
  wipe_stack(&v);

}

static fixnum log_maxpage_bound=sizeof(fixnum)*8-1;
static fixnum mem_bound=sizeof(fixnum)*8-1;

int
mbrk(void *v) {

  ufixnum uv=(ufixnum)v,uc=(ufixnum)sbrk(0),ux,um;
  fixnum m=((1UL<<(sizeof(fixnum)*8-1))-1);

  if (uv<uc) {
    um=uv;
    ux=uc;
  } else {
    um=uc;
    ux=uv;
  }

  if (((fixnum)(ux-um))<0)
    return mbrk((void *)uc+(uv<uc ? -m : m)) || mbrk(v);

  return uc==(ufixnum)sbrk(uv-uc) ? 0 : -1;

}
    
static char *
next_line(int l,ufixnum *s) {

  size_t n;
  ssize_t i;
  char *p;

  if (*s && (n=strlen(FN1))) {
    memmove(FN1,FN1+n+1,sizeof(FN1)-(n+1));
    *s-=n+1;
  }

  massert(*s<sizeof(FN1));
  massert((i=read(l,FN1+*s,sizeof(FN1)-*s))>=0);
  *s+=i;

  if (!(p=memchr(FN1,'\n',*s)))
    return NULL;

  *p=0;
  return FN1;

}

static void
parse_meminfo_decimal(char *c,char *h,ufixnum *t) {

  ufixnum n=strlen(h);

  if (memcmp(h,c,n))
    return;

  massert(sscanf(c+n,"%ld kB",t)==1);
  *t<<=10;
  *t>>=PAGEWIDTH;

}

static void
parse_proc_meminfo(ufixnum *t,ufixnum *f,ufixnum *st,ufixnum *sf) {

  int l;
  ufixnum r=0;
  char *s="/proc/meminfo",*c;

  *t=*f=*st=*sf=0;

  massert((l=open(s,O_RDONLY))>=0);

  for (;(c=next_line(l,&r));) {
    parse_meminfo_decimal(c,"MemTotal:",t);
    parse_meminfo_decimal(c,"MemFree:",f);
    parse_meminfo_decimal(c,"SwapTotal:",st);
    parse_meminfo_decimal(c,"SwapFree:",sf);
  }

  massert(!close(l));

}


#if defined(__CYGWIN__)||defined(__MINGW32__)

#include <windows.h>

static ufixnum
get_phys_pages_no_malloc(char n,char ramp) {

  MEMORYSTATUS m;

  m.dwLength=sizeof(m);
  GlobalMemoryStatus(&m);
  return m.dwTotalPhys>>PAGEWIDTH;

}

#elif defined (DARWIN)

#include <sys/sysctl.h>

static ufixnum
get_phys_pages_no_malloc(char n,char ramp) {

  uint64_t s;
  size_t z=sizeof(s);
  int m[2]={CTL_HW,HW_MEMSIZE};

  if (sysctl(m,2,&s,&z,NULL,0)==0)
    return s>>PAGEWIDTH;

  return 0;

}

#elif defined(__sun__)

static ufixnum
get_phys_pages_no_malloc(char n,char ramp) {

  return sysconf(_SC_PHYS_PAGES);

}

#elif defined(FREEBSD)

#include <sys/types.h>
#include <sys/sysctl.h>

static ufixnum
get_phys_pages_no_malloc(char n,char ramp) {

  size_t i,len=sizeof(i);

  return (sysctlbyname("hw.physmem",&i,&len,NULL,0) ? 0 : i)>>PAGEWIDTH;

}

#else /*Linux*/

static ufixnum
get_phys_pages_no_malloc(char freep,char ramp) {

  ufixnum t,f,st,sf;

  parse_proc_meminfo(&t,&f,&st,&sf);

  return (freep ? (ramp ? f : f+sf) : (ramp ? t : t+st));

}


#endif

static ufixnum
get_phys_pages1(char freep,char ramp) {

  ufixnum p=get_phys_pages_no_malloc(freep,ramp);
#if defined(__GNU__)/*FIXME -- no mmap overcommit as of yet*/
  p>>=1;
#endif
  return p;

}

static int
acceptable_log_maxpage_bound(ufixnum l) {

  ufixnum max=(sizeof(fixnum)<<3)-1;

  return l>max ? 0 :
    (l==max ? 1 :
     (void *)heap_end-data_start+CEI(rb_pointer-rb_begin(),PAGESIZE) < (1UL<<(l+1)));
}

static void
get_gc_environ(void) {

  const char *e;

  mem_multiple=1.0;
  if ((e=getenv("GCL_MEM_MULTIPLE"))) {
    massert(sscanf(e,"%lf",&mem_multiple)==1);
    massert(mem_multiple>=0.0);
  }

  mem_bound=sizeof(fixnum)*8-1;
  if ((e=getenv("GCL_MEM_BOUND"))) {
    ufixnum f;
    massert(sscanf(e,"%lud",&f)==1);
    if (acceptable_log_maxpage_bound(f)) {
      mem_bound=f;
      mem_multiple=1.0;
    }
  }

  gc_alloc_min=0.05;
  if ((e=getenv("GCL_GC_ALLOC_MIN"))) {
    massert(sscanf(e,"%lf",&gc_alloc_min)==1);
    massert(gc_alloc_min>=0.0);
  }

  gc_page_min=0.5;
  if ((e=getenv("GCL_GC_PAGE_MIN"))||(e=getenv("GCL_GC_PAGE_THRESH"))) {/*legacy support*/
    massert(sscanf(e,"%lf",&gc_page_min)==1);
    massert(gc_page_min>=0.0);
  }

  gc_page_max=0.75;
  if ((e=getenv("GCL_GC_PAGE_MAX"))) {
    massert(sscanf(e,"%lf",&gc_page_max)==1);
    massert(gc_page_max>=0.0);
  }

  multiprocess_memory_pool=getenv("GCL_MULTIPROCESS_MEMORY_POOL");
  if (multiprocess_memory_pool &&
      (*multiprocess_memory_pool=='t' || *multiprocess_memory_pool=='T'))/*GCL 2.6 compatability*/
    multiprocess_memory_pool=getenv("HOME");

  wait_on_abort=0;
  if ((e=getenv("GCL_WAIT_ON_ABORT")))
    massert(sscanf(e,"%lu",&wait_on_abort)==1);

}

static void
setup_maxpages(double scale) {

  void *beg=data_start ? data_start : sbrk(0);
  ufixnum maxpages=real_maxpage-page(beg),npages,i;

  for (npages=0,i=t_start;i<t_other;i++)
    npages+=tm_table[i].tm_maxpage=tm_table[i].tm_npage;

  massert(scale*maxpages>=npages);

  maxpages*=scale;
  phys_pages*=scale;
  real_maxpage=maxpages+page(beg);

  resv_pages=available_pages=0;
  available_pages=check_avail_pages();

  resv_pages=available_pages/100;
  available_pages-=resv_pages;

}


static void *
next_shared_lib_map_no_malloc(void)  {

#if !defined(DARWIN) && !defined(__CYGWIN__) && !defined(__MINGW32__) && !defined(__MINGW64__)/*FIXME*/

  char *c,rwx[4];
  ufixnum a,e,s=(ufixnum)sbrk(0),r;
  int l;

  massert((l=open("/proc/self/maps",O_RDONLY))!=-1);

  for (a=r=0;(a<=s || !memcmp(rwx,"---",3)) && (c=next_line(l,&r));)
    sscanf(c,"%lx-%lx %3s",&a,&e,rwx);

  massert(!close(l));

  return (void *)(c ? a : -1);

#else

  return (void *)-1;

#endif
}

static void *stack_map_base=(void *)-1;
void *shared_lib_start=(void *)-1;

static int
set_real_maxpage(void *beg) {

  void *end,*cp;
  ufixnum mp,sz;

  end=(void *)ROUNDDN((void *)-1,PAGESIZE);
  mp=page(end-beg);

  mp=ufmin(mp,get_phys_pages1(0,0));

  sz=ufmin(mem_bound,log_maxpage_bound);
  sz=(1UL<<sz)+((1UL<<sz)-1);
  mp=ufmin(mp,page(sz));

#if defined(LOW_IM_FIX)
  cp=(void *)(ufixnum)LOW_IM_FIX;
#elif defined(IM_FIX_BASE)
  cp=(void *)IM_FIX_BASE;
#else
  cp=(void *)-1;
#endif
  cp=cp<beg ? (void *)-1 : cp;
  mp=ufmin(mp,page(cp-beg));

  cp=alloca(1);
  cp=cp<stack_map_base ? cp : stack_map_base;
  cp=cp<beg ? (void *)-1 : cp;
  mp=ufmin(mp,page(cp-beg));

  cp=shared_lib_start=next_shared_lib_map_no_malloc();
  cp=cp<beg ? (void *)-1 : cp;
  mp=ufmin(mp,page(cp-beg));

  real_maxpage=mp+page(beg);

  return 0;

}

int
update_real_maxpage(void) {

  void *beg;

#ifdef __MINGW32__
  static fixnum n;

  if (!n) {
    init_shared_memory();
    n=1;
  }
#endif

  get_gc_environ();

  massert(beg=data_start ? data_start : sbrk(0));
  set_real_maxpage(beg);

  phys_pages=ufmin(get_phys_pages1(0,1)+page(beg),real_maxpage)-page(beg);

  setup_maxpages(mem_multiple);

  return 0;

}

static int
minimize_image(void) {

  fixnum i;
  
  empty_relblock();
  nrbpage=0;
  resize_hole(0,t_relocatable,0);

#if defined(BSD) || defined(ATT)  
  mbrk(core_end=heap_end);
#endif
  
  cbgbccount = tm_table[t_contiguous].tm_adjgbccnt = tm_table[t_contiguous].tm_opt_maxpage = 0;
  rbgbccount = tm_table[t_relocatable].tm_adjgbccnt = tm_table[t_relocatable].tm_opt_maxpage = 0;
  for (i = 0;  i < (int)t_end;  i++)
    tm_table[i].tm_gbccount = tm_table[i].tm_adjgbccnt = tm_table[i].tm_opt_maxpage = 0;
  
  return 0;
  
}

DEFUN("SET-LOG-MAXPAGE-BOUND",object,fSset_log_maxpage_bound,SI,1,1,NONE,II,OO,OO,OO,(fixnum l),"") {

  GBC(t_relocatable);
  if (acceptable_log_maxpage_bound(l)) {
    unsetenv("GCL_MEM_MULTIPLE");/*invoking this function overrides mem_multiple*/
    unsetenv("GCL_MEM_BOUND");/*invoking this function overrides mem_multiple*/
    minimize_image();
    log_maxpage_bound=l;
    update_real_maxpage();
    maybe_set_hole_from_maxpages();
  }

  return (object)log_maxpage_bound;

}

#include <dlfcn.h>

size_t
dir_name_length(const char *s) {
  size_t m;
  const char *z;
  for (m=strlen(s),z=kcl_self+m;z>s && z[-1]!='/';z--,m--);
  return m;
}

int initializing_boot=0;

void
init_boot(void) {

  char *sysd=getenv("GCL_SYSDIR"),*d=sysd ? sysd : kcl_self;
#ifndef __CYGWIN__
  void *v,*q;
#endif
  char *z,*s="libboot.so";
  size_t m=sysd ? strlen(sysd) : dir_name_length(kcl_self),n=m+strlen(s)+1;
  object omp=sSAoptimize_maximum_pagesA->s.s_dbind;

  sSAoptimize_maximum_pagesA->s.s_dbind=Cnil;
  z=alloca(n);
  snprintf(z,n,"%-*.*s%s",(int)m,(int)m,d,s);
#ifndef __CYGWIN__
  if (!(v=dlopen(z,RTLD_LAZY|RTLD_GLOBAL)))
    printf("%s\n",dlerror());
  if (!(q=dlsym(v,"gcl_init_boot")))
    printf("%s\n",dlerror());
#endif
  initializing_boot=1;
#ifdef __CYGWIN__
  {
    extern void gcl_init_boot(void);
    gcl_init_boot();
  }
#else
  ((void (*)())q)();
#endif
  initializing_boot=0;
  sSAoptimize_maximum_pagesA->s.s_dbind=omp;

}

int in_pre_gcl=0;
object def_env1[2]={(object)1,Cnil},*def_env=def_env1+1;
object src_env1[2]={(object)1,Cnil},*src_env=src_env1+1;

#ifdef HAVE_MPROTECT
#include <sys/mman.h>
int
gcl_mprotect(void *v,unsigned long l,int p) {

  int i;
  char b[80];

  if ((i=mprotect(v,l,p))) {
    snprintf(b,sizeof(b),"mprotect failure: %p %lu %d\b",v,l,p);
    perror(b);
  }

  return i;

}
#endif

DEFVAR("*CODE-BLOCK-RESERVE*",sSAcode_block_reserveA,SI,Cnil,"");

#define HAVE_GCL_CLEANUP

void
gcl_cleanup(int gc) {

  if (getenv("GCL_WAIT"))
    sleep(30);

#if defined(USE_CLEANUP)
  {extern void _cleanup(void);_cleanup();}
#endif

#ifdef USE_GPROF
  gprof_cleanup();
#endif

  if (gc) {

    saving_system=TRUE;
    GBC(t_other);
    saving_system=FALSE;

    minimize_image();

    raw_image=FALSE;
    cs_org=0;
    msbrk_end();

  }

  close_pool();

}

/*gcc boolean expression tail position bug*/
static char *stack_to_be_allocated;

int
stack_ret(char *s,unsigned long size) {
  int r,i;
  for (i=r=0;i<size;i++)
    r^=((unsigned char)s[i])|((ufixnum)(s+i));
  return r;
}

int
get_stack_to_be_allocated(unsigned long size) {
  stack_to_be_allocated=alloca(size);
  memset(stack_to_be_allocated,0,size);
  return stack_ret(stack_to_be_allocated,size);
}

DEFUN("EQUAL-TAIL-RECURSION-CHECK",object,fSequal_tail_recursion_check,SI,1,1,NONE,II,OO,OO,OO,(fixnum s),"") {
  object x0=make_list(s/sizeof(object)),x1=make_list(s/sizeof(object));
  char *w;
  get_stack_to_be_allocated(s);
  fLequal(x0,x1);
  for (w=stack_to_be_allocated;w<stack_to_be_allocated+s && !*w;w++);
  RETURN1((object)(w-stack_to_be_allocated));
}

#if !defined(DARWIN)&&!defined(__MINGW32__)

static int
mbin(const char *s,char *o) {

  struct stat ss;

  if (!stat(s,&ss) && (ss.st_mode&S_IFMT)==S_IFREG && !access(s,R_OK|X_OK)) {
    massert(realpath(s,o));
    return 1;
  }

  return 0;

}

static int
which(const char *n,char *o) {

  char *s;

  if (strchr(n,'/'))
    return mbin(n,o);

  massert(snprintf(FN1,sizeof(FN1),"%s",getenv("PATH"))>1);
  for (s=NULL;(s=strtok(s ? NULL : FN1,":"));) {

    massert(snprintf(FN2,sizeof(FN2),"%s/%s",s,n));
    if (mbin(FN2,o))
      return 1;

  }

  return 0;

}

#endif

DEFUN("KCL-SELF",object,fSkcl_self,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  return make_simple_string(kcl_self);

}

int
main(int argc, char **argv, char **envp) {

  GET_FULL_PATH_SELF(kcl_self);
  *argv=kcl_self;

  vs_top = vs_base = vs_org;
  ihs_top = ihs_org-1;
  bds_top = bds_org-1;
  frs_top = frs_org-1;

#include "cstack.h"

  gcl_init_alloc(alloca(1));

  setbuf(stdin, stdin_buf); 
  setbuf(stdout, stdout_buf);
  setbuf(stderr, stderr_buf);
#ifdef _WIN32
  _fmode = _O_BINARY;
  _setmode( _fileno( stdin ), _O_BINARY );
  _setmode( _fileno( stdout ), _O_BINARY );
  _setmode( _fileno( stderr ), _O_BINARY );
#endif
  ARGC = argc;
  ARGV = argv;
  ENVP = envp;
  
  if (raw_image) {

    printf("GCL (GNU Common Lisp)  %s  %ld pages\n",LISP_IMPLEMENTATION_VERSION,real_maxpage);
    fflush(stdout);
    
    initlisp();
    ihs_top++;lex_new();/*FIXME*/
    
    GBC_enable = TRUE;
    
    gcl_init_init();
  
    sLApackageA->s.s_dbind = user_package;
    
  } else {

    terminal_io->sm.sm_object0->sm.sm_fp = stdin;
    terminal_io->sm.sm_object1->sm.sm_fp = stdout;
    standard_error->sm.sm_fp = stderr;

    gcl_init_big1();
#ifdef USE_READLINE
    gcl_init_readline_function();
#endif
#ifdef NEED_STACK_CHK_GUARD
    __stack_chk_guard=random_ulong();/*Cannot be safely set inside a function which returns*/
#endif

    if (in_pre_gcl) init_boot();

  }

#ifdef _WIN32
  detect_wine();
#endif

  sSAlisp_maxpagesA->s.s_dbind = make_fixnum(real_maxpage);

  ihs_push(Cnil);
  lex_new();
  vs_base = vs_top;
  
  interrupt_enable = TRUE;
  install_default_signals();
    
  do 
    super_funcall(sStop_level);
  while (type_of(sSAmultiply_stacksA->s.s_dbind)==t_fixnum && multiply_stacks(fix(sSAmultiply_stacksA->s.s_dbind)));
    
  return 0;

}

/* catch certain signals */
void install_segmentation_catcher(void)
{
  unblock_signals(SIGSEGV,SIGSEGV);
  unblock_signals(SIGBUS,SIGBUS);
  (void) gcl_signal(SIGSEGV,segmentation_catcher);
  (void) gcl_signal(SIGBUS,segmentation_catcher);
}

void
do_gcl_abort(void) {
  if (wait_on_abort)
    sleep(wait_on_abort);
  gcl_cleanup(0);
  abort();
}

int catch_fatal=1;
void
error(char *s)
{
        if (catch_fatal>0 && interrupt_enable )
            {catch_fatal = -1;
#ifdef SGC
    if (sgc_enabled)
      sgc_quit();
    if (sgc_enabled==0)
#endif
      install_segmentation_catcher();
    {
      FEerror("Caught fatal error [memory may be damaged]: ~a",1,str(s));
    }
  }
  printf("\nUnrecoverable error: %s.\n", s);
  fflush(stdout);
#ifdef UNIX
  gcl_abort();
#endif
}

static void
initlisp(void) {

        void *v=&v;

	if (NULL_OR_ON_C_STACK(v) == 0
#if defined(IM_FIX_BASE)
             || NULL_OR_ON_C_STACK(IM_FIX_BASE) == 0
             || NULL_OR_ON_C_STACK((IM_FIX_BASE|IM_FIX_LIM)) == 0
#endif
	    /* || NULL_OR_ON_C_STACK(vv) */
	    || NULL_OR_ON_C_STACK(pagetoinfo(first_data_page))
	    || NULL_OR_ON_C_STACK(core_end-1)) {
	  /* check person has correct definition of above */
	  emsg("%p %d "
#if defined(IM_FIX_BASE)
		  "%p %d %p %d "
#endif
		  "%p %d %p %d\n",
		  v,NULL_OR_ON_C_STACK(v),
#if defined(IM_FIX_BASE)
		  (void *)IM_FIX_BASE,NULL_OR_ON_C_STACK(IM_FIX_BASE),
		  (void *)(IM_FIX_BASE|IM_FIX_LIM),NULL_OR_ON_C_STACK(IM_FIX_BASE|IM_FIX_LIM),
#endif
		  pagetoinfo(first_data_page),NULL_OR_ON_C_STACK(pagetoinfo(first_data_page)),
		  core_end-1,NULL_OR_ON_C_STACK(core_end-1));
	  error("NULL_OR_ON_C_STACK macro invalid");
	}

	Cnil->fw=0;
	set_type_of(Cnil,t_symbol);
 	Cnil->c.c_cdr=Cnil;
 	Cnil->s.s_dbind = Cnil;
	Cnil->s.s_sfdef = NOT_SPECIAL;
	Cnil->s.s_name = make_simple_string("NIL");
 	Cnil->s.s_gfdef = OBJNULL;
 	Cnil->s.s_plist = Cnil;
 	Cnil->s.s_hpack = Cnil;
 	Cnil->s.s_stype = (short)stp_constant;
 	Cnil->s.s_mflag = FALSE;
	Cnil->s.s_hash = ihash_equal1(Cnil,0);
	
	Ct->fw=0;
	set_type_of(Ct,t_symbol);
 	Ct->s.s_dbind = Ct;
	Ct->s.tt=1;
 	Ct->s.s_sfdef = NOT_SPECIAL;
	Ct->s.s_name = make_simple_string("T");
 	Ct->s.s_gfdef = OBJNULL;
 	Ct->s.s_plist = Cnil;
 	Ct->s.s_hpack = Cnil;
 	Ct->s.s_stype = (short)stp_constant;
 	Ct->s.s_mflag = FALSE;
	Ct->s.s_hash = ihash_equal1(Ct,0);
	
	gcl_init_symbol();

	gcl_init_package();

	Cnil->s.s_hpack = lisp_package;
	import(Cnil, lisp_package);
	export(Cnil, lisp_package);

	Ct->s.s_hpack = lisp_package;
	import(Ct, lisp_package);
	export(Ct, lisp_package);

	sLlambda = make_ordinary("LAMBDA");
	sSlambda_block = make_si_ordinary("LAMBDA-BLOCK");
	sSlambda_closure = make_si_ordinary("LAMBDA-CLOSURE");
	sSlambda_block_closure = make_si_ordinary("LAMBDA-BLOCK-CLOSURE");
	sLspecial = make_ordinary("SPECIAL");

	NewInit();
	init_boot();

	gcl_init_typespec();
	gcl_init_number();
	gcl_init_character();

	gcl_init_read();
	gcl_init_bind();
	gcl_init_pathname();
	gcl_init_print();
	gcl_init_GBC();

	gcl_init_unixfasl();
	gcl_init_unixsys();
	gcl_init_unixsave();

	gcl_init_alloc_function();
	gcl_init_array_function();
	gcl_init_character_function();
	gcl_init_file_function();
	gcl_init_list_function();
	gcl_init_package_function();
	gcl_init_pathname_function();
	gcl_init_predicate_function();
	gcl_init_print_function();
	gcl_init_read_function();
	gcl_init_sequence_function();
#if  defined(KCLOVM) || defined(RUN_PROCESS)
	gcl_init_socket_function();
#endif	
	gcl_init_structure_function();
	gcl_init_string_function();
	gcl_init_symbol_function();
	gcl_init_typespec_function();
	gcl_init_hash();
	gcl_init_cfun();

	gcl_init_unixfsys();
	gcl_init_unixtime();
	gcl_init_eval();
	gcl_init_lex();
	gcl_init_prog();
	gcl_init_catch();
	gcl_init_block();
        gcl_init_macros();
	gcl_init_conditional();
	gcl_init_reference();
	gcl_init_assignment();
	gcl_init_multival();
	gcl_init_error();
	gcl_init_let();
	gcl_init_mapfun();
	gcl_init_iteration();
	gcl_init_toplevel();

	gcl_init_cmpaux();

	init_main();

	gcl_init_format();
	gcl_init_links();

	gcl_init_fat_string();
	gcl_init_sfasl();
#ifdef CMAC
	gcl_init_cmac();
#endif	
#ifdef USE_READLINE
	gcl_init_readline();
#endif
	gcl_init_safety();

}
object
vs_overflow(void)
{
	if (vs_limit > vs_org + stack_multiple *  VSSIZE)
		error("value stack overflow");
	vs_limit += STACK_OVER*VSGETA;
	FEerror("Value stack overflow.", 0);
	return Cnil;
}

void
bds_overflow(void) {
	--bds_top;
	if (bds_limit > bds_org + stack_multiple *  BDSSIZE) {
            error("bind stack overflow");
        }
	bds_limit += STACK_OVER  *BDSGETA;
	FEerror("Bind stack overflow.", 0);
}

void
frs_overflow(void) {
	--frs_top;
	if (frs_limit > frs_org + stack_multiple *  FRSSIZE)
		error("frame stack overflow");
	frs_limit += STACK_OVER* FRSGETA;
	FEerror("Frame stack overflow.", 0);
}

void
ihs_overflow(void) {
	--ihs_top;
	if (ihs_limit > ihs_org + stack_multiple *  IHSSIZE)
		error("invocation history stack overflow");
	ihs_limit += STACK_OVER*IHSGETA;
	FEerror("Invocation history stack overflow.", 0);
}

void
segmentation_catcher(int i, long code, void *scp, char *addr) {
#if !defined(_WIN32) && !defined(__CYGWIN__)
  void *faddr;
  faddr=GET_FAULT_ADDR(sig,code,scp,addr); 

  if ((cstack_dir==-1 && faddr < (void *)cs_limit && (void *)cs_limit-faddr <= PAGESIZE) ||
      (cstack_dir==1 && faddr > (void *)cs_limit && faddr-(void *)cs_limit <= PAGESIZE))
    FEerror("Control stack overflow.",0); /*FIXME -- provide getrlimit here.*/
  else 
    printf("Segmentation violation: c stack ok:signalling error");
#endif
  error("Segmentation violation.");
}

DEFUN("BYE",object,fSbye,SI,0,1,NONE,OI,OO,OO,OO,(fixnum exit_code,...),"") {

  if (!INIT_NARGS(0)) exit_code=0;

#ifdef UNIX
  exit(exit_code);
#else
  RETURN(1,int,exit_code, 0); 
#endif

}

DEFUN("QUIT",object,fSquit,SI,0,1,NONE,OI,OO,OO,OO,(fixnum exitc),"") {
  return FFN(fSbye)(exitc); 
}
 

static void
FFN(siLargc)(void) {
  check_arg(0);
  vs_push(make_fixnum(ARGC));
}

static void
FFN(siLargv)(void) {
  int i=0;
  
  check_arg(1);
  if (type_of(vs_base[0]) != t_fixnum ||
      (i = fix(vs_base[0])) < 0 ||
      i >= ARGC)
    FEerror("Illegal argument index: ~S.", 1, vs_base[0]);
  vs_base[0] = make_simple_string(ARGV[i]);

}

static void
FFN(siLgetenv)(void) {

  char name[256];
  int i;
  char *value;
  extern char *getenv(const char *);
  
  check_arg(1);
  check_type_string(&vs_base[0]);
  if (VLEN(vs_base[0]) >= 256)
    FEerror("Too long name: ~S.", 1, vs_base[0]);
  for (i = 0;  i < VLEN(vs_base[0]);  i++)
    name[i] = vs_base[0]->st.st_self[i];
  name[i] = '\0';
  if ((value = getenv(name)) != NULL) {
#ifdef _WIN32
    fix_filename(NULL, value);
#endif    
    vs_base[0] = make_simple_string(value);
#ifdef FREE_GETENV_RESULT
    free(value);
    
#endif		
    }
  else
    vs_base[0] = Cnil;

}

object *vs_marker;

static void
FFN(siLmark_vs)(void) {
  check_arg(0);
  vs_marker = vs_base;
  vs_base[0] = Cnil;
}

static void
FFN(siLcheck_vs)(void) {
  check_arg(0);
  if (vs_base != vs_marker)
    FEerror("Value stack is flawed.", 0);
  vs_base[0] = Cnil;
}

DEFUN("CATCH-FATAL",object,fScatch_fatal,SI,1,1,NONE,OI,OO,OO,OO,(fixnum i),"") {
  catch_fatal=i;
  return Cnil;
}

DEFUN("RESET-STACK-LIMITS",object,fSreset_stack_limits,SI,0,1,NONE,OO,OO,OO,OO,(object cs_org_reset,...),"") {

  if (!INIT_NARGS(0)) cs_org_reset=Cnil;

  if(catch_fatal <0) catch_fatal=1;
#ifdef SGC	
  {extern int fault_count ; fault_count = 0;}
#endif 
  if (vs_top < vs_org + stack_multiple *  VSSIZE)
    vs_limit = vs_org + stack_multiple *  VSSIZE;
  else
    error("can't reset vs_limit");
  if (bds_top < bds_org + stack_multiple *  BDSSIZE)
    bds_limit = bds_org + stack_multiple *  BDSSIZE;
  else
    error("can't reset bds_limit");
  if (frs_top < frs_org + stack_multiple *  FRSSIZE)
    frs_limit = frs_org + stack_multiple *  FRSSIZE;
  else
    error("can't reset frs_limit");
  if (ihs_top < ihs_org + stack_multiple *  IHSSIZE)
    ihs_limit = ihs_org + stack_multiple *  IHSSIZE;
  else
    error("can't reset ihs_limit");

  if (cs_org_reset!=Cnil) {

    cs_org=alloca(1);

#ifdef __ia64__
    {
      extern void * GC_save_regs_in_stack();
      cs_org2=GC_save_regs_in_stack();
    }
#endif
    /* reset_cstack_limit(i); */

  }

  RETURN1(Cnil);

}

#define COPYSTACK(org,p,typ,lim,top,geta,size) \
  {unsigned long topl=top-org;\
   bcopy(org,p,(lim-org)*sizeof(typ));\
   org=p;\
   top=org+topl;\
   lim=org+stack_multiple*size;\
   p=lim+(STACK_OVER+1)*geta;\
   }

static int
multiply_stacks(int m) {  
  void *p;
  int vs,bd,frs,ihs;
  stack_multiple=stack_multiple*m;
#define ELTSIZE(x) (((char *)((x)+1)) - ((char *) x))
  vs  = (stack_multiple*VSSIZE  + (STACK_OVER+1)*VSGETA)* ELTSIZE(vs_org);
  bd  = (stack_multiple*BDSSIZE + (STACK_OVER+1)*BDSGETA)*ELTSIZE(bds_org);
  frs = (stack_multiple*FRSSIZE + (STACK_OVER+1)*FRSGETA)*ELTSIZE(frs_org);
  ihs = (stack_multiple*IHSSIZE + (STACK_OVER+1)*IHSGETA)*ELTSIZE(ihs_org);
  if (stack_space==0)
    enter_mark_origin(&stack_space);
  massert(!((vs+bd+frs+ihs)%sizeof(fixnum)));
  stack_space=fSmake_vector(make_fixnum(aet_fix),(vs+bd+frs+ihs)/sizeof(fixnum),Cnil,Cnil,Cnil,0,Ct,OBJNULL);
  p=stack_space->v.v_self;
  COPYSTACK(vs_org,p,object,vs_limit,vs_top,VSGETA,VSSIZE);
  COPYSTACK(bds_org,p,struct bds_bd,bds_limit,bds_top,BDSGETA,BDSSIZE);
  COPYSTACK(frs_org,p,struct frame,frs_limit,frs_top,FRSGETA,FRSSIZE);
  COPYSTACK(ihs_org,p,struct invocation_history,ihs_limit,ihs_top,IHSGETA,IHSSIZE);
  vs_base=vs_top;
  return stack_multiple;
}

DEFVAR("*NO-INIT*",sSAno_initA,SI,Cnil,"");

LFD(siLinit_system)(void) {
  check_arg(0);
  gcl_init_system(sSAno_initA);
  vs_base[0] = Cnil;
  vs_top=vs_base+1;
}

static void
FFN(siLuser_init)(void) {
  check_arg(0);
  sLApackageA->s.s_dbind = user_package;
  user_init();
  vs_base[0] = Cnil;
}

DEFUN("NANI",object,fSnani,SI,1,1,NONE,OI,OO,OO,OO,(fixnum address),"") {

  RETURN1((object)address);

}

DEFUN("ADDRESS",object,fSaddress,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {

  RETURN1(x);

}

static void
FFN(siLinitialization_failure)(void) {
  check_arg(0);
  printf("lisp initialization failed\n");
  do_gcl_abort();
}

DEFUN("IDENTITY",object,fLidentity,LISP,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  RETURN1 (x0);
}


DEFUN("LISP-IMPLEMENTATION-VERSION",object,fLlisp_implementation_version,LISP,0,0,NONE,OO,OO,OO,OO,(void),"") {
  RETURN1((make_simple_string(LISP_IMPLEMENTATION_VERSION)));
}

static void
FFN(siLsave_system)(void) {
  
#ifdef HAVE_YP_UNBIND
  extern object truename(),namestring();
  check_arg(1);
  /* prevent subsequent consultation of yp by getting
     truename now*/
  vs_base[0]=namestring(truename(vs_base[0]));
  {char name[200];
  char *dom = name;
  if (0== getdomainname(dom,sizeof(name)))
    yp_unbind(dom);}
#endif
  
#ifdef DO_BEFORE_SAVE
  DO_BEFORE_SAVE
#endif	
    
  siLsave();

}

DEFVAR("*LISP-MAXPAGES*",sSAlisp_maxpagesA,SI,make_fixnum(real_maxpage),"");
DEFVAR("*SYSTEM-DIRECTORY*",sSAsystem_directoryA,SI,Cnil,"");
DEFVAR("*MULTIPLY-STACKS*",sSAmultiply_stacksA,SI,Cnil,"");
DEF_ORDINARY("TOP-LEVEL",sStop_level,SI,"");
DEFVAR("*COMMAND-ARGS*",sSAcommand_argsA,SI,sLnil,"");

static void
init_main(void) {

  /* make_function("BY", Lby); */
  make_si_function("ARGC", siLargc);
  make_si_function("ARGV", siLargv);
  
  make_si_function("GETENV", siLgetenv);
  
  make_si_function("MARK-VS", siLmark_vs);
  make_si_function("CHECK-VS", siLcheck_vs);
  make_si_function("INIT-SYSTEM", siLinit_system);
  make_si_function("USER-INIT", siLuser_init);
  make_si_function("INITIALIZATION-FAILURE",
		   siLinitialization_failure);
  
  siClisp_pagesize =
    make_si_constant("LISP-PAGESIZE", make_fixnum(PAGESIZE));
  
  
  {object features;
  
#define ADD_FEATURE(name) \
	 features=  make_cons(make_keyword(name),features)
  
  features=    make_cons(make_keyword("COMMON"),
			 make_cons(make_keyword("KCL"), Cnil));
  ADD_FEATURE("AKCL");
  ADD_FEATURE("GCL");
#ifdef BROKEN_O4_OPT
  ADD_FEATURE("BROKEN_O4_OPT");
#endif
#ifdef GMP
  ADD_FEATURE("GMP");
#endif	 
#ifdef GCL_GPROF
  ADD_FEATURE("GPROF");
#endif	 
  
#ifndef _WIN32
  ADD_FEATURE("UNIX");
#endif

#if defined ( DARWIN )
  ADD_FEATURE("DARWIN");
#endif

#if defined ( _WIN32 ) 
  ADD_FEATURE("WINNT");
  ADD_FEATURE("WIN32");
#endif

#if defined(__CYGWIN__)
  ADD_FEATURE("CYGWIN");
#endif

#ifdef IEEEFLOAT
  ADD_FEATURE("IEEE-FLOATING-POINT");
#endif
#ifdef SGC
  ADD_FEATURE("SGC");
#endif	 

  ADD_FEATURE(HOST_CPU);
  ADD_FEATURE(HOST_KERNEL);
#ifdef HOST_SYSTEM
  ADD_FEATURE(HOST_SYSTEM);
#endif
#ifdef  BSD
  ADD_FEATURE("BSD");
#endif
  
#if !defined(DOUBLE_BIGENDIAN)
  ADD_FEATURE("CLX-LITTLE-ENDIAN");
#endif
  
#ifndef PECULIAR_MACHINE
#define BIGM    (int)((((unsigned int)(-1))/2))	 
  { 
/*      int ONEM = -1; */
    int Bigm  = BIGM;
    int Smallm = -BIGM-1;
    int Seven = 7;
    int Three = 3;
    if ( (Smallm / Seven)  < 0
	 && (Smallm / (-Seven))  > 0
	 && (Bigm / (-Seven)) < 0 
	 && ((-Seven) / Three) == -2
	 && (Seven / (-Three)) == -2
	 && ((-Seven)/ (-Three)) == 2)
      { ADD_FEATURE("TRUNCATE_USE_C");
      }  }
#endif
  
#ifdef USE_READLINE
#ifdef READLINE_IS_EDITLINE
  ADD_FEATURE("EDITLINE");
#else
  ADD_FEATURE("READLINE");
#endif
#endif
#if !defined(USE_DLOPEN)
  ADD_FEATURE("NATIVE-RELOC");
#if defined(HAVE_LIBBFD) 
  ADD_FEATURE("BFD");
#endif
#endif
  ADD_FEATURE("UNEXEC");
#ifdef HAVE_XGCL
  ADD_FEATURE("XGCL");
#endif

#ifdef HAVE_GNU_LD
  ADD_FEATURE("GNU-LD");
#endif
  
#ifndef NO_C99
  ADD_FEATURE("C99");
#endif	 

#ifdef STATIC_LINKING
  ADD_FEATURE("STATIC");
#endif	 

#if SIZEOF_LONG==8
  ADD_FEATURE("64BIT");
#endif	 
  ADD_FEATURE("INTDIV");
  ADD_FEATURE("DYNAMIC-EXTENT");

#ifdef LARGE_MEMORY_MODEL
  ADD_FEATURE("LARGE-MEMORY-MODEL");
#endif

  make_special("*FEATURES*",features);}
  
  make_si_function("SAVE-SYSTEM", siLsave_system);
/*   make_si_sfun("CATCH-FATAL",siLcatch_fatal,ARGTYPE1(f_fixnum)); */
  /* make_si_function("WARN-VERSION",Lidentity); */
  
}

#if defined(HAVE_DIS_ASM_H) && defined(OUTPUT_ARCH)

#include "dis-asm.h"

static char b[4096],*bp;

static int
my_fprintf(void *v,const char *f,...) {
  va_list va;
  int r;
  va_start(va,f);
  bp+=(r=vsnprintf(bp,sizeof(b)-(bp-b),f,va));
  va_end(va);
  return r;
}

static int
my_fprintf_styled(void *v,enum disassembler_style,const char *f,...) {
  va_list va;
  int r;
  va_start(va,f);
  bp+=(r=vsnprintf(bp,sizeof(b)-(bp-b),f,va));
  va_end(va);
  return r;
}

static int
my_read(bfd_vma memaddr, bfd_byte *myaddr, unsigned int length, struct disassemble_info *dinfo) {
  memcpy(myaddr,(void *)(long)memaddr,length);
  return 0;
}

static void
my_pa(bfd_vma addr,struct disassemble_info *dinfo) {
  dinfo->fprintf_func(dinfo->stream,"%p",(void *)(long)addr);
}

#endif


DEFUN("DISASSEMBLE-INSTRUCTION",object,fSdisassemble_instruction,SI,1,1,NONE,OI,OO,OO,OO,(fixnum addr),"") {

#if defined(HAVE_DIS_ASM_H) && defined(OUTPUT_ARCH)

  static disassemble_info i;
  void *v;
  void * (*s)();
  fixnum j,j1,k;
  object x;

  if ((v=dlopen("libopcodes.so",RTLD_NOW))) {
    if ((s=dlsym(v,"init_disassemble_info"))) {
      s(&i, stdout,(fprintf_ftype) my_fprintf,my_fprintf_styled);
      i.read_memory_func=my_read;
      i.print_address_func=my_pa;
#if defined(OUTPUT_MACH)
      i.mach=OUTPUT_MACH;
#endif
      if ((s=dlsym(v,"disassembler"))) {
	disassembler_ftype disasm=(disassembler_ftype)(ufixnum)s(OUTPUT_ARCH,false,0,NULL);/*bfd_mach_x86_64*/
	bp=b;
	disasm(addr,&i);
	my_fprintf(NULL," ;");
	x=make_simple_string(b);

	j1=j=(addr-16)&(~16UL);
	bp=b;
	for (k=0;k<16;k++) {
	  j+=disasm(j,&i);
	  my_fprintf(NULL," ;");
	}
	return MMcons(x,MMcons(make_simple_string(b),make_fixnum(j-j1)));
      }
    }
    massert(!dlclose(v));
  }

#endif

  return MMcons(make_simple_string("fnop ;"),make_fixnum(0));

}

typedef struct {
  enum type tt;
  struct typemanager *tp;
} Tbl;

#define Tblof(a_)       {(a_),tm_of(a_)}
#define tblookup(a_,b_) ({Tbl *tb=tb1;(b_)=(a_);for (;tb->tt && tb->b_!=(b_);tb++);tb->tt;})
#define mtm_of(a_)      (a_)>=t_other ? NULL : tm_of(a_)

DEFUN("FUNCTION-BY-ADDRESS",object,fSfunction_by_address,SI,1,1,NONE,OI,OO,OO,OO,(fixnum ad),"") {

  ufixnum m=-1,mm,j;
  void *o;
  object x,xx=Cnil;
  Tbl tb1[]={Tblof(t_function),{0}};
  struct typemanager *tp;
  enum type tt;
  struct pageinfo *v;

  if (VALID_DATA_ADDRESS_P(ad))
    for (v=cell_list_head;v;v=v->next)
      if (tblookup(mtm_of(v->type),tp))
	for (o=pagetochar(page(v)),j=tp->tm_nppage;j--;o+=tp->tm_size)
	  if (tblookup(type_of((x=o)),tt))
	    if (!is_free(x) && (mm=ad-(ufixnum)x->fun.fun_self)<m) {
	      m=mm;
	      xx=x;
	    }

  return xx;

}
