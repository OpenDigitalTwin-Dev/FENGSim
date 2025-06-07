#define MP386
#include "att.h"
/* #include "386.h" */
/* #include "fcntl.h" */

#define DBEGIN _dbegin
#define DBEGIN_TY unsigned long
extern DBEGIN_TY _dbegin;

/* size to use for mallocs done  */
/* #define BABY_MALLOC_SIZE 0x5000 */

#define RECREATE_HEAP recreate_heap1();

#ifdef IN_UNIXTIME
#undef ATT
#define BSD
#endif

#define IS_DIR_SEPARATOR(x) ((x=='/')||(x=='\\'))

#undef NEED_GETWD
#ifdef IN_UNIXFSYS
#undef ATT
#define BSD
#endif

/* on most machines this will test in one instruction
   if the pointe/r is on the C stack or the 0 pointer
   in winnt our heap starts at DBEGIN
   */
/*  #define NULL_OR_ON_C_STACK(y)\ */
/*      (((unsigned int)(y)) == 0 ||  \ */
/*       (((unsigned int)(y)) < DBEGIN && ((unsigned int)(y)) &0xf000000)) */
/* #define NULL_OR_ON_C_STACK(y) (((void *)(y)) < ((void *)0x400000)) */
     
      


     

#define HAVE_SIGACTION
/* a noop */

#define brk(x) printf("not doing break\n");
#include <stdarg.h>     
#include <stdio.h>
#define UNIXSAVE "unexnt.c"

#define MAXPATHLEN 260
#define SEPARATE_SFASL_FILE "sfaslcoff.c"
#define SPECIAL_RSYM "rsym_nt.c"

#define HAVE_AOUT "wincoff.h"
/* we dont need to worry about zeroing fp->_base , to prevent  */

 /* must use seek to go to beginning of string table */
/* #define MUST_SEEK_TO_STROFF */
/* #define N_STROFF(hdr)   ((&hdr)->f_symptr+((&hdr)->f_nsyms)*SYMESZ) */

#define TO_NUMBER(ptr,type) (*((type *)(void *)(ptr)))

#define SEEK_TO_END_OFILE(fp) seek_to_end_ofile(fp)
		
#define RUN_PROCESS

#define	IEEEFLOAT
  
#define I386

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("I386"); ADD_FEATURE("WINNT")


/* include some low level routines for maxima */
#define CMAC

#define RELOC_FILE "rel_coff.c"

#undef  LISTEN_FOR_INPUT
#define LISTEN_FOR_INPUT(fp) do { \
  int c = 0; \
  if ((((FILE *)fp)->_r <= 0) && (ioctl(((FILE *)fp)->_file, FIONREAD, &c), c<=0)) \
    return 0; \
} while (0)

/* adjust the start to the offset */
#define ADJUST_RELOC_START(j) \
	the_start = memory->cfd.cfd_start + \
	  (j == DATA_NSCN ? textsize : 0);
	

#define IF_ALLOCATE_ERR \
	if (core_end != sbrk(0))\
         {char * e = sbrk(0); \
	if (e - core_end < 0x10000 ) { \
	  int i; \
	  for (i=page(core_end); i < page(e); i++) { \
	     \
	  } \
	  core_end = e; \
	} \
          else  \
        error("Someone allocated my memory!");} \
	if (core_end != (sbrk(PAGESIZE*(n - m))))

#include <limits.h>
#include <sys/stat.h>
#define GET_FULL_PATH_SELF(a_) do {				\
    static char q[PATH_MAX];					\
    massert(which("/proc/self/exe",q) || which(argv[0],q));	\
    (a_)=q;							\
  } while(0)

/* Begin for cmpinclude */


/* End for cmpinclude */

#define SF(a_) ((siginfo_t *)a_)

#define FPE_CODE(i_,v_) make_fixnum((long)fSfpe_code((long)FFN(fSfnstsw)(),(long)FFN(fSstmxcsr)()))
/* #define FPE_CODE(i_,v_) make_fixnum((fixnum)SF(i_)->si_code) */
#define FPE_ADDR(i_,v_) make_fixnum((fixnum)SF(i_)->si_addr)
#define FPE_CTXT(v_) Cnil

#define FPE_INIT Cnil

#undef HAVE_MPROTECT /*buggy on cygwin and unnecessary*/
