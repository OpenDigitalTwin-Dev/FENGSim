#include "bsd.h"

#ifndef __ELF__
#error Linux systems use ELF
#endif

#undef HAVE_AOUT
/* #define HAVE_AOUT <elf.h> */
#define HAVE_ELF

/* Seeking to the end of ELF data is a little messy... */
#include <link.h>
#define SEEK_TO_END_OFILE(fp) seek_to_end_ofile(fp)


#define GET_FAULT_ADDR(sig,code,sv,a) ((siginfo_t *)code)->si_addr
#define MPROTECT_ACTION_FLAGS SA_RESTART | SA_ONSTACK | SA_SIGINFO
#define INSTALL_MPROTECT_HANDLER \
do {static struct sigaction action; \
      action.sa_handler = (void *)memprotect_handler; \
      action.sa_flags = MPROTECT_ACTION_FLAGS; \
      /*action.sa_restorer = 0;*/ \
      sigemptyset(&action.sa_mask); \
      sigaddset(&action.sa_mask,SIGINT); \
      sigaddset(&action.sa_mask,SIGALRM); \
      sigaction(SIGSEGV,&action,0); \
      sigaction(SIGBUS,&action,0);} while (0)

#undef SETUP_SIG_STACK

/* #define ELF_TEXT_BASE  0x0/\* DBEGIN *\/ */

/* #undef SET_REAL_MAXPAGE */
/* #define SET_REAL_MAXPAGE do { struct rlimit data_rlimit; \ */
/*                               extern char etext; \ */
/* 			       real_maxpage = MAXPAGE ;\ */
/*      	getrlimit(RLIMIT_DATA, &data_rlimit); \ */
/* 	real_maxpage = ((unsigned long)&etext/PAGESIZE \ */
/* 			+ data_rlimit.rlim_cur/PAGESIZE - ELF_TEXT_BASE/PAGESIZE); \ */
/* 	if (real_maxpage > MAXPAGE) \ */
/* 		real_maxpage = MAXPAGE ; } while(0) */

#ifdef USE_DLOPEN
#define SPECIAL_RSYM "rsym_elf.c"
#define SEPARATE_SFASL_FILE "fasldlsym.c"
#else
#ifdef HAVE_LIBBFD
#define SEPARATE_SFASL_FILE "sfaslbfd.c"
#else
/* #if !defined(__i386__) && !defined(__sparc__) */
/* #error Can only do non-bfd relocs for i386 and sparc */
/* #endif */
#define SPECIAL_RSYM "rsym_elf.c"
#define SEPARATE_SFASL_FILE "sfaslelf.c"
#endif
#endif

#define UNEXEC_USE_MAP_PRIVATE
#define UNIXSAVE "unexelf.c"

#undef HAVE_SIGVEC
#define HAVE_SIGACTION
#ifndef HAVE_SV_ONSTACK
#define SV_ONSTACK 0
#endif

/* unblock signals m and n, and set val to signal_mask(m) | signal_mask(n)
   if they were set */
/* #define SIG_UNBLOCK_SIGNALS(val,m,n) \ */
/*     current_mask = sigblock(0);  \ */
/*     sigsetmask(~(sigmask(m)) & ~(sigmask(n)) & current_mask); \ */
/*     result = (current_mask & sigmask(m) ? signal_mask(m) : 0) \ */
/*       | (current_mask & sigmask(n) ? signal_mask(n) : 0); */
#define HAVE_SIGPROCMASK

#define RUN_PROCESS

#define	IEEEFLOAT
       
/* #define HAVE_XDR */

#define USE_ULONG_

/*   How to check for input */
#undef LISTEN_FOR_INPUT
#define LISTEN_FOR_INPUT(fp) \
do { int c = 0; \
  if((((FILE *)fp)->_IO_read_ptr >= ((FILE *)fp)->_IO_read_end)	\
     && (ioctl(((FILE *)fp)->_fileno, FIONREAD, &c),c<=0))	\
     return 0;} while (0)

/* #define DATA_BEGIN((TXTRELOC+header.a_text+(SEGSIZ-1)) & ~(SEGSIZ-1)); */
#define DATA_BEGIN (char *)(char *)N_DATADDR(header);

#define	PAGSIZ		(NBPG)
#define	SEGSIZ		(NBPG * CLSIZE)
#define	TXTRELOC	0

#define USE_DIRENT
#define GETPATHNAME
#define PATHNAME_CACHE	10

/* get the fileno of a FILE* */
#define FILENO(x) fileno(x)

#define ULONG_DEFINED

#undef LD_COMMAND
#define LD_COMMAND(command,main,start,input,ldarg,output) \
  sprintf(command, "ld -d -S -N -x -A %s -T %x %s %s -o %s", \
            main,start,input,ldarg,output)

#define SET_SESSION_ID() (setpgrp() ? -1 : 0)

#define CLEANUP_CODE \
  setbuf(stdin,0); \
   setbuf(stdout,0);

#include <limits.h>
#include <sys/stat.h>
#define GET_FULL_PATH_SELF(a_) do {				\
    static char q[PATH_MAX];					\
    massert(which("/proc/self/exe",q) || which(argv[0],q));	\
    (a_)=q;							\
  } while(0)


#define UC(a_) ((ucontext_t *)a_)
#define SF(a_) ((siginfo_t *)a_)

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))

/* #define FPE_CODE(i_) make_fixnum((fixnum)SF(i_)->si_code) */
#ifdef __i386__
#define FPE_CODE(i_,v_) make_fixnum((fixnum)FFN(fSfpe_code)(UC(v_)->uc_mcontext.fpregs->sw,((struct _fpstate *)UC(v_)->uc_mcontext.fpregs)->mxcsr))
#define FPE_ADDR(i_,v_) make_fixnum((UC(v_)->uc_mcontext.fpregs->tag!=-1) ? UC(v_)->uc_mcontext.fpregs->ipoff : (fixnum)SF(i_)->si_addr)
#define FPE_SET_CTXT_ADDR(c_,a_) ({void *_c=(c_);fixnum _a=fix(a_);(UC(_c)->uc_mcontext.gregs[REG_EIP]=(_a));})
#define FPE_CLR_CTXT_CWD(c_) ({void *_c=(c_);UC(_c)->uc_mcontext.fpregs->cw|=FE_ALL_EXCEPT;((struct _fpstate *)UC(c_)->uc_mcontext.fpregs)->mxcsr|=(FE_ALL_EXCEPT<<7);})
#else
#define FPE_CODE(i_,v_) make_fixnum((fixnum)FFN(fSfpe_code)(UC(v_)->uc_mcontext.fpregs->swd,((struct _fpstate *)UC(v_)->uc_mcontext.fpregs)->mxcsr))
#define FPE_ADDR(i_,v_) make_fixnum(UC(v_)->uc_mcontext.fpregs->fop ? UC(v_)->uc_mcontext.fpregs->rip : (fixnum)SF(i_)->si_addr)
#define FPE_SET_CTXT_ADDR(c_,a_) ({void *_c=(c_);fixnum _a=fix(a_);(UC(_c)->uc_mcontext.gregs[REG_RIP]=(_a));})
#define FPE_CLR_CTXT_CWD(c_) ({void *_c=(c_);UC(_c)->uc_mcontext.fpregs->cwd|=FE_ALL_EXCEPT;UC(_c)->uc_mcontext.fpregs->mxcsr|=(FE_ALL_EXCEPT<<7);})
#endif
#define FPE_CTXT(v_)							\
  list(3,make_fixnum((fixnum)&UC(v_)->uc_mcontext.gregs),		\
       make_fixnum((fixnum)&UC(v_)->uc_mcontext.fpregs->_st),		\
       make_fixnum((fixnum)&((struct _fpstate *)UC(v_)->uc_mcontext.fpregs)->_xmm))

#define MC(b_) v.uc_mcontext.b_
#define REG_LIST(a_) MMcons(make_fixnum(sizeof(a_)),make_fixnum(sizeof(*a_)))
#define MCF(b_) (((struct _fpstate *)MC(fpregs))->b_)

#ifdef __x86_64__
#define FPE_RLST "R8 R9 R10 R11 R12 R13 R14 R15 RDI RSI RBP RBX RDX RAX RCX RSP RIP EFL CSGSFS ERR TRAPNO OLDMASK CR2"
#elif defined(__i386__)
#define FPE_RLST "GS FS ES DS EDI ESI EBP ESP EBX EDX ECX EAX TRAPNO ERR EIP CS EFL UESP SS"
#else
#error Missing reg list
#endif

#define FPE_INIT ({ucontext_t v;list(3,MMcons(make_simple_string(({const char *s=FPE_RLST;s;})),REG_LIST(MC(gregs))),\
				     REG_LIST(MCF(_st)),REG_LIST(MCF(_xmm)));})

#else

#define FPE_TCODE(x_) \
  {ufixnum _x=(x_),_y=0;			\
   switch(_x) {					\
   case FPE_FLTINV: _y=FE_INVALID;break;	\
   case FPE_FLTDIV: _y=FE_DIVBYZERO;break;	\
   case FPE_FLTOVF: _y=FE_OVERFLOW;break;	\
   case FPE_FLTUND: _y=FE_UNDERFLOW;break;	\
   case FPE_FLTRES: _y=FE_INEXACT;break;	\
   }						\
   _y;						\
  }
#define FPE_CODE(i_,v_) make_fixnum(FPE_TCODE((fixnum)SF(i_)->si_code))
#define FPE_ADDR(i_,v_) make_fixnum((fixnum)SF(i_)->si_addr)
#define FPE_CTXT(v_) Cnil

#define FPE_INIT Cnil

#endif

#undef sbrk
#define sbrk msbrk
#define INITIALIZE_BRK msbrk_init();
