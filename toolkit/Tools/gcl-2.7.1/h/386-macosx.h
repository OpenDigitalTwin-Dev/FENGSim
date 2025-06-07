/*
    GCL config file for Mac OS X.
    
    To be used with the following configure switches :
        --enable-debug (optional)
        --enable-machine=powerpc-macosx
        --disable-statsysbfd
        --enable-custreloc
    
    Aurelien Chanudet <aurelien.chanudet(at)m4x.org>
*/

/* For those who are using ACL2, please remember to enlarge your shell stack (ulimit -s 8192).  */

#include "bsd.h"

#define DARWIN

/* Mac OS X has its own executable file format (Mach-O).  */
#undef HAVE_AOUT
#undef HAVE_ELF


/** sbrk(2) emulation  */

/* Alternatively, we could use the global variable vm_page_size.  */
#define PAGEWIDTH 12

/* The following value determines the running process heap size.  */
/* #define BIG_HEAP_SIZE   0x50000000 */

extern char *mach_mapstart;
extern char *mach_maplimit;
extern char *mach_brkpt;

extern char *get_dbegin ();

#include <unistd.h> /* to get sbrk defined */
extern void *my_sbrk(long incr);
#define sbrk my_sbrk


/** (si::save-system "...") a.k.a. unexec implementation  */

/* The implementation of unexec for GCL is based on Andrew Choi's work for Emacs.
   Previous pioneering implementation of unexec for Mac OS X by Steve Nygard.  */
#define UNIXSAVE "unexmacosx.c"

#undef malloc
#define malloc my_malloc

#undef free
#define free my_free

#undef realloc
#define realloc my_realloc

#undef valloc
#define valloc my_valloc

#undef calloc
#define calloc my_calloc


/** Dynamic loading implementation  */

/* The sfasl{bfd,macosx,macho}.c files are included from sfasl.c.  */
#ifdef HAVE_LIBBFD
#define SEPARATE_SFASL_FILE "sfaslbfd.c"
#else
#define SPECIAL_RSYM "rsym_macosx.c"
#define SEPARATE_SFASL_FILE "sfaslmacho.c"
#endif

/* The file has non Mach-O stuff appended.  We need to know where the Mach-O stuff ends.  */
#include <stdio.h>
extern int seek_to_end_ofile (FILE *);
#define SEEK_TO_END_OFILE(fp) seek_to_end_ofile(fp)

/** Stratified garbage collection implementation [ (si::sgc-on t) ]  */

/* Mac OS X has sigaction (this is needed in o/usig.c)  */
#define HAVE_SIGACTION

/* Copied from {Net,Free,Open}BSD.h  */
/* Modified according to Camm's instructions on April 15, 2004.  */
#define HAVE_SIGPROCMASK

/* until the sgc/save problem can be fixed.  20050114 CM*/
/* #define SGC */

#define MPROTECT_ACTION_FLAGS (SA_SIGINFO | SA_RESTART)

#define INSTALL_MPROTECT_HANDLER                        \
do {                                                    \
  static struct sigaction sact;                         \
  sigfillset (&(sact.sa_mask));                         \
  sact.sa_flags = MPROTECT_ACTION_FLAGS;                \
  sact.sa_sigaction = (void (*) ()) memprotect_handler; \
  sigaction (SIGBUS, &sact, 0);                         \
  sigaction (SIGSEGV, &sact, 0);                        \
} while (0);

/* si_addr not containing the faulting address is a bug in Darwin.
   Work around this by looking at the dar field of the exception state.  */
#define GET_FAULT_ADDR(sig,code,sv,a) ((siginfo_t *)code)->si_addr
/* #define GET_FAULT_ADDR(sig,code,scp,addr) ((char *) (((ucontext_t *) scp)->uc_mcontext->es.dar)) */

/*
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/ucontext.h>

void handler (int sig, siginfo_t *info, void *scp)
{
     ucontext_t *uc = (ucontext_t *)scp;
     fprintf(stderr, "addr = 0x%08lx\n", uc->uc_mcontext->es.dar);
     _exit(99);
}

int main(void)
{
     struct sigaction sact;
     int ret;

     sigfillset(&(sact.sa_mask));
     sact.sa_flags = SA_SIGINFO;
     sact.sa_sigaction = (void (*)())handler;
     ret = sigaction (SIGBUS, &sact, 0);
     return *(int *)0x43;
}
*/


/** Misc stuff  */

#define IEEEFLOAT
       
/* Mac OS X does not have _fileno as in linux.h. Nor does it have _cnt as in bsd.h.
   Let's see what we can do with this declaration found in {Net,Free,Open}BSD.h.  */
#undef LISTEN_FOR_INPUT
#define LISTEN_FOR_INPUT(fp)                                            \
do {int c=0;                                                            \
  if (((FILE *)fp)->_r <=0 && (c=0, ioctl(((FILE *)fp)->_file, FIONREAD, &c), c<=0)) \
        return(FALSE);                                                  \
} while (0)

#define GET_FULL_PATH_SELF(a_)                              \
do {                                                        \
extern int _NSGetExecutablePath (char *, unsigned long *);  \
unsigned long bufsize = 1024;                               \
static char buf [1024];                                     \
static char fub [1024];                                     \
if (_NSGetExecutablePath (buf, &bufsize) != 0) {            \
    error ("_NSGetExecutablePath failed");                  \
}                                                           \
if (realpath (buf, fub) == 0) {                             \
    error ("realpath failed");                              \
}                                                           \
(a_) = fub;                                                 \
} while (0)

#ifdef _LP64
#define C_GC_OFFSET 4
#include <mach-o/x86_64/reloc.h>
#define RELOC_H "mach64_i386_reloc.h"
#else
#define RELOC_H "mach32_i386_reloc.h"
#endif


#define UC(a_) ((ucontext_t *)a_)
#define SF(a_) ((siginfo_t *)a_)

#define FPE_CODE(i_,v_) make_fixnum(FFN(fSfpe_code)(*(fixnum *)&UC(v_)->uc_mcontext->__fs.__fpu_fsw,UC(v_)->uc_mcontext->__fs.__fpu_mxcsr))
#define FPE_ADDR(i_,v_) make_fixnum(UC(v_)->uc_mcontext->__fs.__fpu_fop ? UC(v_)->uc_mcontext->__fs.__fpu_ip : (fixnum)SF(i_)->si_addr)
#define FPE_CTXT(v_) list(3,make_fixnum((fixnum)&UC(v_)->uc_mcontext->__ss), \
			  make_fixnum((fixnum)&UC(v_)->uc_mcontext->__fs.__fpu_stmm0), \
			  make_fixnum((fixnum)&UC(v_)->uc_mcontext->__fs.__fpu_xmm0))


#define MC(b_) v.uc_mcontext->b_
#define REG_LIST(a_,b_) MMcons(make_fixnum(a_*sizeof(b_)),make_fixnum(sizeof(b_)))
#define MCF(b_) ((MC(__fs)).b_)

#ifdef __x86_64__
#define FPE_RLST "RAX RBX RCX RDX RDI RSI RBP RSP R8 R9 R10 R11 R12 R13 R14 R15 RIP RFLAGS CS FS GS"
#elif defined(__i386__)
#define FPE_RLST "GS FS ES DS EDI ESI EBP ESP EBX EDX ECX EAX TRAPNO ERR EIP CS EFL UESP SS"
#else
#error Missing reg list
#endif

#define FPE_INIT ({ucontext_t v;list(3,MMcons(make_simple_string(({const char *s=FPE_RLST;s;})),REG_LIST(21,MC(__ss))),	\
				     REG_LIST(8,MCF(__fpu_stmm0)),REG_LIST(16,MCF(__fpu_xmm0)));})


#include <sys/param.h>/*PATH_MAX MAXPATHLEN*/
#undef MIN
#undef MAX
