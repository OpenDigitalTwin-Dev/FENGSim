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

/* Processor cache synchronization code.  This is based on powerpc-linux.h (Debian ppc).
   See equivalent code in dyld.  See also vm_msync declared in <mach/vm_maps.h>.  */
#define CLEAR_CACHE_LINE_SIZE 32
#define CLEAR_CACHE                                                             \
do {                                                                            \
  void *v=memory->cfd.cfd_start,*ve=v+memory->cfd.cfd_size;                     \
  v=(void *)((unsigned long)v & ~(CLEAR_CACHE_LINE_SIZE - 1));                  \
  for (;v<ve;v+=CLEAR_CACHE_LINE_SIZE)                                          \
  asm __volatile__                                                              \
    ("dcbst 0,%0\n\tsync\n\ticbi 0,%0\n\tsync\n\tisync": : "r" (v) : "memory"); \
} while(0)


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
#define GET_FAULT_ADDR(sig,code,scp,addr) ((char *) (((ucontext_t *) scp)->uc_mcontext->es.dar))

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

#define RELOC_H "mach32_ppc_reloc.h"
