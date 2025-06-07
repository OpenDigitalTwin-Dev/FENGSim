/*
 * FreeBSD.h for gcl
 *
 * Ported by Mark Murray
 *  Looked at previous versions by Hsu, Werkowsksi, Tobin, and Mogart.
 *
 */

#ifndef __ELF__
#error FreeBSD systems use ELF
#endif

#if defined(__i386__)
#define __ELF_NATIVE_CLASS 32
#endif
#if defined(__alpha__) || defined(__sparc64__) || defined(__ia64__)
#define __ELF_NATIVE_CLASS 64
#endif

#if !defined(ElfW)
#define ElfW(a) Mjoin(Elf,Mjoin(__ELF_NATIVE_CLASS,Mjoin(_,a)))
#endif
#define ELFW(a) Mjoin(ELF,Mjoin(__ELF_NATIVE_CLASS,Mjoin(_,a)))
 
/* OpenBSD needs sys/types.h included before link.h, which is included
   in linux.h */
#include <sys/types.h>
#if defined(HAVE_ELF_H)
#include <elf.h>
#elif defined(HAVE_ELF_ABI_H)
#include <elf_abi.h>
#endif
#include "linux.h"

#if defined(__i386__)
#define I386
#endif

#define ADDITIONAL_FEATURES					\
		     ADD_FEATURE("386BSD");			\
                     ADD_FEATURE("FreeBSD");

#define USE_ATT_TIME

#undef LISTEN_FOR_INPUT
#define LISTEN_FOR_INPUT(fp)					\
do {								\
	int c = 0;						\
								\
	if (							\
		(fp)->_r <= 0 &&				\
		(ioctl(((FILE *)fp)->_file, FIONREAD, &c), c <= 0)	\
	)							\
		return(FALSE);					\
} while (0)

#ifdef IN_GBC
#include <sys/types.h>
#endif

#if defined(IN_UNIXTIME)
# include <time.h>
#endif

/*#define UNEXEC_USE_MAP_PRIVATE*/
#define UNIXSAVE "unexelf.c"

#ifdef CLOCKS_PER_SEC
#define HZ CLOCKS_PER_SEC
#else
#define HZ 128
#endif
/* #define ss_base ss_sp */

/* begin for GC */
#define PAGEWIDTH 12		/* i386 sees 4096 byte pages */
/* end for GC */

#define HAVE_SIGPROCMASK
#define SIG_STACK_SIZE (SIGSTKSZ/sizeof(double))

/*
 * The next two defines are for SGC,
 *	one of which needs to go in cmpinclude.h.
 */
#define SIGPROTV SIGBUS

/* Begin for cmpinclude */
#define SGC	/* can mprotect pages and so selective gc will work */
/* End for cmpinclude */
