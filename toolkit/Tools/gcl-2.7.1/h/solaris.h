#ifndef __ELF__
#define __ELF__
#endif
#define ElfW(a) Elf32_ ## a
#if !defined(HAVE_LIBBFD) && !defined(USE_DLOPEN)
#define __ELF_NATIVE_CLASS 32
#include <sys/elf_SPARC.h>
#endif
#include "linux.h"

#ifdef IN_GBC
#undef MPROTECT_ACTION_FLAGS
#define MPROTECT_ACTION_FLAGS SA_RESTART|SA_SIGINFO
#define GET_FAULT_ADDR(sig,code,sv,a) \
 ((siginfo_t *)code)->si_addr
/*  #define GET_FAULT_ADDR(sig,code,sv,a) \ */
/*      ((void *)(*((char ***)(&code)))[44]) */
#endif

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("SUN"); \
      	             ADD_FEATURE("SPARC")

#define	SPARC
#define SGC

#define PTR_ALIGN 8

#undef LISTEN_FOR_INPUT
#undef SIG_UNBLOCK_SIGNALS
#define NO_SYSTEM_TIME_ZONE

void bcopy (const void *,void *,size_t);
void bzero(void *,size_t);
int bcmp(const void *,const void *,size_t);

#if SIZEOF_LONG==4
#define RELOC_H "elf32_sparc_reloc.h"
#else
#define RELOC_H "elf64_sparc_reloc.h"
#define SPECIAL_RELOC_H "elf64_sparc_reloc_special.h"
void unwind() __attribute__((optimize("O0")));/*FIXME*/
#endif

