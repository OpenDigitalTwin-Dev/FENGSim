#include "linux.h"

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("SH4"); \
      	             ADD_FEATURE("")

#define	SH4
#define SGC


#ifdef IN_SFASL
#include <sys/mman.h>
#define CLEAR_CACHE {\
   void *p=memory->cfd.cfd_start,*pe=p+memory->cfd.cfd_size; \
   p=(void *)((unsigned long)p & ~(PAGESIZE-1)); \
   for (;p<pe;p++) /*+=PAGESIZE?*/ asm __volatile__ ("ocbp @%0\n\t": : "r" (p) : "memory");\
}
#endif
#define RELOC_H "elf32_sh4_reloc.h"

#define NEED_STACK_CHK_GUARD

/* #define DEFINED_REAL_MAXPAGE (1UL<<18) /\*FIXME brk probe broken*\/ */
