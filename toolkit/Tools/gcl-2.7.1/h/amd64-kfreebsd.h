#include "linux.h"

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("BSD386"); \
      	             ADD_FEATURE("MC68020")


#define	I386
#define SGC

/* Apparently stack pointers can be 4 byte aligned, at least &argc -- CM */
#define C_GC_OFFSET 4

#define RELOC_H "elf64_i386_reloc.h"

#define BRK_DOES_NOT_GUARANTEE_ALLOCATION
#define FREEBSD
