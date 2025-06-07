#include "linux.h"

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("BSD386"); \
      	             ADD_FEATURE("MC68020")


#define	I386
#define SGC

#define RELOC_H "elf32_i386_reloc.h"

#define BRK_DOES_NOT_GUARANTEE_ALLOCATION
#define FREEBSD
