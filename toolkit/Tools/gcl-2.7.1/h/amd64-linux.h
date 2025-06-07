#include "linux.h"

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("BSD386"); \
      	             ADD_FEATURE("MC68020")

#define	I386
#define SGC

/* Apparently stack pointers can be 4 byte aligned, at least &argc -- CM */
#define C_GC_OFFSET 4

#define RELOC_H "elf64_i386_reloc.h"
#define MAX_CODE_ADDRESS (1L<<31)/*large memory model broken gcc 4.8*/
#define MAX_DEFAULT_MEMORY_MODEL_CODE_ADDRESS (1UL<<31)
#define LARGE_MEMORY_MODEL /*working -mcmodel=large giving unrestricted code load addresses*/
