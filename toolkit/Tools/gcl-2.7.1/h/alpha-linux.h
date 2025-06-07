#include "linux.h"

#define SGC

#define RELOC_H "elf64_alpha_reloc.h"
#define SPECIAL_RELOC_H "elf64_alpha_reloc_special.h"
#define PAL_imb		134
#define imb() __asm__ __volatile__ ("call_pal %0 #imb" : : "i" (PAL_imb) : "memory")
#define CLEAR_CACHE imb()

/*FIXME probe broken in recent kernels, no access*/
/* #define DEFINED_REAL_MAXPAGE (1UL<<18) /\*FIXME brk probe broken*\/ */
