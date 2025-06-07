#include "linux.h"

#define SGC

/* Apparently stack pointers can be 4 byte aligned, at least &argc -- CM */
#define C_GC_OFFSET 4

#define RELOC_H "elf64_loongarch64_reloc.h"
#define SPECIAL_RELOC_H "elf64_loongarch64_reloc_special.h"
/* #define MAX_CODE_ADDRESS (1L<<31)/\*large memory model broken gcc 4.8*\/ */

#define NEED_STACK_CHK_GUARD
