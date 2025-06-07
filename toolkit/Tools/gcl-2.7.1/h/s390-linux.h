#include "linux.h"

#define SGC

#if SIZEOF_LONG == 8
#define C_GC_OFFSET 4
#define RELOC_H "elf64_s390_reloc.h"
#define SPECIAL_RELOC_H "elf64_sparc_reloc_special.h"
#else
#define RELOC_H "elf32_s390_reloc.h"
#endif
