#include <mach-o/x86_64/reloc.h>

#define GOT_RELOC(ri) ri->r_type==X86_64_RELOC_GOT_LOAD||ri->r_type==X86_64_RELOC_GOT


  case X86_64_RELOC_UNSIGNED:		// for absolute addresses

     if (ri->r_extern || !ri->r_pcrel) 
      store_val(q,~0L,ri->r_pcrel ? a-rel : a);

    break; 
  case X86_64_RELOC_GOT_LOAD:		// a MOVQ load of a GOT entry
  case X86_64_RELOC_GOT:		// a MOVQ load of a GOT entry

    got+=n1[ri->r_symbolnum].n_desc-1;
    *got=a;
    a=(ul)got;

  case X86_64_RELOC_SIGNED:		// for signed 32-bit displacement
  case X86_64_RELOC_BRANCH:		// a CALL/JMP instruction with 32-bit displacement

     if (ri->r_extern || !ri->r_pcrel) 	   
       store_val(q,MASK(32),(ri->r_pcrel ? a-((ul)q+4) : a)+(signed)(*q&MASK(32)));

    break;

