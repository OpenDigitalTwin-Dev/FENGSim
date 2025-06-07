#include <mach-o/ppc/reloc.h>

  case PPC_RELOC_VANILLA:

    add_val(q,~0L,ri->r_pcrel ? a-rel : a);

    break;

  case PPC_RELOC_JBSR:

    redirect_trampoline(ri,sec1->addr+ri[1].r_address,rel,sec1,io1,n1,&a);
    if (!ri->r_extern)
      return 0;

    if (ovchk(a,~MASK(26)))
      store_val(q,MASK(26),a|0x3); 
    else if (ovchk(a-(ul)q,~MASK(26)))
      store_val(q,MASK(26),(a-(ul)q)|0x1); 
	
    break;

  case PPC_RELOC_SECTDIFF:
  case PPC_RELOC_HI16_SECTDIFF:
  case PPC_RELOC_LO16_SECTDIFF:
  case PPC_RELOC_HA16_SECTDIFF:
  case PPC_RELOC_LO14_SECTDIFF:
  case PPC_RELOC_LOCAL_SECTDIFF:
  case PPC_RELOC_PAIR:
    break;
