#define R_PPC64_PLTSEQ  119  /*FIXME not in elf.h*/
#define R_PPC64_PLTCALL 120

#define ha(x_) ((((x_) >> 16) + (((x_) & 0x8000) ? 1 : 0)) & 0xffff)
#define lo(x_) ((x_) & 0xffff)

    case R_PPC64_REL16_HA: 
      store_val(where,MASK(16),ha(s+a-p));
      break;
    case R_PPC64_PLT16_HA:
      gote=got+sym->st_size-1;
      *gote=s+a;
      massert(toc);
      store_val(where,MASK(16),ha((ul)gote-toc->st_value));
      break;
    case R_PPC64_PLT16_LO_DS:
      gote=got+sym->st_size-1;
      *gote=s+a;
      massert(toc);
      store_val(where,MASK(16),lo((ul)gote-toc->st_value));/*>>2*/
      break;
    case R_PPC64_PLTSEQ:
    case R_PPC64_PLTCALL:
      break;
    case R_PPC64_TOC16_HA: 
      massert(toc);
      store_val(where,MASK(16),ha(s+a-toc->st_value));
      break;
    case R_PPC64_TOC16_LO_DS: 
      massert(toc);
      store_val(where,MASK(16),lo(s+a-toc->st_value));/*>>2*/
      break;
    case R_PPC64_REL16_LO:
      store_val(where,MASK(16),lo(s+a-p));
      break;
    case R_PPC64_TOC16_LO:
      massert(toc);
      store_val(where,MASK(16),lo(s+a-toc->st_value));
      break;
    case R_PPC64_ADDR64:
      store_val(where,~0L,(s+a));
      break;
    case R_PPC64_TOC:
      massert(toc);
      store_val(where,~0L,toc->st_value);
      break;
    case R_PPC64_REL32:
      store_val(where,MASK(32),(s+a-p));
      break;
