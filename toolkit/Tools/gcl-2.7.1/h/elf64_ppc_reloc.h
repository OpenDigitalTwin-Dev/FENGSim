#define ha(x_) ((((x_) >> 16) + (((x_) & 0x8000) ? 1 : 0)) & 0xffff)
#define lo(x_) ((x_) & 0xffff)
#define m(x_) ((void *)((ul)(x_)-6))

    case R_PPC64_TOC16_HA: 
      store_val(m(where),MASK(16),ha(s+a-toc));
      break;
    case R_PPC64_TOC16_LO_DS: 
      store_val(m(where),MASK(16),lo(s+a-toc));/*>>2*/
      break;
    case R_PPC64_TOC16_LO:
      store_val(m(where),MASK(16),lo(s+a-toc));
      break;
    case R_PPC64_ADDR64:
      store_val(where,~0L,(s+a));
      break;
    case R_PPC64_TOC:
      store_val(where,~0L,toc);
      break;
    case R_PPC64_REL32:
      store_val(where,MASK(32)<<32,(s+a-p)<<32);
      break;
