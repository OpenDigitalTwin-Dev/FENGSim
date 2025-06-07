    case R_PARISC_PCREL17F:
      s+=a-pltgot;
      s=((long)s)>>2;
      massert(ovchks(s,~MASK(17)));		  
      s&=MASK(17);
      *where=(0x39<<26)|(0x13<<21)|ASM17(s); /* b,l -> be,l */
      break;
    case R_PARISC_PCREL21L:
      s+=a;
      s-=p+11;
      s>>=11;
      store_valu(where,MASK(21),ASM21(s));
      break;
    case R_PARISC_DIR21L:
      s+=a;
      s>>=11;
      store_valu(where,MASK(21),ASM21(s));
      break;
    case R_PARISC_PCREL14R:
      s+=a;
      s-=p+11;
      s&=MASK(11);
      store_valu(where,MASK(14),s<<1);
      break;
    case R_PARISC_DIR14R:
      s+=a;
      s&=MASK(11);
      store_valu(where,MASK(14),s<<1);
      break;
    case R_PARISC_DIR17R:
      s+=a;
      s&=MASK(11);
      store_valu(where,MASK(17),s<<1);
      break;
    case R_PARISC_LTOFF21L:
    case R_PARISC_DPREL21L:
      s-=pltgot;
      s>>=11;
      store_valu(where,MASK(21),ASM21(s));
      break;
    case R_PARISC_LTOFF14R:
    case R_PARISC_DPREL14R:
      s-=pltgot;
      s&=MASK(11);
      store_valu(where,MASK(14),s<<1);
      store_valu(where,MASK(6)<<26,0xd<<26); /*ldw -> ldo*/
      break;
    case R_PARISC_PLABEL32:
    case R_PARISC_SEGREL32:
    case R_PARISC_DIR32:
      store_val(where,~0L,s+a);
      break;
