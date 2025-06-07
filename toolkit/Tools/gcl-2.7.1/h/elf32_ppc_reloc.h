    case R_PPC_REL24: /*FIXME, this is just for mcount, why longcall doesn't work is unknown */
      s+=a;
      if (ovchks(s,~MASK(26)))
        store_val(where,MASK(26),s|0x3);
      else  if (ovchks(s-p,~MASK(26)))
        store_val(where,MASK(26),(s-p)|0x1); 
      else massert(!"REL24 overflow");
        break;
    case R_PPC_REL32: 
      store_val(where,~0L,s+a-p);
      break;
    case R_PPC_ADDR16_HA:
      s+=a;
      s+=s&0x8000 ? 1<<16 : 0;
      store_val(where,~MASK(16),s&0xffff0000);
      break;
    case R_PPC_ADDR16_LO:
      store_val(where,~MASK(16),(s+a)<<16);
      break;
    case R_PPC_ADDR32:
      store_val(where,~0L,s+a);
      break;
