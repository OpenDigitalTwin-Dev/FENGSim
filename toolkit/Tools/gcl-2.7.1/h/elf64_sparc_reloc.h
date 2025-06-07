  case R_SPARC_WDISP30:
    store_ivals((int *)where,MASK(30),((long)(s+a-p))>>2);
    break;
    
  case R_SPARC_HI22:
    store_ival((int *)where,MASK(22),(s+a)>>10);
    break;
    
  case R_SPARC_LO10:
    store_ival((int *)where,MASK(10),s+a);
    break;

  case R_SPARC_OLO10:
    store_ival((int *)where,MASK(10),s+a);
    add_ival((int *)where,MASK(13),ELF_R_ADDEND(r->r_info));
    break;

  case R_SPARC_13:
    store_ivalu((int *)where,MASK(13),s+a);
    break;
    
  case R_SPARC_32:
  case R_SPARC_UA32:
    store_ivalu((int *)where,MASK(32),s+a);
    break;
    
  case R_SPARC_64:
  case R_SPARC_UA64:
    store_valu(where,~0L,s+a);
    break;
