  case     R_390_32:
    add_ivals((int *)where,MASK(32),s+a);
    break;
  case     R_390_64:
    add_val(where,~0L,s+a);
    break;
  case     R_390_PC32:
    add_ivals((int *)where,MASK(32),s+a-p);
    break;
  case     R_390_PC32DBL:
  case     R_390_PLT32DBL:/*FIXME think about this*/
    add_ivals((int *)where,MASK(32),(s+a-p)>>1);
    break;
