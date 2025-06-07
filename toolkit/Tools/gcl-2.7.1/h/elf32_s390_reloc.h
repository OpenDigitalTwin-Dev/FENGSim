  case     R_390_32:
    add_val(where,~0L,s+a);
    break;
    
  case     R_390_PC32:
    add_val(where,~0L,s+a-p);
    break;

 case     R_390_PC32DBL:
    add_val(where,~0L,(s+a-p)>>1);
    break;
