  case     R_68K_32:
    add_val(where,~0L,s+a);
    break;
  case     R_68K_PC32:
    add_val(where,~0L,s+a-p);
    break;
