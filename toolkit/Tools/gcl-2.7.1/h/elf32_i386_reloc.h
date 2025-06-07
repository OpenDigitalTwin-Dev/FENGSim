  case     R_386_32:
    add_val(where,~0L,s+a);
    break;
    
  case     R_386_PC32:
    add_val(where,~0L,s+a-p);
    break;
    
