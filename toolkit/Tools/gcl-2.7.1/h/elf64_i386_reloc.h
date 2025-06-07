    case R_X86_64_32:
      add_val(where,MASK(32),s+a);
      break;
    case R_X86_64_32S:
      add_vals(where,MASK(32),s+a);
      break;
    case R_X86_64_64:
      add_val(where,~0L,s+a);
      break;
    case R_X86_64_PC32:
    case R_X86_64_PLT32:
      massert(ovchks(s+a-p,~MASK(32)));		  
      add_val(where,MASK(32),s+a-p);
      break;
