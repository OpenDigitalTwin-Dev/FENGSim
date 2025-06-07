    case R_ALPHA_GPDISP:
      gotoff=(ul)(got+HIGH(a)-1);
      s=gotoff-p;
      store_val(where,MASK(16),(s-(short)s)>>16);
      store_val((void *)where+LOW(a),MASK(16),s);
      break;
    case R_ALPHA_SREL32:
      store_val(where,MASK(32),s+a-p);
      break;
    case R_ALPHA_GPREL32:
      store_val(where,MASK(32),s+LOW(a)-(ul)(got+HIGH(a)-1));
      break;
    case R_ALPHA_LITUSE:
    case R_ALPHA_HINT:
      break;
    case R_ALPHA_REFQUAD:
      store_val(where,~0L,s+a);
      break;
    case R_ALPHA_REFLONG:
      store_val(where,MASK(32),s+a);
      break;
    case R_ALPHA_LITERAL:
      s+=LOW(a);
      a=HIGH(a)-1;
      if (s>=ggot1 && s<ggote) {
        massert(!write_stub(s,got+a));
      } else 
        got[a]=s;
      store_vals(where,MASK(16),(ul)(got+a)-gotoff);
      break;
    case R_ALPHA_GPRELHIGH:
      s+=a-gotoff;
      store_val(where,MASK(16),(s-(short)s)>>16);
      break;
    case R_ALPHA_GPRELLOW:
      store_val(where,MASK(16),s+a-gotoff);
      break;
    case R_ALPHA_TLS_GD_HI:
      store_vals(where,MASK(21),((long)(s+a-(p+4)))>>2);
      break;
