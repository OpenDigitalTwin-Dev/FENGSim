#define R_ARM_MOVW_ABS_NC 43
#define R_ARM_MOVT_ABS    44
#define R_ARM_CALL 28
#define R_ARM_V4BX 40
    case R_ARM_MOVW_ABS_NC:
      s+=a;
      s&=0xffff;
      s=(s&0xfff)|((s>>12)<<16);
      add_vals(where,~0L,s);
      break;
    case R_ARM_MOVT_ABS:
      s+=a;
      s>>=16;
      s=(s&0xfff)|((s>>12)<<16);
      add_vals(where,~0L,s);
      break;
    case R_ARM_CALL:
    case R_ARM_JUMP24:
      {
	long x=((long)(s+a-p))/4;
	if (abs(x)&(~MASK(23))) {
          got+=(sym->st_size-1)*tz;
	  memcpy(got,tramp,sizeof(tramp));
	  /*recurse on relocate?*/
          got[sizeof(tramp)/sizeof(*got)]=s;
	  x=((long)got-p)/4;
	}
	add_vals(where,MASK(24),x);
      }
      break;
    case R_ARM_V4BX:
    case R_ARM_ABS32:
      add_vals(where,~0L,s+a);
      break;
