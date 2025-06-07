    case R_AARCH64_ABS64: /* .xword: (S+A) */
      store_val(where,~0L,s+a);
      break;
    case R_AARCH64_ABS32: /* .word:  (S+A) */
      store_val(where,MASK(32),s+a);
      break;
    case R_AARCH64_JUMP26: /* B:      ((S+A-P) >> 2) & 0x3ffffff.  */
    case R_AARCH64_CALL26: /* BL:     ((S+A-P) >> 2) & 0x3ffffff.  */
      {
	long x=((long)(s+a-p))/4;
	if (abs(x)&(~MASK(25))) {
	  if (a) {
	    got+=gotp;
	    gotp+=tz;
	  } else
	    got+=(sym->st_size-1)*tz;
	  *got++=s+a;
	  memcpy(got,tramp,sizeof(tramp));
	  x=((long)got-p)/4;
	}
	store_vals(where,MASK(26),x);
      }
      break;
    case R_AARCH64_ADR_PREL_PG_HI21: /* ADRH:   ((PG(S+A)-PG(P)) >> 12) & 0x1fffff */
#define PG(x) ((x) & ~0xfff)
      s = ((long)(PG(s+a)-PG(p))) / 0x1000;
      store_val(where,MASK(2) << 29, (s & 0x3) << 29);
      store_val(where,MASK(19) << 5, (s & 0x1ffffc) << 3);
#undef PG
      break;
    case R_AARCH64_ADD_ABS_LO12_NC: /* ADD:    (S+A) & 0xfff */
      store_val(where,MASK(12) << 10,(s+a) << 10);
      break;
    case R_AARCH64_LDST8_ABS_LO12_NC: /* LD/ST8: (S+A) & 0xfff */
      store_val(where,MASK(12) << 10,((s+a) & 0xfff) << 10);
      break;
    case R_AARCH64_LDST16_ABS_LO12_NC: /* LD/ST16: (S+A) & 0xffe */
      store_val(where,MASK(12) << 10,((s+a) & 0xffe) << 9);
      break;
    case R_AARCH64_LDST32_ABS_LO12_NC: /* LD/ST32: (S+A) & 0xffc */
      store_val(where,MASK(12) << 10,((s+a) & 0xffc) << 8);
      break;
    case R_AARCH64_LDST64_ABS_LO12_NC: /* LD/ST64: (S+A) & 0xff8 */
      store_val(where,MASK(12) << 10,((s+a) & 0xff8) << 7);
      break;
    case R_AARCH64_LDST128_ABS_LO12_NC: /* LD/ST128: (S+A) & 0xff0 */
      store_val(where,MASK(12) << 10,((s+a) & 0xff0) << 6);
      break;
    case R_AARCH64_PREL64:
      store_val(where,~0L,(s+a-p));
      break;
    case R_AARCH64_PREL32:
      store_val(where,MASK(32),(s+a-p));
      break;
    case R_AARCH64_PREL16:
      store_val(where,MASK(16),(s+a-p));
      break;
