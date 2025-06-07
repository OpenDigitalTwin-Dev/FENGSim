#define get_insn_page(x) ((x) & ~0xffful)
#define get_page_delta(dest, pc) ({                  \
  ul res = get_insn_page(dest) - get_insn_page(pc);  \
  if ((dest) & 0x800)                                  \
    res += 0x1000ul - 0x100000000ul;                 \
  if (res & 0x80000000)                              \
    res += 0x100000000ul;                            \
  res;                                               \
})
#define get_page_low(dest) ((dest) & 0xfff)
#define bdest (((long)((s+a)-p))>>2)
#define bgdest (((long)(((ul)got)-p))>>2)

    case R_LARCH_RELAX:
    case R_LARCH_ALIGN:
      massert(!emsg("Unsupport relaxation, please compile with '-mno-relax -Wa,-mno-relax'\n"));
      break;
    case R_LARCH_64:
      store_val(where,~0L,(s+a));
      break;
    case R_LARCH_32:
      store_val(where,MASK(32),(s+a));
      break;
    case R_LARCH_32_PCREL:
      store_val(where,MASK(32),(s+a)-p);
      break;
    case R_LARCH_ADD6:
      add_val(where,MASK(6),(s+a));
      break;
    case R_LARCH_ADD8:
      add_val(where,MASK(8),(s+a));
      break;
    case R_LARCH_ADD16:
      add_val(where,MASK(16),(s+a));
      break;
    case R_LARCH_ADD32:
      add_val(where,MASK(32),(s+a));
      break;
    case R_LARCH_ADD64:
      add_val(where,~0L,(s+a));
      break;
    case R_LARCH_SUB6:
      add_val(where,MASK(6),-(s+a));
      break;
    case R_LARCH_SUB8:
      add_val(where,MASK(8),-(s+a));
      break;
    case R_LARCH_SUB16:
      add_val(where,MASK(16),-(s+a));
      break;
    case R_LARCH_SUB32:
      add_val(where,MASK(32),-(s+a));
      break;
    case R_LARCH_SUB64:
      add_val(where,~0L,-(s+a));
      break;
    case R_LARCH_B16:
      store_val(where,MASK(16)<<10,bdest<<10);
      break;
    case R_LARCH_B21:
      store_val(where,(MASK(16)<<10)|MASK(5),bdest<<10|((bdest>>16)&0x1f));
      break;
    case R_LARCH_B26:
      {
	if ((bdest&(~MASK(25)))==0||((~bdest)&(~MASK(25)))==0) {
	  store_val(where,MASK(26),bdest<<10|((bdest>>16)&0x3ff));
	  break;
	}
	if (!(sym->st_size&0x2))
	  massert(!emsg("Unresolved R_LARCH_B26 symbol\n"));
	got+=(sym->st_size>>2)+(sym->st_size&0x1?1:0);
	store_val(where,MASK(26),bgdest<<10|((bgdest>>16)&0x3ff));
	memcpy(got,tramp,sizeof(tramp));
	store_val(got,MASK(20)<<5,(get_insn_page(s+a)-get_insn_page((ul)got))>>12<<5);
	store_val((ul*)((ul)got+4),MASK(16)<<10,(((s+a)>>2)&0x3ff)<<10);
      }
      break;
    case R_LARCH_PCALA_HI20:
      store_val(where,MASK(20)<<5,get_page_delta(s+a,p)>>12<<5);
      break;
    case R_LARCH_PCALA_LO12:
      store_val(where,MASK(12)<<10,get_page_low(s+a)<<10);
      break;
    case R_LARCH_GOT_PC_HI20:
      got+=sym->st_size>>2;
      *got=s+a;
      store_val(where,MASK(20)<<5,get_page_delta((ul)got,p)>>12<<5);
      break;
    case R_LARCH_GOT_PC_LO12:
      got+=sym->st_size>>2;
      // *got=s+a;
      store_val(where,MASK(12)<<10,get_page_low((ul)got)<<10);
      break;
