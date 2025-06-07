#define riscv_high(a_) ((a_)+(((a_)&0x800) ? (1<<12) : 0))

    case R_RISCV_HI20:
      store_val(where,MASK(20)<<12,riscv_high(s+a));
      break;
    case R_RISCV_RELAX:/*FIXME figure out how to delete instructions efficiently*/
      break;
    case R_RISCV_LO12_I:
      store_val(where,MASK(12)<<20,(s+a)<<20);
      break;
    case R_RISCV_LO12_S:
      store_val(where,MASK(5)<<7,(s+a)<<7);
      store_val(where,MASK(7)<<25,(s+a)<<20);
      break;
    case R_RISCV_CALL:
    case R_RISCV_CALL_PLT:
      store_val(where,MASK(20)<<12,riscv_high(s+a-p));
      store_val((void *)where+4,MASK(12)<<20,(s+a-p)<<20);
      break;
    case R_RISCV_BRANCH:
    case R_RISCV_RVC_BRANCH:
    case R_RISCV_RVC_JUMP:
    case R_RISCV_JAL:
      break;
    case R_RISCV_64:
      store_val(where,~0L,(s+a));
      break;
    case R_RISCV_32:
      store_val(where,MASK(32),(s+a));
      break;
    case R_RISCV_32_PCREL:
      store_val(where,MASK(32),(s+a)-p);
      break;
    case R_RISCV_ADD8:
      add_val(where,MASK(8),(s+a));
      break;
    case R_RISCV_ADD16:
      add_val(where,MASK(16),(s+a));
      break;
    case R_RISCV_ADD32:
      add_val(where,MASK(32),(s+a));
      break;
    case R_RISCV_ADD64:
      add_val(where,~0L,(s+a));
      break;
    case R_RISCV_SUB6:
      add_val(where,MASK(6),-(s+a));
      break;
    case R_RISCV_SUB8:
      add_val(where,MASK(8),-(s+a));
      break;
    case R_RISCV_SUB16:
      add_val(where,MASK(16),-(s+a));
      break;
    case R_RISCV_SUB32:
      add_val(where,MASK(32),-(s+a));
      break;
    case R_RISCV_SUB64:
      add_val(where,~0L,-(s+a));
      break;
    case R_RISCV_SET6:
      store_val(where,MASK(6),(s+a));
      break;
    case R_RISCV_SET8:
      store_val(where,MASK(8),(s+a));
      break;
    case R_RISCV_SET16:
      store_val(where,MASK(16),(s+a));
      break;
    case R_RISCV_SET32:
      store_val(where,MASK(32),(s+a));
      break;
