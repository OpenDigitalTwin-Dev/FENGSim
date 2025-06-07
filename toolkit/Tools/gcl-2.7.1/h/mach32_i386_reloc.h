  case GENERIC_RELOC_VANILLA:
    
    redirect_trampoline(ri,*q,rel,sec1,io1,n1,&a);
    if (ri->r_extern)
      store_val(q,~0L,ri->r_pcrel ? a-rel : a);
    else if (!ri->r_pcrel)
      add_val(q,~0L,a);
    
    break;

  case GENERIC_RELOC_LOCAL_SECTDIFF:
  case GENERIC_RELOC_SECTDIFF:
  case GENERIC_RELOC_PAIR:
    break;

