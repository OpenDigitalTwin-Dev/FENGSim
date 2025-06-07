    case R_MIPS_JALR:
      break;
    case R_MIPS_GPREL32:
      add_val(where,~0L,s+a-(ul)got);
      break;
    case R_MIPS_26:
      if (((s+a)>>28)!=(((ul)where)>>28)) {
	gote=got+sym->st_size-1;
	massert(!write_26_stub(s+a,got,gote));
	store_val(where,MASK(26),((ul)gote)>>2);
      } else
        add_val(where,MASK(26),(s+a)>>2);
      break;
    case R_MIPS_32:
      add_val(where,~0L,s+a);
      break;
    case R_MIPS_GOT16:
      if (sym->st_shndx) { /* this should be followed by a LO16 */
	store_val(where,0xffe00000,0x3c000000); 
	r->r_info=ELF_R_INFO(ELF_R_SYM(r->r_info),R_MIPS_HI16);
	relocate(sym1,r,a,start,got,gote);
	break;
      }
    case R_MIPS_CALL16:
      gote=got+sym->st_size-1;
      store_val(where,MASK(16),((void *)gote-(void *)got));
      if (s>=ggot && s<ggote) {
        massert(!write_stub(s,got,gote));
      } else
        *gote=s;
      break;
    case R_MIPS_HI16:
      if (sym->st_other) s=gpd=(ul)got-(sym->st_other==2 ? 0 : (ul)where);
      if (!hr) hr=r;
      if (a) add_vals(where,MASK(16),(s>>16)+a);
      break;
    case R_MIPS_LO16:
      if (sym->st_other) s=gpd ? gpd : ({massert(sym->st_other==2);(ul)got;});
      a=*where&MASK(16);
      if (a&0x8000) a|=0xffff0000; 
      a+=s&MASK(16);
      a+=(a&0x8000)<<1; 
      store_val(where,MASK(16),a);
      a=0x10000|(a>>16);
      for (hr=hr ? hr : r;--r>=hr;)
	if (ELF_R_TYPE(r->r_info)==R_MIPS_HI16)
	  relocate(sym1,r,a,start,got,gote);
      hr=NULL;gpd=0;
      break;
