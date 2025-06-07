#define R_ARM_THM_CALL        10
#define R_ARM_THM_MOVW_ABS_NC 47
#define R_ARM_THM_MOVW_ABS    48
     case R_ARM_THM_JUMP24:
      {
    	long x=(long)(s+a-p);
    	if (abs(x)&(~MASK(23))) {

          got+=(sym->st_size-1)*tz;
    	  memcpy(got,tramp,sizeof(tramp));

	  r->r_offset=(void *)got-(void *)start;
	  r->r_info=ELF_R_INFO(ELF_R_SYM(r->r_info),R_ARM_THM_MOVW_ABS_NC);
	  relocate(sym1,r,0,start,got,gote);

	  r->r_offset=(void *)(got+1)-(void *)start;
	  r->r_info=ELF_R_INFO(ELF_R_SYM(r->r_info),R_ARM_THM_MOVW_ABS);
	  relocate(sym1,r,0,start,got,gote);

    	  x=((long)got-p);
    	}
        if (ELF_ST_TYPE(sym->st_info)==STT_FUNC) x|=1;
        x-=4; /*FIXME maybe drop 4 and add_val below*/
        x=((long)x>>1);
        store_val(where,MASK(11)<<16,(x&0x7ff)<<16);
        store_val(where,MASK(10),x>>11);
        store_val(where,MASK(1)<<(16+11),(~((x>>21&0x1)^(x>>23&0x1)))<<(16+11));
        store_val(where,MASK(1)<<(16+13),(~((x>>22&0x1)^(x>>23&0x1)))<<(16+13));
        store_val(where,MASK(1)<<10,(x>>23&0x1)<<10);
      }
      break;
    case R_ARM_THM_CALL:
      {
    	long x=(long)(s+a-p);
    	if (abs(x)&(~MASK(22))) {
          got+=(sym->st_size-1)*tz;
    	  memcpy(got,tramp,sizeof(tramp));

	  r->r_offset=(void *)got-(void *)start;
	  r->r_info=ELF_R_INFO(ELF_R_SYM(r->r_info),R_ARM_THM_MOVW_ABS_NC);
	  relocate(sym1,r,0,start,got,gote);

	  r->r_offset=(void *)(got+1)-(void *)start;
	  r->r_info=ELF_R_INFO(ELF_R_SYM(r->r_info),R_ARM_THM_MOVW_ABS);
	  relocate(sym1,r,0,start,got,gote);

    	  x=((long)got-p);
    	}
        if (ELF_ST_TYPE(sym->st_info)==STT_FUNC) x|=1;
        x-=4; /*FIXME maybe drop 4 and add_val below*/
        x=((long)x>>1);
        store_val(where,MASK(11),x>>11);
        store_val(where,MASK(11)<<16,(x&0x7ff)<<16);
      }
      break;
    case R_ARM_THM_MOVW_ABS_NC:
      s+=a;
      if (ELF_ST_TYPE(sym->st_info)==STT_FUNC) s|=1;
      s&=0xffff;
      s=((s>>12)&0xf)|(((s>>11)&0x1)<<10)|((s&0xff)<<16)|(((s>>8)&0x7)<<28);
      add_vals(where,~0L,s);
      break;
    case R_ARM_THM_MOVW_ABS:
      s+=a;
      s>>=16;
      s=((s>>12)&0xf)|(((s>>11)&0x1)<<10)|((s&0xff)<<16)|(((s>>8)&0x7)<<28);
      add_vals(where,~0L,s);
      break;
    case R_ARM_ABS32:
      add_vals(where,~0L,s+a);
      break;
