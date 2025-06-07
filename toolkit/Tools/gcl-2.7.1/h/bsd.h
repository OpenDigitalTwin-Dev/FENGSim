#define BSD 1
#define UNIX
#define AV
#define SFASL

/* #define HAVE_AOUT <a.out.h> */


#define MEM_SAVE_LOCALS	\
  struct exec header;\
  int stsize

#define READ_HEADER 	fread(&header, sizeof(header), 1, original); \
	data_begin=DATA_BEGIN; \
	data_end = core_end; \
	original_data = header.a_data; \
	header.a_data = data_end - data_begin; \
	header.a_bss = 0; \
	fwrite(&header, sizeof(header), 1, save);

#define FILECPY_HEADER \
	filecpy(save, original, header.a_text - sizeof(header));

#define  COPY_TO_SAVE \
  filecpy(save, original, header.a_syms+header.a_trsize+header.a_drsize); \
  fread(&stsize, sizeof(stsize), 1, original); \
  fwrite(&stsize, sizeof(stsize), 1, save); \
filecpy(save, original, stsize - sizeof(stsize))


#define NUMBER_OPEN_FILES getdtablesize() 


extern char etext[];

#define INIT_ALLOC heap_end = core_end = PCEI(sbrk(0),PAGESIZE);

#define SYM_EXTERNAL_P(sym) ((sym)->n_type & N_EXT)
     
#define cs_check(x) 


#define LD_COMMAND(command,main,start,input,ldarg,output) \
  sprintf(command, "ld -d -N -x -A %s -T %x %s %s -o %s", \
            main,start,input,ldarg,output)

#define SYM_UNDEF_P(sym) ((N_SECTION(sym)) == N_UNDEF)
#define NUM_AUX(sym) 0

       /* the section like N_ABS,N_TEXT,.. */


  /* We have socket utilities, and can fork off a process
   and get a stream connection with it */
#define RUN_PROCESS

/* #define HAVE_XDR */

#define WANT_VALLOC  

  /* if there is no input there return false */
#define LISTEN_FOR_INPUT(fp) \
  if(((FILE *)fp)->_cnt <=0 && (c=0,ioctl(((FILE *)fp)->_file, FIONREAD, &c),c<=0)) \
     return 0

 /* have sys/ioctl.h */
#define HAVE_IOCTL
  

#define HAVE_SIGVEC
  
