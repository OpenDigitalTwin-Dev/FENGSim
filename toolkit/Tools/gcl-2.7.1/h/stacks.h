#ifndef VSSIZE
#define VSSIZE 512*1024
#endif

#define VSGETA 128
object value_stack[VSSIZE + (STACK_OVER +1) *VSGETA],*vs_org=value_stack,*vs_limit=value_stack+VSSIZE;     

#ifndef BDSSIZE
#define BDSSIZE		8*1024
#endif
#define	BDSGETA		64
struct bds_bd bind_stack[BDSSIZE + (STACK_OVER +1)* BDSGETA],*bds_org=bind_stack,*bds_limit=bind_stack+BDSSIZE;

     
#ifndef IHSSIZE
#define	IHSSIZE		32*1024
#endif
#define	IHSGETA		96
struct invocation_history ihs_stack[IHSSIZE + (STACK_OVER +1) * IHSGETA],*ihs_org=ihs_stack,*ihs_limit=ihs_stack+IHSSIZE;     


#ifndef FRSSIZE
#define FRSSIZE		8*1024
#endif
#define	FRSGETA		96
struct frame frame_stack[FRSSIZE + (STACK_OVER +1) * FRSGETA],*frs_org=frame_stack,*frs_limit=frame_stack+FRSSIZE;

