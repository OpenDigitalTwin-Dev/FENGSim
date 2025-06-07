#define VFUN_NARG_BIT   (1 <<11) 
#define MVRET_BIT       (1 <<10) 
#define SFUN_RETURN_MASK 0x300
#define SFUN_ARG_TYPE_MASK (~0xfff)

#define SFUN_RETURN_TYPE(s) \
  ((enum ftype)(((s) & SFUN_RETURN_MASK) >> 8))

#define SFUN_START_ARG_TYPES(x) (x=(x>>10))
#define SFUN_NEXT_TYPE(x) ((enum ftype)((x=(x>>2))& 3))

#define MAX_C_ARGS 9

/*          ...xx|xx|xxxx|xxxx|   
       ret  Narg     */

/*    a9a8a7a6a5a4a3a4a3a2a1a0rrrrnnnnnnnn
         ai=argtype(i)         ret   nargs
 */
#define SFUN_NARGS(x)   (x & 0xff) /* 8 bits */
#define RESTYPE(x)      (x<<8)   /* 2 bits */
/* set if the VFUN_NARGS = m ; has been set correctly */
#define ARGTYPE(i,x)    ((x) <<(12+(i*2)))
#define ARGTYPE1(x)     (1 | ARGTYPE(0,x))
#define ARGTYPE2(x,y)   (2 | ARGTYPE(0,x)  | ARGTYPE(1,y))
#define ARGTYPE3(x,y,z) (3 | ARGTYPE(0,x) | ARGTYPE(1,y) | ARGTYPE(2,z))

