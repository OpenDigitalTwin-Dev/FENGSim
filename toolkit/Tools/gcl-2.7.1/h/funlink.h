#ifndef FUNLINK_H
#define FUNLINK_H

/* the link_desc, is an INT which carries the call information
   for all uses of that link.   It tells whether fcall.nargs is
   set before the call, whether the VFUN_FUN is set, (to pass in
   a closure function) or if the number of values is set after the
   call.  It gives the min and max number of args and the result
   type expected.   It describes the arg types.
   enum F_arg_flags


*/
    
/*
A link arg descriptor:
    a6a5a4a3a2a1a0rrmmmmmmfffllllll
    l = least number of args passed
    m = max number of args passed
    f = flags bits set according to F_arg_flags, There are F_end flag bits.
    r = result type in F_arg_types
    ai = i'th arg type in F_arg_types
*/

/* 2^6 is the limit on the number of args */
#define F_NARG_WIDTH 6
#define F_START_TYPES_POS   (2* F_NARG_WIDTH + F_end )
enum F_arg_flags
{ F_requires_nargs, /* if set, then caller must store VFUN_NARGS with number
		       of args passed.   F_ARGD is used to set up the argd,
		       and it sets this if minargs < maxargs.   */
  F_caller_sets_one_val, /* If set, then the CALLER will look after setting the
		       fcall.nvalues to 1, if necessary (eg the call is at the
		       end of a function, or if multiple-values-list invokes
		       the function.)  If foo is proclaimed to return exactly
		       one value, then the CALLER might set this flag in the
		       link argd, or it might do it in the case we have (setq
		       x (foo)) or (values (foo)).   
		      
		       If this flag is not set, then the CALLED function is
		       responsible for setting the number of values in
		       fcall.nvalues, and also for always returning as C value
		       Cnil, in the case that it sets fcall.nvalues == 0.  */
  F_requires_fun_passed, /* if set, the caller must set VFUN_FUN to the
			    calling function.  This is used by closures, but
			    could be used by other things i suppose. */
  F_end               /* 1 bigger than the largest flag */
  };
enum F_arg_types
{ F_object,
  F_int,  
  F_double_ptr,
  F_shortfloat  
  };

/* Make a mask for bits i < j, masking j-i bits */
#define MASK_RANGE(i,j)  ((~(~0UL << (j-i)))<< i)

#define F_PLAIN(x) (((x) & MASK_RANGE( F_START_TYPES_POS,31)) == 0)
#define ARG_LIMIT 63

/* We allow 2 bits for encoding arg types and return type */
#define F_TYPE_WIDTH 2
#define F_MIN_ARGS(x) (x & MASK_RANGE(0,F_NARG_WIDTH))
#define F_NARGS(x) F_MIN_ARGS(x)
#define F_ARG_FLAGS_P(x,flag) (x & (1 << (F_NARG_WIDTH +  flag)))
#define F_ARG_FLAGS(x) ((x >> F_NARG_WIDTH) & MASK_RANGE(0,F_end))
#define F_MAX_ARGS(x) ((x >> (F_NARG_WIDTH + F_end )) \
		       & MASK_RANGE(0,F_NARG_WIDTH))

#define BITS_PER_CHAR 8
#define MAX_ARGS 63
#define F_TYPES(x) (((x) >> F_START_TYPES_POS ) \
		       & MASK_RANGE(0, sizeof(int)*BITS_PER_CHAR - F_START_TYPES_POS))
#define F_RESULT_TYPE(x) (F_TYPES(x) & MASK_RANGE(0,F_TYPE_WIDTH))
#define F_ARG_LIMIT ((1<< F_NARG_WIDTH) -1)

/* make an argd slot
   where flags and argtypes are already set up as fields
 */		       
#define F_ARGD(min,max,flags, argtypes) \
      (min | ((flags | (max-min ? (1<<F_requires_nargs) : 0) \
	       << F_NARG_WIDTH)) \
       | (max << (F_NARG_WIDTH+F_end)) \
       | (argtypes<< (2* F_NARG_WIDTH + F_end )))

#define ONE_VAL  (1 << (F_NARG_WIDTH + F_caller_sets_one_val))
#define CLOS (1 << (F_NARG_WIDTH + F_requires_fun_passed))
#define VARARG (1 << (F_NARG_WIDTH + F_requires_nargs))
   /* the following may be used as an argument to DEFUN even in the case
      of varargs, since the F_ARGD macro detects minargs<maxargs and sets this.*/
#define NONE 0    

/* we dont want to define all these two letter macros... all the time */

#ifndef NO_DEFUN
#define OO (F_object | F_object << F_TYPE_WIDTH)
#define OI (F_object | F_int << F_TYPE_WIDTH)
#define OD (F_object | F_double_ptr << F_TYPE_WIDTH)
#define IO (F_int | F_object << F_TYPE_WIDTH)
#define II (F_int | F_int << F_TYPE_WIDTH)
#define ID (F_int | F_double_ptr << F_TYPE_WIDTH)
#define DO (F_double_ptr | F_object << F_TYPE_WIDTH)
#define DI (F_double_ptr | F_int << F_TYPE_WIDTH)
#define DD (F_double_ptr | F_double_ptr << F_TYPE_WIDTH)
#endif

#define ARGTYPES(a,b,c,d) \
  (a | (b << (2* F_TYPE_WIDTH)) | (c << (4* F_TYPE_WIDTH)) | (d << (6*F_TYPE_WIDTH)))


#define PUSH_FIRST_VAL(x) int nvals = 1 ; object result = (x)
#define PUSH_VAL(x) if (nvals>=sizeof(fcall.values)/sizeof(*fcall.values) \
                       FEerror("Too many function call values"); \
                     else fcall.values[nvals++] = (x)
#define RETURN_VALS   fcall.nvalues= nvals; return result;} 0

#define FUNCALL(n,form) (VFUN_NARGS=n,form)
 
#endif
