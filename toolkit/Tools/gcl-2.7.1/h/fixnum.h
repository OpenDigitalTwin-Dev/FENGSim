#if defined (LOW_SHFT)

#define LOW_IM_FIX (1L<<(LOW_SHFT-1))
#define INT_IN_BITS(a_,b_) ({fixnum _a=(fixnum)(a_);_a>>(b_)==_a>>(CHAR_SIZE*SIZEOF_LONG-1);})

#define      make_imm_fixnum(a_)        ((object)(fixnum)a_)
#define       fix_imm_fixnum(a_)        ((fixnum)a_)
#define      mark_imm_fixnum(a_)        ({if (is_unmrkd_imm_fixnum(a_)) (a_)=((object)((fixnum)(a_)+(LOW_IM_FIX<<1)));})
#define    unmark_imm_fixnum(a_)        ({if (is_marked_imm_fixnum(a_)) (a_)=((object)((fixnum)(a_)-(LOW_IM_FIX<<1)));})
#define        is_imm_fixnum(a_)        ((fixnum)(a_)>=-LOW_IM_FIX && ((fixnum)(a_)<(fixnum)OBJNULL))/* (labs((fixnum)(a_))<=(fixnum)OBJNULL) */
#define is_unmrkd_imm_fixnum(a_)        is_imm_fix(a_)/* (labs((fixnum)(a_))<=LOW_IM_FIX) */
#define is_marked_imm_fixnum(a_)        ((fixnum)(a_)>=LOW_IM_FIX && ((fixnum)(a_)<(fixnum)OBJNULL))/* (is_imm_fixnum(a_)&&!is_unmrkd_imm_fixnum(a_)) */
#define           is_imm_fix(a_)        INT_IN_BITS(a_,LOW_SHFT-1)
#elif defined (IM_FIX_BASE) && defined(IM_FIX_LIM)
#define      make_imm_fixnum(a_)        ((object)((a_)+(IM_FIX_BASE+(IM_FIX_LIM>>1))))
#define       fix_imm_fixnum(a_)        ((fixnum)(((ufixnum)(a_))-(IM_FIX_BASE+(IM_FIX_LIM>>1))))
#define      mark_imm_fixnum(a_)        ((a_)=((object)(((ufixnum)(a_)) | IM_FIX_LIM)))
#define    unmark_imm_fixnum(a_)        ((a_)=((object)(((ufixnum)(a_)) &~ IM_FIX_LIM)))
#define        is_imm_fixnum(a_)        (((ufixnum)(a_))>=IM_FIX_BASE)
#define is_unmrkd_imm_fixnum(a_)        (is_imm_fixnum(a_)&&!is_marked_imm_fixnum(a_))
#define is_marked_imm_fixnum(a_)        (((ufixnum)(a_))&IM_FIX_LIM)
#define           is_imm_fix(a_)        (!(((a_)+(IM_FIX_LIM>>1))&-IM_FIX_LIM))
/* #define        un_imm_fixnum(a_)        ((a_)=((object)(((fixnum)(a_))&~(IM_FIX_BASE)))) */
#else
#define      make_imm_fixnum(a_)        make_fixnum1(a_)
#define       fix_imm_fixnum(a_)        ((a_)->FIX.FIXVAL)
#define      mark_imm_fixnum(a_)        
#define    unmark_imm_fixnum(a_)        
#define        is_imm_fixnum(a_)        0
#define is_unmrkd_imm_fixnum(a_)        0
#define is_marked_imm_fixnum(a_)        0
#define           is_imm_fix(a_)        0
/* #define        un_imm_fixnum(a_)         */
#endif

#define make_fixnum(a_)  ({register fixnum _q1=(a_);register object _q3;\
                          _q3=is_imm_fix(_q1) ? make_imm_fixnum(_q1) : make_fixnum1(_q1);_q3;})
#define CMPmake_fixnum(a_) make_fixnum(a_)/*FIXME*/
#define fix(a_)          ({register object _q2=(a_);register fixnum _q4;\
                          _q4=is_imm_fixnum(_q2) ? fix_imm_fixnum(_q2) :  (_q2)->FIX.FIXVAL;_q4;})
#define Mfix(a_)         fix(a_)
#define small_fixnum(a_) make_fixnum(a_) /*make_imm_fixnum(a_)*/
#define set_fix(a_,b_)   ((a_)->FIX.FIXVAL=(b_))

