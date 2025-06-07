;; Copyright (C) 2024 Camm Maguire
(in-package :compiler)

;; The optimizers have been redone to allow more flags
;; The old style optimizations  correspond to the first 2
;; flags.  
;; ( arglist result-type flags {string | function})

;; meaning of the flags slot.
;       '((allocates-new-storage ans); might invoke gbc
;	 (side-effect-p set)        ; no effect on arguments
;	 (constantp)                ; always returns same result,
;	                            ;double eval ok.
;	 (result-type-from-args rfa); if passed args of matching
;					;type result is of result type
;         (is)))                     ;; extends the `integer stack'.
;    (cond ((member flag v :test 'eq)
;
;;;   valid properties are 'inline-always 'inline-safe 'inline-unsafe

;; Note:   The order of the properties is important, since the first
;; one whose arg types and result type can be matched will be chosen.


(or (fboundp 'flags) (load "../cmpnew/cmpeval.lsp"))

;;BOOLE
(push '((t t t) t #.(compiler::flags) "immnum_bool(#0,#1,#2)") (get 'boole 'compiler::inline-always))
(push '((fixnum t t) t #.(compiler::flags) "immnum_boole(#0,#1,#2)") (get 'boole 'compiler::inline-always))

;;BOOLE3
; (push '((fixnum fixnum fixnum) fixnum #.(flags rfa)INLINE-BOOLE3)
;   (get 'boole3 'inline-always))

;;FP-OKP
 (push '((t) boolean #.(flags set rfa)
  "@0;(type_of(#0)==t_stream? ((#0)->sm.sm_fp)!=0: 0 )")
   (get 'fp-okp 'inline-unsafe))
(push '((stream) boolean #.(flags set rfa)"((#0)->sm.sm_fp)!=0")
   (get 'fp-okp 'inline-unsafe))

;;LDB1
 (push '((fixnum fixnum fixnum) fixnum #.(flags)
  "((((~(-1 << (#0))) << (#1)) & (#2)) >> (#1))")
   (get 'si::ldb1 'inline-always))

;;LONG-FLOAT-P
 (push '((t) boolean #.(flags rfa)"type_of(#0)==t_longfloat")
   (get 'long-float-p 'inline-always))

;;COMPLEX-P
 (push '((t) boolean #.(flags)"type_of(#0)==t_complex")
   (get 'si::complexp 'inline-always))

;;SFEOF
 (push `((t) boolean #.(flags set rfa) ,(lambda (x) (add-libc "feof") (wt "(((int(*)(void *))dlfeof)((" x ")->sm.sm_fp))")))
   (get 'sfeof 'inline-unsafe))


;;SGETC1
 (push `((t) fixnum #.(flags set rfa) ,(lambda (x) (add-libc "getc") (wt "(((int(*)(void *))dlgetc)((" x ")->sm.sm_fp))")))
   (get 'sgetc1 'inline-unsafe))

;;SPUTC
 (push `((fixnum t) fixnum #.(flags set rfa) ,(lambda (x y) (add-libc "putc") (wt "(((int(*)(int,void *))dlputc)(" x ",(" y ")->sm.sm_fp))")))
   (get 'sputc 'inline-always))
(push `((character t) fixnum #.(flags set rfa) ,(lambda (x y) (add-libc "putc") (wt "(((int(*)(int,void *))dlputc)(char_code(" x "),(" y ")->sm.sm_fp))")))
   (get 'sputc 'inline-always))

;;FORK
 (push `(() t #.(flags) ,(lambda nil (add-libc "memset")(add-libc "pipe")(add-libc "close")(add-libc "fork")(wt "myfork()")))
   (get 'si::fork 'inline-unsafe))

;;READ-POINTER-OBJECT
 (push '((t) t #.(flags ans set)"read_pointer_object(#0)")
   (get 'si::read-pointer-object 'inline-unsafe))

;;WRITE-POINTER-OBJECT
 (push '((t t) t #.(flags ans set)"write_pointer_object(#0,#1)")
   (get 'si::write-pointer-object 'inline-unsafe))

;;READ-BYTE1
 ;; (push '((t t) t #.(flags rfa ans set)"read_byte1(#0,#1)")
 ;;   (get 'read-byte1 'inline-unsafe))

;;READ-CHAR1
 (push '((t t) t #.(flags rfa ans set)"read_char1(#0,#1)")
   (get 'read-char1 'inline-unsafe))

;;SHIFT<<
 (push '((fixnum fixnum) fixnum #.(flags)"((#0) << (#1))")
   (get 'shift<< 'inline-always))

;;SHIFT>>
 (push '((fixnum fixnum) fixnum #.(flags set rfa)"((#0) >> (- (#1)))")
   (get 'shift>> 'inline-always))

;;SHORT-FLOAT-P
 (push '((t) boolean #.(flags rfa)"type_of(#0)==t_shortfloat")
   (get 'short-float-p 'inline-always))

;;SIDE-EFFECTS
 (push '(nil t #.(flags)"Ct")
   (get 'side-effects 'inline-always))

;;STACK-CONS  ;;FIXME update this
; (push '((fixnum t t) t #.(flags)
;  "(STcons#0.t=t_cons,STcons#0.m=0,STcons#0.c_car=(#1),
;              STcons#0.c_cdr=(#2),(object)&STcons#0)")
;   (get 'stack-cons 'inline-always))

;;SUBLIS1
;;  (push '((t t t) t #.(flags rfa ans set)SUBLIS1-INLINE)
;;    (get 'sublis1 'inline-always))

;;FIXME the MAX and MIN optimized  arg evaluations aren't logically related to side effects
;;      but we need to save the intermediate results in any case to avoid exponential
;;      growth in nested expressions.  set added to flags for now here and in analogous
;;      constructs involving ?.  CM 20041129

;;ABS
; (si::putprop 'abs 'abs-propagator 'type-propagator)
 (push '((t) t #.(compiler::flags) "immnum_abs(#0)") (get 'abs 'compiler::inline-always))       
 (push '(((integer #.(1+ most-negative-fixnum) #.most-positive-fixnum)) (integer 0 #.most-positive-fixnum) #.(flags)"abs(#0)")
   (get 'abs 'inline-always))
 (push '((short-float) (short-float 0.0) #.(flags)"fabs(#0)") ;;FIXME ranged floating point types
   (get 'abs 'inline-always))
 (push '((long-float) (long-float 0.0) #.(flags)"fabs(#0)")
   (get 'abs 'inline-always))
 (push '(((real 0.0)) t #.(flags)"#0")
   (get 'abs 'inline-always))
 (push '(((and cnum (real 0.0))) cnum #.(flags)"#0")
   (get 'abs 'inline-always))

;;VECTOR-TYPE
 (push '((t fixnum) boolean #.(flags rfa)
  "@0;(type_of(#0) == t_vector && (#0)->v.v_elttype == (#1))")
   (get 'vector-type 'inline-always))

;; ;;SYSTEM:ASET
;; (push '((t t t) t #.(flags set)"aset1(#1,fixint(#2),#0)")
;;    (get 'system:aset 'inline-always))
;; (push '((t t fixnum) t #.(flags set)"aset1(#1,#2,#0)")
;;    (get 'system:aset 'inline-always))
;; (push '((t t t) t #.(flags set)"aset1(#1,fix(#2),#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((t (array t) fixnum) t #.(flags set)"(#1)->v.v_self[#2]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((character (array character) fixnum) character #.(flags rfa set)"(#1)->ust.ust_self[#2]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array fixnum) fixnum) fixnum #.(flags set rfa)"(#1)->fixa.fixa_self[#2]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array signed-short) fixnum) fixnum #.(flags rfa set)"((short *)(#1)->ust.ust_self)[#2]=(#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array signed-char) fixnum) fixnum #.(flags rfa set)"((#1)->ust.ust_self)[#2]=(#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array unsigned-short) fixnum) fixnum #.(flags rfa set)
;;   "((unsigned short *)(#1)->ust.ust_self)[#2]=(#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array unsigned-char) fixnum) fixnum #.(flags rfa set)"((#1)->ust.ust_self)[#2]=(#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((short-float (array short-float) fixnum) short-float #.(flags rfa set)"(#1)->sfa.sfa_self[#2]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((long-float (array long-float) fixnum) long-float #.(flags rfa set)"(#1)->lfa.lfa_self[#2]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((t t t t) t #.(flags set)
;;   "@1;aset(#1,fix(#2)*(#1)->a.a_dims[1]+fix(#3),#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((t (array t) fixnum fixnum) t #.(flags set)
;;   "@1;(#1)->a.a_self[(#2)*(#1)->a.a_dims[1]+#3]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((character (array character) fixnum fixnum) character
;; 	#.(flags rfa set)
;;   "@1;(#1)->ust.ust_self[(#2)*(#1)->a.a_dims[1]+#3]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((fixnum (array fixnum) fixnum fixnum) fixnum #.(flags set rfa)
;;   "@1;(#1)->fixa.fixa_self[(#2)*(#1)->a.a_dims[1]+#3]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((short-float (array short-float) fixnum fixnum) short-float #.(flags rfa set)
;;   "@1;(#1)->sfa.sfa_self[(#2)*(#1)->a.a_dims[1]+#3]= (#0)")
;;    (get 'system:aset 'inline-unsafe))
;; (push '((long-float (array long-float) fixnum fixnum) long-float #.(flags rfa set)
;;   "@1;(#1)->lfa.lfa_self[(#2)*(#1)->a.a_dims[1]+#3]= (#0)")
;;    (get 'system:aset 'inline-unsafe))

;;SYSTEM:FILL-POINTER-SET
 (push '((t fixnum) seqind #.(flags rfa set)"(((#0)->st.st_fillp)=(((#0)->st.st_hasfillp) ? (#1) : ((#0)->st.st_fillp)))")
   (get 'system:fill-pointer-set 'inline-unsafe))
 (push '(((vector) seqind) seqind #.(flags rfa set)"(((#0)->st.st_fillp)=(((#0)->st.st_hasfillp) ? (#1) : ((#0)->st.st_fillp)))")
   (get 'system:fill-pointer-set 'inline-always))

;;SYSTEM:FIXNUMP
;;  (push '((t) boolean #.(flags rfa)"type_of(#0)==t_fixnum")
;;    (get 'system:fixnump 'inline-always))
;; (push '((fixnum) boolean #.(flags rfa)"1")
;;    (get 'system:fixnump 'inline-always))

;;SYSTEM:SEQINDP
;;  (push '((t) boolean #.(flags rfa) #.(format nil "(type_of(#0)==t_fixnum && ({fixnum _t=fix(#0);_t>=0 && _t<=~s;}))" array-dimension-limit))
;;    (get 'system::seqindp 'inline-always))
;; (push '((fixnum) boolean #.(flags rfa)#.(format nil "(#0>=0 && #0<=~s)" array-dimension-limit))
;;    (get 'system::seqindp 'inline-always))
;; (push '((seqind) boolean #.(flags rfa)"1")
;;    (get 'system::seqindp 'inline-always))

;;SYSTEM:HASH-SET
(push '((t t t) t #.(flags rfa) "@2;(sethash(#0,#1,#2),#2)") (get 'si::hash-set 'inline-always));FIXME
;(push '((t t t) t #.(flags rfa) "@2;(sethash_with_check(#0,#1,#2),#2)") (get 'si::hash-set 'inline-always))

;;SYSTEM:MV-REF
 (push '((fixnum) t #.(flags)"(MVloc[(#0)])")
   (get 'system:mv-ref 'inline-always))

;;SYSTEM:PUTPROP
 (push '((t t t) t #.(flags set)"putprop(#0,#1,#2)")
   (get 'system:putprop 'inline-always))

;;SYSTEM:SET-MV
 (push '((fixnum t) t #.(flags)"(MVloc[(#0)]=(#1))")
   (get 'system:set-mv 'inline-always))

;;SYSTEM:SPUTPROP
 (push '((symbol t t) t #.(flags set)"fSsputprop(#0,#1,#2)")
   (get 'system:sputprop 'inline-always))

;;SYSTEM:STRUCTURE-DEF
 (push '((t) t #.(flags)"(#0)->str.str_def")
   (get 'system:structure-def 'inline-unsafe))
 (push '((structure) structure #.(flags)"(#0)->str.str_def")
   (get 'system:structure-def 'inline-always))

;;SYSTEM:STRUCTURE-LENGTH
 ;; (push '((t) fixnum #.(flags rfa)"S_DATA(#0)->length")
 ;;   (get 'system:structure-length 'inline-unsafe))

;;SYSTEM:STRUCTURE-REF
 (push '((t t fixnum) t #.(flags ans)"structure_ref(#0,#1,#2)")
   (get 'system:structure-ref 'inline-always))

;;SYSTEM:STRUCTURE-SET
 (push '((t t fixnum t) t #.(flags set)"structure_set(#0,#1,#2,#3)")
   (get 'system:structure-set 'inline-always))


;;SYSTEM:gethash1
 ;; (push '((t t) t #.(flags)"({struct htent *e=gethash(#0,#1);e->hte_key != OBJNULL ? e->hte_value : Cnil;})")
 ;;   (get 'system:gethash1 'inline-always))

;;SYSTEM:SVSET
;;  (push '((t t t) t #.(flags set)"aset1(#0,fixint(#1),#2)")
;;    (get 'system:svset 'inline-always))
;; (push '((t fixnum t) t #.(flags set)"aset1(#0,#1,#2)")
;;    (get 'system:svset 'inline-always))
;; (push '((t t t) t #.(flags set)"((#0)->v.v_self[fix(#1)]=(#2))")
;;    (get 'system:svset 'inline-unsafe))
;; (push '((t fixnum t) t #.(flags set)"(#0)->v.v_self[#1]= (#2)")
;;    (get 'system:svset 'inline-unsafe))

;;ASH
;(si::putprop 'ash 'ash-propagator 'type-propagator)
(push '((t t) t #.(compiler::flags) "immnum_shft(#0,#1)") (get 'ash 'compiler::inline-always))
(push '(((integer 0 0) t) fixnum #.(flags rfa)"0")
   (get 'ash 'inline-always))
(push '((fixnum (integer 0 #.(integer-length most-positive-fixnum))) fixnum #.(flags)"((#0)<<(#1))")
   (get 'ash 'inline-always))
(push '((fixnum (integer #.most-negative-fixnum 0)) fixnum #.(flags set)
	#.(concatenate 'string "@1;(-(#1)&"
		       (write-to-string (logxor -1 (integer-length most-positive-fixnum)))
		       "? ((#0)>=0 ? 0 : -1) : (#0)>>-(#1))"))
   (get 'ash 'inline-always))


;;+
(push '((t t) t #.(flags ans)"immnum_plus(#0,#1)") (get 'si::number-plus 'inline-always))
(push '((cnum cnum) cnum #.(flags)"(#0)+(#1)") (get 'si::number-plus 'inline-always))


;;-
;(push '((t) t #.(flags ans)"immnum_negate(#0)") (get '- 'inline-always))
;(push '((cnum) cnum #.(flags)"-(#0)") (get '- 'inline-always))
;(push '(((integer #.most-negative-fixnum #.most-negative-fixnum)) t #.(flags)"immnum_negate(#0)") (get '- 'inline-always))

(push '((t t) t #.(flags ans)"immnum_minus(#0,#1)") (get 'si::number-minus 'inline-always))
(push '((cnum cnum) cnum #.(flags)"(#0)-(#1)") (get 'si::number-minus 'inline-always))
(push '(((integer 0 0) t) t #.(flags ans)"immnum_negate(#1)") (get 'si::number-minus 'inline-always))
(push '(((integer 0 0) cnum) cnum #.(flags ans)"-(#1)") (get 'si::number-minus 'inline-always))

;;*
;(si::putprop '* 'super-range 'type-propagator)
(push '((t t) t #.(flags ans)"immnum_times(#0,#1)") (get 'si::number-times 'inline-always))
(push '((fixnum fixnum) integer #.(flags ans rfa)"safe_mul(#0,#1)") (get 'si::number-times 'inline-always))
(push '((cnum cnum) cnum #.(flags)"(#0)*(#1)") (get 'si::number-times 'inline-always))


;;/
(push '((t t) t #.(flags ans) "number_divide(#0,#1)") (get 'si::number-divide 'inline-always))
(push '((cnum cnum) cnum #.(flags) "(#0)/(#1)") (get 'si::number-divide 'inline-always))

;;/=
 (push '((t t) boolean #.(flags rfa)"immnum_ne(#0,#1)")
   (get '/= 'inline-always))
(push '((cnum cnum) boolean #.(flags rfa)"(#0)!=(#1)") (get '/= 'inline-always))

;;<
 (push '((t t) boolean #.(flags rfa)"immnum_lt(#0,#1)") (get '< 'inline-always))
(push '((creal creal) boolean #.(flags rfa)"(#0)<(#1)") (get '< 'inline-always))

;;compiler::objlt
 (push '((t t) boolean #.(flags rfa)"((object)(#0))<((object)(#1))") (get 'si::objlt 'inline-always))

;;<=
 (push '((t t) boolean #.(flags rfa)"immnum_le(#0,#1)") (get '<= 'inline-always))
(push '((creal creal) boolean #.(flags rfa)"(#0)<=(#1)") (get '<= 'inline-always))

;;=
 (push '((t t) boolean #.(flags rfa)"immnum_eq(#0,#1)") (get '= 'inline-always))
(push '((cnum cnum) boolean #.(flags rfa)"(#0)==(#1)") (get '= 'inline-always))

;;>
 (push '((t t) boolean #.(flags rfa)"immnum_gt(#0,#1)") (get '> 'inline-always))
(push '((creal creal) boolean #.(flags rfa)"(#0)>(#1)") (get '> 'inline-always))

;;>=
 (push '((t t) boolean #.(flags rfa)"immnum_ge(#0,#1)") (get '>= 'inline-always))
(push '((creal creal) boolean #.(flags rfa)"(#0)>=(#1)") (get '>= 'inline-always))

;;APPEND
;;  (push '((t t) t #.(flags ans)"append(#0,#1)")
;;    (get 'append 'inline-always))

;;ARRAY-DIMENSION
;(push '((t fixnum) fixnum #.(flags rfa)"@01;(type_of(#0)==t_array ? (#0)->a.a_dims[(#1)] : (#0)->v.v_dim)")
;   (get 'array-dimension 'inline-unsafe))

;;CMP-ARRAY-DIMENSION
;; (setf (symbol-function 'cmp-array-dimension) (symbol-function 'array-dimension))
;; (push '(cmp-array-dimension-inline-types nil #.(flags itf) cmp-array-dimension-inline)
;;    (get 'cmp-array-dimension 'inline-always))

;;ARRAY-TOTAL-SIZE
 (push '((t) fixnum #.(flags rfa)"((#0)->st.st_dim)")
   (get 'array-total-size 'inline-unsafe))

;;ARRAYP
 (push '((t) boolean #.(flags rfa)
  "@0;({enum type _tp=type_of(#0);_tp>=t_string && _tp<=t_array;})")
   (get 'arrayp 'inline-always))

;;ATOM
 (push '((t) boolean #.(flags rfa)"atom(#0)")
   (get 'atom 'inline-always))

;;BIT-VECTOR-P
 (push '((t) boolean #.(flags rfa)"({enum type tp=type_of(#0);tp==t_bitvector||tp==t_simple_bitvector;})")
   (get 'bit-vector-p 'inline-always))

;;HASH-TABLE-P
 (push '((t) boolean #.(flags)"(type_of(#0)==t_hashtable)")
   (get 'hash-table-p 'inline-always))

;;RANDOM-STATE-P
 (push '((t) boolean #.(flags)"(type_of(#0)==t_random)")
   (get 'random-state-p 'inline-always))

;;RANDOM-STATE-P
 (push '((t) boolean #.(flags)"(type_of(#0)==t_random)")
   (get 'random-state-p 'inline-always))

;;PACKAGEP
 (push '((t) boolean #.(flags)"(type_of(#0)==t_package)")
   (get 'packagep 'inline-always))

;;STREAMP
 (push '((t) boolean #.(flags)"(type_of(#0)==t_stream)")
   (get 'streamp 'inline-always))

;;READTABLEP
 (push '((t) boolean #.(flags)"(type_of(#0)==t_readtable)")
   (get 'readtablep 'inline-always))

;;COMPOUND PREDICATES
;; (dolist (l '(integerp rationalp floatp realp numberp vectorp arrayp compiled-function-p))
;;   (push
;;    `((t) boolean #.(flags) ,(substitute #\_ #\- (concatenate 'string (string-downcase l) "(#0)")))
;;    (get l 'inline-always)))


;;BOUNDP
 (push '((t) boolean #.(flags rfa)"(#0)->s.s_dbind!=OBJNULL")
   (get 'boundp 'inline-unsafe))
 (push '((symbol) boolean #.(flags rfa)"(#0)->s.s_dbind!=OBJNULL")
   (get 'boundp 'inline-always))

;;CONS-CAR
; (push '((list) t #.(flags rfa)"(#0)->c.c_car") (get 'si::cons-car 'inline-always))
;;CONS-CDR
; (push '((list) t #.(flags rfa)"(#0)->c.c_cdr") (get 'si::cons-cdr 'inline-always))

;;CHAR-CODE
; (push '((character) fixnum #.(flags rfa)"(#0)")
;   (get 'char-code 'inline-always))

;;CHAR/=
(push '((t t) boolean #.(flags rfa)"!eql(#0,#1)")
   (get 'char/= 'inline-unsafe))
(push '((t t) boolean #.(flags rfa)"char_code(#0)!=char_code(#1)")
   (get 'char/= 'inline-unsafe))
(push '((character character) boolean #.(flags rfa)"(#0)!=(#1)")
   (get 'char/= 'inline-unsafe))

;;CHAR<
 (push '((character character) boolean #.(flags rfa)"(#0)<(#1)")
   (get 'char< 'inline-always))

;;CHAR<=
 (push '((character character) boolean #.(flags rfa)"(#0)<=(#1)")
   (get 'char<= 'inline-always))

;;CHAR=
 (push '((t t) boolean #.(flags rfa)"eql(#0,#1)")
   (get 'char= 'inline-unsafe))
(push '((t t) boolean #.(flags rfa)"char_code(#0)==char_code(#1)")
   (get 'char= 'inline-unsafe))
(push '((character character) boolean #.(flags rfa)"(#0)==(#1)")
   (get 'char= 'inline-unsafe))

;;CHAR>
 (push '((character character) boolean #.(flags rfa)"(#0)>(#1)")
   (get 'char> 'inline-always))

;;CHAR>=
 (push '((character character) boolean #.(flags rfa)"(#0)>=(#1)")
   (get 'char>= 'inline-always))

;;CHARACTERP
 (push '((t) boolean #.(flags rfa)"type_of(#0)==t_character")
   (get 'characterp 'inline-always))

;;RPLACA
 (push '((cons t) t #.(flags)"@0;((#0)->c.c_car=(#1),(#0))")
   (get 'rplaca 'inline-always))
 (push '((t t) t #.(flags)"@0;((#0)->c.c_car=(#1),(#0))")
   (get 'rplaca 'inline-unsafe))

;;RPLACD
 (push '((cons t) t #.(flags)"@0;((#0)->c.c_cdr=(#1),(#0))")
   (get 'rplacd 'inline-always))
 (push '((t t) t #.(flags)"@0;((#0)->c.c_cdr=(#1),(#0))")
   (get 'rplacd 'inline-unsafe))

;;CODE-CHAR
; (push '((fixnum) character #.(flags)"(#0)")
;   (get 'code-char 'inline-always))

;;CONS
 (push '((t t) t #.(flags ans)"make_cons(#0,#1)")
   (get 'cons 'inline-always))
;; (push '((t t) dynamic-extent #.(flags ans)"ON_STACK_CONS(#0,#1)")
;;    (get 'cons 'inline-always))

;;CONSP
 (push '((t) boolean #.(flags rfa)"consp(#0)")
   (get 'consp 'inline-always))

;;DIGIT-CHAR-P
; (push '((character) (or null (integer 0 9)) #.(flags rfa)"@0; ((#0) <= '9' && (#0) >= '0')")
;   (get 'digit-char-p 'inline-always))

;;ENDP
 (push '((t) boolean #.(flags rfa)"endp(#0)")
       (get 'endp 'inline-safe))
;(push '((t) boolean #.(flags rfa)"(#0)==Cnil")
;      (get 'endp 'inline-unsafe))

;;EQ
 (push '((t t) boolean #.(flags rfa)"(#0)==(#1)")
   (get 'eq 'inline-always))
 (push '((cnum cnum) boolean #.(flags rfa)"(#0)==(#1)")
   (get 'eq 'inline-always))
;(push '((fixnum fixnum) boolean #.(flags rfa)"0")
;   (get 'eq 'inline-always))

;;EQL
 (push '((t t) boolean #.(flags rfa)"eql(#0,#1)")
       (get 'eql 'inline-always))
(push '((cnum cnum) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'eql 'inline-always))
(push '((character character) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'eql 'inline-always))
;;FIXME -- floats?

;;EQUAL
 (push '((t t) boolean #.(flags rfa)"equal(#0,#1)")
       (get 'equal 'inline-always))
(push '((cnum cnum) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equal 'inline-always))
(push '((character character) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equal 'inline-always))

;;EQUALP
 (push '((t t) boolean #.(flags rfa)"equalp(#0,#1)")
      (get 'equalp 'inline-always))
 (push '((fixnum fixnum) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equalp 'inline-always))
 (push '((short-float short-float) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equalp 'inline-always))
 (push '((long-float long-float) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equalp 'inline-always))
 (push '((character character) boolean #.(flags rfa)"(#0)==(#1)")
      (get 'equalp 'inline-always))

;;EXPT
 (push '((t t) t #.(flags ans)"number_expt(#0,#1)")
   (get 'expt 'inline-always))
(push `((fixnum fixnum) fixnum #.(flags) "fixnum_expt((#0),(#1))")
      (get 'expt 'inline-always))
(push `(((integer 2 2) fixnum) fixnum #.(flags) "(1L<<(#1))")
      (get 'expt 'inline-always))



;; ;;si::FILL-POINTER-INTERNAL
;;  (push '((t) seqind #.(flags rfa)"((#0)->v.v_fillp)")
;;    (get 'si::fill-pointer-internal 'inline-unsafe))
;;  (push '((vector) seqind #.(flags rfa)"((#0)->v.v_fillp)")
;;    (get 'si::fill-pointer-internal 'inline-always))

;;ARRAY-HAS-FILL-POINTER-P
 (push '((t) boolean #.(flags rfa)"((#0)->v.v_hasfillp)")
   (get 'array-has-fill-pointer-p 'inline-unsafe))
 (push '((vector) boolean #.(flags rfa)"((#0)->v.v_hasfillp)")
   (get 'array-has-fill-pointer-p 'inline-always))

;;FIRST
;;  (push '((t) t #.(flags)"car(#0)")
;;    (get 'first 'inline-safe))
;(push '((t) t #.(flags)"CMPcar(#0)")
;   (get 'first 'inline-unsafe))

;;FLOATP
 (push '((t) boolean #.(flags rfa)
  "@0;type_of(#0)==t_shortfloat||type_of(#0)==t_longfloat")
   (get 'floatp 'inline-always))

;;FLOOR
; (push '((fixnum fixnum) fixnum #.(flags rfa)
;  "@01;(#0>=0&&(#1)>0?(#0)/(#1):ifloor(#0,#1))")
;   (get 'floor 'inline-always))
;(si::putprop 'floor 'floor-propagator 'type-propagator)
(push '((t t) t #.(compiler::flags) "immnum_floor(#0,#1)") (get 'floor 'compiler::inline-always))
#+intdiv
(push '((fixnum fixnum) (returns-exactly fixnum fixnum) #.(flags rfa set)
	 "@01;({fixnum _t=(#0)/(#1);_t=((#0)<=0 && (#1)<=0) || ((#0)>=0 && (#1)>=0) || ((#1)*_t==(#0)) ? _t : _t-1;@1((#0)-_t*(#1))@ _t;})")
   (get 'floor 'inline-always))

;;CEILING
;(si::putprop 'ceiling 'floor-propagator 'type-propagator)
(push '((t t) t #.(compiler::flags) "immnum_ceiling(#0,#1)") (get 'ceiling 'compiler::inline-always))
#+intdiv
(push '((fixnum fixnum) (returns-exactly fixnum fixnum) #.(flags rfa set)
	 "@01;({fixnum _t=(#0)/(#1);_t=((#0)<=0 && (#1)>=0) || ((#0)>=0 && (#1)<=0) || ((#1)*_t==(#0)) ? _t : _t+1;@1((#0)-_t*(#1))@ _t;})")
   (get 'ceiling 'inline-always))

;;FOURTH
;;  (push '((t) t #.(flags)"cadddr(#0)")
;;    (get 'fourth 'inline-safe))
;(push '((t) t #.(flags)"CMPcadddr(#0)")
;   (get 'fourth 'inline-unsafe))

;;FIFTH
;;  (push '((t) t #.(flags)"cadr(cdddr(#0))")
;;    (get 'fifth 'inline-safe))
;(push '((t) t #.(flags)"CMPcadr(CMPcdddr(#0))")
;   (get 'fifth 'inline-unsafe))

;;SIXTH
;;  (push '((t) t #.(flags)"caddr(cdddr(#0))")
;;    (get 'sixth 'inline-safe))
;(push '((t) t #.(flags)"CMPcaddr(CMPcdddr(#0))")
;   (get 'sixth 'inline-unsafe))

;;SEVENTH
;;  (push '((t) t #.(flags)"cadddr(cdddr(#0))")
;;    (get 'seventh 'inline-safe))
;(push '((t) t #.(flags)"CMPcadddr(CMPcdddr(#0))")
;   (get 'seventh 'inline-unsafe))

;;EIGHTH
;;  (push '((t) t #.(flags)"cadr(cdddr(cdddr(#0)))")
;;    (get 'eighth 'inline-safe))
;(push '((t) t #.(flags)"CMPcadr(CMPcdddr(CMPcdddr(#0)))")
;   (get 'eighth 'inline-unsafe))

;;NINTH
;;  (push '((t) t #.(flags)"caddr(cdddr(cdddr(#0)))")
;;    (get 'ninth 'inline-safe))
;(push '((t) t #.(flags)"CMPcaddr(CMPcdddr(CMPcdddr(#0)))")
;   (get 'ninth 'inline-unsafe))

;;TENTH
;;  (push '((t) t #.(flags)"cadddr(cdddr(cdddr(#0)))")
;;    (get 'tenth 'inline-safe))
;(push '((t) t #.(flags)"CMPcadddr(CMPcdddr(CMPcdddr(#0)))")
;   (get 'tenth 'inline-unsafe))

;;GET
 (push '((t t t) t #.(flags)"get(#0,#1,#2)")
   (get 'get 'inline-always))
(push '((t t) t #.(flags)"get(#0,#1,Cnil)")
   (get 'get 'inline-always))

;;INTEGERP
 (push '((t) boolean #.(flags rfa)
  "@0;({enum type _tp=type_of(#0);_tp==t_fixnum||_tp==t_bignum;})")
   (get 'integerp 'inline-always))
(push '((fixnum) boolean #.(flags rfa)"1")
   (get 'integerp 'inline-always))


;;KEYWORDP
 (push '((t) boolean #.(flags rfa)
  "@0;(type_of(#0)==t_symbol&&(#0)->s.s_hpack==keyword_package)")
   (get 'keywordp 'inline-always))

;;ADDRESS
 (push '((t) fixnum #.(flags rfa)"((fixnum)(#0))")
   (get 'si::address 'inline-always))

;;NANI
 (push '((fixnum) t #.(flags rfa)"((object)(#0))")
   (get 'si::nani 'inline-always))


;;LENGTH
 (push '((t) fixnum #.(flags rfa)"length(#0)")
   (get 'length 'inline-always))
(push '((vector) seqind #.(flags rfa)"((#0)->v.v_hasfillp ? (#0)->v.v_fillp : (#0)->v.v_dim)")
   (get 'length 'inline-always))

;;LIST
(push '((t *) list #.(flags ans rfa) LIST-INLINE);proper-list can get bumped
   (get 'list 'inline-always))

;;LIST*
(push '((t *) list #.(flags ans rfa) LIST*-INLINE)
   (get 'list* 'inline-always))

;;CONS
 (push '((t t) t #.(flags ans) CONS-INLINE)
   (get 'cons 'inline-always))

;;LISTP
 (push '((t) boolean #.(flags rfa)"listp(#0)")
   (get 'listp 'inline-always))

;;si::spice-p
 (push '((t) boolean #.(flags)"@0;type_of(#0)==t_spice")
   (get 'si::spice-p 'inline-always))

;;LOGNAND
(push '((t t) t #.(compiler::flags) "immnum_nand(#0,#1)") (get 'lognand 'compiler::inline-always))
;;LOGNOR
(push '((t t) t #.(compiler::flags) "immnum_nor(#0,#1)") (get 'lognor 'compiler::inline-always))
;;LOGEQV
(push '((t t) t #.(compiler::flags) "immnum_eqv(#0,#1)") (get 'logeqv 'compiler::inline-always))

;;LOGANDC1
(push '((t t) t #.(compiler::flags) "immnum_andc1(#0,#1)") (get 'logandc1 'compiler::inline-always))
;;LOGANDC2
(push '((t t) t #.(compiler::flags) "immnum_andc2(#0,#1)") (get 'logandc2 'compiler::inline-always))
;;LOGORC1
(push '((t t) t #.(compiler::flags) "immnum_orc1(#0,#1)") (get 'logorc1 'compiler::inline-always))
;;LOGORC1
(push '((t t) t #.(compiler::flags) "immnum_orc2(#0,#1)") (get 'logorc2 'compiler::inline-always))


;;LOGAND
 (push '((t t) t #.(flags)"immnum_and((#0),(#1))")
   (get 'logand 'inline-always))
 (push '((fixnum fixnum) fixnum #.(flags rfa)"((#0) & (#1))")
   (get 'logand 'inline-always))

;;LOGANDC1
 (push '((fixnum fixnum) fixnum #.(flags rfa)"(~(#0) & (#1))")
   (get 'logandc1 'inline-always))

;;LOGANDC2
 (push '((fixnum fixnum) fixnum #.(flags rfa)"((#0) & ~(#1))")
   (get 'logandc2 'inline-always))

;;LOGIOR
 (push '((t t) t #.(flags)"immnum_ior((#0),(#1))")
   (get 'logior 'inline-always))
 (push '((fixnum fixnum) fixnum #.(flags rfa)"((#0) | (#1))")
   (get 'logior 'inline-always))

;;LOGXOR
 (push '((t t) t #.(flags)"immnum_xor((#0),(#1))")
   (get 'logxor 'inline-always))
 (push '((fixnum fixnum) fixnum #.(flags rfa)"((#0) ^ (#1))")
   (get 'logxor 'inline-always))

;;LOGNOT
 (push '((t) t #.(flags)"immnum_not(#0)")
   (get 'lognot 'inline-always))
 (push '((fixnum) fixnum #.(flags rfa)"(~(#0))")
   (get 'lognot 'inline-always))

;;MAKE-LIST
 (push '((seqind) proper-list #.(flags ans rfa) MAKE-LIST-INLINE)
   (get 'make-list 'inline-always))
 (push '(((integer 0 0)) null #.(flags rfa) "Cnil")
   (get 'make-list 'inline-always))

;;INTEGER-LENGTH
(push '((t) t #.(compiler::flags) "immnum_length(#0)") (get 'integer-length 'compiler::inline-always))
(push '((fixnum) fixnum #.(flags rfa set) 
	#.(format nil "({register fixnum _x=labs(#0),_t=~s;for (;_t>=0 && !((_x>>_t)&1);_t--);_t+1;})" (integer-length most-positive-fixnum)))
   (get 'integer-length 'inline-always))


;;MAX
(push '((t t) t #.(flags) "immnum_max(#0,#1)");"@01;(number_compare(#0,#1)>=0?(#0):#1)"
    (get 'max 'inline-always));FIXME
;(push '((t t) t #.(flags set)"@01;({register int _r=number_compare(#0,#1); fixnum_float_contagion(_r>=0 ? #0 : #1,_r>=0 ? #1 : #0);})")
;   (get 'max 'inline-always))
(push '((creal creal) long-float #.(flags set)"@01;((double)((#0)>=(#1)?(#0):#1))")
   (get 'max 'inline-always))
(push '((creal creal) short-float #.(flags set)"@01;((float)((#0)>=(#1)?(#0):#1))")
   (get 'max 'inline-always))
(push '((creal creal) fixnum #.(flags set)"@01;((fixnum)((#0)>=(#1)?(#0):#1))")
   (get 'max 'inline-always))

;;MIN
(push '((t t) t #.(flags) "immnum_min(#0,#1)");"@01;(number_compare(#0,#1)<=0?(#0):#1)"
    (get 'min 'inline-always));FIXME
;(push '((t t) t #.(flags set)"@01;({register int _r=number_compare(#0,#1); fixnum_float_contagion(_r<=0 ? #0 : #1,_r<=0 ? #1 : #0);})")
;   (get 'min 'inline-always))
(push '((creal creal) long-float #.(flags set)"@01;((double)((#0)<=(#1)?(#0):#1))")
   (get 'min 'inline-always))
(push '((creal creal) short-float #.(flags set)"@01;((float)((#0)<=(#1)?(#0):#1))")
   (get 'min 'inline-always))
(push '((creal creal) fixnum #.(flags set)"@01;((fixnum)((#0)<=(#1)?(#0):#1))")
   (get 'min 'inline-always))


;;MOD
 (push '((t t) t #.(compiler::flags) "immnum_mod(#0,#1)") (get 'mod 'compiler::inline-always))
#+intdiv
 (push '((fixnum fixnum) fixnum #.(flags rfa set)"@01;({register fixnum _t=(#0)%(#1);((#1)<0 && _t<=0) || ((#1)>0 && _t>=0) ? _t : _t + (#1);})")
   (get 'mod 'inline-always))

;;CMP-NTHCDR
(push '((seqind t) list #.(flags rfa)"({register fixnum _i=#0;register object _x=#1;for (;_i--;_x=_x->c.c_cdr);_x;})")
   (get 'cmp-nthcdr 'inline-unsafe))
(push '(((not seqind) proper-list) null #.(flags rfa)"Cnil")
   (get 'cmp-nthcdr 'inline-unsafe))
(push '((seqind proper-list) proper-list #.(flags rfa)"({register fixnum _i=#0;register object _x=#1;for (;_i--;_x=_x->c.c_cdr);_x;})")
   (get 'cmp-nthcdr 'inline-always))
(push '(((and (integer 0) (not seqind)) proper-list) null #.(flags rfa)"Cnil")
   (get 'cmp-nthcdr 'inline-always))


;;NULL
 (push '((t) boolean #.(flags rfa)"(#0)==Cnil")
   (get 'null 'inline-always))

;;RATIONALP
 (push '((t) boolean #.(flags rfa)"@0;rationalp(#0)")
   (get 'rationalp 'inline-always))

;;REALP
 (push '((t) boolean #.(flags rfa)"@0;realp(#0)")
   (get 'realp 'inline-always))

;;NUMBERP
 (push '((t) boolean #.(flags rfa)"@0;numberp(#0)")
   (get 'numberp 'inline-always))

;;EQL-IS-EQ
 (push '((t) boolean #.(flags rfa)"@0;eql_is_eq(#0)")
   (get 'eql-is-eq 'inline-always))
 (push '((fixnum) boolean #.(flags rfa)"@0;(is_imm_fix(#0))")
   (get 'eql-is-eq 'inline-always))

;;EQUAL-IS-EQ
 (push '((t) boolean #.(flags rfa)"@0;equal_is_eq(#0)")
   (get 'equal-is-eq 'inline-always))
 (push '((fixnum) boolean #.(flags rfa)"@0;(is_imm_fix(#0))")
   (get 'equal-is-eq 'inline-always))

;;EQUALP-IS-EQ
 (push '((t) boolean #.(flags rfa)"@0;equalp_is_eq(#0)")
   (get 'equalp-is-eq 'inline-always))

;;PRIN1
 (push '((t t) t #.(flags set)"prin1(#0,#1)")
   (get 'prin1 'inline-always))
(push '((t) t #.(flags set)"prin1(#0,Cnil)")
   (get 'prin1 'inline-always))

;;PRINC
 (push '((t t) t #.(flags set)"princ(#0,#1)")
   (get 'princ 'inline-always))
(push '((t) t #.(flags set)"princ(#0,Cnil)")
   (get 'princ 'inline-always))

;;PRINT
 (push '((t t) t #.(flags set)"print(#0,#1)")
   (get 'print 'inline-always))
(push '((t) t #.(flags set)"print(#0,Cnil)")
   (get 'print 'inline-always))

;;RATIOP
(push '((t) boolean #.(flags rfa) "type_of(#0)==t_ratio")
      (get 'ratiop 'inline-always))

;;REM
(push '((t t) t #.(compiler::flags) "immnum_rem(#0,#1)") (get 'rem 'compiler::inline-always))
#+intdiv
(push '((fixnum fixnum) fixnum #.(flags rfa)"((#0)%(#1))")
   (get 'rem 'inline-always))

;;SECOND
;;  (push '((t) t #.(flags)"cadr(#0)")
;;    (get 'second 'inline-safe))
;(push '((t) t #.(flags)"CMPcadr(#0)")
;   (get 'second 'inline-unsafe))

;;STRING
 (push '((t) t #.(flags ans)"coerce_to_string(#0)")
   (get 'string 'inline-always))

;;PATHNAME-DESIGNATORP
(push '((t) boolean #.(flags)"pathname_designatorp(#0)")
      (get 'si::pathname-designatorp 'inline-always))

;;PATHNAMEP
(push '((t) boolean #.(flags)"type_of(#0)==t_pathname")
      (get 'pathnamep 'inline-always))

;;STRINGP
 (push '((t) boolean #.(flags rfa)"({enum type tp=type_of(#0);tp==t_string||tp==t_simple_string;})")
   (get 'stringp 'inline-always))

;;SVREF
;;  (push '((t t) t #.(flags ans)"aref1(#0,fixint(#1))")
;;    (get 'svref 'inline-always))
;; (push '((t fixnum) t #.(flags ans)"aref1(#0,#1)")
;;    (get 'svref 'inline-always))
(push '((t t) t #.(flags)"(#0)->v.v_self[fix(#1)]")
   (get 'svref 'inline-unsafe))
(push '((t fixnum) t #.(flags)"(#0)->v.v_self[#1]")
   (get 'svref 'inline-unsafe))

;;SYMBOL-NAME
 ;; (push '((t) string #.(flags ans rfa)"symbol_name(#0)")
 ;;   (get 'symbol-name 'inline-always))

;;SYMBOL-VALUE
(push '((t) t #.(flags) "((#0)->s.s_dbind)")
    (get 'symbol-value 'inline-unsafe))

;;SYMBOL-FUNCTION FIXME
(push '((t) (or cons function) #.(flags rfa) "({register object _sym=#0;_sym->s.s_sfdef!=NOT_SPECIAL ? make_cons(sLspecial,make_fixnum((long)_sym->s.s_sfdef)) : (_sym->s.s_mflag ? make_cons(sSmacro,_sym->s.s_gfdef) : _sym->s.s_gfdef);})")
      (get 'symbol-function 'inline-unsafe))

;;FUNCALLABLE-SYMBOL-FUNCTION
(push '((t) function #.(flags rfa) "#0->s.s_gfdef")
      (get 'funcallable-symbol-function 'inline-always))

;;SI::FBOUNDP-SYM
(push '((t) boolean #.(flags rfa) "@0;(#0->s.s_sfdef!=NOT_SPECIAL || #0->s.s_gfdef!=OBJNULL)")
      (get 'si::fboundp-sym 'inline-unsafe))
(push '((symbol) boolean #.(flags rfa) "@0;(#0->s.s_sfdef!=NOT_SPECIAL || #0->s.s_gfdef!=OBJNULL)")
      (get 'si::fboundp-sym 'inline-always))

;;TERPRI
 (push '((t) t #.(flags set)"terpri(#0)")
   (get 'terpri 'inline-always))
(push '(nil t #.(flags set)"terpri(Cnil)")
   (get 'terpri 'inline-always))

;;THIRD
;;  (push '((t) t #.(flags)"caddr(#0)")
;;    (get 'third 'inline-safe))
;(push '((t) t #.(flags)"CMPcaddr(#0)")
;   (get 'third 'inline-unsafe))

;;TRUNCATE

(push '((t t) t #.(compiler::flags) "immnum_truncate(#0,#1)") (get 'truncate 'compiler::inline-always))
#+intdiv
(push '((fixnum fixnum) (returns-exactly fixnum fixnum) #.(flags rfa)"({fixnum _t=(#0)/(#1);@1(#0)-_t*(#1)@ _t;})")
   (get 'truncate 'inline-always))
(push '((fixnum) (returns-exactly fixnum fixnum) #.(flags rfa)"({fixnum _t=(#0);@1(#0)-_t@ _t;})")
   (get 'truncate 'inline-always))
(push '((short-float) (returns-exactly fixnum short-float) #.(flags rfa)"({float _t=(#0);@1(#0)-_t@ _t;})")
   (get 'truncate 'inline-always))
(push '((long-float) (returns-exactly fixnum long-float) #.(flags rfa)"({double _t=(#0);@1(#0)-_t@ _t;})")
   (get 'truncate 'inline-always))

;;COMPLEXP
 (push '((t) boolean #.(flags rfa) "type_of(#0)==t_complex")
   (get 'complexp 'inline-always))

;;COMPLEX
 (push '((t t) complex #.(flags) "make_complex(#0,#1)")
   (get 'complex 'inline-always))
 (push '((short-float short-float) fcomplex #.(flags) "(#0 + I * #1)")
   (get 'complex 'inline-always))
 (push '((long-float long-float) dcomplex #.(flags) "(#0 + I * #1)")
   (get 'complex 'inline-always))


;;VECTORP
 (push '((t) boolean #.(flags rfa)
  "@0;({enum type _tp=type_of(#0);_tp>=t_string && _tp<=t_vector;})")
   (get 'vectorp 'inline-always))

;;SEQUENCEP
 (push '((t) boolean #.(flags rfa)
  "@0;(listp(#0) || ({enum type _tp=type_of(#0);_tp>=t_string && _tp<=t_vector;}))")
   (get 'sequencep 'inline-always))

;;FUNCTIONP
 (push '((t) boolean #.(flags rfa) "(functionp(#0))")
   (get 'functionp 'inline-always))

;;COMPILED-FUNCTION-P
 (push '((t) boolean #.(flags rfa) "(compiled_functionp(#0))")
   (get 'compiled-function-p 'inline-always))

;; ;;WRITE-CHAR
;; (push '((t) t #.(flags set)
;;  "@0;(writec_stream(char_code(#0),sLAstandard_outputA->s.s_dbind),(#0))")
;;   (get 'write-char 'inline-unsafe))

;;CMOD
 (push '((t) t #.(flags) "cmod(#0)")
   (get 'system:cmod 'inline-always))

;;CTIMES
 (push '((t t) t #.(flags) "ctimes(#0,#1)")
   (get 'system:ctimes 'inline-always))

;;CPLUS
 (push '((t t) t #.(flags) "cplus(#0,#1)")
   (get 'system:cplus 'inline-always))

;;CDIFFERENCE
 (push '((t t) t #.(flags) "cdifference(#0,#1)")
   (get 'system:cdifference 'inline-always))

;;si::static-inverse-cons
(push '((t) t #.(compiler::flags) "({object _y=(object)fixint(#0);is_imm_fixnum(_y) ? Cnil : (is_imm_fixnum(_y->c.c_cdr) ? _y : (_y->d.f||_y->d.e ? Cnil : _y));})") (get 'si::static-inverse-cons 'compiler::inline-always))
(push '((fixnum) t #.(compiler::flags) "({object _y=(object)#0;is_imm_fixnum(_y) ? Cnil : (is_imm_fixnum(_y->c.c_cdr) ? _y : (_y->d.f||_y->d.e ? Cnil : _y));})") (get 'si::static-inverse-cons 'compiler::inline-always))
(push '((t) t #.(compiler::flags) "({object _y=(object)fix(#0);is_imm_fixnum(_y) ? Cnil : (is_imm_fixnum(_y->c.c_cdr) ? _y : (_y->d.f||_y->d.e ? Cnil : _y));})") (get 'si::static-inverse-cons 'compiler::inline-unsafe))
(push '((fixnum) t #.(compiler::flags) "({object _y=(object)#0;is_imm_fixnum(_y) ? Cnil : (is_imm_fixnum(_y->c.c_cdr) ? _y : (_y->d.f||_y->d.e ? Cnil : _y));})") (get 'si::static-inverse-cons 'compiler::inline-unsafe))

;;SI::NEXT-HASH-TABLE-INDEX
 (push '((t t) fixnum #.(flags rfa) 
	 "({fixnum _i;for (_i=fix(#1);_i<(#0)->ht.ht_size && (#0)->ht.ht_self[_i].hte_key==OBJNULL;_i++);_i==(#0)->ht.ht_size ? -1 : _i;})")
   (get 'si::next-hash-table-index 'inline-unsafe))
 (push '((t fixnum) fixnum #.(flags rfa) 
	 "({fixnum _i;for (_i=(#1);_i<(#0)->ht.ht_size && (#0)->ht.ht_self[_i].hte_key==OBJNULL;_i++);_i==(#0)->ht.ht_size ? -1 : _i;})")
   (get 'si::next-hash-table-index 'inline-unsafe))

;;SI::HASH-ENTRY-BY-INDEX
 (push '((t t) t #.(flags) "(#0)->ht.ht_self[fix(#1)].hte_value")
   (get 'si::hash-entry-by-index 'inline-unsafe))
 (push '((t fixnum) t #.(flags) "(#0)->ht.ht_self[(#1)].hte_value")
   (get 'si::hash-entry-by-index 'inline-unsafe))

;;SI::HASH-KEY-BY-INDEX
 (push '((t t) t #.(flags) "(#0)->ht.ht_self[fix(#1)].hte_key")
   (get 'si::hash-key-by-index 'inline-unsafe))
 (push '((t fixnum) t #.(flags) "(#0)->ht.ht_self[(#1)].hte_key")
   (get 'si::hash-key-by-index 'inline-unsafe))

;;si::GENSYM0
(push '(nil symbol #.(flags ans set rfa) "fSgensym0()") (get 'si::gensym0 'inline-always))

;;si::GENSYM1S
(push '((string) symbol #.(flags ans set rfa) "fSgensym1s(#0)") (get 'si::gensym1s 'inline-always))

;;si::GENSYM1IG
(push '((t) symbol #.(flags ans set rfa) "fSgensym1ig(#0)") (get 'si::gensym1ig 'inline-always))

;;SI::HASH-SET
 (push '((t t t) t #.(flags) "@2;(sethash(#0,#1,#2),#2)")
   (get 'si::hash-set 'inline-unsafe))

;;New C ffi
;;

;(push '((t fixnum opaque *) opaque #.(flags rfa) "(#0(#1))(#2#*)") (get 'addr-call 'inline-always))
;(push '((t fixnum) opaque #.(flags rfa) "(#0(#1))()") (get 'addr-call 'inline-always))

(push '(((member :address) t) fixnum #.(flags rfa) "object_to_fixnum(#1)") (get 'unbox 'inline-always))
(push '(((member :address) fixnum) fixnum #.(flags rfa) "(#1)") (get 'unbox 'inline-always))

;; (defun register-key (l tt)
  
;;   (push `(((member ,l) t t t) ,tt ,(flags rfa) "((#1)->#2.#3)") 
;; 	(get 'el 'inline-always))
;;   (push `(((member ,l) t t t seqind) ,tt ,(flags rfa) "((#1)->#2.#3[#4])")
;; 	(get 'el 'inline-always))
;;   (push `((,tt (member ,l) t t t) ,tt ,(flags rfa) "((#2)->#3.#4=(#0))")
;; 	(get 'set-el 'inline-always))
;;   (push `((,tt (member ,l) t t t seqind) ,tt ,(flags rfa) "((#2)->#3.#4[#5]=(#0))")
;; 	(get 'set-el 'inline-always))

;; )

(deftype stdesig nil '(or string symbol character))
(deftype longfloat nil 'long-float)
(deftype shortfloat nil 'short-float)
(deftype hashtable nil 'hash-table)
(deftype ocomplex nil 'complex)
(deftype bitvector nil 'bit-vector)
(deftype random nil 'random-state)
(deftype cfun nil 'function);FIXME
					; (deftype cclosure nil 'function);FIXME
					; (deftype closure nil 'function);FIXME
					; (deftype sfun nil 'function);FIXME
(deftype ifun nil 'function);FIXME
					; (deftype vfun nil 'function);FIXME
(deftype ustring nil 'string);FIXME
(deftype fixarray nil '(array fixnum))
(deftype sfarray nil '(array short-float))
(deftype lfarray nil '(array long-float))



;;si::c-type
(push '((t) #.(cmp-unnorm-tp (c-type-propagator 'si::c-type #tt)) #.(flags rfa) "type_of(#0)")
      (get 'si::c-type 'inline-always))

(push '((long-float) short-float #.(flags rfa) "((float)#0)" ) (get 'si::long-to-short 'inline-always))
(push '((t) short-float #.(flags) "((float)lf(#0))" ) (get 'si::long-to-short 'inline-unsafe))
(push '((long-float) short-float #.(flags rfa) "((float)#0)" ) (get 'si::long-to-short 'inline-unsafe))

(push '((bignum) long-float #.(flags) "big_to_double(#0)" ) (get 'si::big-to-double 'inline-always))
(push '((t) long-float #.(flags) "big_to_double(#0)" ) (get 'si::big-to-double 'inline-unsafe))
(push '((bignum) long-float #.(flags) "big_to_double(#0)" ) (get 'si::big-to-double 'inline-unsafe))

(push '(((complex)) t #.(flags) "(#0)->cmp.cmp_real")   (get 'complex-real 'inline-always))
(push '((fcomplex) short-float #.(flags) "creal(#0)")   (get 'complex-real 'inline-always))
(push '((dcomplex) long-float  #.(flags) "creal(#0)")   (get 'complex-real 'inline-always))

(push '((t) t         #.(flags) "(#0)->cmp.cmp_real")   (get 'complex-real 'inline-unsafe));FIXME
(push '((fcomplex) short-float #.(flags) "creal(#0)")   (get 'complex-real 'inline-unsafe))
(push '((dcomplex) long-float  #.(flags) "creal(#0)")   (get 'complex-real 'inline-unsafe))

(push '(((complex)) t #.(flags) "(#0)->cmp.cmp_imag")   (get 'complex-imag 'inline-always))
(push '((fcomplex) short-float #.(flags) "cimag(#0)")   (get 'complex-imag 'inline-always))
(push '((dcomplex) long-float  #.(flags) "cimag(#0)")   (get 'complex-imag 'inline-always))

(push '((t) t         #.(flags) "(#0)->cmp.cmp_imag")   (get 'complex-imag 'inline-unsafe));FIXME
(push '((fcomplex) short-float #.(flags) "cimag(#0)")   (get 'complex-imag 'inline-unsafe))
(push '((dcomplex) long-float  #.(flags) "cimag(#0)")   (get 'complex-imag 'inline-unsafe))

(push '((ratio)  integer        #.(flags rfa) "(#0)->rat.rat_num") (get 'ratio-numerator 'inline-always))
(push '((ratio)  integer        #.(flags rfa) "(#0)->rat.rat_den") (get 'ratio-denominator 'inline-always))

(push `((long-float) boolean #.(flags rfa) ,(lambda (x) (add-libc "isinf") (wt "(((int(*)(double))dlisinf)(" x "))"))) (get 'si::isinf 'inline-always))
(push `((long-float) boolean #.(flags rfa) ,(lambda (x) (add-libc "isnan") (wt "(((int(*)(double))dlisnan)(" x "))"))) (get 'si::isnan 'inline-always))


;;LOGCOUNT
(push '((t) t #.(compiler::flags) "immnum_count(#0)") (get 'logcount 'compiler::inline-always))
;;LOGBITP
(push '((t t) boolean #.(compiler::flags) "immnum_bitp(#0,#1)") (get 'logbitp 'compiler::inline-always))

;;LOGNAND
(push '((t t) t #.(compiler::flags) "immnum_nand(#0,#1)") (get 'lognand 'compiler::inline-always))
;;LOGNOR
(push '((t t) t #.(compiler::flags) "immnum_nor(#0,#1)") (get 'lognor 'compiler::inline-always))
;;LOGEQV
(push '((t t) t #.(compiler::flags) "immnum_eqv(#0,#1)") (get 'logeqv 'compiler::inline-always))

;;LOGANDC1
(push '((t t) t #.(compiler::flags) "immnum_andc1(#0,#1)") (get 'logandc1 'compiler::inline-always))
;;LOGANDC2
(push '((t t) t #.(compiler::flags) "immnum_andc2(#0,#1)") (get 'logandc2 'compiler::inline-always))
;;LOGORC1
(push '((t t) t #.(compiler::flags) "immnum_orc1(#0,#1)") (get 'logorc1 'compiler::inline-always))
;;LOGORC1
(push '((t t) t #.(compiler::flags) "immnum_orc2(#0,#1)") (get 'logorc2 'compiler::inline-always))
       

;;LOGTEST
(push '((t t) boolean #.(compiler::flags) "immnum_logt(#0,#1)") (get 'logtest 'compiler::inline-always))

;LDB
(push '(((cons fixnum fixnum) fixnum) fixnum #.(compiler::flags) "fixnum_ldb(fix(#0->c.c_car),fix(#0->c.c_cdr),#1)") (get 'ldb 'compiler::inline-always))
;LDB-TEST
(push '(((cons fixnum fixnum) fixnum) boolean #.(compiler::flags) "fixnum_ldb(fix(#0->c.c_car),fix(#0->c.c_cdr),#1)") (get 'ldb-test 'compiler::inline-always))
;DPB
(push '((fixnum (cons fixnum fixnum) fixnum) t #.(compiler::flags) "fixnum_dpb(fix(#1->c.c_car),fix(#1->c.c_cdr),#0,#2)") (get 'dpb 'compiler::inline-always))
;DEPOSIT-FIELD
(push '((fixnum (cons fixnum fixnum) fixnum) t #.(compiler::flags) "fixnum_dpf(fix(#1->c.c_car),fix(#1->c.c_cdr),#0,#2)") (get 'deposit-field 'compiler::inline-always))


;;MINUSP
(push '((t) boolean #.(flags) "immnum_minusp(#0)") (get 'minusp 'inline-always));"number_compare(small_fixnum(0),#0)>0"
;;PLUSP
(push '((t) boolean #.(flags) "immnum_plusp(#0)") (get 'plusp 'inline-always));"number_compare(small_fixnum(0),#0)>0"
;;ZEROP
(push '((t) boolean #.(flags) "immnum_zerop(#0)") (get 'zerop 'inline-always));"number_compare(small_fixnum(0),#0)==0"


;;EVENP
(push '((t) boolean #.(compiler::flags) "immnum_evenp(#0)") (get 'evenp 'compiler::inline-always))
;;ODDP
(push '((t) boolean #.(compiler::flags) "immnum_oddp(#0)") (get 'oddp 'compiler::inline-always))

;;SIGNUM
(push '((t) t #.(compiler::flags) "immnum_signum(#0)") (get 'signum 'compiler::inline-always))




(setf (get :boolean 'lisp-type) 'boolean)
(setf (get :void 'lisp-type) nil)
(setf (get :cnum 'lisp-type) 'cnum)
(setf (get :creal 'lisp-type) 'creal)
(dolist (l '((:float      "make_shortfloat"      short-float     cnum)
	     (:double     "make_longfloat"       long-float      cnum)
	     (:character  "code_char"            character       cnum)
	     (:char       "make_fixnum"          char            cnum)
	     (:short      "make_fixnum"          short           cnum)
	     (:int        "make_fixnum"          int             cnum)
	     (:uchar      "make_fixnum"          unsigned-char   cnum)
	     (:ushort     "make_fixnum"          unsigned-short  cnum)
	     (:uint       "make_fixnum"          unsigned-int    cnum)
	     (:fixnum     "make_fixnum"          fixnum          cnum)
	     (:long       "make_fixnum"          fixnum          cnum)
	     (:fcomplex   "make_fcomplex"        fcomplex        cnum)
	     (:dcomplex   "make_dcomplex"        dcomplex        cnum)
	     (:string     "make_simple_string"   string)
	     (:object     ""                     t)
	     (:char*      nil                    nil             (array character)   "->st.st_self")
	     (:float*     nil                    nil             (array short-float) "->sfa.sfa_self")
	     (:double*    nil                    nil             (array long-float)  "->lfa.lfa_self")
	     (:long*      nil                    nil             (array fixnum)      "->fixa.fixa_self")
	     (:void*      nil                    nil             (array t)           "->v.v_self")))
  (setf (get (car l) 'lisp-type) (if (cadr l) (caddr l) (cadddr l)))
  (when (cadr l)
    (push `(((member ,(car l)) opaque) t #.(flags rfa) ,(strcat (cadr l) "(#1)"))
	  (get 'box   'inline-always))
    (push `(((member ,(car l)) t) opaque #.(flags rfa) ,(if (eq (car l) :object) "(#1)" (strcat "object_to_" (car l) "(#1)")))
	  (get 'unbox 'inline-always)))
  (when (cadddr l)
    (push `(((member ,(car l)) ,(cadddr l)) opaque
	    #.(flags rfa) ,(if (fifth l) (strcat "(#1)" (fifth l)) (strcat "(" (car l) ")" "(#1)")))
	  (get 'unbox 'inline-always))))

(dolist (l '(char short long int integer keyword character real string structure symbol fixnum))
  (let ((s (intern (symbol-name l) 'keyword)))
    (setf (get s 'lisp-type) l)))

(dolist (l '((object t)(plist proper-list)(float short-float)(double long-float)
	     (pack (or null package)) (direl (or keyword null string))))
  (let ((s (intern (symbol-name (car l)) 'keyword)))
    (setf (get s 'lisp-type) (cadr l))))

(defvar *box-alist* (mapcar (lambda (x) (cons x (cadr (assoc (get x 'lisp-type) *c-types*))))
			    '(:char :fixnum :float :double :fcomplex :dcomplex)))

(do-symbols
 (s :keyword)
 (let ((z (get s 'lisp-type :opaque)))
   (unless (eq z :opaque)
     (setf (get s 'cmp-lisp-type) (or (cadr (assoc (get s 'lisp-type) *c-types*)) (cmp-norm-tp z))))))
