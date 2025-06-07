;;; CMPTYPE  Type information.
;;;
;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
;; Copyright (C) 2024 Camm Maguire

;; This file is part of GNU Common Lisp, herein referred to as GCL
;;
;; GCL is free software; you can redistribute it and/or modify it under
;;  the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;; 
;; GCL is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
;; License for more details.
;; 
;; You should have received a copy of the GNU Library General Public License 
;; along with GCL; see the file COPYING.  If not, write to the Free Software
;; Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.


(in-package :compiler)

(defstruct (binding (:print-function (lambda (x s i) (s-print 'binding (binding-repeatable x) (si::address x) s))))
  form
  repeatable)

(defun naltp (tp &aux (catp (car (atomic-tp tp))))
  (labels ((g (x) (when (binding-p x) (return-from naltp catp)))
	   (f (x) (if (consp x) (or (g (car x)) (f (cdr x))) (g x))))
    (f catp)))

(defun explode-nalt (tp)
  (labels ((g (x)
	     (if (binding-p x)
		 (if (binding-form x) (cmp-unnorm-tp (info-type (cadr (binding-form x)))) t)
		 `(member ,x)))
	   (f (x) (if (consp x) `(cons ,(g (car x)) ,(f (cdr x))) (g x))))
  (cmp-norm-tp (f tp))))

(defun cons-type-p (ktp tp)
  (let ((a (type-and ktp tp)))
    (when (consp a)
      (member-if #'identity (cdr (caar (third a))) :key 'car))))

(defun needs-explode (t1 t2 &aux (n (naltp t1)))
  (when (and n (cons-type-p (if (typep n 'proper-cons) #tproper-cons #tsi::improper-cons) t2))
    n))


(defun ctp-and (t1 t2 &aux n)
  (cond ((setq n (needs-explode t1 t2)) (when (tp-and (explode-nalt n) t2) t1))
	((setq n (needs-explode t2 t1)) (when (tp-and (explode-nalt n) t1) t2))
	((tp-and t1 t2))))

(defun ctp<= (t1 t2 &aux n)
  (cond ((setq n (needs-explode t1 t2)) (when (tp-and (explode-nalt n) t2) t));hash
	((tp<= t1 t2))))

(defun null-list (x) (when (plusp x) (make-list x :initial-element #tnull)))

(defun type-and (x y)
  (cond ((eq x '*) y)
	((eq y '*) x)
	((and (cmpt x) (cmpt y))
	 (let ((lx (length x))(ly (length y)))
	   (cons (if (when (eql lx ly)
		       (when (eq (car x) (car y))
			 (eq (car x) 'returns-exactly)))
		     'returns-exactly 'values)
		 (mapcar 'type-and
			 (append (cdr x) (null-list (- ly lx)))
			 (append (cdr y) (null-list (- lx ly)))))))
	((cmpt x) (type-and (or (cadr x) #tnull) y))
	((cmpt y) (type-and x (or (cadr y) #tnull)))
	((ctp-and x y))))

(defun type-or1 (x y)
  (cond ((eq x '*) x)
	((eq y '*) y)
	((and (cmpt x) (cmpt y))
	 (let ((lx (length x))(ly (length y)))
	   (cons (if (when (eql lx ly)
		    (when (eq (car x) (car y))
		      (eq (car x) 'returns-exactly)))
		  'returns-exactly 'values)
		 (mapcar 'type-or1
			 (append (cdr x) (null-list (- ly lx)))
			 (append (cdr y) (null-list (- lx ly)))))))
	((cmpt x) (type-or1 x `(returns-exactly ,y)))
	((cmpt y) (type-or1 `(returns-exactly ,x) y))
	((tp-or x y))))

(defun type<= (x y)
  (cond ((eq y '*))
	((eq x '*) nil)
	((and (cmpt x) (cmpt y))
	 (do ((x (cdr x) (cdr x))(y (cdr y) (cdr y)))
	     ((and (not x) (not y)) t)
	     (unless (type<= (if x (car x) #tnull) (if y (car y) #tnull))
	       (return nil))))
	((cmpt x) (type<= x `(returns-exactly ,y)));FIXME
	((cmpt y) (type<= `(returns-exactly ,x) y))
	((ctp<= x y))))

(defun type>= (x y) (type<= y x))

(defun type= (x y) (when (type<= y x) (type<= x y)))


(defun get-sym (args)
  (intern (apply 'concatenate 'string (mapcar 'string args))))


(defvar *c-types* (mapcar (lambda (x &aux (y (pop x)))
			    (list* y (cmp-norm-tp y) x))
`((nil                   nil                nil                nil                ""                 ""                   "object ")
  (null                  nil                nil                inline-cond        ""                 ""                   "object ")
  (true                  nil                nil                inline-cond        ""                 ""                   "object ")
  (boolean               nil                nil                inline-cond        ""                 ""                   "object ")
  (character             wt-character-loc   nil                inline-character   "char_code"        "code_char"          "int8_t ")
  (bit                   wt-char-loc        return-char        inline-char        "fix"              "make_fixnum"        "int8_t ")
  (non-negative-char     wt-char-loc        return-char        inline-char        "fix"              "make_fixnum"        "int8_t ")
  (unsigned-char         wt-char-loc        return-char        inline-char        "fix"              "make_fixnum"        "uint8_t ")
  (signed-char           wt-char-loc        return-char        inline-char        "fix"              "make_fixnum"        "int8_t ")
  (char                  wt-char-loc        return-char        inline-char        "fix"              "make_fixnum"        "int8_t ")
  (non-negative-short    wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int16_t ")
  (unsigned-short        wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "uint16_t ")
  (signed-short          wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int16_t ")
  (short                 wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int16_t ")
  (non-negative-int      wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int32_t ")
  (unsigned-int          wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "uint32_t ")
  (signed-int            wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int32_t ")
  (int                   wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "int32_t ")
  (non-negative-fixnum   wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "fixnum ")
  (fixnum                wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "fixnum ")
  (long                  wt-fixnum-loc      return-fixnum      inline-fixnum      "fix"              "make_fixnum"        "fixnum ")
  (short-float           wt-short-float-loc return-short-float inline-short-float "sf"               "make_shortfloat"    "float ")
  (long-float            wt-long-float-loc  return-long-float  inline-long-float  "lf"               "make_longfloat"     "double ")
  (creal                 nil                nil                nil                ""                 ""                   "")
  (fcomplex              wt-fcomplex-loc    return-fcomplex    inline-fcomplex    "sfc"              "make_fcomplex"      "fcomplex ")
  (dcomplex              wt-dcomplex-loc    return-dcomplex    inline-dcomplex    "lfc"              "make_dcomplex"      "dcomplex ")
  (cnum                  nil                nil                nil                ""                 ""                   "")
  (string                wt-string-loc      return-string      inline-string      "object_to_string" "make_simple_string" "char *")
  (t                     wt-loc             return-object      inline             ""                 ""                   "object "))))


(defconstant +c-global-arg-types-syms+   `(fixnum)) ;FIXME (long-float short-float) later
(defconstant +c-local-arg-types-syms+    (union +c-global-arg-types-syms+ '(char fixnum long-float short-float fcomplex dcomplex)))
(defconstant +c-local-var-types-syms+    (union +c-local-arg-types-syms+ '(char fixnum long-float short-float fcomplex dcomplex)))


(defvar +value-types+
  (mapcar (lambda (x)
	    (cons (cadr (assoc x *c-types*)) (get-sym `(,x "-VALUE"))))
	  (list* 'character +c-local-var-types-syms+)))


(defconstant +return-alist+
  (mapcar (lambda (x)
	    (cons (if (eq x 'object) x (cadr (assoc x *c-types*)))
		  (get-sym `("RETURN-" ,x))))
	  (cons 'object +c-local-arg-types-syms+)))

(defconstant +wt-loc-alist+
  `((object . wt-loc)
    ,@(mapcar (lambda (x)
		(cons (cadr (assoc x *c-types*))
		      (get-sym `("WT-" ,x "-LOC"))))
	      (cons 'string +c-local-var-types-syms+))))

(defconstant +inline-types-alist+
  `(,@(mapcar (lambda (x)
		(cons (cadr (assoc x *c-types*))
		      (case x
			    ((t) 'inline)
			    (boolean 'inline-cond)
			    (otherwise (get-sym `("INLINE-" ,x))))))
	      (list* 'boolean t 'string +c-local-var-types-syms+))))

(defconstant +c-global-arg-types+
  (mapcar (lambda (x) (cadr (assoc x *c-types*))) +c-global-arg-types-syms+))

(defconstant +c-local-arg-types+
  (mapcar (lambda (x) (cadr (assoc x *c-types*))) +c-local-arg-types-syms+))

(defconstant +c-local-var-types+
  (mapcar (lambda (x) (cadr (assoc x *c-types*))) +c-local-var-types-syms+))

(defconstant +wt-c-var-alist+
  (nconc
   (mapcar (lambda (x &aux (z (assoc x *c-types*)))
	     (cons (cadr z) (seventh z)))
	   (list* 'character 'string +c-local-var-types-syms+))
   `((object . ""))))

(defconstant +to-c-var-alist+
  (nconc
   (mapcar (lambda (x &aux (z (assoc x *c-types*)))
	     (cons (cadr z) (sixth z)))
	   (list* 'character 'string +c-local-var-types-syms+))
   `((object . ""))))

(defconstant +c-type-string-alist+
  (mapcar (lambda (x &aux (z (assoc x *c-types*)))
	    (cons (cadr z) (eighth z)))
	  `(t bit character signed-char non-negative-char unsigned-char
	      signed-short non-negative-short unsigned-short fixnum
	      non-negative-fixnum signed-int non-negative-int unsigned-int
	      long-float short-float fcomplex dcomplex)))

(defconstant +cmp-array-types+
  (mapcar (lambda (x) (cadr (assoc x *c-types*))) +array-types+))

(defconstant +wt-c-rep-alist+
  (nconc
   (mapcar (lambda (x &aux (z (assoc x *c-types*)))
	     (cons (cadr z) (eighth z)))
	   (list* nil +c-local-var-types-syms+))
   `((object . "object "))))

(defconstant +defentry-c-rep-alist+
  (mapcar (lambda (x &aux (z (assoc x *c-types*)))
	    (cons (cadr z) (eighth z)))
	  '(char #+64bit signed-int fixnum short-float long-float string t)))

(defconstant +cmp-type-alist+
  (mapcar (lambda (x) (cons (cmp-norm-tp (car x)) (cdr x))) +type-alist+))


;FIXME?
(defconstant +promoted-c-types+
  (nconc
   (mapcar (lambda (x) (cadr (assoc x *c-types*)))
	   '(nil null boolean))
   +c-local-var-types+))


(defconstant +clzl0+ (let ((x (1- fixnum-length))) (cmp-norm-tp `(integer ,x ,x))))

(defconstant +coersion-alist+
  (mapcar (lambda (x)
	    (cons (cadr (assoc x *c-types*)) (get-sym `(,x "-LOC"))))
	  +c-local-var-types-syms+))

(defconstant +number-inlines+
  (mapcar 'cdr
	  (remove-if-not (lambda (x)
			   (type>= #tnumber (car x)))
			 +inline-types-alist+)))


(defstruct opaque)


(defmacro nil-to-t (x) `(or ,x t))

(defun name-to-sd (x &aux (sd (when (symbolp x) (get x 's-data))))
  (unless sd
    (error "The structure ~a is undefined." x))
  sd)

(defvar *tmpsyms* nil)
(defun tmpsym nil
  (let ((x (or (pop *tmpsyms*) (gensym))))
    (setf (symbol-plist x) '(tmp t))
    x))
(defconstant +tmpsyms+ (let ((*gensym-counter* 0)) (mapl (lambda (x) (rplaca x (tmpsym))) (make-list 1000))))

(defconstant +opaque+ (gensym))

(defvar *car-limit* -1);1)
(defvar *cdr-limit* -1);5)
(defun cons-tp-limit (x i j)
  (declare (seqind i j))
  (cond ((> i *car-limit*) nil)
	((> j *cdr-limit*) nil)
	((atom x) t)
	((and (cons-tp-limit (car x) (1+ i) 0) (cons-tp-limit (cdr x) i (1+ j))))))

(defun cons-tp-limit-tp (x i j)
  (declare (seqind i j))
  (cond ((> i *car-limit*) nil)
	((> j *cdr-limit*) nil)
	((atom x))
	((not (eq (car x) 'cons)))
	((and (cons-tp-limit-tp (cadr x) (1+ i) 0) (cons-tp-limit-tp (caddr x) i (1+ j))))))

(defun object-type (thing) (object-tp thing))

(defconstant +real-contagion-list+ si::+range-types+)




(defun get-inf (x)
  (etypecase
      x
    (integer (if (plusp x) '+iinf (if (minusp x) '-iinf 'inan)))
    (rational (if (plusp x) '+rinf (if (minusp x) '-rinf 'rnan)))
    (float (float (if (plusp x) +inf (if (minusp x) -inf nan)) x))))

(defun pole-type (y)
  (etypecase
   y
   (integer 'integer)
   (short-float 'short-float)
   (long-float 'long-float)))


(defun tp-to-inf (tp x)
  (ecase tp
	 (integer (if x '-iinf '+iinf))
	 (ratio (if x '-rinf '+rinf))
	 (short-float (if x -sinf +sinf))
	 (long-float (if x -inf +inf))))

(defun bnds-to-bounds (tp x)
  (when x
    (let ((y (pop x)))
      (cons (cond ((eq y '*) (tp-to-inf tp x))
		  ((consp y) (cons (if x 1 -1) y))
		  (y))
	    (bnds-to-bounds tp x)))))


(defun bound-num (x)
  (cond ((member x '(+iinf +rinf)) +sinf)
	((member x '(-iinf -rinf)) -sinf)
	((member x '(inan rnan)) snan)
	((consp x) (cadr x))
	(x)))
	
(defun rat-bound-p (x)
  (or (member x '(+iinf -iinf inan +rinf -rinf rnan))
      (rationalp (if (consp x) (cadr x) x))))
(defun int-bound-p (x)
  (or (member x '(+iinf -iinf inan))
      (integerp (if (consp x) (cadr x) x))))


;; (& ^ \| ~)
;; (logand logior logxor logeqv logandc1 logandc2 logorc1 logorc2 lognand lognor lognot)
;; (+ - * max min si::number-plus si::number-times si::number-minus pexpt /-pole)
;; (gcd lcm)
;; (mod rem)
;; (floor ceiling truncate round ffloor fceiling ftruncate fround)
;; (ash >> << integer-length clzl ctzl abs )


(defun ?rationalize (x f r &aux (z (or (eq f '/-pole) (member-if-not 'int-bound-p r))))
  (cond ((not (member f '(+ - * max min si::number-plus si::number-times si::number-minus pexpt /-pole))) x);closed functions over rationals plus pexpt
	((member-if-not 'rat-bound-p r) x)
	((unless (int-bound-p (cadr r)) (eq f 'pexpt)) x)
	((isinf x) (if (plusp x) (if z '+rinf '+iinf) (if z '-rinf '-iinf)))
	((isnan x) (if z 'rnan 'inan))
	((numberp x) (rational x))
	(x)))

(defun ?list-bound (x r)
  (if (when (and (numberp x) (member-if 'consp r))
	(not (or (isinf x) (isnan x))))
      (list x)
      x))

(defun pole-d (x)
  (if (consp x) (car x) 0))

(defun pole-check (f r)
  (si::break-on-floating-point-exceptions :suspend t)
  (prog1
      (apply f (if (when (symbolp f) (get f 'pole)) r (mapcar 'bound-num r)))
    (si::break-on-floating-point-exceptions :suspend nil)))


(defun mfc1 (f &rest r)
  (?list-bound (?rationalize (pole-check f r) f r) r))

(defun infp (x m)
  (member x (if m `(-iinf -rinf ,-sinf ,-inf) `(+iinf +rinf ,+sinf ,+inf))))

(defun nanp (x)
  (or (member x '(inan rnan)) (isnan x)))

(defun minmax1 (tp m)
  (or
   (reduce (lambda (y x &aux (x (if (when (consp x) (eq (cdr x) 'incl)) (car x) x)))
	    (cond ((eq y '*) y)
		  ((infp x m) '*)
		  ((infp x (not m)) (or y 0))
		  ((nanp x) y);'*
		  ((not y) x)
		  ((funcall (if m '< '>) (if (consp x) (car x) x) (if (consp y) (car y) y)) x)
		  ((when (atom x) (when (consp y) (eql x (car y)))) x)
		  (y)))
	   tp :initial-value nil)
   '*))

(defun inf-tp (x)
  (case x
    ((+iinf -iinf inan) 'integer)
    ((+rinf -rinf rnan) 'ratio)))

(defun mk-tp2 (tp &aux (mm1 (minmax1 tp t)) (mm2 (minmax1 tp nil)))
  (reduce
   'type-or1
   (mapcar (lambda (x)
	     (when (member-if (lambda (y &aux (y (if (listp y) (if (integerp (car y)) 1/2 (car y)) y)))
				(or (typep y x) (eq (inf-tp y) x)))
			      tp)
	       (cmp-norm-tp `(,(if (eq x 'ratio) 'rational x) ,mm1 ,mm2))))
	   si::+range-types+)
   :initial-value nil))


(defun outer-merge (&rest r &aux (z (pop r)))
  (mapcan (lambda (z)
	    (mapcar (lambda (x) (cons z x))
		    (if r (apply 'outer-merge r) (list nil))))
	  z))

(defun ar-merge (&rest r)
  (mapcar (lambda (x &aux (p1 (pop x))(b1 (real-bnds x)))
	    (unless (car b1) (return-from ar-merge nil))
	    (bnds-to-bounds p1 b1))
	  r))

(defun mk-contagion-rep (f ?complex r &aux (i 1))
  (apply f
	 (mapcar (lambda (x &aux (p (pop x)) (y (contagion-irep (incf i) p)))
		   (if ?complex (complex y (contagion-irep (incf i) p)) y))
		 r)))

(defun dsrg (f &rest r &aux (z (apply 'ar-merge r)))
  (if z
   (let* ((v (mapcar (lambda (x) (apply 'mfc1 f x)) (apply 'outer-merge z)))
	  (vc (remove-if-not 'mfc-complexp v))
	  (vr (set-difference v vc)))
     (reduce 'type-or1
	     (mapcar 'complex-contagion vc)
	     :initial-value  (mk-tp2 vr)))
   (complex-contagion (mk-contagion-rep f t r))))

(defun super-range (f &rest r)
  (reduce 'type-or1
	  (mapcar (lambda (x) (apply 'dsrg f x))
		  (apply 'outer-merge (mapcar 'range-decomp r)))
	  :initial-value nil))





















; libm standard poles
; / mod rem truncate etc.  0 two-sided
; atanh +-1  branch-cut
; log 0 branch-cut
; expt/pow 0 neg  same as /
; lgamma/tgamma neg int

(defconstant +small-rat+ (rational least-positive-long-float))

(defun contagion-irep (x tp)
  (case tp
	(ratio (if (or (= 0 x) (= 1 x)) x (+ x (/ 1 x))))
	(integer x)
	(otherwise (coerce x tp))))

(defconstant +cmp-range-types+
  (let ((z '(integer ratio short-float long-float)))
    (nconc (mapcar (lambda (x) (cons x (cmp-norm-tp x))) z)
	   (mapcar (lambda (x)
		     (cons x
			   (case x
			     (integer #t(complex rational))
			     (ratio #t(and (complex rational) (not (complex integer))))
			     (otherwise (cmp-norm-tp `(complex ,x))))))
		   z))))


(defun complex-contagion (z &aux (z (if (listp z) (car z) z)))
  (cadar (member z
		 '#.(mapcar (lambda (x &aux (x `(complex ,x))) (list x (cmp-norm-tp x)))
			   '(integer rational short-float long-float real))
		 :test 'typep :key 'car)))

(defun mfc-complexp (x &aux (x (if (listp x) (car x) x)))
  (complexp x))

(defun range-decomp (tp)
  (mapcan (lambda (x &aux (f (pop x))(z (type-and tp x))) (when z (list (cons f z)))) +cmp-range-types+))


(dolist (l '(si::number-plus si::number-minus si::number-times + - * exp tanh sinh asinh))
  (si::putprop l 'super-range 'type-propagator))

(defun atan-propagator (f t1 &optional (t2 nil t2p))
  (if t2p
      (type-or1
       (super-range f (type-and #tnon-negative-real t1) (type-and #tnon-negative-real t2))
       (type-or1
	(super-range f (type-and #tnon-negative-real t1) (type-and #tnegative-real t2))
	(type-or1
	 (super-range f (type-and #tnegative-real t1) (type-and #tnon-negative-real t2))
	 (super-range f (type-and #tnegative-real t1) (type-and #tnegative-real t2)))))
    (super-range f t1)))
(si::putprop 'atan 'atan-propagator 'type-propagator)

(defun float-propagator (f t1 &optional (t2 (or (type-and #tfloat t1) #t(member 0.0)))
			 &aux (t1 (type-and #treal t1))(t2 (type-and #tfloat t2)))
  (when (and t1 t2)
    (super-range f t1 t2)))
(setf (get 'float 'type-propagator) 'float-propagator)

(defun bit-type (tp)
  (cond ((not tp) tp)
	((atomic-tp tp) tp)
	((type>= #tinteger tp)
	 (let* ((tp (list* 'integer (real-bnds tp)))
		(l (cadr tp))
		(l (if (consp l) (car l) l))
		(h (caddr tp))
		(h (if (consp h) (car h) h))
		(h (if (eq h '*) h (if (>= h 0) (1- (ash 1 (integer-length h))) -1)))
		(l (if (eq l '*) l (if (< l 0) (- (ash 1 (integer-length l))) 0))))
	   (cmp-norm-tp `(integer ,l ,h))))))


(defun logand2-propagator (f t1 t2)
  (when (and (type>= #tfixnum t2) (type>= #tfixnum t1));FIXME
    (let ((t1 (bit-type t1))(t2 (bit-type t2)))
      (super-range '*
		   (if (and (atomic-tp t1) (atomic-tp t2)) #t(integer 1 1) #t(integer 0 1))
		   (type-or1
		    (super-range f
				 (type-and #tnon-negative-integer t1)
				 (type-and #tnon-negative-integer t2))
		    (type-or1
		     (super-range f
				  (type-and #tnegative-integer t1)
				  (type-and #tnon-negative-integer t2))
		     (type-or1
		      (super-range f
				   (type-and #tnon-negative-integer t1)
				   (type-and #tnegative-integer t2))
		      (super-range f
				   (type-and #tnegative-integer t1)
				   (type-and #tnegative-integer t2)))))))))

(dolist (l '(& ^ \|))
  (si::putprop l 'logand2-propagator 'type-propagator))

(defun logand1-propagator (f t1)
  (when (type>= #tfixnum t1);FIXME
    (super-range '*
		 #t(integer 0 1)
		 (super-range f t1))))

(si::putprop '~ 'logand1-propagator 'type-propagator)

(defun logand-propagator (f &optional (t1 nil t1p) (t2 nil t2p) &rest r)
  (cond (r (apply 'logand-propagator f (logand-propagator f t1 t2) (car r) (cdr r)))
	(t2p (logand2-propagator f t1 t2))
	(t1p (logand1-propagator f t1))
	((not t1p) (super-range f))))

(dolist (l '(logand logior logxor logeqv logandc1 logandc2 logorc1 logorc2 lognand lognor lognot))
  (si::putprop l 'logand-propagator 'type-propagator))

(defun min-max-propagator (f &optional (t1 nil t1p) (t2 nil t2p))
  (cond (t2p (super-range f (type-and #treal t1) (type-and #treal t2)))
	(t1p (super-range f (type-and #treal t1)))))
(si::putprop 'max 'min-max-propagator 'type-propagator)
(si::putprop 'min 'min-max-propagator 'type-propagator)

(defun /-pole (x y &aux (d (pole-d y))(x (bound-num x))(y (bound-num y)))
  (if (zerop y)
      (get-inf (* x (if (floatp y) (float d y) d)))
    (let ((x (/ x y)))
      (if (integerp x) (cons x 'incl) x))))
(setf (get '/-pole 'pole) t)

(defun /-propagator (f t1 &optional t2)
  (cond (t2
	 (reduce 'type-or1
		 (mapcar (lambda (x) (super-range '/-pole t1 (type-and t2 x)))
			 '(#tcomplex #tpositive-real #tnegative-real))
		 :initial-value nil))
	(t1 (/-propagator f (object-tp 1) t1))))
(si::putprop '/ '/-propagator 'type-propagator)
(si::putprop 'si::number-divide '/-propagator 'type-propagator)

(defun real-imag-tp (x rp)
  (when (consp x)
    (case (car x)
      (member (reduce (lambda (y x) (type-or1 y (object-tp (if rp (realpart x) (imagpart x))))) (cdr x)
		      :initial-value nil))
      (or (reduce (lambda (y x) (type-or1 y (real-imag-tp x rp))) (cdr x) :initial-value nil))
      (complex (cmp-norm-tp (cadr x)))
      (si::complex* (cmp-norm-tp (if rp (cadr x) (caddr x)))))))


(defun complex-real-imag-type-propagator (f t1 rp)
  (declare (ignore f))
  (when (type>= #tcomplex t1)
    (reduce (lambda (&rest r) (when r (apply 'type-or1 r)))
	    (mapcar (lambda (x) (real-imag-tp (si::tp-type (cdr x)) rp))
		    (range-decomp t1)))))
(defun complex-real-type-propagator (f t1)
  (complex-real-imag-type-propagator f t1 t))

(defun complex-imag-type-propagator (f t1)
  (complex-real-imag-type-propagator f t1 nil))
(si::putprop 'si::complex-real 'complex-real-type-propagator 'type-propagator)
(si::putprop 'si::complex-imag 'complex-imag-type-propagator 'type-propagator)
(si::putprop 'c-ocomplex-real 'complex-real-type-propagator 'type-propagator)
(si::putprop 'c-ocomplex-imag 'complex-imag-type-propagator 'type-propagator)

(defun tp-contagion (tp c &aux (s #tshort-float)(l #tlong-float))
  (cond ((type>= c s)
	 (if (type>= s tp) tp
	     (cmp-norm-tp `(short-float ,@(real-bnds tp)))))
	((type>= c l)
	 (if (type>= l tp) tp
	     (cmp-norm-tp `(long-float ,@(real-bnds tp)))))
	(tp)))


(defun c-type-propagator (f t1)
  (declare (ignore f))
  (cmp-norm-tp
   (cons 'member
	 (reduce (lambda (y x)
		    (when (type-and t1 (car x))
		      (pushnew (c-type (eval (cdr x))) y))
		    y) si::+rn+ :initial-value nil))))
(si::putprop 'c-type 'c-type-propagator 'type-propagator)


(defconstant +e+ 2.7182818284590451)

(defun log-pole (&rest r &aux (x (pop r))(d (pole-d x))(x (bound-num x))(x (if (integerp x) (float x) x)))
  (if (zerop x)
      (let ((x (coerce -inf (pole-type x))))
	(if (plusp d) x (complex (realpart x) (float +pi+ (realpart x)))))
    (apply 'log x (mapcar 'bound-num r))))
(setf (get 'log-pole 'pole) t)

(defun log-propagator (f t1 &rest r)
  (declare (ignore f))
  (reduce 'type-or1
	  (mapcar (lambda (x) (apply 'super-range 'log-pole (type-and t1 x) r))
		  '(#tcomplex #tpositive-real #tnegative-real))
	  :initial-value nil))
(si::putprop 'log 'log-propagator 'type-propagator)

(defun last-cons-type (tp &optional l)
  (cond ((and l (atom tp)) tp)
	((and (consp tp) (eq (car tp) 'cons) 
	      (cddr tp) (not (cdddr tp)))
	 (last-cons-type (caddr tp) t))))

(defun cdr-propagator (f t1 &aux (t1 (type-and #tlist t1)))
  (declare (ignore f))
  (cond ((type>= #tnull t1) t1) ;FIXME clb ccb do-setq-tp
	((let ((a1 (atomic-tp t1)))
	   (when a1
	     (let ((tp (cdar a1)))
	       (unless (binding-p tp)
		 (object-type tp))))));FIXME bind-type?
	((and (consp t1) (eq (car t1) 'cons)) (caddr t1))
	((type>= #tproper-list t1) #tproper-list)))
(si::putprop 'cdr 'cdr-propagator 'type-propagator)

(defun make-list-propagator (f t1 &rest r &aux (a (atomic-tp t1)))
  (declare (ignore f r))
  (cond ((and (type>= #t(integer 0 5) t1) a)
;	 (object-type (make-list (cadr t1))))
	 (cmp-norm-tp (reduce (lambda (y x) (declare (ignore x)) `(cons t ,y)) (make-list (car a)) :initial-value 'null)))
	(#tproper-list)))
(si::putprop 'make-list 'make-list-propagator 'type-propagator)


(defun nth-cons-tp (n tp)
  (cond ((= n 0) tp)
	((and (consp tp) (eq 'cons (car tp)) (cddr tp) (not (cdddr tp)))
	 (nth-cons-tp (1- n) (caddr tp)))))

(defun nthcdr-propagator (f t1 t2)
  (declare (ignore f))
  (let ((t1 (type-and #tinteger t1))
	(t2 (type-and #tlist t2)))
    (cond ((type>= #tnull t2) t2) ;FIXME clb ccb do-setq-tp
	  ((type>= #t(integer 0 0) t1) t2)
	  ((and (consp t2) (eq (car t2) 'cons) (atomic-tp t1) (typep (cadr t1) 'seqind)) 
	   (nth-cons-tp (cadr t1) t2))
	  ((type>= #tproper-list t2) #tproper-list))))
(si::putprop 'nthcdr 'nthcdr-propagator 'type-propagator)

(defun bump-pcons (v p)
  (let ((tp (if p #tproper-cons #tcons)))
    (unless (type>= (var-type v) tp)
      (when (type>= #tproper-cons (var-type v))
	(do-setq-tp v nil tp)
	(mapc (lambda (x) (bump-pcons x p)) (var-aliases v))))))

(defun bump-pconsa (v ctp)
  (let ((tp (cons-propagator 'cons ctp (cdr-propagator 'cdr (var-type v)))))
    (unless (type>= (var-type v) tp)
      (do-setq-tp v nil tp)
      (mapc (lambda (x) (bump-pconsa x ctp)) (var-aliases v)))))


(defun bump-cons-tp-if (f tp)
  (dolist (v *vars*)
    (when (var-p v)
      (unless (tp<= tp (var-type v))
	(when (funcall f (var-type v))
	  (keyed-cmpnote (list (var-name v) 'type-propagation 'type 'bump-cons-tp-if)
			 "Bumping var ~s cons type ~s -> ~s, tp ~s"
			 (var-name v) (cmp-unnorm-tp (var-type v)) (cmp-unnorm-tp (tp-or (var-type v) tp)) (cmp-unnorm-tp tp))
	  (do-setq-tp v 'bump-cons-tp-if (tp-or (var-type v) tp))))
      (let ((s (var-store v)))
	(when (listp s);FIXME
	  (dolist (b s)
	    (let* ((fm (binding-form b))(i (when fm (cadr fm)))(itp (when i (info-type i))))
	      (when (and fm (funcall f itp))
		(setf (info-type i) (type-or1 itp tp))))))))))

(defun c1rplacd (args)
  (let* ((info (make-info :flags (iflags side-effects)))
	 (nargs (c1args args info))
	 (tp1 (type-and #tcons (info-type (cadar nargs))))(tp2 (info-type (cadadr nargs)))
	 (c1 (car (atomic-tp tp1)))(atp2 (atomic-tp tp2))(a2 (car atp2)) itp)
    (c1side-effects nil)
    (cond
      ((and c1 atp2 (not (eq c1 a2)) (if (typep c1 'proper-cons) (typep a2 'proper-list) (typep a2 '(not proper-list))))
       (when *restore-vars-env*
	 (keyed-cmpnote (list 'type-mod-unwind) "Winding type ~s for rplacd" tp1)
	 (pushnew (list tp1 (mcpt tp1)) *restore-vars* :key 'car))
       (setf (cdr c1) a2 itp tp1))
      ((and (typep c1 'proper-cons) (type<= tp2 #tproper-list))
       (bump-cons-tp-if (lambda (x &aux (c (car (atomic-tp x)))) (when (consp c) (tailp c1 c))) #tproper-cons))
      ((and (typep c1 'si::improper-cons) (type<= tp2 #t(not proper-list)))
       (bump-cons-tp-if (lambda (x &aux (c (car (atomic-tp x)))) (when (consp c) (tailp c1 c))) #tsi::improper-cons))
      ((and (type<= tp1 #tproper-cons) (type-and tp2 #t(not proper-list)))
       (bump-cons-tp-if (lambda (x) (type-and x #tproper-cons)) #tcons))
      ((and (type<= tp1 #tsi::improper-cons) (type-and tp2 #tproper-list))
       (bump-cons-tp-if (lambda (x) (type-and x #tsi::improper-cons)) #tcons)))
    (setf (info-type info) (or itp (if (type<= tp2 #tproper-list) #tproper-cons #tcons)))
    (list 'call-global info 'rplacd nargs)))
(si::putprop 'rplacd 'c1rplacd 'c1)

(defun c1rplaca (args)
  (let* ((info (make-info :flags (iflags side-effects)))
	 (nargs (c1args args info))
	 (tp1 (info-type (cadar nargs)))
	 (atp (car (atomic-tp tp1)))
;	 (atp1 (car (atomic-tp (narg-list-type (cdr nargs))))))
	 (atp1 (car (atomic-tp (info-type (cadadr nargs))))))
    (c1side-effects nil)
    (when (consp atp) 
      (when (eq atp atp1) (setq atp1 (copy-list atp1)))
      (when *restore-vars-env*
	(keyed-cmpnote (list 'type-mod-unwind)	"Winding type ~s for rplaca" tp1)
	(let ((tp (info-type (cadar nargs)))) (pushnew (list tp (mcpt tp)) *restore-vars* :key 'car)))
      (setf (car atp) (or atp1 (new-bind))))
    (when (eq (caar nargs) 'var)
      (bump-pconsa (caaddr (car nargs)) (info-type (cadadr nargs))))
    (setf (info-type info) (cons-propagator 'cons (info-type (cadadr nargs))
					    (cdr-propagator 'cdr (info-type (cadar nargs)))))
    (list 'call-global info 'rplaca nargs)))
(si::putprop 'rplaca 'c1rplaca 'c1)

(defun cons-propagator (f t1 t2 &aux tmp)
  (declare (ignore f))
  (cond ((let ((a1 (atomic-tp t1))
	       (a2 (atomic-tp t2)))
	   (and a1 a2 (object-type (cons (car a1) (car a2))))))
	((cons-tp-limit (setq tmp `(cons ,t1 ,t2)) 0 0) (cmp-norm-tp tmp))
	((type>= #tproper-list t2) #tproper-cons)
	(#tcons)))
(si::putprop 'cons 'cons-propagator 'type-propagator)

(defun carcdr-c1form-narg (fm)
  (case (car fm)
    (inline (carcdr-c1form-narg (fourth fm)))
    (let* (carcdr-c1form-narg (fifth fm)))
    (lit (fourth (car (eighth fm))))))

(defun get-binding-form (b &aux (v (get-var b)))
  (if v
      (c1var v);FIXME? go through c1var when possible to pick up var-type
      (let ((f (when (and (binding-p b) (binding-repeatable b)) (binding-form b))))
	(when (check-vs (find-vs f))
	  f))))


(defun co1carcdr (f x);FIXME c1 prop?
  (let* ((c1form (mi1 f x));NOTE this cannot be exponential/double eval, esp. for ACL2!
	 (narg (carcdr-c1form-narg c1form))
	 (atp (when (and narg (ignorable-form narg)) (atomic-tp (info-type (cadr narg)))))
	 (tp (car atp))
	 (b (when (consp tp) (funcall f tp))));FIXME nil
    (typecase b
      (null c1form)
      (binding (or (get-binding-form b) c1form))
      (otherwise (atomic-type-constant-value atp)))))
(setf (get 'car 'co1) 'co1carcdr)
(setf (get 'cdr 'co1) 'co1carcdr)

(defun car-propagator (f t1 &aux (t1 (type-and #tlist t1)))
  (declare (ignore f))
  (cond ((type>= #tnull t1) t1) ;FIXME clb ccb do-setq-tp
	((let ((a1 (atomic-tp t1)))
	   (when a1
	     (let ((tp (caar a1)))
	       (unless (binding-p tp)
		 (object-type tp))))))
	((and (consp t1) (eq (car t1) 'cons)) (cadr t1))))
(si::putprop 'car 'car-propagator 'type-propagator)

(defun contagion (t1 t2)
  (car (member (type-or1 t1 t2) `(,#tlong-float ,#tshort-float #tratio #tinteger)
	       :test 'type-and)))
(defun mod-propagator (f t1 t2 &aux (t1 (type-and #treal t1))(t2 (type-and #treal t2))
				 (r1 (range-decomp t1))(r2 (range-decomp t2)))
  (declare (ignore f))
  (cond
   ((cdr r1) (reduce 'type-or1 (mapcar (lambda (x) (mod-propagator f (cdr x) t2)) r1) :initial-value nil))
   ((cdr r2) (reduce 'type-or1 (mapcar (lambda (x) (mod-propagator f t1 (cdr x))) r2) :initial-value nil))
   ((let ((a (atomic-tp t1))(b (atomic-tp t2)))
      (when (and a b)
	(unless (zerop (car b))
	  (object-tp (mod (car a) (car b)))))))
   ((and (type>= #treal t1) (type>= #treal t2))
	 (let* ((tp (super-range '* #t(integer 0 1) t2));FIXME this might break for integers in the future
		(r (real-bnds tp))
		(r (labels ((b (x) (if (when (numberp x) (not (zerop x))) (list x) x)))
		     (list (b (car r)) (b (cadr r))))))
	   (type-and (contagion t1 t2) (cmp-norm-tp (cons 'real r)))))))
(si::putprop 'mod 'mod-propagator 'type-propagator)

(defun random-propagator (f t1 &optional t2)
  (declare (ignore t2))
  (mod-propagator f (super-range '* #t(integer 0 1) t1) t1))
(si::putprop 'random 'random-propagator 'type-propagator)

(defun lgcd2-propagator (f t1 t2 t3
			 &aux (a1 (car (atomic-tp t1)))(a2 (car (atomic-tp t2)))
			   (a3 (car (atomic-tp t3))))
  (cond ((and a1 a2 a3) (object-type (funcall f a1 a2 a3)))
	((type-and #t(not (integer 0 0)) (super-range '* #t(integer 0 1) (super-range 'min t1 t2))))))
(si::putprop 'si::lgcd2 'lgcd2-propagator 'type-propagator)


(defun rem-propagator (f t1 t2 &aux (ta (abs-propagator 'abs t2)))
  (let ((tm (mod-propagator f t1 t2)))
    (when tm
      (cond ((type>= #tnon-negative-real t1)
	     (type-or1 (type-and #tnon-negative-real tm)
		       (super-range '+ (type-and #tnon-positive-real tm) ta)))
	    ((type>= #tnon-positive-real t1)
	     (type-or1 (type-and #tnon-positive-real tm)
		       (super-range '- (type-and #tnon-negative-real tm) ta)))
	    ((type-or1 tm (super-range '- tm)))))))
(si::putprop 'rem 'rem-propagator 'type-propagator)

(defun floor-propagator (f t1 &optional (t2 #t(member 1))
			   &aux (t1 (type-and #treal t1))(t2 (type-and #treal t2))
			   (i (member f '(floor truncate round ceiling))))
  (let* ((sr (super-range (lambda (x)
			    (cond ((isinf x) (if i (if (> x 0) '+iinf '-iinf) x))
				  ((isnan x) (if i 'inan x))
				  ((funcall f x))))
			  (/-propagator '/ t1 t2)))
	 (sr (if i (type-and #tinteger sr) sr)))
    (when sr
      `(returns-exactly
	,sr
	,(cond ((member f '(floor ffloor))       (mod-propagator f t1 t2))
	       ((member f '(ceiling fceiling))   (super-range '- (mod-propagator f t1 t2)))
	       ((member f '(truncate ftruncate round fround)) (rem-propagator f t1 t2)))))))

(dolist (l '(floor ceiling truncate round ffloor fceiling ftruncate fround))
  (si::putprop l 'floor-propagator 'type-propagator)
  (si::putprop l t 'c1no-side-effects))

(defun ash-propagator (f t1 t2)
  (and
   (type>= #tfixnum t1)
   (type>= #t(integer #.most-negative-fixnum #.(integer-length most-positive-fixnum)) t2)
   (super-range f t1 t2)))
(si::putprop 'ash 'ash-propagator 'type-propagator)
(si::putprop 'si::mpz_mul_2exp 'ash-propagator 'type-propagator)
(si::putprop 'si::mpz_fdiv_q_2exp 'ash-propagator 'type-propagator)

(defun <<-propagator (f t1 t2)
  (when (type>= #tfixnum t1)
    (super-range
     (lambda (x y)
       (if (when (typep y 'fixnum)
	     (> (- #.(1+ (integer-length most-positive-fixnum)) (integer-length x)) y))
	   (funcall f x y) (return-from <<-propagator nil)))
     t1 t2)))
(si::putprop 'si::<< '<<-propagator 'type-propagator)

(defun >>-propagator (f t1 t2)
  (when (and (type>= #tfixnum t1) (type>= #t(integer 0 #.(integer-length most-positive-fixnum)) t2))
   (super-range f t1 t2)))
(si::putprop 'si::>> '>>-propagator 'type-propagator)

(defun pexpt (x y) ;; x>=0, y>=0
  (typecase
   y
   ((real 0 0) (1+ y))
   ((integer 1000) (get-inf x))
   (otherwise (expt x y))))

(defun expt-propagator (f t1 t2)
  (declare (ignore f))
  (when (type>= #tnon-negative-real t1)
    (when (type>= #treal t2)
      (type-or1
       (super-range 'pexpt t1 (type-and #tnon-negative-real t2))
       (/-propagator '/ (super-range 'pexpt t1 (super-range '- (type-and #tnegative-real t2))))))))
(si::putprop 'expt 'expt-propagator 'type-propagator)

;; (defun exp-propagator (f t1)
;;   (declare (ignore f))
;;   (expt-propagator 'expt (if (type>= #tshort-float t1) (object-type (float +e+ 0.0s0)) (object-type +e+)) t1))
;; (si::putprop 'exp 'exp-propagator 'type-propagator)

(defun integer-length-propagator (f t1)
  (when (type>= #tfixnum t1)
    (type-or1 (super-range f (type-and #tnon-negative-real t1))
	      (super-range f (type-and #tnon-positive-real t1)))))
(si::putprop 'integer-length 'integer-length-propagator 'type-propagator)
;(defconstant +clzl0+ (let ((x (1+ (si::clzl 1)))) (cmp-norm-tp `(integer ,x ,x))))
;(defconstant +clzl0+ (let ((x (1- si::fixnum-length))) (cmp-norm-tp `(integer ,x ,x))))

(defun bnd-clzl (x y)
  (let* ((lx (si::clzl x))(ly (si::clzl y))(m (if (if (minusp x) (plusp y) (minusp y)) (si::clzl 0) lx)))
    (cmp-norm-tp `(integer ,(min lx ly m) ,(max lx ly m)))))

(defun clzl-propagator (f t1 &aux (t1 (type-and #tfixnum t1)));FIXME wrap
  (declare (ignorable f))
  (unless (type<= #tfixnum t1)
    (if (atom t1) (apply 'bnd-clzl (real-bnds t1))
	(reduce 'type-or1
		(mapcar (lambda (x) (bnd-clzl (car x) (cdr x)))
			(cdr (assoc 'integer (caaddr t1))))
		:initial-value nil))))
(si::putprop 'si::clzl 'clzl-propagator 'type-propagator)
(si::putprop 'si::clzl t 'cmp-inline);FIXME no declaim

(defun bnd-ctzl (x y &optional (i 0) res)
  (if (eql x y)
      (cmp-norm-tp (cons 'member (cons (+ (if (zerop x) 0 i) (si::ctzl x)) res)))
      (bnd-ctzl (>> (if (oddp x) (1+ x) x) 1) (>> (if (oddp y) (1- y) y) 1) (1+ i) (cons i res))))

(defun ctzl-propagator (f t1 &aux (t1 (type-and #tfixnum t1)))
  (declare (ignorable f))
  (unless (type<= #tfixnum t1)
    (if (atom t1) (apply 'bnd-ctzl (real-bnds t1))
	(reduce 'type-or1
		(mapcar (lambda (x) (bnd-ctzl (car x) (cdr x)))
			(cdr (assoc 'integer (caaddr t1))))
		:initial-value nil))))
(si::putprop 'si::ctzl 'ctzl-propagator 'compiler::type-propagator)
(si::putprop 'si::ctzl t 'compiler::cmp-inline);FIXME no declaim

(defun abs-propagator (f t1)
  (when t1
    (type-and #tnon-negative-real
	      (type-or1
	       (let ((t1 (type-and t1 #tcomplex)))
		 (when t1
		   (super-range
		    '+
		    (abs-propagator f (complex-real-type-propagator 'complex-real t1))
		    (abs-propagator f (complex-imag-type-propagator 'complex-imag t1)))))
	       (let ((t1 (type-and #treal t1)))
		 (type-or1 t1 (super-range '- t1)))))))
(si::putprop 'abs 'abs-propagator 'type-propagator)

(defun cosh-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #t(not real)))
	    (type-or1 (super-range f (type-and t1 #tnon-negative-real))
		      (super-range f (type-and t1 #tnegative-real)))))
(si::putprop 'cosh 'cosh-propagator 'type-propagator)

(defun shrnfm (t1 m o &aux (sf (type>= #tshort-float t1))
		  (m (if sf (float m 0.0s0) m))
		  (o (if sf (float o 0.0s0) o)));FIXME
  (let* ((r (real-bnds t1))
	 (s (if (numberp (car r)) (ftruncate (+ o (car r)) m) 0))
	 (k (cmp-norm-tp `(real ,(- o) (,(- m o)))))
	 (st (super-range '- t1 (object-tp (* s m)))))
    (type-and
     k
     (type-or1
      st
      (super-range '- st (object-tp m))))));FIXME max period

(defconstant +pi+ (atan 0 -1))
(defconstant +pid2+ (* 0.5 (atan 0 -1)))

(defun float-proxy-propagator (f t1)
  (declare (ignore f))
  (reduce 'type-or1
	  (mapcar (lambda (x)
		    (super-range (lambda (x)
				   (cond ((isinf x) (if (> x 0) +inf -inf))
					 ((isnan x) nan)
					 ((float x))))
				 (type-and t1 x)))
		  '(#tnegative-real #tnon-negative-real))
	  :initial-value nil))
(si::putprop 'si::big-to-double 'float-proxy-propagator 'type-propagator)
(si::putprop 'si::ratio-to-double 'float-proxy-propagator 'type-propagator)

(defun sqrt-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (type-or1 (super-range f (type-and t1 #tnon-negative-real))
		      (super-range 'sqrt (type-and t1 #tnegative-real)))))
(si::putprop 'sqrt 'sqrt-propagator 'type-propagator)

(defun cos-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (let ((z (shrnfm (type-and t1 #treal) (* 2 +pi+) +pi+)))
	      (type-or1 (super-range f (type-and z #tnon-negative-real))
			(super-range f (type-and z #tnegative-real))))))
(si::putprop 'cos 'cos-propagator 'type-propagator)

(defun sin-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (let ((z (shrnfm (type-and t1 #treal) (* 2 +pi+) +pid2+)))
	      (type-or1 (super-range f (type-and z (cmp-norm-tp `(real * (,+pid2+)))))
			(super-range f (type-and z (cmp-norm-tp `(real ,+pid2+))))))));FIXME
(si::putprop 'sin 'sin-propagator 'type-propagator)

(defun tan-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (let ((z (shrnfm (type-and t1 #treal) +pi+ +pid2+)))
	      (type-or1 (super-range f (type-and z #tnon-negative-real))
			(super-range f (type-and z #tnegative-real))))))
(si::putprop 'tan 'tan-propagator 'type-propagator)

(defun asin-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (type-or1 (super-range (lambda (x) (funcall f (/ x 6))) 
				   (super-range '* #t(integer 6 6) (type-and t1 #t(real -1 1))))
		      (super-range f (type-and t1 #t(not (real -1 1)))))))
(si::putprop 'asin 'asin-propagator 'type-propagator)
(si::putprop 'acos 'asin-propagator 'type-propagator)


(defun atanh-pole (x &aux (d (pole-d x))(x (bound-num x))(x (if (integerp x) (float x) x)))
  (cond ((= x 1) (let ((x (coerce +inf (pole-type x))))
		   (if (minusp d) x (complex (realpart x) (float +pid2+ (realpart x))))))
	((= x -1) (let ((x (coerce -inf (pole-type x))))
		    (if (plusp d) x (complex (realpart x) (float +pid2+ (realpart x))))))
	((atanh x))))
(setf (get 'atanh-pole 'pole) t)

(defun atanh-propagator (f t1)
  (declare (ignore f))
  (reduce 'type-or1
	  (mapcar (lambda (x) (super-range 'atanh-pole (type-and t1 x)))
		  `(,#tcomplex ,#t(real * (-1)) ,#t(real (-1) (1)) ,#t(real (1))))
	  :initial-value nil))
(si::putprop 'atanh 'atanh-propagator 'type-propagator)

(defun acosh-propagator (f t1)
  (type-or1 (super-range f (type-and t1 #tcomplex))
	    (type-or1 (super-range f (type-and t1 #t(real 1)))
		      (super-range f (type-and t1 #t(real * (1)))))))
(si::putprop 'acosh 'acosh-propagator 'type-propagator)

(defun make-vector-propagator (f et st &rest r)
  (declare (ignore f))
  (cmp-norm-tp `(,(if (and (type>= #tnull (pop r))
			   (type>= #tnull (pop r))
			   (type>= #tnull (car r)))
		      'simple-array 'array)
		 ,(or (car (atomic-tp et)) '*)
		 (,(or (car (atomic-tp st)) '*)))))
(si::putprop 'si::make-vector 'make-vector-propagator 'type-propagator)
(defun make-array1-propagator (f &rest r)
  (declare (ignore f))
  (cmp-norm-tp `(array ,(or (car (atomic-tp (car r))) '*)
		       ,(or (let* ((x (car (atomic-tp (sixth r))));FIXME centralize
				   (x (if (integerp x) (make-list x :initial-element '*) x)))
			      (mapcar (lambda (x) (if (integerp x) x '*)) x))
			    '*))))
(si::putprop 'si::make-array1 'make-array1-propagator 'type-propagator)

(defun promoted-c-type (type &aux r)
  (let ((type (coerce-to-one-value type)))
    (cond ((eq type 'object) type);FIXME
	  ((setq r (member type +promoted-c-types+ :test 'type<=))
	   (car r))
	  (#tt))))

(defun single-type-p (type)
  (if (listp type)
      (case (car type)
	    (returns-exactly (when (cdr type) (unless (cddr type) (cadr type))))
	    (values nil)
	    (otherwise type))
    (unless (eq type '*) type)))

(defun coerce-to-one-value (type) (type-and type t))

(defun individuals (f x)
  (mapcan (lambda (y &aux (y (if (listp y) (car y) y)))
	    (when (funcall f y)
	      (list (cons (car x) y))))
	  (cdr x)))

(defun ntp-kingdoms-with-individuals (ntp)
  (mapcan (lambda (x)
	    (case (car x)
	      ((complex-integer complex-integer-ratio complex-ratio-integer
				complex-short-float complex-long-float)
	       (individuals 'complexp x))
	      ((integer ratio))
	      ((short-float long-float);FIXME conceptually this should not be here.
	       (mapcan (lambda (y)
			 (when (consp y)
			   (labels ((d (z &aux (z (if (listp z) (car z) z)))
				      (when (or (isinf z) (isnan z))
					(list (cons (car x) z)))))
			     (append (d (car y)) (d (cdr y))))))
		       (cdr x)))
	      ((std-instance structure funcallable-std-instance)
	       (individuals (lambda (y) (not (eq 'top (si::std-def y)))) x))
	      ((proper-cons si::improper-cons);FIXME
	       (mapcan (lambda (y)
			 (when (listp y)
			   (append (ntp-kingdoms-with-individuals (car y))
				   (ntp-kingdoms-with-individuals (cadr y))
				   (when (caddr y) (list (cons (car x) (caddr y))))
				   (when (cadddr y) (list (cons (car x) (car (cadddr y))))))))
		       (cdr x)))
	      (#.(mapcar 'cdr si::*all-array-types*)  (individuals 'arrayp x))
	      (otherwise (individuals (lambda (y) (not (eq y t))) x))))
	  (car ntp)))

(defun kingdoms-with-individuals (tp)
  (when (consp tp)
    (let ((ntp (caddr tp)))
      (if (caddr ntp)
	  (mapcar 'car (car (si::ntp-and (caar ntp) (si::ntp-not (cadar ntp)))))
	  (ntp-kingdoms-with-individuals ntp)))))


(declaim (inline bump-individuals))
(defun bump-individuals (f tp)
  (cond ((cmpt tp) (cons (car tp) (mapcar (lambda (x) (bump-individuals f x)) (cdr tp))))
	((let* ((x (kingdoms-with-individuals tp))
		(x (remove-if-not (lambda (x) (if (consp x) (funcall f (cdr x)) t)) x))
		(x (remove-duplicates (mapcar (lambda (x) (if (consp x) (car x) x)) x))))
	   (if x
	       (type-or1 (cmp-norm-tp (cons 'or x)) tp)
	       tp)))))

(declaim (inline unprintable-individual-p))
(defun unprintable-individual-p (x)
  (typecase x
    (float (or (isnan x) (isinf x)));t
    ((or string bit-vector number random-state character symbol pathname) nil)
    (cons (or (unprintable-individual-p (car x)) (unprintable-individual-p (cdr x))))
    ((array t) (some 'unprintable-individual-p x))
    ;FIXME assumes structure elements are printable
    (structure (si::s-data-print-function (c-structure-def x)))
    (t t)))

(defun export-type (type)
  (bump-individuals 'unprintable-individual-p type))



(defun bump-tp (tp)
  (cond ((eq tp '*) tp)
	((and (consp tp) (member (car tp) '(values returns-exactly)))
	 `(,(car tp) ,@(mapcar 'bump-tp (cdr tp))))
	((type>= tp #tnull) (type-or1 #tnull (bump-tp (type-and #t(not null) tp))))
	((tsrch tp))
	(t)))




(defun c-structure-def-propagator (f t1)
  (declare (ignore f))
  (when (symbolp t1)
    (let ((tem (get t1 's-data)))
      (when tem (object-type tem)))))
(setf (get 'c-structure-def 'type-propagator) 'c-structure-def-propagator)

(defun structure-name-propagator (f t1)
  (declare (ignore f))
  (when (symbolp t1)
    (when (get t1 's-data)
      (object-type t1))))
(setf (get 'si::structure-name 'type-propagator) 'structure-name-propagator)


(defun expand-type-propagator (f t1 &aux (a (atomic-tp t1))(b (car a)));FIXME organization
  (when a
    (when (constant-type-p b)
      (object-type (funcall f b)))))
(dolist (l 'si::(expand-array-element-type
		 cmp-norm-tp sequence-tp-nonsimple-p sequence-tp-lengths make-sequence-element-type
		 expand-deftype sdata-includes
		 lookup-simple-typep-fn lookup-typep-fn))
  (setf (get l 'compiler::c1no-side-effects) t)
  (setf (get l 'compiler::type-propagator) 'compiler::expand-type-propagator))

(defun improper-consp-type-propagator (f t1 &optional t2)
  (declare (ignore f t2))
  (cond ((not (type-and #tsi::improper-cons t1)) #tnull)
	((type>= #tsi::improper-cons t1) #ttrue)))
(dolist (l 'si::(improper-consp))
  (setf (get l 'compiler::c1no-side-effects) t)
  (setf (get l 'compiler::type-propagator) 'compiler::improper-consp-type-propagator))

(defun symbol-gfdef-propagator (f t1 &aux (a (atomic-tp t1)))
  (declare (ignore f))
  (if a (object-type (funid-to-fn (car a))) #tfunction));FIXME 0
(setf (get 'c-symbol-gfdef 'type-propagator) 'symbol-gfdef-propagator)
