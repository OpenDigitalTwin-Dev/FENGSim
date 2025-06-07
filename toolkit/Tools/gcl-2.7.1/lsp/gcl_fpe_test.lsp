;; Copyright (C) 2024 Camm Maguire
(in-package :si)

#.`(defun test-fpe (f a r &optional chk &aux cc (o (mapcan (lambda (x) (list x t)) (break-on-floating-point-exceptions))))
     (flet ((set-break (x) (when (keywordp r)
			     (apply 'break-on-floating-point-exceptions (append (unless x o) (list r x))))))
       (let* ((rr (handler-case (unwind-protect (progn (set-break t) (apply f a)) (set-break nil))
				,@(mapcar (lambda (x &aux (x (car x))) `(,x (c) (setq cc c) ,(intern (symbol-name x) :keyword)))
					  (append si::+fe-list+ '((arithmetic-error)(error)))))))
	 (print (list* f a r rr (when cc (list cc (arithmetic-error-operation cc) (arithmetic-error-operands cc)))))
	 (assert (eql r rr))
	 (when (and chk cc)
	   (unless (eq 'fnop (cadr (member :op (arithmetic-error-operation cc))))
	     (assert (eq (symbol-function f) (cadr (member :fun (arithmetic-error-operation cc)))))
	     (assert (or (every 'equalp (mapcar (lambda (x) (if (numberp x) x (coerce x 'list))) a)
				(arithmetic-error-operands cc))
			 (every 'equalp (nreverse (mapcar (lambda (x) (if (numberp x) x (coerce x 'list))) a))
				(arithmetic-error-operands cc)))))))))

#+(or x86_64 i386)
(progn
  (eval-when
      (compile eval)
    (defmacro deft (n rt args &rest code)
      `(progn
	 (clines ,(nstring-downcase
		   (apply 'concatenate 'string
			  "static " (symbol-name rt) " " (symbol-name n) "("
			  (apply 'concatenate 'string 
				 (mapcon (lambda (x) (list* (symbol-name (caar x)) " " (symbol-name (cadar x)) 
							    (when (cdr x) (list ", ")))) args)) ") " code)))
	 (defentry ,n ,(mapcar 'car args) (,rt ,(string-downcase (symbol-name n)))))))
  
  (deft fdivp object ((object x) (object y))
    "{volatile double a=lf(x),b=lf(y),c;"
    "__asm__ __volatile__ (\"fldl %1;fldl %0;fdivp %%st,%%st(1);fstpl %2;fwait\" "
    ": \"=m\" (a), \"=m\" (b) : \"m\" (c));"
    "return make_longfloat(c);}")
  
  (deft fsqrt object ((object x))
    "{volatile double a=lf(x),c;"
    "__asm__ __volatile__ (\"fldl %0;fsqrt ;fstpl %1;fwait\" "
    ": \"=m\" (a): \"m\" (c));"
    "return make_longfloat(c);}")

  (deft divpd object ((object x) (object y) (object z))
    "{__asm__ __volatile__ (\"movapd %0,%%xmm0;movapd %1,%%xmm1;divpd %%xmm0,%%xmm1;movapd %%xmm1,%2\" "
    ": \"=m\" (*(char *)x->a.a_self), \"=m\" (*(char *)y->a.a_self) : \"m\" (*(char *)z->a.a_self));"
    "return z;}")
  
  (deft divpdm object ((object x) (object y) (object z))
    "{__asm__ __volatile__ (\"movapd %1,%%xmm1;divpd %0,%%xmm1;movapd %%xmm1,%2\" "
    ": \"=m\" (*(char *)x->a.a_self), \"=m\" (*(char *)y->a.a_self) : \"m\" (*(char *)z->a.a_self));"
    "return z;}")
  
  (deft divps object ((object x) (object y) (object z))
    "{__asm__ __volatile__ (\"movaps %0,%%xmm0;movaps %1,%%xmm1;divps %%xmm0,%%xmm1;movaps %%xmm1,%2\" "
    ": \"=m\" (*(char *)x->a.a_self), \"=m\" (*(char *)y->a.a_self) : \"m\" (*(char *)z->a.a_self));"
    "return z;}")
  
  (deft divpsm object ((object x) (object y) (object z))
    "{__asm__ __volatile__ (\"movaps %1,%%xmm1;divps %0,%%xmm1;movaps %%xmm1,%2\" "
    ": \"=m\" (*(char *)x->a.a_self), \"=m\" (*(char *)y->a.a_self) : \"m\" (*(char *)z->a.a_self));"
    "return z;}")
  
  (deft divsd object ((object x) (object y))
    "{volatile double a=lf(x),b=lf(y),c;"
    "__asm__ __volatile__ (\"movsd %0,%%xmm0;movsd %1,%%xmm1;divsd %%xmm1,%%xmm0;movsd %%xmm0,%2\" "
    ": \"=m\" (a), \"=m\" (b) : \"m\" (c));"
    "return make_longfloat(c);}")
  
  (deft divsdm object ((object x) (object y))
    "{volatile double a=lf(x),b=lf(y),c;"
    "__asm__ __volatile__ (\"movsd %0,%%xmm0;divsd %1,%%xmm0;movsd %%xmm0,%2\" "
    ": \"=m\" (a), \"=m\" (b) : \"m\" (c));"
    "return make_longfloat(c);}")
  
  (deft divss object ((object x) (object y))
    "{volatile float a=sf(x),b=sf(y),c;"
    "__asm__ __volatile__ (\"movss %0,%%xmm0;movss %1,%%xmm1;divss %%xmm1,%%xmm0;movss %%xmm0,%2\" "
    ": \"=m\" (a), \"=m\" (b) : \"m\" (c));"
    "return make_shortfloat(c);}")
  
  (deft divssm object ((object x) (object y))
    "{volatile float a=sf(x),b=sf(y),c;"
    "__asm__ __volatile__ (\"movss %0,%%xmm0;divss %1,%%xmm0;movss %%xmm0,%2\" "
    ": \"=m\" (a), \"=m\" (b) : \"m\" (c));"
    "return make_shortfloat(c);}")
  
  (deft sqrtpd object ((object x) (object y) (object z))
    "{__asm__ __volatile__ (\"movapd %0,%%xmm0;movapd %1,%%xmm1;sqrtpd %%xmm0,%%xmm1;movapd %%xmm1,%2\" "
    ": \"=m\" (*(char *)x->a.a_self), \"=m\" (*(char *)y->a.a_self) : \"m\" (*(char *)z->a.a_self));"
    "return z;}")
  
  (eval-when
      (compile load eval)
    (deft c_array_self fixnum ((object x)) "{return (fixnum)x->a.a_self;}")
    (defun c-array-eltsize (x) (ecase (array-element-type x) (short-float 4) (long-float 8)))
    (defun make-aligned-array (alignment size &rest r
					 &aux (ic (member :initial-contents r)) y
					 (c (cadr ic))
					 (r (append (ldiff-nf r ic) (cddr ic)))
					 (a (apply 'make-array (+ alignment size) (list* :static t r))))
      (setq y (map-into
	       (apply 'make-array size
		      :displaced-to a
		      :displaced-index-offset (truncate (- alignment (mod (c_array_self a) alignment)) (c-array-eltsize a))
		      r)
	       'identity c))
      (assert (zerop (mod (c_array_self y) 16)))
      y))
  
  (setq fa (make-aligned-array 16 4 :element-type 'short-float :initial-contents '(1.2s0 2.3s0 3.4s0 4.1s0))
	fb (make-aligned-array 16 4 :element-type 'short-float)
	fc (make-aligned-array 16 4 :element-type 'short-float :initial-contents '(1.3s0 2.4s0 3.5s0 4.6s0))
	fx (make-aligned-array 16 4 :element-type 'short-float :initial-contents (make-list 4 :initial-element most-positive-short-float))
	fm (make-aligned-array 16 4 :element-type 'short-float :initial-contents (make-list 4 :initial-element least-positive-normalized-short-float))
	fn (make-aligned-array 16 4 :element-type 'short-float :initial-contents (make-list 4 :initial-element -1.0s0))
	fr (make-aligned-array 16 4 :element-type 'short-float))
  
  (setq da (make-aligned-array 16 2 :element-type 'long-float :initial-contents '(1.2 2.3))
	db (make-aligned-array 16 2 :element-type 'long-float)
	dc (make-aligned-array 16 2 :element-type 'long-float :initial-contents '(1.3 2.4))
	dx (make-aligned-array 16 2 :element-type 'long-float :initial-contents (make-list 2 :initial-element most-positive-long-float))
	dm (make-aligned-array 16 2 :element-type 'long-float :initial-contents (make-list 2 :initial-element least-positive-normalized-long-float))
	dn (make-aligned-array 16 2 :element-type 'long-float :initial-contents (make-list 2 :initial-element -1.0))
	dr (make-aligned-array 16 2 :element-type 'long-float))
  
  (test-fpe 'fdivp (list 1.0 2.0) 0.5 t)
  (test-fpe 'fdivp (list 1.0 0.0) :division-by-zero t)
  (test-fpe 'fdivp (list 0.0 0.0) :floating-point-invalid-operation t)
  (test-fpe 'fdivp (list most-positive-long-float least-positive-normalized-long-float) :floating-point-overflow);fstpl
  (test-fpe 'fdivp (list least-positive-normalized-long-float most-positive-long-float) :floating-point-underflow);fstpl
  (test-fpe 'fdivp (list 1.2 1.3) :floating-point-inexact);post args
  
  (test-fpe 'divpd (list da da dr) dr t)
  (test-fpe 'divpd (list db da dr) :division-by-zero t)
  (test-fpe 'divpd (list db db dr) :floating-point-invalid-operation t)
  (test-fpe 'divpd (list dm dx dr) :floating-point-overflow t)
  (test-fpe 'divpd (list dx dm dr) :floating-point-underflow t)
  (test-fpe 'divpd (list da dc dr) :floating-point-inexact t)
  
  (test-fpe 'divpdm (list da da dr) dr t)
  (test-fpe 'divpdm (list db da dr) :division-by-zero t)
  (test-fpe 'divpdm (list db db dr) :floating-point-invalid-operation t)
  (test-fpe 'divpdm (list dm dx dr) :floating-point-overflow t)
  (test-fpe 'divpdm (list dx dm dr) :floating-point-underflow t)
  (test-fpe 'divpdm (list da dc dr) :floating-point-inexact t)
  
  
  (test-fpe 'divps (list fa fa fr) fr t)
  (test-fpe 'divps (list fb fa fr) :division-by-zero t)
  (test-fpe 'divps (list fb fb fr) :floating-point-invalid-operation t)
  (test-fpe 'divps (list fm fx fr) :floating-point-overflow t)
  (test-fpe 'divps (list fx fm fr) :floating-point-underflow t)
  (test-fpe 'divps (list fa fc fr) :floating-point-inexact t)
  
  (test-fpe 'divpsm (list fa fa fr) fr t)
  (test-fpe 'divpsm (list fb fa fr) :division-by-zero t)
  (test-fpe 'divpsm (list fb fb fr) :floating-point-invalid-operation t)
  (test-fpe 'divpsm (list fm fx fr) :floating-point-overflow t)
  (test-fpe 'divpsm (list fx fm fr) :floating-point-underflow t)
  (test-fpe 'divpsm (list fa fc fr) :floating-point-inexact t)
  
  
  
  (test-fpe 'divsd (list 1.0 2.0) 0.5 t)
  (test-fpe 'divsd (list 1.0 0.0) :division-by-zero t)
  (test-fpe 'divsd (list 0.0 0.0) :floating-point-invalid-operation t)
  (test-fpe 'divsd (list most-positive-long-float least-positive-normalized-long-float) :floating-point-overflow t)
  (test-fpe 'divsd (list least-positive-normalized-long-float most-positive-long-float) :floating-point-underflow t)
  (test-fpe 'divsd (list 1.2 2.3) :floating-point-inexact t)
  
  (test-fpe 'divsdm (list 1.0 2.0) 0.5 t)
  (test-fpe 'divsdm (list 1.0 0.0) :division-by-zero t)
  (test-fpe 'divsdm (list 0.0 0.0) :floating-point-invalid-operation t)
  (test-fpe 'divsdm (list most-positive-long-float least-positive-normalized-long-float) :floating-point-overflow t)
  (test-fpe 'divsdm (list least-positive-normalized-long-float most-positive-long-float) :floating-point-underflow t)
  (test-fpe 'divsdm (list 1.2 2.3) :floating-point-inexact t)
  
  (test-fpe 'divss (list 1.0s0 2.0s0) 0.5s0 t)
  (test-fpe 'divss (list 1.0s0 0.0s0) :division-by-zero t)
  (test-fpe 'divss (list 0.0s0 0.0s0) :floating-point-invalid-operation t)
  (test-fpe 'divss (list most-positive-short-float least-positive-normalized-short-float) :floating-point-overflow t)
  (test-fpe 'divss (list least-positive-normalized-short-float most-positive-short-float) :floating-point-underflow t)
  (test-fpe 'divss (list 1.2s0 2.3s0) :floating-point-inexact t)
  
  (test-fpe 'divssm (list 1.0s0 2.0s0) 0.5s0 t)
  (test-fpe 'divssm (list 1.0s0 0.0s0) :division-by-zero t)
  (test-fpe 'divssm (list 0.0s0 0.0s0) :floating-point-invalid-operation t)
  (test-fpe 'divssm (list most-positive-short-float least-positive-normalized-short-float) :floating-point-overflow t)
  (test-fpe 'divssm (list least-positive-normalized-short-float most-positive-short-float) :floating-point-underflow t)
  (test-fpe 'divssm (list 1.2s0 2.3s0) :floating-point-inexact t)
  
  (test-fpe 'sqrtpd (list da db dr) dr t)
  (test-fpe 'sqrtpd (list dn db dr) :floating-point-invalid-operation t)
  (test-fpe 'sqrtpd (list da db dr) :floating-point-inexact t))


(defun l/ (x y) (declare (long-float x y)) (/ x y))
(defun s/ (x y) (declare (short-float x y)) (/ x y))
;(defun lsqrt (x) (declare (long-float x)) (lit :double "sqrt(" (:double x) ")"));(the long-float (|libm|:|sqrt| x))


(test-fpe 'l/ (list 1.0 2.0) 0.5 t)
(test-fpe 'l/ (list 1.0 0.0) :division-by-zero t)
(test-fpe 'l/ (list 0.0 0.0) :floating-point-invalid-operation t)
(test-fpe 'l/ (list most-positive-long-float least-positive-normalized-long-float) :floating-point-overflow t)
(test-fpe 'l/ (list least-positive-normalized-long-float most-positive-long-float) :floating-point-underflow t)
(test-fpe 'l/ (list 1.2 1.3) :floating-point-inexact t)

(test-fpe 's/ (list 1.0s0 2.0s0) 0.5s0 t)
(test-fpe 's/ (list 1.0s0 0.0s0) :division-by-zero t)
(test-fpe 's/ (list 0.0s0 0.0s0) :floating-point-invalid-operation t)
(test-fpe 's/ (list most-positive-short-float least-positive-normalized-short-float) :floating-point-overflow t)
(test-fpe 's/ (list least-positive-normalized-short-float most-positive-short-float) :floating-point-underflow t)
(test-fpe 's/ (list 1.2s0 1.3s0) :floating-point-inexact t)

(test-fpe 'fsqrt (list 4.0) 2.0 t)
(test-fpe 'fsqrt (list -1.0) :floating-point-invalid-operation t)
;(test-fpe 'lsqrt (list -1.0) :floating-point-invalid-operation t)
;(test-fpe '|libm|:|sqrt| (list -1.0) :floating-point-invalid-operation t)
(test-fpe 'fsqrt (list 1.2) :floating-point-inexact t)
