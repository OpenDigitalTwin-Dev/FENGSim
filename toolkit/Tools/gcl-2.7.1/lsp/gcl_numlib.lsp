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


;;;;    numlib.lsp
;;;;
;;;;                           number routines


(in-package :system)

(defun powm (a b c)
  (declare (optimize (safety 1)))
  (check-type a integer)
  (check-type b (integer 0))
  (check-type c (integer 0))
  (if (typep b 'fixnum)
      (gmp:mpz_powm_ui a b c)
      (gmp:mpz_powm a b c)))
(declaim (inline powm))

(defconstant imag-one #C(0.0d0 1.0d0))

(defun isqrt (i)
  (declare (optimize (safety 1)))
  (check-type i (integer 0))
  (typecase
   i
   (fixnum (do* ((y 0 (floor i x))
		 (x (ash 1 (ceiling (integer-length i) 2)) (+ (ash x -1) (ash y -1) (logand x y 1))))
		((<= x y) x)))
   (otherwise (mpz_sqrt i))))

(deftype bytespec nil `(cons (integer 0) (integer 0)))

(defun byte (size position)
  (declare (optimize (safety 1)))
  (check-type size (integer 0))
  (check-type position (integer 0))
  (cons size position))

(defun byte-position (bytespec)
  (declare (optimize (safety 1)))
  (check-type bytespec cons)
  (cdr bytespec))

(defun byte-size (bytespec)
  (declare (optimize (safety 1)))
  (check-type bytespec cons)
  (car bytespec))

(defun ldb (bytespec integer)
  (declare (optimize (safety 1)))
  (check-type bytespec bytespec)
  (check-type integer integer)
  (logand (ash integer (- (byte-position bytespec)))
	  (1- (ash 1 (byte-size bytespec)))))

(defun ldb-test (bytespec integer)
  (declare (optimize (safety 1)))
  (check-type bytespec bytespec)
  (check-type integer integer)
  (not (zerop (ldb bytespec integer))))

(defun dpb (newbyte bytespec integer &aux (z (1- (ash 1 (byte-size bytespec)))))
  (declare (optimize (safety 1)))
  (check-type newbyte integer)
  (check-type bytespec bytespec)
  (check-type integer integer)
  (logior (logandc2 integer (ash z (byte-position bytespec)))
	  (ash (logand newbyte z) (byte-position bytespec))))

(defun deposit-field (newbyte bytespec integer &aux (z (ash (1- (ash 1 (byte-size bytespec))) (byte-position bytespec))))
  (declare (optimize (safety 1)))
  (check-type newbyte integer)
  (check-type bytespec bytespec)
  (check-type integer integer)
  (logior (logandc2 integer z) (logand newbyte z)))

(defun mask-field (bytespec integer)
  (declare (optimize (safety 1)))
  (check-type bytespec bytespec)
  (check-type integer integer)
  (logand integer (ash (1- (ash 1 (byte-size bytespec))) (byte-position bytespec))))


(defun phase (x)
  (declare (optimize (safety 1)))
  (check-type x number)
  (if (= 0 x) 0.0
    (atan (imagpart x) (realpart x))))

(defun signum (x) 
  (declare (optimize (safety 1)))
  (check-type x number)
  (if (zerop x) x
      (typecase x
	(rational (if (minusp x) -1 1))
	(short-float (if (minusp x) -1.0s0 1.0s0))
	(long-float (if (minusp x) -1.0 1.0))
	(fcomplex (/ x (abs x)))
	(dcomplex (/ x (abs x)))
	(complex (let* ((y (max (abs (realpart x)) (abs (imagpart x))))
			(z (complex (/ (realpart x) y) (/ (imagpart x) y))))
		   (/ z (abs z)))))))

(defun cis (x) 
  (declare (optimize (safety 1)))
  (check-type x real)
  (exp (* #c(0 1) (float x))))


(defun ffloor (x &optional (y 1.0s0))
  (declare (optimize (safety 1)))
  (check-type x real)
  (check-type y real)
  (multiple-value-bind (i r) (floor x y)
    (values (float i (if (floatp x) x 1.0)) r)))

(defun fceiling (x &optional (y 1.0s0))
  (declare (optimize (safety 1)))
  (check-type x real)
  (check-type y real)
  (multiple-value-bind (i r) (ceiling x y)
    (values (float i (if (floatp x) x 1.0)) r)))

(defun ftruncate (x &optional (y 1.0s0))
  (declare (optimize (safety 1)))
  (check-type x real)
  (check-type y real)
  (multiple-value-bind (i r) (truncate x y)
    (values (float i (if (floatp x) x 1.0)) r)))

(defun fround (x &optional (y 1.0s0))
  (declare (optimize (safety 1)))
  (check-type x real)
  (check-type y real)
  (multiple-value-bind (i r) (round x y)
    (values (float i (if (floatp x) x 1.0)) r)))


(defun logtest (x y) 
  (declare (optimize (safety 1)))
  (check-type x integer)
  (check-type y integer)
  (not (zerop (logand x y))))

(defconstant +make-complex-alist+
  `((complex-integer #tinteger #tinteger)
    (complex-integer-ratio #tinteger #tratio)
    (complex-ratio-integer #tratio #tinteger)
    (complex-ratio #tratio #tratio)
    (complex-short-float #tshort-float #tshort-float)
    (complex-long-float #tlong-float #tlong-float)))

(eval-when (compile eval) (defmacro complex-tt (s) (or (position s +make-complex-alist+ :key 'car) (baboon))))

(defun complex (rp &optional (ip (typecase rp (rational 0)(short-float 0.0s0)(long-float 0.0))))
  (declare (optimize (safety 1)))
  (check-type rp real)
  (check-type ip real)
  (typecase rp
    (integer
     (typecase ip
       ((integer 0 0) rp)
       (integer       (make-complex #.(complex-tt complex-integer)       rp            ip))
       (ratio         (make-complex #.(complex-tt complex-integer-ratio) rp            ip))
       (short-float   (make-complex #.(complex-tt complex-short-float)   (float rp ip) ip))
       (long-float    (make-complex #.(complex-tt complex-long-float)    (float rp ip) ip))))
    (ratio
     (typecase ip
       ((integer 0 0) rp)
       (integer       (make-complex #.(complex-tt complex-ratio-integer) rp            ip))
       (ratio         (make-complex #.(complex-tt complex-ratio)         rp            ip))
       (short-float   (make-complex #.(complex-tt complex-short-float)   (float rp ip) ip))
       (long-float    (make-complex #.(complex-tt complex-long-float)    (float rp ip) ip))))
    (short-float
     (typecase ip
       (rational      (make-complex #.(complex-tt complex-short-float)   rp            (float ip rp)))
       (short-float   (make-complex #.(complex-tt complex-short-float)   rp            ip))
       (long-float    (make-complex #.(complex-tt complex-long-float)    (float rp ip) ip))))
    (long-float       (make-complex #.(complex-tt complex-long-float)    rp            (float ip rp)))))

(defun make-complex-propagator (f t1 t2 t3 &aux (i -1))
  (declare (ignore f))
  (reduce 'tp-or
	  (mapcan (lambda (x)
		    (when (tp-and t1 (object-tp (incf i)))
		      (list (cmp-norm-tp `(complex* ,(cmp-unnorm-tp (tp-and t2 (cadr x))) ,(cmp-unnorm-tp (tp-and t3 (caddr x))))))))
		  +make-complex-alist+)
	  :initial-value nil))
(setf (get 'make-complex 'type-propagator) 'make-complex-propagator)

(defun float-digits (x)
  (declare (optimize (safety 1)))
  (check-type x float);FIXME etypecase
  (typecase x
    (short-float 24)
    (t 53)))

(defun float-precision (x)
  (declare (optimize (safety 1)))
  (check-type x float);FIXME etypecase
  (typecase x
    ((member 0.0 0.0s0) 0)
    (short-float 24)
    (t 53)))

(defun float-sign (x &optional (y 1.0))
  (declare (optimize (safety 1)))
  (check-type x float)
  (check-type y float)
  (let ((y (float (abs y) x)))
    (if (minusp x) (- y) y)))

(defun float-radix (x)
  (declare (optimize (safety 1)))
  (check-type x float);FIXME etypecase
  2)

(defun atomic-tp-propagator (f &rest r);  tp &aux (atp (atomic-tp tp)))
  (declare (dynamic-extent r))
  (unless (member-if-not 'atomic-tp r)
    (let ((l (multiple-value-list (apply f (mapcar (lambda (x) (car (atomic-tp x))) r)))))
      (if (cdr l)
	  `(returns-exactly ,@(mapcar 'object-tp l))
	  (object-tp (car l))))))

(dolist (l '(integer-decode-float decode-float scale-float));float-radix float-digits float-precision float-sign
  (setf (get l 'type-propagator) 'atomic-tp-propagator (get l 'compiler::c1no-side-effects) t))

(declaim (inline fryi))
(defun fryi (x a)
  (labels ((fryn (x a) (abs (- (* x (denominator a)) (numerator a))))
	   (fryk (x a b &aux (c (fryn x a))(d (fryn x b))
		    (kf 0.8);heuristic guard against overshoot
		    (cf (* c kf))(df (* d kf)))
	     (cond ((> cf d 0) (values (truncate (/ cf d))))
		   ((> df c 0) (values (truncate (/ df c))))
		   (1)))
	   (med (a b k)
	     (/ (+ (numerator a)   (* k (numerator b)))
		(+ (denominator a) (* k (denominator b)))))
	   (fry (x a b)
	     (cond ((= (float a x) x) a)
		   ((= (float b x) x) b)
		   ((< (med a b 1) x) (fry x (med a b (fryk x a b)) b))
		   ((fry x a (med b a (fryk x a b)))))))
    (fry x a (1+ a))))

(defun rationalize (x)
  (declare (optimize (safety 1)))
  (check-type x real)
  (typecase x
    (rational x)
    (float
     (if (isnan x)
	 (rational x)
	 (multiple-value-bind
	       (f r) (truncate x)
	   (cond ((minusp r) (fryi x (1- f)))
		 ((zerop r) f)
		 ((fryi x f))))))))
