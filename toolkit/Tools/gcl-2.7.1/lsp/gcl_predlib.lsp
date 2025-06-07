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


;;;;    predlib.lsp
;;;;
;;;;                              predicate routines

(in-package :system)

(export '(int void static 
	  non-standard-object-compiled-function
	  interpreted-function
	  non-logical-pathname
	  non-standard-base-char true gsym
	  std-instance
	  funcallable-std-instance
	  hash-table-eq hash-table-eql hash-table-equal hash-table-equalp
	  +type-alist+
	  sequencep ratiop short-float-p long-float-p
	  eql-is-eq equal-is-eq equalp-is-eq eql-is-eq-tp equal-is-eq-tp equalp-is-eq-tp
	  +array-types+
	  +aet-type-object+
	  returns-exactly
	  immfix
	  file-input-stream file-output-stream file-io-stream file-probe-stream
	  string-input-stream string-output-stream
	  proper-sequence proper-sequencep proper-cons proper-consp
	  fcomplex dcomplex
	  cnum-type spice
	  resolve-type
	  ldiff-nf))

(defun ldiff-nf-with-last (l tl &aux r rp)
  (declare (optimize (safety 1)))
  (check-type l proper-list)
  (labels ((srch (x)
	     (if (eq x tl) (values r rp)
		 (let ((tmp (cons (car x) nil)))
		   (setq rp (if rp (cdr (rplacd rp tmp)) (setq r tmp)))
		   (srch (cdr x))))))
    (if tl (srch l) (values l nil))))
(setf (get 'ldiff-nf-with-last 'cmp-inline) t)

(defun ldiff-nf (l tl) (values (ldiff-nf-with-last l tl)))
(setf (get 'ldiff-nf 'cmp-inline) t)

(setf (get 'ldiff-nf 'cmp-inline) t)
;(declaim (inline ldiff-nf))

(defconstant +array-type-alist+ (mapcar (lambda (x) (cons x (intern (string-concatenate "ARRAY-" (string x)))))
					+array-types+))


#+(and pre-gcl raw-image)
(defun array-offset (x) (c-array-offset x))
#+(and pre-gcl raw-image)
(defmacro check-type (&rest r) nil)
#+(and pre-gcl raw-image)
(defmacro assert (&rest r) nil)

(defun ratiop (x) (and (rationalp x) (not (integerp x))))

(defun upgraded-complex-part-type (type &optional environment) 
  (declare (ignore environment) (optimize (safety 2)))
  type)

(defmacro check-type-eval (place type)
  `(values (assert (typep ,place ,type) (,place) 'type-error :datum ,place :expected-type ,type)));fixme

;;; COERCE function.
;(defconstant +coerce-list+ '(list vector string array character short-float
;				  long-float float complex function null cons))

(defconstant +objnull+ (objnull))

(defun coerce (object type &aux ntype (atp (listp type)) (ctp (if atp (car type) type)) (tp (when atp (cdr type))))
  (declare (optimize (safety 2))) ;(print (list 'coerce object type))
;  (check-type type (or (member function) type-spec));FIXME
  (case ctp
	(function
	 (let ((object object))
;	   (check-type object (or function (and symbol (not boolean)) (cons (member lambda) t)))
	   (typecase
	    object
	    (function object) 
	    (symbol
	     (let* ((f (c-symbol-gfdef object))(fi (address f))(m (c-symbol-mflag object)))
	       (check-type fi (and fixnum (not (integer #.+objnull+ #.+objnull+))))
	       (check-type m  (integer 0 0))
	       f))
	    (cons (the function (eval object))))))
	;FIXME member
	((list cons vector string array member simple-array non-simple-array)
	 (if (typep object type) object (replace (make-sequence type (length object)) object)))
	(character (character object))
	(short-float (float object 0.0S0))
	(long-float (float object 0.0L0))
	(float (float object))
	(complex (if (typep object type) object
	 (let ((rtp (or (car tp) t)))
	   (complex (coerce (realpart object) rtp) (coerce (imagpart object) rtp)))))
	(otherwise 
	 (cond ((typep object type) object)
	       ((setq ntype (expand-deftype type)) (coerce object ntype))
	       ((check-type-eval object type))))))


(defconstant +ifb+ (- (car (last (multiple-value-list (si::heap-report))))))
(defconstant +ifr+ (ash (- +ifb+)  -1))
(defconstant +ift+ (when (> #.+ifr+ 0) '(integer #.(- +ifr+) #.(1- +ifr+))))


(defun eql-is-eq (x)
  (typecase
   x
   (immfix t)
   (number nil)
   (otherwise t)))
(setf (get 'eql-is-eq 'cmp-inline) t)

;To pevent typep/predicate loops
;(defun eql-is-eq (x) (typep x (funcall (get 'eql-is-eq-tp 'deftype-definition))))
(defun equal-is-eq (x) (typep x (funcall (macro-function (get 'equal-is-eq-tp 'deftype-definition)) nil nil)));FIXME
(defun equalp-is-eq (x) (typep x (funcall (macro-function (get 'equalp-is-eq-tp 'deftype-definition)) nil nil)))

(defun seqindp (x) (and (fixnump x) (>= x 0) (< x array-dimension-limit)))
(si::putprop 'seqindp t 'cmp-inline)

(defun standard-charp (x)
  (when (characterp x)
    (standard-char-p x)))

(defun non-standard-base-char-p (x)
  (and (characterp x) (not (standard-char-p x))))

(defun improper-consp (s &optional (f nil fp) (z (if fp f s)))
  (cond ((atom z) (when fp (when z t)))
	((atom (cdr z)) (when (cdr z) t))
	((eq s f))
	((improper-consp (cdr s) (cddr z)))))


(defconstant most-negative-immfix (or (cadr +ift+) 1))
(defconstant most-positive-immfix (or (caddr +ift+) -1))


(defun gsym-p (x) (when x (unless (eq x t) (unless (keywordp x) (symbolp x)))))

(defun sequencep (x)
  (or (listp x) (vectorp x)))

(defun short-float-p (x)
  (= (c-type x) #.(c-type 0.0s0)))
;  (and (floatp x) (eql x (float x 0.0s0))))

(defun long-float-p (x)
  (= (c-type x) #.(c-type 0.0)))
;  (and (floatp x) (eql x (float x 0.0))))

(defun fcomplexp (x)
  (and (complexp x) (short-float-p (realpart x)) (short-float-p (imagpart x))))

(defun dcomplexp (x)
  (and (complexp x) (long-float-p (realpart x)) (long-float-p (imagpart x))))

(defun proper-consp (x)
  (and (consp x) (not (improper-consp x))))

(defun proper-listp (x)
  (or (null x) (proper-consp x)))

(defun proper-sequencep (x)
  (or (vectorp x) (proper-listp x)))

(defun type-list-p (spec r &aux s)
  (not (member-if-not (lambda (x &aux (q (member x r)))
			(or (when q (setq s (car q) r (cdr q)) q)
			    (unless (eq s '&allow-other-keys)
			      (when (typep x (if (eq s '&key) '(cons keyword (cons type-spec null)) 'type-spec))
				(if (eq s '&rest) (setq s '&allow-other-keys) t))))) spec)))

(defun arg-list-type-p (x) (type-list-p x '(&optional &rest &key)))

(defun values-list-type-p (x)
  (if (when (listp x) (eq (car x) 'values))
      (type-list-p (cdr x) '(&optional &rest &allow-other-keys))
    (typep x 'type-spec)))

(defun structurep (x)
  (typecase x (structure t)))

(defconstant +type-alist+ '((null . null)
	  (not-type . not)
          (symbol . symbolp)
          (eql-is-eq-tp . eql-is-eq)
          (equal-is-eq-tp . equal-is-eq)
          (equalp-is-eq-tp . equalp-is-eq)
          (keyword . keywordp)
	  ;; (non-logical-pathname . non-logical-pathname-p)
	  (logical-pathname . logical-pathname-p)
	  (proper-cons . proper-consp)
	  (proper-list . proper-listp)
	  (proper-sequence . proper-sequencep)
;	  (non-keyword-symbol . non-keyword-symbol-p)
	  (gsym . gsym-p)
	  (standard-char . standard-charp)
	  (non-standard-base-char . non-standard-base-char-p)
;	  (interpreted-function . interpreted-function-p)
	  (real . realp)
	  (float . floatp)
	  (short-float . short-float-p)
	  (long-float . long-float-p)
	  (fcomplex . fcomplexp)
	  (dcomplex . dcomplexp)
	  (array . arrayp)
	  (vector . vectorp)
	  (bit-vector . bit-vector-p)
	  (string . stringp)
	  (complex . complexp)
	  (ratio . ratiop)
	  (sequence . sequencep)
          (atom . atom)
          (cons . consp)
          (list . listp)
          (seqind . seqindp)
          (fixnum . fixnump)
          (integer . integerp)
          (rational . rationalp)
          (number . numberp)
          (character . characterp)
          (package . packagep)
          (stream . streamp)
          (pathname . pathnamep)
          (readtable . readtablep)
          (hash-table . hash-table-p)
          (hash-table-eq . hash-table-eq-p)
          (hash-table-eql . hash-table-eql-p)
          (hash-table-equal . hash-table-equal-p)
          (hash-table-equalp . hash-table-equalp-p)
          (random-state . random-state-p)
          (structure . structurep)
          (function . functionp)
          (immfix . immfixp)
	  (improper-cons . improper-consp)
          ;; (compiled-function . compiled-function-p)
          ;; (non-generic-compiled-function . non-generic-compiled-function-p)
          ;; (generic-function . generic-function-p)
	  ))

(dolist (l +type-alist+)
  (when (symbolp (cdr l)) 
    (putprop (cdr l) (car l) 'predicate-type)))


(defconstant +singleton-types+ '(null true gsym keyword standard-char
				      non-standard-base-char 
				      package
				      broadcast-stream concatenated-stream echo-stream
				      file-input-stream file-output-stream
				      file-io-stream file-probe-stream
				      string-input-stream string-output-stream
				      file-synonym-stream non-file-synonym-stream two-way-stream 
				      non-logical-pathname logical-pathname
				      readtable 
				      hash-table-eq hash-table-eql hash-table-equal hash-table-equalp
				      random-state
				      interpreted-function
				      non-standard-object-compiled-function
				      spice))


(defconstant +range-types+ `(integer ratio short-float long-float))
(defconstant +complex-types+ `(integer ratio short-float long-float))

(mapc (lambda (x) (setf (get x 'cmp-inline) t))
      '(lremove lremove-if lremove-if-not lremove-duplicates lreduce))

(defun lremove (q l &key (key #'identity) (test #'eql) &aux r rp (p l))
  (declare (proper-list l));FIXME
  (mapl (lambda (x)
		(when (funcall test q (funcall key (car x)))
		  (let ((y (ldiff-nf p x)))
		    (setq rp (last (if rp (rplacd rp y) (setq r y)))
			  p (cdr x))))) l)
  (cond (rp (rplacd rp p) r)
	(p)))

(defun lremove-if (f l) (lremove f l :test 'funcall))
(defun lremove-if-not (f l) (lremove (lambda (x) (not (funcall f x))) l :test 'funcall))

(defun lremove-duplicates (l &key (test #'eql))
  (lremove-if (lambda (x) (member x (setq l (cdr l)) :test test)) l))


(defun lreduce (f l &key (key #'identity) (initial-value nil ivp))
  (labels ((rl (s &optional (res initial-value)(ft ivp))
	       (if s (rl (cdr s) (let ((k (funcall key (car s)))) (if ft (funcall f res k) k)) t)
		 (if ft res (values (funcall f))))))
	  (rl l)))

(defun rational (x)
  (declare (optimize (safety 1)))
  (check-type x real)
  (if (rationalp x) x ;too early for typecase
      (multiple-value-bind
	    (i e s) (integer-decode-float x)
	(let ((x (if (>= e 0) (ash i e) (/ i (ash 1 (- e))))))
	  (if (>= s 0) x (- x))))))

(defun ordered-intersection-eq (l1 l2)
  (let (z zt)
    (do ((l l1 (cdr l))) ((not l))
      (when (memq (car l) l2)
	(setf zt (let ((p (cons (car l) nil))) (if zt (cdr (rplacd zt p)) (setf z p))))))
    z))

(defun ordered-intersection (l1 l2)
  (let (z zt)
    (do ((l l1 (cdr l))) ((not l))
      (when (member (car l) l2)
	(setf zt (let ((p (cons (car l) nil))) (if zt (cdr (rplacd zt p)) (setf z p))))))
    z))

(defun expand-array-element-type (type)
   (or (car (member type +array-types+ :test (lambda (x y) (unless (eq y t) (subtypep x y))))) t))

#.`(defun upgraded-array-element-type (type &optional environment)
     (declare (ignore environment) (optimize (safety 1)))
     (case type
	   ((nil t) type)
	   ,@(mapcar (lambda (x) `(,x type)) (cons '* (lremove t +array-types+)))
	   (otherwise (expand-array-element-type type))))

;; CLASS HACKS

(eval-when
 (compile eval)
 (defmacro clh nil
  `(progn
     ,@(mapcar (lambda (x &aux (f (when (eq x 'find-class) `(&optional ep))) (z (intern (string-concatenate "SI-" (symbol-name x)))))
		 `(defun ,z (o ,@f &aux e (x ',x) (fn (load-time-value nil)))
		    (declare (notinline find-class));to enable recompile file in ansi image
		    (setq fn (or fn (and (fboundp 'classp) (fboundp x) x)))
		    (cond (fn (values (funcall fn o ,@(cdr f))))
			  ((setq e (get ',z 'early)) (values (funcall e o ,@(cdr f)))))))
	       '(classp class-precedence-list find-class class-name class-of class-direct-subclasses))
     (let (fun)
       (defun si-class-finalized-p (x)
	 (unless fun (let* ((p (find-package "PCL"))(sym (when p (find-symbol "CLASS-FINALIZED-P" p))))
		       (when (and sym (fboundp sym) (fboundp 'classp))
			 (setq fun (symbol-function sym)))))
	 (when (and fun (funcall fun x)) t)))
     )))
(clh)

(let ((h (make-hash-table :test 'eq)))
  (defun si-cpl-or-nil (x)
    (or (gethash x h)
	(let ((y (when (si-class-finalized-p x) (si-class-precedence-list x))))
	  (when y (setf (gethash x h) y))))))

;(defun si-cpl-or-nil (x) (when (si-class-finalized-p x) (si-class-precedence-list x)))

(defun is-standard-class (object &aux (o (load-time-value nil)))
  (and (si-classp object)
       (member (or o (setq o (si-find-class 'standard-object)))
	       (si-cpl-or-nil object))
       object))

(defun find-standard-class (object)
  (when (symbolp object)
    (is-standard-class (si-find-class object nil))))

(defun coerce-to-standard-class (object)
  (is-standard-class (if (symbolp object) (si-find-class object nil) object)))


(defun get-included (name)
  (cons name (mapcan 'get-included (sdata-included (get name 's-data)))))

    
;; set by unixport/init_kcl.lsp
;; warn if a file was comopiled in another version
(defvar *gcl-extra-version* nil)
(defvar *gcl-minor-version* nil)
(defvar *gcl-major-version* nil)
(defvar *gcl-git-tag* nil)
(defvar *gcl-release-date*  nil)

(defun warn-version (majvers minvers extvers)
  (and *gcl-major-version* *gcl-minor-version* *gcl-extra-version*
       (or (not (eql extvers *gcl-extra-version*))
	   (not (eql minvers *gcl-minor-version*))
	   (not (eql majvers *gcl-major-version*)))
       *load-verbose*
       (format t "[compiled in GCL ~a.~a.~a] " majvers minvers extvers)))

(defconstant +array-typep-alist+
  (mapcar (lambda (x)
	    (cons x
		  (mapcar (lambda (y &aux (q (intern (string-concatenate (string x) "-" (string y))))
				     (f (intern (string-concatenate (string q) "-SIMPLE-TYPEP-FN"))))
			    (list* y q f))
			  (list* '* nil +array-types+))))
	  '(array simple-array non-simple-array matrix vector)))

