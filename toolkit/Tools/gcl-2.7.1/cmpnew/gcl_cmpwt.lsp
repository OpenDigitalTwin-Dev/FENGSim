;;; CMPWT  Output routines.
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

(defstruct (fasd (:type vector))
  stream
  table
  eof
  direction
  package
  index
  filepos
  table_length
  evald_forms ; list of forms eval'd. (load-time-eval)
  )

(si::freeze-defstruct 'fasd)

(defvar *fasd-ops*
'(  d_nil         ;/* dnil: nil */
  d_eval_skip    ;    /* deval o1: evaluate o1 after reading it */
  d_delimiter    ;/* occurs after d_listd_general and d_new_indexed_items */
  d_enter_vector ;     /* d_enter_vector o1 o2 .. on d_delimiter  make a cf_data with
		  ;  this length.   Used internally by gcl.  Just make
		  ;  an array in other lisps */
  d_cons        ; /* d_cons o1 o2: (o1 . o2) */
  d_dot         ;
  d_list    ;/* list* delimited by d_delimiter d_list,o1,o2, ... ,d_dot,on
		;for (o1 o2       . on)
		;or d_list,o1,o2, ... ,on,d_delimiter  for (o1 o2 ...  on)
	      ;*/
  d_list1   ;/* nil terminated length 1  d_list1o1   */
  d_list2   ; /* nil terminated length 2 */
  d_list3
  d_list4
  d_eval
  d_short_symbol
  d_short_string
  d_short_fixnum
  d_short_symbol_and_package
  d_bignum
  d_fixnum
  d_string
  d_objnull
  d_structure
  d_package
  d_symbol
  d_symbol_and_package
  d_end_of_file
  d_standard_character
  d_vector
  d_array
  d_begin_dump
  d_general_type
  d_sharp_equals ;              /* define a sharp */
  d_sharp_value
  d_sharp_value2
  d_new_indexed_item
  d_new_indexed_items
  d_reset_index
  d_macro
  d_reserve1
  d_reserve2
  d_reserve3
  d_reserve4
  d_indexed_item3 ;      /* d_indexed_item3 followed by 3bytes to give index */
  d_indexed_item2  ;      /* d_indexed_item2 followed by 2bytes to give index */
  d_indexed_item1 
  d_indexed_item0    ;  /* This must occur last ! */
))

;(require 'FASDMACROS "../cmpnew/gcl_fasdmacros.lsp")
(eval-when (compile eval)
;  (require 'FASDMACROS "../cmpnew/gcl_fasdmacros.lsp")

(defmacro put-op (op str)
  `(write-byte ,(or (position op *fasd-ops*)
		    (error "illegal op")) ,str))

(defmacro put2 (n str)
  `(progn  (write-bytei ,n 0 ,str)
	   (write-bytei  ,n 1 ,str)))
  
(defmacro write-bytei (n i str)
  `(write-byte (the fixnum (ash (the fixnum ,n) >> ,(* i 8))) ,str))
  


;(defmacro data-inits () `(first *data*))
;(defmacro data-dl () `(second *data*))

)

(defun wt-comment (message &optional (symbol nil))
  (princ "
/*	" *compiler-output1*)
  (let* ((mlist (and symbol (list (string symbol))))
	 (mlist (cons message mlist)))
    (dolist (s mlist)
      (declare (string s))
      (dotimes (n (length s))
		 (let ((c (schar s n)))
		   (declare (character c))
		   (unless (char= c #\/)
		     (princ c *compiler-output1*))))))
  (princ "	*/
" *compiler-output1*)
  nil
  )

(defun wt1 (form)
  (cond ((or (stringp form) (integerp form) (characterp form))
         (princ form *compiler-output1*))
        ((or (typep form 'long-float)
             (typep form 'short-float))
         (format *compiler-output1* "~10,,,,,,'eG" form))
        ((or (typep form 'fcomplex)
             (typep form 'dcomplex))
	 (wt "(" (realpart form) " + I * " (imagpart form) ")"))
        (t (wt-loc form)))
  nil)

(defun wt-h1 (form)
  (cond ((consp form)
         (let ((fun (get (car form) 'wt)))
              (if fun
                  (apply fun (cdr form))
                  (cmpiler-error "The location ~s is undefined." form))))
        (t (princ form *compiler-output2*)))
  nil)

(defvar *fasd-data*)

(defvar *hash-eq* nil)
(defvar *run-hash-equal-data-checking* nil)
(defun memoized-hash-equal (x depth);FIXME implement all this in lisp
  (declare (fixnum depth)(inline si::hash-set))
  (unless *run-hash-equal-data-checking*
    (return-from memoized-hash-equal 0))
  (unless *hash-eq* (setq *hash-eq* (make-hash-table :test 'eq)))
  (address
   (or (gethash x *hash-eq*)
       (setf (gethash x *hash-eq*)
	     (nani
	      (if (> depth 3) 0
		  (if (typep x 'cons)
		      (logxor (setq depth (the fixnum (1+ depth)));FIXME?
			      (logxor
			       (memoized-hash-equal (car x) depth)
			       (memoized-hash-equal (cdr x) depth)))
		      (si::hash-equal x depth))))))))

(defun push-data-incf (x)
  (declare (ignore x));FIXME
  (incf *next-vv*))

(defun wt-data1 (expr)
  (terpri *compiler-output-data*)
  (prin1 expr *compiler-output-data*))


(defun add-init (x &optional endp &aux (tem (cons (memoized-hash-equal x -1000) x)))
  (if endp
      (nconc *data* (list tem))
    (push tem *data*))
  x)

(defun add-dl (x &optional endp &aux (tem (cons (memoized-hash-equal x -1000) x)))
  (if endp
      (nconc (data-dl) (list tem))
    (push tem (data-dl)))
  x)

(defun verify-datum (v)
  (unless (eql (pop v) (memoized-hash-equal v -1000))
    (cmpwarn "A form or constant:~% ~s ~%has changed during the eval compile procedure!.~%  The changed form will be the one put in the compiled file" v))
  v)

(defun wt-fasd-element (x)
  (si::find-sharing-top x (fasd-table (car *fasd-data*)))
  (si::write-fasd-top x (car *fasd-data*)))

(defun wt-data2 (x)
  (let ((*print-radix* nil)
        (*print-base* 10)
        (*print-circle* t)
        (*print-pretty* nil)
        (*print-level* nil)
        (*print-length* nil)
        (*print-case* :downcase)
        (*print-gensym* t)
        (*print-array* t)
        (*print-readably* (not *compiler-compile*))
	;;This forces the printer to add the float type in the .data file.
	(*READ-DEFAULT-FLOAT-FORMAT* 'long-float)
        (si::*print-package* t)
        (si::*print-structure* t))
    (if *fasd-data*
	(wt-fasd-element x)
	(wt-data1 x))))


(defun wt-data-file nil
  (when *prof-p* (add-init `(si::mark-memory-as-profiling)))
  (wt-data2 (1+ *next-vv*))
  (cond (*compiler-compile*;FIXME, clean this up
	 (setq *compiler-compile-data* (mapcar 'verify-datum (nreverse *data*)))
	 (wt-data2 `(mapc 'eval *compiler-compile-data*)))
	;; Carefully allow sharing across all data but preseve eval order
	((wt-data2 `'(progn ,@(mapcar (lambda (x) (cons '|#,| (verify-datum x))) (nreverse *data*))))))
  (when *fasd-data*
    (si::close-fasd (car *fasd-data*))))

(defun wt-data-begin ())
(defun wt-data-end ())

(defmacro wt (&rest forms &aux (fl nil))
  (dolist (form forms (cons 'progn (reverse (cons nil fl))))
    (if (stringp form)
        (push `(princ ,form *compiler-output1*) fl)
        (push `(wt1 ,form) fl))))

(defmacro wt-h (&rest forms &aux (fl nil))
  (cond ((endp forms) '(princ "
" *compiler-output2*))
        ((stringp (car forms))
         (dolist (form (cdr forms)
                         (list* 'progn `(princ ,(concatenate 'string "
" (car forms)) *compiler-output2*) (reverse (cons nil fl))))
                   (if (stringp form)
                       (push `(princ ,form *compiler-output2*) fl)
                       (push `(wt-h1 ,form) fl))))
        (t (dolist (form forms
                           (list* 'progn '(princ "
" *compiler-output2*) (reverse (cons nil fl))))
                     (if (stringp form)
                         (push `(princ ,form *compiler-output2*) fl)
                         (push `(wt-h1 ,form) fl))))))

(defmacro wt-nl (&rest forms &aux (fl nil))
  (cond ((endp forms) '(princ "
	" *compiler-output1*))
        ((stringp (car forms))
         (dolist (form (cdr forms)
                         (list* 'progn `(princ ,(concatenate 'string "
	" (car forms)) *compiler-output1*) (reverse (cons nil fl))))
                   (if (stringp form)
                       (push `(princ ,form *compiler-output1*) fl)
                       (push `(wt1 ,form) fl))))
        (t (dolist (form forms
                           (list* 'progn '(princ "
	" *compiler-output1*) (reverse (cons nil fl))))
                     (if (stringp form)
                         (push `(princ ,form *compiler-output1*) fl)
                         (push `(wt1 ,form) fl))))))

(defmacro wt-nl1 (&rest forms &aux (fl nil))
  (cond ((endp forms) '(princ "
" *compiler-output1*))
        ((stringp (car forms))
         (dolist (form (cdr forms)
                         (list* 'progn `(princ ,(concatenate 'string "
" (car forms)) *compiler-output1*) (nreverse (cons nil fl))))
                   (if (stringp form)
                       (push `(princ ,form *compiler-output1*) fl)
                       (push `(wt1 ,form) fl))))
        (t (dolist (form forms
                           (list* 'progn '(princ "
" *compiler-output1*) (nreverse (cons nil fl))))
                     (if (stringp form)
                         (push `(princ ,form *compiler-output1*) fl)
                         (push `(wt1 ,form) fl))))))

