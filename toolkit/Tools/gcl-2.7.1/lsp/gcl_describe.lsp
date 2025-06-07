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


;;;;    describe.lsp
;;;;
;;;;                           DESCRIBE and INSPECT


;; (in-package 'lisp)

;; (export '(describe inspect))


(in-package :system)


(defvar *inspect-level* 0)
(defvar *inspect-history* nil)
(defvar *inspect-mode* nil)

(defvar *old-print-level* nil)
(defvar *old-print-length* nil)


(defun inspect-read-line ()
  (do ((char (read-char *query-io*) (read-char *query-io*)))
      ((or (char= char #\Newline) (char= char #\Return)))))

(defun read-inspect-command (label object allow-recursive)
  (unless *inspect-mode*
    (inspect-indent-1)
    (if allow-recursive
        (progn (princ label) (inspect-object object))
        (format t label object))
    (return-from read-inspect-command nil))
  (loop
    (inspect-indent-1)
    (if allow-recursive
        (progn (princ label)
               (inspect-indent)
               (prin1 object))
        (format t label object))
    (write-char #\Space)
    (force-output)
    (case (do ((char (read-char *query-io*) (read-char *query-io*)))
              ((and (char/= char #\Space) (char/= #\Tab)) char))
      ((#\Newline #\Return)
       (when allow-recursive (inspect-object object))
       (return nil))
      ((#\n #\N)
       (inspect-read-line)
       (when allow-recursive (inspect-object object))
       (return nil))
      ((#\s #\S) (inspect-read-line) (return nil))
      ((#\p #\P)
       (inspect-read-line)
       (let ((*print-pretty* t) (*print-level* nil) (*print-length* nil))
            (prin1 object)
            (terpri)))
      ((#\a #\A) (inspect-read-line) (throw 'abort-inspect nil))
      ((#\u #\U)
       (return (values t (prog1
                          (eval (read-preserving-whitespace *query-io*))
                          (inspect-read-line)))))
      ((#\e #\E)
       (dolist (x (multiple-value-list
                   (multiple-value-prog1
                    (eval (read-preserving-whitespace *query-io*))
                    (inspect-read-line))))
               (write x
                      :level *old-print-level*
                      :length *old-print-length*)
               (terpri)))       
      ((#\q #\Q) (inspect-read-line) (throw 'quit-inspect nil))
      (t (inspect-read-line)
         (terpri)
         (format t
                 "Inspect commands:~%~
		n (or N or Newline):	inspects the field (recursively).~%~
		s (or S):		skips the field.~%~
		p (or P):		pretty-prints the field.~%~
		a (or A):		aborts the inspection ~
					of the rest of the fields.~%~
		u (or U) form:		updates the field ~
					with the value of the form.~%~
		e (or E) form:		evaluates and prints the form.~%~
		q (or Q):		quits the inspection.~%~
		?:			prints this.~%~%")))))

(defmacro inspect-recursively (label object &optional place)
  (if place
      `(multiple-value-bind (update-flag new-value)
            (read-inspect-command ,label ,object t)
         (when update-flag (setf ,place new-value)))
      `(when (read-inspect-command ,label ,object t)
             (princ "Not updated.")
             (terpri))))

(defmacro inspect-print (label object &optional place)
  (if place
      `(multiple-value-bind (update-flag new-value)
           (read-inspect-command ,label ,object nil)
         (when update-flag (setf ,place new-value)))
      `(when (read-inspect-command ,label ,object nil)
             (princ "Not updated.")
             (terpri))))
          
(defun inspect-indent ()
  (fresh-line)
  (format t "~V@T"
          (* 4 (if (< *inspect-level* 8) *inspect-level* 8))))

(defun inspect-indent-1 ()
  (fresh-line)
  (format t "~V@T"
          (- (* 4 (if (< *inspect-level* 8) *inspect-level* 8)) 3)))


(defun inspect-symbol (symbol)
  (let ((p (symbol-package symbol)))
    (cond ((null p)
           (format t "~:@(~S~) - uninterned symbol" symbol))
          ((eq p (find-package "KEYWORD"))
           (format t "~:@(~S~) - keyword" symbol))
          (t
           (format t "~:@(~S~) - ~:[internal~;external~] symbol in ~A package"
                   symbol
                   (multiple-value-bind (b f)
                                        (find-symbol (symbol-name symbol) p)
                     (declare (ignore b))
                     (eq f :external))
                   (package-name p)))))

  (when (boundp symbol)
        (if *inspect-mode*
            (inspect-recursively "value:"
                                 (symbol-value symbol)
                                 (symbol-value symbol))
            (inspect-print "value:~%   ~S"
                           (symbol-value symbol)
                           (symbol-value symbol))))

  (do ((pl (symbol-plist symbol) (cddr pl)))
      ((endp pl))
    (unless (and (symbolp (car pl))
                 (or (eq (symbol-package (car pl)) (find-package 'system))
                     (eq (symbol-package (car pl)) (find-package 'compiler))))
      (if *inspect-mode*
          (inspect-recursively (format nil "property ~S:" (car pl))
                               (cadr pl)
                               (get symbol (car pl)))
          (inspect-print (format nil "property ~:@(~S~):~%   ~~S" (car pl))
                         (cadr pl)
                         (get symbol (car pl))))))
  
  (when (print-doc symbol t)
        (format t "~&-----------------------------------------------------------------------------~%"))
  )

(defun inspect-package (package)
  (format t "~S - package" package)
  (when (package-nicknames package)
        (inspect-print "nicknames:  ~S" (package-nicknames package)))
  (when (package-use-list package)
        (inspect-print "use list:  ~S" (package-use-list package)))
  (when  (package-used-by-list package)
         (inspect-print "used-by list:  ~S" (package-used-by-list package)))
  (when (package-shadowing-symbols package)
        (inspect-print "shadowing symbols:  ~S"
                       (package-shadowing-symbols package))))

(defun inspect-character (character)
  (format t
          (cond ((standard-char-p character) "~S - standard character")
                ((characterp character) "~S - character")
                (t "~S - character"))
          character)
  (inspect-print "code:  #x~X" (char-code character))
  (inspect-print "bits:  ~D" (char-bits character))
  (inspect-print "font:  ~D" (char-font character)))

(defun inspect-number (number)
  (case (type-of number)
    (fixnum (format t "~S - fixnum (32 bits)" number))
    (bignum (format t "~S - bignum" number))
    (ratio
     (format t "~S - ratio" number)
     (inspect-recursively "numerator:" (numerator number))
     (inspect-recursively "denominator:" (denominator number)))
    (complex
     (format t "~S - complex" number)
     (inspect-recursively "real part:" (realpart number))
     (inspect-recursively "imaginary part:" (imagpart number)))
    ((short-float single-float)
     (format t "~S - short-float" number)
     (multiple-value-bind (signif expon sign)
          (integer-decode-float number)
       (declare (ignore sign))
       (inspect-print "exponent:  ~D" expon)
       (inspect-print "mantissa:  ~D" signif)))
    ((long-float double-float)
     (format t "~S - long-float" number)
     (multiple-value-bind (signif expon sign)
          (integer-decode-float number)
       (declare (ignore sign))
       (inspect-print "exponent:  ~D" expon)
       (inspect-print "mantissa:  ~D" signif)))))

(defun inspect-cons (cons)
  (format t
          (case (car cons)
            ((lambda lambda-block lambda-closure lambda-block-closure)
             "~S - function")
            (quote "~S - constant")
            (t "~S - cons"))
          cons)
  (when *inspect-mode*
        (do ((i 0 (1+ i))
             (l cons (cdr l)))
            ((atom l)
             (inspect-recursively (format nil "nthcdr ~D:" i)
                                  l (cdr (nthcdr (1- i) cons))))
          (inspect-recursively (format nil "nth ~D:" i)
                               (car l) (nth i cons)))))

(defun inspect-string (string)
  (format t (if (simple-string-p string) "~S - simple string" "~S - string")
          string)
  (inspect-print  "dimension:  ~D"(array-dimension string 0))
  (when (array-has-fill-pointer-p string)
        (inspect-print "fill pointer:  ~D"
                       (fill-pointer string)
                       (fill-pointer string)))
  (when *inspect-mode*
        (dotimes (i (array-dimension string 0))
                 (inspect-recursively (format nil "aref ~D:" i)
                                      (char string i)
                                      (char string i)))))

(defun inspect-vector (vector)
  (format t (if (simple-vector-p vector) "~S - simple vector" "~S - vector")
          vector)
  (inspect-print  "dimension:  ~D" (array-dimension vector 0))
  (when (array-has-fill-pointer-p vector)
        (inspect-print "fill pointer:  ~D"
                       (fill-pointer vector)
                       (fill-pointer vector)))
  (when *inspect-mode*
        (dotimes (i (array-dimension vector 0))
                 (inspect-recursively (format nil "aref ~D:" i)
                                      (aref vector i)
                                      (aref vector i)))))

(defun inspect-array (array)
  (format t (if (adjustable-array-p array)
                "~S - adjustable aray"
                "~S - array")
          array)
  (inspect-print "rank:  ~D" (array-rank array))
  (inspect-print "dimensions:  ~D" (array-dimensions array))
  (inspect-print "total size:  ~D" (array-total-size array)))

(defun inspect-structure (x &aux name)
  (format t "Structure of type ~a ~%Byte:[Slot Type]Slot Name   :Slot Value"
	  (setq name (type-of x)))
  (let* ((sd (get name 'si::s-data))
	 (spos (s-data-slot-position sd)))
    (dolist (v (s-data-slot-descriptions sd))
	    (format t "~%~4d:~@[[~s] ~]~20a:~s"   
		    (aref spos (nth 4 v))
		    (let ((type (nth 2 v)))
		      (if (eq t type) nil type))
		    (car v)
		    (structure-ref1 x (nth 4 v))))))
    
  
(defun inspect-object (object &aux (*inspect-level* *inspect-level*))
  (inspect-indent)
  (when (and (not *inspect-mode*)
             (or (> *inspect-level* 5)
                 (member object *inspect-history*)))
        (prin1 object)
        (return-from inspect-object))
  (incf *inspect-level*)
  (push object *inspect-history*)
  (catch 'abort-inspect
         (cond ((symbolp object) (inspect-symbol object))
               ((packagep object) (inspect-package object))
               ((characterp object) (inspect-character object))
               ((numberp object) (inspect-number object))
               ((consp object) (inspect-cons object))
               ((stringp object) (inspect-string object))
               ((vectorp object) (inspect-vector object))
               ((arrayp object) (inspect-array object))
	       ((structurep object)(inspect-structure object))
               (t (format t "~S - ~S" object (type-of object))))))


(defun describe (object &optional stream
			&aux (*standard-output* (cond ((eq stream t) *terminal-io*) ((not stream) *standard-output*) (stream)))
			     (*inspect-mode* nil)
                             (*inspect-level* 0)
                             (*inspect-history* nil)
                             (*print-level* nil)
                             (*print-length* nil))
;  "The lisp function DESCRIBE."
  (declare (optimize (safety 2)))
  (terpri)
  (catch 'quit-inspect (inspect-object object))
  (terpri)
  (values))

(defun inspect (object &aux (*inspect-mode* t)
                            (*inspect-level* 0)
                            (*inspect-history* nil)
                            (*old-print-level* *print-level*)
                            (*old-print-length* *print-length*)
                            (*print-level* 3)
                            (*print-length* 3))
;  "The lisp function INSPECT."
  (declare (optimize (safety 2)))
  (read-line)
  (princ "Type ? and a newline for help.")
  (terpri)
  (catch 'quit-inspect (inspect-object object))
  (terpri)
  (values))

(defun print-doc (symbol &optional (called-from-apropos-doc-p nil)
                         &aux (f nil) x)
  (flet ((doc1 (doc ind)
           (setq f t)
           (format t
                   "~&-----------------------------------------------------------------------------~%~53S~24@A~%~A"
                   symbol ind doc))
         (good-package ()
           (if (eq (symbol-package symbol) (find-package "LISP"))
               (find-package "SYSTEM")
               *package*)))

    (cond ((special-operator-p symbol)
           (doc1 (or (real-documentation symbol 'function) "")
                 (if (macro-function symbol)
                     "[Special form and Macro]"
                     "[Special form]")))
          ((macro-function symbol)
           (doc1 (or (real-documentation symbol 'function) "") "[Macro]"))
          ((fboundp symbol)
           (doc1
            (or (real-documentation symbol 'function)
                (if (consp (setq x (function-lambda-expression (symbol-function symbol))))
                    (case (car x)
                          (lambda (format nil "~%Args: ~S" (cadr x)))
                          (lambda-block (format nil "~%Args: ~S" (caddr x)))
                          (lambda-closure
                           (format nil "~%Args: ~S" (car (cddddr x))))
                          (lambda-block-closure
                           (format nil "~%Args: ~S" (cadr (cddddr x))))
                          (t ""))
                    ""))
            "[Function]"))
          ((setq x (real-documentation symbol 'function))
           (doc1 x "[Macro or Function]")))

    (cond ((constantp symbol)
           (unless (and (eq (symbol-package symbol) (find-package "KEYWORD"))
                        (null (real-documentation symbol 'variable)))
             (doc1 (or (real-documentation symbol 'variable) "") "[Constant]")))
          ((si:specialp symbol)
           (doc1 (or (real-documentation symbol 'variable) "")
                 "[Special variable]"))
          ((or (setq x (real-documentation symbol 'variable)) (boundp symbol))
           (doc1 (or x "") "[Variable]")))

    (cond ((setq x (real-documentation symbol 'type))
           (doc1 x "[Type]"))
          ((setq x (get symbol 'deftype-form))
           (let ((*package* (good-package)))
             (doc1 (format nil "~%Defined as: ~S~%See the doc of DEFTYPE." x)
                   "[Type]"))))

    (cond ((setq x (real-documentation symbol 'structure))
           (doc1 x "[Structure]"))
          ((setq x (get symbol 'defstruct-form))
           (doc1 (format nil "~%Defined as: ~S~%See the doc of DEFSTRUCT." x)
                 "[Structure]")))

    (cond ((setq x (real-documentation symbol 'setf))
           (doc1 x "[Setf]"))
          ((setq x (get symbol 'setf-update-fn))
           (let ((*package* (good-package)))
             (doc1 (format nil "~%Defined as: ~S~%See the doc of DEFSETF."
                           `(defsetf ,symbol ,(get symbol 'setf-update-fn)))
                   "[Setf]")))
          ((setq x (get symbol 'setf-lambda))
           (let ((*package* (good-package)))
             (doc1 (format nil "~%Defined as: ~S~%See the doc of DEFSETF."
                           `(defsetf ,symbol ,@(get symbol 'setf-lambda)))
                   "[Setf]")))
          ((setq x (get symbol 'setf-method))
           (let ((*package* (good-package)))
             (doc1
              (format nil
                "~@[~%Defined as: ~S~%See the doc of DEFINE-SETF-METHOD.~]"
                (if (consp x)
                    (case (car x)
                          (lambda `(define-setf-method ,@(cdr x)))
                          (lambda-block `(define-setf-method ,@(cddr x)))
                          (lambda-closure `(define-setf-method ,@(cddddr x)))
                          (lambda-block-closure
                           `(define-setf-method ,@(cdr (cddddr x))))
                          (t nil))
                    nil))
            "[Setf]"))))
    )
  (idescribe (symbol-name symbol))
  (if called-from-apropos-doc-p
      f
      (progn (if f
                 (format t "~&-----------------------------------------------------------------------------")
                 (format t "~&No documentation for ~:@(~S~)." symbol))
             (values))))

(defun apropos-doc (string &optional (package 'lisp) &aux f (package (or package (list-all-packages))))
  (setq string (string string))
  (do-symbols (symbol package) ;FIXME?  do-symbols takes package list
	      (when (search string (string symbol))
		(setq f (or (print-doc symbol t) f))))
  (if f
      (format t "~&-----------------------------------------------------------------------------")
      (format t "~&No documentation for ~S in ~:[any~;~A~] package."
              string package
              (and package (package-name (coerce-to-package package)))))
  (values))

