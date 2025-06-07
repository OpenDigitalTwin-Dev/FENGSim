;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sun Apr  5 22:26:59 1998
;;;; Contains: Testing of CL Features related to "CONS", part 25

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; setting of C*R accessors

(loop
    for fn in '(car cdr caar cadr cdar cddr
		caaar caadr cadar caddr cdaar cdadr cddar cdddr
		caaaar caaadr caadar caaddr cadaar cadadr caddar cadddr
		cdaaar cdaadr cdadar cdaddr cddaar cddadr cdddar cddddr)
    do
      (let ((level (- (length (symbol-name fn)) 2)))
	(eval `(deftest ,(intern
			  (concatenate 'string
			    (symbol-name fn)
			    "-SET-ALT")
			  :cl-test)
		   (let ((x (create-c*r-test ,level)))
		     (and
		      (setf (,fn x) 'a)
		      (eql (,fn x) 'a)
		      (setf (,fn x) 'none)
		      (equal x (create-c*r-test ,level))
		      ))
		 t))))

(loop
    for (fn len) in '((first 1) (second 2) (third 3) (fourth 4)
		      (fifth 5) (sixth 6) (seventh 7) (eighth 8)
		      (ninth 9) (tenth 10))
    do
      (eval
       `(deftest ,(intern
		   (concatenate 'string
		     (symbol-name fn)
		     "-SET-ALT")
		   :cl-test)
	    (let ((x (make-list 20 :initial-element nil)))
	      (and
	       (setf (,fn x) 'a)
	       (loop
		   for i from 1 to 20
		   do (when (and (not (eql i ,len))
				 (nth (1- i) x))
			(return nil))
		   finally (return t))
	       (eql (,fn x) 'a)
	       (nth ,(1- len) x)))
	  a)))
