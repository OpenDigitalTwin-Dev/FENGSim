;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 07:59:10 1998
;;;; Contains: Package test code, part 04

(in-package :cl-test)
(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; intern

(deftest intern.1
  (progn
    (safely-delete-package "TEMP1")
    (let ((p (make-package "TEMP1"))
	  (i 0) x y)
      (multiple-value-bind* (sym1 status1)
	  (find-symbol "FOO" p)
	(intern (progn (setf x (incf i)) "FOO")
		(progn (setf y (incf i)) p))
	(multiple-value-bind* (sym2 status2)
	    (find-symbol "FOO" p)
	  (and (eql i 2)
	       (eql x 1)
	       (eql y 2)
	       (null sym1)
	       (null status1)
	       (string= (symbol-name sym2) "FOO")
	       (eqt (symbol-package sym2) p)
	       (eqt status2 :internal)
	       (progn (delete-package p) t))))))
  t)

(deftest intern.2
  (progn
    (safely-delete-package "TEMP1")
    (let ((p (make-package "TEMP1")))
      (multiple-value-bind* (sym1 status1)
	  (find-symbol "FOO" "TEMP1")
	(intern "FOO" "TEMP1")
	(multiple-value-bind* (sym2 status2)
	    (find-symbol "FOO" p)
	  (and (null sym1)
	       (null status1)
	       (string= (symbol-name sym2) "FOO")
	       (eqt (symbol-package sym2) p)
	       (eqt status2 :internal)
	       (progn (delete-package p) t))))))
  t)

(deftest intern.error.1
  (classify-error (intern))
  program-error)

(deftest intern.error.2
  (classify-error (intern "X" "CL" nil))
  program-error)
