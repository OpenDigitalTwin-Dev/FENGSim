;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 08:04:19 1998
;;;; Contains: Package test code, part 11

(in-package :cl-test)
(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; unexport

(deftest unexport.1
  (progn
    (safely-delete-package "X")
    (let* ((p (make-package "X" :use nil))
	   (r (export (intern "X" p) p))
	   (i 0) x y)
      (multiple-value-bind*
       (sym1 access1)
       (find-symbol "X" p)
       (unexport (progn (setf x (incf i)) sym1)
		 (progn (setf y (incf i)) p))
       (multiple-value-bind*
	(sym2 access2)
	(find-symbol "X" p)
	(and (eqt r t)
	     (eql i 2) (eql x 1) (eql y 2)
	     (eqt sym1 sym2)
	     (eqt access1 :external)
	     (eqt access2 :internal)
	     (equal (symbol-name sym1) "X")
	     t)))))
  t)
	
(deftest unexport.2
  (progn
    (safely-delete-package "X")
    (let* ((p (make-package "X" :use nil))
	   (r (export (intern "X" p) p)))
      (multiple-value-bind*
       (sym1 access1)
       (find-symbol "X" p)
       (unexport (list sym1) "X")
       (multiple-value-bind*
	(sym2 access2)
	(find-symbol "X" p)
	(and (eqt sym1 sym2)
	     (eqt r t)
	     (eqt access1 :external)
	     (eqt access2 :internal)
	     (equal (symbol-name sym1) "X")
	     t)))))
  t)

(deftest unexport.3
  (progn
    (safely-delete-package "X")
    (let* ((p (make-package "X" :use nil))
	   (r1 (export (intern "X" p) p))
	   (r2 (export (intern "Y" p) p)))
      (multiple-value-bind*
       (sym1 access1)
       (find-symbol "X" p)
       (multiple-value-bind*
	(sym1a access1a)
	(find-symbol "Y" p)
	(unexport (list sym1 sym1a) '#:|X|)
	(multiple-value-bind*
	 (sym2 access2)
	 (find-symbol "X" p)
	 (multiple-value-bind*
	  (sym2a access2a)
	  (find-symbol "Y" p)
	  (and (eqt sym1 sym2)
	       (eqt sym1a sym2a)
	       (eqt r1 t)
	       (eqt r2 t)
	       (eqt access1 :external)
	       (eqt access2 :internal)
	       (eqt access1a :external)
	       (eqt access2a :internal)
	       (equal (symbol-name sym1) "X")
	       (equal (symbol-name sym1a) "Y")
	       t)))))))
  t)

(deftest unexport.4
  (progn
    (safely-delete-package "X")
    (let* ((p (make-package "X" :use nil))
	   (r (export (intern "X" p) p)))
      (multiple-value-bind*
       (sym1 access1)
       (find-symbol "X" p)
       (unexport (list sym1) #\X)
       (multiple-value-bind*
	(sym2 access2)
	(find-symbol "X" p)
	(and (eqt sym1 sym2)
	     (eqt r t)
	     (eqt access1 :external)
	     (eqt access2 :internal)
	     (equal (symbol-name sym1) "X")
	     t)))))
  t)

;; Check that it signals a package error when unexporting
;;  an inaccessible symbol

(deftest unexport.5
  (classify-error
   (progn
     (when (find-package "X") (delete-package "X"))
     (unexport 'a (make-package "X" :use nil))
     nil))
  package-error)

;; Check that internal symbols are left alone

(deftest unexport.6
  (progn
    (when (find-package "X") (delete-package "X"))
    (let ((p (make-package "X" :use nil)))
      (let* ((sym (intern "FOO" p))
	     (r (unexport sym p)))
	(multiple-value-bind*
	 (sym2 access)
	 (find-symbol "FOO" p)
	 (and (eqt r t)
	      (eqt access :internal)
	      (eqt sym sym2)
	      (equal (symbol-name sym) "FOO")
	      t)))))
  t)

(deftest unexport.error.1
  (classify-error (unexport))
  program-error)

(deftest unexport.error.2
  (classify-error (unexport 'xyz "CL-TEST" nil))
  program-error)

