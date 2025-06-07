;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 08:08:41 1998
;;;; Contains: Package test code, part 15

(in-package :cl-test)
(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; use-package

(deftest use-package.1
  (progn
    (safely-delete-package "H")
    (safely-delete-package "G")
    (let* ((pg (make-package "G" :use nil))
	   (ph (make-package "H" :use nil))
	   (sym1 (intern "FOO" pg))
	   (i 0) x y)
      (and
       (eqt (export sym1 pg) t)
       (null (package-used-by-list pg))
       (null (package-used-by-list ph))
       (null (package-use-list pg))
       (null (package-use-list ph))
       (eqt (use-package (progn (setf x (incf i)) pg)
			 (progn (setf y (incf i)) ph))
	    t)  ;; "H" will use "G"
       (eql i 2) (eql x 1) (eql y 2)
       (multiple-value-bind (sym2 access)
	   (find-symbol "FOO" ph)
	 (and
	  (eqt access :inherited)
	  (eqt sym1 sym2)))
       (equal (package-use-list ph) (list pg))
       (equal (package-used-by-list pg) (list ph))
       (null (package-use-list pg))
       (null (package-used-by-list ph))
       (eqt (unuse-package pg ph) t)
       (null (find-symbol "FOO" ph)))))
  t)

(deftest use-package.2
  (progn
    (safely-delete-package "H")
    (safely-delete-package "G")
    (let* ((pg (make-package "G" :use nil))
	   (ph (make-package "H" :use nil))
	   (sym1 (intern "FOO" pg)))
      (and
       (eqt (export sym1 pg) t)
       (null (package-used-by-list pg))
       (null (package-used-by-list ph))
       (null (package-use-list pg))
       (null (package-use-list ph))
       (eqt (use-package "G" "H") t)  ;; "H" will use "G"
       (multiple-value-bind (sym2 access)
	   (find-symbol "FOO" ph)
	 (and
	  (eqt access :inherited)
	  (eqt sym1 sym2)))
       (equal (package-use-list ph) (list pg))
       (equal (package-used-by-list pg) (list ph))
       (null (package-use-list pg))
       (null (package-used-by-list ph))
       (eqt (unuse-package pg ph) t)
       (null (find-symbol "FOO" ph)))))
  t)

(deftest use-package.3
  (progn
    (safely-delete-package "H")
    (safely-delete-package "G")
    (let* ((pg (make-package "G" :use nil))
	   (ph (make-package "H" :use nil))
	   (sym1 (intern "FOO" pg)))
      (and
       (eqt (export sym1 pg) t)
       (null (package-used-by-list pg))
       (null (package-used-by-list ph))
       (null (package-use-list pg))
       (null (package-use-list ph))
       (eqt (use-package '#:|G| '#:|H|) t)  ;; "H" will use "G"
       (multiple-value-bind (sym2 access)
	   (find-symbol "FOO" ph)
	 (and
	  (eqt access :inherited)
	  (eqt sym1 sym2)))
       (equal (package-use-list ph) (list pg))
       (equal (package-used-by-list pg) (list ph))
       (null (package-use-list pg))
       (null (package-used-by-list ph))
       (eqt (unuse-package pg ph) t)
       (null (find-symbol "FOO" ph)))))
  t)

(deftest use-package.4
  (progn
    (safely-delete-package "H")
    (safely-delete-package "G")
    (let* ((pg (make-package "G" :use nil))
	   (ph (make-package "H" :use nil))
	   (sym1 (intern "FOO" pg)))
      (and
       (eqt (export sym1 pg) t)
       (null (package-used-by-list pg))
       (null (package-used-by-list ph))
       (null (package-use-list pg))
       (null (package-use-list ph))
       (eqt (ignore-errors (use-package #\G #\H))
	    t)  ;; "H" will use "G"
       (multiple-value-bind (sym2 access)
	   (find-symbol "FOO" ph)
	 (and
	  (eqt access :inherited)
	  (eqt sym1 sym2)))
       (equal (package-use-list ph) (list pg))
       (equal (package-used-by-list pg) (list ph))
       (null (package-use-list pg))
       (null (package-used-by-list ph))
       (eqt (unuse-package pg ph) t)
       (null (find-symbol "FOO" ph)))))
  t)

;; use lists of packages

(deftest use-package.5
  (let ((pkgs '("H" "G1" "G2" "G3"))
	(vars '("FOO1" "FOO2" "FOO3")))
    (dolist (p pkgs)
      (safely-delete-package p)
      (make-package p :use nil))
    (and
     (every (complement #'package-use-list) pkgs)
     (every (complement #'package-used-by-list) pkgs)
     (every #'(lambda (v p)
		(export (intern v p) p))
	    vars (cdr pkgs))
     (progn
       (dolist (p (cdr pkgs)) (intern "MINE" p))
       (eqt (use-package (cdr pkgs) (car pkgs)) t))
     (every #'(lambda (v p)
		(eqt (find-symbol v p)
		     (find-symbol v (car pkgs))))
	    vars (cdr pkgs))
     (null (find-symbol "MINE" (car pkgs)))
     (every #'(lambda (p)
		(equal (package-used-by-list p)
		       (list (find-package (car pkgs)))))
	    (cdr pkgs))
     (equal (sort-package-list (package-use-list (car pkgs)))
	    (mapcar #'find-package (cdr pkgs)))
     (every (complement #'package-use-list) (cdr pkgs))
     (null (package-used-by-list (car pkgs)))))
  t)

;; Circular package use

(deftest use-package.6
  (progn
    (safely-delete-package "H")
    (safely-delete-package "G")
    (let ((pg (make-package "G"))
	  (ph (make-package "H"))
	  sym1 sym2 sym3 sym4
	  a1 a2 a3 a4)
      (prog1
	  (and
	   (export (intern "X" pg) pg)
	   (export (intern "Y" ph) ph)
	   (use-package pg ph)
	   (use-package ph pg)
	   (progn
	     (multiple-value-setq
		 (sym1 a1) (find-symbol "X" pg))
	     (multiple-value-setq
		 (sym2 a2) (find-symbol "Y" ph))
	     (multiple-value-setq
		 (sym3 a3) (find-symbol "Y" pg))
	     (multiple-value-setq
		 (sym4 a4) (find-symbol "X" ph))
	     (and
	      (eqt a1 :external)
	      (eqt a2 :external)
	      (eqt a3 :inherited)
	      (eqt a4 :inherited)
	      (eqt sym1 sym4)
	      (eqt sym2 sym3)
	      (eqt (symbol-package sym1) pg)
	      (eqt (symbol-package sym2) ph)
	      (unuse-package pg ph)
	      (unuse-package ph pg))))
	(safely-delete-package pg)
	(safely-delete-package ph))))
  t)

;; Also: need to check that *PACKAGE* is used as a default

(deftest use-package.error.1
  (classify-error (use-package))
  program-error)

(deftest use-package.error.2
  (progn
    (safely-delete-package "UPE2A")
    (safely-delete-package "UPE2")
    (make-package "UPE2" :use ())
    (make-package "UPE2A" :use ())
    (classify-error (use-package "UPE2" "UPE2A" nil)))
  program-error)
