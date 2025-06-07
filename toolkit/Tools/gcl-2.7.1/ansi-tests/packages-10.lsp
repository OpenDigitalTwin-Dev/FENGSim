;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 08:03:36 1998
;;;; Contains: Package test code, part 10

(in-package :cl-test)
(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; with-package-iterator

(deftest with-package-iterator.1
  (with-package-iterator-internal (list (find-package "COMMON-LISP-USER")))
  t)

(deftest with-package-iterator.2
  (with-package-iterator-external (list (find-package "COMMON-LISP-USER")))
  t)

(deftest with-package-iterator.3
  (with-package-iterator-inherited (list (find-package "COMMON-LISP-USER")))
  t)

(deftest with-package-iterator.4
  (with-package-iterator-all (list (find-package "COMMON-LISP-USER")))
  t)

;;; Should test on some packages containing shadowed symbols,
;;; multiple inheritance

(deftest with-package-iterator.5
  (with-package-iterator-all '("A"))
  t)

(deftest with-package-iterator.6
  (with-package-iterator-all '(#:|A|))
  t)

(deftest with-package-iterator.7
  (with-package-iterator-all '(#\A))
  t)

(deftest with-package-iterator.8
  (with-package-iterator-internal (list (find-package "A")))
  t)

(deftest with-package-iterator.9
  (with-package-iterator-external (list (find-package "A")))
  t)

(deftest with-package-iterator.10
  (with-package-iterator-inherited (list (find-package "A")))
  t)

;;; Check that if no access symbols are provided, a program error is
;;; raised
#|
(deftest with-package-iterator.11
    (handler-case
	(progn
	  (test-with-package-iterator (list (find-package "COMMON-LISP-USER")))
	  nil)
      (program-error () t)
      (error (c) c))
  t)
|#

;;; Paul Werkowski" <pw@snoopy.mv.com> pointed out that
;;; that test is broken.  Here's a version of the replacement
;;; he suggested.
;;
;;; I'm not sure if this is correct either; it depends on
;;; whether with-package-iterator should signal the error
;;; at macro expansion time or at run time.
;;
;;; PFD 01-18-03:  I should rewrite this to use CLASSIFY-ERROR, which
;;;  uses EVAL to avoid that problem.

(deftest with-package-iterator.11
  (handler-case (macroexpand-1
		 '(with-package-iterator (x "COMMON-LISP-USER")))
		(program-error () t)
		(error (c) c))
  t)

;;; Apply to all packages
(deftest with-package-iterator.12
  (loop
   for p in (list-all-packages) count
   (handler-case
    (progn
      (format t "Package ~S~%" p)
      (not (with-package-iterator-internal (list p))))
    (error (c)
	   (format "Error ~S on package ~A~%" c p)
	   t)))
  0)

(deftest with-package-iterator.13
  (loop
   for p in (list-all-packages) count
   (handler-case
    (progn
      (format t "Package ~S~%" p)
      (not (with-package-iterator-external (list p))))
    (error (c)
	   (format "Error ~S on package ~A~%" c p)
	   t)))
  0)

(deftest with-package-iterator.14
  (loop
   for p in (list-all-packages) count
   (handler-case
    (progn
      (format t "Package ~S~%" p)
      (not (with-package-iterator-inherited (list p))))
    (error (c)
	   (format t "Error ~S on package ~S~%" c p)
	   t)))
  0)
