;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 07:49:34 1998
;;;; Contains: Package test code, part 01

(in-package :cl-test)
(declaim (optimize (safety 3)))

;; Test find-symbol, with the various combinations of
;; package designators

(deftest find-symbol.1
  (find-symbol "aBmAchb1c")
  nil nil)

(deftest find-symbol.2
  (find-symbol "aBmAchb1c" "CL")
  nil nil)

(deftest find-symbol.3
  (find-symbol "aBmAchb1c" "COMMON-LISP")
  nil nil)

(deftest find-symbol.4
  (find-symbol "aBmAchb1c" "KEYWORD")
  nil nil)

(deftest find-symbol.5
  (find-symbol "aBmAchb1c" "COMMON-LISP-USER")
  nil nil)

(deftest find-symbol.6
  (find-symbol (string '#:car) "CL")
  car :external)

(deftest find-symbol.7
  (find-symbol (string '#:car) "COMMON-LISP")
  car :external)

(deftest find-symbol.8
  (values (find-symbol (string '#:car) "COMMON-LISP-USER"))
  car #| :inherited |# )

(deftest find-symbol.9
  (find-symbol (string '#:car) "CL-TEST")
  car :inherited)

(deftest find-symbol.10
  (find-symbol (string '#:test) "KEYWORD")
  :test :external)

(deftest find-symbol.11
  (find-symbol (string '#:find-symbol.11) "CL-TEST")
  find-symbol.11 :internal)

(deftest find-symbol.12
  (find-symbol "FOO" #\A)
  A::FOO :external)

(deftest find-symbol.13
  (progn
    (intern "X" (find-package "A"))
    (find-symbol "X" #\A))
  A::X :internal)

(deftest find-symbol.14
  (find-symbol "FOO" #\B)
  A::FOO :inherited)

(deftest find-symbol.15
  (find-symbol "FOO" "B")
  A::FOO :inherited)

(deftest find-symbol.16
  (find-symbol "FOO" (find-package "B"))
  A::FOO :inherited)

(deftest find-symbol.order.1
  (let ((i 0) x y)
    (values
     (find-symbol (progn (setf x (incf i)) (string '#:car))
		  (progn (setf y (incf i)) "COMMON-LISP"))
     i x y))
  car 2 1 2)

(deftest find-symbol.error.1
  (classify-error (find-symbol))
  program-error)

(deftest find-symbol.error.2
  (classify-error (find-symbol "CAR" "CL" nil))
  program-error)