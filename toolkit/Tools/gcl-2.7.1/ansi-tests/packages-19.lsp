;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Tue May  5 17:22:49 1998
;;;; Contains: Packages test code, part 19.  Tests of the keyword package.
;;;;           See also cl-symbols.lsp (for keywordp test cases)

(in-package :cl-test)
(declaim (optimize (safety 3)))

;; Check that each keyword satisfies keywordp

(deftest keyword.1
  (do-symbols (s "KEYWORD" t)
    (unless (keywordp s)
      (return (list s nil))))
  t)

;; Every keyword is external
(deftest keyword.2
  (do-symbols (s "KEYWORD" t)
    (multiple-value-bind (s2 access)
	(find-symbol (symbol-name s) "KEYWORD")
      (unless (and (eqt s s2)
		   (eqt access :external))
	(return (list s2 access)))))
  t)

;; Every keyword evaluates to itself
(deftest keyword.3
  (do-symbols (s "KEYWORD" t)
    (unless (eqt s (eval s))
      (return (list s (eval s)))))
  t)


;;; Other error tests

(deftest package-shadowing-symbols.error.1
  (classify-error (package-shadowing-symbols))
  program-error)

(deftest package-shadowing-symbols.error.2
  (classify-error (package-shadowing-symbols "CL" nil))
  program-error)

(deftest package-use-list.error.1
  (classify-error (package-use-list))
  program-error)

(deftest package-use-list.error.2
  (classify-error (package-use-list "CL" nil))
  program-error)

(deftest package-used-by-list.error.1
  (classify-error (package-used-by-list))
  program-error)

(deftest package-used-by-list.error.2
  (classify-error (package-used-by-list "CL" nil))
  program-error)

