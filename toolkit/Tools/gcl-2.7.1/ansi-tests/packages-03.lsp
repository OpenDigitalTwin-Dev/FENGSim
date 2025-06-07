;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 07:51:26 1998
;;;; Contains: Package test code, part 03

(in-package :cl-test)
(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; list-all-packages

;; list-all-packages returns a list
(deftest list-all-packages.1
  (numberp (ignore-errors (list-length (list-all-packages))))
  t)

;; The required packages are present
(deftest list-all-packages.2
  (subsetp
   (list (find-package "CL")
	 (find-package "CL-USER")
	 (find-package "KEYWORD")
	 (find-package "A")
	 (find-package "RT")
	 (find-package "CL-TEST")
	 (find-package "B"))
   (list-all-packages))
  t)

;; The list returned has only packages in it
(deftest list-all-packages.3
  (notnot-mv (every #'packagep (list-all-packages)))
  t)

;; It returns a list of the same packages each time it is called
(deftest list-all-packages.4
  (let ((p1 (list-all-packages))
	(p2 (list-all-packages)))
    (and (subsetp p1 p2)
	 (subsetp p2 p1)))
  t)

(deftest list-all-packages.error.1
  (classify-error (list-all-packages nil))
  program-error)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; package-name

(deftest package-name.1
  (ignore-errors (package-name "A"))
  "A")

(deftest package-name.2
  (ignore-errors (package-name #\A))
  "A")

(deftest package-name.3
  (ignore-errors (package-name "Q"))
  "A")

(deftest package-name.4
  (ignore-errors (package-name #\Q))
  "A")

(deftest package-name.5
  (notnot-mv (member (classify-error (package-name "NOT-THERE"))
		     '(type-error package-error)))
  t)

(deftest package-name.6
  (notnot-mv (member (classify-error (package-name #\*))
		     '(type-error package-error)))
  t)

(deftest package-name.6a
  (notnot-mv (member (classify-error (locally (package-name #\*) t))
		     '(type-error package-error)))
  t)

(deftest package-name.7
  (package-name "CL")
  #.(string '#:common-lisp))

(deftest package-name.8
  (package-name "COMMON-LISP")
  #.(string '#:common-lisp))

(deftest package-name.9
  (package-name "COMMON-LISP-USER")
  #.(string '#:common-lisp-user))

(deftest package-name.10
  (package-name "CL-USER")
  #.(string '#:common-lisp-user))

(deftest package-name.11
  (package-name "KEYWORD")
  #.(string '#:keyword))

(deftest package-name.12
  (package-name (find-package "CL"))
  #.(string '#:common-lisp))

(deftest package-name.13
  (let* ((p (make-package "TEMP1"))
	 (pname1 (package-name p)))
    (rename-package "TEMP1" "TEMP2")
    (let ((pname2 (package-name p)))
      (safely-delete-package p)
      (list pname1 pname2 (package-name p))))
  ("TEMP1" "TEMP2" nil))

;; (find-package (package-name p)) == p for any package p
(deftest package-name.14
  (loop
   for p in (list-all-packages) count
   (not
    (let ((name (package-name p)))
      (and (stringp name)
	   (eqt (find-package name) p)))))
  0)

;; package-name applied to a package's name
;; should return an equal string
(deftest package-name.15
  (loop
   for p in (list-all-packages) count
   (not (equal (package-name p)
	       (ignore-errors (package-name (package-name p))))))
  0)

(deftest package-name.error.1
  (classify-error (package-name))
  program-error)

(deftest package-name.error.2
  (classify-error (package-name "CL" nil))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; package-nicknames

(deftest package-nicknames.1
  (ignore-errors (package-nicknames "A"))
  ("Q"))

(deftest package-nicknames.2
  (ignore-errors (package-nicknames #\A))
  ("Q"))

(deftest package-nicknames.3
  (ignore-errors (package-nicknames ':|A|))
  ("Q"))

(deftest package-nicknames.4
  (ignore-errors (package-nicknames "B"))
  nil)

(deftest package-nicknames.5
  (ignore-errors (package-nicknames #\B))
  nil)

(deftest package-nicknames.6
  (ignore-errors (package-nicknames '#:|B|))
  nil)

(deftest package-nicknames.7
  (ignore-errors
    (subsetp '(#.(string '#:cl))
	     (package-nicknames "COMMON-LISP")
	     :test #'string=))
  t)

(deftest package-nicknames.8
  (ignore-errors
    (notnot
     (subsetp '(#.(string '#:cl-user))
	      (package-nicknames "COMMON-LISP-USER")
	      :test #'string=)))
  t)

(deftest package-nicknames.9
  (classify-error (package-nicknames 10))
  type-error)

(deftest package-nicknames.9a
  (classify-error (locally (package-nicknames 10) t))
  type-error)

(deftest package-nicknames.10
  (ignore-errors (package-nicknames (find-package "A")))
  ("Q"))

(deftest package-nicknames.11
  (notnot-mv (member (classify-error (package-nicknames "NOT-A-PACKAGE-NAME"))
		     '(type-error package-error)))
  t)


;; (find-package n) == p for each n in (package-nicknames p),
;; for any package p
(deftest package-nicknames.12
  (loop
   for p in (list-all-packages) sum
   (loop
    for nk in (package-nicknames p) count
	 (not
	  (and (stringp nk)
	       (eqt p (find-package nk))))))
  0)

(deftest package-nicknames.error.1
  (classify-error (package-nicknames))
  program-error)

(deftest package-nicknames.error.2
  (classify-error (package-nicknames "CL" nil))
  program-error)
