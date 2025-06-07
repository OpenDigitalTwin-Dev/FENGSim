;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 08:07:31 1998
;;;; Contains: Package test code, part 18

(in-package :cl-test)
(declaim (optimize (safety 3)))

(declaim (special *universe*))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; packagep, typep * 'package

(deftest packagep.1
  (loop
   for x in *universe* count
   (unless (eqt (not (packagep x))
		(not (typep x 'package)))
	   (format t
		   "(packagep ~S) = ~S, (typep x 'package) = ~S~%"
		   x (packagep x) x (typep x 'package))
	   t))
  0)

;;; *package* is always a package

(deftest packagep.2
  (not-mv (packagep *package*))
  nil)

(deftest packagep.error.1
  (classify-error (packagep))
  program-error)

(deftest packagep.error.2
  (classify-error (packagep nil nil))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; package-error

(deftest package-error.1
  (not
   (typep (make-condition 'package-error :package "CL")
	  'package-error))
  nil)

(deftest package-error.2
  (not
   (typep (make-condition 'package-error
			  :package (find-package "CL"))
	  'package-error))
  nil)

(deftest package-error.3
  (subtypep* 'package-error 'error)
  t t)

(deftest package-error.4
   (not
    (typep (make-condition 'package-error
			   :package (find-package '#:|CL|))
	   'package-error))
  nil)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; package-error-package

(deftest package-error-package.1
  (eqt (find-package (package-error-package
		      (make-condition 'package-error
				      :package "CL")))
       (find-package "CL"))
  t)

(deftest package-error-package.2
  (eqt (find-package (package-error-package
		      (make-condition 'package-error
				      :package (find-package "CL"))))
       (find-package "CL"))
  t)

(deftest package-error-package.3
  (eqt (find-package (package-error-package
		      (make-condition 'package-error
				      :package '#:|CL|)))
       (find-package "CL"))
  t)

(deftest package-error-package.4
  (eqt (find-package (package-error-package
		      (make-condition 'package-error
				      :package #\A)))
       (find-package "A"))
  t)

(deftest package-error-package.error.1
  (classify-error (package-error-package))
  program-error)

(deftest package-error-package.error.2
  (classify-error
   (package-error-package
    (make-condition 'package-error :package #\A)
    nil))
  program-error)
