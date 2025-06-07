;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Apr 25 19:20:29 1998
;;;; Contains: Package test code, part 17

(in-package :cl-test)
(declaim (optimize (safety 3)))

(deftest do-symbols.1
  (equalt
   (remove-duplicates
    (sort-symbols (let ((all nil))
		    (do-symbols (x "B" all) (push x all)))))
   (list (find-symbol "BAR" "B")
	 (find-symbol "FOO" "A")))
  t)

;;
;; Test up some test packages
;;

(defun collect-symbols (pkg)
  (remove-duplicates
   (sort-symbols
    (let ((all nil))
      (do-symbols (x pkg all) (push x all))))))

(defun collect-external-symbols (pkg)
  (remove-duplicates
   (sort-symbols
    (let ((all nil))
      (do-external-symbols (x pkg all) (push x all))))))

(deftest do-symbols.2
    (collect-symbols "DS1")
  (DS1:A DS1:B DS1::C DS1::D))

(deftest do-symbols.3
    (collect-symbols "DS2")
  (DS2:A DS2::E DS2::F DS2:G DS2:H))

(deftest do-symbols.4
  (collect-symbols "DS3")
  (DS1:A DS3:B DS2:G DS2:H DS3:I DS3:J DS3:K DS3::L DS3::M))

(deftest do-symbols.5
  (remove-duplicates
   (collect-symbols "DS4")
   :test #'(lambda (x y)
	     (and (eqt x y)
		  (not (eqt x 'DS4::B)))))
  (DS1:A DS1:B DS2::F DS3:G DS3:I DS3:J DS3:K DS4::X DS4::Y DS4::Z))


(deftest do-external-symbols.1
    (collect-external-symbols "DS1")
  (DS1:A DS1:B))

(deftest do-external-symbols.2
    (collect-external-symbols "DS2")
  (DS2:A DS2:G DS2:H))

(deftest do-external-symbols.3
    (collect-external-symbols "DS3")
  (DS1:A DS3:B DS2:G DS3:I DS3:J DS3:K))

(deftest do-external-symbols.4
    (collect-external-symbols "DS4")
  ())

(deftest do-external-symbols.5
    (equalt (collect-external-symbols "KEYWORD")
	    (collect-symbols "KEYWORD"))
  t)

;; Test that do-symbols, do-external-symbols work without
;; a return value (and that the default return value is nil)

(deftest do-symbols.6
  (do-symbols (s "DS1") (declare (ignore s)) t)
  nil)

(deftest do-external-symbols.6
  (do-external-symbols (s "DS1") (declare (ignore s)) t)
  nil)

;; Test that do-symbols, do-external-symbols work without
;; a package being specified

(deftest do-symbols.7
  (let ((x nil)
	(*package* (find-package "DS1")))
    (declare (special *package*))
    (list
     (do-symbols (s) (push s x))
     (sort-symbols x)))
  (nil (DS1:A DS1:B DS1::C DS1::D)))

(deftest do-external-symbols.7
  (let ((x nil)
	(*package* (find-package "DS1")))
    (declare (special *package*))
    (list
     (do-external-symbols (s) (push s x))
     (sort-symbols x)))
  (nil (DS1:A DS1:B)))

;; Test that the tags work in the tagbody,
;;  and that multiple statements work

(deftest do-symbols.8
  (handler-case
   (let ((x nil))
     (list
      (do-symbols
       (s "DS1")
       (when (equalt (symbol-name s) "C") (go bar))
       (push s x)
       (go foo)
       bar
       (push t x)
       foo)
      (sort-symbols x)))
   (error (c) c))
  (NIL (DS1:A DS1:B DS1::D T)))

(deftest do-external-symbols.8
  (handler-case
   (let ((x nil))
     (list
      (do-external-symbols
       (s "DS1")
       (when (equalt (symbol-name s) "A") (go bar))
       (push s x)
       (go foo)
       bar
       (push t x)
       foo)
      (sort-symbols x)))
   (error (c) c))
  (NIL (DS1:B T)))



  
