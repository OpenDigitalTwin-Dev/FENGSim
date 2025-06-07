;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:37:56 1998
;;;; Contains: Testing of CL Features related to "CONS", part 11

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; ldiff, tailp

(deftest ldiff.1
  (let* ((x (copy-tree '(a b c d e f)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x (cdddr x))))
      (and (check-scaffold-copy x xcopy)
	   result)))
  (a b c))

(deftest ldiff.2
  (let* ((x (copy-tree '(a b c d e f)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x 'a)))
      (and
       (check-scaffold-copy x xcopy)
       (zerop
	(loop
	 for a on x and b on result count
	 (eqt a b)))
       result)))
  (a b c d e f))

;; Works when the end of the dotted list is a symbol
(deftest ldiff.3
  (let* ((x (copy-tree '(a b c d e . f)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x 'a)))
      (and
       (check-scaffold-copy x xcopy)
       result)))
  (a b c d e . f))

;; Works when the end of the dotted list is a fixnum
(deftest ldiff.4
  (let* ((n 18)
	 (x (list* 'a 'b 'c 18))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x n)))
      (and
       (check-scaffold-copy x xcopy)
       result)))
  (a b c))

;; Works when the end of the dotted list is a larger
;; integer (that is eql, but probably not eq).
(deftest ldiff.5
  (let* ((n 18000000000000)
	 (x (list* 'a 'b 'c (1- 18000000000001)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x n)))
      (and
       (check-scaffold-copy x xcopy)
       result)))
  (a b c))

;; Test works when the end of a dotted list is a string
(deftest ldiff.6
  (let* ((n (copy-seq "abcde"))
	 (x (list* 'a 'b 'c n))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x n)))
      (if (equal result (list 'a 'b 'c))
	  (check-scaffold-copy x xcopy)
	result)))
  t)

;; Check that having the cdr of a dotted list be string-equal, but
;; not eql, does not result in success
(deftest ldiff.7
  (let* ((n (copy-seq "abcde"))
	 (x (list* 'a 'b 'c n))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (ldiff x (copy-seq n))))
      (if (equal result x)
	  (check-scaffold-copy x xcopy)
	result)))
  t)

;; Check that on failure, the list returned by ldiff is
;; a copy of the list, not the list itself.

(deftest ldiff.8
  (let ((x (list 'a 'b 'c 'd)))
    (let ((result (ldiff x '(e))))
      (and (equal x result)
	   (loop
	    for c1 on x
	    for c2 on result
	    count (eqt c1 c2)))))
  0)

(deftest ldiff.order.1
  (let ((i 0) x y)
    (values
     (ldiff (progn (setf x (incf i))
		   (list* 'a 'b 'c 'd))
	    (progn (setf y (incf i))
		   'd))
     i x y))
  (a b c) 2 1 2)       

;; Error checking

(deftest ldiff.error.1
  (classify-error (ldiff 10 'a))
  type-error)

;; Single atoms are not dotted lists, so the next
;; case should be a type-error
(deftest ldiff.error.2
  (classify-error (ldiff 'a 'a))
  type-error)

(deftest ldiff.error.3
  (classify-error (ldiff (make-array '(10) :initial-element 'a) '(a)))
  type-error)

(deftest ldiff.error.4
    (classify-error (ldiff 1.23 t))
  type-error)

(deftest ldiff.error.5
    (classify-error (ldiff #\w 'a))
  type-error)

(deftest ldiff.error.6
  (classify-error (ldiff))
  program-error)

(deftest ldiff.error.7
  (classify-error (ldiff nil))
  program-error)

(deftest ldiff.error.8
  (classify-error (ldiff nil nil nil))
  program-error)

;; Note!  The spec is ambiguous on whether this next test
;; is correct.  The spec says that ldiff should be prepared
;; to signal an error if the list argument is not a proper
;; list or dotted list.  If listp is false, the list argument
;; is neither (atoms are not dotted lists).
;;
;; However, the sample implementation *does* work even if
;; the list argument is an atom.
;;
#|
(defun ldiff-12-body ()
  (loop
   for x in *universe*
   count (and (not (listp x))
	      (not (eqt 'type-error
			(catch-type-error (ldiff x x)))))))

(deftest ldiff-12
    (ldiff-12-body)
  0)
|#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; tailp

(deftest tailp.1
  (let ((x (copy-tree '(a b c d e . f))))
    (and
     (tailp x x)
     (tailp (cdr x) x)
     (tailp (cddr x) x)
     (tailp (cdddr x) x)
     (tailp (cddddr x) x)
     t))
  t)

;; The next four tests test that tailp handles dotted lists.  See
;; TAILP-NIL:T in the X3J13 documentation.

(deftest tailp.2
  (notnot-mv (tailp 'e (copy-tree '(a b c d . e))))
  t)

(deftest tailp.3
  (tailp 'z (copy-tree '(a b c d . e)))
  nil)

(deftest tailp.4
  (notnot-mv (tailp 10203040506070
		    (list* 'a 'b (1- 10203040506071))))
  t)

(deftest tailp.5
  (let ((x "abcde")) (tailp x (list* 'a 'b (copy-seq x))))
  nil)

(deftest tailp.error.5
  (classify-error (tailp))
  program-error)

(deftest tailp.error.6
  (classify-error (tailp nil))
  program-error)

(deftest tailp.error.7
  (classify-error (tailp nil nil nil))
  program-error)

;; Test that tailp does not modify its arguments

(deftest tailp.6
    (let* ((x (copy-list '(a b c d e)))
	   (y (cddr x)))
      (let ((xcopy (make-scaffold-copy x))
	    (ycopy (make-scaffold-copy y)))
	(and
	 (tailp y x)
	 (check-scaffold-copy x xcopy)
	 (check-scaffold-copy y ycopy))))
  t)

;; Note!  The spec is ambiguous on whether this next test
;; is correct.  The spec says that tailp should be prepared
;; to signal an error if the list argument is not a proper
;; list or dotted list.  If listp is false, the list argument
;; is neither (atoms are not dotted lists).
;;
;; However, the sample implementation *does* work even if
;; the list argument is an atom.
;;

#|
(defun tailp.7-body ()
  (loop
      for x in *universe*
      count (and (not (listp x))
		 (eqt 'type-error
		     (catch-type-error (tailp x x))))))

(deftest tailp.7
    (tailp.7-body)
  0)
|#
    
(deftest tailp.order.1
  (let ((i 0) x y)
    (values
     (notnot
      (tailp (progn (setf x (incf i)) 'd)
	     (progn (setf y (incf i)) '(a b c . d))))
     i x y))
  t 2 1 2)

