;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:35:15 1998
;;;; Contains: Testing of CL Features related to "CONS", part 7

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; nconc

(deftest nconc.1
  (nconc)
  nil)

(deftest nconc.2
  (nconc (copy-tree '(a b c d e f)))
  (a b c d e f))

(deftest nconc.3
  (nconc 1)
  1)

(deftest nconc.4
  (let ((x (list 'a 'b 'c))
	(y (list 'd 'e 'f)))
    (let ((ycopy (make-scaffold-copy y)))
      (let ((result (nconc x y)))
	(and
	 (check-scaffold-copy y ycopy)
	 (eqt (cdddr x) y)
	 result))))
  (a b c d e f))

(deftest nconc.5
  (let ((x (list 'a 'b 'c)))
    (nconc x x)
    (and
     (eqt (cdddr x) x)
     (null (list-length x))))
  t)

(deftest nconc.6
  (let ((x (list 'a 'b 'c))
	(y (list 'd 'e 'f 'g 'h))
	(z (list 'i 'j 'k)))
    (let ((result (nconc x y z 'foo)))
      (and
       (eqt (nthcdr 3 x) y)
       (eqt (nthcdr 5 y) z)
       (eqt (nthcdr 3 z) 'foo)
       result)))
  (a b c d e f g h i j k . foo))

(deftest nconc.7
  (nconc (copy-tree '(a . b))
	 (copy-tree '(c . d))
	 (copy-tree '(e . f))
	 'foo)
  (a c e . foo))

(deftest nconc.order.1
  (let ((i 0) x y z)
    (values
     (nconc (progn (setf x (incf i)) (copy-list '(a b c)))
	    (progn (setf y (incf i)) (copy-list '(d e f)))
	    (progn (setf z (incf i)) (copy-list '(g h i))))
     i x y z))
  (a b c d e f g h i) 3 1 2 3)

(deftest nconc.order.2
  (let ((i 0))
    (values
     (nconc (incf i))
     i))
  1 1)	    

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; append

(deftest append.1
  (append)
  nil)

(deftest append.2
  (append 'x)
  x)

(deftest append.3
  (let ((x (list 'a 'b 'c 'd))
	(y (list 'e 'f 'g)))
    (let ((xcopy (make-scaffold-copy x))
	  (ycopy (make-scaffold-copy y)))
      (let ((result (append x y)))
	(and
	 (check-scaffold-copy x xcopy)
	 (check-scaffold-copy y ycopy)
	 result))))
  (a b c d e f g))

(deftest append.4
  (append (list 'a) (list 'b) (list 'c)
	  (list 'd) (list 'e) (list 'f)
	  (list 'g) 'h)
  (a b c d e f g . h))

(deftest append.5
  (append nil nil nil nil nil nil nil nil 'a)
  a)

(deftest append.6
  (append-6-body)
  0)

(deftest append.order.1
  (let ((i 0) x y z)
    (values
     (append (progn (setf x (incf i)) (copy-list '(a b c)))
	     (progn (setf y (incf i)) (copy-list '(d e f)))
	     (progn (setf z (incf i)) (copy-list '(g h i))))
     i x y z))
  (a b c d e f g h i) 3 1 2 3)

(deftest append.order.2
  (let ((i 0)) (values (append (incf i)) i))
  1 1)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; revappend

(deftest revappend.1
    (let* ((x (list 'a 'b 'c))
	   (y (list 'd 'e 'f))
	   (xcopy (make-scaffold-copy x))
	   (ycopy (make-scaffold-copy y))
	   )
      (let ((result (revappend x y)))
	(and
	 (check-scaffold-copy x xcopy)
	 (check-scaffold-copy y ycopy)
	 (eqt (cdddr result) y)
	 result)))
  (c b a d e f))

(deftest revappend.2
    (revappend (copy-tree '(a b c d e)) 10)
  (e d c b a . 10))

(deftest revappend.3
    (revappend nil 'a)
  a)

(deftest revappend.4
    (revappend (copy-tree '(a (b c) d)) nil)
  (d (b c) a))

(deftest revappend.order.1
  (let ((i 0) x y)
    (values
     (revappend (progn (setf x (incf i)) (copy-list '(a b c)))
		(progn (setf y (incf i)) (copy-list '(d e f))))
     i x y))
  (c b a d e f) 2 1 2)

(deftest revappend.error.1
  (classify-error (revappend))
  program-error)

(deftest revappend.error.2
  (classify-error (revappend nil))
  program-error)

(deftest revappend.error.3
  (classify-error (revappend nil nil nil))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; nreconc

(deftest nreconc.1
  (let* ((x (list 'a 'b 'c))
	 (y (copy-tree '(d e f)))
	 (result (nreconc x y)))
    (and (equal y '(d e f))
	 result))
  (c b a d e f))

(deftest nreconc.2
  (nreconc nil 'a)
  a)

(deftest nreconc.order.1
  (let ((i 0) x y)
    (values
     (nreconc (progn (setf x (incf i)) (copy-list '(a b c)))
	      (progn (setf y (incf i)) (copy-list '(d e f))))
     i x y))
  (c b a d e f) 2 1 2)

(deftest nreconc.error.1
  (classify-error (nreconc))
  program-error)

(deftest nreconc.error.2
  (classify-error (nreconc nil))
  program-error)

(deftest nreconc.error.3
  (classify-error (nreconc nil nil nil))
  program-error)
