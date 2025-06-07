;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 10:23:31 1998
;;;; Contains: Testing of CL Features related to "CONS", part 18

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; get-properties

(deftest get-properties.1
  (get-properties nil nil)
  nil nil nil)

(deftest get-properties.2
  (get-properties '(a b) nil)
  nil nil nil)

(deftest get-properties.3
  (get-properties '(a b c d) '(a))
  a b (a b c d))

(deftest get-properties.4
  (get-properties '(a b c d) '(c))
  c d (c d))

(deftest get-properties.5
  (get-properties '(a b c d) '(c a))
  a b (a b c d))

(deftest get-properties.6
  (get-properties '(a b c d) '(b))
  nil nil nil)

(deftest get-properties.7
  (get-properties '("aa" b c d) (list (copy-seq "aa")))
  nil nil nil)

(deftest get-properties.8
  (get-properties '(1000000000000 b c d) (list (1+ 999999999999)))
  nil nil nil)

(deftest get-properties.9
  (let* ((x (copy-list '(a b c d e f g h a c)))
	 (xcopy (make-scaffold-copy x))
	 (y (copy-list '(x y f g)))
	 (ycopy (make-scaffold-copy y)))
    (multiple-value-bind
	(indicator value tail)
	(get-properties x y)
      (and
       (check-scaffold-copy x xcopy)
       (check-scaffold-copy y ycopy)
       (eqt tail (nthcdr 6 x))
       (values indicator value tail))))
  g h (g h a c))

(deftest get-properties.order.1
  (let ((i 0) x y)
    (values
     (multiple-value-list
      (get-properties (progn (setf x (incf i)) '(a b c d))
		      (progn (setf y (incf i)) '(c))))
     i x y))
  (c d (c d)) 2 1 2)

(deftest get-properties.error.1
  (classify-error (get-properties))
  program-error)

(deftest get-properties.error.2
  (classify-error (get-properties nil))
  program-error)

(deftest get-properties.error.3
  (classify-error (get-properties nil nil nil))
  program-error)

	   
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; getf

(deftest getf.1
  (getf nil 'a)
  nil)

(deftest getf.2
  (getf nil 'a 'b)
  b)

(deftest getf.3
  (getf '(a b) 'a)
  b)

(deftest getf.4
  (getf '(a b) 'a 'c)
  b)

(deftest getf.5
  (let ((x 0))
    (values
     (getf '(a b) 'a (incf x))
     x))
  b 1)

(deftest getf.order.1
  (let ((i 0) x y)
    (values
     (getf (progn (setf x (incf i)) '(a b))
	   (progn (setf y (incf i)) 'a))
     i x y))
  b 2 1 2)
		  
(deftest getf.order.2
  (let ((i 0) x y z)
    (values
     (getf (progn (setf x (incf i)) '(a b))
	   (progn (setf y (incf i)) 'a)
	   (setf z (incf i)))
     i x y z))
  b 3 1 2 3)		  

(deftest setf-getf.1
  (let ((p (copy-list '(a 1 b 2))))
    (setf (getf p 'c) 3)
    ;; Must check that only a, b, c have properties
    (and
     (eqlt (getf p 'a) 1)
     (eqlt (getf p 'b) 2)
     (eqlt (getf p 'c) 3)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b c))))
      0)
     t))
  t)

(deftest setf-getf.2
  (let ((p (copy-list '(a 1 b 2))))
    (setf (getf p 'a) 3)
    ;; Must check that only a, b have properties
    (and
     (eqlt (getf p 'a) 3)
     (eqlt (getf p 'b) 2)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b))))
      0)
     t))
  t)    

(deftest setf-getf.3
  (let ((p (copy-list '(a 1 b 2))))
    (setf (getf p 'c 17) 3)
    ;; Must check that only a, b, c have properties
    (and
     (eqlt (getf p 'a) 1)
     (eqlt (getf p 'b) 2)
     (eqlt (getf p 'c) 3)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b c))))
      0)
     t))
  t)

(deftest setf-getf.4
  (let ((p (copy-list '(a 1 b 2))))
    (setf (getf p 'a 17) 3)
    ;; Must check that only a, b have properties
    (and
     (eqlt (getf p 'a) 3)
     (eqlt (getf p 'b) 2)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b))))
      0)
     t))
  t)

(deftest setf-getf.5
  (let ((p (copy-list '(a 1 b 2)))
	(foo nil))
    (setf (getf p 'a (progn (setf foo t) 0)) 3)
    ;; Must check that only a, b have properties
    (and
     (eqlt (getf p 'a) 3)
     (eqlt (getf p 'b) 2)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b))))
      0)
     foo))
  t)

(deftest setf-getf.order.1
  (let ((p (list (copy-list '(a 1 b 2))))
	(cnt1 0) (cnt2 0) (cnt3 0))
    (setf (getf (car (progn (incf cnt1) p)) 'c (incf cnt3))
	  (progn (incf cnt2) 3))
    ;; Must check that only a, b, c have properties
    (and
     (eqlt cnt1 1)
     (eqlt cnt2 1)
     (eqlt cnt3 1)
     (eqlt (getf (car p) 'a) 1)
     (eqlt (getf (car p) 'b) 2)
     (eqlt (getf (car p) 'c) 3)
     (eqlt
      (loop
       for ptr on (car p) by #'cddr count
       (not (member (car ptr) '(a b c))))
      0)
     t))
  t)

(deftest setf-getf.order.2
  (let ((p (list (copy-list '(a 1 b 2))))
	(i 0) x y z w)
    (setf (getf (car (progn (setf x (incf i)) p))
		(progn (setf y (incf i)) 'c)
		(setf z (incf i)))
	  (progn (setf w (incf i)) 3))
    ;; Must check that only a, b, c have properties
    (and
     (eqlt i 4)
     (eqlt x 1)
     (eqlt y 2)
     (eqlt z 3)
     (eqlt w 4)
     (eqlt (getf (car p) 'a) 1)
     (eqlt (getf (car p) 'b) 2)
     (eqlt (getf (car p) 'c) 3)
     (eqlt
      (loop
       for ptr on (car p) by #'cddr count
       (not (member (car ptr) '(a b c))))
      0)
     t))
  t)

(deftest incf-getf.1
  (let ((p (copy-list '(a 1 b 2))))
    (incf (getf p 'b))
    ;; Must check that only a, b have properties
    (and
     (eqlt (getf p 'a) 1)
     (eqlt (getf p 'b) 3)
     (eqlt
      (loop
       for ptr on p by #'cddr count
       (not (member (car ptr) '(a b))))
      0)
     t))
  t)

(deftest incf-getf.2
  (let ((p (copy-list '(a 1 b 2))))
    (incf (getf p 'c 19))
    ;; Must check that only a, b have properties
    (and
     (eqlt (getf p 'a) 1)
     (eqlt (getf p 'b) 2)
     (eqlt (getf p 'c) 20)
     (eqlt
	(loop
	 for ptr on p by #'cddr count
	 (not (member (car ptr) '(a b c))))
	0)
     t))
  t)

(deftest push-getf.1
  (let ((p nil))
    (values
     (push 'x (getf p 'a))
     p))
  (x) (a (x)))

(deftest getf.error.1
  (classify-error (getf))
  program-error)

(deftest getf.error.2
  (classify-error (getf nil))
  program-error)

(deftest getf.error.3
  (classify-error (getf nil nil nil nil))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; remf

(deftest remf.1
  (let ((x nil))
    (values (remf x 'a) x))
  nil ())

(deftest remf.2
  (let ((x (list 'a 'b)))
    (values (not (null (remf x 'a))) x))
  t ())

(deftest remf.3
  (let ((x (list 'a 'b 'a 'c)))
    (values (not (null (remf x 'a))) x))
  t (a c))

(deftest remf.4
  (let ((x (list 'a 'b 'c 'd)))
    (values
     (and (remf x 'c) t)
     (loop
      for ptr on x by #'cddr count
      (not (eqt (car ptr) 'a)))))
  t 0)

(deftest remf.order.1
  (let ((i 0) x y
	(p (make-array 1 :initial-element (copy-list '(a b c d e f)))))
    (values
     (notnot
      (remf (aref p (progn (setf x (incf i)) 0))
	    (progn (setf y (incf i))
		   'c)))
     (aref p 0)
     i x y))
  t (a b e f) 2 1 2)

  