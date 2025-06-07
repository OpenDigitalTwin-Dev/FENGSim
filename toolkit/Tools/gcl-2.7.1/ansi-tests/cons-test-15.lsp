;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:40:12 1998
;;;; Contains: Testing of CL Features related to "CONS", part 15

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; mapc

(deftest mapc.1
  (mapc #'list nil)
  nil)

(deftest mapc.2
  (let ((x 0))
    (let ((result
	   (mapc #'(lambda (y) (incf x y))
		 '(1 2 3 4))))
      (list result x)))
  ((1 2 3 4) 10))

(deftest mapc.3
  (let ((x 0))
    (list
     (mapc #'(lambda (y z) (declare (ignore y z)) (incf x))
	   (make-list 5 :initial-element 'a)
	   (make-list 5 ))
     x))
  ((a a a a a) 5))

(deftest mapc.4
  (let ((x 0))
    (list
     (mapc #'(lambda (y z) (declare (ignore y z)) (incf x))
	   (make-list 5 :initial-element 'a)
	   (make-list 10))
     x))
  ((a a a a a) 5))

(deftest mapc.5
  (let ((x 0))
    (list
     (mapc #'(lambda (y z) (declare (ignore y z)) (incf x))
	   (make-list 5 :initial-element 'a)
	   (make-list 3))
     x))
  ((a a a a a) 3))

(defvar *mapc.6-var* nil)
(defun mapc.6-fun (x)
  (push x *mapc.6-var*)
  x)

(deftest mapc.6
  (let* ((x (copy-list '(a b c d e f g h)))
	 (xcopy (make-scaffold-copy x)))
    (setf *mapc.6-var* nil)
    (let ((result (mapc 'mapc.6-fun x)))
      (and (check-scaffold-copy x xcopy)
	   (eqt result x)
	   *mapc.6-var*)))
  (h g f e d c b a))

(deftest mapc.order.1
  (let ((i 0) x y z)
    (values
     (mapc (progn (setf x (incf i))
		  #'list)
	   (progn (setf y (incf i))
		  '(a b c))
	   (progn (setf z (incf i))
		  '(1 2 3)))
     i x y z))
  (a b c) 3 1 2 3)

(deftest mapc.error.1
  (classify-error (mapc #'identity 1))
  type-error)

(deftest mapc.error.2
  (classify-error (mapc))
  program-error)

(deftest mapc.error.3
  (classify-error (mapc #'append))
  program-error)

(deftest mapc.error.4
  (classify-error (locally (mapc #'identity 1) t))
  type-error)

(deftest mapc.error.5
  (classify-error (mapc #'cons '(a b c)))
  program-error)

(deftest mapc.error.6
  (classify-error (mapc #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)

(deftest mapc.error.7
  (classify-error (mapc #'car '(a b c)))
  type-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; mapcar

(deftest mapcar.1
  (mapcar #'1+ nil)
  nil)

(deftest mapcar.2
  (let* ((x (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result (mapcar #'1+ x)))
      (and (check-scaffold-copy x xcopy)
	   result)))
  (2 3 4 5))

(deftest mapcar.3
  (let* ((n 0)
	 (x (copy-list '(a b c d)))
	 (xcopy (make-scaffold-copy x)))
    (let ((result
	   (mapcar #'(lambda (y) (declare (ignore y)) (incf n))
		   x)))
      (and (check-scaffold-copy x xcopy)
	   result)))
  (1 2 3 4))

(deftest mapcar.4
  (let* ((n 0)
	 (x (copy-list '(a b c d)))
	 (xcopy (make-scaffold-copy x))
	 (x2 (copy-list '(a b c d e f)))
	 (x2copy (make-scaffold-copy x2))
	 (result
	  (mapcar #'(lambda (y z) (declare (ignore y z)) (incf n))
		  x x2)))
    (and (check-scaffold-copy x xcopy)
	 (check-scaffold-copy x2 x2copy)
	 (list result n)))
  ((1 2 3 4) 4))
  
(deftest mapcar.5
  (let* ((n 0)
	 (x (copy-list '(a b c d)))
	 (xcopy (make-scaffold-copy x))
	 (x2 (copy-list '(a b c d e f)))
	 (x2copy (make-scaffold-copy x2))
	 (result
	  (mapcar #'(lambda (y z) (declare (ignore y z)) (incf n))
		  x2 x)))
    (and (check-scaffold-copy x xcopy)
	 (check-scaffold-copy x2 x2copy)
	 (list result n)))
  ((1 2 3 4) 4))

(deftest mapcar.6
 (let* ((x (copy-list '(a b c d e f g h)))
	 (xcopy (make-scaffold-copy x)))
    (setf *mapc.6-var* nil)
    (let ((result (mapcar 'mapc.6-fun x)))
      (and (check-scaffold-copy x xcopy)
	   (list *mapc.6-var* result))))
 ((h g f e d c b a) (a b c d e f g h)))

(deftest mapcar.order.1
  (let ((i 0) x y z)
    (values
     (mapcar (progn (setf x (incf i))
		    #'list)
	     (progn (setf y (incf i))
		    '(a b c))
	     (progn (setf z (incf i))
		    '(1 2 3)))
     i x y z))
  ((a 1) (b 2) (c 3))
  3 1 2 3)

(deftest mapcar.error.1
  (classify-error (mapcar #'identity 1))
  type-error)

(deftest mapcar.error.2
  (classify-error (mapcar))
  program-error)

(deftest mapcar.error.3
  (classify-error (mapcar #'append))
  program-error)

(deftest mapcar.error.4
  (classify-error (locally (mapcar #'identity 1) t))
  type-error)

(deftest mapcar.error.5
  (classify-error (mapcar #'car '(a b c)))
  type-error)

(deftest mapcar.error.6
  (classify-error (mapcar #'cons '(a b c)))
  program-error)

(deftest mapcar.error.7
  (classify-error (mapcar #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; mapcan

(deftest mapcan.1
  (mapcan #'list nil)
  nil)

(deftest mapcan.2
  (mapcan #'list (copy-list '(a b c d e f)))
  (a b c d e f))

(deftest mapcan.3
  (let* ((x (list 'a 'b 'c 'd))
	 (xcopy (make-scaffold-copy x))
	 (result (mapcan #'list x)))
    (and
     (= (length x) (length result))
     (check-scaffold-copy x xcopy)
     (loop
      for e1 on x
      and e2 on result
      count (or (eqt e1 e2) (not (eql (car e1) (car e2)))))))
  0)

(deftest mapcan.4
  (mapcan #'list
	  (copy-list '(1 2 3 4))
	  (copy-list '(a b c d)))
  (1 a 2 b 3 c 4 d))

(deftest mapcan.5
  (mapcan #'(lambda (x y) (make-list y :initial-element x))
	  (copy-list '(a b c d))
	  (copy-list '(1 2 3 4)))
  (a b b c c c d d d d))

(defvar *mapcan.6-var* nil)
(defun mapcan.6-fun (x)
  (push x *mapcan.6-var*)
  (copy-list *mapcan.6-var*))

(deftest mapcan.6
  (progn
    (setf *mapcan.6-var* nil)
    (mapcan 'mapcan.6-fun (copy-list '(a b c d))))
  (a b a c b a d c b a))

(deftest mapcan.order.1
  (let ((i 0) x y z)
    (values
     (mapcan (progn (setf x (incf i))
		    #'list)
	     (progn (setf y (incf i))
		    '(a b c))
	     (progn (setf z (incf i))
		    '(1 2 3)))
     i x y z))
  (a 1 b 2 c 3)
  3 1 2 3)

(deftest mapcan.8
  (mapcan #'(lambda (x y) (make-list y :initial-element x))
	  (copy-list '(a b c d))
	  (copy-list '(1 2 3 4 5 6)))
  (a b b c c c d d d d))

(deftest mapcan.9
  (mapcan #'(lambda (x y) (make-list y :initial-element x))
	  (copy-list '(a b c d e f))
	  (copy-list '(1 2 3 4)))
  (a b b c c c d d d d))

(deftest mapcan.10
  (mapcan #'list
	  (copy-list '(a b c d))
	  (copy-list '(1 2 3 4))
	  nil)
  nil)

(deftest mapcan.11
  (mapcan (constantly 1) (list 'a))
  1)

(deftest mapcan.error.1
  (classify-error (mapcan #'identity 1))
  type-error)

(deftest mapcan.error.2
  (classify-error (mapcan))
  program-error)

(deftest mapcan.error.3
  (classify-error (mapcan #'append))
  program-error)

(deftest mapcan.error.4
  (classify-error (locally (mapcan #'identity 1) t))
  type-error)

(deftest mapcan.error.5
  (classify-error (mapcan #'car '(a b c)))
  type-error)

(deftest mapcan.error.6
  (classify-error (mapcan #'cons '(a b c)))
  program-error)

(deftest mapcan.error.7
  (classify-error (mapcan #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; mapl

(deftest mapl.1
  (mapl #'list nil)
  nil)

(deftest mapl.2
  (let* ((a nil)
	 (x (copy-list '(a b c)))
	 (xcopy (make-scaffold-copy x))
	 (result
	  (mapl #'(lambda (y) (push y a))
		x)))
    (and
     (check-scaffold-copy x xcopy)
     (eqt result x)
     a))
  ((c) (b c) (a b c)))

(deftest mapl.3
  (let* ((a nil)
	 (x (copy-list '(a b c d)))
	 (y (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (mapl #'(lambda (xtail ytail)
		    (setf a
			  (append (mapcar #'list xtail ytail)
				  a)))
		x y)))
    (and
     (eqt result x)
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     a))
  ((d 4) (c 3) (d 4) (b 2) (c 3) (d 4)
   (a 1) (b 2) (c 3) (d 4)))

(deftest mapl.4
  (let* ((a nil)
	 (x (copy-list '(a b c d)))
	 (y (copy-list '(1 2 3 4 5 6 7 8)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (mapl #'(lambda (xtail ytail)
		    (setf a
			  (append (mapcar #'list xtail ytail)
				  a)))
		x y)))
    (and
     (eqt result x)
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     a))
  ((d 4) (c 3) (d 4) (b 2) (c 3) (d 4)
   (a 1) (b 2) (c 3) (d 4)))

(deftest mapl.5
  (let* ((a nil)
	 (x (copy-list '(a b c d e f g)))
	 (y (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (mapl #'(lambda (xtail ytail)
		    (setf a
			  (append (mapcar #'list xtail ytail)
				  a)))
		x y)))
    (and
     (eqt result x)
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     a))
  ((d 4) (c 3) (d 4) (b 2) (c 3) (d 4)
   (a 1) (b 2) (c 3) (d 4)))

(deftest mapl.order.1
  (let ((i 0) x y z)
    (values
     (mapl (progn
	     (setf x (incf i))
	     (constantly nil))
	   (progn
	     (setf y (incf i))
	     '(a b c))
	   (progn
	     (setf z (incf i))
	     '(1 2 3)))
     i x y z))
  (a b c) 3 1 2 3)

(deftest mapl.error.1
  (classify-error (mapl #'identity 1))
  type-error)

(deftest mapl.error.2
  (classify-error (mapl))
  program-error)

(deftest mapl.error.3
  (classify-error (mapl #'append))
  program-error)

(deftest mapl.error.4
  (classify-error (locally (mapl #'identity 1) t))
  type-error)

(deftest mapl.error.5
  (classify-error (mapl #'cons '(a b c)))
  program-error)

(deftest mapl.error.6
  (classify-error (mapl #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)

(deftest mapl.error.7
  (classify-error (mapl #'caar '(a b c)))
  type-error)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; maplist

(deftest maplist.1
  (maplist #'list nil)
  nil)

(deftest maplist.2
  (let* ((x (copy-list '(a b c)))
	 (xcopy (make-scaffold-copy x))
	 (result (maplist #'identity x)))
    (and (check-scaffold-copy x xcopy)
	 result))
  ((a b c) (b c) (c)))

(deftest maplist.3
  (let* ((x (copy-list '(a b c d)))
	 (y (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (maplist #'append x y)))
    (and
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     result))
  ((a b c d 1 2 3 4)
   (b c d 2 3 4)
   (c d 3 4)
   (d 4)))

(deftest maplist.4
  (let* ((x (copy-list '(a b c d)))
	 (y (copy-list '(1 2 3 4 5)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (maplist #'append x y)))
    (and
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     result))
  ((a b c d 1 2 3 4 5)
   (b c d 2 3 4 5)
   (c d 3 4 5)
   (d 4 5)))

(deftest maplist.5
  (let* ((x (copy-list '(a b c d e)))
	 (y (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (maplist #'append x y)))
    (and
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     result))
  ((a b c d e 1 2 3 4)
   (b c d e 2 3 4)
   (c d e 3 4)
   (d e 4)))

(deftest maplist.6
  (maplist 'append '(a b c) '(1 2 3))
  ((a b c 1 2 3) (b c 2 3) (c 3)))

(deftest maplist.7
  (maplist #'(lambda (x y) (nth (car x) y))
	   '(0 1 0 1 0 1 0)
	   '(a b c d e f g)
	   )
  (a c c e e g g))

(deftest maplist.order.1
  (let ((i 0) x y z)
    (values
     (maplist
      (progn
	(setf x (incf i))
	#'(lambda (x y) (declare (ignore x)) (car y)))
      (progn
	(setf y (incf i))
	'(a b c))
      (progn
	(setf z (incf i))
	     '(1 2 3)))
     i x y z))
  (1 2 3) 3 1 2 3)

(deftest maplist.error.1
  (classify-error (maplist #'identity 'a))
  type-error)

(deftest maplist.error.2
  (classify-error (maplist #'identity 1))
  type-error)

(deftest maplist.error.3
  (classify-error (maplist #'identity 1.1323))
  type-error)

(deftest maplist.error.4
  (classify-error (maplist #'identity "abcde"))
  type-error)

(deftest maplist.error.5
  (classify-error (maplist))
  program-error)

(deftest maplist.error.6
  (classify-error (maplist #'append))
  program-error)

(deftest maplist.error.7
  (classify-error (locally (maplist #'identity 'a) t))
  type-error)

(deftest maplist.error.8
  (classify-error (maplist #'caar '(a b c)))
  type-error)

(deftest maplist.error.9
  (classify-error (maplist #'cons '(a b c)))
  program-error)

(deftest maplist.error.10
  (classify-error (maplist #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; mapcon

(deftest mapcon.1
  (mapcon #'(lambda (x) (append '(a) x nil)) nil)
  nil)

(deftest mapcon.2
  (let* ((x (copy-list '(1 2 3 4)))
	 (xcopy (make-scaffold-copy x))
	 (result
	  (mapcon #'(lambda (y) (append '(a) y nil)) x)))
    (and
     (check-scaffold-copy x xcopy)
     result))
  (a 1 2 3 4 a 2 3 4 a 3 4 a 4))

(deftest mapcon.3
  (let* ((x (copy-list '(4 2 3 2 2)))
	 (y (copy-list '(a b c d e f g h i j k l)))
	 (xcopy (make-scaffold-copy x))
	 (ycopy (make-scaffold-copy y))
	 (result
	  (mapcon #'(lambda (xt yt)
		      (subseq yt 0 (car xt)))
		  x y)))
    (and
     (check-scaffold-copy x xcopy)
     (check-scaffold-copy y ycopy)
     result))
  (a b c d b c c d e d e e f))

(deftest mapcon.4
  (mapcon (constantly 1) (list 'a))
  1)

(deftest mapcon.order.1
  (let ((i 0) x y z)
    (values
     (mapcon (progn (setf x (incf i))
		    #'(lambda (x y) (list (car x) (car y))))
	     (progn (setf y (incf i))
		    '(a b c))
	     (progn (setf z (incf i))
		    '(1 2 3)))
     i x y z))
  (a 1 b 2 c 3)
  3 1 2 3)

(deftest mapcon.error.1
  (classify-error (mapcon #'identity 1))
  type-error)

(deftest mapcon.error.2
  (classify-error (mapcon))
  program-error)

(deftest mapcon.error.3
  (classify-error (mapcon #'append))
  program-error)

(deftest mapcon.error.4
  (classify-error (locally (mapcon #'identity 1) t))
  type-error)

(deftest mapcon.error.5
  (classify-error (mapcon #'caar '(a b c)))
  type-error)

(deftest mapcon.error.6
  (classify-error (mapcon #'cons '(a b c)))
  program-error)

(deftest mapcon.error.7
  (classify-error (mapcon #'cons '(a b c) '(1 2 3) '(4 5 6)))
  program-error)
