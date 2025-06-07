;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Wed Apr  1 21:49:43 1998
;;;; Contains: Testing of CL Features related to "CONS", part 23

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; set-exclusive-or

(deftest set-exclusive-or.1
    (set-exclusive-or nil nil)
  nil)

(deftest set-exclusive-or.2
    (let ((result
	   (set-exclusive-or-with-check '(a b c) nil)))
      (check-set-exclusive-or '(a b c) nil result))
  t)

(deftest set-exclusive-or.3
    (let ((result
	   (set-exclusive-or-with-check '(a b c d e f) '(f b d))))
      (check-set-exclusive-or '(a b c d e f) '(f b d) result))
  t)

(deftest set-exclusive-or.4
    (sort
     (copy-list
      (set-exclusive-or-with-check (shuffle '(1 2 3 4 5 6 7 8))
				 '(10 101 4 74 2 1391 7 17831)))
     #'<)
  (1 3 5 6 8 10 74 101 1391 17831))

(deftest set-exclusive-or.5
    (check-set-exclusive-or
     nil
     '(a b c d e f g h)
     (set-exclusive-or-with-check nil '(a b c d e f g h)))
  t)

(deftest set-exclusive-or.6
  (set-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :key nil)
  (c))

(deftest set-exclusive-or.7
    (set-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'eq)
  (c))

(deftest set-exclusive-or.7-a
    (set-exclusive-or-with-check '(d a b e) '(a b c d e) :test #'eq)
  (c))

(deftest set-exclusive-or.8
    (set-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'eql)
  (c))

(deftest set-exclusive-or.8-a
    (set-exclusive-or-with-check '(e d b a) '(a b c d e) :test #'eql)
  (c))

(deftest set-exclusive-or.8-b
    (set-exclusive-or-with-check '(a b c d e) '(d a b e)
				 :test-not (complement #'eql))
  (c))

(deftest set-exclusive-or.9
    (set-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'equal)
  (c))

(deftest set-exclusive-or.10
    (set-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'eq)
  (c))

(deftest set-exclusive-or.11
    (set-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'eql)
  (c))

(deftest set-exclusive-or.12
    (set-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'equal)
  (c))

(deftest set-exclusive-or.13
    (do-random-set-exclusive-ors 100 100)
  nil)

(deftest set-exclusive-or.14
    (set-exclusive-or-with-check '((a . 1) (b . 2) (c . 3012))
			       '((a . 10) (c . 3))
			       :key 'car)
  ((b . 2)))

(deftest set-exclusive-or.15
    (set-exclusive-or-with-check '((a . xx) (b . 2) (c . 3))
			       '((a . 1) (c . 3313))
			       :key #'car)
  ((b . 2)))

(deftest set-exclusive-or.16
    (set-exclusive-or-with-check '((a . xx) (b . 2) (c . 3))
			       '((a . 1) (c . 3313))
			       :key #'car
			       :test-not (complement #'eql))
  ((b . 2)))

;;
;; Check that set-exclusive-or does not invert
;; the order of the arguments to the test function
;;
(deftest set-exclusive-or.17
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (set-exclusive-or-with-check
	list1 list2
	:test #'(lambda (s1 s2)
		  (when (or (member s1 list2)
			    (member s2 list1))
		    (return-from fail 'failed)))))))
  t)

(deftest set-exclusive-or.17-a
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (set-exclusive-or-with-check
	list1 list2
	:key #'identity
	:test #'(lambda (s1 s2)
		  (when (or (member s1 list2)
			    (member s2 list1))
		    (return-from fail 'failed)))))))
  t)

(deftest set-exclusive-or.18
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (set-exclusive-or-with-check
	list1 list2
	:test-not
	#'(lambda (s1 s2)
	    (when (or (member s1 list2)
		      (member s2 list1))
	      (return-from fail 'failed))
	    t)))))
  t)

(deftest set-exclusive-or.18-a
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (set-exclusive-or-with-check
	list1 list2
	:key #'identity
	:test-not
	#'(lambda (s1 s2)
	    (when (or (member s1 list2)
		      (member s2 list1))
	      (return-from fail 'failed))
	    t)))))
  t)

;;; Order of argument evaluation tests

(deftest set-exclusive-or.order.1
  (let ((i 0) x y)
    (values
     (sort
      (set-exclusive-or (progn (setf x (incf i))
			       (list 1 2 3 4))
			(progn (setf y (incf i))
			       (list 1 3 6 10)))
      #'<)
     i x y))
  (2 4 6 10) 2 1 2)

(deftest set-exclusive-or.order.2
  (let ((i 0) x y z)
    (values
     (sort
      (set-exclusive-or (progn (setf x (incf i))
			       (list 1 2 3 4))
			(progn (setf y (incf i))
			       (list 1 3 6 10))
			:test (progn (setf z (incf i))
				     #'eql))
      #'<)
     i x y z))
  (2 4 6 10) 3 1 2 3)

(deftest set-exclusive-or.order.3
  (let ((i 0) x y z w)
    (values
     (sort
      (set-exclusive-or (progn (setf x (incf i))
			       (list 1 2 3 4))
			(progn (setf y (incf i))
			       (list 1 3 6 10))
			:test (progn (setf z (incf i))
				     #'eql)
			:key (progn (setf w (incf i)) nil))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)

(deftest set-exclusive-or.order.4
  (let ((i 0) x y z w)
    (values
     (sort
      (set-exclusive-or (progn (setf x (incf i))
			       (list 1 2 3 4))
			(progn (setf y (incf i))
			       (list 1 3 6 10))
			:key (progn (setf z (incf i)) nil)
			:test (progn (setf w (incf i))
				     #'eql))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)

(deftest set-exclusive-or.order.5
  (let ((i 0) x y z w)
    (values
     (sort
      (set-exclusive-or (progn (setf x (incf i))
			       (list 1 2 3 4))
			(progn (setf y (incf i))
			       (list 1 3 6 10))
			:key (progn (setf z (incf i)) nil)
			:key (progn (setf w (incf i))
				    (complement #'eql)))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)


;;; Keyword tests

(deftest set-exclusive.allow-other-keys.1
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :bad t :allow-other-keys t)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.allow-other-keys.2
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t :bad t)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.allow-other-keys.3
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t :bad t
			  :test #'(lambda (x y) (= x (1- y))))
	#'<)
  (1 6))

(deftest set-exclusive.allow-other-keys.4
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.allow-other-keys.5
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys nil)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.allow-other-keys.6
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t
			  :allow-other-keys nil)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.allow-other-keys.7
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t
			  :allow-other-keys nil
			  '#:x 1)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.keywords.8
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :test #'eql
			  :test #'/=)
	#'<)
  (1 2 5 6))

(deftest set-exclusive.keywords.9
  (sort (set-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :test #'/=
			  :test #'eql)
	#'<)
  nil)

(deftest set-exclusive-or.error.1
  (classify-error (set-exclusive-or))
  program-error)

(deftest set-exclusive-or.error.2
  (classify-error (set-exclusive-or nil))
  program-error)

(deftest set-exclusive-or.error.3
  (classify-error (set-exclusive-or nil nil :bad t))
  program-error)

(deftest set-exclusive-or.error.4
  (classify-error (set-exclusive-or nil nil :key))
  program-error)

(deftest set-exclusive-or.error.5
  (classify-error (set-exclusive-or nil nil 1 2))
  program-error)

(deftest set-exclusive-or.error.6
  (classify-error (set-exclusive-or nil nil :bad t :allow-other-keys nil))
  program-error)

(deftest set-exclusive-or.error.7
  (classify-error (set-exclusive-or (list 1 2) (list 3 4) :test #'identity))
  program-error)

(deftest set-exclusive-or.error.8
  (classify-error (set-exclusive-or (list 1 2) (list 3 4) :test-not #'identity))
  program-error)

(deftest set-exclusive-or.error.9
  (classify-error (set-exclusive-or (list 1 2) (list 3 4) :key #'cons))
  program-error)

(deftest set-exclusive-or.error.10
  (classify-error (set-exclusive-or (list 1 2) (list 3 4) :key #'car))
  type-error)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; nset-exclusive-or

(deftest nset-exclusive-or.1
    (nset-exclusive-or nil nil)
  nil)

(deftest nset-exclusive-or.2
    (let ((result
	   (nset-exclusive-or-with-check '(a b c) nil)))
      (check-set-exclusive-or '(a b c) nil result))
  t)

(deftest nset-exclusive-or.3
    (let ((result
	   (nset-exclusive-or-with-check '(a b c d e f) '(f b d))))
      (check-set-exclusive-or '(a b c d e f) '(f b d) result))
  t)

(deftest nset-exclusive-or.4
    (sort
     (copy-list
      (nset-exclusive-or-with-check (shuffle '(1 2 3 4 5 6 7 8))
				 '(10 101 4 74 2 1391 7 17831)))
     #'<)
  (1 3 5 6 8 10 74 101 1391 17831))

(deftest nset-exclusive-or.5
    (check-set-exclusive-or
     nil
     '(a b c d e f g h)
     (nset-exclusive-or-with-check nil '(a b c d e f g h)))
  t)

(deftest nset-exclusive-or.6
  (nset-exclusive-or-with-check '(a b c d e) '(d a b e)
				:key nil)
  (c))

(deftest nset-exclusive-or.7
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'eq)
  (c))

(deftest nset-exclusive-or.7-a
    (nset-exclusive-or-with-check '(d a b e) '(a b c d e) :test #'eq)
  (c))

(deftest nset-exclusive-or.8
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'eql)
  (c))

(deftest nset-exclusive-or.8-a
    (nset-exclusive-or-with-check '(e d b a) '(a b c d e) :test #'eql)
  (c))

(deftest nset-exclusive-or.8-b
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e)
				  :test-not (complement #'eql))
  (c))

(deftest nset-exclusive-or.9
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e) :test #'equal)
  (c))

(deftest nset-exclusive-or.10
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'eq)
  (c))

(deftest nset-exclusive-or.11
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'eql)
  (c))

(deftest nset-exclusive-or.12
    (nset-exclusive-or-with-check '(a b c d e) '(d a b e)
			       :test 'equal)
  (c))

(deftest nset-exclusive-or.13
    (do-random-nset-exclusive-ors 100 100)
  nil)

(deftest nset-exclusive-or.14
    (nset-exclusive-or-with-check '((a . 1) (b . 2) (c . 3012))
			       '((a . 10) (c . 3))
			       :key 'car)
  ((b . 2)))

(deftest nset-exclusive-or.15
    (nset-exclusive-or-with-check '((a . xx) (b . 2) (c . 3))
			       '((a . 1) (c . 3313))
			       :key #'car)
  ((b . 2)))

(deftest nset-exclusive-or.16
    (nset-exclusive-or-with-check '((a . xx) (b . 2) (c . 3))
			       '((a . 1) (c . 3313))
			       :key #'car
			       :test-not (complement #'eql))
  ((b . 2)))

;;
;; Check that nset-exclusive-or does not invert
;; the order of the arguments to the test function
;;
(deftest nset-exclusive-or.17
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (nset-exclusive-or-with-check
	list1 list2
	:test #'(lambda (s1 s2)
		  (when (or (member s1 list2)
			    (member s2 list1))
		    (return-from fail 'failed)))))))
  t)

(deftest nset-exclusive-or.17-a
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (nset-exclusive-or-with-check
	list1 list2
	:key #'identity
	:test #'(lambda (s1 s2)
		  (when (or (member s1 list2)
			    (member s2 list1))
		    (return-from fail 'failed)))))))
  t)

(deftest nset-exclusive-or.18
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (nset-exclusive-or-with-check
	list1 list2
	:test-not
	#'(lambda (s1 s2)
	    (when (or (member s1 list2)
		      (member s2 list1))
	      (return-from fail 'failed))
	    t)))))
  t)

(deftest nset-exclusive-or.18-a
  (let ((list1 '(a b c d))
	(list2 '(e f g h)))
    (block fail
      (notnot-mv
       (nset-exclusive-or-with-check
	list1 list2
	:key #'identity
	:test-not
	#'(lambda (s1 s2)
	    (when (or (member s1 list2)
		      (member s2 list1))
	      (return-from fail 'failed))
	    t)))))
  t)

;;; Order of argument evaluation tests

(deftest nset-exclusive-or.order.1
  (let ((i 0) x y)
    (values
     (sort
      (nset-exclusive-or (progn (setf x (incf i))
				(list 1 2 3 4))
			 (progn (setf y (incf i))
				(list 1 3 6 10)))
      #'<)
     i x y))
  (2 4 6 10) 2 1 2)

(deftest nset-exclusive-or.order.2
  (let ((i 0) x y z)
    (values
     (sort
      (nset-exclusive-or (progn (setf x (incf i))
				(list 1 2 3 4))
			 (progn (setf y (incf i))
				(list 1 3 6 10))
			 :test (progn (setf z (incf i))
				      #'eql))
      #'<)
     i x y z))
  (2 4 6 10) 3 1 2 3)

(deftest nset-exclusive-or.order.3
  (let ((i 0) x y z w)
    (values
     (sort
      (nset-exclusive-or (progn (setf x (incf i))
				(list 1 2 3 4))
			 (progn (setf y (incf i))
				(list 1 3 6 10))
			 :test (progn (setf z (incf i))
				      #'eql)
			 :key (progn (setf w (incf i)) nil))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)

(deftest nset-exclusive-or.order.4
  (let ((i 0) x y z w)
    (values
     (sort
      (nset-exclusive-or (progn (setf x (incf i))
				(list 1 2 3 4))
			 (progn (setf y (incf i))
				(list 1 3 6 10))
			 :key (progn (setf z (incf i)) nil)
			 :test (progn (setf w (incf i))
				      #'eql))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)

(deftest nset-exclusive-or.order.5
  (let ((i 0) x y z w)
    (values
     (sort
      (nset-exclusive-or (progn (setf x (incf i))
				(list 1 2 3 4))
			 (progn (setf y (incf i))
				(list 1 3 6 10))
			 :key (progn (setf z (incf i)) nil)
			 :key (progn (setf w (incf i))
				     (complement #'eql)))
      #'<)
     i x y z w))
  (2 4 6 10) 4 1 2 3 4)


;;; Keyword tests

(deftest nset-exclusive.allow-other-keys.1
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :bad t :allow-other-keys t)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.allow-other-keys.2
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t :bad t)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.allow-other-keys.3
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t :bad t
			  :test #'(lambda (x y) (= x (1- y))))
	#'<)
  (1 6))

(deftest nset-exclusive.allow-other-keys.4
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.allow-other-keys.5
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys nil)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.allow-other-keys.6
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t
			  :allow-other-keys nil)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.allow-other-keys.7
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :allow-other-keys t
			  :allow-other-keys nil
			  '#:x 1)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.keywords.8
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :test #'eql
			  :test #'/=)
	#'<)
  (1 2 5 6))

(deftest nset-exclusive.keywords.9
  (sort (nset-exclusive-or (list 1 2 3 4) (list 3 4 5 6)
			  :test #'/=
			  :test #'eql)
	#'<)
  nil)

;;; Error tests

(deftest nset-exclusive-or.error.1
  (classify-error (nset-exclusive-or))
  program-error)

(deftest nset-exclusive-or.error.2
  (classify-error (nset-exclusive-or nil))
  program-error)

(deftest nset-exclusive-or.error.3
  (classify-error (nset-exclusive-or nil nil :bad t))
  program-error)

(deftest nset-exclusive-or.error.4
  (classify-error (nset-exclusive-or nil nil :key))
  program-error)

(deftest nset-exclusive-or.error.5
  (classify-error (nset-exclusive-or nil nil 1 2))
  program-error)

(deftest nset-exclusive-or.error.6
  (classify-error (nset-exclusive-or nil nil :bad t :allow-other-keys nil))
  program-error)

(deftest nset-exclusive-or.error.7
  (classify-error (nset-exclusive-or (list 1 2) (list 3 4) :test #'identity))
  program-error)

(deftest nset-exclusive-or.error.8
  (classify-error (nset-exclusive-or (list 1 2) (list 3 4) :test-not #'identity))
  program-error)

(deftest nset-exclusive-or.error.9
  (classify-error (nset-exclusive-or (list 1 2) (list 3 4) :key #'cons))
  program-error)

(deftest nset-exclusive-or.error.10
  (classify-error (nset-exclusive-or (list 1 2) (list 3 4) :key #'car))
  type-error)

