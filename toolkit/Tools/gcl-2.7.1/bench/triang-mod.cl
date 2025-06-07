;; $Header$
;; $Locker$

;;; TRIANG -- Board game benchmark.
;;; In converting to common lisp eq compares of fixnums have been changed
;;; to eql and the type of the board vectors has been declared.

(declaim (special board seq a b c))
(declaim (type (vector fixnum ) board seq a b c))

(defvar answer)
(defvar final)

(defun triang-setup ()
  (setq board (make-array 16 :element-type 'fixnum :initial-element 1))
  (setq seq (make-array 14 :element-type 'fixnum :initial-element 0))
  (setq a
    (make-array
      37
      :element-type 'fixnum 
      :initial-contents
      '(1 2 4 3 5 6 1 3 6 2 5 4 11 12 13 7 8 4 4 7 11 8 12
	13 6 10 15 9 14 13 13 14 15 9 10 6 6)))
  (setq b (make-array
	    37 
            :element-type 'fixnum
            :initial-contents
	    '(2 4 7 5 8 9 3 6 10 5 9 8 12 13 14 8 9 5
	      2 4 7 5 8 9 3 6 10 5 9 8 12 13 14 8 9 5 5)))
  (setq c (make-array
	    37 
            :element-type 'fixnum
            :initial-contents
	    '(4 7 11 8 12 13 6 10 15 9 14 13 13 14 15 9 10 6
	      1 2 4 3 5 6 1 3 6 2 5 4 11 12 13 7 8 4 4)))
  (setf (aref board 5) 0))

(defun last-position ()
  (do ((i 1 (the fixnum (+ i 1))))
      ((= i 16) 0)
    (declare (fixnum i))
    (if (eql 1 (aref board i))
	(return i))))

(defun try (i depth)
  (declare (fixnum i depth))
  (cond ((= depth 14) 
	 (let ((lp (last-position)))
	   (unless (member lp final :test #'eql)
	     (push lp final)))
    ;;;     (format t "~&~s"  (cdr (simple-vector-to-list seq)))
	 (push (cdr (simple-vector-to-list seq))
	       answer) t) 		; this is a hack to replace LISTARRAY
	((and (eql 1 (aref board (aref a i)))
	      (eql 1 (aref board (aref b i)))
	      (eql 0 (aref board (aref c i))))
	 (setf (aref board (aref a i)) 0)
	 (setf (aref board (aref b i)) 0)
	 (setf (aref board (aref c i)) 1)
	 (setf (aref seq depth) i)
	 (do ((j 0 (the fixnum (+ j 1)))
	      (depth (the fixnum (+ depth 1))))
	     ((or (= j 36)
		  (try j depth)) ())
	   (declare (fixnum j depth)))
	 (setf (aref board (aref a i)) 1) 
	 (setf (aref board (aref b i)) 1)
	 (setf (aref board (aref c i)) 0) ())))

(defun simple-vector-to-list (seq)
  (do ((i (- (length seq) 1) (1- i))
       (res))
      ((< i 0)
       res)
    (declare (fixnum i))
    (declare (type (array fixnum) seq))
    (push (aref  seq i) res)))
		
(defun gogogo (i)
  (let ((answer ())
	(final ()))
    (try i 1)))

(defun testtriang ()
  (triang-setup)
  (print (time (gogogo 22))))
