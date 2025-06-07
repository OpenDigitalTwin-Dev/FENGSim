;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defun mask (nbits &optional (off 0))
  (if (eql nbits fixnum-length)
      -1
      (<< (~ (<< -1 nbits)) (end-shft off nbits))))
(setf (get 'mask 'compiler::cmp-inline) t)

(defun b<< (x y)
#+clx-little-endian (<< x y)
#-clx-little-endian (>> x y))
(setf (get 'b<< 'compiler::cmp-inline) t)

(defun b>> (x y)
#+clx-little-endian (>> x y)
#-clx-little-endian (<< x y))
(setf (get 'b>> 'compiler::cmp-inline) t)


(defun merge-word (x y m) (\| (& x m) (& y (~ m))))
(setf (get 'merge-word 'compiler::cmp-inline) t)

(defun bit-array-fixnum (a i n)
  (if (<= 0 i n)
      (*fixnum (c-array-self a) i nil 0)
      0))
(setf (get 'bit-array-fixnum 'compiler::cmp-inline) t)

(defun set-bit-array-fixnum (a i v);(a i n v)
;  (assert (<= 0 i n))
  (*fixnum (c-array-self a) i t v))
(setf (get 'set-bit-array-fixnum 'compiler::cmp-inline) t)

(defun gw (a i n od)
  (cond ((zerop od) (bit-array-fixnum a i n))
	((plusp od)
	 (merge-word
	  (b>> (bit-array-fixnum a i n) od)
	  (b<< (bit-array-fixnum a (1+ i) n) (- fixnum-length od))
	  (mask (- fixnum-length od))))
	((merge-word
	  (b>> (bit-array-fixnum a (1- i) n) (+ fixnum-length od))
	  (b<< (bit-array-fixnum a i n) (- od))
	  (mask (- od))))))
(setf (get 'gw 'compiler::cmp-inline) t)

(defun bit-array-op (fn ba1 ba2 &optional rba (so1 0) (so2 0) (so3 0) n
		     &aux
		       (rba (case rba
			      ((t) ba1)
			      ((nil) (make-array (array-dimensions ba1) :element-type 'bit))
			      (otherwise rba))))
  (let* ((o3 (+ so3 (array-offset rba)))
	 (y (or n (array-total-size rba)));min
	 (o1 (+ so1 (array-offset ba1)))
	 (n1 (ceiling (+ o1 (array-total-size ba1)) fixnum-length))
	 (o1 (- o1 o3))
	 (o2 (+ so2 (array-offset ba2)))
	 (n2 (ceiling (+ o2 (array-total-size ba2)) fixnum-length))
	 (o2 (- o2 o3)))
    
    (multiple-value-bind
     (nw rem) (floor (+ o3 y) fixnum-length)
      
     (let ((i 0)(n3 (if (zerop rem) nw (1+ nw))))
       
       (when (plusp o3)
	 (set-bit-array-fixnum
	  rba i
	  (merge-word
	   (funcall fn (gw ba1 i n1 o1) (gw ba2 i n2 o2)) 
	   (bit-array-fixnum rba i n3)
	   (mask (min y (- fixnum-length o3)) o3)))
	 (incf i))
       
       (do nil ((>= i nw))
	   (set-bit-array-fixnum
	    rba i
	    (funcall fn (gw ba1 i n1 o1) (gw ba2 i n2 o2)))
	   (incf i))
       
       (when (and (plusp rem) (eql i nw))
	 (set-bit-array-fixnum
	  rba i
	  (merge-word
	   (funcall fn (gw ba1 i n1 o1) (gw ba2 i n2 o2))
	   (bit-array-fixnum rba i n3)
	   (mask rem))))
       
       rba))))
(setf (get 'bit-array-op 'compiler::cmp-inline) t)

(defun copy-bit-vector (a i b j n)
  (bit-array-op (lambda (x y) (declare (ignore x)) y) a b t i j i n))


;FIXME array-dimensions allocates....
(defvar *bit-array-dimension-check-ref* nil)

(defun bit-array-dimension-check (y &aux (r (array-rank *bit-array-dimension-check-ref*)))
  (when (eql r (array-rank y))
    (dotimes (i r t)
      (unless (eql (array-dimension *bit-array-dimension-check-ref* i) (array-dimension y i))
	(return nil)))))
(setf (get 'bit-array-dimension-check 'compiler::cmp-inline) t)

(eval-when
 (compile eval)
 (defmacro defbitfn (f fn &aux (n (eq f 'bit-not)))
   `(defun ,f (x ,@(unless n `(y)) &optional rz)
      (declare (optimize (safety 1)))
      (check-type x (array bit))
      ,@(unless n `((check-type y (array bit))))
      (check-type rz (or boolean (array bit)))
      (let ((*bit-array-dimension-check-ref* x),@(unless n '((y y)))(rz rz))
	,@(unless n '((check-type y (and (array bit) (satisfies bit-array-dimension-check)))))
	(check-type rz (or boolean (and (array bit) (satisfies bit-array-dimension-check))))
	(bit-array-op ,fn x ,(if n 'x 'y) rz)))))


(defbitfn bit-and #'&)
(defbitfn bit-ior #'\|)
(defbitfn bit-xor #'^)
(defbitfn bit-eqv   (lambda (x y) (~ (^ x y))))
(defbitfn bit-nand  (lambda (x y) (~ (& x y))))
(defbitfn bit-nor   (lambda (x y) (~ (\| x y))))
(defbitfn bit-andc1 (lambda (x y) (& (~ x) y)))
(defbitfn bit-andc2 (lambda (x y) (& x (~ y))))
(defbitfn bit-orc1  (lambda (x y) (\| (~ x) y)))
(defbitfn bit-orc2  (lambda (x y) (\| x (~ y))))
(defbitfn bit-not   (lambda (x y) (declare (ignore y)) (~ x)))

(defun baset (v x &rest r)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type x (array bit))
  (apply 'aset v x r))
(setf (get 'baset 'compiler::cmp-inline) t)
(defun sbaset (v x &rest r)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type x (simple-array bit))
  (apply 'aset v x r))
(setf (get 'sbaset 'compiler::cmp-inline) t)
