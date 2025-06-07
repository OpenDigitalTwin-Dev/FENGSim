;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
;; Copyright (C) 2024 Camm Maguire

;; This file is part of GNU Common Lisp, herein referred to as GCL
;;
;; GCL is free software; you can redistribute it and/or modify it under
;;  the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;; 
;; GCL is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
;; License for more details.
;; 
;; You should have received a copy of the GNU Library General Public License 
;; along with GCL; see the file COPYING.  If not, write to the Free Software
;; Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.


;;;;    listlib.lsp
;;;;
;;;;                        list manipulating routines

; Rewritten 11 Feb 1993 by William Schelter and Gordon Novak to use iteration
; rather than recursion, as needed for large data sets.

(in-package :system)

(eval-when
 (compile eval)
 
  (defmacro collect (r rp form)
    `(let ((tmp ,form))
       (setq ,rp (cond (,rp (rplacd ,rp tmp) tmp) ((setq ,r tmp))))))

  (defmacro cons-length (x)
    (declare (optimize (safety 2)))
    `(let ((,x ,x))
       (if (not ,x) 0
	   (do ((i 1 (1+ i))(s ,x (cdr s))(f (cdr ,x) (cddr f)))
	       ((>= i array-dimension-limit) (- array-dimension-limit))
	     (cond ((eq s f) (return i))
		   ((endp f) (return (1+ (- (+ i i)))))
		   ((endp (cdr f)) (return (- (+ i i))))))))))

#.(let (r)
    (labels
	((f (n r) (when (plusp n) (cons `(pop ,r) (f (1- n) r))))
	 (d (n &aux (s (intern (format nil "AFC~a" n))))
	   (setq r (cons (cons n s) r))
	   `(progn (declaim (inline ,s))
		   (defun ,s (f s r)
		     (declare (function f)(proper-list r))
		     (values (funcall f s ,@(f n 'r))))))
	 (a (n) (when (plusp n) (cons (d n) (a (1- n))))))
      `(progn
	 ,@(a (- call-arguments-limit 2))
	 (defconstant +afc-syms+ ',r))))

(declaim (inline afc-sym))
(defun afc-sym (n)
  (labels ((f (n s) (when s (if (eql (caar s) n) (cdar s) (f n (cdr s))))))
    (f n +afc-syms+)))

(defun mapl (fd list &rest r &aux (fun (coerce fd 'function)))
  (declare (optimize (safety 1))(dynamic-extent r)(notinline make-list));FIXME
  (check-type fd function-designator)
  (check-type list proper-list)
  (labels ((lmap (f x) (when x (funcall f x) (lmap f (cdr x))))
	   (lmapr (f x) (lmap f x) x))
    (if (not r) (lmapr fun list);compiler accelerator
	(let* ((lr (length r))(q (make-list lr))(nf (afc-sym lr)))
	  (declare (dynamic-extent q))
	  (labels ((a-cons (x) (check-type x list) (or x (return-from mapl list)))
		   (last nil (lmapr (lambda (x) (rplaca x (if r (a-cons (pop r)) (a-cons (cdar x))))) q)))
	    ;cannot apply as fun might capture (last) via &rest
	    (lmapr (lambda (x) (funcall nf fun x (last))) list))))))


(defun mapc (fd list &rest r &aux (fun (coerce fd 'function)))
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type fd function-designator)
  (check-type list proper-list)
  (if (not r) (mapl (lambda (x) (funcall fun (car x))) list);compiler accelerator
      (let* ((lr (length r))(q (make-list lr))(nf (afc-sym lr)))
	(declare (dynamic-extent q))
	(apply 'mapl (lambda (x &rest r) (funcall nf fun (car x) (mapl (lambda (x) (setf (car x) (car (pop r)))) q))) list r))))


(defun mapcar (fd list &rest r &aux (fun (coerce fd 'function)) res rp)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type fd function-designator)
  (check-type list proper-list)
  (apply 'mapc (lambda (x &rest z &aux (tem (cons (apply fun x z) nil)))
		 (setq rp (if rp (cdr (rplacd rp tem)) (setq res tem)))) list r)
  res)

(defun mapcan (fd list &rest r &aux (fun (coerce fd 'function)) res rp)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type fd function-designator)
  (check-type list proper-list)
  (apply 'mapc (lambda (x &rest z &aux (tem (apply fun x z)))
		 (if rp (rplacd rp tem) (setq res tem))
		 (when (consp tem) (setq rp (last tem)))) list r)
  res)

(defun maplist (fd list &rest r &aux (fun (coerce fd 'function)) res rp)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type fd function-designator)
  (check-type list proper-list)
  (apply 'mapl (lambda (x &rest z &aux (tem (cons (apply fun x z) nil)))
		 (setq rp (if rp (cdr (rplacd rp tem)) (setq res tem)))) list r)
  res)

(defun mapcon(fd list &rest r &aux (fun (coerce fd 'function)) res rp)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type fd function-designator)
  (check-type list proper-list)
  (apply 'mapl (lambda (x &rest z &aux (tem (apply fun x z)))
		 (if rp (rplacd rp tem) (setq res tem))
		 (when (consp tem) (setq rp (last tem)))) list r)
  res)

(defun endp (x)
  (declare (optimize (safety 2)))
  (check-type x list)
  (not x))

(defun nthcdr (n x)
  (declare (optimize (safety 2)))
  (check-type n (integer 0))
  (check-type x list)
  (when x ;FIXME?
    (let ((n (cond ((<= n array-dimension-limit) n) 
		   ((let ((j (cons-length x))) (when (> j 0) (mod n j))))
		   ((return-from nthcdr nil)))))
      (labels ((lnthcdr (x n) (if (or (<= n 0) (endp x)) x (lnthcdr (cdr x) (1- n)))))
	(lnthcdr x n)))))

(defun last (x &optional (n 1));FIXME check for circle
  (declare (optimize (safety 2)))
  (check-type x list)
  (check-type n (integer 0))
  (let* ((n (min array-dimension-limit n))
	 (w (cond ((= n 1) (cdr x))
		  ((do ((n n (1- n))(w x (cdr w))) ((<= n 0) w)
		       (unless (consp w) (return-from last x)))))))
    (do ((x x (cdr x)) (w w (cdr w)))
	((atom w) x))))

(defun butlast (x &optional (n 1));FIXME check for circle
  (declare (optimize (safety 2)))
  (check-type x list)
  (check-type n (integer 0))
  (let* ((n (min array-dimension-limit n))
	 (w (cond ((= n 1) (cdr x))
		  ((do ((n n (1- n))(w x (cdr w))) ((<= n 0) w)
		       (unless (consp w) (return-from butlast nil)))))))
    (do (r rp (x x (cdr x)) (w w (cdr w)))
	((atom w) r)
	(let ((tmp (cons (car x) nil))) (collect r rp tmp)))))

(defun nbutlast (x &optional (n 1));FIXME check for circle
  (declare (optimize (safety 2)))
  (check-type x list)
  (check-type n (integer 0))
  (let* ((n (min array-dimension-limit n))
	 (w (cond ((= n 1) (cdr x))
		  ((do ((n n (1- n))(w x (cdr w))) ((<= n 0) w)
		       (unless (consp w) (return-from nbutlast nil)))))))
    (do ((r x) (rp nil x) (x x (cdr x)) (w w (cdr w)))
	((atom w) (when rp (rplacd rp nil) r)))))

(defun ldiff (l tl &aux r rp)
  (declare (optimize (safety 1)))
  (check-type l list)
  (labels ((srch (x)
	     (cond ((eql x tl) (when rp (rplacd rp nil)) r)
		   ((atom x) (when rp (rplacd rp x)) r)
		   (t (let ((tmp (cons (car x) (cdr x))))
			(setq rp (if rp (cdr (rplacd rp tmp)) (setq r tmp)))
			(srch (cdr x)))))))
    (srch l)))

(defun tailp (tl l)
  (declare (optimize (safety 1)))
  (check-type l list)
  (labels ((srch (x)
	     (or (eql x tl)
		 (unless (atom x)
		   (srch (cdr x))))))
    (srch l)))

(defun list-length (l)
  (declare (optimize (safety 2)))
  (check-type l list)
  (cond ((endp l) 0) 
	((endp (setq l (cdr l))) 1)
	((endp (setq l (cdr l))) 2)
	((endp (setq l (cdr l))) 3)
	((endp (setq l (cdr l))) 4)
	((let ((x (cons-length l)))
	   (when (<= x 0) (+ 4 (- x)))))))

(defun make-list (n &key initial-element)
  (declare (optimize (safety 2)))
  (check-type n seqind)
  (do (r (n n (1- n))) ((<= n 0) r)
      (push initial-element r)))

(defun rest (l)
  (declare (optimize (safety 2)))
  (check-type l list)
  (cdr l))

(defun acons (key datum alist)
  (declare (optimize (safety 2)))
  (cons (cons key datum) alist))

(defun pairlis (k d &optional a)
  (declare (optimize (safety 1)))
  (check-type k proper-list)
  (check-type d proper-list)
  (mapc (lambda (x y) (setq a (acons x y a))) k d)
  a)

(defun copy-list (l)
  (declare (optimize (safety 2)))
  (check-type l list)
  (do (r rp (l l (cdr l))) ((atom l) (when rp (rplacd rp l)) r)
      (let ((tmp (cons (car l) nil))) (collect r rp tmp))))

(defun copy-alist (l)
  (declare (optimize (safety 1)))
  (check-type l proper-list)
  (maplist (lambda (x &aux (e (car x))) (if (consp e) (cons (car e) (cdr e)) e)) l))


	
(defun nconc (&rest l)
  (declare (optimize (safety 1))(dynamic-extent l))
  (if (cdr l)
      (let ((x (pop l))(y (apply 'nconc l)))
	(etypecase x (cons (rplacd (last x) y) x)(null y)))
      (car l)))

(defun nreconc (list tail &aux r)
  (declare (optimize (safety 1)))
  (check-type list proper-list)
  (mapl (lambda (x) (when r (setq tail (rplacd r tail))) (setq r x)) list)
  (if r (rplacd r tail) tail))

(defun nth (n x)
  (declare (optimize (safety 2)))
  (check-type n (integer 0))
  (check-type x list)
  (car (nthcdr n x)))

(defun first (x)   (declare (optimize (safety 2))) (check-type x list) (car x))
(defun second (x)  (declare (optimize (safety 2))) (check-type x list) (cadr x))
(defun third (x)   (declare (optimize (safety 2))) (check-type x list) (caddr x))
(defun fourth (x)  (declare (optimize (safety 2))) (check-type x list) (cadddr x))
(defun fifth (x)   (declare (optimize (safety 2))) (check-type x list) (car (cddddr x)))
(defun sixth (x)   (declare (optimize (safety 2))) (check-type x list) (cadr (cddddr x)))
(defun seventh (x) (declare (optimize (safety 2))) (check-type x list) (caddr (cddddr x)))
(defun eighth (x)  (declare (optimize (safety 2))) (check-type x list) (cadddr (cddddr x)))
(defun ninth (x)   (declare (optimize (safety 2))) (check-type x list) (car (cddddr (cddddr x))))
(defun tenth (x)   (declare (optimize (safety 2))) (check-type x list) (cadr (cddddr (cddddr x))))

; Courtesy Paul Dietz
(defmacro nth-value (n expr)
  (declare (optimize (safety 2)))
  `(nth ,n (multiple-value-list ,expr)))

(defun copy-tree (tr)
  (declare (optimize (safety 2)))
  (do (st cs a (g (sgen))) (nil)
      (declare (dynamic-extent st cs))
      (cond ((atom tr)
	     (do nil ((or (not cs) (eq g (car cs))))
		 (setq a (pop cs) st (cdr st) tr (cons a tr)))
	     (unless cs (return tr))
	     (setf (car cs) tr tr (cdar st)))
	    ((setq st (cons tr st) cs (cons g cs) tr (car tr))))))


(defun append (&rest l)
  (declare (optimize (safety 1))(dynamic-extent l))
  (if (cdr l)
      (let ((x (pop l))(y (apply 'append l)))
	(check-type x proper-list)
	(if (typep y 'proper-list)
	    (let (r rp) (mapc (lambda (x) (collect r rp (cons x nil))) x) (when rp (rplacd rp y)) (or r y))
	    (labels ((f (x) (if x (cons (car x) (f (cdr x))) y))) (f x))))
      (car l)))



(defun revappend (list tail)
  (declare (optimize (safety 1)))
  (check-type list proper-list)
  (mapc (lambda (x) (setq tail (cons x tail))) list)
  tail)

(defun not (x)
  (if x nil t))

(defun null (x)
  (if x nil t))

(defun get-properties (p i &aux s)
  (declare (optimize (safety 1)));FIXME, safety 2 and no check-type loses signature info
  (check-type p proper-list)
  (check-type i proper-list)
  (cond ((endp p) (values nil nil nil))
	((member (setq s (car p)) i :test 'eq) (values s (cadr p) p))
	(t (let ((p (cdr p)))
	     (check-type p proper-cons);FIXME, cons loses proper in return
	     (get-properties (cdr p) i)))))

(defun rplaca (x y)
  (declare (optimize (safety 1)))
  (check-type x cons)
  (c-set-cons-car x y)
  x)

(defun rplacd (x y)
  (declare (optimize (safety 1)))
  (check-type x cons)
  (c-set-cons-cdr x y)
  x)

;(defun listp (x) (typep x 'list));(typecase x (list t)))
(defun consp (x) (when x (listp x)))
(defun atom (x) (not (consp x)))

(defun getf (l i &optional d)
  (declare (optimize (safety 1)))
  (check-type l proper-list)
  (cond ((endp l) d) 
	((eq (car l) i) (cadr l))
	((let ((l (cdr l)))
	   (check-type l cons)
	   (getf (cdr l) i d)))))

(defun identity (x) x)


#-pre-gcl
(eval-when (compile) (load (merge-pathnames "gcl_defseq.lsp" *compile-file-pathname*)))

(defseq member ((item) list :list t)
  (unless (mapl (lambda (x) (when (test item (car x)) (return-from member x))) list)))

(defseq assoc ((item) list :list t)
  (unless (mapc (lambda (x);check-type dropped at safety 1 in assoc-if/not
		  (unless (listp x) (error 'type-error :datum x :expected-type 'list))
		  (when (and x (test item (car x))) (return-from assoc x)))
		list)))

(defseq rassoc ((item) list :list t)
  (unless (mapc (lambda (x)
		  (unless (listp x) (error 'type-error :datum x :expected-type 'list))
		  (when (and x (test item (cdr x))) (return-from rassoc x)))
		list)))

(defseq intersection (nil (l1 l2) :list t)
  (mapcan (lambda (x) (when (member (key x) l2 :test #'test) (cons x nil))) l1))


(defseq union (nil (l1 l2) :list t)
  (let (rp)
    (prog1 (or (mapcan (lambda (x)
			 (unless (member (key x) l2 :test #'test)
			   (setq rp (cons x nil))))
		       l1)
	       l2)
      (when rp (rplacd rp l2)))))


(defseq set-difference (nil (l1 l2) :list t)
  (mapcan (lambda (x)
	    (unless (member (key x) l2 :test #'test)
	      (cons x nil)))
	  l1))


(defseq set-exclusive-or (nil (l1 l2) :list t)
  (let (rp (rr (copy-list l2)))
    (prog1 (or (mapcan (lambda (x &aux (k (key x)))
			 (if (member k l2 :test #'test)
			     (unless (setq rr (delete k rr :test #'test)))
			     (setq rp (cons x nil))))
		       l1)
	       rr)
      (when rp (rplacd rp rr)))))

(defseq nintersection (nil (l1 l2) :list t)
  (let (r rp)
    (mapl (lambda (x)
	    (when (member (key (car x)) l2 :test #'test)
	      (if rp (rplacd rp x) (setq r x)) (setq rp x)))
	  l1)
    (when rp (rplacd rp nil))
    r))

(defseq nunion (nil (l1 l2) :list t)
  (let (r rp)
    (mapl (lambda (x)
	    (unless (member (key (car x)) l2 :test #'test)
	      (if rp (rplacd rp x) (setq r x))(setq rp x)))
	  l1)
    (when rp (rplacd rp l2))
    (or r l2)))

(defseq nset-difference (nil (l1 l2) :list t)
  (let (r rp)
    (mapl (lambda (x)
	    (unless (member (key (car x)) l2 :test #'test)
	      (if rp (rplacd rp x) (setq r x))(setq rp x)))
	  l1)
    (when rp (rplacd rp nil))
    r))


(defseq nset-exclusive-or (nil (l1 l2) :list t)
  (let (r rp (rr (copy-list l2)))
    (mapl (lambda (x &aux (k (key (car x))))
	    (if (member k l2 :test #'test)
		(unless (setq rr (delete k rr :test #'test)))
		(progn (if rp (rplacd rp x) (setq r x))(setq rp x))))
	  l1)
    (when rp (rplacd rp rr))
    (or r rr)))

(defseq subsetp (nil (l1 l2) :list t)
  (mapc (lambda (x)
	  (unless (member (key x) l2 :test #'test)
	    (return-from subsetp nil)))
	l1)
  t)


(defseq subst ((n o) tr :list tree)
  (do (st cs a c rep (g (sgen))) (nil)
      (declare (dynamic-extent st cs))
      (setq rep (test o tr))
      (cond ((or rep (atom tr))
	     (setq tr (if rep n tr))
	     (do nil ((or (not cs) (eq g (car cs))))
		 (setq a (pop cs) c (pop st) tr (if (and (eq a (car c)) (eq tr (cdr c))) c (cons a tr))))
	     (if cs (setf (car cs) tr tr (cdar st)) (return tr)))
	    ((setq st (cons tr st) cs (cons g cs) tr (car tr))))))


(defseq nsubst ((n o) tr :list tree)
  (do (st cs rep (g (sgen))) (nil)
      (declare (dynamic-extent st cs))
      (setq rep (test o tr))
      (cond ((or rep (atom tr))
	     (setq tr (if rep n tr))
	     (do nil ((or (not cs) (eq g (car cs))))
		 (setf (caar st) (pop cs) (cdar st) tr tr (pop st)))
	     (if cs (setf (car cs) tr tr (cdar st)) (return tr)))
	    ((setq st (cons tr st) cs (cons g cs) tr (car tr))))))


(defseq sublis (nil (al tr) :list tree)
  (or (unless al tr)
      (do (st cs a c rep (g (sgen))) (nil)
	(declare (dynamic-extent st cs))
	(setq rep (assoc (key tr) al :test #'test-no-key))
	(cond ((or rep (atom tr))
	       (setq tr (if rep (cdr rep) tr))
	       (do nil ((or (not cs) (eq g (car cs))))
		 (setq a (pop cs) c (pop st) tr (if (and (eq a (car c)) (eq tr (cdr c))) c (cons a tr))))
	       (if cs (setf (car cs) tr tr (cdar st)) (return tr)))
	      ((setq st (cons tr st) cs (cons g cs) tr (car tr)))))))

(defseq nsublis (nil (al tr) :list tree)
  (or (unless al tr)
      (do (st cs rep (g (sgen))) (nil)
	(declare (dynamic-extent st cs))
	(setq rep (assoc (key tr) al :test #'test-no-key))
	(cond ((or rep (atom tr))
	       (setq tr (if rep (cdr rep) tr))
	       (do nil ((or (not cs) (eq g (car cs))))
		 (setf (caar st) (pop cs) (cdar st) tr tr (pop st)))
	       (if cs (setf (car cs) tr tr (cdar st)) (return tr)))
	      ((setq st (cons tr st) cs (cons g cs) tr (car tr)))))))


(defseq adjoin ((item) list :list t :+ifnot nil)
  (if (member (key item) list :test #'test)
      list
    (cons item list)))


(defseq tree-equal (nil (tr1 tr2) :list tree2 :nokey t)
  (do (st1 cs1 st2 (g (sgen))) (nil)
    (declare (dynamic-extent st1 cs1 st2))
    (cond ((and (atom tr1) (consp tr2)) (return nil))
	  ((and (consp tr1) (atom tr2)) (return nil))
	  ((atom tr1)
	   (unless (test tr1 tr2) (return nil))
	   (do nil ((or (not cs1) (eq g (car cs1))))
	     (setq cs1 (cdr cs1) tr1 (pop st1) tr2 (pop st2)))
	   (unless cs1 (return t))
	   (setf (car cs1) tr1 tr1 (cdar st1) tr2 (cdar st2)))
	  ((setq st1 (cons tr1 st1) cs1 (cons g cs1) tr1 (car tr1)
		 st2 (cons tr2 st2) tr2 (car tr2))))))
