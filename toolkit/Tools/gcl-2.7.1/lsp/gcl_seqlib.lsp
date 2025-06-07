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


;;;;   seqlib.lsp
;;;;
;;;;                           sequence routines

(in-package :system)


(defun length (x)
  (declare (optimize (safety 1)))
  (check-type x proper-sequence)
  (labels ((ll (x i) (declare (seqind i)) (if x (ll (cdr x) (1+ i)) i)))
	  (if (listp x)
	      (ll x 0)
	    (if (array-has-fill-pointer-p x)
		(fill-pointer x)
	      (array-dimension x 0)))))

(defun elt (seq n)
  (declare (optimize (safety 1)))
  (check-type seq sequence)
  (check-type n seqind)
  (assert (< n (length seq)) () 'type-error :datum n :expected-type `(integer 0 (,(length seq))))
  (if (listp seq) (nth n seq) (aref seq n)))

(declaim (inline elt-set))
(defun elt-set (seq n v)
  (declare (optimize (safety 1)))
  (check-type seq sequence)
  (check-type n seqind)
  (assert (< n (length seq)) () 'type-error :datum n :expected-type `(integer 0 (,(length seq))))
  (if (listp seq) (setf (nth n seq) v) (setf (aref seq n) v)))

(defun nreverse (s)
  (declare (optimize (safety 1)))
  (check-type s proper-sequence)
  (labels ((lr (tl &optional hd) (if tl (lr (cdr tl) (rplacd tl hd)) hd))
	   (la (&optional (i 0) (j (1- (length s))))
	       (cond ((< i j) (set-array s i s j t) (la (1+ i) (1- j))) (s))))
	  (if (listp s) (lr s) (la))))

(defun reverse (s)
  (declare (optimize (safety 1)))
  (check-type s sequence);FIXME
  (labels ((lr (tl &optional hd) (if tl (lr (cdr tl) (cons (car tl) hd)) hd))
	   (la (&optional (ls (length s)) (r (make-array ls :element-type (array-element-type s))) (i 0) (j (1- ls)))
	       (cond ((and (< i ls) (>= j 0)) (set-array r i s j) (la ls r (1+ i) (1- j))) (r))))
	  (if (listp s) (lr s) (la))))


(defun subseq (s start &optional end)
  (declare (optimize (safety 1)))
  (check-type s sequence)
  (check-type start seqind)
  (check-type end (or null seqind))

  (if (listp s)
      (let ((s (nthcdr start s)))
	(ldiff s (when end (nthcdr (- end start) s))))
      (let* ((ls (length s))(n (- (if (when end (< end ls)) end ls) start)))
	(set-array-n (make-array n :element-type (array-element-type s)) 0 s start n))))


#-pre-gcl
(eval-when (compile) (load (merge-pathnames "gcl_defseq.lsp" *compile-file-pathname*)))

(defseq find ((item) s)
  (labels ((find-loop (i p)
	     (unless (or (< i start) (>= i end) (when l (endp p)))
	       (let ((el (el p i)))
		 (when (test item el)
		   (return-from find el)))
	       (find-loop (if jj i (+ i j)) (cdr p)))))
    (find-loop (if from-end (1- end) start) (when l (or r s)))))

(defseq position ((item) s)
  (labels ((position-loop (i p)
	     (unless (or (< i start) (>= i end) (when l (endp p)))
	       (when (test item (el p i))
		 (return-from position i))
	       (position-loop (+ i j) (cdr p)))))
    (position-loop (if from-end (1- end) start) (when l (or r s)))))


(defseq count ((item) s)
  (labels ((count-loop (i p k)
	     (if (or (< i start) (>= i end) (>= k end) (when l (endp p))) (the seqbnd k);FIXME
		 (count-loop (if jj i (+ i j)) (cdr p) (+ k (if (test item (el p i)) 1 0))))))
    (count-loop (if from-end (1- end) start) (when l (or r s)) 0)))


(defseq remove ((item) seq :count t)
  (let* ((indl (cons (unless l lsa) nil))(inds (when from-end indl)) indp indt)
    (declare (dynamic-extent inds indl indt));FIXME consider removing indices for lists
    (labels ((remove-loop (i p)
	       (unless (or (< i start) (>= i end) (when l (endp p)) (<= cnt 0))
		 (when (test item (el p i))
		   (cond (from-end (push (hd p i) inds)) ((setq indt (cons (hd p i) nil)) (collect indt inds indp)))
		   (when count (decf cnt)))
		 (remove-loop (if jj i (+ i j)) (cdr p)))))
      (remove-loop (if from-end (1- end) start) (when l (or r s)))
      (unless from-end (collect indl inds indp)))
    (unless (cdr inds) (return-from remove seq))
    (cond ((listp seq)
	   (let (w r rp)
	     (dolist (ind inds r)
	       (declare (proper-list ind));FIXME
	       (do ((q (if w (cdr w) seq) (cdr q))) ((eq q (or ind q)) (unless ind (collect q r rp)) (setq w ind))
		 (collect (cons (car q) nil) r rp)))))
	  ((let* ((q (make-array (- lsa (1- (length inds))) :element-type (array-element-type s))))
	     (do* ((inds inds (cdr inds))(n -1 nn)(nn (car inds) (car inds))(k 0 (1+ k)))((not inds) q)
	       (declare (seqind nn k));FIXME
	       (set-array-n q (- n (1- k)) seq (1+ n) (- nn n))))))))

(defseq delete ((item) seq :count t)
  (let* ((indl (cons (unless l lsa) nil))(inds (when from-end indl)) indp indt)
    (declare (dynamic-extent inds indl indt))
    (labels ((delete-loop (i p)
	       (unless (or (< i start) (>= i end) (when l (endp p)) (<= cnt 0))
		 (when (test item (el p i))
		   (cond (from-end (push (hd p i) inds)) ((setq indt (cons (hd p i) nil)) (collect indt inds indp)))
		   (when count (decf cnt)))
		 (delete-loop (if jj i (+ i j)) (cdr p)))))
      (delete-loop (if from-end (1- end) start) (when l (or r s)))
      (unless from-end (collect indl inds indp)))
    (unless (cdr inds) (return-from delete seq))
    (cond ((listp seq)
	   (let (w r rp)
	     (dolist (ind inds r)
	       (declare (proper-list ind));FIXME
	       (do ((q (if w (cdr w) seq) (cdr q))) ((eq q (or ind q)) (unless ind (collect q r rp)) (setq w ind))
		 (collect q r rp)))))
	  ((let* ((lq (- lsa (1- (length inds))))
		  (q (if (array-has-fill-pointer-p seq) seq (make-array lq :element-type (array-element-type s)))))
	     (do* ((inds inds (cdr inds))(n -1 nn)(nn (car inds) (car inds))(k 0 (1+ k)))((not inds) (when (eq seq q) (setf (fill-pointer q) lq)) q)
	       (declare (seqind nn k));FIXME
	       (set-array-n q (- n (1- k)) seq (1+ n) (- nn n))))))))


(defseq nsubstitute ((new item) seq :count t)
  (labels ((nsubstitute-loop (i p)
	     (if (or (< i start) (>= i end) (when l (endp p)) (<= cnt 0)) seq
		 (progn (when (test item (el p i))
			  (cond (l (setf (car (hd p i)) new))((setf (aref seq i) new)))
			  (when count (decf cnt)))
			(nsubstitute-loop (if jj i (+ i j)) (cdr p))))))
    (nsubstitute-loop (if from-end (1- end) start) (when l (or r s)))))


(defseq substitute ((new item) seq :count t)
  (let* ((indl (cons (unless l lsa) nil))(inds (when from-end indl)) indp indt)
    (declare (dynamic-extent inds indl indt))
    (labels ((substitute-loop (i p)
	       (unless (or (< i start) (>= i end) (when l (endp p)) (<= cnt 0))
		 (when (test item (el p i))
		   (cond (from-end (push (hd p i) inds)) ((setq indt (cons (hd p i) nil)) (collect indt inds indp)))
		   (when count (decf cnt)))
		 (substitute-loop (if jj i (+ i j)) (cdr p)))))
      (substitute-loop (if from-end (1- end) start) (when l (or r s)))
      (unless from-end (collect indl inds indp)))
    (unless (cdr inds) (return-from substitute seq))
    (cond ((listp seq)
	   (let (w r rp)
	     (dolist (ind inds r)
	       (declare (proper-list ind));FIXME
	       (do ((q (if w (cdr w) seq) (cdr q))) ((eq q (or ind q)) (collect (if ind (cons new nil) q) r rp) (setq w ind))
		 (collect (cons (car q) nil) r rp)))))
	  ((let* ((q (make-array lsa :element-type (array-element-type s))))
	     (do* ((inds inds (cdr inds))(n -1 nn)(nn (car inds) (car inds)))((not inds) q)
	       (declare (seqind nn));FIXME
	       (set-array-n q (1+ n) seq (1+ n) (- nn n))
	       (when (cdr inds) (setf (aref q nn) new))))))))


(defseq remove-duplicates (nil seq)
  (let ((e (if l (- end start) end))(st (if l 0 start)))
    (declare (seqbnd e st));FIXME
    (remove-if (lambda (x)
		 (position x (if (unless from-end l) (setq e (1- e) s (cdr s)) s)
			   :start (if (or l from-end) st (incf st))
			   :end (if from-end (decf e) e)
			   :test (lambda (x y) (test (key x) y))))
	       seq :start start :end end :from-end from-end)))

(defseq delete-duplicates (nil seq)
  (let ((e (if l (- end start) end))(st (if l 0 start)))
    (declare (seqbnd e st));FIXME
    (delete-if (lambda (x)
		 (position x (if (unless from-end l) (setq e (1- e) s (cdr s)) s)
			   :start (if (or l from-end) st (incf st))
			   :end (if from-end (decf e) e)
			   :test (lambda (x y) (test (key x) y))))
	       seq :start start :end end :from-end from-end)))


(defun reduce (fd s &key key from-end (start 0) end (initial-value nil ivp) 
		&aux (kf (when key (coerce key 'function)))(f (coerce fd 'function))
		  (l (listp s))(e (or end (if l (1- array-dimension-limit) (length s)))))
  (declare (optimize (safety 1)))
  (check-type fd function-designator)
  (check-type s sequence)
  (check-type key (or null function-designator))
  (check-type start seqind)
  (check-type end (or null seqind))
  (labels ((k (s i &aux (z (if l (car s) (aref s i)))) (if kf (funcall kf z) z))
	   (fc (r k) (values (funcall f (if from-end k r) (if from-end r k))))
	   (rl (s i res)
	     (cond ((or (>= i e) (when l (endp s))) res)
		   (from-end (fc (rl (if l (cdr s) s) (1+ i) (if ivp res (k s i))) (if ivp (k s i) res)))
		   ((rl (if l (cdr s) s) (1+ i) (fc res (k s i)))))))
    (let ((s (if l (nthcdr start s) s)))
      (cond (ivp (rl s start initial-value))
	    ((or (>= start e) (when l (endp s))) (values (funcall f)))
	    ((rl (if l (cdr s) s) (1+ start) (k s start)))))))

(defun every (pred seq &rest seqs &aux (pred (coerce pred 'function)))
  (declare (optimize (safety 1))(dynamic-extent seqs))
  (check-type pred function-designator)
  (check-type seq proper-sequence)
  (apply 'map nil (lambda (x &rest r) (unless (apply pred x r) (return-from every nil))) seq seqs)
  t)

(defun some (pred seq &rest seqs &aux (pred (coerce pred 'function)))
  (declare (optimize (safety 1))(dynamic-extent seqs))
  (check-type pred function-designator)
  (check-type seq proper-sequence)
  (apply 'map nil (lambda (x &rest r &aux (v (apply pred x r))) (when v (return-from some v))) seq seqs))

(defun notevery (pred seq &rest seqs)
  (declare (optimize (safety 1))(dynamic-extent seqs))
  (check-type pred function-designator)
  (check-type seq proper-sequence)
  (not (apply 'every pred seq seqs)))

(defun notany (pred seq &rest seqs)
  (declare (optimize (safety 1))(dynamic-extent seqs))
  (check-type pred function-designator)
  (check-type seq proper-sequence)
  (not (apply 'some pred seq seqs)))


(defun seqtype (sequence)
  (cond ((listp sequence) 'list)
        ((stringp sequence) 'string)
        ((bit-vector-p sequence) 'bit-vector)
        ((vectorp sequence) (list 'vector (array-element-type sequence)))
        (t (error "~S is not a sequence." sequence))))


(defun fill (sequence item &key (start 0) end)
  (declare (optimize (safety 1)))
  (check-type sequence proper-sequence)
  (check-type start (or null seqind))
  (check-type end (or null seqind))
  (nsubstitute-if item (lambda (x) (declare (ignore x)) t) sequence :start start :end end))

(defun replace (s1 s2 &key (start1 0) end1 (start2 0) end2 &aux (os1 s1) s3)
  (declare (optimize (safety 1))(notinline make-list)(dynamic-extent s3))
  (check-type s1 sequence)
  (check-type s2 sequence)
  (check-type start1 seqind)
  (check-type start2 seqind)
  (check-type end1 (or null seqind))
  (check-type end2 (or null seqind))
  (let* ((lp1 (listp s1)) (lp2 (listp s2))
	 (e1 (or end1 (if lp1 (1- array-dimension-limit) (length s1))))
	 (e2 (or end2 (if lp2 (1- array-dimension-limit) (length s2)))))
    (if (unless (or lp1 lp2) (eq (array-element-type s1) (array-element-type s2)))
	(set-array-n s1 start1 s2 start2 (min (- e1 start1) (- e2 start2)))
	(progn
	  (when (and (eq s1 s2) (> start1 start2))
	    (setq s3 (make-list (length s2)) s2 (replace s3 s2) lp2 t e2 (1- array-dimension-limit)))
	  (do ((i1 start1 (1+ i1))(i2 start2 (1+ i2))
	       (s1 (if lp1 (nthcdr start1 s1) s1) (if lp1 (cdr s1) s1))
	       (s2 (if lp2 (nthcdr start2 s2) s2) (if lp2 (cdr s2) s2)))
	      ((or (not s1) (>= i1 e1) (not s2) (>= i2 e2)) os1)
	    (let ((e2 (if lp2 (car s2) (aref s2 i2))))
	      (if lp1 (setf (car s1) e2) (setf (aref s1 i1) e2))))))))



(defseq mismatch (nil (s1 s2))
  (let* ((s2 (or r s))(i2 (if from-end (1- end2) start2))(j (if from-end -1 1)))
    (or (let ((x (position-if-not
		  (lambda (x)
		    (unless (or (< i2 start2) (>= i2 end2) (when l (endp s2)))
		      (let ((el (el s2 i2)))
			(incf i2 j)(setq s2 (if l (cdr s2) s2))
			(test (key x) el))))
		  s1 :from-end from-end :start start1 :end end1)))
	  (when x (if from-end (1+ x) x)))
	(unless (or (< i2 start2) (>= i2 end2) (when l (endp s2)))
	    (if from-end start1 (let ((ln1 (length s1))) (if end1 (min end1 ln1) ln1)))))))

(defun nonregexp-string-p (str s e)
  (when (and (stringp str) (zerop s) (if e (eql e (length str)) t));FIXME frame
    (map nil (lambda (x) (case (char-code x) (#.(mapcar 'char-code (coerce "\\^$.|?*+()[]{}" 'list)) (return-from nonregexp-string-p nil)))) str)
    t))
(declaim (inline nonregexp-string-p))

(defseq search (nil (s1 s2));consider (position-if-not 'eql-is-eq s1 :start start1 :end end1)
  (unless (or test test-not key from-end)
    (when (and (stringp s2) (nonregexp-string-p s1 start1 end1))
      (let ((x (string-match s1 s2 start2 end2)))
	(return-from search (unless (minusp x) x)))))
  (let ((n (max 0 (- (or end1 (length s1)) start1))))
    (do ((p (when l (if from-end (nthcdr (max 0 (1- n)) r) s)) (cdr p))
	 (i (if from-end (- end2 n) start2) (if (>= i end2) (return nil) (+ i (if from-end -1 1)))));keep i seqbnd
	((or (< i start2) (> i (- end2 n)) (when l (endp p))))
      (unless (mismatch s1 (or (if l (if from-end (car p) p)) s2)
			:test (lambda (x y) (test (key x) y))
			:start1 start1 :start2 (if p 0 i) :end1 end1 :end2 (if p n (+ i n)))
	(return i)))))

(eval-when (compile eval)

  (defmacro mrotatef (a b &aux (s (sgen "MRF-S"))) `(let ((,s ,a)) (setf ,a ,b ,b ,s)))

  (defmacro raref (a seq i j l)
    `(if ,l
	 (mrotatef (car (aref ,a ,i)) (car (aref ,a ,j)))
	 (set-array ,seq ,i ,seq ,j t)))

  (defmacro garef (a seq i l) `(if ,l (car (aref ,a ,i)) (aref ,seq ,i))))

(defun sort (seq pred &key (key 'identity))
  (declare (optimize (safety 1)))
  (check-type seq proper-sequence)
  (let* ((ll (length seq))
	 (list (listp seq))
	 (a (when list (make-array ll))))
    (when list
      (do ((fi 0 (1+ fi)) (l seq (cdr l))) ((>= fi ll)) (setf (aref a fi) l)))
    (do ((ii (list ll 0))) ((not ii) seq)
	(declare (dynamic-extent ii))
	(let* ((ls (pop ii)) (fi (pop ii)))
	  (declare (seqind ls fi))
	  (do nil ((>= fi (1- ls)))
	    (let* ((spi (+ fi (random (- ls fi))))
		   (sp (garef a seq spi list))
		   (sp (if key (funcall key sp) sp)))
	      (raref a seq fi spi list)
	      (do ((lf fi) (rt ls)) ((>= lf rt))
		(declare (seqind lf rt));FIXME
		(do ((q t)) 
		    ((or (>= (if q (incf lf) lf) (if q rt (decf rt)))
			 (let* ((f (garef a seq (if q lf rt) list))
				(f (if key (funcall key f) f)))
			   (and (not (funcall pred (if q f sp) (if q sp f)))
				(setq q (not q)))))))
		(let* ((r (< lf rt))
		       (f (if r lf fi))
		       (s (if r rt (setq spi (1- lf)))))
		  (raref a seq f s list)))
	      (let* ((ospi (1+ spi))
		     (b   (< (- ls ospi) (- spi fi)))
		     (lf  (if b ospi 0))
		     (rt  (if b 0 spi))
		     (b1  (if b (> (- ls lf) 1) (> (- rt fi) 1)))
		     (ns  (if b lf fi))
		     (ns1 (if b ls rt))
		     (nls (if b spi ls))
		     (nfi (if b fi ospi)))
		(when b1
		  (push ns ii) (push ns1 ii))
		(setq ls nls fi nfi))))))))


(defun stable-sort (sequence predicate &key key)
  (declare (optimize (safety 1)))
  (check-type sequence proper-sequence)
  (typecase 
   sequence
   (list (list-merge-sort sequence predicate key))
   (string (sort sequence predicate :key key))
   (bit-vector (sort sequence predicate :key key))
   (otherwise 
    (replace sequence (list-merge-sort (coerce sequence 'list) predicate key)))))

(eval-when (compile eval)
  (defmacro f+ (x y) `(the fixnum (+ (the fixnum ,x) (the fixnum ,y))))
  (defmacro f- (x y) `(the fixnum (- (the fixnum ,x) (the fixnum ,y)))))

(defun merge (result-type sequence1 sequence2 predicate
	      &key (key #'identity)
	      &aux (l1 (length sequence1)) (l2 (length sequence2)))
  (declare (optimize (safety 1)))
  (declare (fixnum l1 l2))
  (when (equal key 'nil) (setq key #'identity))
  (do ((newseq (make-sequence result-type (the fixnum (f+ l1 l2))))
       (j 0 (f+ 1  j))
       (i1 0)
       (i2 0))
      ((and (= i1 l1) (= i2 l2)) newseq)
    (declare (fixnum j i1 i2))
    (cond ((and (< i1 l1) (< i2 l2))
	   (cond ((funcall predicate
			   (funcall key (elt sequence1 i1))
			   (funcall key (elt sequence2 i2)))
		  (setf (elt newseq j) (elt sequence1 i1))
		  (setf  i1 (f+ 1  i1)))
		 ((funcall predicate
			   (funcall key (elt sequence2 i2))
			   (funcall key (elt sequence1 i1)))
		  (setf (elt newseq j) (elt sequence2 i2))
		  (setf  i2 (f+ 1  i2)))
		 (t
		  (setf (elt newseq j) (elt sequence1 i1))
		  (setf  i1 (f+ 1  i1)))))
          ((< i1 l1)
	   (setf (elt newseq j) (elt sequence1 i1))
	   (setf  i1 (f+ 1  i1)))
	  (t
	   (setf (elt newseq j) (elt sequence2 i2))
	   (setf  i2 (f+ 1  i2))))))

(defmacro with-hash-table-iterator ((name hash-table) &body body)
  (declare (optimize (safety 1)))
  (let ((table (sgen))
	(ind (sgen))
	(size (sgen)))
    `(let* ((,table ,hash-table)
	    (,ind -1)
	    (,size (1- (hash-table-size ,table))))
       (macrolet ((,name nil
			 `(do nil ((>= ,',ind ,',size))
			      (let* ((e (hashtable-self ,',table (incf ,',ind)))
				     (k (htent-key e)))
				(unless (eql +objnull+ k)
				  (return (values t (nani k) (htent-value e))))))))
		 ,@body))))
		 

(defun copy-seq (s) 
  (declare (optimize (safety 1)))
  (check-type s sequence)
  (if (listp s)
      (copy-list s)
    (let* ((n (length s))
	   (o (make-array n :element-type (array-element-type s))))
      (set-array-n o 0 s 0 n))))

