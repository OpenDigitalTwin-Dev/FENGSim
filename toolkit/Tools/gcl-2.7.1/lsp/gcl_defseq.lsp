(in-package :si)

(defmacro defseq (n (il seq &key count list (+ifnot il) nokey) &body body
		  &aux (ts (sgen))(tsc (sgen))(kf (sgen))
		    (q2 (listp seq))(seq1 (when q2 (car seq)))(seq (if q2 (cadr seq) seq))
		    (st (if list 'proper-list 'proper-sequence))
		    (start (unless list (if q2 'start2 'start)))(end (unless list (if q2 'end2 'end)))
		    (keys `(&key ,@(unless nokey '(key))
				 ,@(unless list `(from-end (,start 0) ,end
							   ,@(when q2 `((start1 0) end1))
							   ,@(when count `(count)))))))
   `(progn
      (defun ,n (,@il ,@(when q2 (list seq1)) ,seq ,@keys test test-not
		 &aux (,ts (coerce (or test test-not #'eql) 'function))
		   (,tsc (cond ((eq ,ts #'eq) 0)
			       ((eq ,ts #'eql) 1)
			       ((eq ,ts #'equal) 2)
			       ((eq ,ts #'equalp) 3)
			       ((eq ,ts #'funcall) 4)
			       (5)))
		   ,@(unless nokey `((,kf (when key (coerce key 'function)))
				     (,kf (unless (eq ,kf #'identity) ,kf))))
		   ,@(unless list `((l (listp ,seq)))))
	(declare (optimize (safety 1)))
	,@(unless (case list ((tree tree2) t)) `((check-type ,seq ,st)))
	,@(when q2 (unless (eq list 'tree2) `((check-type ,seq1 ,st))))
	(check-type test (or null function-designator))
	(check-type test-not (or null function-designator))
	,@(unless nokey `((check-type key (or null function-designator))))
	,@(unless list
	    `((check-type ,start seqind)
	      (check-type ,end (or null seqbnd))
	      ,@(when q2 `((check-type start1 seqind)
			   (check-type end1 (or null seqbnd))))))
	,@(when count `((check-type count (or null integer))))
	(and test test-not (error "both test and test not supplied"))
	(let* ,(unless list
		 `((lsa (if l (1- array-dimension-limit) (length ,seq)))
		   (jj (unless ,end l))(j (if from-end -1 1))(,end (or ,end lsa))
		   ,@(when count `((cnt (or count (1- array-dimension-limit)))
				   (cnt (min (1- array-dimension-limit) (max 0 cnt)))))
		   r (s (if l (nthcdr ,start ,seq) ,seq))))
	  ,@(unless list
	      `((declare (dynamic-extent r)(ignorable j jj))
		(when (and l from-end)
		  (do ((p s (cdr p))(i ,start (1+ i))) ((or (when l (endp p)) (>= i ,end)) (setq ,end (min ,end i)))
		    (push p r)))))
	  (labels (,@(unless list
		       `((el (p i) (if l (if from-end (caar p) (car p)) (aref ,seq i)))
			 (hd (p i) (if l (if from-end (car p) p) i))))
		   ,@(unless nokey `((key (x) (if ,kf (funcall ,kf x) x))))
                     (test-no-key (x y)
		       (if (case ,tsc
			     (0 (eq x y))
			     (1 (eql x y))
			     (2 (equal x y))
			     (3 (equalp x y))
			     (4 (funcall x y))
			     (otherwise (funcall ,ts x y)))
			   (not test-not) test-not))
		     (test (x y) (test-no-key x ,(if nokey 'y '(key y)))))
	    (declare (ignorable #'test ,@(unless list `(#'el #'hd))))
	    (when (case ,tsc
		    ((1 2 3)
		     (or ,@(unless list
			     `((unless l (array-eql-is-eq ,seq))
			       ,@(when seq1 `((unless (listp ,seq1) (array-eql-is-eq ,seq1))))))
			 ,@(when il
			     (let ((i (car (last il))))
			       `((case ,tsc
				   (1 (eql-is-eq ,i))
				   (2 (equal-is-eq ,i))
				   (3 (equalp-is-eq ,i)))))))))
	      (setq ,tsc 0))
	    (macrolet ((collect (a b c) `(let ((tmp ,a)) (setq ,c (if ,c (cdr (rplacd ,c tmp)) (setq ,b tmp))))))
	      ,@body))))
      ,@(when +ifnot
	  (let* ((s (sgen))(tk (sgen))(new (when (cdr il) (list (car il))))
		 (x `(defun ,s (,@new fd ,seq ,@keys)
		      (declare (optimize (safety 1)))
		      (check-type fd function-designator)
		      ,@(unless (case list ((tree tree2) t)) `((check-type ,seq ,st)))
		      (check-type key (or null function-designator))
		      ,@(unless list
			  `((check-type ,start seqind)
			    (check-type ,end (or null seqbnd))))
		      ,@(when count `((check-type count (or null integer))))
		      (,n ,@new (coerce fd 'function) ,seq ,tk #'funcall
			  :key key ,@(unless list `(:from-end from-end :start start :end end))
			  ,@(when count `(:count count))))));try apply
	    (list (sublis `((,s . ,(intern (string-concatenate (string n) "-IF")))(,tk . :test)) x)
		  (sublis `((,s . ,(intern (string-concatenate (string n) "-IF-NOT")))(,tk . :test-not)) x))))))
