(in-package :si)

(defun lenel (x lp)
  (case x (:wild 1)(:wild-inferiors 2)(:absolute (if lp -1 0))(:relative (if lp 0 -1))
	((:unspecific nil :newest) -1)(otherwise (length x))))

(defun next-match (&optional (i 1) (k -1) (m (1- (ash (length *match-data*) -1))))
  (cond ((< k (match-beginning i) (match-end i)) i)
	((< i m) (next-match (1+ i) k m))
	(i)))

(defun mme2 (s lel lp &optional (b 0) (i (next-match)) r el
	       &aux (e (+ b (lenel (car lel) lp)))(j (match-beginning i))(k (match-end i)))
  (cond
   ((< (- b 2) j k (+ e 2))
    (let* ((z (car lel))(b1 (max b j))(e1 (min k e))
	   (z (if (or (< b b1) (< e1 e)) (subseq z (- b1 b) (- e1 b)) z))
	   (r (if el r (cons nil r))))
      (mme2 s lel lp b (next-match i k) (cons (cons z (car r)) (cdr r)) (or el (car lel)))))
   ((< (1- j) b e (1+ k))
    (let ((r (if el r (cons nil r))))
      (mme2 s (cdr lel) lp (1+ e) i (cons (cons (car lel) (car r)) (cdr r)) (or el (list (car lel))))))
   ((consp el)
    (let* ((cr (nreverse (car r))))
      (mme2 s lel lp b (next-match i k) (cons (cons (car el) (list cr)) (cdr r)))))
   (el
    (let* ((cr (nreverse (car r))))
      (mme2 s (cdr lel) lp (1+ e) i (cons (cons el cr) (cdr r)))))
   (lel (mme2 s (cdr lel) lp (1+ e) i (cons (car lel) r)))
   ((nreverse r))))

(defun do-repl (x y)
  (labels ((r (x l &optional (b 0) &aux (f (string-match #v"\\*" x b)))
	      (if (eql f -1) (if (eql b 0) x (subseq x b))
		(concatenate 'string (subseq x b f) (or (car l) "") (r x (cdr l) (1+ f))))))
    (r y x)))

(defun dir-p (x) (when (consp x) (member (car x) '(:absolute :relative))))

(defun source-portion (x y)
  (cond
   ((or (dir-p x) (dir-p y))
    (mapcan (lambda (z &aux (w (source-portion
				(if y (when (wild-dir-element-p z) (setf x (member-if 'listp x)) (pop x)) z)
				(when y z))))
   	      (if (listp w) w (list w))) (or y x)))
   ((if y (eq y :wild-inferiors) t) (if (listp x) (if (listp (cadr x)) (cadr x) (car x)) x));(or  y)
   ((eq y :wild) (if (listp x) (car x) x));(or  y)
   ((stringp y) (do-repl (when (listp x) (unless (listp (cadr x)) (cdr x))) y))
   (y)))

(defun list-toggle-case (x f)
  (typecase x
    (string (values (funcall f x)))
    (cons (mapcar (lambda (x) (list-toggle-case x f)) x))
    (otherwise x)))

(defun mme3 (sx px flp tlp)
  (list-toggle-case
   (lnp (mme2 sx (pnl1 (mlp px)) flp))
   (cond ((eq flp tlp) 'identity)
	 (flp 'string-downcase)
	 (tlp 'string-upcase))))

(defun translate-pathname (source from to &key
				  &aux (psource (pathname source))
				  (pto (pathname to))
				  (match (pathname-match-p source from)))
  (declare (optimize (safety 1)))
  (check-type source pathname-designator)
  (check-type from pathname-designator)
  (check-type to pathname-designator)
  (check-type match (not null))
  (apply 'make-pathname :host (pathname-host pto) :device (pathname-device pto)
	 (mapcan 'list +pathname-keys+
		 (mapcar 'source-portion
			 (mme3 (namestring source) psource (typep psource 'logical-pathname) (typep pto 'logical-pathname))
			 (mlp pto)))))

(defun translate-logical-pathname (spec &key &aux (p (pathname spec)))
  (declare (optimize (safety 1)))
  (check-type spec pathname-designator)
  (typecase p
    (logical-pathname
     (let ((rules (assoc p (logical-pathname-translations (pathname-host p)) :test 'pathname-match-p)))
       (unless rules
	 (error 'file-error :pathname p :format-control "No matching translations"))
       (translate-logical-pathname (apply 'translate-pathname p rules))))
    (otherwise p)))
    
