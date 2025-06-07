;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defun in-package-internal (n &aux (p (find-package n)))
  (or (when p (setq *package* p))
      (progn
	(restart-case (error 'package-error :package n)
	  (store-value (v)
	    :report (lambda (s) (format s "Supply a new package name"))
	    :interactive read-evaluated-form
	    (setq n v)))
	(in-package-internal n))))

(defmacro in-package (name)
  `(in-package-internal ',name))


;FIXME called from C
(defun pprint-insert-conditional-newlines (st)
  (if (>= (string-match #v"[^\n\r ] +" st) 0)
      (concatenate
       'string
       (subseq st 0 (match-end 0))
       "~:_"
       (pprint-insert-conditional-newlines (subseq st (match-end 0))))
      st))

;FIXME called from C
(defun pprint-check-format-string (st)
  (let ((j (>= (string-match #v"~>" st) 0))
	(pp (>= (string-match #v"~:@?>|~:?@?_|~[0-9]*:?I|~[0-9]+,[0-9]+:?@?T]|~:?@?W" st) 0)))
    (assert (not (and j pp)))
    j))

;FIXME called from C
(defun pprint-quit (x h s count)
  (cond
    ((or (and x (atom x)) (and *print-circle* h (gethash x h)))
     (when (>= count 0) (write-string ". " s))
     (write x :stream s)
     t)
    ((and *print-length* (>= count *print-length* 0))
     (write-string "..." s)
     t)
    ((and (< count 0) *print-level* (> *prin-level* *print-level*))
     (write-string "#" s)
     t)))

(defmacro pprint-logical-block ((s x &key (prefix "") (per-line-prefix "") (suffix ""))
				&body body &aux (count (gensym)))
  (declare (optimize (safety 1)))
  `(let* ((*print-line-prefix* ,per-line-prefix)(*prin-level* (1+ *prin-level*)))
     (check-type *print-line-prefix* string)
     (flet ((do-pref (x h)
	      (if (pprint-quit x h ,s -1)
		  (return-from do-pref nil)
		  (write-string ,prefix ,s)))
	    (do-suf (x h) (declare (ignore x h)) (write-string ,suffix ,s));FIXME
	    (do-pprint (x h &aux (,count 0))
	      (macrolet
		  ((pprint-pop nil
		     '(if (pprint-quit x h ,s ,count)
		       (return-from do-pprint nil)
		       (progn (incf ,count)(pop x))))
		   (pprint-exit-if-list-exhausted nil
		     '(unless x (return-from do-pprint nil))))
		,@body)))
       (write-int1 ,x ,s #'do-pprint #'do-pref #'do-suf))))

(defun pprint-fill (s list &optional (colon-p t) at-sign-p)
  (declare (ignore at-sign-p))
  (unless (listp list) (setq colon-p nil))
  (pprint-logical-block (s list :prefix (if colon-p "(" "")
                                :suffix (if colon-p ")" ""))
    (pprint-exit-if-list-exhausted)
    (loop (write (pprint-pop) :stream s)
          (pprint-exit-if-list-exhausted)
          (write-char #\Space s)
          (pprint-newline :fill s))))

(defun pprint-tabular (s list &optional (colon-p t) at-sign-p (tabsize nil))
  (declare (ignore at-sign-p))
  (when (null tabsize) (setq tabsize 16))
  (pprint-logical-block (s list :prefix (if colon-p "(" "")
                                :suffix (if colon-p ")" ""))
    (pprint-exit-if-list-exhausted)
    (loop
     (write (pprint-pop) :stream s)
     (pprint-exit-if-list-exhausted)
     (write-char #\Space s)
     (pprint-tab :section-relative 0 tabsize s)
     (pprint-newline :fill s))))

(defun pprint-linear (s list &optional (colon-p t) at-sign-p)
  (declare (ignore at-sign-p))
  (unless (listp list) (setq colon-p nil))
  (pprint-logical-block (s list :prefix (if colon-p "(" "")
                                :suffix (if colon-p ")" ""))
    (pprint-exit-if-list-exhausted)
    (loop (write (pprint-pop) :stream s)
          (pprint-exit-if-list-exhausted)
          (write-char #\Space s)
     (pprint-newline :linear s))))

(defun coerce-to-stream (strm)
  (case strm ((nil) *standard-output*) ((t) *terminal-io*)(otherwise strm)))

(defun pprint-tab (kind colnum colinc &optional strm)
  (declare (optimize (safety 1)))
  (check-type kind (member :line :section :line-relative :section-relative))
  (check-type colnum (integer 0))
  (check-type colinc (integer 0))
  (check-type strm (or boolean stream));FIXME output-stream
  (when *print-pretty*
    (pprint-queue-codes (coerce-to-stream strm) (get kind 'fixnum) colnum colinc)))


(defun pprint-indent (kind n &optional stream)
  (declare (optimize (safety 1)))
  (check-type kind (member :current :block))
  (check-type n real)
  (check-type stream (or boolean stream))
  (when *print-pretty*
    (let* ((stream (coerce-to-stream stream)))
      (unless (pprint-miser-style stream)
	(pprint-queue-codes stream (get kind 'fixnum) (round n))))))

	   
(defun pprint-newline (kind &optional stream)
  (declare (optimize (safety 1)))
  (check-type kind (member :linear :miser :fill :mandatory))
  (check-type stream (or boolean stream))
  (when *print-pretty*
    (let ((stream (coerce-to-stream stream)))
      (pprint-queue-codes
       stream
       (get (case kind
	      (:miser (if (pprint-miser-style stream) :linear (return-from pprint-newline nil)))
	      (:fill (if (pprint-miser-style stream) :linear kind))
	      (otherwise kind))
	    'fixnum)))))

(defvar *print-pprint-dispatch* (list nil))

(defun pprint-make-dispatch (table)
  `(lambda (x)
     (typecase x
       ,@(mapcar (lambda (x) `(,(car x) (values ',(cadr x) t)))
		 table)
       (otherwise (values nil nil)))))

(defun set-pprint-dispatch (spec fun &optional (pri 0) (tab *print-pprint-dispatch*)
			    &aux (x (assoc spec (car tab) :test 'equal)))
  (declare (optimize (safety 1)))
  (check-type spec type-spec)
  (check-type fun (or null function-name function))
  (check-type pri real)
  (check-type tab (cons list (or null function)))
	      
  (if x
      (setf (cadr x) fun (caddr x) pri)
      (push (list spec fun pri) (car tab)))
  (sort (car tab) '> :key 'caddr)
  (setf (cdr tab) (compile nil (pprint-make-dispatch (car tab))))
  nil)

(defun pprint-dispatch (obj &optional (table *print-pprint-dispatch*))
  (declare (optimize (safety 1)))
  (check-type table (cons list (or null function)))
  (if (cdr table)
      (funcall (cdr table) obj)
      (values nil nil)))

(defun copy-pprint-dispatch (&optional (tab *print-pprint-dispatch*))
  (declare (optimize (safety 1)))
  (check-type tab (or null (cons list (or null function))))
  (cons (car tab) (cdr tab)))
