;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defun match-beginning (i &aux (v *match-data*))
  (declare ((vector fixnum) v)(seqind i))
  (the (or (integer -1 -1 ) seqind) (aref v i)))
(defun match-end (i &aux (v *match-data*))
  (declare ((vector fixnum) v)(seqind i))
  (the (or (integer -1 -1 ) seqind) (aref v (+ i (ash (length v) -1)))))

(declaim (inline match-beginning match-end))

(defun dir-conj (x) (if (eq x :relative) :absolute :relative))

(defvar *up-key* :up)

(defun element (x b i key &optional def)
  (let* ((z (if (> i b) (subseq x b i) def));(make-array (- i b) :element-type 'character :displaced-to x :displaced-index-offset b)
	 (w (assoc key '((:host . nil)
			 (:device . nil)
			 (:directory . ((".." . :up)("*" . :wild)("**" . :wild-inferiors)))
			 (:name . (("*" . :wild)))
			 (:type . (("*" . :wild)))
			 (:version . (("*" . :wild)("NEWEST" . :newest))))))
	 (w (assoc z (cdr w) :test 'string-equal))
	 (z (if w (cdr w) z)))
    (if (eq z :up) *up-key* z)))

(defun dir-parse (x sep sepfirst &optional (b 0))
  (when (stringp x)
    (let ((i (position sep x :start b)));string-match spoils outer match results
      (when i
	(let* ((y (dir-parse x sep sepfirst (1+ i)))
	       (z (element x b i :directory))
	       (y (if z (cons z y) y)))
	  (if (zerop b)
	      (cons (if (zerop i) sepfirst (dir-conj sepfirst)) y)
	    y))))))

(defun match-component (x i k &optional (boff 0) (eoff 0))
  (element x (+ (match-beginning i) boff) (+ (match-end i) eoff) k))

(defun version-parse (x)
  (typecase x
    (string (when (plusp (length x)) (version-parse (parse-integer x))))
    (otherwise x)))

(defconstant +generic-logical-pathname-regexp+ (compile-regexp (to-regexp-or-namestring (make-list (length +logical-pathname-defaults+)) t t)))

(defun logical-pathname-parse (x &optional host def (b 0) (e (length x)) &aux (x (string-upcase x)))
  (when (and (eql b (string-match +generic-logical-pathname-regexp+ x b e)) (eql (match-end 0) e))
    (let ((mhost (match-component x 1 :host 0 -1)))
      (when (and host mhost)
	(unless (string-equal host mhost)
	    (error 'error :format-control "Host part of ~s does not match ~s" :format-arguments (list x host))))
      (let ((host (or host mhost (pathname-host def))))
	(when (logical-pathname-host-p host)
	  (let* ((dir (dir-parse (match-component x 2 :none) #\; :relative))
		 (edir (expand-home-dir dir)))
	  (make-pathname :host host
			 :device :unspecific
			 :directory edir
			 :name (match-component x 6 :name)
			 :type (match-component x 8 :type 1)
			 :version (version-parse (match-component x 11 :version 1))
			 :namestring (when (and mhost (eql b 0) (eql e (length x)) (eq dir edir)) x))))))))

(defconstant +generic-physical-pathname-regexp+ (compile-regexp (to-regexp-or-namestring (make-list (length +physical-pathname-defaults+)) t nil)))

(defun expand-home-dir (dir)
  (if (and (eq (car dir) :relative) (stringp (cadr dir)) (eql #\~ (aref (cadr dir) 0)))
      (append (dir-parse (home-namestring (cadr dir)) #\/ :absolute) (cddr dir))
    dir))

(defun pathname-parse (x b e)
  (when (and (eql b (string-match +generic-physical-pathname-regexp+ x b e)) (eql (match-end 0) e))
    (let* ((dir (dir-parse (match-component x 1 :none) #\/ :absolute))
	   (edir (expand-home-dir dir)))
      (make-pathname :directory edir
		     :name (match-component x 3 :name)
		     :type (match-component x 4 :type 1)
		     :namestring (when (and (eql b 0) (eql e (length x)) (eq dir edir)) x)))))

(defun path-stream-name (x)
  (check-type x pathname-designator)
  (typecase x
    (synonym-stream (path-stream-name (symbol-value (synonym-stream-symbol x))))
    (stream (path-stream-name (c-stream-object1 x)))
    (otherwise x)))

(defun parse-namestring (thing &optional host (default-pathname *default-pathname-defaults*) &rest r &key (start 0) end junk-allowed)
  (declare (optimize (safety 1))(dynamic-extent r))
  (check-type thing pathname-designator)
  (check-type host (or null (satisfies logical-pathname-translations)))
  (check-type default-pathname pathname-designator)
  (check-type start seqind)
  (check-type end (or null seqind))
  
  (typecase thing
    (string (let* ((e (or end (length thing)))
		   (l (logical-pathname-parse thing host default-pathname start e))
		   (l (or l (unless host (pathname-parse thing start e)))))
	      (cond (junk-allowed (values l (max 0 (match-end 0))))
		    (l (values l e))
		    ((error 'parse-error :format-control "~s is not a valid pathname on host ~s" :format-arguments (list thing host))))))
    (stream (apply 'parse-namestring (path-stream-name thing) host default-pathname r))
    (pathname
     (when host
       (unless (string-equal host (pathname-host thing))
	 (error 'file-error :pathname thing :format-control "Host does not match ~s" :format-arguments (list host))))
     (values thing start))))

(defun pathname (spec)
  (declare (optimize (safety 1)))
  (check-type spec pathname-designator)
  (if (typep spec 'pathname) spec (values (parse-namestring spec))))

(defun sharp-p-reader (stream subchar arg)
  (declare (ignore subchar arg))
  (let ((x (parse-namestring (read stream)))) x))

(defun sharp-dq-reader (stream subchar arg);FIXME arg && read-suppress
  (declare (ignore subchar arg))
  (unread-char #\" stream)
  (let ((x (parse-namestring (read stream)))) x))

(set-dispatch-macro-character #\# #\p 'sharp-p-reader)
(set-dispatch-macro-character #\# #\P 'sharp-p-reader)
(set-dispatch-macro-character #\# #\" 'sharp-dq-reader)
