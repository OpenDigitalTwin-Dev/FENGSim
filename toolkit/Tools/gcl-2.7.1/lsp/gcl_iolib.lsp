;; -*-Lisp-*-
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


;;;;   iolib.lsp
;;;;
;;;;        The IO library.


(in-package :si)

(defun concatenated-stream-streams (stream)
  (declare (optimize (safety 2)))
  (check-type stream concatenated-stream)
  (c-stream-object0 stream))
(defun broadcast-stream-streams (stream)
  (declare (optimize (safety 2)))
  (check-type stream broadcast-stream)
  (c-stream-object0 stream))
(defun two-way-stream-input-stream (stream)
  (declare (optimize (safety 2)))
  (check-type stream two-way-stream)
  (c-stream-object0 stream))
(defun echo-stream-input-stream (stream)
  (declare (optimize (safety 2)))
  (check-type stream echo-stream)
  (c-stream-object0 stream))
(defun two-way-stream-output-stream (stream)
  (declare (optimize (safety 2)))
  (check-type stream two-way-stream)
  (c-stream-object1 stream))
(defun echo-stream-output-stream (stream)
  (declare (optimize (safety 2)))
  (check-type stream echo-stream)
  (c-stream-object1 stream))
(defun synonym-stream-symbol (stream)
  (declare (optimize (safety 2)))
  (check-type stream synonym-stream)
  (c-stream-object0 stream))

(defun maybe-clear-input (&optional (x *standard-input*))
  (typecase x
    (synonym-stream (maybe-clear-input (symbol-value (synonym-stream-symbol x))))
    (two-way-stream (maybe-clear-input (two-way-stream-input-stream x)))
    (stream (when (terminal-input-stream-p x) (clear-input t)))))

(defmacro with-open-stream ((var stream) . body)
  (declare (optimize (safety 1)))
  (multiple-value-bind (ds b) (find-declarations body)
    `(let ((,var ,stream))
       ,@ds
       (unwind-protect
	   (progn ,@b)
         (close ,var)))))

(defun make-string-input-stream (string &optional (start 0) end)
  (declare (optimize (safety 1)))
  (check-type string string)
  (check-type start seqind)
  (check-type end (or null seqind))
  (let ((l (- (or end (length string)) start)))
    (make-string-input-stream-int
     (make-array l :element-type (array-element-type string) :displaced-to string :displaced-index-offset start :fill-pointer 0)
     0 l)))

(defun get-string-input-stream-index (stream &aux (s (c-stream-object0 stream)))
  (+ (fill-pointer s) (multiple-value-bind (a b) (array-displacement s) (declare (ignore a)) b)))

(defmacro with-input-from-string ((var string &key index (start 0) end) . body)
  (declare (optimize (safety 2)))
  (multiple-value-bind (doc decls ctps body) (parse-body-header body)
    (declare (ignore doc))
    `(let ((,var (make-string-input-stream ,string ,start ,end)))
       ,@decls
       ,@ctps
       (multiple-value-prog1
	   (progn ,@body)
	 ,@(when index
	     `((setf ,index (get-string-input-stream-index ,var))))))))



(defvar *sosm* (make-string-output-stream))

(defun get-sosm nil
  (when *sosm*
    (setf (fill-pointer (c-stream-object0 *sosm*)) 0)
    *sosm*))

(defmacro with-output-to-string ((var &optional string &key (element-type ''character)) . body)
  (declare (optimize (safety 2)))
  (multiple-value-bind (doc decls ctps body) (parse-body-header body)
    (declare (ignorable doc))
    `(let* ((,var ,(if string
		       `(progn ,element-type (make-string-output-stream-from-string ,string))
		       `(or (get-sosm) (make-string-output-stream :element-type ,element-type))))
	    (*sosm* (unless (eq ,var *sosm*) *sosm*)))
       ,@decls
       ,@ctps
       ,@body
       ,@(unless string `((get-output-stream-string ,var))))))


(defun read-from-string (string &optional (eof-error-p t) eof-value
                         &key (start 0) end preserve-whitespace)
  (declare (optimize (safety 2)))
  (check-type string string)
  (check-type start seqind)
  (check-type end (or null seqind))
  (let ((stream (make-string-input-stream string start (or end (length string)))))
    (values (if preserve-whitespace
		(read-preserving-whitespace stream eof-error-p eof-value)
	      (read stream eof-error-p eof-value))
	    (get-string-input-stream-index stream))))


(defun write (x &key stream 
		  ((:array           *print-array*)            *print-array*)
		  ((:base            *print-base*)             *print-base*)
		  ((:case            *print-case*)             *print-case*)
		  ((:circle          *print-circle*)           *print-circle*)
		  ((:escape          *print-escape*)           *print-escape*)
		  ((:gensym          *print-gensym*)           *print-gensym*)
		  ((:length          *print-length*)           *print-length*)
		  ((:level           *print-level*)            *print-level*)
		  ((:lines           *print-lines*)            *print-lines*)
		  ((:miser-width     *print-miser-width*)      *print-miser-width*)
		  ((:pprint-dispatch *print-pprint-dispatch*)  *print-pprint-dispatch*)
		  ((:pretty          *print-pretty*)           *print-pretty*)
		  ((:radix           *print-radix*)            *print-radix*)
		  ((:readably        *print-readably*)         *print-readably*)
		  ((:right-margin    *print-right-margin*)     *print-right-margin*))
  (write-int x stream))

(defun write-to-string (x &rest r &aux (stream (or (get-sosm) (make-string-output-stream)))(*sosm* nil))
  (declare (optimize (safety 1))(dynamic-extent r))
  (apply 'write x :stream stream r)
  (get-output-stream-string stream))

(defun prin1-to-string (object &aux (stream (or (get-sosm) (make-string-output-stream)))(*sosm* nil))
  (declare (optimize (safety 2)))
  (prin1 object stream)
  (get-output-stream-string stream))

(defun princ-to-string (object &aux (stream (or (get-sosm) (make-string-output-stream)))(*sosm* nil))
  (declare (optimize (safety 2)))
  (princ object stream)
  (get-output-stream-string stream))

(defun file-string-length (ostream object)
  (declare (optimize (safety 2)))
  (let ((ostream (if (typep ostream 'broadcast-stream) 
		     (car (last (broadcast-stream-streams ostream)))
		   ostream)))
    (cond ((not ostream) 1)
	  ((subtypep (stream-element-type ostream) 'character)
	   (length (let ((*print-escape* nil)) (write-to-string object)))))))

(defmacro with-temp-file ((s pn) (tmp ext) &rest body) 
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header body)
   (declare (ignore doc))
   `(let* ((,s (temp-stream ,tmp ,ext)) 
	   (,pn (stream-object1 ,s))) 
      ,@decls
      ,@ctps
      (unwind-protect (progn ,@body) (progn (close ,s) (delete-file ,s))))))

(defmacro with-open-file ((stream . filespec) . body)
  (declare (optimize (safety 1)))
  (multiple-value-bind (doc decls ctps body) (parse-body-header body)
    (declare (ignore doc))
    `(let ((,stream (open ,@filespec)))
       ,@decls
       ,@ctps
       (unwind-protect (progn ,@body)
	 (when ,stream (close ,stream))))))

(defun y-or-n-p (&optional string &rest args)
  (declare (optimize (safety 1)))
  (when string (format *query-io* "~&~?  (Y or N) " string args))
  (let ((reply (symbol-name (read *query-io*))))
    (cond ((string-equal reply "Y") t)
	  ((string-equal reply "N") nil)
	  ((apply 'y-or-n-p string args)))))

(defun yes-or-no-p (&optional string &rest args)
  (declare (optimize (safety 1)))
  (when string (format *query-io* "~&~?  (Yes or No) " string args))
  (let ((reply (symbol-name (read *query-io*))))
    (cond ((string-equal reply "YES") t)
	  ((string-equal reply "NO") nil)
	  ((apply 'yes-or-no-p string args)))))

(defun sharp-a-reader (stream subchar arg)
  (declare (ignore subchar) (optimize (safety 2)))
  (let ((initial-contents (read stream nil nil t)))
    (unless *read-suppress*
      (do ((i 0 (1+ i))
	   (d nil (cons (length ic) d))
	   (ic initial-contents (if (zerop (length ic)) ic (elt ic 0))))
	  ((>= i arg) (make-array (nreverse d) :initial-contents initial-contents))))))

(set-dispatch-macro-character #\# #\a 'sharp-a-reader)
(set-dispatch-macro-character #\# #\a 'sharp-a-reader (standard-readtable))
(set-dispatch-macro-character #\# #\A 'sharp-a-reader)
(set-dispatch-macro-character #\# #\A 'sharp-a-reader (standard-readtable))

;; defined in defstruct.lsp
(set-dispatch-macro-character #\# #\s 'sharp-s-reader)
(set-dispatch-macro-character #\# #\s 'sharp-s-reader (standard-readtable))
(set-dispatch-macro-character #\# #\S 'sharp-s-reader)
(set-dispatch-macro-character #\# #\S 'sharp-s-reader (standard-readtable))

(defvar *dribble-stream* nil)
(defvar *dribble-io* nil)
(defvar *dribble-namestring* nil)
(defvar *dribble-saved-terminal-io* nil)

(defun dribble (&optional (pathname "DRIBBLE.LOG" psp))
  (declare (optimize (safety 1)))
  (cond ((not psp)
         (when (null *dribble-stream*) (error "Not in dribble."))
         (if (eq *dribble-io* *terminal-io*)
             (setq *terminal-io* *dribble-saved-terminal-io*)
             (warn "*TERMINAL-IO* was rebound while DRIBBLE is on.~%~
                   You may miss some dribble output."))
         (close *dribble-stream*)
         (setq *dribble-stream* nil)
         (format t "~&Finished dribbling to ~A." *dribble-namestring*))
        (*dribble-stream*
         (error "Already in dribble (to ~A)." *dribble-namestring*))
        (t
         (let* ((namestring (namestring pathname))
                (stream (open pathname :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)))
           (setq *dribble-namestring* namestring
                 *dribble-stream* stream
                 *dribble-saved-terminal-io* *terminal-io*
                 *dribble-io* (make-two-way-stream
                               (make-echo-stream *terminal-io* stream)
                               (make-broadcast-stream *terminal-io* stream))
                 *terminal-io* *dribble-io*)
           (multiple-value-bind (sec min hour day month year)
               (get-decoded-time)
             (format t "~&Starts dribbling to ~A (~d/~d/~d, ~d:~d:~d)."
                     namestring year month day hour min sec))))))

; simple formatter macro

(defmacro formatter ( control-string )
  (declare (optimize (safety 2)))
  `(progn
     (lambda (*standard-output* &rest arguments)                                
       (let ((*format-unused-args* nil))
	 (apply 'format t ,control-string arguments)
	 *format-unused-args*))))

(defun stream-external-format (s)
  (declare (optimize (safety 1)))
  (check-type s stream)
  :default)



(defmacro with-standard-io-syntax (&body body)
  (declare (optimize (safety 2)))
  `(let* ((*package* (find-package :cl-user))
	  (*print-array* t)
	  (*print-base* 10)
	  (*print-case* :upcase)
	  (*print-circle* nil)
	  (*print-escape* t)
	  (*print-gensym* t)
	  (*print-length* nil)
	  (*print-level* nil)
	  (*print-lines* nil)
	  (*print-miser-width* nil)
	  (*print-pprint-dispatch* *print-pprint-dispatch*);FIXME
	  (*print-pretty* nil)
	  (*print-radix* nil)
	  (*print-readably* t)
	  (*print-right-margin* nil)
	  (*read-base* 10)
	  (*read-default-float-format* 'single-float)
	  (*read-eval* t)
	  (*read-suppress* nil)
	  (*readtable* (copy-readtable (standard-readtable))))
     ,@body))

(defmacro print-unreadable-object
	  ((object stream &key type identity) &body body)
  (declare (optimize (safety 2)))
  (let ((q `(princ " " ,stream)))
    `(if *print-readably* 
	 (error 'print-not-readable :object ,object)
       (progn
	 (princ "#<" ,stream)
	 ,@(when type `((prin1 (type-of ,object) ,stream) ,q))
	 ,@body
	 ,@(when identity
	     (let ((z `(princ (address ,object) ,stream)))
	       (if (and (not body) type) (list z) (list q z))))
	 (princ ">" ,stream)
	 nil))))
;     (print-unreadable-object-function ,object ,stream ,type ,identity ,(when body `(lambda nil ,@body)))))

(defmacro with-compile-file-syntax (&body body)
  `(let ((*print-radix* nil)
	 (*print-base* 10)
	 (*print-circle* t)
	 (*print-pretty* nil)
	 (*print-level* nil)
	 (*print-length* nil)
	 (*print-case* :downcase)
	 (*print-gensym* t)
	 (*print-array* t)
	 (*print-package* t)
	 (*print-structure* t))
     ,@body))

(defmacro with-compilation-unit (opt &rest body)   
  (declare (optimize (safety 2)))
  (declare (ignore opt)) 
  `(multiple-value-prog1
       (progn ,@body)
;     (do-recomp)
     ))


(defun restrict-stream-element-type (tp)
  (cond ((member tp '(unsigned-byte signed-byte)) tp)
	((or (member tp '(character :default)) (subtypep tp 'character)) 'character)
	((subtypep tp 'integer)
	 (let* ((ntp (tp-bnds (cmp-norm-tp tp)))
		(min (car ntp))(max (cdr ntp))
		(s (if (or (eq min '*) (< min 0)) 'signed-byte 'unsigned-byte))
		(lim (unless (or (eq min '*) (eq max '*)) (max (integer-length min) (integer-length max))))
		(lim (if (and lim (eq s 'signed-byte)) (1+ lim) lim)))
	   (if lim `(,s ,lim) s)))
	((check-type tp (member character integer)))))

(defun load-pathname-exists (z)
  (or (probe-file z)
      (when *allow-gzipped-file*
	(when (probe-file (string-concatenate (namestring z) ".gz"))
	  z))))

(defun load-pathname (p print if-does-not-exist external-format
			&aux (pp (merge-pathnames p))
			(epp (reduce (lambda (y x) (or y (load-pathname-exists (translate-pathname x "" p))))
				     '(#P".o" #P".lsp" #P".lisp" #P"") :initial-value nil)));FIXME newest?
  (if epp
      (let* ((*load-pathname* pp)(*load-truename* epp))
	(with-open-file
	 (s epp :external-format external-format)
	 (if (member (peek-char nil s nil 'eof) '#.(mapcar 'code-char (list 127 #xcf #xce #x4c #x64)))
	     (load-fasl s print)
	   (let ((*standard-input* s)) (load-stream s print)))))
    (when if-does-not-exist
      (error 'file-error :pathname pp :format-control "File does not exist."))))

(defun load (p &key (verbose *load-verbose*) (print *load-print*) (if-does-not-exist :error)
	       (external-format :default) &aux (*readtable* *readtable*)(*package* *package*))
  (declare (optimize (safety 1)))
  (check-type p (or stream pathname-designator))
  (when verbose (format t ";; Loading ~s~%" p))
  (prog1
      (typecase p
	(pathname-designator (load-pathname (pathname p) print if-does-not-exist external-format))
	(stream (load-stream p print)))
    (when verbose (format t ";; Finished loading ~s~%" p))))

(defun ensure-directories-exist (ps &key verbose &aux created)
  (declare (optimize (safety 1)))
  (check-type ps pathname-designator)
  (when (wild-pathname-p ps)
    (error 'file-error :pathname ps :format-control "Pathname is wild"))
  (labels ((d (x y &aux (z (ldiff-nf x y)) (n (namestring (make-pathname :directory z))))
	      (when (when z (stringp (car (last z))))
		(unless (eq :directory (stat n))
		  (mkdir n)
		  (setq created t)
		  (when verbose (format *standard-output* "Creating directory ~s~%" n))))
	      (when y (d x (cdr y)))))
    (let ((pd (pathname-directory ps)))
      (d pd (cdr pd)))
    (values ps created)))

(defun get-byte-stream-nchars (s)
  (let* ((tp (stream-element-type s))(ctp (cmp-norm-tp tp)))
    (labels ((ts (i) (when (<= i 32)
		       (if (tp<= ctp (cmp-norm-tp `(unsigned-byte ,(* i char-length))))
			   i (ts (1+ i))))))
      (cond ((tp<= ctp #tcharacter) 1)
	    ((ts 0))
	    (1)))))

(defun parse-integer (s &key start end (radix 10) junk-allowed)
  (declare (optimize (safety 1)))
  (parse-integer-int s start end radix junk-allowed))

(defun write-byte (j s &aux (i j))
  (declare (optimize (safety 1)))
  (check-type j integer)
  (check-type s stream)
  (dotimes (k (get-byte-stream-nchars s) j)
    (write-char (code-char (logand i #.(1- (ash 1 char-length)))) s)
    (setq i (ash i #.(- char-length)))))


(defun read-byte (s &optional (eof-error-p t) eof-value &aux (i 0))
  (declare (optimize (safety 1)))
  (check-type s stream)
  (dotimes (k (get-byte-stream-nchars s) i)
    (setq i (logior i (ash (let ((ch (read-char s eof-error-p eof-value)))
			     (if (eq ch eof-value) (return ch) (char-code ch)))
			   (* k char-length))))))


(defun read-sequence (seq strm &key (start 0) end
		      &aux (l (listp seq))(seqp (when l (nthcdr start seq)))
			(cp (eq (stream-element-type strm) 'character)))
  (declare (optimize (safety 1)));FIXME
  (check-type seq sequence)
  (check-type strm stream)
  (check-type start (integer 0))
  (check-type end (or null (integer 0)))
  (labels ((set-cons (x z) (check-type x cons) (setf (car x) z) (cdr x)))
    (the seqbnd
	 (reduce (lambda (y x &aux (z (if cp (read-char strm nil 'eof) (read-byte strm nil 'eof))))
		   (declare (seqind y)(ignorable x))
		   (when (eq z 'eof) (return-from read-sequence y))
		   (if l (setq seqp (set-cons seqp z)) (setf (aref seq y) z))
		   (1+ y))
		 seq :initial-value start :start start :end end))))


(defun write-sequence (seq strm &key (start 0) end
			   &aux (cp (eq (stream-element-type strm) 'character)))
  (declare (optimize (safety 1)))
  (check-type seq sequence)
  (check-type strm stream)
  (check-type start (integer 0))
  (check-type end (or null (integer 0)))
  (reduce (lambda (y x)
		   (declare (seqind y))
		   (if cp (write-char x strm) (write-byte x strm))
		   (1+ y))
	 seq :initial-value start :start start :end end)
  seq)

(defun open (f &key (direction :input)
	       (element-type 'character)
	       (if-exists nil iesp)
	       (if-does-not-exist nil idnesp)
	       (external-format :default) &aux (pf (pathname f)))
  (declare (optimize (safety 1)))
  (check-type f pathname-designator)
  (when (wild-pathname-p pf)
    (error 'file-error :pathname pf :format-control "Pathname is wild."))
  (let* ((s (open-int (namestring (translate-logical-pathname pf)) direction
		      (restrict-stream-element-type element-type)
		      if-exists iesp if-does-not-exist idnesp external-format)))
    (when (typep s 'stream) (c-set-stream-object1 s pf) s)))


(defun file-length (x)
  (declare (optimize (safety 1)))
  (check-type x (or broadcast-stream path-stream))
  (if (typep x 'broadcast-stream)
      (let ((s (broadcast-stream-streams x))) (if s (file-length (car (last s))) 0))
    (multiple-value-bind (tp sz) (stat x)
      (declare (ignore tp))
      (values (truncate sz (get-byte-stream-nchars x))))))

(defun file-position (x &optional (pos :start pos-p))
  (declare (optimize (safety 1)))
  (check-type x (or broadcast-stream file-stream string-stream))
  (check-type pos (or (member :start :end) (integer 0)))
  (typecase x
    (broadcast-stream
     (let ((s (car (last (broadcast-stream-streams x)))))
       (if s (if pos-p (file-position s pos) (file-position s)) 0)))
    (string-stream
     (let* ((st (c-stream-object0 x))(l (length st))(d (array-dimension st 0))
	    (p (case pos (:start 0) (:end l) (otherwise pos))))
       (if pos-p (when (<= p d) (setf (fill-pointer st) p)) l)))
    (otherwise
     (let ((n (get-byte-stream-nchars x))
	   (p (case pos (:start 0) (:end (file-length x)) (otherwise pos))))
       (if pos-p (when (fseek x (* p n)) p) (/ (ftell x) n))))))
