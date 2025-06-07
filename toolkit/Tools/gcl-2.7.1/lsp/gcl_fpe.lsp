;; Copyright (C) 2024 Camm Maguire
(in-package :fpe)

(import 'si::(disassemble-instruction feenableexcept fedisableexcept fld *fixnum *float *double
				      *fixnum *ushort *uint fun-name
				      si-class-direct-subclasses si-class-name si-find-class
				      +fe-list+ +mc-context-offsets+ floating-point-error 
				      function-by-address clines defentry))
(export '(break-on-floating-point-exceptions read-instruction))

(eval-when
    (eval compile)

  (defconstant +feallexcept+ (reduce 'logior (mapcar 'caddr +fe-list+)))


  (defun moff (i r) (* i (cdr r)))
  
  (defun stl (s &aux (s (if (stringp s) (make-string-input-stream s) s))(x (read s nil 'eof)))
    (unless (eq x 'eof) (cons x (stl s))))

  (defun ml (r) (when r (make-list (truncate (car r) (cdr r)))))

  (defun mcgr (r &aux (i -1))
    (mapcar (lambda (x y) `(defconstant ,x ,(moff (incf i) r))) (when r (stl (pop r))) (ml r)))
  
  (defun mcr (p r &aux (i -1))
    (mapcar (lambda (x) `(defconstant ,(intern (concatenate 'string p (write-to-string (incf i))) :fpe) ,(moff i r)))
	    (ml r)))

  (defmacro deft (n rt args &rest code)
  `(progn
     (clines ,(nstring-downcase 
	       (apply 'concatenate 'string
			   (symbol-name rt) " " (symbol-name n) "("
			   (apply 'concatenate 'string 
				  (mapcon (lambda (x) (list* (symbol-name (caar x)) " " (symbol-name (cadar x)) 
							     (when (cdr x) (list ", ")))) args))
			   ") "
			   code)))
     (defentry ,n ,(mapcar 'car args) (,rt ,(string-downcase (symbol-name n)))))))

#.`(progn ,@(mcgr (first +mc-context-offsets+)))
#.`(progn ,@(mcr "ST" (second +mc-context-offsets+)))
#.`(progn ,@(mcr "XMM" (third +mc-context-offsets+)))


(defconstant +top-readtable+ (let ((*readtable* (copy-readtable)))
			       (set-syntax-from-char #\, #\Space)
			       (set-syntax-from-char #\; #\a)
			       (set-macro-character #\0 '0-reader)
			       (set-macro-character #\$ '0-reader)
			       (set-macro-character #\- '0-reader)
			       (set-macro-character #\% '%-reader)
			       (set-macro-character #\( 'paren-reader)
			       *readtable*))
(defconstant +sub-readtable+ (let ((*readtable* (copy-readtable +top-readtable+)))
			       (set-syntax-from-char #\0 #\a)
			       *readtable*))
(defvar *offset* 0)
(defvar *insn* nil)
(defvar *context* nil)


(defun rf (addr w)
  (ecase w (4 (*float addr 0 nil nil)) (8 (*double addr 0 nil nil))))

(defun ref (addr p w &aux (i -1)) 
  (if p 
      (map-into (make-list (truncate 16 w)) (lambda nil (rf (+ addr (* w (incf i))) w)))
    (rf addr w)))

(defun gref (addr &aux (z (symbol-name *insn*))(lz (length z))(lz (if (eql (aref z (- lz 3)) #\2) (- lz 3) lz))
		  (f (eql #\F (aref z 0))))
  (ref addr (unless f (eql (aref z (- lz 2)) #\P)) (if (or f (eql (aref z (1- lz)) #\D)) 8 4)))

(defun reg-lookup (x) (*fixnum (+ (car *context*) (symbol-value x)) 0 nil nil))

(defun st-lookup (x) (fld (+ (cadr *context*) (symbol-value x))))
(defun xmm-lookup (x) (gref (+ (caddr *context*) (symbol-value x))))


(defun lookup (x &aux (z (symbol-name x)))
  (case (aref z 0)
    (#\X (xmm-lookup x))
    (#\S (st-lookup x))
    (otherwise (reg-lookup x))))

(defun %-reader (stream subchar &aux (*readtable* +sub-readtable+)(*package* (find-package :fpe)))
  (declare (ignore subchar))
  (let ((x (read stream)))
    (lookup (if (eq x 'st)
		(intern (concatenate 'string (symbol-name x)
				     (write-to-string
				      (if (eql (peek-char nil stream nil 'eof) #\()
					  (let ((ch (read-char stream))(x (read stream))(ch (read-char stream)))
					    (declare (ignore ch))
					    x)
					0))) :fpe) x))))

(defun 0-reader (stream subchar &aux a (s 1)(*readtable* +sub-readtable+))

  (when (eql subchar #\$) (setq a t subchar (read-char stream)))
  (when (eql subchar #\-) (setq s -1 subchar (read-char stream)))
  (assert (eql subchar #\0))
  (assert (eql (read-char stream) #\x))

  (let* ((*read-base* 16)(x (* s (read stream))))
    (if a x (let ((*offset* x)) (read stream)))))

(defun paren-reader (stream subchar &aux (*readtable* +sub-readtable+))
  (declare (ignore subchar))
  (let* ((x (read-delimited-list #\) stream)))
    (gref (+ *offset* (pop x) (if x (* (pop x) (car x)) 0)))))

(defun read-operands (s context &aux (*context* context))
  (read-delimited-list #\; s))

(defun read-instruction (addr context &aux (*readtable* +top-readtable+)
			      (i (car (disassemble-instruction addr)))(s (make-string-input-stream (substitute #\; #\# i)))
			      (*insn* (read s)))
  (cons i (cons *insn* (when context (read-operands s context)))))


(defun fe-enable (a)
  (declare (fixnum a))
  (fedisableexcept)
  (feenableexcept a))


#.`(let ((fpe-enabled 0))
     (defun break-on-floating-point-exceptions 
       (&key suspend ,@(mapcar (lambda (x) `(,(car x) (logtest ,(caddr x) fpe-enabled))) +fe-list+) &aux r)
       (fe-enable
	(if suspend 0
	  (setq fpe-enabled 
		(logior
		 ,@(mapcar (lambda (x)
			     `(cond (,(car x) (push ,(intern (symbol-name (car x)) :keyword) r) ,(caddr x))
				    (0)))
			   +fe-list+)))))
       r))

(defun subclasses (class)
  (when class
    (cons class (mapcan 'subclasses (si-class-direct-subclasses class)))))

(defun code-condition (code)
  (or (reduce (lambda (y x) (if (subtypep y x) (si-class-name x) y))
	  (reduce (lambda (&rest r) (when r (apply 'intersection r)))
		  (mapcar (lambda (x) (subclasses (si-find-class (car x))))
			  (remove code +fe-list+ :key 'caddr :test-not 'logtest)))
	  :initial-value nil)
      'arithmetic-error))
	 
(defun floating-point-error (code addr context)
  (break-on-floating-point-exceptions :suspend t)
  (restart-case
   (unwind-protect
       (let* ((fun (function-by-address addr))(m (read-instruction addr context)))
	 ((lambda (&rest r) (apply 'error (if (find-package :conditions) r (list (format nil "~s" r)))))
	  (code-condition code)
	  :operation (list :insn (pop m) :op (pop m) :fun fun :addr addr) :operands m :function-name (when fun (fun-name fun))))
     (break-on-floating-point-exceptions))
   (continue nil :report (lambda (s) (format s "Continue disabling floating point exception trapping"))
	     (apply 'break-on-floating-point-exceptions (mapcan (lambda (x) (list x nil)) (break-on-floating-point-exceptions))))))
