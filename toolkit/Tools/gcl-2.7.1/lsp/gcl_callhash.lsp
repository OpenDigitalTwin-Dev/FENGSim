;; Copyright (C) 2024 Camm Maguire
;; -*-Lisp-*-

(in-package :si);FIXME this belongs in :compiler

(export '(*split-files* *sig-discovery* compress-fle))

(defstruct (call (:type list) (:constructor make-call))
  sig callees src file props name)
(defvar *cmr* nil)
(defvar *keep-state* nil)
(defvar *sig-discovery* nil)
(defvar *split-files* nil)

(defun break-state (sym x)
  (format t "Breaking state function ~s due to definition of ~s~%" x sym)
  (let ((o (old-src x)))
    (mapc (lambda (x) (remprop x 'state-function)) (car o))
    (mapc (lambda (x y) (unless (eq sym x) (eval `(defun ,x ,@(cdr y))))) (car o) (cadr o))
    (mapc (lambda (y) (push y *cmr*) (add-recompile y 'state-function (sig x) nil)) (car o)) 
    (fmakunbound x)
    (unintern x)))

(defconstant +et+ (mapcar (lambda (x) (cons (cmp-norm-tp x) x)) 
			  '(list cons proper-list proper-sequence sequence boolean null true array vector number immfix bfix bignum integer
				 function-designator
				 ratio short-float long-float float real number pathname hash-table function)))

(defvar *sig-discovery-props* nil)

(defun symbol-function-plist (sym &aux (fun (symbol-to-function sym)))
  (when fun (c-function-plist fun)))

(defun sym-plist (sym &aux (pl (symbol-function-plist sym)))
  (when pl
    (or (cdr (assoc sym *sig-discovery-props*)) pl)))
				  
(defun needs-recompile (sym)
  (let* ((plist (sym-plist sym))
	 (callees (cadr plist)))
    (mapc (lambda (x &aux (s (car x)) (cmp-sig (cdr x))(act-sig (car (sym-plist s))))
	    (unless (eq sym s)
	      (when act-sig
		(unless (sig= cmp-sig act-sig);Can be sig= if we don't hash, or eq
		  (return-from needs-recompile (list (list sym s cmp-sig act-sig)))))))
	  callees)
    nil))

(defun all-conflicts (&aux r q)
  (do-all-symbols (sym (sort q (lambda (x y)
				 (cond ((member (caar x) (cadr y)) 1)
				       ((member (caar y) (cadr x)) -1)
				       (0)))))
    (let* ((plist (sym-plist sym))(callees (cadr plist)))
      (mapc (lambda (x &aux (s (car x)) (cmp-sig (cdr x))(act-sig (car (sym-plist s))))
	      (unless (eq sym s)
		(when act-sig
		  (unless (sig= cmp-sig act-sig);Can be sig= if we don't hash, or eq
		    (pushnew sym (cadar (pushnew (list (car (pushnew (list s cmp-sig act-sig) r :test 'equal)) nil) q :key 'car :test 'equal)))))))
	    callees)
      nil)))

(defun same-file-all-callees (x y fn)
;  (let ((z (remove-if-not (lambda (x) (equal (file x) fn)) (callees x)))) ;FIXME remove inline
  (let (z)
    (dolist (l (callees x))
      (when (equal fn (file l));FIXME eq
	(push l z)))
    (do ((l (set-difference z y) (cdr l))
	 (r (union z y) (same-file-all-callees (car l) r fn)))
	((endp l) r))))

(defun same-file-all-callers (x y fn)
;  (let ((z (remove-if-not (lambda (x) (equal (file x) fn)) (callers x))));FIXME remove inline
  (let (z)
    (dolist (l (callers x))
      (when (equal fn (file l));FIXME eq
	(push l z)))
    (do ((l (set-difference z y) (cdr l))
	 (r (union z y) (same-file-all-callers (car l) r fn)))
	((endp l) r))))

;; (defun all-callees (x y)
;;   (let ((z (gethash x *ach*)))
;;     (if z (union z y)
;;       (let ((z (call-callees (gethash x *call-hash-table*))))
;; 	(do ((l (set-difference z y) (cdr l))
;; 	     (r (union z y) (all-callees (car l) r)))
;; 	    ((endp l) 
;; 	     (unless (intersection z y) (setf (gethash x *ach*) (set-difference r y)))
;; 	     r))))))

;; (defun all-callers (x y)
;;   (let ((z (gethash x *acr*)))
;;     (if z (union z y)
;;       (let ((z (call-callers (gethash x *call-hash-table*))))
;; 	(do ((l (set-difference z y) (cdr l))
;; 	     (r (union z y) (all-callers (car l) r)))
;; 	    ((endp l) 
;; 	     (unless (intersection z y) (setf (gethash x *acr*) (set-difference r y)))
;; 	     r))))))

(defun nsyms (n &optional syms)
  (declare (seqind n))
  (cond ((= n 0) (nreverse syms))
	((nsyms (1- n) (cons (gensym) syms)))))

(defun max-types (sigs &optional res)
  (cond ((not res) (max-types (cdr sigs) (ldiff-nf (caar sigs) (member '* (caar sigs)))))
	((not sigs) res)
	((max-types (cdr sigs) 
		    (let ((z (ldiff-nf (caar sigs) (member '* (caar sigs)))))
		      (append
		       (mapcar (lambda (x y) (or (not (equal x y)) x)) z res)
		     (early-nthcdr (length z) res)))))))

(defun early-nthcdr (i x)
  (declare (seqind i))
  (cond ((= 0 i) x)
	((early-nthcdr (1- i) (cdr x)))))

(defun old-src (stfn &optional src syms sts srcs)
  (cond (stfn (old-src nil (function-src stfn) syms sts srcs))
	((atom src) nil)
	((eq (car src) 'labels)
	 (list (mapcar 'car (cadr src)) 
	       (mapcar (lambda (x) (if (eq (caadr x) 'funcall) (cadadr x) (caadr x))) (cddr (caddr src)))))
	((or (old-src stfn (car src) syms sts srcs) (old-src stfn (cdr src) syms sts srcs)))))

(defun lambda-vars (ll)
  (remove '&optional (mapcar (lambda (x) (if (consp x) (car x) x)) ll)))

(defun inlinef (n syms sts fns)
    (unless (member-if (lambda (x) (intersection '(&rest &key &aux &allow-other-keys) (cadr x))) fns)
      (let* ((lsst (1- (length sts)))
	     (tps (max-types (mapcar 'sig syms)))
	     (min (reduce 
		   'min 
		   (mapcar (lambda (x) (length (ldiff-nf (cadr x) (member '&optional (cadr x))))) fns)
		   :initial-value 64));FIXME
	     (max (reduce 'max (mapcar (lambda (x) (length (lambda-vars (cadr x)))) fns) :initial-value 0))
	     (reqs (nsyms min))
	     (opts (nsyms (- max min)))
	     (ll (append reqs (when (> max min) (cons '&optional opts))))
	     (all (reverse (append reqs opts))))
	`(defun ,n ,(cons 'state ll)
	   (declare (fixnum state) ,@(mapcar 'list tps reqs))
	   ,@(let (d (z (cddr (car fns)))) 
	       (when (stringp (car z)) (pop z))
	       (do nil ((or (not z) (not (consp (car z))) (not (eq (caar z) 'declare))) (nreverse d)) 
		   (let ((q (pop z)))
		     (when (and (consp (cadr q)) (eq 'optimize (caadr q))) 
		       (push q d)))))

	   (labels
	    ,(mapcan (lambda (x y z)
		       `((,x ,(cadr y) (,n ,z ,@(lambda-vars (cadr y)))))) syms fns sts)
	    (case state
		  ,@(mapcar
		     (lambda (x y)
		       `(,(if (= x lsst) 'otherwise x) 
			 (funcall ,y ,@(reverse (early-nthcdr (- max (length (lambda-vars (cadr y)))) all)))))
		     sts fns)))))))

(defun sig (x) (let ((h (call x))) (when h (call-sig h))))
(defun signature (x) (readable-sig (sig x)))
(defun props (x) (let ((h (call x))) (when h (call-props h))))
(defun src (x) (let ((h (call x))) (when h (call-src h))))
(defun file (x) (let ((h (call x))) (when h (call-file h))))
;; (defun file (x) (let* ((f (if (functionp x) x (symbol-to-function x)))
;; 		       (d (when f (address (c-function-data f)))))
;; 		  (when d
;; 		    (unless (eql d +objnull+)
;; 		      (c-cfdata-name (nani d))))))
(defun name (x) (let ((h (call x))) (when h (call-name h))))
(defun callees (x) (let ((h (call x))) (when h (call-callees h))))
;(defun callers (x) (get x 'callers))

;; (defun *s (x) 
;;   (let ((p (find-package x)))
;;     (remove-if-not
;;      (lambda (y) (eq (symbol-package y) p)) 
;;      (let (r) 
;;        (maphash (lambda (x y) (when (eq '* (cadr (call-sig y))) (push x r))) *call-hash-table*)
;;        r))))

(defun mutual-recursion-peers (sym)
  (unless (or (get sym 'state-function) (get sym 'mutual-recursion-group))
    (let ((y (sig sym)))
      (when (eq '* (cadr y))
	(let ((e (same-file-all-callees sym nil (file sym)))
	      (r (same-file-all-callers sym nil (file sym))))
	  (remove-if-not
	   (lambda (x) 
	     (and (eq (symbol-package x) (symbol-package sym))
		  (let ((h (call x)))
		    (when h (eq '* (cadr (call-sig h)))))))
	   (intersection e r)))))))

;(defun mutual-recursion-peers (sym)
;  (unless (or (get sym 'state-function) (get sym 'mutual-recursion-group))
;    (let ((y (sig sym)))
;      (when (eq '* (cadr y)) 
;	(let* ((e (same-file-all-callees sym nil (file sym)))
;	       (r (same-file-all-callers sym nil (file sym)))
;	       (i (intersection e r))
;	       (i1 (remove-if-not (lambda (x) (get x 'mutual-recursion-group)) i))
;	       (i2 (set-difference i i1))
;	       (i (remove-duplicates (union (mapcan (lambda (x) (list (get x 'mutual-recursion-group))) i1) i2))))
;	  (mapc (lambda (x) (break-state x x)) i1)
;	  (remove-if-not (lambda (x) (eq '* (cadr (sig x)))) i))))))

;	  (remove-if (lambda (x) (get x 'mutual-recursion-group))
;		     (remove-if-not (lambda (x) (eq '* (cadr (sig x)))) i)))))))

(defun convert-to-state (sym)
  (let ((syms (mutual-recursion-peers sym)))
    (when (and (remove sym syms) (member sym syms))
      (let* ((fns (mapcar 'function-src syms))
	     (n (intern (symbol-name (gensym (symbol-name sym))) (symbol-package sym)))
	     (*keep-state* n)
	     (sts (let (sts) (dotimes (i (length syms) (nreverse sts)) (push i sts))))
	     (ns (inlinef n syms sts fns)))
	(when ns
	  (eval ns)
	  (mapc (lambda (x y z) (let ((z (cadr z))) (eval `(defun ,x ,z (,n ,y ,@(lambda-vars z)))))) syms sts fns)
	  (mapc (lambda (x) (putprop x n 'state-function)) syms)
;	  (dolist (l syms) (add-hash l nil (list (list n)) nil nil))
	  (putprop n syms 'mutual-recursion-group)
	  (add-recompile n 'mutual-recursion nil nil)
	  n)))))
    
(defun temp-prefix nil
  (concatenate 'string
	       *tmp-dir*
	       "gazonk_"
	       (write-to-string (let ((p (getpid))) (if (>= p 0) p (- p))))
	       "_"));FIXME

(defun compiler-state-fns nil
  (let ((p (find-package "COMPILER")))
    (when p
      (do-symbols 
       (s p)
       (when (member s *cmr*)
	 (let* ((x (convert-to-state s))(*keep-state* x))
	   (when x
	     (compile x)
	     (mapc 'compile (get x 'mutual-recursion-group)))))))))

(defun callers (sym &aux r)
  (do-all-symbols (s r)
    (when (member sym (callees s) :key 'car)
      (push s r))))

(defun callers-p (sym &aux (fn (or (macro-function sym)(symbol-function sym))))
  (do-all-symbols (s)
    (when (member sym (callees s) :key 'car)
      (return-from callers-p t))
    (when (member sym (symbol-plist s))
      (return-from callers-p t))
    (when (member fn (symbol-plist s))
      (return-from callers-p t))
    (when (member-if (lambda (x) (when (or (symbolp x)(functionp x)) (member sym (callees x) :key 'car))) (symbol-plist s))
      (return-from callers-p t))))

(defun dead-code (ps &aux r)
  (let ((p (find-package ps)))
    (when p
      (do-symbols
       (s p r)
       (when (fboundp s)
	 (unless (macro-function s)
	   (multiple-value-bind
	    (s k)
	    (find-symbol (symbol-name s) p)
	     (when (eq k :internal)
	       (unless (callers-p s)
		 (push s r))))))))))

(defun gen-discovery-props (&aux (*sig-discovery* t) q)
  (do-all-symbols (s) (let ((x (needs-recompile s))) (when x (pushnew (caar x) q))))
  (when q
    (format t "~%Pass 1 signature discovery on ~s functions ..." (length q))
    (mapc (lambda (x) (format t "~s~%" x) (compile x)) q)
    (gen-discovery-props)))

(defun do-recomp2 (sp fl &aux *sig-discovery-props* *compile-verbose* r)
  (gen-discovery-props)
  (dolist (s (gen-all-ftype-symbols))
    (let* ((f (or (file s) ""))
	   (sig (car (sym-plist s))))
      (when (and sig (member f fl :test 'string=));e.g. fns in o/, interpreted, wrong-file
	(push (list s sig) r))))
  (write-sys-proclaims1 sp r))

(defvar *do-recomp-output-dir* nil)
;;FIXME not always idempotent
(defun do-recomp (&optional cdebug &rest excl &aux *sig-discovery-props* *compile-verbose*)
  (gen-discovery-props)
  (let* ((fl (mapcar 'car *sig-discovery-props*))
	 (fl (remove-duplicates (mapcar (lambda (x &aux (f (file x))) (when f (namestring f))) fl) :test 'string=))
	 (fl (set-difference fl excl :test (lambda (x y) (search y x)))))
    (when cdebug (compiler::cdebug))
    (format t "~%Recompiling original source files ...~%")
    (mapc (lambda (x)
	    (format t "~s~%" x)
	    (compile-file x :output-file
			  (merge-pathnames
			   (make-pathname :type "o" :name (pathname-name x))
			   (if *do-recomp-output-dir* (truename *do-recomp-output-dir*) x))))
	  (remove nil fl))))

(defun gen-all-ftype-symbols (&aux r)
  (do-all-symbols (s r)
    (when (fboundp s)
      (unless (or (macro-function s) (special-operator-p s))
	(pushnew s r)))))

