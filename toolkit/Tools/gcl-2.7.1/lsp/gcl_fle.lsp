;; Copyright (C) 2024 Camm Maguire
;; (in-package :lisp)

;; (export '(function-lambda-expression))

(in-package :si)
(export 'fle)

;; (export '(blocked-body-name parse-body-header))


;; (defun parse-body-header (x &optional doc decl ctps &aux (a (car x)))
;;   (cond 
;;    ((unless (or doc ctps) (and (stringp a) (cdr x))) (parse-body-header (cdr x) a decl ctps))
;;    ((unless ctps (when (consp a) (eq (car a) 'declare)))  (parse-body-header (cdr x) doc (cons a decl) ctps))
;;    ((when (consp a) (eq (car a) 'check-type)) (parse-body-header (cdr x) doc decl (cons a ctps)))
;;    (t (values doc (nreverse decl) (nreverse ctps) x))))

;; (defun parse-body-header (x &optional doc decl ctps)
;;   (let* ((a (car x))
;; 	 (q (macroexpand a)));FIXME is this correct?  clisp doesn't seem to think so
;;   (cond 
;;    ((unless (or doc ctps) (and (stringp q) (cdr x))) (parse-body-header (cdr x) q decl ctps))
;;    ((unless ctps (when (consp q) (eq (car q) 'declare)))  (parse-body-header (cdr x) doc (cons q decl) ctps))
;;    ((when (consp a) (eq (car a) 'check-type)) (parse-body-header (cdr x) doc decl (cons a ctps)))
;;    (t (values doc (nreverse decl) (nreverse ctps) x)))))

;; (defun make-blocked-lambda (ll decls ctps body block)
;;   (let ((body (if (eq block (blocked-body-name body)) body `((block ,block ,@body)))))
;;     `(lambda ,ll ,@decls ,@ctps ,@body)))

(defun block-lambda (ll block body)
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header body)
   (declare (ignore doc))
   (make-blocked-lambda ll decls ctps body block)))
       
;; (defun find-doc (x &optional y)
;;   (declare (ignore y))
;;   (multiple-value-bind
;;    (doc decls ctps body)
;;    (parse-body-header x)
;;    (values doc decls (nconc ctps body))))

;; (defun blocked-body-name (body)
;;   (when (and (not (cdr body))
;; 	     (consp (car body))
;; 	     (eq (caar body) 'block))
;;     (cadar body)))

(defun get-blocked-body-name (x)
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header (cddr x))
   (declare (ignore doc decls ctps))
   (blocked-body-name body)))


(defun compress-src (src)
  (let* ((w (make-string-output-stream))
	 (ss (si::open-fasd w :output nil nil)))
    (si::find-sharing-top src (aref ss 1))
    (si::write-fasd-top src ss)
    (si::close-fasd ss)
    (get-output-stream-string w)))

(defun uncompress-src (fun)
  (let* ((h   (call fun))
	 (fas (when h (call-src h)))
	 (fas (unless (fixnump fas) fas))
	 (ss  (if (stringp fas) (open-fasd (make-string-input-stream fas) :input 'eof nil) fas))
	 (out (if (vectorp ss) (read-fasd-top ss) ss))
	 (es  (when (eq (car out) 'lambda-closure) (cadr out)))
	 (env (when es (function-env fun 0))))
    (when env
      (setq out (list* (car out)
		       (mapcar (lambda (x) (list (pop x) (nth (- (length es) (car x)) env))) es)
		       (cddr out))))
    (when (vectorp ss)
      (close-fasd ss))
    out))

(defun fle (x) 
  (typecase
   x
   (function (function-lambda-expression x))
   (symbol (when (fboundp x) (unless (special-operator-p x)
			       (unless (macro-function x) (function-lambda-expression (symbol-function x))))))))


(defun function-lambda-expression (y &aux z) 
  (declare (optimize (safety 1)))
  (check-type y function)
  (let ((x (uncompress-src y)))
    (case (car x)
	  (lambda (values x nil (get-blocked-body-name x)))
	  (lambda-block (values (block-lambda (caddr x) (cadr x) (cdddr x)) nil (cadr x)))
	  (lambda-closure (values (setq z (cons 'lambda (cddr (cddr x))))  (cadr x) (get-blocked-body-name z)))
	  (lambda-block-closure (values
				 (block-lambda (caddr (cdddr x)) (cadr (cdddr x)) (cddr (cddr (cddr x)))) 
				 (cadr x) (fifth x)))
	  (otherwise (values nil t nil)))))

(defun function-src (sym)
  (let ((fun (if (symbolp sym) (symbol-to-function sym) sym)));FIXME
    (values (function-lambda-expression fun))))

