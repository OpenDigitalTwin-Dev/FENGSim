;; -*-Lisp-*-
;;; CMPFLET  Flet, Labels, and Macrolet.
;;;
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


(in-package :compiler)

(si:putprop 'flet 'c1flet 'c1special)
(si:putprop 'flet 'c2flet 'c2)
(si:putprop 'labels 'c1labels 'c1special)
(si:putprop 'labels 'c2labels 'c2)
(si:putprop 'macrolet 'c1macrolet 'c1special)
;;; c2macrolet is not defined, because MACROLET is replaced by PROGN
;;; during Pass 1.
(si:putprop 'call-local 'c2call-local 'c2)

(defstruct (fun (:print-function (lambda (x s i) (s-print 'fun (fun-name x) (si::address x) s))))
           name			;;; Function name.
           ref			;;; Referenced or not.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the vs-address for the
           			;;; function closure, or NIL.
           ref-ccb		;;; Cross closure reference.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the vs-address for the
           			;;; function closure, or NIL.
           cfun			;;; The cfun for the function.
           level		;;; The level of the function.

	   info                 ;;; fun-info;  CM, 20031008
	                        ;;; collect info structure when processing
	                        ;;; function lambda list in flet and labels
	                        ;;; and pass upwards to call-local and call-global
	                        ;;; to determine more accurately when
	                        ;;; args-info-changed-vars should prevent certain
	                        ;;; inlining
	                        ;;; examples: (defun foo (a) (flet ((%f8 nil (setq a 0)))
	                        ;;;     (let ((v9 a)) (- (%f8) v9))))
	                        ;;;           (defun foo (a) (flet ((%f8 nil (setq a 2)))
                                ;;;     (* a (%f8))))
	   (call (make-list 6));FIXME
	   vv src c1 c1cb fn)

(defun local-fun-fn (id)
  (let* ((fun (local-fun-p id)))
    (when fun (fun-fn fun))))

;; (defun local-fun-fun (id)
;;   (let* ((fun (local-fun-p id)))
;;     (when fun (car (atomic-tp (info-type (cadr (fun-prov fun))))))))

;; (defun local-fun-src (id)
;;   (let ((fun (local-fun-fun id)));FUN-SRC?
;;     (when fun (function-lambda-expression fun))))

(defun local-fun-src (id)
  (let ((fun (local-fun-p id)))
    (when fun (fun-src fun))))

(si::freeze-defstruct 'fun)

(defvar *funs* nil)

;;; During Pass 1, *funs* holds a list of fun objects, local macro definitions
;;; and the symbol 'CB' (Closure Boundary).  'CB' will be pushed on *funs*
;;; when the compiler begins to process a closure.  A local macro definition
;;; is a list ( macro-name expansion-function).

(defvar *restore-vars-env* nil)
(defun repl-lst (l m &optional o)
  (typecase l
    (cons (cond ((consp m) (setf (car l) (repl-lst (car l) (car m) o) (cdr l) (repl-lst (cdr l) (cdr m) o)) l)(m)))
    (t (if (eql l m) l (if o (new-bind) m)))))

(defun repl-tp (tp m &optional o)
  (unless (equal tp m)
    (let* ((atp (atomic-tp tp))(am (atomic-tp m)))
      (when (and atp am);FIXME redundant?
	(repl-lst (car atp) (car am) o))))
  tp)

(defmacro with-restore-vars (&rest body &aux (rv (sgen "WRV-"))(wns (sgen "WRVW-")))
  `(let (,rv (,wns *warning-note-stack*))
     (declare (ignorable ,rv))
     (labels ((keep-vars nil (setq ,rv *restore-vars*)(keep-warnings))
	      (keep-warnings nil (setq ,wns *warning-note-stack*))
	      (pop-restore-vars nil
	       (setq *warning-note-stack* ,wns)
	       (mapc (lambda (l &aux (v (pop l))(tp (pop l))(st (pop l)))
		       (cond ((var-p v)
			      (keyed-cmpnote (list (var-name v) 'type-propagation 'type)
					     "Restoring var type on ~s from ~s to ~s"
					     (var-name v) (cmp-unnorm-tp (var-type v)) (cmp-unnorm-tp tp))
			      (setf (var-type v) tp (var-store v) st))
			     (t
			      (keyed-cmpnote (list 'type-mod-unwind)	"Unwinding type ~s ~s" v tp)
			      (repl-tp v tp))))
		     (ldiff-nf *restore-vars* ,rv))))
       (declare (ignorable #'keep-vars))
       (prog1
	   (let (*restore-vars* (*restore-vars-env* *vars*))
	     (unwind-protect (progn ,@body) (pop-restore-vars)))
	 (mapc (lambda (l)
		 (when (member (car l) *restore-vars-env*)
		   (pushnew l *restore-vars* :key 'car)))
	       ,rv)))))


(defun ref-environment (&aux inner)
  (dolist (fun *funs*)
    (when (or (eq fun 'cb) (eq fun 'lb))
      (setq inner (or inner fun))))
  (when (eq inner 'cb)
    (ref-inner inner)))

(defun bump-closure-lam-sig (lam)
  (flet ((nt (x) (type-or1 x #tt)))
	(mapc (lambda (x) (setf (var-type x) (nt (var-type x)))) (caaddr lam))
	(let ((i (cadar (last lam))))
	  (setf (info-type i) (nt (info-type i))))
	(lam-e-to-sig lam)))

(defun process-local-fun (b fun def tp)
  (let* ((name (fun-name fun))
	 (lam (do-fun name (cons name (cdr def)) (fun-call fun) (member fun *funs*) b))
	 (res (list fun lam)))

    ;closures almost always called anonymously which will be slow unless argd is 0
    (unless (tailp (member-if-not 'fun-p *funs*) (member fun *funs*))
      (setf (car (fun-call fun)) (bump-closure-lam-sig lam)))

    (ref-environment);FIXME?
    (setf (fun-cfun fun) (next-cfun))
    (add-info (fun-info fun) (cadr lam));FIXME copy-info?
    (setf (info-type (fun-info fun)) (cadar (fun-call fun)))
    (setf (info-type (cadr lam)) tp)
    res))

;; (defun process-local-fun (b fun def tp)
;;   (let* ((name (fun-name fun))
;; 	 (lam (do-fun name (cons name (cdr def)) (fun-call fun) (member fun *funs*) b))
;; ;	 (cvs (let (r) (do-referred (v (cadr lam)) (when (and (var-p v) (var-cbb v)) (push v r))) r))
;; 	 (res (list fun lam))
;; ;	 (l (si::interpreted-function-lambda (cadr tp)))
;; 	 )

;;     ;closures almost always called anonymously which will be slow unless argd is 0
;;     (when (or (eq b 'cb) (fun-ref-ccb fun)) (setf (car (fun-call fun)) (bump-closure-lam-sig lam)))

;;     (ref-environment)
;;     (setf (fun-cfun fun) (next-cfun))
;; ;    (setf (cadr l) cvs)
;;     (add-info (fun-info fun) (cadr lam));FIXME copy-info?
;;     (setf (info-type (fun-info fun)) (cadar (fun-call fun)))
;;     (setf (info-type (cadr lam)) tp)
;;     res))


(defun ref-funs (form funs)
  (ref-obs form funs 
	   (lambda (x) (setf (fun-ref-ccb x) t))
	   (lambda (x) (declare (ignore x)))
	   (lambda (x) (setf (fun-ref x) t))))

(defun effective-safety-src (src &aux (n (pop src))(ll (pop src)))
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header src)
   `(,n ,ll ,@(when doc (list doc))
	,@(cons `(declare (optimize (safety ,(this-safety-level)))) decls)
	,@ctps
	,@body)))

(defvar *local-fun-inline-limit* 200)

(defun c1flet-labels (labels args &aux body ss ts is other-decl (info (make-info))
			     defs1 fnames (ofuns *funs*) (*funs* *funs*)(*top-level-src* *top-level-src*))

  (when (endp args) (too-few-args 'flet 1 0))

  (dolist (def (car args) (setq defs1 (nreverse defs1)))
    (let* ((x (car def))(y (si::funid-sym x))) (unless (eq x y) (setq def (cons y (cdr def)))))
    (cmpck (or (endp def) (endp (cdr def))) "The function definition ~s is illegal." def)
    (when labels
      (cmpck (member (car def) fnames) "The function ~s was already defined." (car def))
      (push (car def) fnames))
    (let* ((def (effective-safety-src def))
	   (src (mark-toplevel-src (si::block-lambda (cadr def) (car def) (cddr def))))
	   (fun (make-fun :name (car def) :src src :info (make-info :type nil :flags (iflags sp-change)))))
      (push fun *funs*)
      (unless (< (cons-count src) *local-fun-inline-limit*)
	(keyed-cmpnote (list (car def) 'notinline)
		       "Blocking inline of large local fun ~s" (car def))
	(pushnew (car def) *notinline*))
      (push (list fun (cdr def)) defs1)))
  
  (let ((*funs* (if labels *funs* ofuns)))
;    (mapc (lambda (x &aux (x (car x))) (setf (fun-fn x) (afe (cons 'df (current-env)) (mf (fun-name x))))) defs1))
    (mapc (lambda (x &aux (x (car x))) (setf (fun-fn x) (mf (fun-name x) x))) defs1))

  (multiple-value-setq (body ss ts is other-decl) (c1body (cdr args) t))
  
  (c1add-globals ss)
  (check-vdecl (mapcar (lambda (x) `(function ,(fun-name (car x)))) defs1) ts is)
  (setq body (c1decl-body other-decl body))
  
  (let ((nf (mapcar 'car defs1)))
    (ref-funs body nf)
    (when labels
      (do (fun) ((not (setq fun (car (member-if (lambda (x) (or (fun-ref x) (fun-ref-ccb x))) nf)))))
	  (setq nf (remove fun nf))
	  (when (fun-ref fun)
	    (ref-funs (fun-c1 fun) nf))
	  (when (fun-ref-ccb fun)
	    (ref-funs (fun-c1cb fun) nf)))))

  (add-info info (cadr body))
  (setf (info-type info) (info-type (cadr body)))

  (mapc (lambda (x &aux (x (car x))) (unless (or (fun-ref x) (fun-ref-ccb x)) (eliminate-src (fun-src x)))) defs1)

  (let* ((funs (mapcar 'car defs1))
	 (fns (mapcar (lambda (x) (caddr (fun-c1   x))) (remove-if-not 'fun-ref funs)))
	 (cls (mapcar (lambda (x) (caddr (fun-c1cb x))) (remove-if-not 'fun-ref-ccb funs))))
    (if (or fns cls)
	(list (if labels 'labels 'flet) info fns cls body)
	body)))


(defun c1flet (args)
  (c1flet-labels nil args))


(defun c2flet-labels (labels local-funs closures body
			     &aux (*vs* *vs*) (oclink *clink*) (*clink* *clink*) 
			     (occb-vs *ccb-vs*) (*ccb-vs* *ccb-vs*))

  (mapc (lambda (def &aux (fun (car def)))
	  (setf (fun-ref fun) (vs-push))
	  (clink (fun-ref fun))
	  (setf (fun-ref-ccb fun) (ccb-vs-push))) closures)

  (mapc (lambda (def &aux (fun (car def)))
	  (when (eq (fun-ref fun) t) (setf (fun-ref fun) (vs-push)))) local-funs)
  
  (let ((*clink*  (if labels *clink*  oclink))
	(*ccb-vs* (if labels *ccb-vs* occb-vs)))

    (mapc (lambda (def &aux (fun (pop def)))
	    (setf (fun-level fun) *level*)
	    (push (list nil *clink* *ccb-vs* fun (car def) *initial-ccb-vs*) *local-funs*)) local-funs)
    
    (when (or local-funs closures) (base-used));fixme
    
    (dolist (def closures)
      
      (let* ((fun (pop def))
	     (lam (car def))
	     (cl (update-closure-indices (fun-call fun)))
	     (sig (car cl))
	     (at (car sig))
	     (rt (cadr sig)))
	
	(push (list 'closure (if (null *clink*) nil (cons 0 0)) *ccb-vs* fun lam) *local-funs*)
      
	(wt-nl)
	(wt-vs* (fun-ref fun))
	(wt "=")

	(setf (fun-vv fun) 
	      (cons '|#,| (export-call-struct cl)))

	(wt-make-cclosure (fun-cfun fun) (fun-name fun) 
			  (fun-vv fun) (new-proclaimed-argd at rt) (argsizes at rt (xa lam)) *clink*)
	(wt ";")
	(wt-nl))))

  (c2expr body))

(defun c2flet (local-funs closures body)
  (c2flet-labels nil local-funs closures body))

(defun c1labels (args)
  (c1flet-labels t args))

(defun c2labels (local-funs closures body)
  (c2flet-labels t local-funs closures body))

(defvar *macrolet-env* nil)

(defun push-macrolet-env (defs)
  (dolist (def defs)
    (cmpck (or (endp def) (not (symbolp (car def))) (endp (cdr def)))
           "The macro definition ~s is illegal." def)
    (push (make-fun :name (car def) :fn (eval (si::defmacro-lambda (pop def) (pop def) def)))
	  *funs*)))

(defun c1macrolet (args &aux body ss ts is other-decl (*funs* *funs*))
  (when (endp args) (too-few-args 'macrolet 1 0))
  (push-macrolet-env (car args))
  (multiple-value-setq (body ss ts is other-decl) (c1body (cdr args) t))
  (c1add-globals ss)
  (check-vdecl nil ts is)
  (c1decl-body other-decl body))

(defun ref-inner (b)
  (when (eq b 'cb)
    (let* ((bv (member b *vars*))
	   (fv (member-if 'is-fun-var (nreverse (ldiff *vars* bv)))))
      (when fv 
	(setf (var-ref (car fv)) t)))))
;; (defun ref-inner (b)
;;   (when (eq b 'cb)
;;     (let* ((bv (member b *vars*))
;; 	   (fv (member-if 'is-fun-var *vars*)))
;;       (when fv 
;; 	(when (tailp bv fv)
;; 	  (setf (var-ref (car fv)) t))))))

;(defvar *local-fun-recursion* nil)
;; (defun c1local-fun (fname &aux ccb prev inner)
;;   (dolist (fun *funs*)
;;     (cond ((eq fun 'cb) (setq ccb t inner (or inner 'cb)))
;; 	  ((eq fun 'lb) (setq inner (or inner 'lb)))
;;           ((eq (fun-name fun) fname)
;; 	   (cond (ccb (ref-inner inner) (setf prev (fun-ref-ccb fun) (fun-ref-ccb fun) t))
;; 		 ((setf prev (fun-ref fun) (fun-ref fun) t)))
;; 	   (unless prev
;; 	     (unless (member fname *local-fun-recursion*)
;; 	       (let* ((*local-fun-recursion* (cons fname *local-fun-recursion*)))
;; 		 (setf (fun-c1 fun) (unfoo (fun-prov fun) (if ccb 'cb 'lb) fun)))))
;; 	   (setf (info-type (fun-info fun)) (cadar (fun-call fun)))
;; 	   (return (list 'call-local (fun-info fun) (list fun ccb)))))))

;; (defun make-fun-c1 (fun b env &optional osig)
;;   (let* ((res (under-env env (c1function (list (fun-src fun) b fun))))
;; 	 (sig (car (fun-call fun))))
;;     (if (and (is-referred fun (cadr res)) (not (eq (cadr osig) (cadr sig))))
;; 	(make-fun-c1 fun b env sig))
;;     res))

;; (defmacro make-local-fun (c1 b f env)
;;   `(progn
;;      (unless (,c1 ,f) (setf (,c1 ,f) t (,c1 ,f) (make-fun-c1 ,f ',b ,env)))
;;      (when (listp (,c1 ,f)) (,c1 ,f))))

(defvar *force-fun-c1* nil)
(defvar *fun-stack* nil)

(defun ifunp (key pred l)
  (car (member-if (lambda (x) (when (fun-p x) (funcall pred x (funcall key x)))) l)))
(defun ifunm (pred i)
  (or  (ifunp 'fun-c1 pred (info-ref i)) (ifunp 'fun-c1cb pred (info-ref-ccb i))))
(defun all-callees (i)
  (when i
    (nconc
     (mapcan (lambda (x) (when (fun-p x) (list (list x)))) (info-ref i))
     (mapcan (lambda (x) (when (fun-p x) (list (list x t)))) (info-ref-ccb i)))))
(defun callee-sigs (i)
  (mapcar (lambda (x) (cons x (car (fun-call (car x))))) (all-callees i)))
(defun invalidate (s)
  (unless (eq s *fun-stack*)
    (let* ((k (car (car s))))
      (keyed-cmpnote (list (fun-name (car k)) 'local) "invalidating local fun ~s" k)
      (if (cdr k) (setf (fun-c1cb (car k)) nil) (setf (fun-c1 (car k)) nil))
      (let ((*fun-stack* s)) (mapc 'invalidate (fourth (car s))))
      (invalidate (cdr s)))))
(defun recursive-loop-funs (s)
  (unless (eq s *fun-stack*)
    (let* ((k (car (car s))))
      (let ((*fun-stack* s)) (mapc 'recursive-loop-funs (fourth (car s))))
      (pushnew (car k) (recursive-loop-funs (cdr s))))));FIXME
(defun fun-stack (key res)
  (list key (car (fun-call (car key))) (callee-sigs (cadr res)) nil res))

(defun make-fun-c1 (fun ccb env &optional prev &aux (c1 (if ccb (fun-c1cb fun) (fun-c1 fun))) (key (cons fun ccb)) tmp)

  (labels ((set (fun val) (if ccb (setf (fun-c1cb fun) val) (setf (fun-c1 fun) val))))
	  
	  (cond (c1
		 (keyed-cmpnote (list (fun-name fun) 'local) "returning finalized value for local fun ~s"  key)
		 c1)
		((setq tmp (assoc key *fun-stack* :test 'equal))
		 (keyed-cmpnote (list (fun-name fun) 'local) "returning trial value for local fun ~s" key)
		 (pushnew *fun-stack* (fourth tmp))
		 (fifth tmp))
		((let* ((ii (keyed-cmpnote (list (fun-name fun) 'local) "processing local fun ~s" key))
			(*fun-stack* (cons (fun-stack key prev) *fun-stack*))
			(res (under-env env (c1function (list (fun-src fun)) (if ccb 'cb 'lb) fun)))
			(fun-stack-prev (pop *fun-stack*))
			(recursive-p (fourth fun-stack-prev))
			(i (cadr res))
			(callees (all-callees i)))
		   (declare (ignore ii))
		   (when recursive-p
		     (setf (info-flags (fun-info fun)) (logior (info-flags (fun-info fun)) (iflags compiler))))
		   (cond ((iflag-p (info-flags i) provisional)
			  (keyed-cmpnote (list (fun-name fun) 'provisional 'local) "local fun ~s provisionally processed" key)
			  res)
			 ((unless (member-if (lambda (x) (assoc x *fun-stack* :test 'equal)) callees)
			    (when recursive-p
			      (or (not (equal (cadr fun-stack-prev) (car (fun-call fun))))
;				  (member-if-not (lambda (x &aux (y (assoc x (caddr fun-stack-prev) :test 'equal))) (when y (equal (cdr y) (car (fun-call (car x)))))) callees)
				  )))
			  (mapc 'invalidate (fourth fun-stack-prev))
			  (keyed-cmpnote (list (fun-name fun) 'local) "reprocessing unfinished local fun ~s on sig mismatch: ~s"
					 key (list (butlast fun-stack-prev 2) (butlast (fun-stack key res) 2)))
			  (make-fun-c1 fun ccb env res))
			 (t
			  (keyed-cmpnote (list (fun-name fun) 'local) "finalizing local fun ~s" key)
			  (set fun res))))))))



(defun c1local-fun (fname &optional cl &aux ccb inner (lf (local-fun-p fname)))
  (dolist (fun *funs*)
    (cond ((not (fun-p fun)) (setq ccb (or (eq fun 'cb) ccb) inner (or inner fun)))
	  ((eq fun lf)
	   (let* ((cl (or ccb cl))
		  (env (fn-get (fun-fn fun) 'df))
		  (fm (make-fun-c1 fun cl env))
		  (info (if fm (copy-info (cadr fm)) (make-info)))
		  (c1fv (when ccb (c1inner-fun-var))))
	     (setf (info-type info) (cadar (fun-call fun)));FIXME
	     (if cl (pushnew fun (info-ref-ccb info)) (pushnew fun (info-ref info)))
	     (when c1fv (add-info info (cadr c1fv)))
	     (return (list 'call-local info (list fun cl ccb) c1fv fm)))))))



(defun sch-local-fun (fname)
  ;;; Returns fun-ob for the local function (not locat macro) named FNAME,
  ;;; if any.  Otherwise, returns FNAME itself.
  (dolist (fun *funs* fname)
    (when (and (not (eq fun 'CB))
               (not (consp fun))
               (eq (fun-name fun) fname))
          (return fun))))

(defun make-inline-arg-str (sig &optional (lev -1))
  (let* ((inl (let (r) (dotimes (i (1+ lev) r) (push i r))))
	 (inl (mapcar (lambda (x) (strcat "base" (write-to-string x))) inl))
	 (inl (if (= lev *level*) (cons "base" (cdr inl)) inl))
	 (va (member '* (car sig)))
	 (inl (dotimes (i (- (length (car sig)) (if va 1 0)) inl) 
		(push (strcat "#" (write-to-string i)) inl)))
	 (inl (if va (cons (if (eq va (car sig)) "#?" "#*") inl) inl))
	 (inl (nreverse inl)))
    (reduce 'strcat (mapcon 
		     (lambda (x) 
		       (if (and (cdr x) (not (member (cadr x) '("#*" "#?") :test 'equal)))
			   (list (car x) ",") (list (car x)))) inl)
	    :initial-value "")))

(defun vfun-wrap (x sig clp &optional ap &aux (ap (when ap (1- ap))))
  (let* ((mv (not (single-type-p (cadr sig))))
	 (va (member '* (car sig)))
	 (nreg (length (ldiffn (car sig) va))))
    (ms "(" 
	(when clp  (concatenate 'string "fcall.fun=" clp  ","))
	(when mv "fcall.valp=(fixnum)#v,")
	(when va "fcall.argd=")
	(when (and va ap) "-")
	(when va "#n")
	(when (and va ap (< ap nreg)) (- ap nreg))
	(when va ",") x ")")))

(defun make-local-inline (fd)
  (let* ((fun (pop fd))
	 (clp (pop fd))
	 (ap  (cadr fd))
	 (sig (car (fun-call fun)))
	 (sig (list (mapcar  (lambda (x) (link-rt x t)) (car sig)) (link-rt (cadr sig) t)))
	 (mv (not (single-type-p (cadr sig))))
	 (nm (c-function-name "L" (fun-cfun fun) (fun-name fun)))
	 (nm (concatenate 'string "(" (rep-type (coerce-to-one-value (cadr sig))) ")" nm))
	 (clp (when clp (ccb-vs-str (fun-ref-ccb fun))))
	 (nm (if clp (ms clp "->fun.fun_self") nm))
	 (inl (g1 clp nm sig ap clp (if clp -1 (fun-level fun)))))
    `(,(car sig) ,(cadr sig) 
      ,(if mv (flags rfa svt) (flags rfa))
      ,inl)))

;; (defun make-local-inline (fd)
;;   (let* ((fun (pop fd))
;; 	 (clp (pop fd))
;; 	 (ap  (cadr fd))
;; 	 (sig (car (fun-call fun)))
;; 	 (sig (list (mapcar  (lambda (x) (link-rt x nil)) (car sig)) (link-rt (cadr sig) nil)))
;; 	 (mv (not (single-type-p (cadr sig))))
;; 	 (nm (c-function-name "L" (fun-cfun fun) (fun-name fun)))
;; 	 (clp (when clp (ccb-vs-str (fun-ref-ccb fun))))
;; 	 (nm (if clp (ms clp "->fun.fun_self") nm))
;; 	 (inl (g1 clp nm sig ap clp (if clp -1 (fun-level fun)))))
;;     `(,(car sig) ,(cadr sig) 
;;       ,(if mv (flags rfa svt) (flags rfa))
;;       ,inl)))

;; (defun make-local-inline (fd)
;;   (let* ((fun (pop fd))
;; 	 (clp (pop fd))
;; 	 (ap  (pop fd))
;; 	 (sig (car (fun-call fun)))
;; 	 (sig (list (mapcar  (lambda (x) (link-rt x nil)) (car sig)) (link-rt (cadr sig) nil)))
;; 	 (mv (not (single-type-p (cadr sig))))
;; 	 (nm (c-function-name "L" (fun-cfun fun) (fun-name fun)))
;; 	 (clp (when clp (ccb-vs-str (fun-ref-ccb fun))))
;; 	 (nm (if clp (ms clp "->fun.fun_self") nm))
;; 	 (inl (g1 clp nm sig ap clp (if clp -1 (fun-level fun)))))
;;     `(,(car sig) ,(cadr sig) 
;;       ,(if mv (flags rfa svt) (flags rfa))
;;       ,inl)))

;; (defun make-local-inline (fd)
;;   (let* ((fun (pop fd))
;; 	 (clp (pop fd))
;; 	 (ap  (pop fd))
;; 	 (sig (car (fun-call fun)))
;; 	 (sig (list (mapcar  (lambda (x) (link-rt x nil)) (car sig)) (link-rt (cadr sig) nil)))
;; 	 (mv (not (single-type-p (cadr sig))))
;; 	 (nm (c-function-name "L" (fun-cfun fun) (fun-name fun)))
;; 	 (nm (if clp (strcat (ccb-vs-str (fun-ref-ccb fun)) "->fun.fun_self") nm))
;; 	 (inl (g0 nm sig ap (when clp (ccb-vs-str (fun-ref-ccb fun))) (if clp -1 (fun-level fun)))))
;;     `(,(car sig) ,(cadr sig) 
;;       ,(if mv (flags rfa svt) (flags rfa))
;;       ,inl)))

(defun c2call-local (fd c1fv lam args &aux (*vs* *vs*))
  (declare (ignore lam c1fv))
  (let ((*inline-blocks* 0))
    (unwind-exit (get-inline-loc (make-local-inline fd) args))
    (close-inline-blocks)))

;; (defun c2call-local (fd args &aux (*vs* *vs*))
;;   (let ((*inline-blocks* 0))
;;     (unwind-exit (get-inline-loc (make-local-inline fd) args))
;;     (close-inline-blocks)))

