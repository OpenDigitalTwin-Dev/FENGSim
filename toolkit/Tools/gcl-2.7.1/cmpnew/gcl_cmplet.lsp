;;; CMPLET  Let and Let*.
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
(eval-when (compile)
  (or (fboundp 'write-block-open) (load "cmplet.lsp")))

(si:putprop 'let 'c1let 'c1special)
(si:putprop 'let 'c2let 'c2)
(si:putprop 'let* 'c1let* 'c1special)
(si:putprop 'let* 'c2let* 'c2)

(defun set-var-init-type (v t1);;FIXME should be in c1make-var
  (when (eq (var-kind v) 'lexical)
    (setq t1 (coerce-to-one-value t1))
    (setf (var-dt v) (var-type v)
	  (var-type v) (ensure-known-type (if *compiler-new-safety* (var-type v) (type-and t1 (var-dt v))))
	  (var-mt v) (var-type v)
	  (var-loc v) (unless (and (eq (var-loc v) 'object)
				   (unless (eq t (var-type v)) (var-type v)))
			(var-loc v)))
    (unless (var-type v)
      (cmpwarn "Type mismatches binding declared ~s variable ~s to type ~s."
	       (cmp-unnorm-tp (var-dt v)) (var-name v) (cmp-unnorm-tp t1)))
    (keyed-cmpnote (list (var-name v) 'type-propagation 'type 'init-type)
		   "Setting init type of ~s to ~s" (var-name v) (cmp-unnorm-tp (var-type v)))))

(defun new-c1progn (f body)
  (let ((info (copy-info (cadr body))))
    (add-info info (cadr f))
    (list 'progn info (if (eq (car body) 'progn) (cons f (caddr body)) (list f body)))))
	
;; (defun side-effects-p (f &optional bl)
;;   (cond ((atom f) nil)
;; 	((eq (car f) 'setq) (let ((v (car (third f)))) (member (var-kind v) '(special global))));FIXME psetq
;; 	((member (car f) '(lambda function foo)) nil)
;; 	((eq (car f) 'call-global)
;; 	 (reduce (lambda (y x) (or y (side-effects-p x))) 
;; 		 (fourth f) :initial-value (not (get (caddr f) 'c1no-side-effects))))
;; 	((eq (car f) 'block) (side-effects-p (cdddr f) (cons (caddr f) bl)))
;; 	((member (car f) '(return return-from)) (or (not (member (caddr f) bl)) (side-effects-p (cdddr f) bl)))
;; 	((member (car f) '(call-local ordinary funcall apply throw princ structure-set go)));FIXME
;; 	((or (side-effects-p (car f) bl) (side-effects-p (cdr f) bl)))))

;; (defun ignorable-form (f)
;;   (cond ((member (car f) '(function lambda)))
;; 	((> (length (info-changed-array (cadr f))) 0) nil)
;; 	((side-effects-p f) nil)
;; 	(t)))

;; (defun have-provfn (form);FIXME provisional flag
;;   (cond ((atom form) (eq form 'provfn))
;; 	((or (have-provfn (car form)) (have-provfn (cdr form))))))

;; (defun provisional-block-trim (n bp fs star)
;;   (declare (ignorable n))
;;   (when *provisional-inline*
;;     (or bp
;; 	(when star
;; 	  (have-provfn (cdr fs))))))

(defun ignorable-form-with-local-unreferenced-changes (form vs)
  (let* ((i (cadr form))(ch (info-ch i))
	 (nch (remove-if (lambda (x) (and (member x vs)
					  (eq (var-kind x) 'lexical)
					  (not (eq (var-ref x) t))
					  (not (var-ref-ccb x)))) ch)))
    (ignorable-form
     (if (eq nch ch)
	 form
       (list* (car form) (let ((i (copy-info i)))(setf (info-ch i) nch) i) (cddr form))))))

(defun trim-vars (vars forms body &optional star)

  (do* (nv nf (vs vars (cdr vs)) (fs forms (cdr fs)) 
	   (av (append vars *vars*)) (fv (cdr av) (cdr fv)))
       ((or (endp vs) (endp fs)) (list nv nf body))
    (let ((var (car vs)) (form (car fs)))
      (cond ((and (eq (var-kind var) 'LEXICAL)
		  (not (eq t (var-ref var))) ;;; This field may be IGNORE.
		  (not (var-ref-ccb var)))
	     (check-vref var)
	     (keyed-cmpnote (list 'var-trim (var-name var))
			    "Trimming ~s; bound form ~a ignorable"
			    (var-name var) (if (ignorable-form form) "" "not "))
	     (unless (ignorable-form-with-local-unreferenced-changes form (cdr vs));(ignorable-form form)
	       (when star (ref-vars form (cdr vs)))
	       (let* ((*vars* (if nf (if star fv *vars*) av))
		      (f (if nf (car nf) body))
		      (np (new-c1progn form f)))
		 (if nf (setf (car nf) np) (setf body np)))))
	    ((push var nv) (when star (ref-vars form (cdr vs))) (push form nf))))))

(defun mvars (args ss is ts star inls);FIXME truncate this and make-c1forms at nil type
  (mapcar (lambda (x)
	    (let* ((n (if (atom x) x (pop x)))
		   (f (unless (atom x) (car x)))
		   (v (c1make-var n ss is ts))
		   (fm (if (and inls (eq f (caar inls))) (cdr (pop inls)) (c1arg f))));FIXME check
	      (set-var-init-type v (info-type (cadr fm)))
	      (when (eq (car fm) 'var) (pushnew (caaddr fm) (var-aliases v)))
	      (maybe-reverse-type-prop (var-type v) fm)
	      (when star (push-var v fm))
	      (cons v fm)))
	  args))

(defun push-var (var form)
  (push var *vars*)
  (push-vbind var form))

(defun c1let-* (args &optional star inls
		     &aux (nm (if star 'let* 'let))
		     (ov *vars*) (*vars* *vars*)
		     ss is ts body other-decls
		     (info (make-info)))

  (when (endp args) (too-few-args nm 1 0))

  (multiple-value-setq (body ss ts is other-decls) (c1body (cdr args) nil))

  (let* ((vs (nreverse (mvars (car args) ss is ts star inls)))
	 (vars (mapcar 'car vs))
	 (forms (mapcar 'cdr vs))
	 (vnames (mapcar 'var-name vars)))

    (unless star (mapc (lambda (x) (push-var (car x) (cdr x))) vs))

    (when (member-if-not 'identity forms :key (lambda (x) (info-type (cadr x))))
      (eliminate-src body)
      (setq body nil))

    (c1add-globals (set-difference ss vnames))
    (check-vdecl vnames ts is)
    (setq body (c1decl-body other-decls body))

    (unless (single-type-p (info-type (cadr body))) ;FIXME infinite recursion
      (let ((mv (car (member-if 'is-mv-var vars))))
	(when mv
	  (ref-vars (c1var (var-name mv)) (list mv)))))

    (ref-vars body vars)

    (dolist (var vars) (setf (var-type var) (var-mt var)));FIXME?
 
    (let* ((*vars* ov)
	   (z (trim-vars vars forms body star))
	   (vars (pop z))
	   (fms (pop z))
	   (body (car z)))

      (dolist (fm fms) (add-info info (cadr fm)))
      (add-info info (cadr body))
      
      (setf (info-type info) (info-type (cadr body)))
      
      (if vars (list nm info vars fms body)
	  (list* (car body) info (cddr body))))))

(defun c1let (args)
  (c1let-* args))
(defun c1let* (args)
  (c1let-* args t))


(defun c2let (vars forms body
                   &aux block-p bindings initials
                        (*unwind-exit* *unwind-exit*)
                        (*vs* *vs*) (*clink* *clink*) (*ccb-vs* *ccb-vs*))
  (do ((vl vars (cdr vl)) (fl forms (cdr fl)) (prev-ss nil))
      ((endp vl))
      (let* ((form (car fl)) (var (car vl))
	     (kind (c2var-kind var)))
	(cond (kind (setf (var-kind var) kind (var-loc var) (cs-push (var-type var) t)))
	      ((eq (var-kind var) 'down)
	       (or (si::fixnump (var-loc var)) (wfs-error)))
	      ((eq (var-kind var) 'special))
	      ((setf (var-ref var) (vs-push)))
	      )
        (if (member (var-kind var) +c-local-var-types+)
	    (push (list 'c2expr* (list 'var var nil) form) initials)
	(case (car form)
              (LOCATION
               (if (can-be-replaced var body)
                   (progn (setf (var-kind var) 'REPLACED (var-loc var) (caddr form)))
		 (push (list var (caddr form)) bindings)))
              (VAR
               (let ((var1 (caaddr form)))
		 (cond ((or (args-info-changed-vars var1 (cdr fl))
			    (and (member (var-kind var1) '(SPECIAL GLOBAL))
				 (member (var-name var1) prev-ss)))
			(push (list 'c2expr*
				    (cond ((eq (var-kind var) 'object)
					   (list 'var var nil))
					  ((eq (var-kind var) 'down)
					;(push (list var) bindings)
					   (list 'down (var-loc var)))
					  ((push (list var) bindings)
					   (unless (integerp (var-ref var)) (setf (var-ref var) (vs-push)))
					   (list 'vs (var-ref var))))
				    form) initials))
		       ((eq (var-kind var) 'replaced))
		       ((and (can-be-replaced var body)
			     (member (var-kind var1) '(LEXICAL REPLACED OBJECT))
			     (null (var-ref-ccb var1))
			     (not (is-changed  var1 (cadr body))))
			(setf (var-kind var) 'REPLACED)
			(setf (var-loc var)
			      (case (var-kind var1)
				    (LEXICAL (list 'vs (var-ref var1)))
				    (REPLACED (var-loc var1))
				    (OBJECT (list 'cvar (var-loc var1)))
				    (otherwise (baboon)))))
		       ((push (list var (list 'var var1 (cadr (caddr form)))) bindings)))))
              (otherwise 
	       (cond ((when (and nil (symbolp (car form));FIXME
				 (get (car form) 'wt-loc)
				 (can-be-replaced var body)
				 (= (var-register var) 1))
			(setf (var-kind var) 'replaced) (var-loc var) form))
		     ((push (list 'c2expr*
				  (cond ((eq (var-kind var) 'object)
					 (list 'var var nil))
					((eq (var-kind var) 'down)
					;(push (list var) bindings)
					 (list 'down (var-loc var)))
					((push (list var) bindings)
					 (unless (integerp (var-ref var)) (setf (var-ref var) (vs-push)))
					 (list 'vs (var-ref var))))
				  form) initials))))))
        (when (eq (var-kind var) 'SPECIAL) (push (var-name var) prev-ss))))
  
  (setq block-p (write-block-open vars))

  (dolist (binding (nreverse initials))
	   (cond ((type>= #tnil (info-type (cadr (third binding)))) 
		  (let ((*value-to-go* 'trash)) (c2expr* (third binding)))
		  (let ((*value-to-go* (second binding))) (c2expr* (c1nil))))
		 ((let ((*value-to-go* (second binding)))
		    (c2expr* (third binding))))))
  (dolist (binding (nreverse bindings))
    (if (cdr binding)
        (c2bind-loc (car binding) (cadr binding))
        (c2bind (car binding))))

  (c2expr body)
  (when block-p (wt "}")))

(defun c2let* (vars forms body
                    &aux (block-p nil)
                    (*unwind-exit* *unwind-exit*)
                    (*vs* *vs*) (*clink* *clink*) (*ccb-vs* *ccb-vs*))
  (do ((vl vars (cdr vl))
       (fl forms (cdr fl)))
      ((endp vl))
      (let* ((form (car fl)) (var (car vl))
	     (kind (c2var-kind var)))
	(when kind (setf (var-kind var) kind (var-loc var) (cs-push (var-type var) t)))
        (unless (member (var-kind var) +c-local-var-types+)
	  (case (car form)
		(LOCATION
		 (cond ((can-be-replaced* var body (cdr fl))
			(setf (var-kind var) 'REPLACED)
			(setf (var-loc var) (caddr form)))
		       ((eq (var-kind var) 'down)
			(or (si::fixnump (var-loc var)) (baboon)))
		       ((member (var-kind var) '(object special)))
		       ((setf (var-ref var) (vs-push)))
		       ))
		(VAR
		 (let ((var1 (caaddr form)))
		   (cond ((and (can-be-replaced* var body (cdr fl))
			       (member (var-kind var1) '(LEXICAL REPLACED OBJECT))
			       (null (var-ref-ccb var1))
			       (not (args-info-changed-vars var1 (cdr fl)))
			       (not (is-changed var1 (cadr body))))
			  (setf (var-kind var) 'REPLACED)
			  (setf (var-loc var)
				(case (var-kind var1)
				      (LEXICAL (list 'vs (var-ref var1)))
				      (REPLACED (var-loc var1))
				      (OBJECT (list 'cvar (var-loc var1)))
				      (t (baboon)))))
			 ((member (var-kind var) '(object special)))
			 ((setf (var-ref var) (vs-push)))
			 )))
		(otherwise
		 (cond ((when (and nil (symbolp (car form));FIXME
				   (get (car form) 'wt-loc)
				   (can-be-replaced var body)
				   (= (var-register var) 1))(print form)
			  (setf (var-kind var) 'replaced) (var-loc var) form))
		       ((member (var-kind var) '(object special)))
		       ((setf (var-ref var) (vs-push)))
;		       ((var-ref var) (setf (var-ref var) (vs-push)))
		       ))))))

  (setq block-p (write-block-open vars))

  (do ((vl vars (cdr vl))
       (fl forms (cdr fl))
       (var nil) (form nil))
      ((null vl))
      (setq var (car vl))(setq form (car fl))
;      (print (list (var-kind var) (car form)))
      (cond
       ((eq (var-kind var) 'replaced))
       ((type>= #tnil (info-type (cadr form))) 
	(let ((*value-to-go* 'trash)) (c2expr* form))
	(c2bind-loc var nil))
       ((member (var-kind var) +c-local-var-types+)
	(let ((*value-to-go* (list 'var var nil)))
	  (c2expr* form)))
       (t
	(case
    	 (car form)
	 (LOCATION (c2bind-loc var (caddr form)))
	 (VAR (c2bind-loc var (list 'var (caaddr form) (cadr (caddr form)))))
	 (t (c2bind-init var form))))))
	   
  (c2expr body)

  (when block-p (wt "}")))

(defun can-be-replaced (var body)
  (and (member (var-kind var) '(LEXICAL OBJECT REPLACED))
       (not (var-cb var))
       (not (var-noreplace var))
       (not (is-changed var (cadr body)))))

;; (defun can-be-replaced (var body)
;;   (and (member (var-kind var) '(LEXICAL OBJECT REPLACED))
;;        (not (var-cb var))
;;        (not (var-store var))
;;        (not (is-changed var (cadr body)))))

;; (defun can-be-replaced (var body)
;;   (and (or (eq (var-kind var) 'LEXICAL)
;; 	   (and (eq (var-kind var) 'object)
;; 		(< (the fixnum (var-register var))
;; 		   (the fixnum *register-min*))))
;;        (null (var-ref-ccb var))
;;        (not (eq (var-loc var) 'clb))
;;        (not (is-changed var (cadr body)))))

(defun can-be-replaced* (var body forms)
  (and (can-be-replaced var body)
       (dolist (form forms t)
         (when (is-changed var (cadr form))
	   (return nil)))))


(defun write-block-open (vars)
  (let ( block-p)
    (dolist
     (var vars)
     (let ((kind (var-kind var)))
       (when (or (eq kind 'object) (member kind +c-local-var-types+))
	     (wt-nl)
	     (unless block-p (wt "{") (setq block-p t))
	     (wt-var-decl var)
	     )))
    block-p ))


