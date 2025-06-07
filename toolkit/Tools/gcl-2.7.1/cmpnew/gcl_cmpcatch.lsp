;;; CMPCATCH  Catch, Unwind-protect, and Throw.
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

(si:putprop 'catch 'c1catch 'c1special)
(si:putprop 'catch 'c2catch 'c2)
(si:putprop 'unwind-protect 'c1unwind-protect 'c1special)
(si:putprop 'unwind-protect 'c2unwind-protect 'c2)
(si:putprop 'throw 'c1throw 'c1special)
(si:putprop 'throw 'c2throw 'c2)

(defun c1catch (args &aux (info (make-info :type #t* :flags (iflags sp-change volatile))))
  (when (endp args) (too-few-args 'catch 1 0))
  (let* ((tag (c1arg (pop args) info))
	 (in (mch))
	 (body (unwind-protect (c1progn args) 
		 (mapc (lambda (x &aux (v (pop x))) 
			 (setf (var-type v) (type-or1 (pop x) (var-type v)));FIXME do-setq-tp
			 (push-vbinds v (car x)));FIXME c1throw/c1return-from
		       in))))
    (add-info info (cadr body))
    (list 'catch info tag body)))

(si:putprop 'push-catch-frame 'set-push-catch-frame 'set-loc)

(defun c2catch (tag body &aux (*vs* *vs*))
  (let ((*value-to-go* '(push-catch-frame))) (c2expr* tag))
  (wt-nl "if(nlj_active)")
  (wt-nl "{nlj_active=FALSE;frs_pop();")
  (unwind-exit 'fun-val 'jump)
  (wt "}")
  (wt-nl "else{")
  (let ((*unwind-exit* (cons 'frame *unwind-exit*)))
       (c2expr body))
  (wt "}")
  )

(defun set-push-catch-frame (loc)
  (add-libc "setjmp")
  (setq *frame-used* t)
  (wt-nl "frs_push(FRS_CATCH," loc ");"))

(defun c1unwind-protect (args &aux (info (make-info :flags (iflags sp-change volatile))) form)
  (when (endp args) (too-few-args 'unwind-protect 1 0))
  (setq form (let ((*blocks* (cons 'lb *blocks*))
                   (*tags* (cons 'lb *tags*))
                   (*funs* (cons 'lb *funs*))
                   (*vars* (cons 'lb *vars*)))
                  (c1expr (car args))))
  (or-ccb-assignments (list form))
  (add-info info (cadr form))
  (setf (info-type info) (info-type (cadr form)))
  (setq args (c1arg (cons 'progn (cdr args))))
  (add-info info (cadr args))
  (list 'unwind-protect info form args))

;; (defun c1unwind-protect (args &aux (info (make-info :sp-change 1)) form)
;;   (incf *setjmps*)
;;   (when (endp args) (too-few-args 'unwind-protect 1 0))
;;   (setq form (let ((*blocks* (cons 'lb *blocks*))
;;                    (*tags* (cons 'lb *tags*))
;;                    (*funs* (cons 'lb *funs*))
;;                    (*vars* (cons 'lb *vars*)))
;;                   (c1arg (car args))))
;;   (add-info info (cadr form))
;;   (setq args (c1progn (cdr args)))
;;   (add-info info (cadr args))
;;   (list 'unwind-protect info form args))

;; (defun c1unwind-protect (args &aux (info (make-info :sp-change 1)) form)
;;   (incf *setjmps*)
;;   (when (endp args) (too-few-args 'unwind-protect 1 0))
;;   (setq form (let ((*blocks* (cons 'lb *blocks*))
;;                    (*tags* (cons 'lb *tags*))
;;                    (*funs* (cons 'lb *funs*))
;;                    (*vars* (cons 'lb *vars*)))
;;                   (c1expr (car args))))
;;   (add-info info (cadr form))
;;   (setq args (c1progn (cdr args)))
;;   (add-info info (cadr args))
;;   (list 'unwind-protect info form args))

(defun c2unwind-protect (form body
                         &aux (*vs* *vs*) (loc (list 'vs (vs-push)))
			 top-data)
  ;;;  exchanged following two lines to eliminate setjmp clobbering warning
  (add-libc "setjmp")
  (setq *frame-used* t)
 (wt-nl "frs_push(FRS_PROTECT,Cnil);")
  (wt-nl "{object tag=Cnil;frame_ptr fr=NULL;object p;bool active;")
  (wt-nl "if(nlj_active){tag=nlj_tag;fr=nlj_fr;active=TRUE;}")
  (wt-nl "else{")
  (let ((*value-to-go* 'top)
	*top-data* )
    (c2expr* form)
    (setq top-data *top-data*))
  (wt-nl "active=FALSE;}")
  (wt-nl loc "=Cnil;")
  (wt-nl "while(vs_base<vs_top)")
  (wt-nl "{" loc "=make_cons(vs_top[-1]," loc ");vs_top--;}")
  (wt-nl) (reset-top)
  (wt-nl "nlj_active=FALSE;frs_pop();")
  (let ((*value-to-go* 'trash)) (c2expr* body))
  (wt-nl "vs_base=vs_top=base+" *vs* ";")
  (base-used)
  (wt-nl "for (p= " loc ";!endp(p);p=p->c.c_cdr) vs_push(p->c.c_car);")
  (wt-nl "if (active) {")
  (wt-nl "unwind(fr,tag);")
  (unwind-exit nil)
  (wt-nl "} else {")
  (unwind-exit 'fun-val nil (if top-data (car top-data)))
  (wt "}}"))

(defun c1no-value (args)
  (declare (ignore args))
  (let ((f (copy-tree (c1nil))))
    (setf (cadr f) (make-info :type #tnil))
    f))
(si::putprop 'si::no-value 'c1no-value 'c1)

(defun c1throw (args &aux (info (make-info :type #tnil :flags (iflags side-effects))) tag)
  (when (or (endp args) (endp (cdr args)))
        (too-few-args 'throw 2 (length args)))
  (unless (endp (cddr args))
          (too-many-args 'throw 2 (length args)))
  (setq tag (c1arg (car args)))
  (add-info info (cadr tag))
  (setq args (c1arg (cadr args)))
  (add-info info (cadr args))
  (list 'throw info tag args))

;; (defun c1throw (args &aux (info (make-info :type #tnil :flags (iflags side-effects))) tag)
;;   (when (or (endp args) (endp (cdr args)))
;;         (too-few-args 'throw 2 (length args)))
;;   (unless (endp (cddr args))
;;           (too-many-args 'throw 2 (length args)))
;;   (setq tag (c1expr (car args)))
;;   (add-info info (cadr tag))
;;   (setq args (c1expr (cadr args)))
;;   (add-info info (cadr args))
;;   (list 'throw info tag args))


(defun c2throw (tag val &aux (*vs* *vs*) loc)
  (wt-nl "{frame_ptr fr;")
  (case (car tag)
    (LOCATION (setq loc (caddr tag)))
    (VAR  (setq loc (cons 'var (third tag))))	
    (t (setq loc (list 'vs (vs-push)))
       (let ((*value-to-go* loc)) (c2expr* tag))))

  (wt-nl "fr=frs_sch_catch(" loc ");")
  (wt-nl "if(fr==NULL) FEerror(\"The tag ~s is undefined.\",1," loc ");")
  (let ((*value-to-go* 'top)) (c2expr* val))
  (wt-nl "unwind(fr," loc ");")
  (unwind-exit nil)
  (wt-nl "}"))


