;; Copyright (C) 2024 Camm Maguire
(in-package :si)
(export '(%structure-name
          %compiled-function-name
          %set-compiled-function-name))
(in-package :pcl)

(eval-when
 (compile eval load)
 (setq  *EVAL-WHEN-COMPILE* t))

(defun %%allocate-instance--class (&aux wrapper slots)
  (let ((i (system:make-structure 'std-instance wrapper slots)))
    (c-set-t-tt i (logior 1 (c-t-tt i)))
    i))

(import '(si::memq) 'pcl)
(defmacro assq (item list) `(assoc ,item ,list :test #'eq))
(defmacro posq (item list) `(position ,item ,list :test #'eq))

(defun printing-random-thing-internal (thing stream)
  (format stream "~O" (si:address thing)))


(defmacro %svref (vector index)
  `(svref (the simple-vector ,vector) (the fixnum ,index)))

(defsetf %svref (vector index) (new-value)
  `(setf (svref (the simple-vector ,vector) (the fixnum ,index))
         ,new-value))

(si::freeze-defstruct 'pcl::std-instance)
(si::freeze-defstruct 'method-call)
(si::freeze-defstruct 'fast-method-call)


(defmacro fmc-funcall (fn pv-cell next-method-call &rest args)
  `(funcall ,fn ,pv-cell ,next-method-call ,@args))

(defun pcl::proclaim-defmethod (x y)
  (declare (ignore y))
  (and (symbolp x)
       (setf (get x 'compiler::proclaimed-closure ) t)))

(import 'si::seqind)

(defun %cclosure-env-nthcdr (n f) (function-env f n))
(defun cclosurep (x) (typep x 'function));(typecase x (compiled-function t)))
(defun %cclosure-env (f) (function-env f 0))
(declaim (inline %cclosure-env-nthcdr cclosurep %cclosure-env))

(defconstant funcallable-instance-closure-size 15)

(defun allocate-funcallable-instance-2 ()
  (let (dummy)
    (declare (ignore dummy))
    (lambda (&rest args)
      (declare (ignore args))
      (setq dummy (make-dummy-var));use dummy to ensure freshly allocated closure
      (called-fin-without-function))))

(defun fun-to-funcallable-instance (fin);This cannot be inlines
  (c-set-t-tt fin (logior 1 (c-t-tt fin)))
  (the si::funcallable-std-instance fin))

(defun allocate-funcallable-instance-1 ()
  (let ((fin (allocate-funcallable-instance-2))
	(env (make-list funcallable-instance-closure-size :initial-element nil)))
    (si::set-function-environment fin env)
    (fun-to-funcallable-instance fin)))

(defun funcallable-instance-p (x) (typep x 'funcallable-std-instance))
(defun std-instance-p (x) (typep x 'std-instance))
(declaim (inline std-instance-p funcallable-instance-p))
(remprop 'std-instance-p 'compiler::co1)

(defun si:%structure-name (x) (si::lit :object "(" (:object x) ")->str.str_def->str.str_self[0]"))
(defun %fboundp (x) (/= 0 (si::address (c-symbol-gfdef x))))
(declaim (inline si:%structure-name %fboundp))

(defun set-function-name-1 (fn new-name ignore)
  (declare (ignore ignore))
  (typecase fn
    (function;compiled-function
     (when (symbolp new-name) (pcl::proclaim-defmethod new-name nil))
     (setf (si::call-name (c-function-plist fn)) new-name)))
  fn)

(defun %set-cclosure (r v)

  (unless (typep r 'function)
    (error "Bad fn 1"))
  (unless (typep v 'function)
    (error "Bad fn 1"))
  (si::use-fast-links nil r)
  (progn (compiler::side-effects) (compiler::lit :object (:object r) "->fun.fun_self=" (:object v) "->fun.fun_self"));FIXME
  (c-set-function-minarg r (c-function-minarg v))
  (c-set-function-maxarg r (c-function-maxarg v))
  (c-set-function-neval r  (c-function-neval v))
  (c-set-function-vv r     (c-function-vv v))
  (c-set-function-data r   (c-function-data v))
  (c-set-function-plist r  (c-function-plist v))
  (c-set-function-argd r   (c-function-argd v))
  (mapl (lambda (x y) (setf (car x) (car y))) (%cclosure-env r) (%cclosure-env v)))

(defun structure-functions-exist-p nil t)

(defun structure-instance-p (x) (typep x 'structure))
(declaim (inline structure-instance-p))
;; (define-compiler-macro structure-instance-p (x)
;;   (once-only (x)
;;     `(and (si:structurep ,x)
;;           (not (eq (si:%structure-name ,x) 'std-instance)))))

(defun structure-type (x)
  (typecase x (structure (si:%structure-name x))));FIXME type-of
(declaim (inline structure-type))

;; (defun structure-type (x)
;;   (and (si:structurep x)
;;        (si:%structure-name x)))

;; (define-compiler-macro structure-type (x)
;;   (once-only (x)
;;     `(and (si:structurep ,x)
;;           (si:%structure-name ,x))))


(defun structure-type-included-type-name (type)
  (or (car (gethash type *structure-table*))
      (let ((includes (si::s-data-includes (get type 'si::s-data))))
	(when includes
	  (si::s-data-name includes)))))

(defun structure-type-internal-slotds (type)
   (si::s-data-slot-descriptions (get type 'si::s-data)))

(defun structure-type-slot-description-list (type)
  (or (cdr (gethash type *structure-table*))
      (mapcan (lambda (slotd)
		(when (and slotd (car slotd))
		  (let ((offset (fifth slotd)))
		    (let ((reader (lambda (x) (si:structure-ref1 x offset)))
			  (writer (lambda (v x) (si:structure-set x type offset v))))
		      (let* ((reader-sym 
			      (let ((*package* *the-pcl-package*))
				(intern (format nil "~s SLOT~D" type offset))))
			     (writer-sym (get-setf-function-name reader-sym))
			     (slot-name (first slotd)))
			(setf (symbol-function reader-sym) reader)
			(setf (symbol-function writer-sym) writer)
			(do-standard-defsetf-1 reader-sym)
			(list (list slot-name
				    (find-symbol (concatenate 'string (symbol-name type) "-" (symbol-name slot-name)) 
						 (or (symbol-package type) *package*))
				    reader-sym
				    writer
				    (third slotd)
				    (second slotd))))))))
              (let ((slotds (structure-type-internal-slotds type))
                    (inc (structure-type-included-type-name type)))
                (if inc
                    (nthcdr (length (structure-type-internal-slotds inc)) slotds)
		  slotds)))))

(defun structure-slotd-name (slotd) (first slotd))
(defun structure-slotd-accessor-symbol (slotd) (second slotd))
(defun structure-slotd-reader-function (slotd) (third slotd))
(defun structure-slotd-writer-function (slotd) (fourth slotd))
(defun structure-slotd-type (slotd) (fifth slotd))
(defun structure-slotd-init-form (slotd) (sixth slotd))

(defun renew-sys-files nil
  ;; packages:
  (compiler::get-packages-ansi
   '(:walker :iterate :pcl :slot-accessor-name)
   "sys-package.lisp")
  (with-open-file (st "sys-package.lisp"
		      :direction :output
		      :if-exists :append)
		  (format st "(lisp::in-package \"SI\")
(export '(%structure-name
          %compiled-function-name
          %set-compiled-function-name))
(in-package \"PCL\")
"))

  (si::do-recomp2 "sys-proclaim.lisp" (mapcar 'namestring (directory "*.*p"))))

	
		 
