;; Copyright (C) 2024 Camm Maguire
(in-package :user)

(eval-when (compile load eval)

(if (find-package :walker)
    (use-package '(:lisp) :walker)
  (make-package :walker :use '(:lisp)))

(if (find-package :iterate)
    (use-package '(:lisp :walker) :iterate)
    (make-package :iterate :use '(:lisp :walker)))

(if (find-package :pcl)
    (use-package '(:walker :iterate :lisp :s) :pcl)
    (make-package :pcl :use '(:walker :iterate :lisp))))

(in-package :pcl)
(defvar *the-pcl-package* (find-package :pcl))
(defun load-truename (&optional errorp) *load-pathname*)
(import 'si::(clines defentry defcfun object void int double))
(import 'si::compiler-let :walker)
(defstruct slot-object)
