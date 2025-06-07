;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(export '(mdlsym mdl lib-name))

(defun lib-name (p)
  (if (or (string= p "") (string= p "libc") (string= p "libm")) "" 
    (string-concatenate #+darwin "/usr/lib/system/" p #+darwin ".dylib" #+cygwin ".dll" #-(or darwin cygwin) ".so")));FIXME

(defun mdl (n p vad)
  (let* ((sym (mdlsym n (lib-name p)))
	 (ad (symbol-value sym))
	 (adp (aref %init vad)))
    (dladdr-set adp ad)
    (dllist-push %memory sym adp)))

(defun mdlsym (str &optional (n "" np))
  (let* ((pk (or (find-package "LIB") (make-package "LIB")))
	 (k  (if np (dlopen n) 0))
	 (ad (dlsym k str))
	 (p (or (dladdr ad t) ""));FIXME work around dladdr here, not posix
	 (psym (intern p pk))
	 (npk (or (find-package psym) (make-package psym :use '(:cl))))
	 (sym (and (shadow str npk) (intern str npk))))
    (export (list psym) pk)
    (export sym npk)
    (set psym k)(set sym ad)
    sym))

