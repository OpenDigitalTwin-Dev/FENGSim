;; By W. Schelter
;; Usage: (si::proclaim-file "foo.lsp") (compile-file "foo.lsp")

(proclaim (quote (optimize (compilation-speed 0) (safety 0) (speed 3)
			   (space 0)
			   (debug 0))))



;; You may wish to adjust the following to output the proclamations
;; for inclusion in a file.  All fixed arg functions should be proclaimed
;; before their references for maximum efficiency.

;; CAVEAT: The following code only checks for fixed args, it does
;; not check for single valuedness BUT does make a proclamation
;; to that efect.  Unfortunately it is impossible to tell about
;; multiple values without doing a full compiler type pass over 
;; all files in the relevant system.   AKCL supports doing such a pass
;; during the compilation of a system, and can thus produce proclaims for
;; a subsequent compilation.  [see emit-fn documentation].


(DEFVAR *DECLARE-T-ONLY* NIL)
(DEFUN PROCLAIM-FILE (NAME &OPTIONAL *DECLARE-T-ONLY*)
  (WITH-OPEN-FILE 
      (FILE NAME
            :DIRECTION :INPUT)
    (LET ((EOF (CONS NIL NIL)))
      (LOOP
       (LET ((FORM (READ FILE NIL EOF)))
         (COND ((EQ EOF FORM) (RETURN NIL))
               ((MAKE-DECLARE-FORM FORM ))))))))

(DEFUN MAKE-DECLARE-FORM (FORM)
; !!!
  (WHEN
        (LISTP FORM)
   (COND ((MEMBER (CAR FORM) '(EVAL-WHEN ))
          (DOLIST (V (CDDR FORM)) (MAKE-DECLARE-FORM V)))
         ((MEMBER (CAR FORM) '(PROGN ))
          (DOLIST (V (CDR FORM)) (MAKE-DECLARE-FORM V)))
         ((MEMBER (CAR FORM) '(IN-PACKAGE DEFCONSTANT))
          (EVAL FORM))
         ((MEMBER (CAR FORM) '(DEFUN))
          (COND
           ((AND
             (listp (CADDR FORM))
             (NOT (MEMBER '&REST (CADDR FORM)))
             (NOT (MEMBER '&BODY (CADDR FORM)))
             (NOT (MEMBER '&KEY (CADDR FORM)))
             (NOT (MEMBER '&OPTIONAL (CADDR FORM))))
             ;;could print  declarations here.
	    (print (list (cadr form) (ARG-DECLARES (THIRD FORM) (cdddr FORM))))
            (FUNCALL 'PROCLAIM
		     `(ftype (function ,(ARG-DECLARES (THIRD FORM) (cdddr FORM))
				       t)
			     ,(cadr form)))
	    ))))))

(DEFUN ARG-DECLARES (ARGS DECLS &AUX ANS)
  (COND ((STRINGP (CAR DECLS)) (SETQ DECLS (CADR DECLS)))
	(T (SETQ DECLS (CAR DECLS))))
  (COND ((AND (not *declare-t-only*)
	       (CONSP DECLS) (EQ (CAR DECLS ) 'DECLARE))
	 (DO ((V ARGS (CDR V)))
	     ((OR (EQ (CAR V) '&AUX)
		  (NULL V))
	      (NREVERSE ANS))
	     (PUSH (DECL-TYPE (CAR V) DECLS) ANS)))
	(T (MAKE-LIST (- (LENGTH args)
			 (LENGTH (MEMBER '&AUX args)))
		      :INITIAL-ELEMENT T))))

(DEFUN DECL-TYPE (V DECLS)
  (DOLIST (D (CDR DECLS))
	  (CASE (CAR D)
		(TYPE (IF (MEMBER V (CDDR D))
			(RETURN-FROM DECL-TYPE (SECOND D))))
		((FIXNUM CHARACTER FLOAT LONG-FLOAT SHORT-FLOAT )
		 (IF (MEMBER V (CDR D)) (RETURN-FROM DECL-TYPE (CAR D))))))
  T)
