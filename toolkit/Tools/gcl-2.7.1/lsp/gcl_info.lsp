;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(eval-when (compile eval)
 (defmacro f (op x y)
   (list op x y)))

(defconstant +crlu+ #v"")
(defconstant +crnp+ #v"[]")

(defvar *info-data* nil)
(defvar *current-info-data* nil)

(defun file-to-string (file &optional (start 0)
			    &aux (si::*ALLOW-GZIPPED-FILE* t)(len 0))
  (with-open-file
   (st file)
   (setq len (file-length st))
   (or (and (<= 0 start ) (<= start len))
       (error "illegal file start ~a" start))
   (let ((tem (make-array (- len start)
			  :element-type 'character)))
     (if (> start 0) (file-position st start))
     (si::fread tem 0 (length tem) st) tem)))

(defun atoi (string start &aux (ans 0) (ch 0)(len (length string)))
  (declare (string string))
  (declare (fixnum start ans ch len) )
  (while (< start len)
    (setq ch (char-code (aref string start)))
    (setq start (+ start 1))
    (setq ch (- ch #.(char-code #\0)))
    (cond ((and (>= ch 0) (< ch 10))
	   (setq ans (+ ch (* 10 ans))))
	  (t (return nil))))
  ans)
  
(defun info-get-tags (file &aux (lim 0) *match-data* tags files
			   (*case-fold-search* t))
  (declare (fixnum lim))
  (let ((s (file-to-string file)) (i 0))
    (declare (fixnum i) (string s))
    (cond ((f >= (string-match #v"[\n]+Indirect:" s 0) 0)
	   (setq i (match-end 0))
	   (setq lim (string-match +crlu+ s i))
	   (while
	       (f >= (string-match #v"\n([^\n]+): ([0-9]+)" s i lim) 0)
	     (setq i (match-end 0))
	     (setq files
		   (cons(cons
			 (atoi s (match-beginning 2))
			 (get-match s 1)
			 )
			files)))))
    (cond ((f >=  (si::string-match #v"[\n]+Tag Table:" s i) 0)
	   (setq i (si::match-end 0))
	   (cond ((f >= (si::string-match +crlu+ s i) 0)
		  (setq tags (subseq s i (si::match-end 0)))))))
    (if files (or tags (info-error "Need tags if have multiple files")))
    (list* tags (nreverse files))))

(defun re-quote-string (x &aux (i 0) ch)
  (declare (fixnum i))
  (let* ((x (if (stringp x) x (string x)))
	 (len (length x))
	 (tem x))
    (while (< i len)
      (setq ch (aref x i))
      (when (position ch "\\()[]+.*|^$?")
	(when (eq x tem)
	  (setq tem 
		(make-array len :adjustable t
			    :element-type 'character :fill-pointer i))
	  (dotimes (j i) (setf (aref tem j) (aref x j))))
	(vector-push-extend #\\ tem))
      (unless (eq tem x) (vector-push-extend ch tem))
      (setq i (+ i 1)))
    (remove-if-not 'standard-char-p tem)))

(defun get-match (string i)
  (subseq string (match-beginning i) (match-end i)))

(defun get-nodes (pat node-string &aux (i 0) ans
		      (*case-fold-search* t) *match-data*)
  (declare (fixnum i))
  (when node-string
	(setq pat
	      (si::string-concatenate "Node: ([^]*" (re-quote-string
						       pat) "[^]*)"))
	(while (f >= (string-match pat node-string i) 0)
	  (setq i (match-end 0))
	  (setq ans (cons (get-match node-string 1) 
			  ans))
	  )
	(nreverse ans)))

(defun get-index-node ()
 (or (third *current-info-data*) 
     (let* (
	    s
	    (node-string (car (nth 1 *current-info-data*)))
	    (node
	     (and node-string (car (get-nodes "Index" node-string)))))
       (when node
	   (setq s (show-info
		    node
		    nil
		    nil
		    ))
	(setf (third *current-info-data*) s)))))

(defun nodes-from-index (pat  &aux (i 0) ans
			      (*case-fold-search* t) *match-data*)
  (let ((index-string (get-index-node)))
    (when index-string
    (setq pat 
	  (si::string-concatenate #u"\n\\* ([^:\n]*" (re-quote-string
						  pat)
				  #u"[^:\n]*):[ \t]+([^\t\n,.]+)"))
    (while (f >= (string-match pat index-string i) 0)
      (setq i (match-end 0))
      (setq ans (cons (cons (get-match index-string 1)
			    (get-match index-string 2))
			  
			  
		      ans))
      )
    (nreverse ans))))

(defun get-node-index (pat node-string &aux (node pat) *match-data*)
  (cond ((null node-string) 0)
	(t
	 (setq pat
	       (si::string-concatenate "Node: "
				       (re-quote-string pat) "([0-9]+)"))
	 (cond ((f >= (string-match pat node-string) 0)
		(atoi node-string (match-beginning 1)))
	       (t (info-error "cant find node ~s" node) 0)))))

(defun all-matches (pat st &aux (start 0) *match-data*)
  (declare (fixnum start))
   (sloop::sloop while (>= (setq start (si::string-match pat st start)) 0)
         do nil;(print start)
         collect (list start (setq start (si::match-end 0)))))



(defmacro node (prop x)
  `(nth ,(position prop '(string begin end header name
				 info-subfile
				 file tags)) ,x)) 

(defun node-offset (node)
  (+ (car (node info-subfile node)) (node begin node)))

(defvar *info-paths*
  '("" "/usr/info/" "/usr/local/lib/info/" "/usr/local/info/"
    "/usr/local/gnu/info/" "/usr/share/info/"))

(defvar *old-lib-directory* nil)
(defun setup-info (name &aux tem file)

  (unless (eq *old-lib-directory* *lib-directory*)
    (setq *old-lib-directory* *lib-directory*)
    (push (string-concatenate *lib-directory* "info/") *info-paths*)
    (setq *info-paths* (fix-load-path *info-paths*)))

  (when (equal name "DIR")
    (setq name "dir"))

  ;; compressed info reading -- search for gzipped files, and open with base filename
;; relying on si::*allow-gzipped-files* to uncompress
  (setq file (file-search name *info-paths* '("" ".info" ".gz") nil))
  (let ((ext (search ".gz" file)))
    (when ext
      (setq file (subseq file 0 ext))))

  (unless file
    (unless (equal name "dir")
      (let* ((tem (show-info "(dir)Top" nil nil))
	     *case-fold-search*)
	(cond ((<= 0 (string-match
		      (string-concatenate "\\(([^(]*" (re-quote-string name) "(.info)?)\\)")
		      tem))
	       (setq file (get-match tem 1)))))))

  (if file
      (let* ((na (namestring file )));(truename file)
	(cond ((setq tem (assoc na *info-data* :test 'equal))
	       (setq *current-info-data* tem))
	      (t   (setq *current-info-data*
			 (list na (info-get-tags na) nil))
		   (setq *info-data* (cons *current-info-data* *info-data*)
			 ))))
      (format t "(not found ~s)" name))
  nil)
			  
(defun get-info-choices (pat type)
      (if (eql type 'index)
	  (nodes-from-index pat )
	(get-nodes pat (car (nth 1 *current-info-data*)))))

(defun add-file (v file &aux (lis v))
  (while lis
    (setf (car lis) (list (car lis) file))
    (setq lis (cdr lis)))
  v)

(defvar *info-window* nil)
(defvar *tk-connection* nil)

(defun info-error (&rest l)
  (if *tk-connection*
      (tk::tkerror (apply 'format nil l))
    (apply 'error l)))

(defvar *last-info-file* nil)
;; cache last file read to speed up lookup since may be gzipped..
(defun info-get-file (pathname)
  (setq pathname
	(merge-pathnames pathname
			 (car *current-info-data*)))
  (cdr 
   (cond ((equal (car *last-info-file*) pathname)
	  *last-info-file*)
	 (t (setq *last-info-file*
		  (cons pathname (file-to-string pathname)))))))

(defun waiting (win)
  (and *tk-connection*
       (fboundp win)
       (winfo :exists win :return 'boolean)
       (funcall win :configure :cursor "watch")))

(defun end-waiting (win) (and (fboundp win)
			   (funcall win :configure :cursor "")))

(defun info-subfile (n  &aux )
;  "For an index N return (START . FILE) for info subfile
; which contains N.   A second value bounding the limit if known
; is returned.   At last file this limit is nil."
  (let ((lis (cdr (nth 1 *current-info-data*)))
	ans lim)
    (and lis (>= n 0)
	   (dolist (v lis)
		 (cond ((> (car v) n )
			(setq lim (car v))
			(return nil)))
		 (setq ans v)
		 ))
    (values (or ans (cons 0 (car *current-info-data*))) lim)))

;;used by search
(defun info-node-from-position (n &aux  (i 0))
  (let* ((info-subfile (info-subfile n))
	 (s (info-get-file (cdr info-subfile)))
	 (end (- n (car info-subfile))))
    (while (f >=  (string-match +crlu+ s i end) 0)
      (setq i (match-end 0)))
    (setq i (- i 1))
    (if (f >= (string-match
	       #v"[\n][^\n]*Node:[ \t]+([^\n\t,]+)[\n\t,][^\n]*\n"  s i) 0)
	(let* ((i (match-beginning 0))
	       (beg (match-end 0))
	       (name (get-match s 1))
	       (end(if (f >= (string-match +crnp+ s beg) 0)
		       (match-beginning 0)
		     (length s)))
	       (node (list* s beg end i name info-subfile
				 *current-info-data*)))
	  node))))
    
(defun show-info (name  &optional position-pattern
			(use-tk *tk-connection*)
			&aux info-subfile *match-data* 
			file
		       (initial-offset 0)(subnode -1))
  (declare (fixnum subnode initial-offset))
;;; (pat . node)
;;; node
;;; (node file)
;;; ((pat . node) file)
;  (print (list name position-pattern use-tk))
  (progn ;decode name
    (cond ((and (consp name) (consp (cdr name)))
	   (setq file (cadr name)
		 name (car name))))
    (cond ((consp name)
	   (setq position-pattern (car name) name (cdr name)))))
  (or (stringp name) (info-error "bad arg"))
  (waiting *info-window*)  
  (cond ((f >= (string-match #v"^\\(([^(]+)\\)([^)]*)" name) 0)
	 ;; (file)node
	 (setq file (get-match name 1))
	 (setq name (get-match name 2))
	 (if (equal name "")(setq name "Top"))))
  (if file  (setup-info file))
  (let ((indirect-index (get-node-index name
					(car (nth 1 *current-info-data*)))))
    (cond ((null  indirect-index)
	   (format t"~%Sorry, Can't find node ~a" name)
	   (return-from show-info nil)))
	
    (setq info-subfile (info-subfile indirect-index))
    (let* ((s
	    (info-get-file (cdr info-subfile)))
	   (start (- indirect-index (car info-subfile))))
      (cond ((f >= (string-match
		    ;; to do fix this ;; see (info)Add  for description;
		    ;;  the 
		    (si::string-concatenate
		     #u"[\n][^\n]*Node:[ \t]+"
		     (re-quote-string name) #u"[,\t\n][^\n]*\n") 
		    s start) 0)
	     (let* ((i (match-beginning 0))
		    (beg (match-end 0))
		    (end(if (f >= (string-match +crnp+ s beg) 0)
			    (match-beginning 0)
			  (length s)))
		    (node (list* s beg end i name info-subfile
				 *current-info-data*)))

	       (cond
		(position-pattern
		 (setq position-pattern (re-quote-string position-pattern))

		 (let (*case-fold-search* )
		   (if (or
			(f >= (setq subnode
				    (string-match
				     (si::string-concatenate
				      #u"\n -+ [A-Za-z ]+: "
				      position-pattern #u"[ \n]")
				     s beg end)) 0)
			(f >= (string-match position-pattern s beg end) 0))
		       (setq initial-offset
			     (- (match-beginning 0) beg))
		     ))))
	       (cond ( use-tk
		       (prog1 (print-node node initial-offset)
			 (end-waiting  *info-window*))
		       )
		     (t
		      (let ((e
			     (if (and (>= subnode 0)
				      (f >=
					 (string-match 
					  #v"\n -+ [a-zA-Z]"
					  s 
					  (let* ((bg (+ beg 1 initial-offset))
						 (sd (string-match #v"\n   " s bg end))
						 (nb (if (minusp sd) bg sd)))
					    nb) 
						       end)
					 0))
				 (match-beginning 0)
			       end)))
			;(print (list  beg initial-offset e end))
			(subseq s (+ initial-offset beg) e )
			;s
			)))))
	    (t (info-error "Cant find node  ~a?" name)
	       (end-waiting  *info-window*)
	       ))
	    )))

(defvar *default-info-files* '( "gcl-si.info" "gcl-tk.info" "gcl.info"))

(defun info-aux (x dirs)
  (sloop for v in dirs
		    do (setup-info v)
		    append (add-file (get-info-choices x 'node) v)
		    append (add-file (get-info-choices x 'index) v)))

(defun info-search (pattern &optional start end &aux limit)
;  "search for PATTERN from START up to END where these are indices in
;the general info file.   The search goes over all files."
  (or start (setq start 0))
  (while start
    (multiple-value-bind
     (file lim)
     (info-subfile start)
     (setq limit lim)
     (and end limit (<  end limit) (setq limit end))

     (let* ((s  (info-get-file (cdr  file)))
	   (beg (car file))
	   (i (- start beg))
	   (leng (length s)))
       (cond ((f >= (string-match pattern s i (if limit (- limit beg) leng)) 0)
	      (return-from info-search (+ beg (match-beginning 0))))))
     (setq start lim)))
  -1)

#+debug ; try searching
(defun try (pat &aux (tem 0) s )
 (while (>= tem 0)
  (cond ((>= (setq tem (info-search pat tem)) 0)
	 (setq s (cdr *last-info-file*))
	 (print (list
		 tem
		 (list-matches s 0 1 2)
		 (car *last-info-file*)
		 (subseq s
			 (max 0 (- (match-beginning 0) 50))
			 (min (+ (match-end 0) 50) (length s)))))
	 (setq tem (+ tem (- (match-end 0) (match-beginning 0))))))))
   
(defun idescribe (name)
    (let* ((items (info-aux name *default-info-files*)))
      (dolist (v items)
	      (when (cond ((consp (car v))
			   (equalp (caar v) name))
			  (t (equalp (car v) name)))
		(format t "~%From ~a:~%" v)
		(princ (show-info v nil nil))))))
  
(defun info (x &optional (dirs *default-info-files*)  &aux wanted
	       *current-info-data* file position-pattern)
  (unless (consp dirs)
    (setq dirs *default-info-files*))
  (let ((tem (info-aux x dirs)))
    (cond
     (*tk-connection*
      (offer-choices tem dirs)
       )
     (t

    (when tem
      (let ((nitems (length tem)))
	  (sloop for i from 0 for name in tem with prev
		 do (setq file nil position-pattern nil)
		 (progn ;decode name
		   (cond ((and (consp name) (consp (cdr name)))
			  (setq file (cadr name)
				name (car name))))
		   (cond ((consp name)
			  (setq position-pattern (car name) name (cdr name)))))
		 (format t "~% ~d: ~@[~a :~]~@[(~a)~]~a." i
			 position-pattern
			 (if (eq file prev) nil (setq prev file)) name))
  	  (if (> (length tem) 1)
	    (format t "~%Enter n, all, none, or multiple choices eg 1 3 : ")
	    (terpri))
	  (let ((line (if (> (length tem) 1) (read-line) "0"))
	        (start 0)
	        val)
	    (while (equal line "") (setq line (read-line)))
	    (while (multiple-value-setq
		    (val start)
		    (read-from-string line nil nil :start start))
	      (cond ((numberp val)
		     (setq wanted (cons val wanted)))
		    (t (setq wanted val) (return nil))))
	    (cond ((consp wanted)(setq wanted (nreverse wanted)))
		  ((symbolp wanted)
		   (setq wanted (and
				 (equal (symbol-name wanted) "ALL")
				 (sloop for i below (length tem) collect i)))))
	    (when wanted
 	      ;; Remove invalid (numerical) answers
	      (setf wanted (remove-if #'(lambda (x)
					  (and (integerp x) (>= x nitems)))
				      wanted))
	      (format t "~%Info from file ~a:" (car *current-info-data*)))
	    (sloop for i in wanted
		   do (princ(show-info (nth i tem)))))))))))

	     
;; idea make info_text window have previous,next,up bindings on keys
;; and on menu bar.    Have it bring up apropos menu. allow selection
;; to say spawn another info_text window.   The symbol that is the window
;; will carry on its plist the prev,next etc nodes, and the string-to-file
;; cache the last read file as well.   Add look up in index file, so that can
;; search an indtqex as well.   Could be an optional arg to show-node
;; 



(defun default-info-hotlist()
  (namestring (merge-pathnames "hotlist" (user-homedir-pathname))))

(defvar *info-window* nil)

(defun add-to-hotlist (node )
  (if (symbolp node) (setq node (get node 'node)))
  (cond
   (node
    (with-open-file
     (st (default-info-hotlist)
	 :direction :output
	 :if-exists :append
	 :if-does-not-exist :create)
     (cond ((< (file-position st) 10)
	    (princ  #u"\nFile:\thotlist\tNode: Top\n\n* Menu: Hot list of favrite info items.\n\n" st)))
     (format st "* (~a)~a::~%" 
	     (node file node)(node name node))))))

(defun list-matches (s &rest l)
  (sloop for i in l 
	 collect
	 (and (f >= (match-beginning i) 0)
	      (get-match s i))))


;;; Local Variables: ***
;;; mode:lisp ***
;;; version-control:t ***
;;; comment-column:0 ***
;;; comment-start: ";;; " ***
;;; End: ***


