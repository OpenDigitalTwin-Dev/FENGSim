;; Copyright (C) 1994 W. Schelter

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

(in-package "TK")



(eval-when (compile eval)
(defmacro f (op x y)
   `(the ,(ecase op (>= 'boolean)((+ -) 'fixnum))
	 (,op (the fixnum ,x) (the fixnum ,y))))
(defmacro while (test &body body)
  `(sloop while ,test do ,@ body))

(or (boundp '*info-window*)
    (si::aload "info"))
)
(defun simple-listbox (w)
  (let ((listbox (conc w '.frame.list))
	(scrollbar(conc w '.frame.scroll)))
    (frame (conc w '.frame))
    (scrollbar scrollbar :relief "sunken" :command
	       (tk-conc w ".frame.list yview"))
    (listbox listbox :yscroll (tk-conc w ".frame.scroll set")
	     :relief "sunken"
	     :setgrid 1)
    (pack scrollbar :side "right" :fill "y")
    (pack listbox :side "left" :expand "yes" :fill "both"))
  (conc w '.frame))
  
  
(defun insert-standard-listbox (w lis &aux print-entry)
  (funcall w :delete 0 'end)
  (setf (get w 'list) lis)
  (setq print-entry (get w 'print-entry))
  (dolist (v lis)
	  (funcall w :insert 'end
		   (if print-entry (funcall print-entry v) v))))

(defun listbox-move (win key |%y|)
  |%y|
  (let ((amt (cdr (assoc key '(("Up" . -1)
			  ("Down" . 1)
			  ("Next" . 10)
			  ("Prior" . -10))
			 :test 'equal))))
    (cond (amt
	   (funcall win :yview
		    (+ (funcall win :nearest 0 :return 'number) amt))))))

(defun new-window (name &aux tem)
  (cond ((not  (fboundp name)) name)
	((winfo :exists name :return 'boolean)
	 (let ((i 2))
	   (while (winfo :exists (setq tem (conc name i )) :return 'boolean)
	     (setq i (+ i 1)))
	   tem))
	(t name)))


(defun insert-info-choices (listbox list &aux file position-pattern prev)
  (funcall listbox :delete 0 'end)
    (sloop for i from 0 for name in list 
		 do  (setq file nil position-pattern nil)
		 (progn ;decode name
		   (cond ((and (consp name) (consp (cdr name)))
			  (setq file (cadr name)
				name (car name))))
		   (cond ((consp name)
			  (setq position-pattern (car name) name (cdr name)))))
		 (funcall listbox :insert 'end
		 (format nil "~@[~a :~]~@[(~a)~]~a." 
			 position-pattern
			 (if (eq file prev) nil (setq prev file)) name)))
  (setf (get listbox 'list)list))

(defun offer-choices (list info-dirs &optional (w (new-window '.info))
			   &aux listbox)
  (toplevel w)
  (simple-listbox w)
  (setq listbox (conc w '.frame.list))
  (insert-info-choices listbox list)
  (bind listbox "<Double-1>"
	#'(lambda ()
	    (show-info
	  (nth (atoi (funcall listbox :curselection :return 'string)
		     0)
	       (get listbox 'list)))))
  (button (conc w '.ok)  :text "Quit " :command `(destroy ',w))
  (frame (conc w '.apro))
  (label(conc w '.apro.label) :text "Apropos: ")
  (entry (conc w '.apro.entry) :relief "sunken")
  (pack	 (conc w '.apro.label) (conc w '.apro.entry) :side "left"
	:expand "yes")
  (pack
      (conc w '.frame) (conc w '.ok)
      (conc w '.apro) :side "top" :fill "both")
  (bind (conc w '.apro.entry) "<KeyPress-Return>"
	#'(lambda()
	    (insert-info-choices
	     listbox
	     (info-aux  (funcall (conc w '.apro.entry)
				 :get :return 'string)
			 info-dirs)
	     )))
  (bind  w "<Enter>" `(focus ',(conc w '.apro.entry)))
  w
)


(defun get-info-apropos (win file type)
  (cond ((and win
	      (winfo :exists win :return 'boolean))
	 (let ((old (get win 'info-data)))
	   (unless (eq old *current-info-data*)
		   (setf (get win 'info-data) *current-info-data*)
		   (funcall (conc win '.frame.list) :delete 0 'end))
	   (raise win)
	   (focus win)
	   win))
	(t (offer-choices file type nil))))
(defun show-info-key (win key)
  (let ((node (get win 'node)) name)
    (or node (info-error "No Node?"))
    (setq name  (if
      (f >= (string-match 
	     (si::string-concatenate key
				 #u":[ \t]+([^\n\t,]+)[\n\t,]")
	     (node string node)
	     (node header node)
	     (node begin node))
	 0)
      (get-match (node string node) 1)))
    (if name (show-info name nil))))
(defun mkinfo (&optional (w '.info_text) &aux textwin menu 
			 )
  (if (winfo :exists w :return 'boolean) (destroy w))
  (toplevel w)
  (wm :title w "Info Text Window")
  (wm :iconname w "Info")
  (frame (setq menu (conc w '.menu )):relief "raised" :borderwidth 1)
  (setq textwin (conc w '.t))
  (pack  menu  :side "top" :fill "x")
  (button (conc menu '.quit) :text "Quit" :command
	  `(destroy ',w))
  
  (menubutton (conc menu '.file) :text "File" :relief 'raised
	      :menu (conc menu '.File '.m) :underline 0)
  (menu (conc menu '.file '.m))
  (funcall (conc menu '.file '.m)
	   :add 'command
	   :label "Hotlist"
	   :command '(show-info (tk-conc "("(default-info-hotlist)
					 ")")
				nil))
  (funcall (conc menu '.file '.m)
	   :add 'command
	   :label "Add to Hotlist"
	   :command `(add-to-hotlist ',textwin))
  (funcall (conc menu '.file '.m)
	   :add 'command
	   :label "Top Dir"
	   :command `(show-info "(dir)" nil))

  (button (conc menu '.next) :text "Next" :relief 'raised
	  :command `(show-info-key ',textwin "Next"))
  (button (conc menu '.prev) :text "Previous" :relief 'raised
	  :command `(show-info-key ',textwin "Prev"))
  (button (conc menu '.up) :text "Up" :relief 'raised
	  :command `(show-info-key ',textwin "Up"))
  (button (conc menu '.info) :text "Info" :relief 'raised
	  :command `(if (winfo :exists ".info")
			(raise '.info)
		      (offer-choices nil si::*default-info-files*)
		      ))
  (button (conc menu '.last) :text "Last" :relief 'raised
	  :command `(info-show-history ',textwin 'last))
  (button (conc menu '.history) :text "History" :relief 'raised
	  :command `(info-show-history ',textwin 'history))

  (pack  (conc menu '.file)
	 (conc menu '.quit)  (conc menu '.next)  (conc menu '.prev)
	  (conc menu '.up)  (conc menu '.prev)  
	  (conc menu '.last) 	  (conc menu '.history) (conc menu '.info)
	  :side "left")
;  (entry (conc menu '.entry) :relief "sunken")
;  (pack (conc menu '.entry) :expand "yes" :fill "x")

;  (pack    (conc menu '.next) 
;	  :side "left")
  
  
  (bind  w "<Enter>" `(focus ',menu))
  
;  (tk-menu-bar menu (conc menu '.next) )
;  (bind menu "<Any-M-KeyPress>" "tk_traverseToMenu %W %A")
  (scrollbar (conc w '.s) :relief "flat" :command (tk-conc w ".t yview"))
  (text textwin :relief "raised" :bd 2
		 :setgrid "true"
	 :state 'disabled)
  (funcall textwin  :configure 
	 :yscrollcommand
	 (scroll-set-fix-xref-closure
	  textwin
	  (conc w '.s))
	 )
  
  (bind menu "<KeyPress-n>" `(show-info-key ',textwin "Next"))
  (bind menu "<KeyPress-u>" `(show-info-key ',textwin "Up"))
  (bind menu "<KeyPress-p>" `(show-info-key ',textwin "Prev"))
  (bind menu "<KeyPress-l>"  (nth 4(funcall (conc menu '.last)
				     :configure :command :return
				     'list-strings)))

;; SEARCHING: this needs to be speeded up and fixed.
;  (bind (conc menu '.entry) "<KeyPress>"
;	`(info-text-search ',textwin ',menu %W %A %K))
;  (bind (conc menu '.entry) "<Control-KeyPress>"
;	`(info-text-search ',textwin ',menu %W %A %K))
	    
;  (bind menu "<KeyPress-s>" #'(lambda () (focus (menu '.entry))))
	    
	    
	

  (pack (conc w '.s) :side 'right :fill "y")
  (pack textwin :expand 'yes :fill 'both)
  (funcall textwin :mark 'set 'insert 0.0)
  (funcall textwin :tag :configure 'bold
	   :font :Adobe-Courier-Bold-O-Normal-*-120-*)
  (funcall textwin :tag :configure 'big :font
	   :Adobe-Courier-Bold-R-Normal-*-140-*)
  (funcall textwin :tag :configure 'verybig :font
	   :Adobe-Helvetica-Bold-R-Normal-*-240-*)
  (funcall textwin :tag :configure 'xref
	   :font :Adobe-Courier-Bold-O-Normal-*-120-* )
  (funcall textwin :tag :configure 'current_xref
	   :underline 1 )
  (funcall textwin :tag :bind 'xref "<Enter>"
  "eval [concat %W { tag add current_xref } [get_tag_range %W xref @%x,%y]]")

  (funcall textwin :tag :bind 'xref "<Leave>"
        "%W tag remove current_xref 0.0 end")   
  (funcall textwin :tag :bind 'xref "<3>" 
	   `(show-this-node ',textwin |%x| |%y|))
  (focus menu)
;;    (bind w "<Any-Enter>" (tk-conc "focus " w ".t"))
  )


(defun info-text-search (textwin menu entry a k &aux again
				 (node (get textwin 'node)))
  (or node (tk-error "cant find node index"))
;  (print (list entry a k ))
  (cond ((equal k "Delete")
	 (let ((n (funcall entry :index 'insert :return 'number)))
	   (funcall entry :delete  (- n 1))))
	((>= (string-match "Control" k) 0))
	((equal a "") (setq again 1))
	((>= (string-match "[^-]" a) 0)
	 (funcall entry :insert 'insert a) (setq again 0))
	(t (focus menu) ))
  (or again (return-from info-text-search nil))
  (print (list 'begin-search  entry a k ))
  
  (let* (
	 (ind (funcall textwin :index 'current :return 'string))
	 (pos (index-to-position ind
				 (node string node)
				 (node  begin node)
				 (node  end node)
				 
				 ))
	 (where 
	  (info-search (funcall entry :get :return 'string)
		       (+ again (node-offset node) pos))))
    ;; to do mark region in reverse video...
    (cond ((>= where 0)
	   (let ((node (info-node-from-position where)))
	     (print-node node (- where (node-offset node)))))
	  (t (funcall entry :flash )))))

(defvar *last-history* nil)
(defun print-node (node initial-offset &aux last)
;  "print text from node possibly positioning window at initial-offset
;from beginning of node"

  (setq last (list node  initial-offset))
  (let ((text '.info_text) textwin tem)
    (or (winfo :exists text :return 'boolean)
	(mkinfo text))
    (setq 
	  textwin (conc text '.t))
    (funcall textwin :configure :state 'normal)
    (cond ((get textwin 'no-record-history)
	   (remprop textwin 'no-record-history))
	  ((setq tem (get textwin 'node))
	   (setq *last-history* nil)
	   (push 
	    (format nil #u"* ~a:\t(~a)~a.\tat:~a"
		  (node name tem)
		      (node file tem)
		      (node name tem)
		       (funcall textwin :index "@0,0" :return 'string)
		       )
		 (get textwin 'history))))
    (setf (get textwin 'node) node)
    (funcall textwin :delete 0.0 'end)
    (funcall textwin :mark :set 'insert "1.0")
    (cond ((> initial-offset 0)
	   ;; insert something to separate the beginning of what
	   ;; we want to show and what goes before.
	   (funcall textwin :insert "0.0" #u"\n")
	   (funcall textwin :mark :set 'display_at 'end)
	   (funcall textwin :mark :set 'insert  'end)
	   (funcall textwin :yview 'display_at)
	   (insert-fontified textwin (node string node)
			     (+  (node begin node) initial-offset)
			     (node end node))
	   (funcall textwin :mark :set 'insert "0.0")
	   (insert-fontified textwin (node string node)
			     (node begin node)
			     (+     (node begin node) initial-offset))
)
	  (t
	   (insert-fontified textwin (node string node)
			     (node begin node)
			     (node end node))))
    (funcall textwin :configure :state 'disabled)
    (raise text)
    textwin
    ))



(defun info-show-history (win type)
  (let ((his (get win 'history)))
    (cond ((stringp type)
	   (if (f >= (string-match #u":\t([^\t]+)[.]\tat:([0-9.]+)" type) 0)
	       (let ((pos (get-match type 2))
		     (w (show-info (get-match type 1) nil)))
		 (setf (get win 'no-record-history) t)
		 (or (equal "1.0" pos)
		     (funcall w :yview pos)))))
	  ((eq type 'last)
	   (info-show-history win (if *last-history*
				      (pop *last-history*)
				    (progn (setq *last-history*
						 (get win 'history))
					   (pop *last-history*)))))
	  ((eq type 'history)
	   (let* ((w '.info_history)
		  (listbox (conc w '.frame.list)))
	     (cond ((winfo :exists w :return 'boolean))
		   (t
		    (toplevel w)
		    (simple-listbox w)
		    (button (conc w '.quit) :text "Quit" :command
			    `(destroy ',w))
		    (pack (conc w '.frame) (conc w '.quit)
			  :expand "yes" :fill 'both)
			  ))
	     (insert-standard-listbox listbox  his)
	     (raise w)
	     (bind listbox "<Double-1>" `(info-show-history
					  ',listbox
					  (car (selection :get
							  :return
							  'list-strings)))))))))



(defun show-this-node (textwin x y)
 (let ((inds (get_tag_range  textwin  'xref "@" :|| x :"," :|| y  :return
		      'list-strings)))
   (cond ((and inds (listp inds) (eql (length inds) 2))
	  (show-info (nsubstitute #\space #\newline
			     (apply textwin :get :return 'string  inds))
		     nil))
	 (t (print inds)))))

(defun scroll-set-fix-xref-closure (wint wins &aux prev)
  #'(lambda (&rest l)
      (or (equal l prev)
	  (progn (setq prev l)
		 (fix-xref wint)
		 (apply wins :set l)))))


(defvar *recursive* nil)

;(defun fix-xref-faster (win &aux   (all'(" ")) tem)
;  (unless
;   *recursive*
;   (let* ((*recursive* t) s
;	  (pat #u"\n\\* ([^:\n]+)::|\n\\* [^:\n]+:[ \t]*(\\([^,\n\t]+\\)[^,.\n\t]*)[^\n]?|\n\\* [^:\n]+:[ \t]*([^,(.\n\t]+)[^\n]?")
;	  (beg (funcall win :index "@0,0 linestart -1 char" :return 'string))
;	  (end (funcall win :index "@0,1000 lineend" :return 'string)))
;     (cond ((or (f >= (string-match "possible_xref"
;		    (funcall win :tag :names beg :return 'string)) 0)
;		(not (equal ""
;			    (setq tem (funcall win :tag :nextrange "possible_xref" beg end
;					:return 'string)))))
;	    (if tem (setq beg (car (list-string tem))))
;	    (let ((s (funcall win :get beg end :return 'string))
;		  (j 0) i)
;	      (with-tk-command
;	       (pp "MultipleTagAdd" no_quote)
;	       (pp win normal)
;	       (pp "xref" normal)
;	       (pp beg normal)
;	       (pp "{" no_quote)
;	       (while (f >= (string-match pat s j) 0)
;		 (setq i (if (f >= (match-beginning 1) 0) 1 2))
;		 (pp (match-beginning i) no_quote)
;		 (pp (match-end i) no_quote)
;		 (setq j (match-end 0))
;		 )
;	       (pp "}" no_quote)
;	       (send-tcl-cmd *tk-connection* tk-command nil)))
;	    (funcall win :tag :remove "possible_xref" beg end)
;	    )))))

(defun fix-xref (win &aux    tem)
  (unless
   *recursive*
   (let* ((*recursive* t) 
	  (pat #u"\n\\* ([^:\n]+)::|\n\\* [^:\n]+:[ \t]*(\\([^,\n\t]+\\)[^,.\n\t]*)[^\n]?|\n\\* [^:\n]+:[ \t]*([^,(.\n\t]+)[^\n]?")
	  (beg (funcall win :index "@0,0 linestart -1 char" :return 'string))
	  (end (funcall win :index "@0,1000 lineend" :return 'string)))
     (cond ((or (f >= (string-match "possible_xref"
		    (funcall win :tag :names beg :return 'string)) 0)
		(not (equal ""
			    (setq tem (funcall win :tag :nextrange
					       "possible_xref" beg end
					:return 'string)))))
	    (if tem (setq beg (car (list-string tem))))
	    (let ((s (funcall win :get beg end :return 'string))
		  (j 0) i)
	      (while (f >= (string-match pat s j) 0)
		(setq i
		      (if (f >= (match-beginning 1) 0) 1
			 (if (f >= (match-beginning 2) 0) 2
			   3)))
		(funcall win :tag :add "xref"
			 beg : "+" : (match-beginning i) : " chars"
			 beg : "+" : (match-end i) : " chars")
		(setq j (match-end 0))))
	    (funcall win :tag :remove "possible_xref" beg end)
	    )))))

(defun insert-fontified (window string beg end)
  "set fonts in WINDOW for string with "
;  (waiting window)
;  (print (list beg end))
  (insert-string-with-regexp
   window string beg end
   #u"\n([^\n]+)\n[.=_*-][.=*_-]+\n|\\*Note ([^:]+)::"
   '((1 section-header)
     (2 "xref")
     ))
  (funcall window :tag :add "possible_xref" "0.0" "end")
  (fix-xref window)
  (end-waiting window)
   )

(defun section-header (win string lis &aux (i (car lis)))
  (let ((mark 'insert))
    (insert-string win  string (match-beginning 0)
		   (match-end i))
    (funcall win :insert mark #u"\n")
    (funcall win :tag :add
	     (cdr (assoc (aref string (f + (match-end i) 2))
			 '((#\= . "verybig")
			   (#\_ . "big")
			   (#\- . "big")
			   (#\. . "bold")
			   (#\* . "bold")
			   )))	
	     "insert - " : (f - (match-end i) (f + (match-beginning i ) -1 ))
	     : " chars"
	     "insert -1 chars")
    ;;make index count be same..
    (let ((n (f - (f - (match-end 0)
		     (match-end i)) 1)))
      (declare (fixnum n))
      (if (>= n 0)
	  (funcall win :insert mark (make-string n )))
      )))


(defun insert-string (win string beg end)
  (and (> end beg)
  (let ((ar (make-array  (- end beg) :element-type 'character
			:displaced-to string :displaced-index-offset beg)))
    (funcall win :insert 'insert ar))))

(defun insert-string-with-regexp (win string beg  end regexp reg-actions
				      &aux (i 0) temi 
				      (*window* win) *match-data*)
  (declare (special *window* *match-data*))
  (declare (fixnum beg end i))
  (while (f >= (string-match regexp string beg end) 0)
    (setq i 1)
    (setq temi nil)
    (loop (or (< i 10) (return nil))
      (cond ((f >= (match-beginning i) 0)
	     (setq temi (assoc i reg-actions))
	     (return nil)))
      (setq i (+ i 1)))
    (cond ;(t nil)
	  ((functionp (second temi))
	   (insert-string win string beg (match-beginning 0))
	   (funcall (second temi) win string temi))
	  ((stringp (second temi))
	   (insert-string win string beg (match-end 0))
	   (dolist
	    (v (cdr temi))
	    (funcall win :tag :add v
		     "insert -" : (f - (match-end 0) (match-beginning i)) : " chars"
		     "insert -" :(f - (match-end 0) (match-end i)): " chars"

		     )
	    ))
	  (t (info-error "bad regexp prop")))
    (setq beg (match-end 0))
    (or (<= beg end) (error "hi")) 
    )
  (insert-string win string beg end))

(defun count-char (ch string beg end &aux (count 0))
;  "Count the occurrences of CH in STRING from BEG to END"
  (declare (character ch))
  (declare (string string))
  (declare (fixnum beg end count))
  (while (< beg end)
    (if (eql (aref string beg) ch) (incf count))
    (incf beg))
  count)

(defun start-of-ith-line (count string beg &optional (end -1))
  (declare (string string))
  (declare (fixnum beg end count))
  (if (< end 0) (setq end (length string)))
  (cond ((eql count 1) beg)
	(t (decf count)
	 (while (< beg end)
	   (if (eql (aref string beg) #\newline)
	       (progn (decf count)
		      (incf beg)
		      (if (<= count 0) (return-from start-of-ith-line beg)))
	     (incf beg)))
	 beg)))
  
(defun index-to-position (index string beg &optional (end -1) &aux (count 0))
; "Find INDEX of form \"line.char\" in STRING with 0.0 at BEG  and
;   up to END.  Result is a fixnum string index"
  (declare (string string index))
  (declare (fixnum beg end count))
  (if (< end 0) (setq end (length string)))
  (let* ((line (atoi index 0))
         (charpos (atoi index (+ 1 (position #\. index)))))
    (declare (fixnum line charpos))
    (setq count (start-of-ith-line line string beg end))
    (print (list count charpos))
    (+ count charpos)))



;;; Local Variables: ***
;;; mode:lisp ***
;;; version-control:t ***
;;; comment-column:0 ***
;;; comment-start: ";;; " ***
;;; End: ***

