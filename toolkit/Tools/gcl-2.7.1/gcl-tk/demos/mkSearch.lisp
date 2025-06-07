;;# mkTextSearch w
(in-package "TK")

;;
;; Create a top-level window containing a text widget that allows you
;; to load a file and highlight all instances of a given string.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(defun mkTextSearch (&optional (w '.search) &aux (textwin (conc w '.t)))
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Text Demonstration - Search and Highlight")
    (wm :iconname w "Text Search")

    (frame (conc w '.file))
    (label (conc w '.file.label) :text "File name:" :width 13 :anchor "w")
    (entry (conc w '.file.entry) :width 40 :relief "sunken" :bd 2
	   :textvariable 'fileName)
    (button (conc w '.file.button) :text "Load File" 
	    :command `(TextLoadFile ',textwin fileName))
    (pack (conc w '.file.label) (conc w '.file.entry) :side "left")
    (pack (conc w '.file.button) :side "left" :pady 5 :padx 10)
    (bind (conc w '.file.entry) "<Return>"
	  `(progn
	     (TextLoadFile ',textwin fileName)
	     (focus (conc ',w '.string.entry))))
    (frame (conc w '.string))
    (label (conc w '.string.label) :text "Search string:" :width 13 :anchor "w")
    (entry (conc w '.string.entry) :width 40 :relief "sunken" :bd 2 
	    :textvariable 'searchString)
    (button (conc w '.string.button) :text "Highlight" 
	    :command `(TextSearch ',textwin searchString "search"))
    (pack (conc w '.string.label) (conc w '.string.entry) :side "left")
    (pack (conc w '.string.button) :side "left" :pady 5 :padx 10)
    (bind (conc w '.string.entry) "<Return>" `(TextSearch
					       ',textwin searchString "search"))

    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (text textwin :relief "raised" :bd 2 :yscrollcommand  (tk-conc w ".s set")
	  :setgrid "true")
    (scrollbar (conc w '.s) :relief "flat" :command (tk-conc w  ".t yview"))
    (pack (conc w '.file) (conc w '.string) :side "top" :fill "x")
    (pack (conc w '.ok) :side "bottom" :fill "x")
    (pack (conc w '.s) :side "right" :fill "y")
    (pack textwin :expand "yes" :fill "both")

    ;; Set up display styles for text highlighting.
    (let* (com
	   (bg (if (> (read-from-string (winfo :depth w)) 1)
		   "SeaGreen4" "black"))
	   on
	       
	   (fun #'(lambda ()
		    (when (myerrorset
			     (progn (funcall textwin
					     :tag
					     :configure "search"
					     :background (if on bg "")
					     :foreground (if on "white" ""))
				    t))
			    (setq on (not on))
			    (myerrorset (after 500 com))
			    ))))
      (setq com (tcl-create-command fun nil nil))
      (setq bil fun)
      (funcall fun ))
    (funcall textwin :insert 0.0
 "
This window demonstrates how to use the tagging facilities in text
widgets to implement a searching mechanism.  First, type a file name
in the top entry, then type <Return> or click on \"Load File\".  Then
type a string in the lower entry and type <Return> or click on
\"Load File\".  This will cause all of the instances of the string to
be tagged with the tag \"search\", and it will arrange for the tag's
display attributes to change to make all of the strings blink.
"
)
    (funcall textwin :mark :set 'insert 0.0)
    (bind w "<Any-Enter>" (tk-conc "focus " w ".file.entry"))
)
(setq fileName "")
(setq searchString "")

;; The utility procedure below loads a file into a text widget,
;; discarding the previous contents of the widget. Tags for the
;; old widget are not affected, however.
;; Arguments:
;;
;; w -		The window into which to load the file.  Must be a
;;		text widget.
;; file -	The name of the file to load.  Must be readable.

(defun TextLoadFile (w file)
  (with-open-file
   (st file)
   (let ((ar (make-array 3000 :element-type 'string-char :fill-pointer 0))
	 (n (file-length st))
	 m)
     (funcall w :delete "1.0" 'end)
     (while (> n 0)
       (setq m (min (array-total-size ar) n))
       (setq n (- n m))
       (si::fread ar 0 m st)
       (setf (fill-pointer ar) m)
       (funcall w :insert 'end ar)))))

     

;; The utility procedure below searches for all instances of a
;; given string in a text widget and applies a given tag to each
;; instance found.
;; Arguments:
;;
;; w -		The window in which to search.  Must be a text widget.
;; string -	The string to search for.  The search is done using
;;		exact matching only;  no special characters.
;; tag -		Tag to apply to each instance of a matching string.

(defun TextSearch (w string tag) 
    (funcall w :tag :remove 'search 0.0 'end)
    (let ((mark "mine")
	  (m (length string)))
      (funcall w :mark :set "mine" "0.0")
      (while (funcall w :compare mark '< 'end :return 'boolean)
	(let ((s (funcall w :get mark mark : " + 3000 chars" :return 'string))
	      (n 0) tem)
	  (while (setq tem (search string s :start2 n))
	    (funcall w :tag :add 'search
		     mark : " + " : tem : " chars"
		     mark : " + " : (setq n (+ tem m)) : " chars"))
	  (funcall w :mark :set mark mark : " + " : (- 3000 m) : " chars")))))

