;;# mkForm w
;;
;; Create a top-level window that displays a bunch of entries with
;; tabs set up to move between them.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")
(defvar *tablist*)

(defun mkForm (&optional (w '.form)) 
    (setq *tablist* nil)
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Form Demonstration")
    (wm :iconname w "Form")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :width "4i" 
	    :text "This window contains a simple form where you can type in the various entries and use tabs to move circularly between the entries.  Click the \"OK\" button or type return when you're done.")
   (dolist (i '(f1 f2 f3 f4 f5))
	(frame (conc w '|.| i) :bd "1m")
	(entry (conc w '|.| i '.entry) :relief "sunken" :width 40)
	(bind (conc w '|.| i '.entry) "<Tab>" '(Tab *tabList*))
	(bind (conc w '|.| i '.entry) "<Return>" `(destroy ',w))
	(label (conc w '|.| i '.label))
	(pack (conc w '|.| i '.entry) :side "right")
	(pack (conc w '|.| i '.label) :side "left")
	(push (conc i '.entry) *tablist*))
    (setq *tablist* (nreverse *tablist*))  
    (funcall (conc w '.f1.label) :config :text "Name: ")
    (funcall (conc w '.f2.label) :config :text "Address: ")
    (funcall (conc w '.f5.label) :config :text "Phone: ")
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.f1) (conc w '.f2) (conc w '.f3)
	  (conc w '.f4) (conc w '.f5) (conc w '.ok) :side "top" :fill "x")

)

;; The procedure below is invoked in response to tabs in the entry
;; windows.  It moves the focus to the next window in the tab list.
;; Arguments:
;;
;; list -	Ordered list of windows to receive focus

(defun Tab (list) 
  (setq i (position (focus :return t) list))
  (cond ((null i) (setq i 0))
	(t (incf i)
	   (if (>=  i (length list) )
	       (setq i 0))))
  (focus (nth i list ))
)

