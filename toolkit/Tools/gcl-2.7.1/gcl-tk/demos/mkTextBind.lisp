;;# mkTextBind w
;;
;; Create a top-level window that illustrates how you can bind
;; Tcl commands to regions of text in a text widget.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")
(defun mkTextBind (&optional (w '.bindings) &aux bold normal
			     (textwin (conc w '.t ) ))
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Text Demonstration - Tag Bindings")
    (wm :iconname w "Text Bindings")
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (scrollbar (conc w '.s) :relief "flat" :command (tk-conc w ".t yview"))
    (text textwin :relief "raised" :bd 2 :yscrollcommand
	  (tk-conc w ".s set") :setgrid "true" 
	    :width 60 :height 28 
	    :font "-Adobe-Helvetica-Bold-R-Normal-*-120-*")

    (pack (conc w '.ok) :side "bottom" :fill "x")
    (pack (conc w '.s) :side "right" :fill "y")
    (pack textwin :expand "yes" :fill "both")

    ;; Set up display styles

    (if (> (read-from-string (winfo :depth w)) 1)
	(progn 
	  (setq bold '(:foreground "red"))
	  (setq normal '(:foreground ""))
	  );;else 
      (progn 
	(setq bold '(:foreground "white" :background "black"))
	(setq normal '(:foreground "" :background ""))
	))
    (funcall textwin :insert 0.0
"The same tag mechanism that controls display styles in text
widgets can also be used to associate Tcl commands with regions
of text, so that mouse or keyboard actions on the text cause
particular Tcl commands to be invoked.  For example, in the
text below the descriptions of the canvas demonstrations have
been tagged.  When you move the mouse over a demo description
the description lights up, and when you press button 3 over a
description then that particular demonstration is invoked.

This demo package contains a number of demonstrations of Tk's
canvas widgets.  Here are brief descriptions of some of the
demonstrations that are available:
"
)
   (let ((blank-lines (format nil "~2%")))
    (insertWithTags textwin 
"1. Samples of all the different types of items that can be
created in canvas widgets." "d1")
    (insertWithTags textwin blank-lines)
    (insertWithTags textwin 
"2. A simple two-dimensional plot that allows you to adjust
the :positions of the data points." "d2")
    (insertWithTags textwin blank-lines)
    (insertWithTags textwin 
"3. Anchoring and justification modes for text items." "d3")
    (insertWithTags textwin blank-lines)
    (insertWithTags textwin 
"4. An editor for arrow-head shapes for line items." "d4")
    (insertWithTags textwin blank-lines)
    (insertWithTags textwin 
"5. A ruler with facilities for editing tab stops." "d5")
    (insertWithTags textwin blank-lines)
    (insertWithTags textwin 
"6. A grid that demonstrates how canvases can be scrolled." "d6"))

    (dolist (tag '("d1" "d2" "d3" "d4" "d5" "d6"))
	(funcall textwin :tag :bind tag "<Any-Enter>"
		 `(,textwin :tag :configure ,tag ,@bold))
	(funcall textwin :tag :bind tag "<Any-Leave>"
		 `(,textwin :tag :configure  ,tag ,@normal))
	)
    (funcall textwin :tag :bind "d1" "<3>" 'mkItems)
    (funcall textwin :tag :bind "d2" "<3>" 'mkPlot)
    (funcall textwin :tag :bind "d3" "<3>" "mkCanvText")
    (funcall textwin :tag :bind "d4" "<3>" "mkArrow")
    (funcall textwin :tag :bind "d5" "<3>" 'mkRuler)
    (funcall textwin :tag :bind "d6" "<3>" "mkScroll")

    (funcall textwin :mark 'set 'insert 0.0)
    (bind w "<Any-Enter>" (tk-conc "focus " w ".t"))
)

;; The procedure below inserts text into a given text widget and
;; applies one or more tags to that text.  The arguments are:
;;
;; w		Window in which to insert
;; text		Text to insert (it's :inserted at the "insert" mark)
;; args		One or more tags to apply to text.  (if :this is empty
;;		then all tags are removed from the text.

(defun insertWithTags (w text &rest args) 
  (let (( start (funcall w :index 'insert :return 'string)))
    (funcall w :insert 'insert text)
    (dolist (v (funcall w :tag "names" start :return 'list-strings))
	    (funcall w :tag 'remove v start "insert"))
    (dolist (i args)
	    (funcall w :tag 'add i start 'insert))))
    

