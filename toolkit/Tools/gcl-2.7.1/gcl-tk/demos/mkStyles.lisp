;;# mkStyles w
;;
;; Create a top-level window with a text widget that demonstrates the
;; various display styles that are available in texts.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")
(defun mkStyles (&optional (w '.styles) &aux (textwin (conc w '.t)) )
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Text Demonstration - Display Styles")
    (wm :iconname w "Text Styles")

    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (scrollbar (conc w '.s) :relief "flat" :command (tk-conc w ".t yview"))
    (text textwin :relief "raised" :bd 2 :yscrollcommand (tk-conc w ".s set") :setgrid "true" 
	    :width 70 :height 28)
    (pack (conc w '.ok) :side "bottom" :fill "x")
    (pack (conc w '.s) :side "right" :fill "y")
    (pack textwin :expand "yes" :fill "both")

    ;; Set up display styles

    (funcall textwin :tag :configure 'bold :font :Adobe-Courier-Bold-O-Normal-*-120-*)
    (funcall textwin :tag :configure 'big :font :Adobe-Courier-Bold-R-Normal-*-140-*)
    (funcall textwin :tag :configure 'verybig :font :Adobe-Helvetica-Bold-R-Normal-*-240-*)
    (if (> (read-from-string (winfo :depth w)) 1)
	(progn 
	  (funcall textwin :tag :configure 'color1 :background "#eed5b7")
	  (funcall textwin :tag :configure 'color2 :foreground "red")
	  (funcall textwin :tag :configure 'raised :background "#eed5b7"
		   :relief "raised" 
		   :borderwidth 1)
	(funcall textwin :tag :configure 'sunken :background "#eed5b7"
		 :relief "sunken" 
		 :borderwidth 1)
    ) ;;else 
 (progn 
	(funcall textwin :tag :configure 'color1 :background "black" :foreground "white")
	(funcall textwin :tag :configure 'color2 :background "black" :foreground "white")
	(funcall textwin :tag :configure 'raised :background "white" :relief "raised" 
		:borderwidth 1)
	(funcall textwin :tag :configure 'sunken :background "white" :relief "sunken" 
		:borderwidth 1)
    ))
    (funcall textwin :tag :configure 'bgstipple :background "black" :borderwidth 0 
	    :bgstipple "gray25")
    (funcall textwin :tag :configure 'fgstipple :fgstipple "gray50")
    (funcall textwin :tag :configure 'underline :underline "on")

    (funcall textwin :insert 0.0 "
Text widgets like this one allow you to display information in a
variety of styles.  Display styles are controlled using a mechanism
called " )
    (insertWithTags textwin "tags" 'bold)
    (insertWithTags textwin ". Tags are just textual names that you can apply to one
or more ranges of characters within a text widget.  You can configure
tags with various display styles.  (if :you do this, then the tagged
characters will be displayed with the styles you chose.  The
available display styles are:
"
)
    (insertWithTags textwin "
1. Font." 'big)
    (insertWithTags textwin "  You can choose any X font, ")
    (insertWithTags textwin "large" "verybig")
    (insertWithTags textwin " or ")
    (insertWithTags textwin "small.
")

    (insertWithTags textwin "
2. Color." 'big)
    (insertWithTags textwin "  You can change either the ")
    (insertWithTags textwin "background" "color1")
    (insertWithTags textwin " or ")
    (insertWithTags textwin "foreground" "color2")
    (insertWithTags textwin "
color, or ")
    (insertWithTags textwin "both" "color1" "color2")
    (insertWithTags textwin ".
")

    (insertWithTags textwin "
3. Stippling." 'big)
    (insertWithTags textwin "  You can cause either the ")
    (insertWithTags textwin "background" 'bgstipple)
    (insertWithTags textwin " or ")
    (insertWithTags textwin "foreground" 'fgstipple)
    (insertWithTags textwin "
information to be drawn with a stipple fill instead of a solid fill.
")
    (insertWithTags textwin "
4. Underlining." 'big)
    (insertWithTags textwin "  You can ")
    (insertWithTags textwin "underline" "underline")
    (insertWithTags textwin " ranges of text.
")
    (insertWithTags textwin "
5. 3-D effects." 'big)
    (insertWithTags textwin
"  You can arrange for the background to be drawn
with a border that makes characters appear either ")
    (insertWithTags textwin "raised" "raised")
    (insertWithTags textwin " or ")
    (insertWithTags textwin "sunken" "sunken")
    (insertWithTags textwin ".
")
    (insertWithTags textwin "
6. Yet to come." 'big)
    (insertWithTags textwin
"  More display effects will be coming soon, such
as the ability to change line justification and perhaps line spacing.")
    (funcall textwin :mark :set 'insert 0.0)
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
    (dolist (v (funcall w :tag :names start :return 'list-strings))
	    (funcall w :tag :remove v start 'insert))
    (dolist (i args)
	    (funcall w :tag :add i start 'insert))))
