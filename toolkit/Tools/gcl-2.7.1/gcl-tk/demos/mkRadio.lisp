(in-package "TK")
;;# mkRadio w
;;
;; Create a top-level window that displays a bunch of radio buttons.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(defun mkRadio (&optional (w '.r1)) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Radiobutton Demonstration")
    (wm :iconname w "Radiobuttons")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300 
	    :text "Two groups of radiobuttons are displayed below.  If you click on a button then the button will become selected exclusively among all the buttons in its group.  A Tcl variable is associated with each group to indicate which of the group's buttons is selected.  Click the \"See Variables\" button to see the current values of the variables.  Click the \"OK\" button when you've seen enough.")
    (frame (conc w '.frame) :borderwidth 10)
    (frame (conc w '.frame2))
    (pack (conc w '.msg) :side "top")
    (pack (conc w '.msg) :side "top")
    (pack (conc w '.frame) :side "top" :fill "x" :pady 10)
    (pack (conc w '.frame2) :side "bottom" :fill "x")

    (frame (conc w '.frame.left))
    (frame (conc w '.frame.right))
    (pack (conc w '.frame.left) (conc w '.frame.right) :side "left" :expand "yes")

    (radiobutton (conc w '.frame.left.b1) :text "Point Size 10" :variable 'size 
	    :relief "flat" :value 10)
    (radiobutton (conc w '.frame.left.b2) :text "Point Size 12" :variable 'size 
	    :relief "flat" :value 12)
    (radiobutton (conc w '.frame.left.b3) :text "Point Size 18" :variable 'size 
	    :relief "flat" :value 18)
    (radiobutton (conc w '.frame.left.b4) :text "Point Size 24" :variable 'size 
	    :relief "flat" :value 24)
    (pack (conc w '.frame.left.b1) (conc w '.frame.left.b2) (conc w '.frame.left.b3) (conc w '.frame.left.b4) 
	    :side "top" :pady 2 :anchor "w")

    (radiobutton (conc w '.frame.right.b1) :text "Red" :variable 'color 
	    :relief "flat" :value "red")
    (radiobutton (conc w '.frame.right.b2) :text "Green" :variable 'color 
	    :relief "flat" :value "green")
    (radiobutton (conc w '.frame.right.b3) :text "Blue" :variable 'color 
	    :relief "flat" :value "blue")
    (radiobutton (conc w '.frame.right.b4) :text "Yellow" :variable 'color 
	    :relief "flat" :value "yellow")
    (radiobutton (conc w '.frame.right.b5) :text "Orange" :variable 'color 
	    :relief "flat" :value "orange")
    (radiobutton (conc w '.frame.right.b6) :text "Purple" :variable 'color 
	    :relief "flat" :value "purple")
    (pack (conc w '.frame.right.b1) (conc w '.frame.right.b2) (conc w '.frame.right.b3) 
	    (conc w '.frame.right.b4) (conc w '.frame.right.b5) (conc w '.frame.right.b6) 
	    :side "top" :pady 2 :anchor "w")

    (button (conc w '.frame2.ok) :text "OK" :command (tk-conc "destroy " w) :width 12)
    (button (conc w '.frame2.vars) :text "See Variables" :width 12
	    :command `(showvars (conc ',w '.dialog) '(size color)))
    (pack (conc w '.frame2.ok) (conc w '.frame2.vars) :side "left" :expand "yes" :fill "x")
)


