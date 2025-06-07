;;# mkLabel w
;;
;; Create a top-level window that displays a bunch of labels.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")
(defun mkLabel (&optional (w '.l1)) 
;    (global :tk_library)
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Label Demonstration")
    (wm :iconname w "Labels")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300 
	    :text "Five labels are displayed below: three textual ones on the left, and a bitmap label and a text label on the right.  Labels are pretty boring because you can't do anything with them.  Click the \"OK\" button when you've seen enough.")
    (frame (conc w '.left))
    (frame (conc w '.right))
    (button (conc w '.ok) :text "OK" :command `(destroy ',w))
    (pack (conc w '.msg) :side "top")
    (pack (conc w '.ok) :side "bottom" :fill "x")
    (pack (conc w '.left) (conc w '.right) :side "left" :expand "yes" :padx 10 :pady 10 :fill "both")

    (label (conc w '.left.l1) :text "First label")
    (label (conc w '.left.l2) :text "Second label, raised just for fun" :relief "raised")
    (label (conc w '.left.l3) :text "Third label, sunken" :relief "sunken")
    (pack (conc w '.left.l1) (conc w '.left.l2) (conc w '.left.l3) 
	    :side "top" :expand "yes" :pady 2 :anchor "w")

    (label (conc w '.right.bitmap) :bitmap "@": *tk-library* : "/demos/images/face" 
	    :borderwidth 2 :relief "sunken")
    (label (conc w '.right.caption) :text "Tcl/Tk Proprietor")
    (pack (conc w '.right.bitmap) (conc w '.right.caption) :side "top")
)
