(in-package "TK")
;;# showVars w var var var '...
;;
;; Create a top-level window that displays a bunch of global variable values
;; and keeps the display up-to-date even when the variables change value
;;
;; Arguments:
;;    w -	Name to use for new top-level window.
;;    var -	Name of variable to monitor.

(defun showVars (w args) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (wm :title w "Variable values")
    (label (conc w '.title) :text "Variable values:" :width 20 :anchor "center" 
	    :font :Adobe-helvetica-medium-r-normal--*-180*)
    (pack (conc w '.title) :side "top" :fill "x")
    (dolist (i args) 
	(frame (conc w '|.| i))
	(label (conc w '|.| i '.name) :text (tk-conc i ": "))
	(label (conc w '|.| i '.value) :textvariable
                             (list (or (get i 'text-variable-type) t) i))
	(pack (conc w '|.| i '.name) (conc w '|.| i '.value) :side "left")
	(pack (conc w '|.| i) :side "top" :anchor "w")
    )
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.ok) :side "bottom" :pady 2)
)
